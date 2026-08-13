import pandas as pd
import duckdb
import os
import time
import json
import threading
import html
import re
import requests
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from huggingface_hub import HfApi
import random
import gc
from cleantext import clean

# ==========================================
# 0. TELEMETRY & PROFILER SETUP
# ==========================================
perf_metrics = {
    'total_prompt_tokens': 0,
    'total_completion_tokens': 0,
    'total_combined_tokens': 0,
    'total_cache_hits': 0,
    'total_cache_misses': 0
}
token_lock = threading.Lock()
stop_telemetry = threading.Event()

def resource_monitor():
    while not stop_telemetry.is_set():
        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        print(f"   [⚙️ TELEMETRY] RAM: {ram.used / (1024**3):.2f}GB / {ram.total / (1024**3):.2f}GB ({ram.percent}%) | CPU: {cpu}%")
        time.sleep(30) 

monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
monitor_thread.start()
script_start_time = time.time()

# ==========================================
# 1. CONFIGURATION & DIRECTORY SETUP
# ==========================================
TARGET_YEAR = os.environ.get("TARGET_YEAR", "2017")
RUN_ID = os.environ.get("GITHUB_RUN_ID", str(int(time.time())))
GITHUB_STARTED_AT = os.environ.get("GITHUB_RUN_STARTED_AT")

if GITHUB_STARTED_AT:
    safe_timestamp = GITHUB_STARTED_AT.replace('T', '_').replace(':', '-').replace('Z', '')
else:
    safe_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

BASE_OUTPUT_DIR = './labelled_output/'
TIMESTAMP_DIR = os.path.join(BASE_OUTPUT_DIR, safe_timestamp)
os.makedirs(TIMESTAMP_DIR, exist_ok=True)

FINAL_CSV_PATH = os.path.join(TIMESTAMP_DIR, f'baseline_tier1_{TARGET_YEAR}.csv')
LEDGER_PATH = os.path.join(BASE_OUTPUT_DIR, 'seen_ids_ledger.txt')

SEED_VALUE = int(RUN_ID) % 100000 
random.seed(SEED_VALUE)

TARGET_SUBREDDITS = [
    'chodi', 'bakchodi', 'sham_sharma_show', 'desimeta',
    'indiandankmemes', 'dankinindia', 'saimansays', 'librandu',
    'unitedstatesofindia', 'indiadiscussion', 'canconfirmiamindian',
    'arrangedmarriage', 'bollyblindsngossip'
]

TARGET_ROWS_PER_JOB = int(os.environ.get("TARGET_ROWS", 12500))
MAX_WORKERS = 10 

OPENCODE_KEY = os.environ.get("OPENCODE_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not OPENCODE_KEY:
    raise ValueError("❌ OPENCODE_KEY environment variable is missing.")

client = OpenAI(api_key=OPENCODE_KEY, base_url="https://opencode.ai/zen/go/v1")
MODEL_NAME = "deepseek-v4-flash"

# ==========================================
# 2. DYNAMIC SYSTEM PROMPT FETCH
# ==========================================
PROMPT_URL = "https://raw.githubusercontent.com/darelphilipo/hinglish_reddit_data/main/prompt/System_Prompt"
prompt_start = time.time()
try:
    print(f"\n🌐 Fetching latest System Prompt from GitHub...")
    response = requests.get(PROMPT_URL, timeout=10)
    response.raise_for_status()
    SYSTEM_PROMPT = response.text.strip()
    print("✅ System Prompt loaded successfully.")
except Exception as e:
    stop_telemetry.set()
    raise RuntimeError(f"❌ Failed to fetch System Prompt from GitHub: {e}")
perf_metrics['prompt_fetch_time'] = time.time() - prompt_start

# ==========================================
# 3. DUCKDB EXTRACTION (WITH NET STATS)
# ==========================================
db_start = time.time()
print(f"\n🦆 [Worker {TARGET_YEAR}] Initializing DuckDB (Dynamic Seed: {SEED_VALUE})...")
con = duckdb.connect()

con.execute("PRAGMA memory_limit='6GB';") 
con.execute("PRAGMA threads=8;")          
con.execute("INSTALL httpfs; LOAD httpfs;")

if HF_TOKEN:
    con.execute(f"CREATE SECRET hf_auth (TYPE HUGGINGFACE, TOKEN '{HF_TOKEN}');")

api = HfApi(token=HF_TOKEN)
all_files = api.list_repo_files("open-index/arctic", repo_type="dataset")

year_files = [f for f in all_files if f.endswith('.parquet') and f'data/comments/{TARGET_YEAR}' in f]
if not year_files:
    stop_telemetry.set()
    raise ValueError(f"❌ No Parquet shards found for year {TARGET_YEAR}.")

if TARGET_ROWS_PER_JOB < 1000:
    max_shards = 10  
elif TARGET_ROWS_PER_JOB < 5000:
    max_shards = 50  
else:
    max_shards = 200 

selected_shards = random.sample(year_files, min(max_shards, len(year_files)))
hf_urls = [f"hf://datasets/open-index/arctic/{f}" for f in selected_shards]
subs_formatted = ", ".join([f"'{s.lower()}'" for s in TARGET_SUBREDDITS])

query = f"""
SELECT id, body, LOWER(subreddit) as subreddit, created_utc, strftime(epoch_ms(created_utc * 1000), '%Y-%m') as year_month
FROM read_parquet({hf_urls})
WHERE LOWER(subreddit) IN ({subs_formatted})
  AND body NOT IN ('[deleted]', '[removed]', '')
  AND length(body) BETWEEN 10 AND 1000
"""

print(f"⏳ Extracting candidate records for {TARGET_YEAR} across {len(selected_shards)} shards...")

duckdb_running = True
def duckdb_heartbeat():
    start_time = time.time()
    net_start = psutil.net_io_counters().bytes_recv
    last_net = net_start
    
    while duckdb_running:
        time.sleep(15)
        if not duckdb_running:
            break
            
        current_time = time.time()
        elapsed = int(current_time - start_time)
        mins, secs = divmod(elapsed, 60)
        
        current_net = psutil.net_io_counters().bytes_recv
        total_downloaded_mb = (current_net - net_start) / (1024 * 1024)
        bytes_in_window = current_net - last_net
        speed_mb_s = (bytes_in_window / (1024 * 1024)) / 15
        last_net = current_net
        
        print(f"   📡 [HF Network] Time: {mins}m {secs}s | Pulled: {total_downloaded_mb:.1f} MB | Speed: {speed_mb_s:.1f} MB/s")

heartbeat_thread = threading.Thread(target=duckdb_heartbeat, daemon=True)
heartbeat_thread.start()

try:
    raw_df = con.query(query).to_df()
    perf_metrics['duckdb_extract_time'] = time.time() - db_start
    print(f"📦 Pulled {len(raw_df)} raw comments in {perf_metrics['duckdb_extract_time']:.2f}s.")
except Exception as e:
    stop_telemetry.set()
    raise RuntimeError(f"❌ DuckDB Extraction crashed: {e}")
finally:
    duckdb_running = False
    heartbeat_thread.join()

# ==========================================
# 4. ADVANCED SANITIZATION & DEDUPLICATION
# ==========================================
sanitize_start = time.time()
print("\n🧹 Running NLP Text Sanitization (HTML, PII, and User Mentions)...")

def sanitize_text(text):
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'/?u/[A-Za-z0-9_-]+', '', text) 
    
    try:
        text = clean(text, 
            fix_unicode=True, to_ascii=False, lower=False, no_line_breaks=True, 
            no_urls=True, replace_with_url="",
            no_emails=True, replace_with_email="",
            no_phone_numbers=True, replace_with_phone_number="",
            no_html=True
        )
    except Exception:
        pass
    return re.sub(r'\s+', ' ', text).strip()

raw_df['body_clean'] = raw_df['body'].apply(sanitize_text)
raw_df = raw_df[raw_df['body_clean'].str.len() > 5]

print("🛡️ Applying aggressive token-saving deduplication...")
raw_df['dedup_key'] = raw_df['body_clean'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

initial_count = len(raw_df)
raw_df.drop_duplicates(subset=['dedup_key'], keep='first', inplace=True)
raw_df.drop(columns=['dedup_key'], inplace=True)
print(f"✂️ Dropped {initial_count - len(raw_df)} spam/copy-paste variants.")

if os.path.exists(LEDGER_PATH):
    with open(LEDGER_PATH, 'r') as f:
        seen_ids = set(line.strip() for line in f if line.strip())
    prev_len = len(raw_df)
    raw_df = raw_df[~raw_df['id'].isin(seen_ids)]
    print(f"🛡️ Filtered out {prev_len - len(raw_df)} comments processed in previous workflow runs.")

# ==========================================
# 5. STRATIFIED BALANCING
# ==========================================
print("\n⚖️ Balancing sampling evenly across Subreddits & Months...")
groups = raw_df.groupby(['subreddit', 'year_month'])
num_groups = len(groups)

if num_groups > 0:
    samples_per_bucket = max(1, TARGET_ROWS_PER_JOB // num_groups)
    balanced_df = (
        raw_df.groupby(['subreddit', 'year_month'], group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), samples_per_bucket), random_state=SEED_VALUE), include_groups=False)
        .reset_index()
    )
    
    if len(balanced_df) < TARGET_ROWS_PER_JOB and len(raw_df) > len(balanced_df):
        remaining_indices = raw_df.index.difference(balanced_df.index)
        needed = TARGET_ROWS_PER_JOB - len(balanced_df)
        backfill_df = raw_df.loc[remaining_indices].sample(n=min(len(raw_df.loc[remaining_indices]), needed), random_state=SEED_VALUE)
        df = pd.concat([balanced_df, backfill_df], ignore_index=True)
    else:
        df = balanced_df
else:
    df = raw_df

if len(df) > TARGET_ROWS_PER_JOB:
    df = df.sample(n=TARGET_ROWS_PER_JOB, random_state=SEED_VALUE).reset_index(drop=True)

df['temp_id'] = df.index.astype(str)
perf_metrics['sanitization_and_balancing_time'] = time.time() - sanitize_start

del raw_df
gc.collect()

print(f"🎯 Final Balanced Pool for {TARGET_YEAR}: {len(df)} rows.")

with open(LEDGER_PATH, 'a') as f:
    for cid in df['id'].tolist():
        f.write(f"{cid}\n")

# ==========================================
# 6. INFERENCE ENGINE (THREAD-SAFE TOKEN TRACKING)
# ==========================================
api_start = time.time()

def label_batch(comments_batch, attempt=1):
    numbered = "\n".join(f'ID: {cid} | Comment: {body}' for cid, body in comments_batch)
    user_prompt = f"Label these {len(comments_batch)} comments strictly following the JSON schema:\n{numbered}"

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            temperature=0.1, max_tokens=2500, response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}}
        )
        
        usage_dict = res.usage.model_dump() if hasattr(res.usage, 'model_dump') else vars(res.usage)
        token_details = usage_dict.get('prompt_tokens_details', {}) or {}
        
        p_tokens = usage_dict.get('prompt_tokens', 0)
        c_tokens = usage_dict.get('completion_tokens', 0)
        t_tokens = usage_dict.get('total_tokens', p_tokens + c_tokens)
        
        c_hits = usage_dict.get('prompt_cache_hit_tokens', token_details.get('cached_tokens', 0))
        c_misses = usage_dict.get('prompt_cache_miss_tokens', p_tokens - c_hits)

        # Thread-safe accumulation using dedicated lock
        with token_lock:
            perf_metrics['total_prompt_tokens'] += p_tokens
            perf_metrics['total_completion_tokens'] += c_tokens
            perf_metrics['total_combined_tokens'] += t_tokens
            perf_metrics['total_cache_hits'] += c_hits
            perf_metrics['total_cache_misses'] += c_misses

        raw_content = res.choices[0].message.content.strip()
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
            raw_content = re.sub(r"\n?```$", "", raw_content).strip()

        parsed_data = json.loads(raw_content)
        parsed_array = parsed_data.get("results", []) if isinstance(parsed_data, dict) else parsed_data
        
        if len(parsed_array) == len(comments_batch):
            for idx, item in enumerate(parsed_array):
                item["temp_id"] = str(comments_batch[idx][0])
            return parsed_array
        raise ValueError("Batch mismatch")
    except Exception as e:
        if attempt <= 4:
            time.sleep(min(2 ** attempt, 10))
            return label_batch(comments_batch, attempt + 1)
        return []

if df.empty:
    stop_telemetry.set()
    print(f"❌ Worker {TARGET_YEAR}: No valid data to label.")
    exit(0)

batches = [list(zip(df["temp_id"], df["body_clean"]))[i:i + 20] for i in range(0, len(df), 20)]
all_labels = []
lock = threading.Lock()

print(f"\n🚀 Running Parallel Inference on {len(df)} rows across {len(batches)} batches...")
with tqdm(total=len(batches), desc=f"Labeling {TARGET_YEAR}") as pbar:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(label_batch, batch) for batch in batches]
        for future in as_completed(futures):
            with lock:
                all_labels.extend(future.result())
            pbar.update(1)

labels_df = pd.DataFrame(all_labels)
labels_df["temp_id"] = labels_df["temp_id"].astype(str)
labels_df.rename(columns={"analysis": "step_by_step_analysis", "pv": "profanity_vulgarity", "tah": "targeted_abuse_harassment", "dhs": "discriminatory_hate_speech", "cst": "caste", "cr": "communal_religious", "rx": "regional_xenophobic", "mg": "misogyny_gender"}, inplace=True)

final_df = df.merge(labels_df, on="temp_id", how="left").drop(columns=["temp_id", "body_clean", "pv", "tah", "dhs", "cst", "cr", "rx", "mg"], errors='ignore')
final_df.to_csv(FINAL_CSV_PATH, index=False)

perf_metrics['api_inference_time'] = time.time() - api_start
perf_metrics['total_script_time'] = time.time() - script_start_time

stop_telemetry.set()
monitor_thread.join()

# ==========================================
# 7. EXPANDED PERFORMANCE & TOKEN LOG
# ==========================================
total_lbl = len(final_df)
p_tokens = perf_metrics['total_prompt_tokens']
c_tokens = perf_metrics['total_completion_tokens']
comb_tokens = perf_metrics['total_combined_tokens']
hits = perf_metrics['total_cache_hits']
misses = perf_metrics['total_cache_misses']

hit_rate = (hits / p_tokens * 100) if p_tokens > 0 else 0.0
avg_tokens_per_comment = (comb_tokens / total_lbl) if total_lbl > 0 else 0.0

print("\n==================================================")
print(" 📈 PIPELINE PERFORMANCE & RESOURCE LOG")
print("==================================================")
print(f"Total Rows Labeled       : {total_lbl:,}")
print(f"Prompt Fetch Time        : {perf_metrics['prompt_fetch_time']:.2f}s")
print(f"DuckDB Extract Time      : {perf_metrics['duckdb_extract_time']:.2f}s")
print(f"Data Prep & Sanitize     : {perf_metrics['sanitization_and_balancing_time']:.2f}s")
print(f"API Inference Time       : {perf_metrics['api_inference_time']:.2f}s")
print(f"Total Workflow Time      : {perf_metrics['total_script_time']:.2f}s")
print("\n--- 💳 COMPLETE API TOKEN METRICS ---")
print(f"Total Input / Prompt Tokens : {p_tokens:,}")
print(f"  ↳ Prompt Cache Hits       : {hits:,} ({hit_rate:.1f}% Cache Hit Rate)")
print(f"  ↳ Prompt Cache Misses     : {misses:,}")
print(f"Total Output / Completion   : {c_tokens:,}")
print(f"Total Combined Tokens       : {comb_tokens:,}")
print(f"Average Tokens / Comment    : {avg_tokens_per_comment:.1f} tokens/comment")
print("==================================================")
print(f"✅ Worker {TARGET_YEAR} Complete! Saved to {FINAL_CSV_PATH}")
