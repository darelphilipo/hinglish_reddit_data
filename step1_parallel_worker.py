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
from concurrent.futures import ThreadPoolExecutor
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
    'total_cache_misses': 0,
    'prompt_fetch_time': 0.0,
    'duckdb_extract_time': 0.0,
    'sanitization_and_balancing_time': 0.0,
    'api_inference_time': 0.0,
    'total_script_time': 0.0
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

BASE_OUTPUT_DIR = './labelled_output/'
CHUNKS_DIR = os.path.join(BASE_OUTPUT_DIR, 'chunks/')
os.makedirs(CHUNKS_DIR, exist_ok=True)

FINAL_CSV_PATH = os.path.join(CHUNKS_DIR, f'baseline_tier1_{TARGET_YEAR}.csv')
LEDGER_PATH = os.path.join(BASE_OUTPUT_DIR, 'seen_ids_ledger.txt')
SUBREDDIT_CONFIG_PATH = "prompt/subreddits.json"
SUBREDDIT_URL = "https://raw.githubusercontent.com/darelphilipo/hinglish_reddit_data/main/prompt/subreddits.json"

SEED_VALUE = int(RUN_ID) % 100000 
random.seed(SEED_VALUE)

TARGET_ROWS_PER_JOB = int(os.environ.get("TARGET_ROWS", 12500))
MAX_WORKERS = 10 

OPENCODE_KEY = os.environ.get("OPENCODE_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not OPENCODE_KEY:
    raise ValueError("❌ OPENCODE_KEY environment variable is missing.")

client = OpenAI(api_key=OPENCODE_KEY, base_url="https://opencode.ai/zen/go/v1")
MODEL_NAME = "deepseek-v4-flash"

# ==========================================
# 2. LOAD SUBREDDITS & DYNAMIC SYSTEM PROMPT
# ==========================================
print(f"\n🔧 Loading Subreddit Configurations (80/20 Target Split)...")
TIER1_SUBS, TIER2_SUBS = [], []
seen_tier2 = set()

# Load subreddits from local file or remote fallback
sub_data = None
if os.path.exists(SUBREDDIT_CONFIG_PATH):
    try:
        with open(SUBREDDIT_CONFIG_PATH, "r", encoding="utf-8") as f:
            sub_data = json.load(f)
    except Exception as e:
        print(f"⚠️ Failed reading local {SUBREDDIT_CONFIG_PATH}: {e}")

if not sub_data:
    try:
        resp = requests.get(SUBREDDIT_URL, timeout=10)
        resp.raise_for_status()
        sub_data = resp.json()
    except Exception as e:
        stop_telemetry.set()
        raise RuntimeError(f"❌ Failed to fetch subreddits.json from GitHub: {e}")

config_toggles = sub_data.get("config", {})
categories = sub_data.get("categories", {})

# Tier 1 = toxicity_focused (80% target)
if config_toggles.get("toxicity_focused", 1) == 1:
    TIER1_SUBS = [s.lower() for s in categories.get("toxicity_focused", [])]

# Tier 2 = all other active categories (20% target)
for cat_name, sub_list in categories.items():
    if cat_name != "toxicity_focused" and config_toggles.get(cat_name, 1) == 1:
        for s in sub_list:
            s_clean = s.lower()
            if s_clean not in seen_tier2 and s_clean not in TIER1_SUBS:
                seen_tier2.add(s_clean)
                TIER2_SUBS.append(s_clean)

T1_QUOTA = max(1, int(TARGET_ROWS_PER_JOB * 0.80))
T2_QUOTA = max(1, TARGET_ROWS_PER_JOB - T1_QUOTA)

print(f"   ↳ Tier 1 (Toxicity Focused): {len(TIER1_SUBS)} subreddits | Quota: {T1_QUOTA:,} rows")
print(f"   ↳ Tier 2 (General / Regional): {len(TIER2_SUBS)} subreddits | Quota: {T2_QUOTA:,} rows")

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
# 3. DUCKDB EXTRACTION (WITH BATCHED RATE-LIMIT SAFETY)
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

max_shards = 10 if TARGET_ROWS_PER_JOB < 1000 else 200
selected_shards = random.sample(year_files, min(max_shards, len(year_files)))

print(f"⏳ Extracting records for {TARGET_YEAR} across {len(selected_shards)} shards (Batched to prevent 429)...")

duckdb_running = True
def duckdb_heartbeat():
    start_time = time.time()
    net_start = psutil.net_io_counters().bytes_recv
    last_net = net_start
    while duckdb_running:
        time.sleep(15)
        if not duckdb_running: break
        current_net = psutil.net_io_counters().bytes_recv
        total_downloaded_mb = (current_net - net_start) / (1024 * 1024)
        speed_mb_s = ((current_net - last_net) / (1024 * 1024)) / 15
        last_net = current_net
        print(f"   📡 [HF Network] Pulled: {total_downloaded_mb:.1f} MB | Speed: {speed_mb_s:.1f} MB/s")

heartbeat_thread = threading.Thread(target=duckdb_heartbeat, daemon=True)
heartbeat_thread.start()

def extract_subreddits(sub_list, tier_name):
    if not sub_list:
        return pd.DataFrame()
    subs_formatted = ", ".join([f"'{s.replace(chr(39), chr(39)+chr(39))}'" for s in sub_list])
    raw_list = []
    chunk_size = 25
    
    for i in range(0, len(selected_shards), chunk_size):
        shard_chunk = selected_shards[i:i + chunk_size]
        hf_urls = [f"hf://datasets/open-index/arctic/{f}" for f in shard_chunk]
        
        query = f"""
        SELECT id, body, LOWER(subreddit) as subreddit, created_utc, strftime(epoch_ms(created_utc * 1000), '%Y-%m') as year_month, '{tier_name}' as tier_label
        FROM read_parquet({hf_urls})
        WHERE LOWER(subreddit) IN ({subs_formatted})
          AND body NOT IN ('[deleted]', '[removed]', '')
          AND length(body) BETWEEN 10 AND 1000
        """
        try:
            temp_df = con.query(query).to_df()
            raw_list.append(temp_df)
        except Exception as e:
            print(f"   ⚠️ [{tier_name}] Rate Limit on batch {i//chunk_size + 1}: {e}. Skipping chunk.")
            time.sleep(5)
            
        time.sleep(1)
        
    return pd.concat(raw_list, ignore_index=True) if raw_list else pd.DataFrame()

try:
    t1_raw_df = extract_subreddits(TIER1_SUBS, "Tier 1")
    t2_raw_df = extract_subreddits(TIER2_SUBS, "Tier 2")
    raw_df = pd.concat([t1_raw_df, t2_raw_df], ignore_index=True)
    perf_metrics['duckdb_extract_time'] = time.time() - db_start
    print(f"📦 Pulled raw comments: Tier 1 = {len(t1_raw_df):,} | Tier 2 = {len(t2_raw_df):,} | Total = {len(raw_df):,}")
    
    if raw_df.empty:
        raise ValueError("All extraction batches failed due to rate limits.")

except Exception as e:
    stop_telemetry.set()
    raise RuntimeError(f"❌ DuckDB Extraction crashed: {e}")
finally:
    duckdb_running = False
    heartbeat_thread.join()

# ==========================================
# 4. SANITIZATION & DEDUPLICATION
# ==========================================
sanitize_start = time.time()
print("\n🧹 Sanitizing text...")

def sanitize_text(text):
    if not isinstance(text, str): return ""
    
    # 1. Unescape HTML entities (&gt; becomes >)
    text = html.unescape(text)
    
    # 2. Strip actual HTML tags using Regex (Safe)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. Replace newlines and tabs with a space
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # 4. Clean Reddit specific formatting
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'/?u/[A-Za-z0-9_-]+', '', text) 
    
    # 5. Strip zero-width characters
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    
    try:
        # 6. Clean text normalization
        text = clean(text, 
            fix_unicode=True, 
            to_ascii=False, 
            lower=False, 
            no_line_breaks=True, 
            no_urls=True, replace_with_url="",
            no_emails=True, replace_with_email="",
            no_phone_numbers=True, replace_with_phone_number=""
        )
    except Exception:
        pass
        
    # 7. Collapse multi-spaces into single space
    return re.sub(r'\s{2,}', ' ', text).strip()

raw_df['body_clean'] = raw_df['body'].apply(sanitize_text)
raw_df = raw_df[raw_df['body_clean'].str.len() > 5]

print("🛡️ Applying aggressive token-saving deduplication...")
initial_count = len(raw_df)
raw_df['dedup_key'] = raw_df['body_clean'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
raw_df.drop_duplicates(subset=['dedup_key'], keep='first', inplace=True)
raw_df.drop(columns=['dedup_key'], inplace=True)
print(f"✂️ Dropped {initial_count - len(raw_df)} spam/copy-paste variants.")

if os.path.exists(LEDGER_PATH):
    with open(LEDGER_PATH, 'r') as f:
        seen = set(line.strip() for line in f if line.strip())
    prev_len = len(raw_df)
    raw_df = raw_df[~raw_df['id'].isin(seen)]
    print(f"🛡️ Filtered out {prev_len - len(raw_df)} comments processed in previous workflow runs.")

# ==========================================
# 5. STRATIFIED 80/20 BALANCING
# ==========================================
print("\n⚖️ Balancing sampling evenly across Subreddits, Months, and Tiers...")

def balance_tier_pool(tier_df, quota):
    if tier_df.empty:
        return pd.DataFrame()
    tier_df = tier_df.copy()
    tier_df['bucket'] = tier_df['subreddit'].astype(str) + "_" + tier_df['year_month'].astype(str)
    groups = tier_df['bucket'].unique()
    sampled_indices = []
    
    if len(groups) > 0:
        samples_per_bucket = max(1, quota // len(groups))
        for bucket in groups:
            b_rows = tier_df[tier_df['bucket'] == bucket]
            sampled_indices.extend(b_rows.sample(n=min(len(b_rows), samples_per_bucket), random_state=SEED_VALUE).index.tolist())
            
    balanced = tier_df.loc[sampled_indices].reset_index(drop=True)
    
    if len(balanced) < quota:
        needed = quota - len(balanced)
        remaining = tier_df.index.difference(balanced.index)
        if not remaining.empty:
            balanced = pd.concat([balanced, tier_df.loc[remaining].sample(n=min(len(remaining), needed), random_state=SEED_VALUE)], ignore_index=True)
            
    return balanced

t1_balanced = balance_tier_pool(raw_df[raw_df['tier_label'] == "Tier 1"], T1_QUOTA)
t2_balanced = balance_tier_pool(raw_df[raw_df['tier_label'] == "Tier 2"], T2_QUOTA)

# Combine tiers and handle surplus/deficit to match TARGET_ROWS_PER_JOB
df = pd.concat([t1_balanced, t2_balanced], ignore_index=True)

if len(df) < TARGET_ROWS_PER_JOB:
    deficit = TARGET_ROWS_PER_JOB - len(df)
    remaining_raw = raw_df.index.difference(df.index)
    if not remaining_raw.empty:
        df = pd.concat([df, raw_df.loc[remaining_raw].sample(n=min(len(remaining_raw), deficit), random_state=SEED_VALUE)], ignore_index=True)

df.drop(columns=['bucket', 'index', 'tier_label'], inplace=True, errors='ignore')
df['temp_id'] = df.index.astype(str)
perf_metrics['sanitization_and_balancing_time'] = time.time() - sanitize_start

print(f"🎯 Final Balanced Pool for {TARGET_YEAR}: {len(df):,} rows (Tier 1: {len(t1_balanced):,} | Tier 2: {len(t2_balanced):,}).")

with open(LEDGER_PATH, 'a') as f:
    for cid in df['id'].tolist(): f.write(f"{cid}\n")

# ==========================================
# 6. INFERENCE ENGINE (THREAD-SAFE & FIXED MERGE)
# ==========================================
api_start = time.time()

def label_batch(comments_batch, attempt=1):
    numbered = "\n".join(f'ID: {cid} | Comment: {body}' for cid, body in comments_batch)
    user_prompt = f"Label these comments:\n{numbered}"
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}], 
            temperature=0.1, 
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}}
        )
        
        usage_dict = res.usage.model_dump() if hasattr(res.usage, 'model_dump') else vars(res.usage)
        token_details = usage_dict.get('prompt_tokens_details', {}) or {}
        p_tokens = usage_dict.get('prompt_tokens', 0)
        c_tokens = usage_dict.get('completion_tokens', 0)
        t_tokens = usage_dict.get('total_tokens', p_tokens + c_tokens)
        c_hits = usage_dict.get('prompt_cache_hit_tokens', token_details.get('cached_tokens', 0))
        c_misses = usage_dict.get('prompt_cache_miss_tokens', p_tokens - c_hits)

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

        content = json.loads(raw_content)
        results = content.get("results", [])
        
        if len(results) == len(comments_batch):
            for idx, item in enumerate(results): item["temp_id"] = str(comments_batch[idx][0])
            return results
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

print(f"\n🚀 Running Parallel Inference on {len(df):,} rows across {len(batches):,} batches...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for result in tqdm(executor.map(label_batch, batches), total=len(batches), desc=f"Labeling {TARGET_YEAR}"): 
        all_labels.extend(result)

labels_df = pd.DataFrame(all_labels)

if 'id' in labels_df.columns: 
    labels_df.drop(columns=["id"], inplace=True)
labels_df["temp_id"] = labels_df["temp_id"].astype(str)

final_df = df.merge(labels_df, on="temp_id", how="left")
final_df.drop(columns=["temp_id", "body_clean"], errors='ignore', inplace=True)
final_df.to_csv(FINAL_CSV_PATH, index=False)

perf_metrics['api_inference_time'] = time.time() - api_start
perf_metrics['total_script_time'] = time.time() - script_start_time
stop_telemetry.set()
monitor_thread.join()

# ==========================================
# 7. COMPLETE PERFORMANCE & RESOURCE LOG
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
