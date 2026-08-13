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
CHUNKS_DIR = os.path.join(BASE_OUTPUT_DIR, 'chunks/')
os.makedirs(CHUNKS_DIR, exist_ok=True) # Ensure chunks folder exists

# All yearly chunks go into the chunks subfolder
FINAL_CSV_PATH = os.path.join(CHUNKS_DIR, f'baseline_tier1_{TARGET_YEAR}.csv')
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

# Dynamic shard sampling
max_shards = 10 if TARGET_ROWS_PER_JOB < 1000 else 200
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
        if not duckdb_running: break
        current_net = psutil.net_io_counters().bytes_recv
        total_downloaded_mb = (current_net - net_start) / (1024 * 1024)
        speed_mb_s = ((current_net - last_net) / (1024 * 1024)) / 15
        last_net = current_net
        print(f"   📡 [HF Network] Pulled: {total_downloaded_mb:.1f} MB | Speed: {speed_mb_s:.1f} MB/s")

heartbeat_thread = threading.Thread(target=duckdb_heartbeat, daemon=True)
heartbeat_thread.start()

try:
    raw_df = con.query(query).to_df()
    perf_metrics['duckdb_extract_time'] = time.time() - db_start
    print(f"📦 Pulled {len(raw_df)} raw comments.")
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
    text = html.unescape(text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'/?u/[A-Za-z0-9_-]+', '', text) 
    return clean(text, fix_unicode=True, to_ascii=False, lower=False, no_line_breaks=True, no_urls=True, no_emails=True, no_phone_numbers=True, no_html=True).strip()

raw_df['body_clean'] = raw_df['body'].apply(sanitize_text)
raw_df = raw_df[raw_df['body_clean'].str.len() > 5]

raw_df['dedup_key'] = raw_df['body_clean'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
raw_df.drop_duplicates(subset=['dedup_key'], keep='first', inplace=True)
raw_df.drop(columns=['dedup_key'], inplace=True)

if os.path.exists(LEDGER_PATH):
    with open(LEDGER_PATH, 'r') as f:
        seen = set(line.strip() for line in f if line.strip())
    raw_df = raw_df[~raw_df['id'].isin(seen)]

# ==========================================
# 5. BULLETPROOF STRATIFIED BALANCING
# ==========================================
raw_df['bucket'] = raw_df['subreddit'].astype(str) + "_" + raw_df['year_month'].astype(str)
groups = raw_df['bucket'].unique()

sampled_indices = []
samples_per_bucket = max(1, TARGET_ROWS_PER_JOB // len(groups))
for bucket in groups:
    b_rows = raw_df[raw_df['bucket'] == bucket]
    sampled_indices.extend(b_rows.sample(n=min(len(b_rows), samples_per_bucket), random_state=SEED_VALUE).index.tolist())

df = raw_df.loc[sampled_indices].reset_index(drop=True)
if len(df) < TARGET_ROWS_PER_JOB:
    needed = TARGET_ROWS_PER_JOB - len(df)
    remaining = raw_df.index.difference(df.index)
    if not remaining.empty:
        df = pd.concat([df, raw_df.loc[remaining].sample(n=min(len(remaining), needed), random_state=SEED_VALUE)], ignore_index=True)

df.drop(columns=['bucket'], inplace=True)
df['temp_id'] = df.index.astype(str)
perf_metrics['sanitization_and_balancing_time'] = time.time() - sanitize_start

with open(LEDGER_PATH, 'a') as f:
    for cid in df['id'].tolist(): f.write(f"{cid}\n")

# ==========================================
# 6. INFERENCE ENGINE (FIXED MERGE)
# ==========================================
api_start = time.time()

def label_batch(comments_batch, attempt=1):
    numbered = "\n".join(f'ID: {cid} | Comment: {body}' for cid, body in comments_batch)
    user_prompt = f"Label these comments:\n{numbered}"
    try:
        res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}], temperature=0.1, response_format={"type": "json_object"})
        
        usage = res.usage.model_dump()
        p, c = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
        with token_lock:
            perf_metrics['total_prompt_tokens'] += p
            perf_metrics['total_completion_tokens'] += c
            perf_metrics['total_combined_tokens'] += p + c

        content = json.loads(res.choices[0].message.content)
        results = content.get("results", [])
        for idx, item in enumerate(results): item["temp_id"] = str(comments_batch[idx][0])
        return results
    except: return []

batches = [list(zip(df["temp_id"], df["body_clean"]))[i:i + 20] for i in range(0, len(df), 20)]
all_labels = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for result in executor.map(label_batch, batches): all_labels.extend(result)

labels_df = pd.DataFrame(all_labels)
if 'id' in labels_df.columns: labels_df.drop(columns=["id"], inplace=True)
labels_df["temp_id"] = labels_df["temp_id"].astype(str)

final_df = df.merge(labels_df, on="temp_id", how="left")
final_df.drop(columns=["temp_id", "body_clean"], errors='ignore', inplace=True)
final_df.to_csv(FINAL_CSV_PATH, index=False)

# ==========================================
# 7. PERFORMANCE LOG
# ==========================================
print("\n==================================================")
print(f"✅ Worker {TARGET_YEAR} Complete! Saved to {FINAL_CSV_PATH}")
