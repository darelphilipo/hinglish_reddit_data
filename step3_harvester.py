import pandas as pd
import duckdb
import os
import math
import collections
import re
import json
import random
import requests
import time
import threading
import psutil
import html
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from tqdm import tqdm
from openai import OpenAI
from huggingface_hub import HfApi
from cleantext import clean

print("🌾 Initializing Autonomous Harvester: 4-Tier Hybrid Engine...")

# ==========================================
# 0. TELEMETRY & PROFILER SETUP
# ==========================================
MAX_WORKERS = 10
perf_metrics = {
    'total_prompt_tokens': 0, 'total_completion_tokens': 0, 'total_combined_tokens': 0,
    'total_cache_hits': 0, 'total_cache_misses': 0, 'generator_tokens': 0
}
token_lock = threading.Lock()
stop_telemetry = threading.Event()
script_start_time = time.time()

def resource_monitor():
    while not stop_telemetry.is_set():
        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        print(f"   [⚙️ SYSTEM] RAM: {ram.used / (1024**3):.2f}GB ({ram.percent}%) | CPU: {cpu}%")
        time.sleep(30) 

monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
monitor_thread.start()

# ==========================================
# 1. LOAD CONFIGS & CALCULATE GOALS
# ==========================================
BASE_OUTPUT_DIR = './labelled_output/'
CHUNKS_DIR = os.path.join(BASE_OUTPUT_DIR, 'chunks/')
os.makedirs(CHUNKS_DIR, exist_ok=True)
MASTER_PATH = os.path.join(BASE_OUTPUT_DIR, 'master_baseline_tier1.csv')
TARGETS_PATH = os.path.join(BASE_OUTPUT_DIR, 'pipeline_targets.json')
SUBREDDIT_CONFIG_PATH = "prompt/subreddits.json"
LEXICON_CONFIG_PATH = "prompt/master_lexicon.json"

if not os.path.exists(MASTER_PATH) or not os.path.exists(TARGETS_PATH):
    print("❌ Master dataset or Blueprint JSON not found. Run Step 1 & 2 first.")
    exit(1)

with open(TARGETS_PATH, 'r') as f:
    blueprint = json.load(f)
TARGETS = blueprint.get("categories", {})

df = pd.read_csv(MASTER_PATH)
shortfalls = {cat: max(0, TARGETS[cat] - df.get(cat, pd.Series([0])).sum()) for cat in TARGETS}

if all(s <= 0 for s in shortfalls.values()):
    stop_telemetry.set()
    print("✅ All toxic target categories met. No harvesting needed.")
    exit(0)

priority_cat = max(shortfalls, key=shortfalls.get)
current_shortfall = shortfalls[priority_cat]
print(f"🎯 Harvester Target Identified: {priority_cat.upper()} | Shortfall: {current_shortfall:,} rows")

# --- LOAD SUBREDDITS & APPLY TOGGLES ---
TIER1_SUBS, TIER2_SUBS = [], []
seen_tier2 = set()
try:
    with open(SUBREDDIT_CONFIG_PATH, "r", encoding="utf-8") as f:
        sub_data = json.load(f)
    config_toggles = sub_data.get("config", {})
    categories = sub_data.get("categories", {})
    
    for cat_name, sub_list in categories.items():
        if config_toggles.get(cat_name, 1) == 1:
            if cat_name == "toxicity_focused":
                TIER1_SUBS.extend(sub_list)
            else:
                for s in sub_list:
                    if s.lower() not in seen_tier2:
                        seen_tier2.add(s.lower())
                        TIER2_SUBS.append(s)
    TIER1_SUBS = list(set(TIER1_SUBS))
    print(f"🔧 Loaded Config: Tier 1 ({len(TIER1_SUBS)} subs), Tier 2/3 ({len(TIER2_SUBS)} subs)")
except Exception as e:
    print(f"⚠️ Failed to load {SUBREDDIT_CONFIG_PATH}: {e}. Falling back to default list.")
    TIER1_SUBS = ['chodi', 'bakchodi', 'IndiaSpeaks']
    TIER2_SUBS = ['india', 'delhi', 'mumbai']

# --- LOAD STATEFUL LEXICON ---
master_lexicon = {}
if os.path.exists(LEXICON_CONFIG_PATH):
    try:
        with open(LEXICON_CONFIG_PATH, "r", encoding="utf-8") as f:
            master_lexicon = json.load(f)
    except Exception: pass
cached_terms = master_lexicon.get(priority_cat.upper(), [])

# ==========================================
# 2. DYNAMIC THRESHOLDING & TF-IDF
# ==========================================
print("\n📊 [DIAGNOSTIC] Phase 1: Dynamic Thresholding & Statistical Seed Generation")
STOPWORDS_URL = "https://raw.githubusercontent.com/darelphilipo/hinglish_reddit_data/main/prompt/hinglish_stopwords.txt"
try:
    resp = requests.get(STOPWORDS_URL, timeout=10)
    static_stopwords = set(line.strip().lower() for line in resp.text.splitlines() if line.strip())
except Exception:
    static_stopwords = {'hai', 'ki', 'aur', 'mein'}

def basic_tokenize(text): return re.findall(r'\b[a-z0-9]+\b', str(text).lower())

global_df_freq = collections.Counter()
# Added Progress Bar for Global Corpus Analysis
for doc in tqdm(df['body'].tolist(), desc="   ↳ Analyzing Global Corpus", leave=False):
    for word in set(basic_tokenize(doc)): global_df_freq[word] += 1

dynamic_stopwords = {w for w, count in global_df_freq.items() if count > int(len(df) * 0.15)}
effective_stopwords = static_stopwords.union(dynamic_stopwords)
print(f"   ↳ Dynamic Threshold Triggered: Auto-purged {len(dynamic_stopwords)} highly frequent filler words.")

target_tf, bg_df_freq = collections.Counter(), collections.Counter()
for doc in df[df[priority_cat] == 1]['body'].tolist(): target_tf.update([w for w in basic_tokenize(doc) if w not in effective_stopwords and len(w)>2])
for doc in df[df[priority_cat] == 0]['body'].tolist():
    for word in set([w for w in basic_tokenize(doc) if w not in effective_stopwords and len(w)>2]): bg_df_freq[word] += 1

tfidf_scores = {w: tf * math.log(max(len(bg_df_freq), 1) / (bg_df_freq.get(w, 0) + 1)) for w, tf in target_tf.items()}
seed_words_only = [w for w, _ in sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10]]
print(f"   ↳ Statistical Seeds Extracted: {seed_words_only}")

# ==========================================
# 3. LLM MULTIPLIER (SEED EXPANSION)
# ==========================================
print("\n🧠 [DIAGNOSTIC] Phase 2: Stateful Semantic Expansion")
OPENCODE_KEY = os.environ.get("OPENCODE_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
client = OpenAI(api_key=OPENCODE_KEY, base_url="https://opencode.ai/zen/go/v1")
MODEL_NAME = "deepseek-v4-flash"

prompt = f"Target category: '{priority_cat}'. We already know these cached terms: {cached_terms[:20]}. Statistically dominant new terms: {seed_words_only}. Generate a JSON object containing a 'keywords' array of 30 completely NEW, highly specific spelling variations, slurs, and slang that users type to bypass filters. Do not repeat known terms. Output strictly JSON: {{\"keywords\": [\"word1\", ...]}}"

try:
    res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
    perf_metrics['generator_tokens'] = res.usage.total_tokens
    llm_keywords = json.loads(res.choices[0].message.content.strip()).get("keywords", [])
    print(f"   ↳ LLM Generated Terms: {llm_keywords}")
except Exception as e:
    print(f"   ⚠️ LLM failed: {e}. Using statistical seeds only.")
    llm_keywords = []

final_keywords = list(dict.fromkeys([k for k in (cached_terms + seed_words_only + llm_keywords) if len(k) > 3]))
print(f"   ↳ Final Deduplicated Lexicon ({len(final_keywords)} terms): {final_keywords}")

# Persist to cache
master_lexicon[priority_cat.upper()] = final_keywords
try:
    with open(LEXICON_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(master_lexicon, f, indent=2, ensure_ascii=False)
except Exception: pass

# ==========================================
# 4. 4-TIER QUOTA EXTRACTION ENGINE
# ==========================================
print("\n🦆 [DIAGNOSTIC] Phase 3: Quota-Controlled Extraction")
CANDIDATE_THRESHOLD = 500
SOFT_THRESHOLD = int(CANDIDATE_THRESHOLD * 0.8)

# Fractional Quotas
T1_QUOTA = max(1, int(SOFT_THRESHOLD * 0.15))
T2_QUOTA = max(1, int(SOFT_THRESHOLD * 0.55))
T3_QUOTA = max(1, int(SOFT_THRESHOLD * 0.25))
T4_QUOTA = max(1, int(SOFT_THRESHOLD * 0.05))

print(f"   ↳ Target Candidates: {SOFT_THRESHOLD} | Quotas -> T1: {T1_QUOTA} | T2: {T2_QUOTA} | T3: {T3_QUOTA} | T4: {T4_QUOTA}")

harvest_df = pd.DataFrame()
safe_keywords = [k.replace("'", "''").lower() for k in final_keywords][:35]
filter_clauses = " OR ".join([f"LOWER(body) LIKE '%{k}%'" for k in safe_keywords if k])

# Statistics Tracker
extraction_stats = {
    "Tier 1 (Core)": collections.defaultdict(int),
    "Tier 2 (Expanded)": collections.defaultdict(int),
    "Tier 3 (Live API)": collections.defaultdict(int),
    "Tier 4 (Wildcard)": collections.defaultdict(int)
}

# --- TIER 3: LIVE ARCTIC API (2025+) ---
def fetch_tier3_live(subreddits, lexicon, max_rows, time_budget=45):
    session = requests.Session()
    start_time = time.time()
    collected = []
    AFTER_2025 = 1735689600 
    
    if not subreddits or not lexicon:
        return pd.DataFrame()

    sampled_subs = random.sample(subreddits, min(len(subreddits), 15))
    sampled_terms = random.sample(lexicon, min(len(lexicon), 10))
    
    total_iters = len(sampled_subs) * len(sampled_terms)
    
    # Progress bar for Tier 3 API Calls
    with tqdm(total=total_iters, desc="      -> [Tier 3] Arctic API Live Search", leave=False) as pbar:
        for sub in sampled_subs:
            if len(collected) >= max_rows or (time.time() - start_time) > time_budget: break
            for term in sampled_terms:
                if len(collected) >= max_rows or (time.time() - start_time) > time_budget: break
                
                params = {"subreddit": sub, "q": term, "after": AFTER_2025, "limit": 100, "sort": "desc"}
                try:
                    resp = session.get("https://arctic-shift.photon-reddit.com/api/comments/search", params=params, timeout=10)
                    if resp.status_code == 429:
                        return pd.DataFrame(collected).drop_duplicates(subset=["id"]) if collected else pd.DataFrame()
                    resp.raise_for_status()
                    
                    for item in resp.json().get("data", []):
                        body = item.get("body", "")
                        if body and body not in ["[removed]", "[deleted]"]:
                            collected.append({"id": item.get("id"), "body": body, "subreddit": sub, "created_utc": item.get("created_utc")})
                    time.sleep(1.0)
                except Exception:
                    pass
                finally:
                    pbar.update(1)
                    
    df = pd.DataFrame(collected)
    if not df.empty:
        df = df.drop_duplicates(subset=["id"])
        if len(df) > max_rows: df = df.sample(max_rows)
    return df

t3_df = fetch_tier3_live(TIER2_SUBS, final_keywords, T3_QUOTA)
if not t3_df.empty:
    harvest_df = pd.concat([harvest_df, t3_df], ignore_index=True)
    extraction_stats["Tier 3 (Live API)"]["2025+"] = len(t3_df)

# --- TIER 1, 2, & 4: DUCKDB ARCHIVE (2017-2024) ---
con = duckdb.connect()
con.execute("PRAGMA memory_limit='6GB'; PRAGMA threads=8; INSTALL httpfs; LOAD httpfs;")
if HF_TOKEN: con.execute(f"CREATE SECRET hf_auth (TYPE HUGGINGFACE, TOKEN '{HF_TOKEN}');")
api = HfApi(token=HF_TOKEN)
all_files = api.list_repo_files("open-index/arctic", repo_type="dataset")

t1_df, t2_df, t4_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
current_year = int(os.environ.get("TARGET_YEAR", 2017))

# Progress bar for DuckDB Time Travel
with tqdm(total=(2024 - current_year + 1), desc="      -> [Tier 1 & 2] DuckDB Time Travel", leave=False) as pbar:
    while current_year <= 2024 and (len(t1_df) < T1_QUOTA or len(t2_df) < T2_QUOTA):
        year_files = [f for f in all_files if f.endswith('.parquet') and f'data/comments/{current_year}' in f]
        if not year_files:
            current_year += 1; pbar.update(1); continue
            
        hf_urls = [f"hf://datasets/open-index/arctic/{f}" for f in random.sample(year_files, min(40, len(year_files)))]

        def duckdb_extract(sub_list, quota, limit=5000):
            if not sub_list:
                sub_clause = ""
            else:
                subs_formatted = ", ".join([f"'{s.replace(chr(39), chr(39)+chr(39))}'" for s in sub_list])
                sub_clause = f"AND LOWER(subreddit) IN ({subs_formatted})"
                
            query = f"""
            SELECT id, body, LOWER(subreddit) as subreddit 
            FROM read_parquet({hf_urls}) 
            WHERE ({filter_clauses}) {sub_clause} AND body NOT IN ('[deleted]', '[removed]', '') 
            LIMIT {limit}
            """
            res_df = con.query(query).to_df()
            if len(res_df) > quota: res_df = res_df.sample(quota)
            return res_df

        # Tier 1
        if len(t1_df) < T1_QUOTA:
            t1_new = duckdb_extract(TIER1_SUBS, T1_QUOTA - len(t1_df))
            if not t1_new.empty:
                t1_df = pd.concat([t1_df, t1_new]).drop_duplicates(subset=['id'])
                extraction_stats["Tier 1 (Core)"][current_year] += len(t1_new)
        
        # Tier 2
        if len(t2_df) < T2_QUOTA:
            t2_new = duckdb_extract(TIER2_SUBS, T2_QUOTA - len(t2_df))
            if not t2_new.empty:
                t2_df = pd.concat([t2_df, t2_new]).drop_duplicates(subset=['id'])
                extraction_stats["Tier 2 (Expanded)"][current_year] += len(t2_new)
            
        current_year += 1
        pbar.update(1)

if not t1_df.empty or not t2_df.empty:
    harvest_df = pd.concat([harvest_df, t1_df, t2_df]).drop_duplicates(subset=['id'])

# --- TIER 4: GLOBAL WILDCARD FALLBACK ---
if len(harvest_df) < SOFT_THRESHOLD:
    deficit = SOFT_THRESHOLD - len(harvest_df)
    t4_year = 2024
    print(f"      -> [Tier 4] Global Fallback Activated. Hunting for {deficit} rows in {t4_year}...")
    t4_files = [f for f in all_files if f.endswith('.parquet') and f'data/comments/{t4_year}' in f]
    if t4_files:
        hf_urls = [f"hf://datasets/open-index/arctic/{f}" for f in random.sample(t4_files, min(40, len(t4_files)))]
        
        query = f"""
        SELECT id, body, LOWER(subreddit) as subreddit 
        FROM read_parquet({hf_urls}) 
        WHERE ({filter_clauses}) AND body NOT IN ('[deleted]', '[removed]', '') 
        LIMIT 5000
        """
        t4_df = con.query(query).to_df()
        if len(t4_df) > deficit: t4_df = t4_df.sample(deficit)
        
        if not t4_df.empty:
            harvest_df = pd.concat([harvest_df, t4_df]).drop_duplicates(subset=['id'])
            extraction_stats["Tier 4 (Wildcard)"][t4_year] += len(t4_df)

if harvest_df.empty:
    stop_telemetry.set()
    print("❌ Exhausted all Tiers. No matching candidates found.")
    exit(0)

# --- PRINT STATISTICAL BREAKDOWN ---
print("\n📈 [EXTRACTION YIELD STATISTICS]")
for tier, year_data in extraction_stats.items():
    total_rows = sum(year_data.values())
    if total_rows > 0:
        year_breakdown = " | ".join([f"{y}: {count} rows" for y, count in sorted(year_data.items())])
        print(f"   ✅ {tier}: {total_rows} total rows -> ({year_breakdown})")
print(f"   -------------------------------------------------")
print(f"   🚀 Total Harvested Pool: {len(harvest_df)} Candidates")

# ==========================================
# 5. SANITIZATION & AUTONOMOUS LABELING
# ==========================================
print(f"\n🛡️ [DIAGNOSTIC] Phase 4: Verification of Candidates")

def sanitize_text(text):
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'/?u/[A-Za-z0-9_-]+', '', text) 
    try: text = clean(text, fix_unicode=True, to_ascii=False, lower=False, no_urls=True, no_emails=True)
    except Exception: pass
    return re.sub(r'\s{2,}', ' ', text).strip()

# Progress bar for sanitization
tqdm.pandas(desc="   ↳ Sanitizing text data", leave=False)
harvest_df['body_clean'] = harvest_df['body'].progress_apply(sanitize_text)
harvest_df['temp_id'] = harvest_df.index.astype(str)

try: SYSTEM_PROMPT = requests.get("https://raw.githubusercontent.com/darelphilipo/hinglish_reddit_data/main/prompt/System_Prompt", timeout=10).text.strip()
except Exception as e: stop_telemetry.set(); raise RuntimeError(f"❌ Failed to fetch System Prompt: {e}")

def label_batch(comments_batch, attempt=1):
    numbered = "\n".join(f'ID: {cid} | Comment: {body}' for cid, body in comments_batch)
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Label these comments:\n{numbered}"}],
            temperature=0.1, response_format={"type": "json_object"}
        )
        
        with token_lock:
            perf_metrics['total_prompt_tokens'] += res.usage.prompt_tokens
            perf_metrics['total_completion_tokens'] += res.usage.completion_tokens

        raw_content = re.sub(r"^```(?:json)?\n?|\n?```$", "", res.choices[0].message.content.strip())
        results = json.loads(raw_content).get("results", [])
        if len(results) == len(comments_batch):
            for idx, item in enumerate(results): item["temp_id"] = str(comments_batch[idx][0])
            return results
        raise ValueError("Batch size mismatch")
    except Exception:
        if attempt <= 3:
            time.sleep(min(2 ** attempt, 10))
            return label_batch(comments_batch, attempt + 1)
        return []

batches = [list(zip(harvest_df["temp_id"], harvest_df["body_clean"]))[i:i + 20] for i in range(0, len(harvest_df), 20)]
all_labels = []

print(f"\n🚀 Running Parallel Inference Engine ({MAX_WORKERS} Workers)...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for result in tqdm(executor.map(label_batch, batches), total=len(batches), desc="Labeling Candidates"): 
        all_labels.extend(result)

labels_df = pd.DataFrame(all_labels)
if 'id' in labels_df.columns: labels_df.drop(columns=["id"], inplace=True)
labels_df["temp_id"] = labels_df["temp_id"].astype(str)
labels_df.rename(columns={"pv": "profanity_vulgarity", "tah": "targeted_abuse_harassment", "dhs": "discriminatory_hate_speech", "cst": "caste", "cr": "communal_religious", "rx": "regional_xenophobic", "mg": "misogyny_gender"}, inplace=True)

final_df = harvest_df.merge(labels_df, on="temp_id", how="inner").drop(columns=["temp_id", "body_clean"], errors='ignore')
final_df = final_df[final_df[priority_cat].notna()]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
FINAL_PATH = os.path.join(CHUNKS_DIR, f'harvested_tier1_{priority_cat}_{timestamp}.csv')
final_df.to_csv(FINAL_PATH, index=False)

stop_telemetry.set()
monitor_thread.join()

print("\n==================================================")
print(" 🏁 HARVESTER YIELD & TELEMETRY REPORT")
print(f"✅ Verified data committed to: {FINAL_PATH}")
