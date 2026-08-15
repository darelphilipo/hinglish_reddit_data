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
# 0. TELEMETRY & PROFILER SETUP[cite: 1]
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
shortfalls = {cat: max(0, TARGETS[cat] - df.get(cat, pd.Series([0])).sum()) for cat in TARGETS}[cite: 1]

if all(s <= 0 for s in shortfalls.values()):
    stop_telemetry.set()
    print("✅ All toxic target categories met. No harvesting needed.")[cite: 1]
    exit(0)

priority_cat = max(shortfalls, key=shortfalls.get)[cite: 1]
current_shortfall = shortfalls[priority_cat]
print(f"🎯 Harvester Target Identified: {priority_cat.upper()} | Shortfall: {current_shortfall:,} rows")[cite: 1]

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
    static_stopwords = set(line.strip().lower() for line in resp.text.splitlines() if line.strip())[cite: 1]
except Exception:
    static_stopwords = {'hai', 'ki', 'aur', 'mein'}

def basic_tokenize(text): return re.findall(r'\b[a-z0-9]+\b', str(text).lower())[cite: 1]

global_df_freq = collections.Counter()
for doc in df['body'].tolist():
    for word in set(basic_tokenize(doc)): global_df_freq[word] += 1[cite: 1]

dynamic_stopwords = {w for w, count in global_df_freq.items() if count > int(len(df) * 0.15)}[cite: 1]
effective_stopwords = static_stopwords.union(dynamic_stopwords)

target_tf, bg_df_freq = collections.Counter(), collections.Counter()
for doc in df[df[priority_cat] == 1]['body'].tolist(): target_tf.update([w for w in basic_tokenize(doc) if w not in effective_stopwords and len(w)>2])[cite: 1]
for doc in df[df[priority_cat] == 0]['body'].tolist():
    for word in set([w for w in basic_tokenize(doc) if w not in effective_stopwords and len(w)>2]): bg_df_freq[word] += 1[cite: 1]

tfidf_scores = {w: tf * math.log(max(len(bg_df_freq), 1) / (bg_df_freq.get(w, 0) + 1)) for w, tf in target_tf.items()}[cite: 1]
seed_words_only = [w for w, _ in sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10]][cite: 1]

# ==========================================
# 3. LLM MULTIPLIER (SEED EXPANSION)[cite: 1]
# ==========================================
print("\n🧠 [DIAGNOSTIC] Phase 2: Stateful Semantic Expansion")
OPENCODE_KEY = os.environ.get("OPENCODE_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
client = OpenAI(api_key=OPENCODE_KEY, base_url="https://opencode.ai/zen/go/v1")[cite: 1]
MODEL_NAME = "deepseek-v4-flash"

prompt = f"Target category: '{priority_cat}'. We already know these cached terms: {cached_terms[:20]}. Statistically dominant new terms: {seed_words_only}. Generate a JSON object containing a 'keywords' array of 30 completely NEW, highly specific spelling variations, slurs, and slang that users type to bypass filters. Do not repeat known terms. Output strictly JSON: {{\"keywords\": [\"word1\", ...]}}"

try:
    res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})[cite: 1]
    perf_metrics['generator_tokens'] = res.usage.total_tokens[cite: 1]
    llm_keywords = json.loads(res.choices[0].message.content.strip()).get("keywords", [])[cite: 1]
except Exception as e:
    print(f"   ⚠️ LLM failed: {e}. Using statistical seeds only.")
    llm_keywords = []

final_keywords = list(dict.fromkeys([k for k in (cached_terms + seed_words_only + llm_keywords) if len(k) > 3]))[cite: 1]
print(f"   ↳ Final Deduplicated Lexicon ({len(final_keywords)} terms)")

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

harvest_df = pd.DataFrame()
safe_keywords = [k.replace("'", "''").lower() for k in final_keywords][:35]
filter_clauses = " OR ".join([f"LOWER(body) LIKE '%{k}%'" for k in safe_keywords if k])

# --- TIER 3: LIVE ARCTIC API (2025+) ---
def fetch_tier3_live(subreddits, lexicon, max_rows, time_budget=45):
    print(f"      -> [Tier 3] Arctic API Live Search (Budget: {time_budget}s, Quota: {max_rows})...")
    session = requests.Session()[cite: 2]
    start_time = time.time()
    collected = []
    AFTER_2025 = 1735689600 
    
    sampled_subs = random.sample(subreddits, min(len(subreddits), 15))
    sampled_terms = random.sample(lexicon, min(len(lexicon), 10))
    
    for sub in sampled_subs:
        if len(collected) >= max_rows or (time.time() - start_time) > time_budget: break
        for term in sampled_terms:
            if len(collected) >= max_rows or (time.time() - start_time) > time_budget: break
            
            params = {"subreddit": sub, "q": term, "after": AFTER_2025, "limit": 100, "sort": "desc"}[cite: 2]
            try:
                resp = session.get("https://arctic-shift.photon-reddit.com/api/comments/search", params=params, timeout=10)[cite: 2]
                if resp.status_code == 429:
                    print("         ⚠️ Arctic API 429 Rate Limit. Backing off Tier 3.")
                    return pd.DataFrame(collected).drop_duplicates(subset=["id"]) if collected else pd.DataFrame()
                resp.raise_for_status()[cite: 2]
                
                for item in resp.json().get("data", []):[cite: 2]
                    body = item.get("body", "")
                    if body and body not in ["[removed]", "[deleted]"]:[cite: 2]
                        collected.append({"id": item.get("id"), "body": body, "subreddit": sub})[cite: 2]
                time.sleep(1.0)  # Throttling to protect rate limits[cite: 2]
            except Exception:
                continue
    df = pd.DataFrame(collected)
    if not df.empty:
        df = df.drop_duplicates(subset=["id"])
        if len(df) > max_rows: df = df.sample(max_rows)
    return df

t3_df = fetch_tier3_live(TIER2_SUBS, final_keywords, T3_QUOTA)
if not t3_df.empty:
    harvest_df = pd.concat([harvest_df, t3_df], ignore_index=True)
    print(f"         Yield: {len(t3_df)} candidates from Live API.")

# --- TIER 1, 2, & 4: DUCKDB ARCHIVE (2017-2024) ---
con = duckdb.connect()[cite: 1]
con.execute("PRAGMA memory_limit='6GB'; PRAGMA threads=8; INSTALL httpfs; LOAD httpfs;")[cite: 1]
if HF_TOKEN: con.execute(f"CREATE SECRET hf_auth (TYPE HUGGINGFACE, TOKEN '{HF_TOKEN}');")[cite: 1]
api = HfApi(token=HF_TOKEN)[cite: 1]
all_files = api.list_repo_files("open-index/arctic", repo_type="dataset")[cite: 1]

t1_df, t2_df, t4_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
current_year = int(os.environ.get("TARGET_YEAR", 2017))[cite: 1]

while current_year <= 2024 and (len(t1_df) < T1_QUOTA or len(t2_df) < T2_QUOTA):
    year_files = [f for f in all_files if f.endswith('.parquet') and f'data/comments/{current_year}' in f][cite: 1]
    if not year_files:
        current_year += 1; continue
        
    hf_urls = [f"hf://datasets/open-index/arctic/{f}" for f in random.sample(year_files, min(40, len(year_files)))][cite: 1]

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
        t1_df = pd.concat([t1_df, t1_new]).drop_duplicates(subset=['id'])
    
    # Tier 2
    if len(t2_df) < T2_QUOTA:
        t2_new = duckdb_extract(TIER2_SUBS, T2_QUOTA - len(t2_df))
        t2_df = pd.concat([t2_df, t2_new]).drop_duplicates(subset=['id'])
        
    current_year += 1

harvest_df = pd.concat([harvest_df, t1_df, t2_df]).drop_duplicates(subset=['id'])
print(f"      -> DuckDB Yield: Tier 1 ({len(t1_df)}), Tier 2 ({len(t2_df)}). Total gathered: {len(harvest_df)}")

# --- TIER 4: GLOBAL WILDCARD FALLBACK ---
if len(harvest_df) < SOFT_THRESHOLD:
    deficit = SOFT_THRESHOLD - len(harvest_df)
    print(f"      -> [Tier 4] Global Fallback. Deficit: {deficit}...")
    t4_df = duckdb_extract([], deficit)
    harvest_df = pd.concat([harvest_df, t4_df]).drop_duplicates(subset=['id'])
    print(f"         Yield: {len(t4_df)} candidates.")

if harvest_df.empty:
    stop_telemetry.set()
    print("❌ Exhausted all Tiers. No matching candidates found.")
    exit(0)

# ==========================================
# 5. SANITIZATION & AUTONOMOUS LABELING[cite: 1]
# ==========================================
print(f"\n🛡️ [DIAGNOSTIC] Phase 4: Verification of {len(harvest_df)} Candidates")

def sanitize_text(text):
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'/?u/[A-Za-z0-9_-]+', '', text) 
    try: text = clean(text, fix_unicode=True, to_ascii=False, lower=False, no_urls=True, no_emails=True)[cite: 1]
    except Exception: pass
    return re.sub(r'\s{2,}', ' ', text).strip()

harvest_df['body_clean'] = harvest_df['body'].apply(sanitize_text)[cite: 1]
harvest_df['temp_id'] = harvest_df.index.astype(str)[cite: 1]

try: SYSTEM_PROMPT = requests.get("https://raw.githubusercontent.com/darelphilipo/hinglish_reddit_data/main/prompt/System_Prompt", timeout=10).text.strip()[cite: 1]
except Exception as e: stop_telemetry.set(); raise RuntimeError(f"❌ Failed to fetch System Prompt: {e}")[cite: 1]

def label_batch(comments_batch, attempt=1):
    numbered = "\n".join(f'ID: {cid} | Comment: {body}' for cid, body in comments_batch)[cite: 1]
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Label these comments:\n{numbered}"}],
            temperature=0.1, response_format={"type": "json_object"}
        )[cite: 1]
        
        with token_lock:
            perf_metrics['total_prompt_tokens'] += res.usage.prompt_tokens[cite: 1]
            perf_metrics['total_completion_tokens'] += res.usage.completion_tokens[cite: 1]

        raw_content = re.sub(r"^```(?:json)?\n?|\n?```$", "", res.choices[0].message.content.strip())[cite: 1]
        results = json.loads(raw_content).get("results", [])[cite: 1]
        if len(results) == len(comments_batch):
            for idx, item in enumerate(results): item["temp_id"] = str(comments_batch[idx][0])[cite: 1]
            return results
        raise ValueError("Batch size mismatch")[cite: 1]
    except Exception:
        if attempt <= 3:
            time.sleep(min(2 ** attempt, 10))[cite: 1]
            return label_batch(comments_batch, attempt + 1)[cite: 1]
        return []

batches = [list(zip(harvest_df["temp_id"], harvest_df["body_clean"]))[i:i + 20] for i in range(0, len(harvest_df), 20)][cite: 1]
all_labels = []

print(f"\n🚀 Running Parallel Inference Engine ({MAX_WORKERS} Workers)...")[cite: 1]
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:[cite: 1]
    for result in tqdm(executor.map(label_batch, batches), total=len(batches), desc="Labeling Candidates"): [cite: 1]
        all_labels.extend(result)[cite: 1]

labels_df = pd.DataFrame(all_labels)[cite: 1]
if 'id' in labels_df.columns: labels_df.drop(columns=["id"], inplace=True)[cite: 1]
labels_df["temp_id"] = labels_df["temp_id"].astype(str)[cite: 1]
labels_df.rename(columns={"pv": "profanity_vulgarity", "tah": "targeted_abuse_harassment", "dhs": "discriminatory_hate_speech", "cst": "caste", "cr": "communal_religious", "rx": "regional_xenophobic", "mg": "misogyny_gender"}, inplace=True)[cite: 1]

final_df = harvest_df.merge(labels_df, on="temp_id", how="inner").drop(columns=["temp_id", "body_clean"], errors='ignore')[cite: 1]
final_df = final_df[final_df[priority_cat].notna()][cite: 1]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')[cite: 1]
FINAL_PATH = os.path.join(CHUNKS_DIR, f'harvested_tier1_{priority_cat}_{timestamp}.csv')[cite: 1]
final_df.to_csv(FINAL_PATH, index=False)[cite: 1]

stop_telemetry.set()[cite: 1]
monitor_thread.join()[cite: 1]

print("\n==================================================")
print(" 🏁 HARVESTER YIELD & TELEMETRY REPORT")
print(f"✅ Verified data committed to: {FINAL_PATH}")[cite: 1]
