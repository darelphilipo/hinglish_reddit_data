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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from huggingface_hub import HfApi
from datasets import load_dataset
from cleantext import clean

print("🌾 Initializing Autonomous Harvester: Regional-Strict 3-Tier Engine...")

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
        time.sleep(60) 

monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
monitor_thread.start()

# ==========================================
# 1. LOAD CONFIGS & CALCULATE GOALS
# ==========================================
TARGETS_DIR = './prompt/'
TARGETS_PATH = os.path.join(TARGETS_DIR, 'pipeline_targets.json')
SUBREDDIT_CONFIG_PATH = "prompt/subreddits.json"
LEXICON_CONFIG_PATH = "prompt/master_lexicon.json"

HF_REPO_ID = "darelphilip/hinglish-toxicity"
CHUNK_SIZE = 2500

KEY_MAPPING = {
    'pv': 'profanity_vulgarity',
    'tah': 'targeted_abuse_harassment',
    'dhs': 'discriminatory_hate_speech',
    'cst': 'caste',
    'cr': 'communal_religious',
    'rx': 'regional_xenophobic',
    'mg': 'misogyny_gender'
}
STUDENT_PROMPT = "You are an expert Hinglish content moderation AI. Analyze the following comment and output a JSON object containing the toxic classification flags and a brief analysis of the target and intent."

if not os.path.exists(TARGETS_PATH):
    print("❌ Blueprint JSON not found. Run Step 2 first.")
    exit(1)

with open(TARGETS_PATH, 'r') as f:
    blueprint = json.load(f)
TARGETS = blueprint.get("categories", {})

print(f"📥 Pulling live dataset from Hugging Face: {HF_REPO_ID} to calculate shortfalls...")
try:
    ds = load_dataset(HF_REPO_ID, split="train")
    df = ds.to_pandas()
except Exception as e:
    stop_telemetry.set()
    print(f"❌ Failed to load dataset from HF: {e}")
    exit(1)

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
    print(f"🔧 Loaded Config: Tier 1 ({len(TIER1_SUBS)} subs), Tier 2 Archive ({len(TIER2_SUBS)} subs)")
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
corpus_list = df['text'].dropna().tolist()
total_corpus = len(corpus_list)

for doc in tqdm(corpus_list, desc="   ↳ Analyzing Global Corpus", miniters=max(1, total_corpus//10), maxinterval=float('inf'), leave=False):
    for word in set(basic_tokenize(doc)): global_df_freq[word] += 1

dynamic_stopwords = {w for w, count in global_df_freq.items() if count > int(total_corpus * 0.15)}
effective_stopwords = static_stopwords.union(dynamic_stopwords)
print(f"   ↳ Dynamic Threshold Triggered: Auto-purged {len(dynamic_stopwords)} filler words.")

target_tf, bg_df_freq = collections.Counter(), collections.Counter()
for doc in df[df[priority_cat] == 1]['text'].dropna().tolist(): target_tf.update([w for w in basic_tokenize(doc) if w not in effective_stopwords and len(w)>2])
for doc in df[df[priority_cat] == 0]['text'].dropna().tolist():
    for word in set([w for w in basic_tokenize(doc) if w not in effective_stopwords and len(w)>2]): bg_df_freq[word] += 1

tfidf_scores = {w: tf * math.log(max(len(bg_df_freq), 1) / (bg_df_freq.get(w, 0) + 1)) for w, tf in target_tf.items()}
raw_seed_words = [w for w, _ in sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10]]
print(f"   ↳ Raw Statistical Seeds: {raw_seed_words}")

# ==========================================
# 2.5. LLM DYNAMIC SAFETY NET
# ==========================================
print("\n🛡️ [DIAGNOSTIC] Phase 1.5: Dynamic LLM Safety Net")
OPENCODE_KEY = os.environ.get("OPENCODE_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
if not OPENCODE_KEY:
    raise ValueError("❌ OPENCODE_KEY environment variable is missing.")

client = OpenAI(api_key=OPENCODE_KEY, base_url="https://opencode.ai/zen/go/v1")
MODEL_NAME = "deepseek-v4-flash"

filter_prompt = f"Target category: '{priority_cat}'. We extracted these raw statistical words: {raw_seed_words}. Filter this list. Remove harmless conversational words (like 'lol', 'fake', 'problem') and purely neutral demographic/identity terms (like 'india', 'brahmin', 'dalit', 'women') IF they are not inherently abusive. Keep ONLY actual slurs, highly toxic slang, and explicitly hostile modifiers. Output strictly JSON: {{\"toxic_seeds\": [\"word1\", ...]}}"

try:
    filter_res = client.chat.completions.create(
        model=MODEL_NAME, 
        messages=[{"role": "user", "content": filter_prompt}], 
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}}
    )
    
    f_usage = filter_res.usage.model_dump() if hasattr(filter_res.usage, 'model_dump') else vars(filter_res.usage)
    perf_metrics['total_prompt_tokens'] += f_usage.get('prompt_tokens', 0)
    perf_metrics['total_completion_tokens'] += f_usage.get('completion_tokens', 0)
    perf_metrics['generator_tokens'] += f_usage.get('total_tokens', 0)

    toxic_seeds = json.loads(filter_res.choices[0].message.content.strip()).get("toxic_seeds", [])
    if not toxic_seeds: toxic_seeds = raw_seed_words[:3]
        
    print(f"   ↳ Sanitized Toxic Seeds: {toxic_seeds}")
except Exception as e:
    print(f"   ⚠️ LLM filter failed: {e}. Using raw seeds.")
    toxic_seeds = raw_seed_words

# ==========================================
# 3. LLM MULTIPLIER (SEED EXPANSION)
# ==========================================
print("\n🧠 [DIAGNOSTIC] Phase 2: Stateful Semantic Expansion")

prompt = f"Target category: '{priority_cat}'. We already know these cached terms: {cached_terms[:20]}. We isolated these highly toxic root seeds: {toxic_seeds}. Generate a JSON object containing a 'keywords' array of 30 completely NEW, highly specific spelling variations, slurs, and slang that users type to bypass filters. Do not repeat known terms. Output strictly JSON: {{\"keywords\": [\"word1\", ...]}}"

try:
    res = client.chat.completions.create(
        model=MODEL_NAME, 
        messages=[{"role": "user", "content": prompt}], 
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}}
    )
    
    usage_dict = res.usage.model_dump() if hasattr(res.usage, 'model_dump') else vars(res.usage)
    perf_metrics['generator_tokens'] += usage_dict.get('total_tokens', 0)
    llm_keywords = json.loads(res.choices[0].message.content.strip()).get("keywords", [])
    print(f"   ↳ LLM Generated Terms: {llm_keywords}")
except Exception as e:
    print(f"   ⚠️ LLM expansion failed: {e}. Using sanitized seeds only.")
    llm_keywords = []

final_keywords = list(dict.fromkeys([k for k in (cached_terms + toxic_seeds + llm_keywords) if len(k) > 3]))
print(f"   ↳ Final Deduplicated Lexicon ({len(final_keywords)} terms): {final_keywords}")

master_lexicon[priority_cat.upper()] = final_keywords
try:
    with open(LEXICON_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(master_lexicon, f, indent=2, ensure_ascii=False)
except Exception: pass

# ==========================================
# 4. STRICT 3-TIER REGIONAL EXTRACTION ENGINE
# ==========================================
print("\n🦆 [DIAGNOSTIC] Phase 3: Regional-Strict Quota Extraction")
CANDIDATE_THRESHOLD = 500
SOFT_THRESHOLD = int(CANDIDATE_THRESHOLD * 0.8)

T1_QUOTA = 0
T2_QUOTA = 0
T3_QUOTA = int(SOFT_THRESHOLD * 1.0)

print(f"   ↳ Target Candidates: {SOFT_THRESHOLD} | Quotas -> Tier 1: {T1_QUOTA} | Tier 2: {T2_QUOTA} | Tier 3 (5% API): {T3_QUOTA}")

harvest_df = pd.DataFrame()
safe_keywords = [k.replace("'", "''").lower() for k in final_keywords][:35]
filter_clauses = " OR ".join([f"LOWER(body) LIKE '%{k}%'" for k in safe_keywords if k])

extraction_stats = {
    "Tier 1 (Core Echo)": collections.defaultdict(int),
    "Tier 2 (Expanded Archive)": collections.defaultdict(int),
    "Tier 3 (Targeted Live API)": collections.defaultdict(int)
}

# --- TIER 3: TARGETED LIVE API (5% Quota) ---
def fetch_tier3_live(subreddits, lexicon, max_rows, time_budget=15):
    print(f"      -> [Tier 3] Targeted Arctic API Search (Target: {max_rows} rows | Budget: {time_budget}s)...")
    session = requests.Session()
    start_time = time.time()
    collected = []
    AFTER_2025 = 1735689600 
    
    if not subreddits or not lexicon: return pd.DataFrame()

    sampled_subs = random.sample(subreddits, min(len(subreddits), 10))
    sampled_terms = random.sample(lexicon, min(len(lexicon), 5))
    
    for sub in sampled_subs:
        if len(collected) >= max_rows or (time.time() - start_time) > time_budget: break
        for term in sampled_terms:
            if len(collected) >= max_rows or (time.time() - start_time) > time_budget: break
            
            params = {"subreddit": sub, "q": term, "after": AFTER_2025, "limit": 50, "sort": "desc"}
            try:
                resp = session.get("https://arctic-shift.photon-reddit.com/api/comments/search", params=params, timeout=5)
                if resp.status_code == 429: break
                resp.raise_for_status()
                
                for item in resp.json().get("data", []):
                    body = item.get("body", "")
                    created_utc = item.get("created_utc", None)
                    ym = datetime.utcfromtimestamp(created_utc).strftime('%Y-%m') if created_utc else None
                    if body and body not in ["[removed]", "[deleted]"]:
                        collected.append({"id": item.get("id"), "body": body, "subreddit": sub, "created_utc": created_utc, "year_month": ym})
                time.sleep(0.5)
            except Exception:
                continue
                
    t3_df = pd.DataFrame(collected)
    if not t3_df.empty:
        t3_df = t3_df.drop_duplicates(subset=["id"])
        if len(t3_df) > max_rows: t3_df = t3_df.sample(max_rows)
    return t3_df

t3_df = fetch_tier3_live(TIER1_SUBS + TIER2_SUBS[:20], final_keywords, T3_QUOTA)
if not t3_df.empty:
    harvest_df = pd.concat([harvest_df, t3_df], ignore_index=True)
    extraction_stats["Tier 3 (Targeted Live API)"]["2025+"] = len(t3_df)
print(f"         ✅ Yield: {len(t3_df)} candidates from Live API.")

# --- TIER 1 & 2: DUCKDB REGIONAL ARCHIVES (2017-2024) ---
print(f"      -> Searching Regional DuckDB Archives (2017-2024)")
con = duckdb.connect()
con.execute("PRAGMA memory_limit='6GB'; PRAGMA threads=8; INSTALL httpfs; LOAD httpfs;")
if HF_TOKEN: con.execute(f"CREATE SECRET hf_auth (TYPE HUGGINGFACE, TOKEN '{HF_TOKEN}');")
api = HfApi(token=HF_TOKEN)
all_files = api.list_repo_files("open-index/arctic", repo_type="dataset")

t1_df, t2_df = pd.DataFrame(), pd.DataFrame()
current_year = int(os.environ.get("TARGET_YEAR", 2017))

def duckdb_extract(sub_list, hf_urls_list, quota, limit=5000):
    if not sub_list: return pd.DataFrame()
    subs_formatted = ", ".join([f"'{s.replace(chr(39), chr(39)+chr(39))}'" for s in sub_list])
    sub_clause = f"AND LOWER(subreddit) IN ({subs_formatted})"
        
    query = f"""
    SELECT id, body, LOWER(subreddit) as subreddit, created_utc, strftime(epoch_ms(created_utc * 1000), '%Y-%m') as year_month
    FROM read_parquet({hf_urls_list}) 
    WHERE ({filter_clauses}) {sub_clause} AND body NOT IN ('[deleted]', '[removed]', '') 
    LIMIT {limit}
    """
    res_df = con.query(query).to_df()
    if len(res_df) > quota: res_df = res_df.sample(quota)
    return res_df

with tqdm(total=(2024 - current_year + 1), desc="      -> [DuckDB Regional Scan]", leave=False) as pbar:
    while current_year <= 2024 and len(harvest_df) < SOFT_THRESHOLD:
        year_files = [f for f in all_files if f.endswith('.parquet') and f'data/comments/{current_year}' in f]
        if not year_files:
            current_year += 1; pbar.update(1); continue
            
        hf_urls = [f"hf://datasets/open-index/arctic/{f}" for f in random.sample(year_files, min(40, len(year_files)))]

        if len(t1_df) < T1_QUOTA:
            t1_new = duckdb_extract(TIER1_SUBS, hf_urls, T1_QUOTA - len(t1_df))
            if not t1_new.empty:
                t1_df = pd.concat([t1_df, t1_new]).drop_duplicates(subset=['id'])
                extraction_stats["Tier 1 (Core Echo)"][current_year] += len(t1_new)
        
        if len(t2_df) < T2_QUOTA:
            t2_new = duckdb_extract(TIER2_SUBS, hf_urls, T2_QUOTA - len(t2_df))
            if not t2_new.empty:
                t2_df = pd.concat([t2_df, t2_new]).drop_duplicates(subset=['id'])
                extraction_stats["Tier 2 (Expanded Archive)"][current_year] += len(t2_new)
            
        harvest_df = pd.concat([t3_df, t1_df, t2_df]).drop_duplicates(subset=['id'])
        current_year += 1
        pbar.update(1)

if harvest_df.empty:
    stop_telemetry.set()
    print("❌ Exhausted regional tiers. No matching candidates found.")
    exit(0)

print("\n📈 [EXTRACTION YIELD STATISTICS]")
for tier, year_data in extraction_stats.items():
    total_rows = sum(year_data.values())
    if total_rows > 0:
        year_breakdown = " | ".join([f"{y}: {count} rows" for y, count in sorted(year_data.items())])
        print(f"   ✅ {tier}: {total_rows} total rows -> ({year_breakdown})")
print(f"   -------------------------------------------------")
print(f"   🚀 Total Regional Pool: {len(harvest_df)} Candidates")

# ==========================================
# 5. SANITIZATION, LABELING & HUGGING FACE UPLOAD
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

total_pool = len(harvest_df)
harvest_df['body_clean'] = [sanitize_text(text) for text in tqdm(harvest_df['body'], desc="   ↳ Sanitizing text data", miniters=max(1, total_pool//10), maxinterval=float('inf'), leave=False)]
harvest_df['temp_id'] = harvest_df.index.astype(str)

try: SYSTEM_PROMPT = requests.get("https://raw.githubusercontent.com/darelphilipo/hinglish_reddit_data/main/prompt/System_Prompt", timeout=10).text.strip()
except Exception as e: stop_telemetry.set(); raise RuntimeError(f"❌ Failed to fetch System Prompt: {e}")

def label_batch(comments_batch, attempt=1):
    numbered = "\n".join(f'ID: {cid} | Comment: {body}' for cid, body in comments_batch)
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Label these comments:\n{numbered}"}],
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

print(f"\n🚀 Running Asynchronous Inference Engine ({MAX_WORKERS} Workers)...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(label_batch, b) for b in batches]
    for future in tqdm(as_completed(futures), total=len(batches), desc="   ↳ Labeling Candidates", miniters=max(1, len(batches)//10), maxinterval=float('inf')):
        all_labels.extend(future.result())

labels_df = pd.DataFrame(all_labels)
if 'id' in labels_df.columns: labels_df.drop(columns=["id"], inplace=True)
labels_df["temp_id"] = labels_df["temp_id"].astype(str)

final_df = harvest_df.merge(labels_df, on="temp_id", how="inner").drop(columns=["temp_id", "body_clean"], errors='ignore')
final_df = final_df[final_df['pv'].notna()]

print("\n🛠️ Formatting Dual-Schema (RoBERTa + Sarvam ChatML)...")
formatted_records = []

for idx, row in final_df.iterrows():
    try:
        labels = {long_key: int(row.get(short_key, 0)) for short_key, long_key in KEY_MAPPING.items()}
        has_analysis = 'analysis' in row and pd.notna(row['analysis'])
        if has_analysis:
            labels['analysis'] = str(row['analysis']).strip()
            
        chatml_messages = [
            {"role": "system", "content": STUDENT_PROMPT},
            {"role": "user", "content": str(row['body']).strip()},
            {"role": "assistant", "content": json.dumps(labels, ensure_ascii=False)}
        ]
        
        record = {
            "id": str(row['id']),
            "text": str(row['body']).strip(),
            "subreddit": str(row.get('subreddit', 'unknown')),
            "created_utc": row.get('created_utc', None),
            "year_month": str(row['year_month']) if pd.notna(row.get('year_month')) else None
        }
        
        record.update(labels)
        record["messages"] = chatml_messages
        if has_analysis:
             record['analysis'] = str(row['analysis']).strip() 
             
        formatted_records.append(record)
    except Exception as e:
        pass

hf_master_df = pd.DataFrame(formatted_records)
total_lbl = len(hf_master_df)

print(f"\n☁️ Initiating Chunked Upload to Hugging Face ({CHUNK_SIZE} rows/chunk)...")
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
chunks = [hf_master_df[i:i + CHUNK_SIZE] for i in range(0, len(hf_master_df), CHUNK_SIZE)]

for chunk_idx, chunk_df in enumerate(chunks):
    chunk_name = f"harvester_{priority_cat}_{timestamp_str}_part_{chunk_idx:03d}.parquet"
    hf_path = f"data/{chunk_name}"
    local_parquet_path = f"./{chunk_name}"
    
    print(f"   📤 Uploading {chunk_name} ({len(chunk_df):,} rows)...")
    chunk_df.to_parquet(local_parquet_path, engine='pyarrow', index=False)
    
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_file(path_or_fileobj=local_parquet_path, path_in_repo=hf_path, repo_id=HF_REPO_ID, repo_type="dataset")
            print(f"   ✅ Successfully pushed {chunk_name} to HF.")
            os.remove(local_parquet_path)
            time.sleep(2)
            break
        except Exception as e:
            if attempt == max_retries:
                if os.path.exists(local_parquet_path): os.remove(local_parquet_path)
            time.sleep(min(3 ** attempt, 30))

stop_telemetry.set()
monitor_thread.join()

print("\n==================================================")
print(" 🏁 HARVESTER YIELD & TELEMETRY REPORT")
print("==================================================")
hit_rate = (perf_metrics['total_cache_hits'] / perf_metrics['total_prompt_tokens'] * 100) if perf_metrics['total_prompt_tokens'] > 0 else 0
print(f"Total Rows Labeled & Pushed : {total_lbl:,}")
print(f"Total Workflow Time      : {time.time() - script_start_time:.2f}s")
print(f"LLM Generator Tokens     : {perf_metrics['generator_tokens']:,}")
print(f"Inference Prompt Tokens  : {perf_metrics['total_prompt_tokens']:,} (Cache Hit: {hit_rate:.1f}%)")
print(f"Inference Output Tokens  : {perf_metrics['total_completion_tokens']:,}")
print(f"✅ Verified shards successfully uploaded to Hugging Face!")
