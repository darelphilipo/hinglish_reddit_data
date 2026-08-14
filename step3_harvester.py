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
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
from huggingface_hub import HfApi
from cleantext import clean

print("🌾 Initializing Autonomous Harvester: Hybrid NLP & Validation Engine...")

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
# 1. LOAD MASTER DATASET & CALCULATE GOAL
# ==========================================
BASE_OUTPUT_DIR = './labelled_output/'
CHUNKS_DIR = os.path.join(BASE_OUTPUT_DIR, 'chunks/')
os.makedirs(CHUNKS_DIR, exist_ok=True)
MASTER_PATH = os.path.join(BASE_OUTPUT_DIR, 'master_baseline_tier1.csv')

if not os.path.exists(MASTER_PATH):
    print("❌ Master dataset not found. Run Step 1 & 2 first.")
    exit(1)

df = pd.read_csv(MASTER_PATH)
TARGETS = {'caste': 1000, 'communal_religious': 1200, 'regional_xenophobic': 1000, 'misogyny_gender': 1000}
shortfalls = {cat: max(0, TARGETS[cat] - df.get(cat, pd.Series([0])).sum()) for cat in TARGETS}

if all(s <= 0 for s in shortfalls.values()):
    stop_telemetry.set()
    print("✅ All target categories met. No harvesting needed.")
    exit(0)

priority_cat = max(shortfalls, key=shortfalls.get)
current_shortfall = shortfalls[priority_cat]
print(f"🎯 Harvester Target Identified: {priority_cat.upper()} | Shortfall: {current_shortfall} rows")

# ==========================================
# 2. DYNAMIC THRESHOLDING & TF-IDF SEED EXTRACTION
# ==========================================
print("\n📊 [DIAGNOSTIC] Phase 1: Dynamic Thresholding & Statistical Seed Generation")
STOPWORDS_URL = "https://raw.githubusercontent.com/darelphilipo/hinglish_reddit_data/main/prompt/hinglish_stopwords.txt"
try:
    resp = requests.get(STOPWORDS_URL, timeout=10)
    resp.raise_for_status()
    static_stopwords = set(line.strip().lower() for line in resp.text.splitlines() if line.strip())
except Exception:
    static_stopwords = {'hai', 'ki', 'aur', 'mein', 'se', 'ko', 'ka', 'ke', 'ye', 'wo', 'the', 'is', 'a', 'to'}

def basic_tokenize(text):
    return re.findall(r'\b[a-z0-9]+\b', str(text).lower())

total_docs = len(df)
max_doc_threshold = int(total_docs * 0.15) # 15% Ceiling

print(f"   ↳ Analyzing global corpus ({total_docs} rows) to establish dynamic baselines...")
global_df_freq = collections.Counter()
for doc in df['body'].tolist():
    for word in set(basic_tokenize(doc)):
        global_df_freq[word] += 1

dynamic_stopwords = {w for w, count in global_df_freq.items() if count > max_doc_threshold}
effective_stopwords = static_stopwords.union(dynamic_stopwords)

print(f"   ↳ Dynamic Threshold Triggered: Auto-purged {len(dynamic_stopwords)} filler words appearing in >15% of comments.")

target_df = df[df[priority_cat] == 1]
bg_df = df[df[priority_cat] == 0]

def get_tokens(text):
    words = basic_tokenize(text)
    return [w for w in words if w not in effective_stopwords and len(w) > 2]

target_tf, bg_df_freq = collections.Counter(), collections.Counter()
for doc in target_df['body'].tolist(): target_tf.update(get_tokens(doc))
for doc in bg_df['body'].tolist():
    for word in set(get_tokens(doc)): bg_df_freq[word] += 1

total_bg_docs = max(len(bg_df), 1)
tfidf_scores = {w: tf * math.log(total_bg_docs / (bg_df_freq.get(w, 0) + 1)) for w, tf in target_tf.items()}
seed_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10]

seed_words_only = [w for w, _ in seed_keywords]

# ==========================================
# 3. LLM MULTIPLIER (SEED EXPANSION)
# ==========================================
print("\n🧠 [DIAGNOSTIC] Phase 2: LLM Semantic Expansion")
OPENCODE_KEY = os.environ.get("OPENCODE_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
client = OpenAI(api_key=OPENCODE_KEY, base_url="https://opencode.ai/zen/go/v1")
MODEL_NAME = "deepseek-v4-flash"

prompt = f"These terms are statistically dominant in Hinglish Reddit comments flagged for '{priority_cat}': {seed_words_only}. Generate a JSON object containing a 'keywords' array of 40 highly specific spelling variations, slurs, and community slang that users type to bypass filters. Output strictly JSON: {{\"keywords\": [\"word1\", ...]}}"

try:
    res = client.chat.completions.create(
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}
    )
    llm_usage = res.usage.model_dump() if hasattr(res.usage, 'model_dump') else vars(res.usage)
    perf_metrics['generator_tokens'] = llm_usage.get('total_tokens', 0)
    llm_keywords = json.loads(res.choices[0].message.content.strip()).get("keywords", [])
except Exception as e:
    print(f"   ⚠️ LLM failed: {e}. Using statistical seeds only.")
    llm_keywords = []

final_keywords = list(dict.fromkeys(seed_words_only + llm_keywords))
print(f"   ↳ Final Deduplicated Lexicon ({len(final_keywords)} terms): {final_keywords}")

# ==========================================
# 4. SURGICAL DUCKDB EXTRACTION (TIME TRAVEL & TIERS)
# ==========================================
print("\n🦆 [DIAGNOSTIC] Phase 3: Targeted Hugging Face Extraction (Multi-Tier)")
con = duckdb.connect()
con.execute("PRAGMA memory_limit='6GB'; PRAGMA threads=8; INSTALL httpfs; LOAD httpfs;")
if HF_TOKEN: con.execute(f"CREATE SECRET hf_auth (TYPE HUGGINGFACE, TOKEN '{HF_TOKEN}');")

api = HfApi(token=HF_TOKEN)
all_files = api.list_repo_files("open-index/arctic", repo_type="dataset")

# --- SUBREDDIT TIER CONFIGURATION ---
TIER_1_SUBS = [
    'chodi', 'bakchodi', 'sham_sharma_show', 'desimeta', 'indiandankmemes', 'dankinindia', 
    'saimansays', 'librandu', 'unitedstatesofindia', 'indiadiscussion', 'canconfirmiamindian', 
    'arrangedmarriage', 'bollyblindsngossip'
]

EXPANSION_DICT = {
    "heavy_1": ["IndiaSpeaks", "india", "indiameme", "funnyIndia", "IndianDankMemes", "CarryMinati", "ipl", "IndianGaming"],
    "heavy_2": ["bollywood", "developersIndia", "UPSC", "IndianStockMarket", "JEENEETards", "Btechtards", "StartUpIndia", "AskIndia"],
    "medium_1": ["indianews", "indiadiscussion", "CriticalThinkingIndia", "unitedstatesofindia", "bihar", "uttarpradesh", "delhi", "karnataka", "TamilNadu", "Maharashtra", "gujarat", "Rajasthan", "bangalore", "mumbai", "chennai", "hyderabad", "kolkata"],
    "medium_2": ["pune", "ahmedabad", "lucknow", "Arrangedmarriage", "RelationshipIndia", "TwoXIndia", "AskIndianWomen", "AskIndianMen", "OffMyChestIndia", "TeenIndia", "IndianTeenagers", "Indiangirlsontinder", "DesiWeddings", "TwentiesIndia", "CricketShitpost", "IndiaCricket", "IndianFootball"],
    "medium_3": ["indiansports", "RCB", "csk", "chessindia", "SaimanSays", "ShahRukhKhan", "SamayRaina", "thugeshh", "beastboyshub", "sunraybee", "FingMemes", "dankrishu", "ViratKohli", "BollyBlindsNGossip", "InstaCelebsGossip", "bollywoodmemes", "BollywoodFashion"],
    "medium_4": ["sharktankindia", "biggboss", "IndianTellyTalk", "DHHMemes", "punjabimusic", "kollywood", "tollywood", "IndianCinema", "BollywoodRealism", "IndianOTTbestof", "AnimeMirchi", "animeindian", "BollywoodMusic", "MalayalamMovies", "IndianHipHopHeads", "IndianStreetBets", "IndiaInvestments"],
    "medium_5": ["personalfinanceindia", "CreditCardsIndia", "CryptoIndia", "mutualfunds", "IndiaTax", "BitcoinIndia", "StockMarketIndia", "FIREIndia", "FatFIREIndia", "IndianStocks", "beermoneyindia", "Frugal_Ind", "CATpreparation", "Indian_Academia", "JEE", "IndiaCareers", "BITSPilani"],
    "medium_6": ["Indians_StudyAbroad", "IndianWorkplace", "ICSE", "CharteredAccountants", "IndiaBusiness", "smallbusinessindia", "CBSE", "indianmedschool", "CarsIndia", "indianrailways", "indianbikes", "AirTravelIndia", "Indianbooks", "IndianArtAndThinking", "indiafood", "IndianArtAI", "hindi"],
    "medium_7": ["IndianFoodPhotos", "IndiaCoffee", "PhotographyIndia", "IndiansRead", "IndiaTech", "GadgetsIndia", "Indiangamers", "XboxIndia", "IndiaPS5", "DesiVideoMemes", "indianmemer", "IndianMeyMeys", "IndianMemeTemplates", "desimemes"],
    "tiny_1": ["AajMaineJana", "Chandigarh", "DesiFragranceAddicts", "DesiKeto", "Fitness_India", "Goa", "HimachalPradesh", "IncredibleIndia", "IndiaNostalgia", "IndiaTrending", "IndianFashionAddicts", "IndianHistory", "IndianMakeupAddicts", "IndianSkincareAddicts", "Indian_flex", "Kerala", "Northeastindia", "Odisha", "SneakersIndia", "SoloTravel_India", "Uttarakhand", "ZyadaKuchNai", "assam", "desitravellers", "gurgaon", "india_tourism", "indiafitcheck", "indianbeautyhauls", "indianfitness", "indiasocial", "watchesindia"]
}

TIER_2_SUBS = list(TIER_1_SUBS)
for cat, subs in EXPANSION_DICT.items(): TIER_2_SUBS.extend(subs)
TIER_2_SUBS = list(set([s.lower() for s in TIER_2_SUBS])) # Deduplicate and lowercase

safe_keywords = [k.replace("'", "''").lower() for k in final_keywords if len(k) > 3][:35]
filter_clauses = " OR ".join([f"LOWER(body) LIKE '%{k}%'" for k in safe_keywords if k])
limit_rows = min(5000, current_shortfall * 5)
CANDIDATE_THRESHOLD = 500

# --- TIME TRAVEL & TIER LOOP ---
start_year = int(os.environ.get("TARGET_YEAR", 2017))
current_year = start_year
harvest_df = pd.DataFrame()

while current_year <= 2024 and len(harvest_df) < CANDIDATE_THRESHOLD:
    print(f"\n   ⏳ Scanning Year: {current_year}")
    year_files = [f for f in all_files if f.endswith('.parquet') and f'data/comments/{current_year}' in f]
    
    if not year_files:
        print(f"      ⚠️ No data found for {current_year}. Skipping...")
        current_year += 1
        continue
        
    selected_shards = random.sample(year_files, min(40, len(year_files)))
    hf_urls = [f"hf://datasets/open-index/arctic/{f}" for f in selected_shards]

    tiers_to_try = [
        (1, "Core Targeting", TIER_1_SUBS),
        (2, "Expanded Targeting", TIER_2_SUBS),
        (3, "Global Wildcard", None)
    ]

    for tier_num, tier_name, sub_list in tiers_to_try:
        print(f"      -> Attempting Tier {tier_num} ({tier_name})...")
        
        sub_clause = ""
        if sub_list:
            subs_formatted = ", ".join([f"'{s}'" for s in sub_list])
            sub_clause = f"AND LOWER(subreddit) IN ({subs_formatted})"
            
        query = f"""
        SELECT id, body, LOWER(subreddit) as subreddit, created_utc, strftime(epoch_ms(created_utc * 1000), '%Y-%m') as year_month
        FROM read_parquet({hf_urls})
        WHERE ({filter_clauses}) 
          {sub_clause}
          AND body NOT IN ('[deleted]', '[removed]', '') 
          AND length(body) BETWEEN 10 AND 1000
        LIMIT {limit_rows}
        """
        
        temp_df = con.query(query).to_df()
        
        if len(temp_df) > len(harvest_df):
            harvest_df = temp_df
            
        print(f"         Yield: {len(temp_df)} candidates.")
        
        if len(harvest_df) >= CANDIDATE_THRESHOLD:
            print("         ✅ Threshold met. Proceeding to verification.")
            break
            
    if len(harvest_df) < CANDIDATE_THRESHOLD:
        current_year += 1

if harvest_df.empty:
    stop_telemetry.set()
    print("❌ Exhausted all Tiers and Years. No matching candidates found.")
    exit(0)

# ==========================================
# 5. SANITIZATION & AUTONOMOUS LABELING
# ==========================================
print("\n🛡️ [DIAGNOSTIC] Phase 4: Verification & Autonomous Labeling")

def sanitize_text(text):
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'/?u/[A-Za-z0-9_-]+', '', text) 
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    
    try:
        text = clean(text, fix_unicode=True, to_ascii=False, lower=False, no_line_breaks=True, 
                     no_urls=True, replace_with_url="", no_emails=True, replace_with_email="", 
                     no_phone_numbers=True, replace_with_phone_number="")
    except Exception:
        pass
    return re.sub(r'\s{2,}', ' ', text).strip()

harvest_df['body_clean'] = harvest_df['body'].apply(sanitize_text)
harvest_df['temp_id'] = harvest_df.index.astype(str)

PROMPT_URL = "https://raw.githubusercontent.com/darelphilipo/hinglish_reddit_data/main/prompt/System_Prompt"
try:
    SYSTEM_PROMPT = requests.get(PROMPT_URL, timeout=10).text.strip()
except Exception as e:
    stop_telemetry.set()
    raise RuntimeError(f"❌ Failed to fetch System Prompt: {e}")

def label_batch(comments_batch, attempt=1):
    numbered = "\n".join(f'ID: {cid} | Comment: {body}' for cid, body in comments_batch)
    user_prompt = f"Label these comments:\n{numbered}"
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            temperature=0.1, response_format={"type": "json_object"}, extra_body={"thinking": {"type": "disabled"}}
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

if 'id' in labels_df.columns: 
    labels_df.drop(columns=["id"], inplace=True)
labels_df["temp_id"] = labels_df["temp_id"].astype(str)

rename_mapping = {"pv": "profanity_vulgarity", "tah": "targeted_abuse_harassment", "dhs": "discriminatory_hate_speech", "cst": "caste", "cr": "communal_religious", "rx": "regional_xenophobic", "mg": "misogyny_gender"}
labels_df.rename(columns=rename_mapping, inplace=True)

final_df = harvest_df.merge(labels_df, on="temp_id", how="inner")
final_df.drop(columns=["temp_id", "body_clean"], errors='ignore', inplace=True)

final_df = final_df[final_df[priority_cat].notna()] 
verified_yield = final_df[priority_cat].sum()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
FINAL_PATH = os.path.join(CHUNKS_DIR, f'harvested_tier1_{priority_cat}_{timestamp}.csv')
final_df.to_csv(FINAL_PATH, index=False)

stop_telemetry.set()
monitor_thread.join()

# ==========================================
# 6. FINAL YIELD REPORT & FUNNEL METRICS
# ==========================================
total_time = time.time() - script_start_time
hit_rate = (perf_metrics['total_cache_hits'] / perf_metrics['total_prompt_tokens'] * 100) if perf_metrics['total_prompt_tokens'] > 0 else 0
success_pct = (verified_yield / len(final_df) * 100) if len(final_df) > 0 else 0
goal_pct = (verified_yield / current_shortfall * 100) if current_shortfall > 0 else 100

print("\n==================================================")
print(" 🏁 HARVESTER YIELD & TELEMETRY REPORT")
print("==================================================")
print(f"Target Category          : {priority_cat.upper()}")
print(f"Starting Shortfall       : {current_shortfall} rows")
print(f"Candidates Extracted     : {len(harvest_df)} rows")
print(f"Verified Positive Yield  : {int(verified_yield)} rows ({success_pct:.1f}% hit rate)")
print(f"Goal Met?                : {goal_pct:.1f}% of shortfall recovered")
print("--------------------------------------------------")
print(f"Total Workflow Time      : {total_time:.2f}s")
print(f"LLM Generator Tokens     : {perf_metrics['generator_tokens']:,}")
print(f"Inference Prompt Tokens  : {perf_metrics['total_prompt_tokens']:,} (Cache Hit: {hit_rate:.1f}%)")
print(f"Inference Output Tokens  : {perf_metrics['total_completion_tokens']:,}")
print("==================================================")
print(f"✅ Verified data committed to: {FINAL_PATH}")
