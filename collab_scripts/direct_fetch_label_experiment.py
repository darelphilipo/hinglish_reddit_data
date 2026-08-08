# @title 🧪 Multi-Subreddit MLOps Experiment (2017 - June 2020)
!pip install -q pandas tqdm openai duckdb
!pip install -q pandas tqdm openai duckdb huggingface_hub

import pandas as pd
import duckdb
import os
import time
import json
import threading
import html
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm
from openai import OpenAI

try:
    from google.colab import userdata, drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
if IN_COLAB:
    print("🔗 Mounting Google Drive...")
    drive.mount('/content/drive', force_remount=False)

OUTPUT_DIR = '/content/drive/MyDrive/Hinglish_Classifier_Project/experiments/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_CSV_PATH = os.path.join(OUTPUT_DIR, 'experiment_5k_2017_2020.csv')
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'experiment_checkpoint.csv')

TARGET_SUBREDDITS = [
    'chodi', 'bakchodi', 'sham_sharma_show', 'desimeta',
    'indiandankmemes', 'dankinindia', 'saimansays', 'librandu',
    'unitedstatesofindia', 'indiadiscussion', 'canconfirmiamindian',
    'arrangedmarriage', 'bollyblindsngossip'
]

# Epoch timestamps for 2017-01-01 00:00:00 UTC to 2020-06-30 23:59:59 UTC
START_EPOCH = 1483228800
END_EPOCH = 1593561599
TOTAL_RECORD_LIMIT = 5000

MAX_WORKERS = 10 

print("🔌 Connecting to OpenCode Go API...")
if IN_COLAB:
    try: OPENCODE_KEY = userdata.get("opencode")
    except: OPENCODE_KEY = userdata.get("OPENCODE_KEY")
else:
    OPENCODE_KEY = os.getenv("OPENCODE_KEY")

client = OpenAI(api_key=OPENCODE_KEY, base_url="https://opencode.ai/zen/go/v1")
MODEL_NAME = "deepseek-v4-flash"

# ==========================================
# 2. DUCKDB LIGHTNING EXTRACTION (Anti-Hang Fix)
# ==========================================
from huggingface_hub import HfApi
import random

print("\n🦆 Initializing DuckDB and Fetching Exact Parquet Paths...")
con = duckdb.connect()
con.execute("PRAGMA memory_limit='5GB';")
con.execute("INSTALL httpfs; LOAD httpfs;")

# 🔑 Authenticate DuckDB using your Colab Secret
if IN_COLAB:
    try:
        print("🔐 Retrieving HF_TOKEN from Colab Secrets...")
        HF_TOKEN = userdata.get('HF_TOKEN')
        con.execute(f"CREATE SECRET hf_auth (TYPE HUGGINGFACE, TOKEN '{HF_TOKEN}');")
        print("✅ Hugging Face authentication successful!")
    except Exception as e:
        print(f"⚠️ Could not load HF_TOKEN: {e}")

con.execute("PRAGMA enable_progress_bar;")
con.execute("PRAGMA enable_print_progress_bar;")

print(f"🕵️ Fetching exact file paths via HF API to prevent metadata hang...")
api = HfApi()

try:
    all_files = api.list_repo_files("open-index/arctic", repo_type="dataset")
    # Filter only for the 2017-2020 comment folders
    target_era_files = [
        f for f in all_files 
        if f.endswith('.parquet') and (
            'data/comments/2017' in f or 
            'data/comments/2018' in f or 
            'data/comments/2019' in f or 
            'data/comments/2020' in f
        )
    ]
    
    # Sample shards to keep extraction lightning-fast (150 shards is plenty to find 5k rows)
    random.seed(42)
    selected_shards = random.sample(target_era_files, min(150, len(target_era_files)))
    hf_urls = [f"hf://datasets/open-index/arctic/{f}" for f in selected_shards]
    print(f"✅ Locked onto {len(hf_urls)} explicit data shards. Booting DuckDB...")
except Exception as e:
    raise RuntimeError(f"❌ Failed to communicate with Hugging Face: {e}")

subs_formatted = ", ".join([f"'{s.lower()}'" for s in TARGET_SUBREDDITS])

fetch_start = time.time()
query = f"""
SELECT 
    id,
    body,
    LOWER(subreddit) as subreddit,
    created_utc,
    strftime(epoch_ms(created_utc * 1000), '%Y-%m') as year_month
FROM read_parquet({hf_urls})
WHERE LOWER(subreddit) IN ({subs_formatted})
  AND created_utc >= {START_EPOCH}
  AND created_utc <= {END_EPOCH}
  AND body NOT IN ('[deleted]', '[removed]', '')
  AND length(body) > 5
"""

try:
    print(f"⏳ Running DuckDB Query across 2017–2020 Parquets...")
    raw_df = con.query(query).to_df()
    print(f"\n📦 Fast-fetch complete! Pulled {len(raw_df)} candidate comments in {time.time() - fetch_start:.2f}s.")
except Exception as e:
    print(f"\n⚠️ DuckDB Fetch Notice: {e}")
    raw_df = pd.DataFrame(columns=['id', 'body', 'subreddit', 'created_utc', 'year_month'])

# Monthly Stratified Sampling
if not raw_df.empty:
    print("⚖️ Balancing dataset across Subreddits and Months...")
    sampled_df = (
        raw_df.groupby(['subreddit', 'year_month'], group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), 100), random_state=42))
        .reset_index(drop=True)
    )
    
    # Cap total dataset at TOTAL_RECORD_LIMIT (5,000)
    if len(sampled_df) > TOTAL_RECORD_LIMIT:
        sampled_df = sampled_df.sample(n=TOTAL_RECORD_LIMIT, random_state=42).reset_index(drop=True)
    
    df = sampled_df
    import gc
    del raw_df
    gc.collect() # Forces Python to instantly free up the RAM
else:
    df = raw_df

print(f"🎯 Final Sample Pool: {len(df)} rows across {df['subreddit'].nunique()} subreddits.")

# ==========================================
# 3. TEXT SANITIZATION (WITH PROGRESS BAR)
# ==========================================
def sanitize_text(text):
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u200b', '')
    return text.strip()

if not df.empty:
    print("\n🧹 Sanitizing text to optimize tokens...")
    # Initialize tqdm for pandas apply to show visual progress bar
    tqdm.pandas(desc="Text Sanitization")
    df['body_clean'] = df['body'].progress_apply(sanitize_text)
    
    df = df[df['body_clean'].str.len() > 0].reset_index(drop=True)
    df['temp_id'] = df.index.astype(str)

# ==========================================
# 4. TEACHER MODEL PROMPT (v6.5) & ENGINE
# ==========================================
SYSTEM_PROMPT = """# Hinglish & English Content Moderation — Teacher Model Labeling Prompt
Version: 6.5 (Batch Array + Full v6.2 English Coverage Restored) | Hindi-English code-mixed and English, Roman script 

Philosophy: When in doubt on harassment/hate speech, output 0. Silencing legitimate speech is worse than missing an edge case. Never flag opinions, criticism, or political disagreement as harassment or hate speech.

Design note on pv (profanity_vulgarity): this flag is a detection signal, not a moderation verdict. It fires whenever explicit profanity is present, regardless of target.

--- 

## OUTPUT FORMAT 
Return ONLY a valid JSON object containing an array under the key "results", no preamble.
CRITICAL: You MUST put 'analysis' first to force logical evaluation before scoring.
Use exactly these short keys for every object to save tokens:

{
  "results": [
    {"id": "<id>", "analysis": "<1 brief sentence analyzing target and intent>", "pv": 0, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0}
  ]
}

Key Legend:
pv = profanity_vulgarity | tah = targeted_abuse_harassment | dhs = discriminatory_hate_speech
cst = caste | cr = communal_religious | rx = regional_xenophobic | mg = misogyny_gender
`dhs` = 1 iff any dimension (cst, cr, rx, mg) = 1. If 0, all dimensions = 0.
The three top-level flags are independent — any combination is valid, including all three at once.

--- 

## THE DECISION PROCEDURE 

Step 1 — Is explicit profanity/vulgarity present anywhere in the text? 
Check for curse words, obscene terms, sexual slurs, or vulgar swearing — Hindi/Hinglish (chutiya, bsdk, mc, bc, gaand, lund, randi, madarchod, behenchod, bhosadike, gandu, harami, kamine) and English (fuck, shit, bitch, cunt, asshole, bastard, motherfucker, whore, slut, dick, prick, retard/retarded) — including obfuscated variants. 
- Present anywhere — even filler, self-directed, or aimed at an abstract target → pv: 1. Otherwise 0.

Step 2 — Is there targeted personal abuse/harassment OR an identity-based stereotype/objectification/hierarchy statement present? 
- Aimed at an abstract entity (government, system, company, policy) → tah: 0 
- Self-directed or reported speech → 0 
- Public figure reverse test: Personal (appearance/character) → 1. Professional (policy) → 0. 
- A specific person insulted, threatened, or personally degraded → tah: 1 

Step 3 — Is an identity dimension (caste/religion/region/gender) load-bearing? 
- Yes → dhs: 1 + correct dimension(s). Applies even with NO profanity present — covert othering, rhetorical questions, mocking religious practices, and conspiracy framing in Hindi or English (e.g. "go back to Pakistan").
- No, generic insult → identity dimensions stay 0. 

Step 4 — Political terms (sanghi/bhakt/libtard) are NOT an identity dimension on their own.

--- 

## CATEGORY DEFINITIONS 

- Caste (cst): chamar, bhangi, neech jaati, dedh, chura, pallan, kanjar, dhobi, mehtar, "quota-wallah", "lower caste" (as insult).
- Communal (cr): katwa, mulla/mulle, kafir, malaun, pajeet, "cow piss drinker", "dung worshipper", "sanghi/bhakt" (only when dehumanizing), "terrorist"/"jihadi" (blanket-smear).
- Regional (rx): bihari/madrasi (pejorative), bimaru, chinki, ghati, bhaiyya, "Porki", "Paki", "chink", "bloody Bihari/Madrasi", "these North/South Indians" (dismissively).
- Gendered (mg): chhakka/hijra, "maal"/"item", "randi khana", "feminazi", "slut", "whore", "asking for it", "women should stay home".

--- 

## KNOWN TRAPS

1. Homonyms: chakka = jackfruit vs slur. BC = date vs profanity.
2. English Homonyms: "dick" or "cock" only trigger pv:1 when used as anatomical vulgarity/insults. Proper names (Dick) get pv:0.
3. India-Pakistan: criticizing Pakistani govt = 0. Blanket-dehumanizing Pakistanis or Indian Muslims = 1 on dhs.
4. Code-switching: Evaluate multi-lingual comments as a single unit.

## ANCHOR EXAMPLES 
[
  {"id": "ex1", "analysis": "English profanity directed at institutional target.", "pv": 1, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex2", "analysis": "Hinglish hate speech targeting religion.", "pv": 1, "tah": 1, "dhs": 1, "cst": 0, "cr": 1, "rx": 0, "mg": 0},
  {"id": "ex3", "analysis": "English xenophobic regional trope.", "pv": 0, "tah": 0, "dhs": 1, "cst": 0, "cr": 0, "rx": 1, "mg": 0},
  {"id": "ex4", "analysis": "English gender hierarchy statement.", "pv": 0, "tah": 0, "dhs": 1, "cst": 0, "cr": 0, "rx": 0, "mg": 1}
]
"""

def label_batch(comments_batch, attempt=1):
    numbered = "\n".join(f'ID: {cid} | Comment: {body}' for cid, body in comments_batch)
    user_prompt = f"Label these {len(comments_batch)} comments strictly following the JSON schema:\n{numbered}"

    try:
        start = time.time()
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            temperature=0.1,
            max_tokens=2500,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}}
        )
        duration = time.time() - start
        
        # Extract API usage stats safely
        usage_dict = res.usage.model_dump() if hasattr(res.usage, 'model_dump') else vars(res.usage)
        token_details = usage_dict.get('prompt_tokens_details', {}) or {}
        cache_hits = usage_dict.get('prompt_cache_hit_tokens', token_details.get('cached_tokens', 0))
        cache_misses = usage_dict.get('prompt_cache_miss_tokens', usage_dict.get('prompt_tokens', 0) - cache_hits)

        # Bulletproof JSON Parsing
        raw_content = res.choices[0].message.content.strip()
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
            raw_content = re.sub(r"\n?```$", "", raw_content).strip()

        try:
            parsed_data = json.loads(raw_content)
        except json.JSONDecodeError:
            match = re.search(r'\[\s*\{.*?\}\s*\]', raw_content, re.DOTALL)
            if match: parsed_data = json.loads(match.group(0))
            else: raise ValueError("JSON parse error")
        
        # Coerce into standard array format
        if isinstance(parsed_data, list): parsed_array = parsed_data
        elif isinstance(parsed_data, dict):
            parsed_array = parsed_data.get("results", [])
            if not parsed_array:
                for val in parsed_data.values():
                    if isinstance(val, list):
                        parsed_array = val
                        break
        else: parsed_array = []
        
        # Verify alignment and return
        if len(parsed_array) == len(comments_batch):
            for idx, item in enumerate(parsed_array):
                item["temp_id"] = str(comments_batch[idx][0])
            tqdm.write(f"  [⚡] Time: {duration:.2f}s | Cache Hits: {cache_hits} | Misses: {cache_misses} | Out tkns: {res.usage.completion_tokens}")
            return parsed_array
            
        raise ValueError(f"Batch mismatch: {len(parsed_array)} / {len(comments_batch)}")
        
    except Exception as e:
        if attempt <= 4:
            wait_time = min(2 ** attempt, 10)
            time.sleep(wait_time)
            return label_batch(comments_batch, attempt + 1)
        return []

# ==========================================
# 5. PIPELINE EXECUTION
# ==========================================
def run_experiment():
    if df.empty:
        print("❌ No data available to label. Stopping pipeline.")
        return

    batches = [list(zip(df["temp_id"], df["body_clean"]))[i:i + 20] for i in range(0, len(df), 20)]
    all_labels = []
    lock = threading.Lock()

    print(f"\n🚀 Running Inference on {len(df)} experiment rows across {len(batches)} batches...")
    
    with tqdm(total=len(batches), desc="API Labeling Progress") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(label_batch, batch) for batch in batches]
            for future in as_completed(futures):
                batch_result = future.result()
                with lock:
                    all_labels.extend(batch_result)
                    if len(all_labels) % 200 < 20 and len(all_labels) > 0:
                        pd.DataFrame(all_labels).to_csv(CHECKPOINT_PATH, index=False)
                pbar.update(1)

    labels_df = pd.DataFrame(all_labels)
    if not labels_df.empty:
        labels_df["temp_id"] = labels_df["temp_id"].astype(str)

        KEY_MAPPING = {
            "analysis": "step_by_step_analysis", "pv": "profanity_vulgarity", "tah": "targeted_abuse_harassment",
            "dhs": "discriminatory_hate_speech", "cst": "caste", "cr": "communal_religious", 
            "rx": "regional_xenophobic", "mg": "misogyny_gender"
        }
        labels_df.rename(columns=KEY_MAPPING, inplace=True)
        
        # Merge back to DF. Dropping "body_clean" so only the raw "body" remains
        final_df = df.merge(labels_df, on="temp_id", how="left").drop(columns=["temp_id", "body_clean", "pv", "tah", "dhs", "cst", "cr", "rx", "mg"], errors='ignore')
        final_df.to_csv(FINAL_CSV_PATH, index=False)
        print(f"\n✅ EXPERIMENT COMPLETE! Saved labeled dataset to:\n   {FINAL_CSV_PATH}")
        
        # Output Statistical Analysis
        analyze_distribution(final_df)
    else:
        print("\n❌ Experiment failed to produce labels.")

# ==========================================
# 6. DISTRIBUTION STATISTICAL REPORT
# ==========================================
def analyze_distribution(rdf):
    print("\n==================================================")
    print(" 📊 EXPERIMENT DATA DISTRIBUTION ANALYSIS")
    print("==================================================")
    total = len(rdf)
    
    rdf['is_toxic'] = rdf[['profanity_vulgarity', 'targeted_abuse_harassment', 'discriminatory_hate_speech']].max(axis=1)
    toxic = rdf['is_toxic'].sum()
    clean = total - toxic
    
    print(f"Total Processed Records : {total}")
    print(f"🟢 Clean Comments       : {clean} ({(clean/total)*100:.1f}%)")
    print(f"🔴 Toxic Comments       : {toxic} ({(toxic/total)*100:.1f}%)\n")
    
    print("--- 🎯 Top-Level & Sub-Category Breakdowns ---")
    flags = [
        ('Profanity / Vulgarity', 'profanity_vulgarity'),
        ('Targeted Abuse', 'targeted_abuse_harassment'),
        ('Hate Speech (dhs)', 'discriminatory_hate_speech'),
        ('  ↳ Caste (cst)', 'caste'),
        ('  ↳ Communal (cr)', 'communal_religious'),
        ('  ↳ Regional (rx)', 'regional_xenophobic'),
        ('  ↳ Misogyny (mg)', 'misogyny_gender')
    ]
    for name, col in flags:
        if col in rdf.columns:
            cnt = rdf[col].sum()
            print(f"{name:<24}: {cnt:>4} ({(cnt/total)*100:>5.1f}%)")

    if 'subreddit' in rdf.columns:
        print("\n--- 🏟️ Toxicity Yield by Subreddit ---")
        sub_stats = rdf.groupby('subreddit')['is_toxic'].agg(['count', 'sum'])
        sub_stats['toxic_pct'] = (sub_stats['sum'] / sub_stats['count']) * 100
        sub_stats = sub_stats.sort_values(by='toxic_pct', ascending=False)
        for sub, row in sub_stats.iterrows():
            print(f"r/{sub:<20}: {int(row['sum']):>3}/{int(row['count']):>3} toxic ({row['toxic_pct']:>5.1f}%)")

run_experiment()

# ==========================================
# 7. HUNTER-SEEKER KEYWORD HARVESTER
# ==========================================
import re
from collections import Counter

def extract_minority_keywords(df, top_n=12):
    print("\n==================================================")
    print(" 🕵️ HUNTER-SEEKER DEBUG ALERTS (KEYWORD HARVESTER)")
    print("==================================================")
    
    # 🛑 Comprehensive Hinglish & English Stop Words
    stop_words = set([
        "hai", "ki", "ko", "se", "aur", "hi", "mein", "pe", "ye", "yeh", "woh", "tha", "thi", 
        "ka", "ke", "toh", "bhi", "na", "nahi", "kar", "liye", "kya", "ek", "jo", "tu", "tum",
        "aap", "ne", "is", "kuch", "koi", "sirf", "sab", "ab", "karna", "baat", "ho", "raha",
        "the", "a", "an", "and", "or", "but", "if", "for", "to", "of", "in", "on", "with", 
        "at", "by", "from", "up", "about", "into", "over", "after", "is", "are", "am", "was", 
        "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "i", "you", 
        "he", "she", "it", "we", "they", "this", "that", "these", "those", "my", "your", "his", 
        "her", "their", "just", "like", "so", "how", "what", "when", "where", "why", "who", 
        "not", "out", "then", "there", "can", "will", "would", "karo", "log", "wale", "kese"
    ])
    
    # The sub-categories we need to aggressively harvest for
    target_columns = ['caste', 'communal_religious', 'regional_xenophobic', 'misogyny_gender']
    
    for col in target_columns:
        if col not in df.columns: 
            continue
            
        # Filter for rows where this specific flag was triggered
        subset = df[df[col] == 1]['body'] 
        
        if subset.empty:
            print(f"[{col}] ⚠️ No data available to analyze.")
            continue
            
        # Combine all text, convert to lowercase, and extract alphanumeric words (3+ chars)
        all_text = " ".join(subset.astype(str).tolist()).lower()
        tokens = re.findall(r'\b[a-z]{3,}\b', all_text)
        
        # Filter out the stop words
        meaningful_words = [word for word in tokens if word not in stop_words]
        
        # Count frequencies
        word_counts = Counter(meaningful_words)
        top_words = word_counts.most_common(top_n)
        
        # Format the output for the console
        signals = ", ".join([f"'{word}' ({count})" for word, count in top_words])
        print(f"[{col:<18}] Signals: {signals}\n")

# To run it on your freshly saved final dataset:
# Ensure you are loading the file we just saved in the previous step
try:
    final_output_df = pd.read_csv(FINAL_CSV_PATH)
    extract_minority_keywords(final_output_df)
except Exception as e:
    print(f"❌ Could not load final CSV for keyword analysis: {e}")
