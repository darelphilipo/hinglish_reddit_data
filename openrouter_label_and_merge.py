import os
import json
import time
import requests
import re
import random
import threading
import psutil
import pandas as pd
import duckdb
import html
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
from huggingface_hub import HfApi
from cleantext import clean
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & OPENROUTER SETUP
# ==========================================
TARGET_ROWS = int(os.environ.get("TARGET_ROWS", 10000))
LIVE_MERGE = os.environ.get("LIVE_MERGE", "false").lower() == "true"
RUN_ID = os.environ.get("GITHUB_RUN_ID", str(int(time.time())))
SEED_VALUE = int(RUN_ID) % 100000 
random.seed(SEED_VALUE)

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not OPENROUTER_KEY:
    raise ValueError("❌ OPENROUTER_KEY environment variable is missing.")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN environment variable is missing (Required to read raw data).")

# Smart Fallback Routing: Nemotron set as default, Gemma as secondary
FALLBACK_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "z-ai/glm-5.2:free"
]

OPENROUTER_REASONING = {
    "enabled": True,
    "effort": "low"  
}

MAX_WORKERS = 2  

HF_REPO_ID = "darelphilip/hinglish-toxicity"
LEDGER_PATH = './seen_ids_ledger.txt'
SUBREDDIT_URL = "https://raw.githubusercontent.com/darelphilipo/hinglish_reddit_data/main/prompt/subreddits.json"
PROMPT_URL = "https://raw.githubusercontent.com/darelphilipo/hinglish_reddit_data/main/prompt/System_Prompt"
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

client = OpenAI(
    api_key=OPENROUTER_KEY, 
    base_url="https://openrouter.ai/api/v1",
    timeout=60.0, 
    default_headers={
        "HTTP-Referer": "https://github.com/darelphilipo/hinglish_reddit_data",
        "X-Title": "Hinglish Toxicity Pipeline"
    }
)
api = HfApi(token=HF_TOKEN)

# ==========================================
# 2. LOAD SUBREDDITS & DYNAMIC SYSTEM PROMPT
# ==========================================
print(f"\n🔧 Loading Subreddit Configurations & Prompt...", flush=True)
try:
    resp = requests.get(SUBREDDIT_URL, timeout=10)
    resp.raise_for_status()
    sub_data = resp.json()
except Exception as e:
    raise RuntimeError(f"❌ Failed to fetch subreddits.json from GitHub: {e}")

config_toggles = sub_data.get("config", {})
categories = sub_data.get("categories", {})

TIER1_SUBS, TIER2_SUBS = [], []
seen_tier2 = set()

if config_toggles.get("toxicity_focused", 1) == 1:
    TIER1_SUBS = [s.lower() for s in categories.get("toxicity_focused", [])]

for cat_name, sub_list in categories.items():
    if cat_name != "toxicity_focused" and config_toggles.get(cat_name, 1) == 1:
        for s in sub_list:
            s_clean = s.lower()
            if s_clean not in seen_tier2 and s_clean not in TIER1_SUBS:
                seen_tier2.add(s_clean)
                TIER2_SUBS.append(s_clean)

T3_QUOTA = TARGET_ROWS 
print(f"   ↳ Quota Target (New Dataset): {T3_QUOTA:,} Rows", flush=True)

try:
    response = requests.get(PROMPT_URL, timeout=10)
    response.raise_for_status()
    SYSTEM_PROMPT = response.text.strip()
    print("✅ System Prompt loaded successfully.", flush=True)
except Exception as e:
    raise RuntimeError(f"❌ Failed to fetch System Prompt from GitHub: {e}")

# ==========================================
# 3. DUCKDB MULTI-TIER EXTRACTION ENGINE
# ==========================================
print(f"\n🦆 Initializing DuckDB Engine (Dynamic Seed: {SEED_VALUE})...", flush=True)
con = duckdb.connect()
con.execute("PRAGMA memory_limit='6GB';") 
con.execute("PRAGMA threads=4;") 
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute(f"CREATE SECRET hf_auth (TYPE HUGGINGFACE, TOKEN '{HF_TOKEN}');")

t3_raw_df = pd.DataFrame()
if T3_QUOTA > 0:
    print(f"🔍 Streaming from darelphilip/reddit_indian_subs (Target: {T3_QUOTA:,} rows)...", flush=True)
    all_active_subs = list(set(TIER1_SUBS + TIER2_SUBS))
    if not all_active_subs:
        all_active_subs = ['indiaspeaks', 'india', 'bihar', 'delhi', 'bangalore', 'developersindia']
        
    subs_formatted = ", ".join([f"'{s.replace(chr(39), chr(39)+chr(39))}'" for s in all_active_subs])
    fetch_limit = max(10000, int(T3_QUOTA * 3.0)) 
    
    t3_query = f"""
    SELECT id, body, LOWER(subreddit) as subreddit, created_utc, strftime(to_timestamp(created_utc), '%Y-%m') as year_month, 'Tier 3' as tier_label
    FROM read_parquet('hf://datasets/darelphilip/reddit_indian_subs/**/*.parquet', union_by_name=True)
    WHERE LOWER(subreddit) IN ({subs_formatted})
      AND body IS NOT NULL
      AND body NOT IN ('[deleted]', '[removed]', '')
      AND length(body) BETWEEN 10 AND 1000
    USING SAMPLE {fetch_limit} ROWS
    """
    try:
        t3_raw_df = con.query(t3_query).to_df()
        print(f"   ✅ Pulled {len(t3_raw_df):,} random raw comments from live dataset.", flush=True)
    except Exception as e:
        print(f"   ❌ Query Failed: {e}", flush=True)

raw_df = t3_raw_df.copy(deep=True)
if raw_df.empty:
    raise ValueError("❌ Extraction returned 0 comments.")

# ==========================================
# 4. SANITIZATION & DEDUPLICATION
# ==========================================
print("\n🧹 Sanitizing text...", flush=True)

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

raw_df['body_clean'] = raw_df['body'].apply(sanitize_text)
raw_df = raw_df[raw_df['body_clean'].str.len() > 5].copy(deep=True)

print("🛡️ Applying aggressive token-saving deduplication...", flush=True)
initial_count = len(raw_df)
raw_df['dedup_key'] = raw_df['body_clean'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
raw_df.drop_duplicates(subset=['dedup_key'], keep='first', inplace=True)
raw_df.drop(columns=['dedup_key'], inplace=True)
print(f"✂️ Dropped {initial_count - len(raw_df)} spam/copy-paste variants.", flush=True)

if os.path.exists(LEDGER_PATH):
    with open(LEDGER_PATH, 'r') as f:
        seen = set(line.strip() for line in f if line.strip())
    prev_len = len(raw_df)
    raw_df = raw_df[~raw_df['id'].isin(seen)]
    print(f"🛡️ Filtered out {prev_len - len(raw_df)} comments processed in previous workflow runs.", flush=True)

if len(raw_df) > TARGET_ROWS:
    df = raw_df.sample(n=TARGET_ROWS, random_state=SEED_VALUE).reset_index(drop=True)
else:
    df = raw_df.reset_index(drop=True)

df.drop(columns=['index', 'tier_label'], inplace=True, errors='ignore')
print(f"🎯 Final Inference Pool: {len(df):,} rows.", flush=True)

# ==========================================
# 5. OPENROUTER INFERENCE ENGINE (WITH FAILOVER)
# ==========================================
def label_batch(comments_batch, attempt=1, model_idx=0):
    numbered = "\n".join(f'ID: {cid} | Comment: {body}' for cid, body in comments_batch)
    user_prompt = f"Label these comments:\n{numbered}"
    
    current_model = FALLBACK_MODELS[model_idx]
    
    try:
        res = client.chat.completions.create(
            model=current_model, 
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}], 
            temperature=0.1, 
            # We keep JSON format, but if a model rejects it, the safety catch below handles it
            response_format={"type": "json_object"},
            extra_body={"reasoning": OPENROUTER_REASONING}
        )
        
        # --- FIX: Safe extraction to prevent 'NoneType' subscript errors ---
        if not hasattr(res, 'choices') or not res.choices:
            raise ValueError(f"API returned empty or invalid choices (Model likely rejected a parameter).")
            
        msg_obj = res.choices[0].message
        raw_content = (msg_obj.content or "").strip()
        # -----------------------------------------------------------------
        
        # --- DEBUG: CHECK FOR THINKING/REASONING ---
        reasoning_text = getattr(msg_obj, 'reasoning', None)
        if reasoning_text:
            print(f"\n[🧠 {current_model} THINKING (API Level)]:\n{reasoning_text[:300]}...\n", flush=True)
        elif "<think>" in raw_content:
            think_block = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)
            if think_block:
                print(f"\n[🧠 {current_model} THINKING (Tag Level)]:\n{think_block.group(1).strip()[:300]}...\n", flush=True)
        # -------------------------------------------

        # Clean up any markdown blocks or leftover think tags before parsing JSON
        raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
            raw_content = re.sub(r"\n?```$", "", raw_content).strip()

        try:
            content = json.loads(raw_content)
        except json.JSONDecodeError:
            raise ValueError(f"Model failed to output valid JSON. Output was: {raw_content[:100]}...")

        # Ensure we are working with a list of results
        results = content.get("results", []) if isinstance(content, dict) else content
        
        if isinstance(results, list) and len(results) == len(comments_batch):
            for idx, item in enumerate(results): 
                if not isinstance(item, dict):
                    raise ValueError(f"Model returned invalid item format in array: {item}")
                item["id"] = str(comments_batch[idx][0])
            return results
            
        raise ValueError(f"Batch mismatch: Expected {len(comments_batch)} results, got {len(results) if isinstance(results, list) else 'non-list'}")
        
    except Exception as e:
        err_msg = str(e)
        
        # Immediate Failover Logic: Catches Congestion, 404s, AND soft-error Invalid Responses
        failover_triggers = ["upstream_provider_shared_pool", "Provider returned error", "404", "unavailable", "empty or invalid choices", "JSONDecodeError"]
        if any(trigger in err_msg for trigger in failover_triggers):
            if model_idx + 1 < len(FALLBACK_MODELS):
                next_model = FALLBACK_MODELS[model_idx + 1]
                if attempt == 1:
                    print(f"\n   🔄 {current_model} is congested/incompatible. Failing over to {next_model}...", flush=True)
                return label_batch(comments_batch, attempt, model_idx + 1)
        
        if attempt <= 5:
            wait_time = min(3 ** attempt, 30)
            print(f"\n   ⏳ OpenRouter Error ({current_model}) (Attempt {attempt}): {e}. Retrying in {wait_time}s...", flush=True)
            time.sleep(wait_time)
            return label_batch(comments_batch, attempt + 1, model_idx)
            
        print(f"\n⚠️ Failed batch after 5 attempts on all models: {e}", flush=True)
        return []
# ==========================================
# 6. DUAL-SCHEMA FORMATTING
# ==========================================
print("\n🛠️ Formatting Dual-Schema (RoBERTa + Sarvam ChatML)...", flush=True)
final_df = final_df.dropna(subset=['pv'])

for short_k, long_k in KEY_MAPPING.items():
    if short_k in final_df.columns and long_k not in final_df.columns:
        final_df[long_k] = final_df[short_k]
    elif long_k not in final_df.columns:
        final_df[long_k] = 0

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
        if has_analysis: record['analysis'] = str(row['analysis']).strip() 
        formatted_records.append(record)
    except Exception as e: pass

hf_master_df = pd.DataFrame(formatted_records)
total_lbl = len(hf_master_df)

# ==========================================
# 7. EXPORT LOGIC (MERGE vs REVIEW)
# ==========================================
print("\n==================================================", flush=True)
print(f" 📊 FINAL RUN DISTRIBUTION (Yield: {total_lbl:,} rows)", flush=True)
print("==================================================", flush=True)
core_cols = list(KEY_MAPPING.values())
toxic_mask = hf_master_df[core_cols].max(axis=1) == 1
total_toxic = int(toxic_mask.sum())
print(f"Toxic Comments: {total_toxic:,} ({total_toxic/max(1, total_lbl)*100:.1f}%)", flush=True)

if LIVE_MERGE:
    print(f"\n☁️ [LIVE_MERGE=TRUE] Initiating Chunked Upload to Hugging Face...", flush=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunks = [hf_master_df[i:i + CHUNK_SIZE] for i in range(0, len(hf_master_df), CHUNK_SIZE)]
    
    for chunk_idx, chunk_df in enumerate(chunks):
        chunk_name = f"openrouter_worker_{timestamp_str}_part_{chunk_idx:03d}.parquet"
        hf_path = f"data/{chunk_name}"
        local_parquet_path = f"./{chunk_name}"
        
        print(f"   📤 Uploading {chunk_name} ({len(chunk_df):,} rows)...", flush=True)
        chunk_df.to_parquet(local_parquet_path, engine='pyarrow', index=False)
        
        try:
            api.upload_file(path_or_fileobj=local_parquet_path, path_in_repo=hf_path, repo_id=HF_REPO_ID, repo_type="dataset")
            print(f"   ✅ Successfully pushed {chunk_name} to HF.", flush=True)
            with open(LEDGER_PATH, 'a') as f:
                for cid in chunk_df['id'].tolist(): f.write(f"{cid}\n")
        except Exception as e:
            print(f"   ❌ Failed to upload chunk: {e}", flush=True)
        finally:
            if os.path.exists(local_parquet_path): os.remove(local_parquet_path)
            
    print("\n✅ Merge to original dataset successfully completed.", flush=True)

else:
    print(f"\n📁 [LIVE_MERGE=FALSE] Saving to local CSV for manual review...", flush=True)
    os.makedirs("output", exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"output/openrouter_labeled_review_{timestamp_str}.csv"
    hf_master_df.to_csv(output_csv, index=False)
    print(f"   ✅ Saved {len(hf_master_df):,} rows to {output_csv}.", flush=True)
    print("   💡 Review this file via GitHub Artifacts. To merge back to Hugging Face, run the workflow with live_merge=true.", flush=True)
