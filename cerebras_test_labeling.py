import os
import sys
import json
import time
import requests
import re
import random
import pandas as pd
import duckdb
import html
from tqdm import tqdm
from openai import OpenAI
from cleantext import clean
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & CEREBRAS SETUP
# ==========================================
# For a test trial on a free tier, default to a much smaller row count
TARGET_ROWS = int(os.environ.get("TARGET_ROWS", 100))
RUN_ID = os.environ.get("GITHUB_RUN_ID", str(int(time.time())))
SEED_VALUE = int(RUN_ID) % 100000 
random.seed(SEED_VALUE)

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not CEREBRAS_API_KEY:
    raise ValueError("❌ CEREBRAS_API_KEY environment variable is missing.")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN environment variable is missing (Required to read raw data).")

MODEL_ID = "qwen-3.8-27b"
# Max 5 req/min on free tier. Batches of 10 comments to keep token limits safe.
BATCH_SIZE = 10 

SUBREDDIT_URL = "https://raw.githubusercontent.com/darelphilipo/hinglish_reddit_data/main/prompt/subreddits.json"
PROMPT_URL = "https://raw.githubusercontent.com/darelphilipo/hinglish_reddit_data/main/prompt/System_Prompt"

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

# Cerebras uses the standard OpenAI SDK structure
client = OpenAI(
    api_key=CEREBRAS_API_KEY, 
    base_url="https://api.cerebras.ai/v1",
    timeout=60.0
)

# Global trackers for free tier monitoring
global_req_count = 0
global_input_tokens = 0
global_output_tokens = 0

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

print(f"   ↳ Quota Target (Test Run): {TARGET_ROWS:,} Rows", flush=True)

try:
    response = requests.get(PROMPT_URL, timeout=10)
    response.raise_for_status()
    SYSTEM_PROMPT = response.text.strip()
    print("✅ System Prompt loaded successfully.", flush=True)
except Exception as e:
    raise RuntimeError(f"❌ Failed to fetch System Prompt from GitHub: {e}")

# ==========================================
# 3. DUCKDB EXTRACTION ENGINE
# ==========================================
print(f"\n🦆 Initializing DuckDB Engine (Dynamic Seed: {SEED_VALUE})...", flush=True)
con = duckdb.connect()
con.execute("PRAGMA memory_limit='6GB';") 
con.execute("PRAGMA threads=4;") 
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute(f"CREATE SECRET hf_auth (TYPE HUGGINGFACE, TOKEN '{HF_TOKEN}');")

raw_df = pd.DataFrame()
if TARGET_ROWS > 0:
    print(f"🔍 Streaming from darelphilip/reddit_indian_subs...", flush=True)
    all_active_subs = list(set(TIER1_SUBS + TIER2_SUBS))
    if not all_active_subs:
        all_active_subs = ['indiaspeaks', 'india', 'bihar', 'delhi', 'bangalore', 'developersindia']
        
    subs_formatted = ", ".join([f"'{s.replace(chr(39), chr(39)+chr(39))}'" for s in all_active_subs])
    fetch_limit = max(1000, int(TARGET_ROWS * 3.0)) 
    
    t3_query = f"""
    SELECT id, body, LOWER(subreddit) as subreddit, created_utc, strftime(to_timestamp(created_utc), '%Y-%m') as year_month
    FROM read_parquet('hf://datasets/darelphilip/reddit_indian_subs/**/*.parquet', union_by_name=True)
    WHERE LOWER(subreddit) IN ({subs_formatted})
      AND body IS NOT NULL
      AND body NOT IN ('[deleted]', '[removed]', '')
      AND length(body) BETWEEN 10 AND 1000
    USING SAMPLE {fetch_limit} ROWS
    """
    
    try:
        raw_df = con.query(t3_query).to_df()
        print(f"   ✅ Pulled {len(raw_df):,} random raw comments.", flush=True)
    except Exception as e:
        print(f"   ❌ DuckDB Network/Query Error: {e}", flush=True)
        sys.exit(1)

if raw_df.empty:
    print("❌ Extraction returned 0 comments. Exiting cleanly.", flush=True)
    sys.exit(1)

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

raw_df['dedup_key'] = raw_df['body_clean'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
raw_df.drop_duplicates(subset=['dedup_key'], keep='first', inplace=True)
raw_df.drop(columns=['dedup_key'], inplace=True)

if len(raw_df) > TARGET_ROWS:
    df = raw_df.sample(n=TARGET_ROWS, random_state=SEED_VALUE).reset_index(drop=True)
else:
    df = raw_df.reset_index(drop=True)

print(f"🎯 Final Inference Pool: {len(df):,} rows.", flush=True)

# ==========================================
# 5. CEREBRAS INFERENCE ENGINE 
# ==========================================
def label_batch(comments_batch, attempt=1):
    global global_req_count, global_input_tokens, global_output_tokens
    
    # Strictly enforce 5 req/min Cerebras Free Tier limit
    if global_req_count > 0:
        time.sleep(15) 

    numbered = "\n".join(f'ID: {cid} | Comment: {body}' for cid, body in comments_batch)
    user_prompt = f"Label these comments:\n{numbered}"
    
    try:
        res = client.chat.completions.create(
            model=MODEL_ID, 
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}], 
            temperature=0.1, 
            response_format={"type": "json_object"},
            reasoning_effort="none"  # DeepSeek Strategy: Turn off native thinking
        )
        
        global_req_count += 1
        
        # Log exact token usage from Cerebras API
        if res.usage:
            in_tok = res.usage.prompt_tokens
            out_tok = res.usage.completion_tokens
            global_input_tokens += in_tok
            global_output_tokens += out_tok
            print(f"\n   [Cerebras Limits] Req: {global_req_count} | Batch In: {in_tok} | Batch Out: {out_tok} | Total MTD: {global_input_tokens+global_output_tokens:,}/90,000", flush=True)

        raw_content = (res.choices[0].message.content or "").strip()

        # Clean up any residual markdown wrappers
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
            raw_content = re.sub(r"\n?```$", "", raw_content).strip()

        try:
            content = json.loads(raw_content)
        except json.JSONDecodeError:
            raise ValueError(f"Model failed to output valid JSON. Output was: {raw_content[:100]}...")

        results = content.get("results", []) if isinstance(content, dict) else content
        
        if isinstance(results, list) and len(results) == len(comments_batch):
            for idx, item in enumerate(results): 
                if not isinstance(item, dict):
                    raise ValueError(f"Model returned invalid item format in array: {item}")
                item["id"] = str(comments_batch[idx][0])
            return results
            
        raise ValueError(f"Batch mismatch: Expected {len(comments_batch)} results, got {len(results) if isinstance(results, list) else 'non-list'}")
        
    except Exception as e:
        if attempt <= 3:
            print(f"\n   ⏳ API Error (Attempt {attempt}): {e}. Retrying in 15s...", flush=True)
            time.sleep(15)
            return label_batch(comments_batch, attempt + 1)
            
        print(f"\n⚠️ Failed batch after 3 attempts: {e}", flush=True)
        return []

if df.empty:
    print(f"❌ Worker: No valid data to label.", flush=True)
    sys.exit(0)

batches = [list(zip(df["id"], df["body_clean"]))[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
all_labels = []

print(f"\n🚀 Running Serial Inference on {len(df):,} rows across {len(batches):,} batches (Cerebras API)...", flush=True)

# Process serially to respect rate limits
for batch in tqdm(batches, desc="Inference Progress"): 
    result = label_batch(batch)
    if result:
        all_labels.extend(result)

labels_df = pd.DataFrame(all_labels)

if labels_df.empty:
    print("❌ All inference requests failed. Check API limits.", flush=True)
    sys.exit(1)

labels_df["id"] = labels_df["id"].astype(str)
df["id"] = df["id"].astype(str)
final_df = df.merge(labels_df, on="id", how="inner")
final_df.drop(columns=["body_clean"], errors='ignore', inplace=True)

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
# 7. EXPORT LOGIC (TEST CSV OUTPUT ONLY)
# ==========================================
print("\n==================================================", flush=True)
print(f" 📊 FINAL RUN DISTRIBUTION (Yield: {total_lbl:,} rows)", flush=True)
print("==================================================", flush=True)
core_cols = list(KEY_MAPPING.values())
toxic_mask = hf_master_df[core_cols].max(axis=1) == 1
total_toxic = int(toxic_mask.sum())
print(f"Toxic Comments: {total_toxic:,} ({total_toxic/max(1, total_lbl)*100:.1f}%)", flush=True)
print(f"Total Cerebras API Tokens Consumed: {global_input_tokens + global_output_tokens:,}")

print(f"\n📁 Saving to local CSV for manual test review...", flush=True)
os.makedirs("output", exist_ok=True)
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = f"output/cerebras_qwen_test_{timestamp_str}.csv"
hf_master_df.to_csv(output_csv, index=False)
print(f"   ✅ Saved {len(hf_master_df):,} rows to {output_csv}.", flush=True)
print("   💡 Review this CSV file to evaluate Qwen 3.8 performance before deploying the merge pipeline.", flush=True)
