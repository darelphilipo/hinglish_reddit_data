import pandas as pd
import json
import os
import time
import requests
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

print("🚀 Initializing Hugging Face Dual-Schema Migration Tool...")

# ==========================================
# 1. CONFIGURATION & SECRETS
# ==========================================
CSV_PATH = './labelled_output/master_baseline_tier1.csv'
LEDGER_PATH = './migration_ledger.json'
CHUNK_SIZE = 2500  # Uploads in batches of 2,500 to create safe checkpoints

HF_TOKEN = os.environ.get("HF_TOKEN")
# ⚠️ CHANGE THIS to your actual Hugging Face username and dataset name
HF_REPO_ID = "darelphilip/hinglish-toxicity" 

if not HF_TOKEN:
    print("❌ Error: HF_TOKEN environment variable is missing.")
    exit(1)

if not os.path.exists(CSV_PATH):
    print(f"❌ Error: Master dataset not found at {CSV_PATH}")
    exit(1)

api = HfApi(token=HF_TOKEN)

# Ensure HF Repo exists
try:
    create_repo(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN, exist_ok=True)
    print(f"✅ Hugging Face repository '{HF_REPO_ID}' is ready.")
except Exception as e:
    print(f"⚠️ Repo check/creation issue: {e}")

# ==========================================
# 2. SET SYSTEM PROMPT (For ChatML)
# ==========================================
print("\n🌐 Setting concise Student Prompt...")
SYSTEM_PROMPT = "You are an expert Hinglish content moderation AI. Analyze the following comment and output a JSON object containing the toxic classification flags and a brief analysis of the target and intent."

# ==========================================
# 3. LOAD DATA & LEDGER STATE
# ==========================================
print("\n📊 Loading Local Master Dataset...")
df = pd.read_csv(CSV_PATH)
initial_rows = len(df)
df.dropna(subset=['body', 'profanity_vulgarity'], inplace=True)
print(f"   ↳ {len(df):,} valid rows available for migration.")

# Initialize or Load Checkpoint Ledger
if os.path.exists(LEDGER_PATH):
    with open(LEDGER_PATH, 'r') as f:
        ledger = json.load(f)
    print(f"📂 Found existing migration ledger. {len(ledger.get('migrated_chunks', []))} chunks already uploaded.")
else:
    ledger = {"migrated_chunks": [], "total_rows_migrated": 0}

# ==========================================
# 4. DUAL-SCHEMA FORMATTING
# ==========================================
LABEL_COLUMNS = [
    'profanity_vulgarity', 'targeted_abuse_harassment', 'discriminatory_hate_speech', 
    'caste', 'communal_religious', 'regional_xenophobic', 'misogyny_gender'
]

print("\n🛠️ Formatting Dual-Schema (RoBERTa + Sarvam ChatML)...")
formatted_records = []

# Using tqdm with 10% update intervals for the formatting phase
update_interval = max(1, len(df) // 10)
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Formatting Rows", miniters=update_interval):
    try:
        # 1. Binary integer targets for RoBERTa
        labels = {col: int(row[col]) for col in LABEL_COLUMNS}
        
        # 2. Add Analysis/Reasoning for Sarvam full LLM training
        has_analysis = 'analysis' in row and pd.notna(row['analysis'])
        if has_analysis:
            labels['analysis'] = str(row['analysis']).strip()
            
        # 3. ChatML format for Instruction Tuning
        chatml_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(row['body']).strip()},
            {"role": "assistant", "content": json.dumps(labels, ensure_ascii=False)}
        ]
        
        # 4. Construct the complete row
        record = {
            "id": str(row.get('id', f"migrated_{idx}")),
            "text": str(row['body']).strip(),
            "subreddit": str(row.get('subreddit', 'unknown')),
            "created_utc": row.get('created_utc', None),
            "year_month": str(row['year_month']) if pd.notna(row.get('year_month')) else None
        }
        
        # Merge binary labels and ChatML messages into the final Hugging Face row
        record.update({col: int(row[col]) for col in LABEL_COLUMNS}) 
        record["messages"] = chatml_messages
        
        # Keep analysis at the top level as well for easy access
        if has_analysis:
             record['analysis'] = str(row['analysis']).strip() 
        
        formatted_records.append(record)
    except Exception as e:
        print(f"⚠️ Skipped row {idx} due to formatting error: {e}")

master_df = pd.DataFrame(formatted_records)

# ==========================================
# 5. SHARDING & HUGGING FACE UPLOAD LOOP
# ==========================================
print(f"\n☁️ Initiating Chunked Upload to Hugging Face ({CHUNK_SIZE} rows/chunk)...")

# Calculate chunks
total_chunks = (len(master_df) // CHUNK_SIZE) + (1 if len(master_df) % CHUNK_SIZE > 0 else 0)
chunks = [master_df[i:i + CHUNK_SIZE] for i in range(0, len(master_df), CHUNK_SIZE)]

for chunk_idx, chunk_df in enumerate(chunks):
    chunk_name = f"migration_base_part_{chunk_idx:03d}.parquet"
    hf_path = f"data/{chunk_name}"
    
    if chunk_name in ledger["migrated_chunks"]:
        print(f"   ⏩ Skipping {chunk_name} (Already uploaded).")
        continue
        
    print(f"\n   📤 Uploading {chunk_name} ({len(chunk_df):,} rows)...")
    local_parquet_path = f"./{chunk_name}"
    
    # Save chunk locally as Parquet
    chunk_df.to_parquet(local_parquet_path, engine='pyarrow', index=False)
    
    # Retry mechanism for HF Rate Limits & Connection Drops
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=local_parquet_path,
                path_in_repo=hf_path,
                repo_id=HF_REPO_ID,
                repo_type="dataset"
            )
            print(f"   ✅ Successfully pushed {chunk_name} to HF.")
            
            # Update Ledger Checkpoint
            ledger["migrated_chunks"].append(chunk_name)
            ledger["total_rows_migrated"] += len(chunk_df)
            with open(LEDGER_PATH, 'w') as f:
                json.dump(ledger, f, indent=4)
                
            os.remove(local_parquet_path) # Clean up temp file
            time.sleep(2) # Brief sleep to respect HF rate limits
            break # Escape retry loop on success
            
        except Exception as e:
            print(f"   ⚠️ Upload failed on attempt {attempt}/{max_retries}: {e}")
            if attempt == max_retries:
                print("\n❌ CRITICAL: Max retries reached. Exiting script. Run again to resume from ledger.")
                if os.path.exists(local_parquet_path): os.remove(local_parquet_path)
                exit(1)
            time.sleep(min(3 ** attempt, 30)) # Exponential backoff

print("\n==================================================")
print(" 🎉 MIGRATION COMPLETE")
print("==================================================")
print(f"Total Rows Migrated : {ledger['total_rows_migrated']:,}")
print(f"Target Repository   : https://huggingface.co/datasets/{HF_REPO_ID}")
print("You can now safely delete the 'labelled_output' folder and master CSV!")
