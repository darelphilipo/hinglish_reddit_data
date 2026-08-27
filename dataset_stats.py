import pandas as pd
import os
from datasets import load_dataset

print("📊 Initializing Hugging Face Dataset Statistical Auditor...")

HF_REPO_ID = "darelphilip/hinglish-toxicity"

# ==========================================
# 1. LOAD DATASET FROM HUGGING FACE
# ==========================================
print(f"\n📥 Pulling live dataset from Hugging Face: {HF_REPO_ID}...")
try:
    ds = load_dataset(HF_REPO_ID, split="train", download_mode="force_redownload")
    df = ds.to_pandas()
    print(f"   ↳ Successfully loaded {len(df):,} rows.")
except Exception as e:
    print(f"❌ Failed to load dataset from HF: {e}")
    exit(1)

total_rows = len(df)
if total_rows == 0:
    print("⚠️ Dataset is empty.")
    exit(0)

# ==========================================
# 2. COMPUTE METRICS & DISTRIBUTIONS
# ==========================================
unique_ids = df['id'].nunique() if 'id' in df.columns else 'N/A'
unique_texts = df['text'].nunique() if 'text' in df.columns else 'N/A'
duplicates = total_rows - unique_texts if isinstance(unique_texts, int) else 0

core_toxic_cols = ['profanity_vulgarity', 'targeted_abuse_harassment', 'discriminatory_hate_speech']
all_label_cols = core_toxic_cols + ['caste', 'communal_religious', 'regional_xenophobic', 'misogyny_gender']

for col in all_label_cols:
    if col not in df.columns:
        df[col] = 0

toxic_mask = df[core_toxic_cols].max(axis=1) == 1
total_toxic = int(toxic_mask.sum())
total_clean = total_rows - total_toxic

if 'text' in df.columns:
    df['text_len'] = df['text'].astype(str).str.len()
    avg_len = df['text_len'].mean()
    max_len = df['text_len'].max()
    min_len = df['text_len'].min()
else:
    avg_len, max_len, min_len = 0, 0, 0

top_subs = df['subreddit'].value_counts().head(5).to_dict() if 'subreddit' in df.columns else {}
ym_dist = df['year_month'].value_counts().sort_index().to_dict() if 'year_month' in df.columns else {}

# ==========================================
# 3. PRINT COMPREHENSIVE REPORT
# ==========================================
print("\n==================================================")
print(" 📈 HINGLISH TOXICITY DATASET - STATISTICAL AUDIT")
print("==================================================")
print(f"Target Repository       : https://huggingface.co/datasets/{HF_REPO_ID}")
print(f"Total Rows (Records)    : {total_rows:,}")
print(f"Unique Comment Texts    : {unique_texts:,} ({duplicates:,} duplicate variants)")
print(f"Unique Reddit IDs       : {unique_ids}")

print("\n--- 🟢 CLASSIFICATION SPLIT ---")
print(f"Clean Comments          : {total_clean:,} ({(total_clean/total_rows)*100:.1f}%)")
print(f"Toxic Comments          : {total_toxic:,} ({(total_toxic/total_rows)*100:.1f}%)")

print("\n--- 🏷️ DETAILED CATEGORY BREAKDOWN ---")
for col in all_label_cols:
    count = int(df[col].sum())
    pct = (count / total_rows) * 100 if total_rows > 0 else 0
    print(f"  - {col:<26} : {count:>6,} rows ({pct:>5.2f}%)")

print("\n--- 📝 TEXT LENGTH METRICS ---")
print(f"Average Character Length: {avg_len:.1f} chars")
print(f"Min / Max Length        : {min_len} / {max_len} chars")

print("\n--- 🌐 TOP 5 SUBREDDITS ---")
for sub, count in top_subs.items():
    print(f"  - r/{sub:<20} : {count:,} rows")

print("\n--- 📅 TIME-SERIES DISTRIBUTION (YEAR-MONTH) ---")
for ym, count in ym_dist.items():
    if pd.notna(ym):
        print(f"  - {ym} : {count:,} rows")

print("==================================================")
print("✅ Statistical audit completed successfully!")
