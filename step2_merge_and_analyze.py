import pandas as pd
import os
import json
from datasets import load_dataset

print("🔄 Initializing Master Dynamic Target Analysis via Hugging Face...")

HF_REPO_ID = "darelphilip/hinglish-toxicity"
TARGETS_DIR = './prompt/'
TARGETS_PATH = os.path.join(TARGETS_DIR, 'pipeline_targets.json')

# ==========================================
# 1. DYNAMIC PIPELINE BLUEPRINT GENERATION
# ==========================================
GLOBAL_GOAL = int(os.environ.get("GLOBAL_DATASET_GOAL", 50000))
CLEAN_GOAL = GLOBAL_GOAL // 2
TOXIC_POOL = GLOBAL_GOAL - CLEAN_GOAL
CATEGORY_GOAL = TOXIC_POOL // 4

targets_dict = {
    "global_goal": GLOBAL_GOAL,
    "clean_data": CLEAN_GOAL,
    "categories": {
        "caste": CATEGORY_GOAL,
        "communal_religious": CATEGORY_GOAL,
        "regional_xenophobic": CATEGORY_GOAL,
        "misogyny_gender": CATEGORY_GOAL
    }
}

os.makedirs(TARGETS_DIR, exist_ok=True)
with open(TARGETS_PATH, 'w') as f:
    json.dump(targets_dict, f, indent=4)

print(f"🎯 Dynamic Blueprint Generated for {GLOBAL_GOAL:,} Total Rows (50/50 Split)")
print(f"   ↳ Target: {CLEAN_GOAL:,} Clean Rows | {CATEGORY_GOAL:,} per Toxic Category")
print(f"   ↳ Saved to {TARGETS_PATH}")

# ==========================================
# 2. PULL LIVE DATASET FROM HUGGING FACE
# ==========================================
print(f"\n📥 Pulling live dataset from Hugging Face: {HF_REPO_ID}...")
try:
    ds = load_dataset(HF_REPO_ID, split="train")
    master_df = ds.to_pandas()
    print(f"   ↳ Successfully loaded {len(master_df):,} rows from Hugging Face.")
except Exception as e:
    print(f"❌ Failed to load dataset from HF: {e}")
    exit(1)

# ==========================================
# 3. MEMORY DEDUPLICATION (For Accurate Counting)
# ==========================================
initial_len = len(master_df)
# We deduplicate on the 'text' column which holds the comment body
master_df.drop_duplicates(subset=['text'], keep='first', inplace=True)
dedup_count = initial_len - len(master_df)
if dedup_count > 0:
    print(f"✂️ Found {dedup_count} cross-run duplicate comments in memory.")

# ==========================================
# 4. ANALYZE FINAL DISTRIBUTIONS
# ==========================================
total_rows = len(master_df)
print("\n==================================================")
print(f" 📊 FINAL MASTER DISTRIBUTION REPORT ({total_rows:,} Rows)")
print("==================================================")

# Safely check for columns in case they are missing
core_toxic_cols = ['profanity_vulgarity', 'targeted_abuse_harassment', 'discriminatory_hate_speech']
for col in core_toxic_cols:
    if col not in master_df.columns:
        master_df[col] = 0

toxic_mask = master_df[core_toxic_cols].max(axis=1) == 1
total_toxic = toxic_mask.sum()
total_clean = total_rows - total_toxic

clean_shortfall = max(0, CLEAN_GOAL - total_clean)
print(f"🟢 Clean Comments  : {total_clean:,} / {CLEAN_GOAL:,} target | Shortfall: {clean_shortfall:,} ({ (total_clean/CLEAN_GOAL)*100 if CLEAN_GOAL>0 else 100:.1f}%)")
print(f"🔴 Toxic Comments  : {total_toxic:,} / {TOXIC_POOL:,} target | Shortfall: {max(0, TOXIC_POOL - total_toxic):,} ({ (total_toxic/TOXIC_POOL)*100 if TOXIC_POOL>0 else 100:.1f}%)\n")

print("--- Toxic Category Breakdown ---")
for col, target in targets_dict['categories'].items():
    if col in master_df.columns:
        count = int(master_df[col].sum())
    else:
        count = 0
    shortfall = max(0, target - count)
    print(f"{col:<22}: {count:>5} / {target:>4} target | Shortfall: {shortfall:,}")

print("\n==================================================")
print("✅ Blueprint generated and real-time HF distribution analysis complete!")
