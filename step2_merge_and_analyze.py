import pandas as pd
import glob
import os
import json

print("🔄 Initializing Master Merge & Dynamic Target Analysis...")

BASE_OUTPUT_DIR = './labelled_output/'
CHUNKS_DIR = os.path.join(BASE_OUTPUT_DIR, 'chunks/')
MASTER_PATH = os.path.join(BASE_OUTPUT_DIR, 'master_baseline_tier1.csv')
TARGETS_PATH = os.path.join(BASE_OUTPUT_DIR, 'pipeline_targets.json')

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

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
with open(TARGETS_PATH, 'w') as f:
    json.dump(targets_dict, f, indent=4)

print(f"🎯 Dynamic Blueprint Generated for {GLOBAL_GOAL:,} Total Rows (50/50 Split)")
print(f"   ↳ Target: {CLEAN_GOAL:,} Clean Rows | {CATEGORY_GOAL:,} per Toxic Category")
print(f"   ↳ Saved to {TARGETS_PATH}")

# ==========================================
# 2. LOCATE & MERGE CHUNKS
# ==========================================
all_files = glob.glob(os.path.join(CHUNKS_DIR, '*.csv'))

if not all_files:
    print("❌ No CSV chunk files found in ./labelled_output/chunks/. Exiting.")
    exit(1)

print(f"\n📦 Found {len(all_files)} dataset chunks. Merging...")

rename_mapping = {
    "pv": "profanity_vulgarity",
    "tah": "targeted_abuse_harassment",
    "dhs": "discriminatory_hate_speech",
    "cst": "caste",
    "cr": "communal_religious",
    "rx": "regional_xenophobic",
    "mg": "misogyny_gender"
}

# Fix: Rename columns BEFORE concatenating to prevent column duplication
df_list = []
for file in all_files:
    temp_df = pd.read_csv(file)
    temp_df.rename(columns=rename_mapping, inplace=True)
    df_list.append(temp_df)

master_df = pd.concat(df_list, ignore_index=True)

# Safety Net: If any duplicate columns somehow survived, merge them by taking the max value (1 overrides 0)
master_df = master_df.groupby(master_df.columns, axis=1).max()

# ==========================================
# 3. DEDUPLICATION
# ==========================================
initial_len = len(master_df)
master_df.drop_duplicates(subset=['body'], keep='first', inplace=True)
dedup_count = initial_len - len(master_df)
print(f"✂️ Dropped {dedup_count} cross-run duplicate comments.")

# ==========================================
# 4. SAVE & ANALYZE
# ==========================================
master_df.to_csv(MASTER_PATH, index=False)

total_rows = len(master_df)
print("\n==================================================")
print(f" 📊 FINAL MASTER DISTRIBUTION REPORT ({total_rows:,} Rows)")
print("==================================================")

# Safely check for columns in case LLM missed any
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
print(f"✅ Master dataset successfully saved and updated at: {MASTER_PATH}")
