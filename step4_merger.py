import pandas as pd
import os
import sys
import time
import json
from datasets import load_dataset

print("🔗 Initializing Master Audit & Loop Controller Engine...")
script_start = time.time()

HF_REPO_ID = "darelphilip/hinglish-toxicity"
TARGETS_PATH = './prompt/pipeline_targets.json'

if not os.path.exists(TARGETS_PATH):
    print("❌ CRITICAL ERROR: Blueprint JSON not found. Run Step 2 first.")
    sys.exit(1)

with open(TARGETS_PATH, 'r') as f:
    blueprint = json.load(f)

TARGETS = blueprint.get("categories", {})
CLEAN_GOAL = blueprint.get("clean_data", 0)
GLOBAL_GOAL = blueprint.get("global_goal", 0)

print(f"\n📥 Pulling live dataset from Hugging Face: {HF_REPO_ID}...")
try:
    ds = load_dataset(HF_REPO_ID, split="train")
    merged_df = ds.to_pandas()
    print(f"   ↳ Successfully loaded {len(merged_df):,} rows from Hugging Face.")
except Exception as e:
    print(f"❌ Failed to load dataset from HF: {e}")
    sys.exit(1)

initial_len = len(merged_df)
merged_df.drop_duplicates(subset=['id'], keep='first', inplace=True)
id_dupes_dropped = initial_len - len(merged_df)

pre_text_dedup = len(merged_df)
merged_df['dedup_hash'] = merged_df['text'].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
merged_df.drop_duplicates(subset=['dedup_hash'], keep='first', inplace=True)
merged_df.drop(columns=['dedup_hash'], inplace=True)
text_dupes_dropped = pre_text_dedup - len(merged_df)

total_dupes = id_dupes_dropped + text_dupes_dropped
if total_dupes > 0:
    print(f"\n🛡️ Deduplication Engine Executed (In-Memory Audit):")
    print(f"   - Blocked {id_dupes_dropped:,} exact ID overlaps.")
    print(f"   - Blocked {text_dupes_dropped:,} semantic copy-pastes.")
    print(f"   - Valid Unique Rows: {len(merged_df):,}")

# ==========================================
# 4. TELEMETRY & FINAL AUDIT REPORTS
# ==========================================

# --- A. CURRENT RUN DISTRIBUTION REPORT ---
print("\n==================================================")
print(" 📊 CURRENT RUN DISTRIBUTION REPORT (TOTAL YIELD)")
print("==================================================")
core_toxic_cols = ['profanity_vulgarity', 'targeted_abuse_harassment', 'discriminatory_hate_speech', 'caste', 'communal_religious', 'regional_xenophobic', 'misogyny_gender']
for col in core_toxic_cols:
    if col not in merged_df.columns: merged_df[col] = 0

total_current_rows = len(merged_df)
current_toxic_mask = merged_df[core_toxic_cols].max(axis=1) == 1
current_total_toxic = int(current_toxic_mask.sum())
current_total_clean = total_current_rows - current_total_toxic

print(f"Total Dataset Yield      : {total_current_rows:,}")
print(f"Total Toxic Records      : {current_total_toxic:,}")
print(f"Total Clean Records      : {current_total_clean:,}")
print("\nCategory Distribution Summary:")
category_keys = {
    'profanity_vulgarity': 'PROFANITY_VULGARITY',
    'targeted_abuse_harassment': 'TARGETED_ABUSE_HARASSMENT',
    'discriminatory_hate_speech': 'DISCRIMINATORY_HATE_SPEECH',
    'caste': 'CASTE',
    'communal_religious': 'COMMUNAL_RELIGIOUS',
    'regional_xenophobic': 'REGIONAL_XENOPHOBIC',
    'misogyny_gender': 'MISOGYNY_GENDER'
}
for col_key, label_name in category_keys.items():
    cnt = int(merged_df[col_key].sum())
    print(f" - {label_name:<30} : {cnt:,}")

print("\n--- 📝 CURRENT RUN SAMPLE EXAMPLES (5 per category) ---")
for col_key, label_name in category_keys.items():
    subset = merged_df[merged_df[col_key] == 1]
    sample_cnt = min(5, len(subset))
    print(f"\n[{label_name} - {sample_cnt} samples]")
    if sample_cnt == 0:
        print("  (No entries found)")
    else:
        for idx, row in subset.sample(n=sample_cnt, random_state=42).iterrows():
            print(f"  • Text: \"{str(row['text'])[:120]}...\"")
            print(f"    Analysis: {row.get('analysis', 'N/A')}")
print("==================================================")

# --- B. FINAL MASTER DISTRIBUTION REPORT ---
print("\n==================================================")
print(" 📊 FINAL MASTER DISTRIBUTION REPORT")
print("==================================================")

final_counts = {cat: merged_df.get(cat, pd.Series([0])).sum() for cat in TARGETS}
toxic_targets_met = True

print(f"{'METRIC':<25} | {'CURRENT':<8} | {'GOAL':<8} | {'STATUS'}")
print("-" * 65)

for cat, goal in TARGETS.items():
    current = int(final_counts[cat])
    if current >= goal:
        status = "✅ MET"
    else:
        status = f"❌ SHORT ({goal - current} needed)"
        toxic_targets_met = False
    print(f"{cat.upper():<25} | {current:<8,} | {goal:<8,} | {status}")

print("-" * 65)

current_clean = total_current_rows - current_toxic_mask.sum()
clean_target_met = current_clean >= CLEAN_GOAL
clean_status = "✅ MET" if clean_target_met else f"❌ SHORT ({CLEAN_GOAL - current_clean} needed)"

print(f"{'CLEAN_BACKGROUND_DATA':<25} | {current_clean:<8,} | {CLEAN_GOAL:<8,} | {clean_status}")
print("==================================================")
print(f"Total Unique Rows        : {total_current_rows:,}")
print(f"Auditor Execution Time   : {time.time() - script_start:.2f}s")
print("==================================================")

# ==========================================
# 5. PIPELINE LOOP TRIGGER
# ==========================================
if toxic_targets_met and clean_target_met:
    print("🎉 SUCCESS: All toxic and clean dataset goals have been met!")
    print("Exiting with Code 0. The autonomous pipeline will now hibernate.")
    sys.exit(0)
elif not toxic_targets_met:
    print("🔄 DEFICIT DETECTED (TOXIC): Returning Exit Code 2 to trigger the Harvester loop.")
    sys.exit(2)
else:
    print("🔄 DEFICIT DETECTED (CLEAN): Toxic goals met, but clean buffer is short.")
    print("Returning Exit Code 3 to trigger Step 1 buffer scraper.")
    sys.exit(3)
