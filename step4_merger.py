import pandas as pd
import os
import sys
import time
import json
from datasets import load_dataset

print("🔗 Initializing Master Audit & Loop Controller Engine...")
script_start = time.time()

# ==========================================
# 1. CONFIGURATION & TARGETS
# ==========================================
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

# ==========================================
# 2. PULL LIVE DATASET FROM HUGGING FACE
# ==========================================
print(f"\n📥 Pulling live dataset from Hugging Face: {HF_REPO_ID}...")
try:
    ds = load_dataset(HF_REPO_ID, split="train")
    merged_df = ds.to_pandas()
    print(f"   ↳ Successfully loaded {len(merged_df):,} rows from Hugging Face.")
except Exception as e:
    print(f"❌ Failed to load dataset from HF: {e}")
    sys.exit(1)

# ==========================================
# 3. IN-MEMORY DEDUPLICATION (For Accurate Audit)
# ==========================================
initial_len = len(merged_df)

# DEDUP LOCK 1: Exact Reddit ID match
merged_df.drop_duplicates(subset=['id'], keep='first', inplace=True)
id_dupes_dropped = initial_len - len(merged_df)

# DEDUP LOCK 2: Semantic text match
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
# 4. TELEMETRY & FINAL AUDIT
# ==========================================
print("\n==================================================")
print(" 📊 MASTER DATASET AUDIT & BALANCING REPORT")
print("==================================================")

final_counts = {cat: merged_df.get(cat, pd.Series([0])).sum() for cat in TARGETS}
toxic_targets_met = True

print(f"{'METRIC':<25} | {'CURRENT':<8} | {'GOAL':<8} | {'STATUS'}")
print("-" * 65)

# 1. Audit Toxic Categories
for cat, goal in TARGETS.items():
    current = int(final_counts[cat])
    
    if current >= goal:
        status = "✅ MET"
    else:
        status = f"❌ SHORT ({goal - current} needed)"
        toxic_targets_met = False

    print(f"{cat.upper():<25} | {current:<8,} | {goal:<8,} | {status}")

print("-" * 65)

# 2. Audit Clean Data
core_toxic_cols = ['profanity_vulgarity', 'targeted_abuse_harassment', 'discriminatory_hate_speech']
for col in core_toxic_cols:
    if col not in merged_df.columns:
        merged_df[col] = 0

final_toxic_mask = merged_df[core_toxic_cols].max(axis=1) == 1
current_clean = len(merged_df) - final_toxic_mask.sum()

clean_target_met = current_clean >= CLEAN_GOAL
clean_status = "✅ MET" if clean_target_met else f"❌ SHORT ({CLEAN_GOAL - current_clean} needed)"

print(f"{'CLEAN_BACKGROUND_DATA':<25} | {current_clean:<8,} | {CLEAN_GOAL:<8,} | {clean_status}")

print("==================================================")
print(f"Total Unique Rows        : {len(merged_df):,}")
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
