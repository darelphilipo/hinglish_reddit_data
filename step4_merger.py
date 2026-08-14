import pandas as pd
import os
import sys
import glob
import time
import re
import json

print("🔗 Initializing Master Merger & Audit Engine...")
script_start = time.time()

# ==========================================
# 1. CONFIGURATION & TARGETS
# ==========================================
BASE_OUTPUT_DIR = './labelled_output/'
CHUNKS_DIR = os.path.join(BASE_OUTPUT_DIR, 'chunks/')
MASTER_PATH = os.path.join(BASE_OUTPUT_DIR, 'master_baseline_tier1.csv')
TARGETS_PATH = os.path.join(BASE_OUTPUT_DIR, 'pipeline_targets.json')

if not os.path.exists(MASTER_PATH) or not os.path.exists(TARGETS_PATH):
    print("❌ CRITICAL ERROR: Master file or Blueprint JSON not found. Run Step 1 & 2 first.")
    sys.exit(1)

with open(TARGETS_PATH, 'r') as f:
    blueprint = json.load(f)

TARGETS = blueprint.get("categories", {})
CLEAN_GOAL = blueprint.get("clean_data", 0)
GLOBAL_GOAL = blueprint.get("global_goal", 0)

# ==========================================
# 2. LOAD MASTER & CALCULATE INITIAL STATE
# ==========================================
master_df = pd.read_csv(MASTER_PATH)
initial_master_len = len(master_df)

# Safely check for columns
core_toxic_cols = ['profanity_vulgarity', 'targeted_abuse_harassment', 'discriminatory_hate_speech']
for col in core_toxic_cols:
    if col not in master_df.columns:
        master_df[col] = 0

initial_toxic_mask = master_df[core_toxic_cols].max(axis=1) == 1
initial_clean_count = initial_master_len - initial_toxic_mask.sum()
initial_counts = {cat: master_df.get(cat, pd.Series([0])).sum() for cat in TARGETS}

print(f"   ↳ Loaded Master Dataset: {initial_master_len:,} rows.")

# ==========================================
# 3. DISCOVER & INGEST CHUNKS
# ==========================================
chunk_files = glob.glob(os.path.join(CHUNKS_DIR, 'harvested_tier1_*.csv'))

if not chunk_files:
    print("   ↳ No new harvested chunks found to merge.")
    chunks_df = pd.DataFrame()
else:
    print(f"   ↳ Found {len(chunk_files)} pending chunk(s). Ingesting...")
    chunk_list = [pd.read_csv(f) for f in chunk_files]
    chunks_df = pd.concat(chunk_list, ignore_index=True)
    print(f"   ↳ Raw rows in chunks: {len(chunks_df):,}")

# ==========================================
# 4. STRICT DEDUPLICATION & MERGE
# ==========================================
if not chunks_df.empty:
    # Append to master
    merged_df = pd.concat([master_df, chunks_df], ignore_index=True)
    
    # DEDUP LOCK 1: Exact Reddit ID match
    pre_id_dedup = len(merged_df)
    merged_df.drop_duplicates(subset=['id'], keep='first', inplace=True)
    id_dupes_dropped = pre_id_dedup - len(merged_df)

    # DEDUP LOCK 2: Semantic text match (strips punctuation/spaces to catch copy-pastes)
    pre_text_dedup = len(merged_df)
    merged_df['dedup_hash'] = merged_df['body'].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    merged_df.drop_duplicates(subset=['dedup_hash'], keep='first', inplace=True)
    merged_df.drop(columns=['dedup_hash'], inplace=True)
    text_dupes_dropped = pre_text_dedup - len(merged_df)

    total_dupes = id_dupes_dropped + text_dupes_dropped
    net_new_rows = len(merged_df) - initial_master_len

    print(f"\n🛡️ Deduplication Engine Executed:")
    print(f"   - Blocked {id_dupes_dropped:,} exact ID overlaps.")
    print(f"   - Blocked {text_dupes_dropped:,} semantic copy-pastes.")
    print(f"   - Net New Rows Added: {net_new_rows:,}")

    # Overwrite Master
    merged_df.to_csv(MASTER_PATH, index=False)
    
    # Cleanup: Delete processed chunks to keep repo clean
    for f in chunk_files:
        os.remove(f)
    print("   ↳ Cleaned up processed chunk files.")
else:
    merged_df = master_df
    total_dupes = 0
    net_new_rows = 0

# ==========================================
# 5. TELEMETRY & FINAL AUDIT
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
    growth = int(current - initial_counts.get(cat, 0))
    growth_str = f"(+{growth})" if growth > 0 else ""
    
    if current >= goal:
        status = "✅ MET"
    else:
        status = f"❌ SHORT ({goal - current} needed)"
        toxic_targets_met = False

    print(f"{cat.upper():<25} | {current:<8,} | {goal:<8,} | {status} {growth_str}")

print("-" * 65)

# 2. Audit Clean Data
for col in core_toxic_cols:
    if col not in merged_df.columns:
        merged_df[col] = 0
final_toxic_mask = merged_df[core_toxic_cols].max(axis=1) == 1
current_clean = len(merged_df) - final_toxic_mask.sum()

clean_growth = int(current_clean - initial_clean_count)
clean_growth_str = f"(+{clean_growth})" if clean_growth > 0 else ""

clean_target_met = current_clean >= CLEAN_GOAL
clean_status = "✅ MET" if clean_target_met else f"❌ SHORT ({CLEAN_GOAL - current_clean} needed)"

print(f"{'CLEAN_BACKGROUND_DATA':<25} | {current_clean:<8,} | {CLEAN_GOAL:<8,} | {clean_status} {clean_growth_str}")

print("==================================================")
print(f"Total Master Rows        : {len(merged_df):,}")
print(f"Total Duplicates Blocked : {total_dupes:,}")
print(f"Merger Execution Time    : {time.time() - script_start:.2f}s")
print("==================================================")

# ==========================================
# 6. PIPELINE LOOP TRIGGER
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
