import pandas as pd
import glob
import os

print("🔄 Initializing Master Merge & Analysis...")

CHUNKS_DIR = './labelled_output/chunks/'
MASTER_PATH = './labelled_output/master_baseline_tier1.csv'

# 1. Locate all CSVs ONLY in the chunks directory
all_files = glob.glob(os.path.join(CHUNKS_DIR, '*.csv'))

if not all_files:
    print("❌ No CSV chunk files found in ./labelled_output/chunks/. Exiting.")
    exit(1)

print(f"📦 Found {len(all_files)} dataset chunks. Merging...")

# 2. Merge all chunks
df_list = [pd.read_csv(file) for file in all_files]
master_df = pd.concat(df_list, ignore_index=True)

# 🐛 FIX: Expand short JSON keys back to full column names
rename_mapping = {
    "pv": "profanity_vulgarity",
    "tah": "targeted_abuse_harassment",
    "dhs": "discriminatory_hate_speech",
    "cst": "caste",
    "cr": "communal_religious",
    "rx": "regional_xenophobic",
    "mg": "misogyny_gender"
}
master_df.rename(columns=rename_mapping, inplace=True)

# 3. Final Cross-Run Deduplication Safety Check
initial_len = len(master_df)
master_df.drop_duplicates(subset=['body'], keep='first', inplace=True)
dedup_count = initial_len - len(master_df)
print(f"✂️ Dropped {dedup_count} cross-run duplicate comments.")

# 4. Save Master File
master_df.to_csv(MASTER_PATH, index=False)

# 5. Print Distribution Stats
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

print(f"🟢 Clean Comments  : {total_clean:,} ({(total_clean/total_rows)*100:.1f}%)")
print(f"🔴 Toxic Comments  : {total_toxic:,} ({(total_toxic/total_rows)*100:.1f}%)\n")

categories = [
    ('caste', 1000), ('communal_religious', 1200), 
    ('regional_xenophobic', 1000), ('misogyny_gender', 1000)
]

for col, target in categories:
    if col in master_df.columns:
        count = master_df[col].sum()
    else:
        count = 0
    shortfall = max(0, target - count)
    print(f"{col:<22}: {count:>5} / {target:>4} target | Shortfall: {shortfall}")

print("\n==================================================")
print(f"✅ Master dataset successfully saved and updated at: {MASTER_PATH}")
