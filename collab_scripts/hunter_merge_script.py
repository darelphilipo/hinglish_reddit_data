# @title 🧬 Merge & Shuffle Master Dataset (Fixed)
import pandas as pd
import os

# 1. Define Paths
OUTPUT_DIR = '/content/drive/MyDrive/Hinglish_Classifier_Project/experiments/'
BASELINE_PATH = os.path.join(OUTPUT_DIR, 'experiment_5k_2017_2020.csv')
HUNTER_PATH = os.path.join(OUTPUT_DIR, 'hunter_caste_misogyny_5k.csv')
MASTER_PATH = os.path.join(OUTPUT_DIR, 'master_training_dataset.csv')

print("🔄 Loading datasets...")
baseline_df = pd.read_csv(BASELINE_PATH)
hunter_df = pd.read_csv(HUNTER_PATH)

print(f"📊 Baseline Rows: {len(baseline_df)} | Hunter Rows: {len(hunter_df)}")

# 2. Concatenate
combined_df = pd.concat([baseline_df, hunter_df], ignore_index=True)
initial_len = len(combined_df)

# 3. Clean up the id_x / id_y collision if it exists
if 'id_x' in combined_df.columns:
    combined_df.rename(columns={'id_x': 'id'}, inplace=True)
if 'id_y' in combined_df.columns:
    combined_df.drop(columns=['id_y'], inplace=True)

# 4. Deduplicate (Using the actual comment text 'body' instead of ID)
# This is an ML best-practice to prevent copy-paste spam from overfitting the model!
combined_df.drop_duplicates(subset=['body'], keep='first', inplace=True)
duplicates_dropped = initial_len - len(combined_df)
print(f"✂️  Dropped {duplicates_dropped} overlapping/duplicate comments.")

# 5. Shuffle (Crucial for ML Training)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Save Master Dataset
combined_df.to_csv(MASTER_PATH, index=False)
print(f"✅ Master dataset merged, shuffled, and saved to:\n   {MASTER_PATH}")

# 7. Final Master Distribution
print("\n==================================================")
print(" 🏆 FINAL MASTER DATASET DISTRIBUTION")
print("==================================================")
total = len(combined_df)
combined_df['is_toxic'] = combined_df[['profanity_vulgarity', 'targeted_abuse_harassment', 'discriminatory_hate_speech']].max(axis=1)
toxic = combined_df['is_toxic'].sum()
clean = total - toxic

print(f"Total Rows: {total}")
print(f"🟢 Clean  : {clean} ({(clean/total)*100:.1f}%)")
print(f"🔴 Toxic  : {toxic} ({(toxic/total)*100:.1f}%)\n")

flags = [
    ('Caste (cst)', 'caste'),
    ('Communal (cr)', 'communal_religious'),
    ('Regional (rx)', 'regional_xenophobic'),
    ('Misogyny (mg)', 'misogyny_gender')
]
for name, col in flags:
    if col in combined_df.columns:
        cnt = combined_df[col].sum()
        print(f"{name:<20}: {cnt:>4} ({(cnt/total)*100:>5.1f}%)")
