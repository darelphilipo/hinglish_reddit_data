# ==========================================
# 4. DUAL-SCHEMA FORMATTING
# ==========================================
LABEL_COLUMNS = [
    'profanity_vulgarity', 'targeted_abuse_harassment', 'discriminatory_hate_speech', 
    'caste', 'communal_religious', 'regional_xenophobic', 'misogyny_gender'
]

print("\n🛠️ Formatting Dual-Schema (RoBERTa + Sarvam ChatML)...")
formatted_records = []

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
