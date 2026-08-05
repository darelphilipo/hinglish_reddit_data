import os
import sys
import time
import uuid
import random
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Features, Value
from huggingface_hub import login, HfApi, CommitOperationDelete

# ==========================================
# DIAGNOSTIC HELPER
# ==========================================
def log_step(step_description):
    """Reads Linux system memory to track exactly when RAM runs out."""
    avail_mb = -1.0
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemAvailable' in line:
                    avail_mb = int(line.split()[1]) / 1024.0
                    break
    except Exception:
        pass
    
    mem_str = f"{avail_mb:,.1f} MB Free" if avail_mb > 0 else "Unknown"
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] [SYS RAM: {mem_str}] => {step_description}", flush=True)

# ==========================================
# SCHEMA DEFINITION
# ==========================================
SCHEMA = Features({
    "id": Value("string"),
    "body": Value("string"),
    "created_utc": Value("int64"),
    "subreddit": Value("string"),
    "score": Value("int64"),
    "controversiality": Value("int64"),
    "collapsed_reason_code": Value("string"),
})

HF_DATASET_REPO = "darelphilip/reddit_indian_subs"
HF_TOKEN = os.getenv("HF_TOKEN")

def get_all_repo_parquet_files(api, repo_id):
    try:
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        return [f for f in all_files if f.endswith(".parquet")]
    except Exception as e:
        log_step(f"[ERROR] Could not list repo files: {e}")
        return []

def load_parquet_group_safely(repo_id, parquet_files, group_label):
    if not parquet_files:
        return None
    try:
        log_step(f"Generating HuggingFace URLs for {len(parquet_files)} files...")
        urls = [f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}" for f in parquet_files]
        
        log_step(f"Downloading and loading {group_label} via load_dataset()...")
        ds = load_dataset("parquet", data_files={"train": urls}, split="train")
        
        log_step(f"Casting schema for {group_label}...")
        ds = ds.cast(SCHEMA)
        
        log_step(f"Successfully loaded '{group_label}': {len(ds)} rows")
        return ds
    except Exception as e:
        log_step(f"[ERROR] Failed to load '{group_label}': {e}")
        return None

def deduplicate_dataset_low_mem(dataset):
    log_step("Starting memory-efficient deduplication. Extracting IDs to Pandas Series...")
    id_series = pd.Series(dataset["id"])
    
    log_step("Calculating unique indices (dropping duplicates)...")
    unique_indices = id_series.drop_duplicates().index.values
    
    log_step(f"Deduplication math complete. Before: {len(dataset)} | After: {len(unique_indices)}")
    
    log_step("Applying native Arrow selection (slicing dataset)...")
    return dataset.select(unique_indices)

def main():
    print("\n" + "="*70)
    print("🚀 SCRIPT VERSION: v5.0 (INCREMENTAL APPEND + NATIVE MEMORY DIAGNOSTICS)")
    print("="*70 + "\n", flush=True)

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set!")
    
    log_step("Logging into Hugging Face...")
    login(token=HF_TOKEN)
    api = HfApi()

    log_step(f"Fetching repository file layout for '{HF_DATASET_REPO}'...")
    all_parquet_files = get_all_repo_parquet_files(api, HF_DATASET_REPO)
    
    # Isolate ONLY the temporary batch files. Ignore the massive 12.5M train split.
    tmp_files = [f for f in all_parquet_files if "tmp_batch_" in f]

    if not tmp_files:
        log_step("No temporary batch files found. Exiting gracefully.")
        sys.exit(0)

    log_step(f"Found {len(tmp_files)} temporary batch file(s) to process.")
    
    # 1. Load the scratch batch files
    ds_tmp = load_parquet_group_safely(HF_DATASET_REPO, tmp_files, "tmp_batches")
    
    if not ds_tmp:
        log_step("Failed to load temporary files. Exiting.")
        sys.exit(1)

    # 2. Deduplicate internally within the new batches
    final_new_dataset = deduplicate_dataset_low_mem(ds_tmp)

    # 3. Export to a local parquet file
    chunk_hash = str(uuid.uuid4())[:8]
    local_filename = f"train_append_{chunk_hash}.parquet"
    
    log_step(f"Saving {len(final_new_dataset)} rows to local disk as '{local_filename}'...")
    final_new_dataset.to_parquet(local_filename)
    log_step(f"File saved successfully. File size: {os.path.getsize(local_filename) / (1024*1024):.2f} MB")

    # 4. Upload directly to Hugging Face
    path_in_repo = f"data/{local_filename}"
    max_push_attempts = 5
    push_succeeded = False
    
    for attempt in range(1, max_push_attempts + 1):
        try:
            log_step(f"Uploading new chunk to '{path_in_repo}' (Attempt {attempt}/{max_push_attempts})...")
            api.upload_file(
                path_or_fileobj=local_filename,
                path_in_repo=path_in_repo,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset"
            )
            log_step("✅ Upload successful! HF will auto-merge this chunk.")
            push_succeeded = True
            break
        except Exception as e:
            wait = random.uniform(5, 12) * attempt
            log_step(f"[WARNING] Upload failed: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    # 5. Clean up local and remote files
    log_step("Deleting local temporary parquet file...")
    if os.path.exists(local_filename):
        os.remove(local_filename)

    if push_succeeded and tmp_files:
        log_step(f"Cleaning up {len(tmp_files)} temporary scratch file(s) from Hugging Face...")
        try:
            ops = [CommitOperationDelete(path_in_repo=f) for f in tmp_files]
            api.create_commit(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                operations=ops,
                commit_message=f"Cleanup: remove consolidated tmp files (Appended chunk {chunk_hash})"
            )
            log_step("✅ Remote cleanup commit successful.")
        except Exception as e:
            log_step(f"[WARNING] Remote cleanup failed: {e}. Will be caught on next run.")

    log_step("🎉 Script completed successfully!")

if __name__ == "__main__":
    main()
