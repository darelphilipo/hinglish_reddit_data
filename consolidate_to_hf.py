import os
import time
import uuid
import random
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Features, Value
from huggingface_hub import login, HfApi, CommitOperationDelete

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
    """Lists all parquet files currently stored in the repository."""
    try:
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        return [f for f in all_files if f.endswith(".parquet")]
    except Exception as e:
        print(f"[!] Could not list repo files: {e}", flush=True)
        return []

def load_parquet_group_safely(repo_id, parquet_files, group_label):
    """Loads a specific set of parquet files directly using data_files URLs."""
    if not parquet_files:
        return None
    try:
        urls = [f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}" for f in parquet_files]
        ds = load_dataset("parquet", data_files={"train": urls}, split="train")
        ds = ds.cast(SCHEMA)
        print(f"  Loaded '{group_label}': {len(ds)} rows from {len(parquet_files)} file(s)", flush=True)
        return ds
    except Exception as e:
        print(f"  [!] Failed to load '{group_label}': {e}. Skipping.", flush=True)
        return None

def deduplicate_dataset_low_mem(dataset):
    """Deduplicates a dataset internally using unique IDs."""
    print("Extracting IDs for memory-efficient deduplication...", flush=True)
    id_series = pd.Series(dataset["id"])
    unique_indices = id_series.drop_duplicates().index.values
    print(f"  Rows before dedupe: {len(dataset)} | Rows after dedupe: {len(unique_indices)}", flush=True)
    return dataset.select(unique_indices)

def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set!")
    login(token=HF_TOKEN)
    api = HfApi()

    print(f"Fetching repository file layout for '{HF_DATASET_REPO}'...", flush=True)
    all_parquet_files = get_all_repo_parquet_files(api, HF_DATASET_REPO)
    
    # Isolate ONLY the temporary batch files. We will NOT load the main train split.
    tmp_files = [f for f in all_parquet_files if "tmp_batch_" in f]

    if not tmp_files:
        print("No temporary batch files found. Nothing to consolidate. Exiting.", flush=True)
        return

    # 1. Load the scratch batch files
    print(f"Found {len(tmp_files)} temporary batch file(s). Loading...", flush=True)
    ds_tmp = load_parquet_group_safely(HF_DATASET_REPO, tmp_files, "tmp_batches")
    
    if not ds_tmp:
        print("Failed to load temporary files. Exiting.", flush=True)
        return

    # 2. Deduplicate internally within the new batches
    final_new_dataset = deduplicate_dataset_low_mem(ds_tmp)

    # 3. Export to a local parquet file
    # We generate a unique hash so it doesn't overwrite existing HF data chunks
    chunk_hash = str(uuid.uuid4())[:8]
    local_filename = f"train_append_{chunk_hash}.parquet"
    
    print(f"\nSaving {len(final_new_dataset)} rows to local file '{local_filename}'...", flush=True)
    final_new_dataset.to_parquet(local_filename)

    # 4. Upload directly to Hugging Face via the API (Incremental Append)
    path_in_repo = f"data/{local_filename}"
    max_push_attempts = 5
    push_succeeded = False
    
    for attempt in range(1, max_push_attempts + 1):
        try:
            print(f"\nUploading new chunk directly to '{path_in_repo}' (attempt {attempt}/{max_push_attempts})...", flush=True)
            api.upload_file(
                path_or_fileobj=local_filename,
                path_in_repo=path_in_repo,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset"
            )
            print("✅ Incremental append successful! Hugging Face will automatically merge this into 'train'.", flush=True)
            push_succeeded = True
            break
        except Exception as e:
            wait = random.uniform(5, 12) * attempt
            print(f"  [!] Upload failed ({e}). Retrying in {wait:.1f}s...", flush=True)
            time.sleep(wait)

    # 5. Clean up local files and remote scratch files
    if os.path.exists(local_filename):
        os.remove(local_filename)

    if push_succeeded and tmp_files:
        print(f"\nCleaning up {len(tmp_files)} temporary scratch file(s) from Hugging Face...", flush=True)
        try:
            ops = [CommitOperationDelete(path_in_repo=f) for f in tmp_files]
            api.create_commit(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                operations=ops,
                commit_message="Cleanup: remove consolidated tmp_batch scratch files",
            )
            print("✅ Cleanup commit successful.", flush=True)
        except Exception as e:
            print(f"  [!] Cleanup commit failed: {e}. Leftover files can be cleaned on the next run.", flush=True)

if __name__ == "__main__":
    main()
