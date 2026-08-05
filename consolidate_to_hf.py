import os
import time
import random
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
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
FINAL_SPLIT = "train"


def get_all_repo_parquet_files(api, repo_id):
    """Lists all parquet files currently stored in the repository."""
    try:
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        return [f for f in all_files if f.endswith(".parquet")]
    except Exception as e:
        print(f"[!] Could not list repo files: {e}", flush=True)
        return []


def load_parquet_group_safely(repo_id, parquet_files, group_label):
    """Loads a specific set of parquet files directly using data_files URLs.
    This bypasses HF split auto-detection crashes."""
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
    """Deduplicates a HuggingFace Dataset by 'id' using PyArrow indices.
    Only loads the 'id' column into memory (~10 MB), keeping the massive
    'body' text column safely in Arrow storage to prevent OOM kills."""
    print("Extracting IDs for memory-efficient deduplication...", flush=True)
    
    # Extract ONLY the ID column to pandas
    id_series = pd.Series(dataset["id"])
    
    # Find unique row indices
    unique_indices = id_series.drop_duplicates().index.values
    
    print(f"  Rows before dedupe: {len(dataset)} | Rows after dedupe: {len(unique_indices)}", flush=True)
    
    # Slice dataset natively in Arrow
    return dataset.select(unique_indices)


def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set!")
    login(token=HF_TOKEN)
    api = HfApi()

    print(f"Fetching repository file layout for '{HF_DATASET_REPO}'...", flush=True)
    all_parquet_files = get_all_repo_parquet_files(api, HF_DATASET_REPO)
    
    if not all_parquet_files:
        print("No Parquet files found in repository. Exiting.", flush=True)
        return

    pieces = []
    files_to_delete = []

    # 1. Isolate existing permanent train files vs tmp_batch files
    train_files = [f for f in all_parquet_files if "train" in f or "data/train-" in f]
    tmp_files = [f for f in all_parquet_files if "tmp_batch_" in f]

    # 2. Load existing train data
    if train_files:
        print("Loading existing 'train' dataset files...", flush=True)
        ds_train = load_parquet_group_safely(HF_DATASET_REPO, train_files, "train_existing")
        if ds_train:
            pieces.append(ds_train)

    # 3. Load scratch batch files
    if tmp_files:
        print(f"Found {len(tmp_files)} temporary batch file(s). Loading...", flush=True)
        ds_tmp = load_parquet_group_safely(HF_DATASET_REPO, tmp_files, "tmp_batches")
        if ds_tmp:
            pieces.append(ds_tmp)
            files_to_delete.extend(tmp_files)

    if not pieces:
        print("No valid datasets loaded. Exiting.", flush=True)
        return

    # 4. Concatenate and deduplicate efficiently
    print("\nConcatenating dataset chunks...", flush=True)
    combined = concatenate_datasets(pieces)
    final_dataset = deduplicate_dataset_low_mem(combined)

    # 5. Push consolidated dataset to 'train'
    max_push_attempts = 5
    push_succeeded = False
    for attempt in range(1, max_push_attempts + 1):
        try:
            print(f"\nPushing consolidated dataset to split '{FINAL_SPLIT}' "
                  f"(attempt {attempt}/{max_push_attempts})...", flush=True)
            
            final_dataset.push_to_hub(
                repo_id=HF_DATASET_REPO, 
                split=FINAL_SPLIT, 
                private=True,
                max_shard_size="500MB"  # Shards cleanly to avoid 5GB upload limits
            )
            print("✅ Consolidation push complete.", flush=True)
            push_succeeded = True
            break
        except Exception as e:
            wait = random.uniform(5, 12) * attempt
            print(f"  [!] Push failed ({e}). Retrying in {wait:.1f}s...", flush=True)
            time.sleep(wait)

    # 6. Delete consumed temporary batch files after successful push
    if push_succeeded and files_to_delete:
        print(f"\nCleaning up {len(files_to_delete)} temporary scratch file(s)...", flush=True)
        try:
            ops = [CommitOperationDelete(path_in_repo=f) for f in files_to_delete]
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
