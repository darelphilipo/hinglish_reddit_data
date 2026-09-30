import os
import sys
import time
import uuid
import random
import pandas as pd
import duckdb
from datasets import load_dataset, Features, Value
from huggingface_hub import HfApi, CommitOperationDelete

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
TMP_MARKER = "tmp_batch_"

def get_all_repo_parquet_files(api, repo_id):
    # FIX: let listing errors propagate. Previously an API failure returned []
    # and the script reported "No temporary batch files found" and exited 0,
    # silently hiding the problem.
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    return [f for f in all_files if f.endswith(".parquet")]

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

def deduplicate_against_global(dataset, existing_files):
    """
    Drop rows whose ID already exists in the MASTER files.

    FIX: the old query used the glob 'data/**/*.parquet', which also matched the
    tmp_batch_* files being consolidated. Every incoming ID was therefore
    "already existing" and 100% of rows were filtered out. We now read only the
    explicit list of non-tmp files.
    """
    log_step("Authenticating DuckDB for global ID check...")
    duckdb.sql(f"CREATE OR REPLACE SECRET hf_secret (TYPE huggingface, TOKEN '{HF_TOKEN}');")

    paths = [f"hf://datasets/{HF_DATASET_REPO}/{f}" for f in existing_files]
    path_list_sql = ", ".join("'" + p.replace("'", "''") + "'" for p in paths)

    existing_ids_set = set()
    if paths:
        log_step(f"Fetching historical IDs from {len(paths)} master file(s) over the network...")
        existing_ids_df = duckdb.sql(f"SELECT id FROM read_parquet([{path_list_sql}])").df()
        existing_ids_set = set(existing_ids_df["id"])
        del existing_ids_df
    else:
        log_step("[WARNING] No master files found; skipping global dedup.")
    log_step(f"Loaded {len(existing_ids_set):,} existing IDs into memory.")

    log_step("Dropping internal duplicates from the new batch...")
    id_series = pd.Series(dataset["id"])
    unique_indices = id_series.drop_duplicates().index.values
    internal_deduped = dataset.select(unique_indices)
    log_step(f"After internal dedup: {len(internal_deduped):,} rows (from {len(dataset):,}).")

    log_step("Filtering out global duplicates already on Hugging Face...")
    final_dataset = internal_deduped.filter(lambda x: x["id"] not in existing_ids_set)

    log_step(f"Global dedup complete. New unique rows to append: {len(final_dataset):,} "
             f"(Down from {len(dataset):,})")
    return final_dataset

def main():
    print("\n" + "=" * 70)
    print("🚀 SCRIPT VERSION: v6.1 (GLOBAL DEDUP FIX + SAFETY GUARDS)")
    print("=" * 70 + "\n", flush=True)

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set!")

    api = HfApi(token=HF_TOKEN)

    log_step(f"Fetching repository file layout for '{HF_DATASET_REPO}'...")
    all_parquet_files = get_all_repo_parquet_files(api, HF_DATASET_REPO)

    # Temp scratch files vs. master files (data_*.parquet + train_append_*.parquet)
    tmp_files = [f for f in all_parquet_files if TMP_MARKER in f]
    existing_files = [f for f in all_parquet_files if TMP_MARKER not in f]

    log_step(f"Repo has {len(existing_files)} master file(s) and {len(tmp_files)} temp file(s).")

    if not tmp_files:
        log_step("No temporary batch files found. Exiting gracefully.")
        sys.exit(0)

    log_step(f"Found {len(tmp_files)} temporary batch file(s) to process.")

    # 1. Load the scratch batch files
    ds_tmp = load_parquet_group_safely(HF_DATASET_REPO, tmp_files, "tmp_batches")

    if ds_tmp is None:
        log_step("Failed to load temporary files. Exiting (temp files kept).")
        sys.exit(1)

    if len(ds_tmp) == 0:
        log_step("[!] Temp files loaded but contain 0 rows. Exiting (temp files kept).")
        sys.exit(1)

    # 2. Deduplicate against the global dataset using DuckDB
    final_new_dataset = deduplicate_against_global(ds_tmp, existing_files)

    # GUARD: never upload an empty chunk and never delete temp files on a zero
    # result. A genuine all-duplicates batch is possible, but a false zero
    # (like the previous bug) destroys data, so fail loudly instead.
    if len(final_new_dataset) == 0:
        log_step(f"[!] 0 new rows out of {len(ds_tmp):,}. Skipping upload and KEEPING temp files "
                 f"so nothing is lost. Investigate before deleting.")
        sys.exit(1)

    # 3. Export to a local parquet file
    chunk_hash = str(uuid.uuid4())[:8]
    local_filename = f"train_append_{chunk_hash}.parquet"

    log_step(f"Saving {len(final_new_dataset):,} rows to local disk as '{local_filename}'...")
    final_new_dataset.to_parquet(local_filename)
    size_mb = os.path.getsize(local_filename) / (1024 * 1024)
    log_step(f"File saved successfully. File size: {size_mb:.2f} MB")

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

    # 5. Clean up local file
    log_step("Deleting local temporary parquet file...")
    if os.path.exists(local_filename):
        os.remove(local_filename)

    if not push_succeeded:
        log_step("[ERROR] Upload failed after all attempts. Temp files kept. Exiting with error.")
        sys.exit(1)

    # 6. Remote cleanup (only reached after a verified, non-empty upload)
    try:
        uploaded_files = api.list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset")
        if path_in_repo not in uploaded_files:
            log_step("[ERROR] Uploaded chunk not visible in repo. Keeping temp files.")
            sys.exit(1)

        log_step(f"Cleaning up {len(tmp_files)} temporary scratch file(s) from Hugging Face...")
        ops = [CommitOperationDelete(path_in_repo=f) for f in tmp_files]
        api.create_commit(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Cleanup: remove consolidated tmp files (Appended chunk {chunk_hash}, "
                           f"{len(final_new_dataset)} rows)"
        )
        log_step("✅ Remote cleanup commit successful.")
    except SystemExit:
        raise
    except Exception as e:
        log_step(f"[WARNING] Remote cleanup failed: {e}. Will be caught on next run.")

    log_step("🎉 Script completed successfully!")

if __name__ == "__main__":
    main()
