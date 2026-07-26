import os
import time
import random
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
from huggingface_hub import login, HfApi
from huggingface_hub.hf_api import CommitOperationDelete
from huggingface_hub.errors import HfHubHTTPError

# Must exactly match the schema in fetch_arctic_to_hf.py. Applied when
# loading each split (in case an existing split on the hub already has a
# null-vs-string type mismatch baked in from before this fix) and again on
# the final push, so 'train' never drifts back into an inconsistent schema.
SCHEMA = Features({
    "id": Value("string"),
    "body": Value("string"),
    "created_utc": Value("int64"),
    "subreddit": Value("string"),
    "score": Value("int64"),
    "controversiality": Value("int64"),
    "collapsed_reason_code": Value("string"),
})

# ==========================================
# CONFIGURATION -- keep in sync with fetch_arctic_to_hf.py
# ==========================================
HF_DATASET_REPO = "darelphilip/reddit_indian_subs"  # CHANGE THIS if needed
HF_TOKEN = os.getenv("HF_TOKEN")

FINAL_SPLIT = "train"

# Must match the batch names defined in fetch_arctic_to_hf.py / the workflow matrix
BATCH_NAMES = [
    "heavy_1", "heavy_2",
    "medium_1", "medium_2", "medium_3", "medium_4", "medium_5", "medium_6", "medium_7",
    "tiny_1",
]

# One-time inclusion of the OLD index-range naming scheme, so leftover splits
# from before the batching rewrite (e.g. tmp_batch_020_040) get swept into
# 'train' rather than sitting there orphaned forever. Safe to leave in
# permanently -- load_split_safely just skips anything that doesn't exist.
LEGACY_BATCH_NAMES = [
    "000_020", "020_040", "040_060", "060_080", "080_100",
    "100_120", "120_140", "140_160", "160_163",
]


def load_split_safely(split_name):
    """Loads a split if it exists; returns None (with a warning) if it doesn't,
    instead of crashing the whole consolidation run. Casts to the pinned
    SCHEMA on load, so a split saved before the schema fix (or one that
    happened to be all-null in some column) gets normalized here rather
    than blowing up concatenate_datasets with a features mismatch."""
    try:
        ds = load_dataset(HF_DATASET_REPO, split=split_name)
        ds = ds.cast(SCHEMA)
        print(f"  Loaded split '{split_name}': {len(ds)} rows", flush=True)
        return ds
    except Exception as e:
        print(f"  [!] Could not load split '{split_name}': {e}. Skipping it.", flush=True)
        return None


def cleanup_consumed_splits(repo_id, split_names):
    """Deletes the repo files backing each split in split_names. Only ever
    called AFTER a confirmed-successful push of the merged data to 'train' --
    this is what makes it safe: every row from these splits is already
    folded into 'train' by the time anything gets deleted, so this is
    tidying up consumed scratch data, not risking loss of unconsolidated data."""
    if not split_names:
        print("No consumed splits to clean up.", flush=True)
        return

    api = HfApi()
    try:
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        print(f"  [!] Could not list repo files for cleanup: {e}. Skipping cleanup this run "
              f"(leftover files just mean next run's cleanup will catch them instead).", flush=True)
        return

    to_delete = [f for f in all_files if any(f"/{name}-" in f or f == name for name in split_names)]
    # Fallback broader match in case of a different file layout than expected
    if not to_delete:
        to_delete = [f for f in all_files if any(name in f for name in split_names)]

    if not to_delete:
        print(f"No repo files matched the {len(split_names)} consumed split(s) -- nothing to delete.", flush=True)
        return

    print(f"Cleaning up {len(to_delete)} file(s) backing {len(split_names)} consumed split(s): "
          f"{to_delete}", flush=True)

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            ops = [CommitOperationDelete(path_in_repo=f) for f in to_delete]
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=ops,
                commit_message="Cleanup: remove consolidated tmp_batch scratch files",
            )
            print("Cleanup commit complete.", flush=True)
            return
        except Exception as e:
            is_conflict = "412" in str(e) or "Precondition Failed" in str(e)
            if attempt < max_attempts:
                wait = random.uniform(3, 10) * attempt
                reason = "branch conflict" if is_conflict else f"transient error ({type(e).__name__}: {e})"
                print(f"Cleanup commit failed -- {reason}. Retrying in {wait:.1f}s...", flush=True)
                time.sleep(wait)
                continue
            print(f"  [!] Cleanup commit failed after {attempt} attempt(s): {e}. "
                  f"Leftover files will just get picked up by the next run's cleanup.", flush=True)
            return


def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set!")
    login(token=HF_TOKEN)

    pieces = []
    loaded_split_names = []  # every tmp_batch split we actually folded in, for cleanup later

    # 1. Load whatever is already permanently saved (empty on the very first run)
    print(f"Loading existing '{FINAL_SPLIT}' split (if it exists)...", flush=True)
    existing = load_split_safely(FINAL_SPLIT)
    if existing is not None:
        pieces.append(existing)

    # 2. Load every batch's scratch split from this run
    print("Loading this run's batch scratch splits...", flush=True)
    loaded_batches = 0
    for batch_name in BATCH_NAMES:
        split_name = f"tmp_batch_{batch_name}"
        ds = load_split_safely(split_name)
        if ds is not None:
            pieces.append(ds)
            loaded_split_names.append(split_name)
            loaded_batches += 1

    print("Sweeping for any leftover legacy-named splits (pre-rewrite naming)...", flush=True)
    legacy_loaded = 0
    for legacy_name in LEGACY_BATCH_NAMES:
        split_name = f"tmp_batch_{legacy_name}"
        ds = load_split_safely(split_name)
        if ds is not None:
            pieces.append(ds)
            loaded_split_names.append(split_name)
            legacy_loaded += 1
    if legacy_loaded:
        print(f"  Found and folded in {legacy_loaded} legacy-named leftover split(s).", flush=True)

    print(f"\nLoaded {loaded_batches}/{len(BATCH_NAMES)} current batch splits this run "
          f"(+ {legacy_loaded} legacy leftovers).", flush=True)
    if loaded_batches < len(BATCH_NAMES):
        print("  [!] Some batches are missing -- check the scrape_and_push job logs above for a "
              "failed/timed-out/cancelled batch. Proceeding with whatever data is available, "
              "since partial data beats no data.", flush=True)

    if not pieces:
        print("Nothing to consolidate (no existing train split and no batch splits found). Exiting.", flush=True)
        return

    # 3. Concatenate everything and dedupe on comment id (handles reruns safely)
    combined = concatenate_datasets(pieces)
    df = combined.to_pandas().drop_duplicates(subset=["id"]).reset_index(drop=True)

    print(f"\nCombined total before dedupe: {sum(len(p) for p in pieces)} rows", flush=True)
    print(f"Combined total after dedupe:  {len(df)} rows", flush=True)

    # 4. Push back as the single permanent split (retry on transient branch conflicts)
    final_dataset = Dataset.from_pandas(df, features=SCHEMA, preserve_index=False)
    max_push_attempts = 5
    push_succeeded = False
    for attempt in range(1, max_push_attempts + 1):
        try:
            print(f"\nPushing consolidated dataset to split '{FINAL_SPLIT}' "
                  f"(attempt {attempt}/{max_push_attempts})...", flush=True)
            final_dataset.push_to_hub(repo_id=HF_DATASET_REPO, split=FINAL_SPLIT, private=True)
            print("✅ Consolidation complete.", flush=True)
            push_succeeded = True
            break
        except Exception as e:
            # Broad catch: HfHubHTTPError (412 conflicts) plus lower-level
            # transport errors (httpx.RemoteProtocolError etc.) that aren't
            # HfHubHTTPError subclasses and would otherwise crash uncaught.
            is_conflict = "412" in str(e) or "Precondition Failed" in str(e)
            if attempt < max_push_attempts:
                wait = random.uniform(3, 10) * attempt
                reason = "branch conflict (412)" if is_conflict else f"transient error ({type(e).__name__}: {e})"
                print(f"Push failed -- {reason}. Retrying in {wait:.1f}s...", flush=True)
                time.sleep(wait)
                continue
            print(f"Push failed after {attempt} attempt(s): {e}", flush=True)
            raise

    # Only clean up scratch files once the merged data is confirmed safely
    # sitting in 'train' -- never delete before a confirmed successful push.
    if push_succeeded:
        print(f"\nCleaning up {len(loaded_split_names)} consumed scratch split(s) now that "
              f"the merge is confirmed in '{FINAL_SPLIT}'...", flush=True)
        cleanup_consumed_splits(HF_DATASET_REPO, loaded_split_names)


if __name__ == "__main__":
    main()
