import os
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import login

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
    instead of crashing the whole consolidation run."""
    try:
        ds = load_dataset(HF_DATASET_REPO, split=split_name)
        print(f"  Loaded split '{split_name}': {len(ds)} rows", flush=True)
        return ds
    except Exception as e:
        print(f"  [!] Could not load split '{split_name}': {e}. Skipping it.", flush=True)
        return None


def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set!")
    login(token=HF_TOKEN)

    pieces = []

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
            loaded_batches += 1

    print("Sweeping for any leftover legacy-named splits (pre-rewrite naming)...", flush=True)
    legacy_loaded = 0
    for legacy_name in LEGACY_BATCH_NAMES:
        split_name = f"tmp_batch_{legacy_name}"
        ds = load_split_safely(split_name)
        if ds is not None:
            pieces.append(ds)
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

    # 4. Push back as the single permanent split
    final_dataset = Dataset.from_pandas(df, preserve_index=False)
    print(f"\nPushing consolidated dataset to split '{FINAL_SPLIT}'...", flush=True)
    final_dataset.push_to_hub(repo_id=HF_DATASET_REPO, split=FINAL_SPLIT, private=True)
    print("✅ Consolidation complete.", flush=True)


if __name__ == "__main__":
    main()
