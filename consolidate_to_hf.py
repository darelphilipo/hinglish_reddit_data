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

# Must match the batch ranges defined in the workflow matrix
BATCH_RANGES = [
    (0, 20), (20, 40), (40, 60), (60, 80), (80, 100),
    (100, 120), (120, 140), (140, 160), (160, 163),
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
    for start, end in BATCH_RANGES:
        split_name = f"tmp_batch_{start:03d}_{end:03d}"
        ds = load_split_safely(split_name)
        if ds is not None:
            pieces.append(ds)
            loaded_batches += 1

    print(f"\nLoaded {loaded_batches}/{len(BATCH_RANGES)} batch splits this run.", flush=True)
    if loaded_batches < len(BATCH_RANGES):
        print("  [!] WARNING: fewer batches than expected -- check the scrape_and_push "
              "job logs above for a failed/timed-out batch before trusting this merge.", flush=True)

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
