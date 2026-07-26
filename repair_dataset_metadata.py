"""
ONE-OFF REPAIR SCRIPT -- run this once, locally, to fix the current broken
repo state. It does NOT touch any actual data files -- it only edits the
dataset card's YAML metadata (README.md front matter) to remove references
to splits whose underlying parquet files no longer exist.

Why this is needed: a previous cleanup step deleted the parquet files for
these stale splits, but never removed their entries from the dataset card's
`configs`/`dataset_info` YAML. Since `datasets` validates the file format
across every listed split in one pass before it will load ANY split, these
7 dangling entries were breaking load_dataset() for the entire repo --
including 'train' itself.

Usage:
    HF_TOKEN=hf_xxx python repair_dataset_metadata.py
"""
import os
from huggingface_hub import DatasetCard

HF_DATASET_REPO = "darelphilip/reddit_indian_subs"  # CHANGE THIS if needed
HF_TOKEN = os.getenv("HF_TOKEN")

# Exactly the splits the error message showed as (None, {}) -- i.e. listed
# in metadata but with no matching files on disk anymore.
STALE_SPLITS = [
    "tmp_batch_020_040",
    "tmp_batch_060_080",
    "tmp_batch_040_060",
    "tmp_batch_080_100",
    "tmp_batch_140_160",
    "tmp_batch_120_140",
    "tmp_batch_160_163",
]


def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set!")

    print(f"Loading dataset card for {HF_DATASET_REPO}...")
    card = DatasetCard.load(HF_DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
    card_dict = card.data.to_dict()

    modified = False

    # Remove stale entries from configs[].data_files
    configs = card_dict.get("configs", [])
    for config in configs:
        data_files = config.get("data_files", [])
        new_data_files = [df for df in data_files if df.get("split") not in STALE_SPLITS]
        if len(new_data_files) != len(data_files):
            removed = len(data_files) - len(new_data_files)
            print(f"  Removing {removed} stale entr(y/ies) from configs[{config.get('config_name')}].data_files")
            config["data_files"] = new_data_files
            modified = True

    # Remove stale entries from dataset_info.splits (handles both the
    # single-dict and multi-config-list shapes datasets can produce)
    dataset_info = card_dict.get("dataset_info")
    if isinstance(dataset_info, list):
        for di in dataset_info:
            splits = di.get("splits", [])
            new_splits = [s for s in splits if s.get("name") not in STALE_SPLITS]
            if len(new_splits) != len(splits):
                print(f"  Removing {len(splits) - len(new_splits)} stale split(s) from dataset_info.splits")
                di["splits"] = new_splits
                modified = True
    elif isinstance(dataset_info, dict):
        splits = dataset_info.get("splits", [])
        new_splits = [s for s in splits if s.get("name") not in STALE_SPLITS]
        if len(new_splits) != len(splits):
            print(f"  Removing {len(splits) - len(new_splits)} stale split(s) from dataset_info.splits")
            dataset_info["splits"] = new_splits
            modified = True

    if not modified:
        print("No stale split references found in metadata -- nothing to repair.")
        return

    from huggingface_hub import DatasetCardData
    card.data = DatasetCardData(**card_dict)

    print("Pushing repaired dataset card...")
    card.push_to_hub(HF_DATASET_REPO, repo_type="dataset",
                      commit_message="Repair: remove stale split metadata for already-deleted files")
    print("✅ Repair complete. load_dataset() should work again now.")


if __name__ == "__main__":
    main()
