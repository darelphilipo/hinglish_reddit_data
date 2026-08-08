**Priority:** 3/5

# Proposal: Subreddit Range Fetch to Hugging Face

## Why

Add a standalone Python script and a separate, manually triggered GitHub
Actions workflow that fetch a date range of Reddit comments for one or more
subreddits via the Arctic Shift API and append the results directly to the
Hugging Face dataset repository as `data/train_append_<8-char-uuid>.parquet`
files.

This provides an operator-controlled backfill path: instead of a fixed monthly
window, an operator supplies a subreddit list and an inclusive start/end month
range, and the script pages through each subreddit-month in ascending order,
cleans the comments, and uploads each non-empty subreddit-month slice directly
to Hugging Face without needing the batch/checkpoint machinery in
`fetch_arctic_to_hf.py`.

## What Changes

- Add a new standalone script `fetch_subreddit_range_to_hf.py` modeled on
  `fetch_arctic_to_hf.py`.
- Add a new `workflow_dispatch`-only workflow
  `.github/workflows/fetch_subreddit_range.yml`.
- Reuse the same pinned schema and the same Arctic Shift pagination,
  retry, rate-limit, and stuck-cursor behavior as `fetch_arctic_to_hf.py`.
- Upload results directly with `huggingface_hub` `HfApi.upload_file` into
  `darelphilip/reddit_indian_subs`.
- Leave existing scripts and workflows unchanged.

## Out Of Scope

- Changes to `fetch_arctic_to_hf.py`, `consolidate_to_hf.py`, or the monthly
  workflow.
- Schema changes or changes to the dataset repository's existing splits.
- Automatic/scheduled triggering of the new workflow.
