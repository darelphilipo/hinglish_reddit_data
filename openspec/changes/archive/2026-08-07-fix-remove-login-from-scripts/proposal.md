**Priority:** 4/5

# Proposal: Remove `login()` from the HF upload scripts

`fetch_arctic_to_hf.py` (line 289) and `consolidate_to_hf.py` (line 93) both
call `login(token=HF_TOKEN)` from `huggingface_hub`. That call validates the
token against the strictly rate-limited `/api/whoami-v2` endpoint, which is a
real HTTP 429 risk during GitHub Actions runs and can crash the script before
any real work begins.

Neither script needs `login()`:

- `fetch_arctic_to_hf.py` uploads via
  `dataset.push_to_hub(repo_id=HF_DATASET_REPO, split=split_name, private=True)`,
  and `huggingface_hub` auto-reads the `HF_TOKEN` env var (which the workflow
  exports), so no explicit token or login is needed.
- `consolidate_to_hf.py` uses `api = HfApi()` with `api.upload_file(...)` and
  `api.create_commit(...)`, and `HfApi()` auto-reads the `HF_TOKEN` env var
  too.

The fix removes the `login()` call and the now-unused `login` import from both
scripts so no whoami request is made at all.

This is a blocking fix: it removes the extra, unused whoami request from both
workflows, preventing startup failures when `/api/whoami-v2` is rate limited.

## Scope

- Remove the `login(token=HF_TOKEN)` call in `fetch_arctic_to_hf.py`.
- Remove `login` from the `from huggingface_hub import login` line in
  `fetch_arctic_to_hf.py`.
- Remove the `login(token=HF_TOKEN)` call in `consolidate_to_hf.py`.
- Remove `login` from the `from huggingface_hub import login, HfApi,
  CommitOperationDelete` line in `consolidate_to_hf.py` so only `HfApi` and
  `CommitOperationDelete` remain imported.
- No behavioral change to fetching, consolidation, uploads, or commits.

## Out Of Scope

- Changes to `fetch_subreddit_range_to_hf.py` or any other script.
- Changes to the fetch, pagination, retry, consolidation, or upload logic.
- Changes to the GitHub Actions workflows.
- Changes to how `HF_TOKEN` is exported or stored.
