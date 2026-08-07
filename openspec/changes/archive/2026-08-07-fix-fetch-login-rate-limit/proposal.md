**Priority:** 4/5

# Proposal: Remove `login()` from the subreddit range fetch script

`fetch_subreddit_range_to_hf.py` calls `login(token=HF_TOKEN)` from
`huggingface_hub` before fetching. That call validates the token via the
strictly rate-limited `/api/whoami-v2` endpoint, which returned HTTP 429
during a GitHub Actions run and crashed the script before any fetching began.

The script does not need `login()` because it already passes
`token=HF_TOKEN` explicitly to every `HfApi.upload_file` call. Removing the
`login()` call (and the now-unused `login` import) eliminates the extra
whoami request, so the script can start fetching without depending on an
endpoint it does not use.

This is a blocking fix: it unblocks the `fetch_subreddit_range` workflow,
which is currently failing at startup before a single page is fetched.

## Scope

- Remove the `login(token=HF_TOKEN)` call in `fetch_subreddit_range_to_hf.py`.
- Remove `login` from the `huggingface_hub` import in
  `fetch_subreddit_range_to_hf.py` so the unused import does not linger.
- No behavioral change to fetching, filtering, deduplication, or uploads.

## Out Of Scope

- Changes to `fetch_arctic_to_hf.py`, `consolidate_to_hf.py`, or any other
  script.
- Changes to the fetch, pagination, retry, or upload logic.
- Changes to the GitHub Actions workflow.
