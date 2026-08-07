# Design: Subreddit Range Fetch to Hugging Face

## Script Shape

Create one standalone script `fetch_subreddit_range_to_hf.py` modeled on
`fetch_arctic_to_hf.py`. It reads everything from environment variables so the
workflow stays declarative:

| Variable | Required | Meaning |
| --- | --- | --- |
| `HF_TOKEN` | yes | Hugging Face write token |
| `SUBREDDITS` | yes | one name or comma-separated names, e.g. `IndiaSpeaks` or `IndiaSpeaks,india,ipl` |
| `START_YEAR` | yes | inclusive range start year |
| `START_MONTH` | yes | inclusive range start month, 1-12 |
| `END_YEAR` | yes | inclusive range end year |
| `END_MONTH` | yes | inclusive range end month, 1-12 |

## Fetching Model

Reuse the exact approach of `fetch_arctic_to_hf.py`:

- Endpoint: `https://arctic-shift.photon-reddit.com/api/comments/search`.
- Query params: `subreddit`, `after`, `before`, `limit=100`, `sort=asc`.
- Hard per-request timeout of 15 seconds enforced with a single-worker
  `ThreadPoolExecutor` (requests' own `timeout` only resets per byte received).
- `MAX_ATTEMPTS = 2`: one try, one retry per page, then raise.
- Log `Retry-After` / `X-RateLimit-*` headers on non-200 responses and when
  `X-RateLimit-Remaining` is low.
- Stuck-cursor protection: if the last comment's `created_utc` equals the
  current `after`, nudge it forward by 1 second.

Unlike `fetch_arctic_to_hf.py`, there is no per-job time budget and no adaptive
fair-share allocation; the run is bounded by the workflow's 60-minute timeout.
A short sleep (1s) between pages keeps the same cadence.

## Month Iteration

Compute the effective range once at startup:

- `range_start = datetime(START_YEAR, START_MONTH, 1)`
- `range_end_exclusive = datetime(END_YEAR, END_MONTH, 1) + 1 month`

Iterate one calendar month at a time from `range_start` to
`range_end_exclusive`. For each month, `after` is the start-of-month epoch and
`before` is the start-of-month-after epoch, so the inclusive start and end
months are both covered.

## Cleaning And Dedup

- Keep only comments whose `body` is non-empty and not `[removed]` or
  `[deleted]`, mirroring `fetch_arctic_to_hf.py`.
- Build rows against the pinned `SCHEMA`:

  ```
  id: string, body: string, created_utc: int64, subreddit: string,
  score: int64, controversiality: int64, collapsed_reason_code: string
  ```

- Deduplicate by comment `id` within the current run only (per-subreddit-month
  slice as it is built, then again across the run's accumulated set). No
  cross-run/global dedup is attempted.

## Empty-Slice Behavior

For each subreddit-month with zero kept comments, log a clear message such as
`r/<sub> YYYY-MM: no data, skipping.` and upload nothing, then continue to the
next subreddit-month.

## Upload Model

Instead of `dataset.push_to_hub`, upload each non-empty subreddit-month slice
directly:

- Build a `Dataset` from the deduplicated rows with the pinned `SCHEMA`, save
  it to a local parquet file, then call
  `HfApi.upload_file(path_or_fileobj, path_in_repo, repo_id, token)`.
- `path_in_repo` is `data/train_append_<8-char-uuid>.parquet` where the uuid is
  freshly generated for each uploaded file (so files never collide across
  concurrent runs and can be swept later).
- Up to 5 attempts with jittered, growing backoff (`random.uniform(3, 10) *
  attempt`), matching the retry style of `push_checkpoint`.

## Failure Handling And Exit Codes

- Per-subreddit/month API failures (after the page retries are exhausted) are
  logged and skipped; the loop continues to the next input.
- After all inputs are processed, the script exits 0 even if individual
  subreddits failed or produced no data.
- The script exits non-zero only for fatal issues:
  - missing `HF_TOKEN`;
  - invalid/incomplete inputs (non-numeric year/month, month outside 1-12,
    empty subreddit list, start range after end range);
  - an upload that is still failing after all 5 attempts for a non-empty
    dataset.

## Logging

Verbose, flushed debug logging throughout: parsed inputs, effective month
range, per-subreddit/month page and row counts, rows fetched / filtered /
kept, retry reasons, rate-limit headers, upload path / row count / file size,
and a final summary. Never log the HF token.

## Workflow Shape

Create `.github/workflows/fetch_subreddit_range.yml`:

- `name: Fetch Subreddit Range to Hugging Face`.
- `on: workflow_dispatch` only, with required string inputs: `subreddits`,
  `start_year`, `start_month`, `end_year`, `end_month`.
- `concurrency.group` set to `fetch-subreddit-range` with
  `cancel-in-progress: false` so overlapping runs queue rather than interleave.
- One job on `ubuntu-latest`, `timeout-minutes: 60`:
  - checkout@v4;
  - setup-python@v5 with Python 3.10;
  - install `requests pandas datasets huggingface_hub python-dateutil urllib3`;
  - run `python fetch_subreddit_range_to_hf.py` with `HF_TOKEN` from
    `${{ secrets.HF_TOKEN }}` and `PYTHONUNBUFFERED=1`, plus the five input
    environment variables.

## Security

The Hugging Face token is passed only through the job environment from the
GitHub Actions secret and is never written to logs or repository files.
