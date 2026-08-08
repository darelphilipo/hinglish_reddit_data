# subreddit-range-fetch Specification

## Purpose
This spec covers the standalone `fetch_subreddit_range_to_hf.py` script and
its manually triggered workflow, including Arctic Shift pagination,
filtering/deduplication, graceful empty handling, direct append uploads to
Hugging Face, failure handling and exit codes, and the manual standalone
workflow.
## Requirements
### Requirement: Standalone subreddit range fetch script

The repository SHALL provide a standalone Python script
(`fetch_subreddit_range_to_hf.py`) that fetches Reddit comments for one or
more subreddits over an inclusive month range via the Arctic Shift API and
uploads non-empty results directly to the Hugging Face dataset repository
`darelphilip/reddit_indian_subs`.

#### Scenario: Single subreddit fetch

- **WHEN** the script runs with `SUBREDDITS=IndiaSpeaks`, a start month, and
  an end month
- **THEN** it pages through every comment in r/IndiaSpeaks for each month in
  the inclusive range in ascending order using the Arctic Shift comments
  search endpoint with `limit=100` and `sort=asc`, filters removed/deleted and
  empty-body comments, and uploads any kept rows to Hugging Face

#### Scenario: Comma-separated multiple subreddits

- **WHEN** `SUBREDDITS` is a comma-separated list such as
  `IndiaSpeaks,india,ipl`
- **THEN** the script parses each subreddit name and processes every
  subreddit in the list over the full month range, treating each
  subreddit-month as an independent unit

#### Scenario: Inclusive multi-month range

- **WHEN** `START_YEAR`/`START_MONTH` and `END_YEAR`/`END_MONTH` define a
  range spanning more than one month
- **THEN** the script iterates one calendar month at a time from the start of
  the start month through the end of the end month inclusive, using
  start-of-month epochs as `after`/`before` bounds for each slice

### Requirement: Arctic Shift fetch behavior

The script SHALL fetch comments with ascending cursor pagination, a hard
per-request timeout, one retry per page, rate-limit header logging, and
stuck-cursor protection, matching `fetch_arctic_to_hf.py`.

#### Scenario: Pagination completes normally

- **WHEN** pages return up to 100 comments and the cursor advances
- **THEN** the script continues to the next page until the range is exhausted
  and records per-page fetched and kept row counts

#### Scenario: Stuck pagination cursor

- **WHEN** the last comment's `created_utc` equals the current `after` cursor
- **THEN** the script nudges the cursor forward by 1 second and continues,
  logging the stuck-cursor event

#### Scenario: Rate-limit headers present

- **WHEN** a response includes `Retry-After` or `X-RateLimit-*` headers
- **THEN** the script logs those headers with the response status so future
  tuning is based on what the server reports

#### Scenario: Page request times out or fails both attempts

- **WHEN** a page request exceeds the hard timeout or fails on both its
  initial attempt and its single retry
- **THEN** the script logs the failure, abandons that subreddit-month, and
  moves on to the next input

### Requirement: Comment filtering and in-run deduplication

The script SHALL keep only comments with a non-empty body that is not
`[removed]` or `[deleted]`, and SHALL deduplicate collected comments by
comment `id` within the run.

#### Scenario: Empty and removed comments are excluded

- **WHEN** collected comments have empty bodies or bodies equal to
  `[removed]` or `[deleted]`
- **THEN** those comments are not kept and are counted in the filtered-out
  rows log

#### Scenario: Duplicate comment ids in a run

- **WHEN** the same comment `id` appears more than once during a run
- **THEN** only the first occurrence is kept for upload and the duplicates are
  dropped

### Requirement: Graceful handling of empty subreddit-months

For each subreddit-month with zero kept comments, the script SHALL log a clear
"no data, skipping" message, upload nothing, and continue.

#### Scenario: A subreddit-month has no comments

- **WHEN** a subreddit-month produces zero kept comments after filtering
- **THEN** the script logs a clear no-data message for that subreddit-month,
  uploads no file for it, and continues processing the remaining inputs

#### Scenario: All subreddit-months are empty

- **WHEN** every subreddit-month in the range yields zero kept comments
- **THEN** the script uploads nothing and still exits 0 after logging the
  summary

### Requirement: Direct append upload to Hugging Face

Non-empty results SHALL be uploaded directly to the dataset repository
`darelphilip/reddit_indian_subs` as a unique
`data/train_append_<8-char-uuid>.parquet` file using `HfApi.upload_file`, with
the pinned schema (id string, body string, created_utc int64, subreddit
string, score int64, controversiality int64, collapsed_reason_code string)
and up to 5 jittered retry attempts.

#### Scenario: Non-empty subreddit-month is uploaded

- **WHEN** a subreddit-month has at least one kept comment
- **THEN** the script serializes those rows with the pinned schema to a
  parquet file and uploads it to
  `data/train_append_<8-char-uuid>.parquet` in `darelphilip/reddit_indian_subs`
  via `HfApi.upload_file`, generating a fresh 8-character UUID for each file

#### Scenario: Transient upload failure is retried

- **WHEN** an upload attempt fails with a transient error
- **THEN** the script retries up to 5 total attempts with jittered, growing
  backoff and logs each retry reason

### Requirement: Failure handling and exit codes

The script SHALL continue past per-subreddit/month API failures and exit 0
after all inputs are processed even if individual subreddits failed or
produced no data, and SHALL exit non-zero only for fatal issues.

#### Scenario: A subreddit-month API failure is not fatal

- **WHEN** a subreddit-month fails all fetch retries
- **THEN** the script logs the failure, continues with the next input, and
  still exits 0 once all inputs have been processed

#### Scenario: Missing HF token is fatal

- **WHEN** `HF_TOKEN` is not set
- **THEN** the script exits non-zero before fetching anything and never logs
  the token

#### Scenario: Invalid or incomplete inputs are fatal

- **WHEN** `SUBREDDITS` is empty, or any of the year/month variables is
  missing, non-numeric, or outside 1-12 for months, or the start range is
  after the end range
- **THEN** the script exits non-zero before fetching anything and logs the
  invalid input

#### Scenario: Upload still failing after all retries

- **WHEN** an upload for a non-empty dataset still fails after all 5 attempts
- **THEN** the script exits non-zero

### Requirement: Manual standalone workflow

The repository SHALL provide a separate GitHub Actions workflow
(`.github/workflows/fetch_subreddit_range.yml`) triggered only by
`workflow_dispatch`, with required inputs `subreddits`, `start_year`,
`start_month`, `end_year`, and `end_month`, that runs the script with `HF_TOKEN`
from the repository secrets.

#### Scenario: Operator manually dispatches the workflow

- **WHEN** an authorized operator manually dispatches the workflow with a
  comma-separated subreddit list and a start/end month range
- **THEN** GitHub Actions checks out the repository, installs Python 3.10 and
  the required dependencies (`requests`, `pandas`, `datasets`,
  `huggingface_hub`, `python-dateutil`, `urllib3`), and runs
  `python fetch_subreddit_range_to_hf.py` with `HF_TOKEN` from secrets,
  `PYTHONUNBUFFERED=1`, and the input values, under a 60-minute timeout

#### Scenario: Workflow is not started automatically

- **WHEN** the monthly schedule occurs or another workflow runs
- **THEN** the subreddit range fetch workflow is not triggered

#### Scenario: Overlapping manual runs are prevented

- **WHEN** a second manual dispatch occurs while one is already running
- **THEN** the second run joins the same concurrency group and does not run
  concurrently with the first

#### Scenario: Script result determines workflow result

- **WHEN** the script exits with a failure status
- **THEN** the workflow run is marked as failed

#### Scenario: The HF token stays secret

- **WHEN** the workflow runs the script
- **THEN** the token is provided only via the `HF_TOKEN` environment variable
  from `${{ secrets.HF_TOKEN }}` and is never logged or written to files

