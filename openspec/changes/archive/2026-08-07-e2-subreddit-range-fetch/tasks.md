# Tasks

- [x] Add `fetch_subreddit_range_to_hf.py` modeled on `fetch_arctic_to_hf.py`.
- [x] Read and validate all input environment variables (`HF_TOKEN`,
      `SUBREDDITS`, `START_YEAR`, `START_MONTH`, `END_YEAR`, `END_MONTH`).
- [x] Parse the subreddit list from the comma-separated `SUBREDDITS` value.
- [x] Compute the inclusive month range and iterate one month at a time.
- [x] Implement Arctic Shift comment search with ascending cursor pagination,
      limit 100, hard per-request timeout, one retry per page, rate-limit
      header logging, and stuck-cursor protection.
- [x] Filter out empty bodies and `[removed]`/`[deleted]` comments.
- [x] Deduplicate collected comments by comment `id` within the run.
- [x] Log a clear "no data, skipping" message for subreddit-months with zero
      kept comments and upload nothing.
- [x] Upload non-empty results directly to
      `darelphilip/reddit_indian_subs` as
      `data/train_append_<8-char-uuid>.parquet` via `HfApi.upload_file` with
      up to 5 jittered retry attempts and the pinned `SCHEMA`.
- [x] Continue on per-subreddit/month failures and exit 0 after processing all
      inputs even if some subreddits failed or produced no data.
- [x] Exit non-zero only for fatal issues (missing `HF_TOKEN`, invalid or
      incomplete inputs, empty subreddit list, upload still failing after all
      retries for a non-empty dataset).
- [x] Add detailed debug logging (inputs, month range, page/row counts,
      retry reasons, rate-limit headers, upload path/row count/file size,
      final summary) without ever logging the HF token.
- [x] Add `.github/workflows/fetch_subreddit_range.yml` with a
      `workflow_dispatch` trigger only, the five required inputs, a concurrency
      group, `ubuntu-latest`, checkout@v4, setup-python@v5 with Python 3.10,
      the required pip packages, `HF_TOKEN` from secrets,
      `PYTHONUNBUFFERED=1`, and a 60-minute timeout.
- [x] Confirm existing scripts and workflows are unchanged.
- [x] Validate the OpenSpec change with
      `openspec validate e2-subreddit-range-fetch`.
