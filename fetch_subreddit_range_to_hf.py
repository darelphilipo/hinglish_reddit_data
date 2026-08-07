import os
import sys
import time
import uuid
import random
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from datasets import Dataset, Features, Value
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Explicit schema -- pinned so a slice where a column happens to be all-null
# (e.g. no [removed]/[deleted] comments in this particular month) doesn't get
# its type inferred as Arrow 'null' and clash with other splits where that
# same column has real string values. This is what caused the last failure.
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
# CONFIGURATION
# ==========================================
HF_DATASET_REPO = "darelphilip/reddit_indian_subs"
HF_TOKEN = os.getenv("HF_TOKEN")

ARCTIC_SHIFT_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"

MAX_ATTEMPTS = 2               # one try, one retry -- then move on
HARD_REQUEST_TIMEOUT = 15      # true wall-clock cap per attempt, regardless of trickling data
PAGE_SLEEP_SECONDS = 1.0       # reduced wait between pages
MAX_UPLOAD_ATTEMPTS = 5        # jittered, growing backoff for each HF upload

_executor = ThreadPoolExecutor(max_workers=1)


def get_secure_session():
    return requests.Session()


session = get_secure_session()


def fatal(message):
    """Log a fatal error and exit non-zero. Used ONLY for issues that should
    abort the whole run before/regardless of fetching."""
    print(f"\n[FATAL] {message}", flush=True)
    sys.exit(1)


def _do_request(params):
    """The actual blocking network call, run in a worker thread so we can
    enforce a true total-duration timeout around it. requests' own `timeout`
    kwarg only resets on each new byte received, so a slow-trickling
    response can hang far longer than the value you pass it."""
    response = session.get(ARCTIC_SHIFT_URL, params=params, timeout=HARD_REQUEST_TIMEOUT)
    return response


def _log_response_headers(response, context):
    """Log rate-limit / retry-after headers whenever we get a non-200,
    so future tuning is based on what the server actually tells us
    instead of trial and error."""
    interesting = {}
    for key in ("Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"):
        if key in response.headers:
            interesting[key] = response.headers[key]
    if interesting:
        print(f"    [headers:{context}] status={response.status_code} {interesting}", flush=True)
    else:
        print(f"    [headers:{context}] status={response.status_code} (no rate-limit headers present)", flush=True)


def fetch_page_with_retries(params):
    """Fetches a single page. One attempt, capped at HARD_REQUEST_TIMEOUT
    seconds. On failure, one retry with the same cap, then raises so the
    caller can abandon this subreddit-month and move on."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time()
        try:
            future = _executor.submit(_do_request, params)
            response = future.result(timeout=HARD_REQUEST_TIMEOUT)
            elapsed = time.time() - t0

            if response.status_code >= 400:
                _log_response_headers(response, context=f"attempt {attempt}")
                response.raise_for_status()

            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining is not None and int(remaining) < 5:
                print(f"    [!] Rate limit getting low: {remaining} requests remaining "
                      f"(resets at {response.headers.get('X-RateLimit-Reset')})", flush=True)

            return response.json(), elapsed

        except FutureTimeoutError:
            elapsed = time.time() - t0
            future.cancel()
            print(f"    [!] attempt {attempt}/{MAX_ATTEMPTS} timed out after {elapsed:.1f}s "
                  f"(no response within {HARD_REQUEST_TIMEOUT}s).", flush=True)
            if attempt == MAX_ATTEMPTS:
                raise
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    [!] attempt {attempt}/{MAX_ATTEMPTS} failed after {elapsed:.1f}s: "
                  f"{type(e).__name__}: {e}", flush=True)
            if attempt == MAX_ATTEMPTS:
                raise


# ==========================================
# INPUT VALIDATION (all fatal before fetching)
# ==========================================
def normalize_subreddits(raw):
    """Strip whitespace, lowercase, drop empties, deduplicate preserving order."""
    seen = set()
    result = []
    for part in raw.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def parse_int_env(name):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        fatal(f"{name} is required but not set.")
    try:
        return int(value)
    except ValueError:
        fatal(f"{name} must be an integer, got '{value}'.")


def parse_inputs():
    """Validate all environment inputs. Any failure exits non-zero before
    anything is fetched. The HF token is never logged."""
    if not HF_TOKEN:
        fatal("HF_TOKEN environment variable is not set.")

    raw_subs = os.getenv("SUBREDDITS")
    if raw_subs is None or raw_subs.strip() == "":
        fatal("SUBREDDITS is required but not set.")
    subreddits = normalize_subreddits(raw_subs)
    if not subreddits:
        fatal("SUBREDDITS parsed to an empty list (no valid subreddit names provided).")

    start_year = parse_int_env("START_YEAR")
    start_month = parse_int_env("START_MONTH")
    end_year = parse_int_env("END_YEAR")
    end_month = parse_int_env("END_MONTH")

    for name, value in (("START_MONTH", start_month), ("END_MONTH", end_month)):
        if not (1 <= value <= 12):
            fatal(f"{name} must be 1-12, got {value}.")

    range_start = datetime(start_year, start_month, 1)
    range_end = datetime(end_year, end_month, 1)
    if range_start > range_end:
        fatal(f"START range ({range_start.strftime('%Y-%m')}) is after "
              f"END range ({range_end.strftime('%Y-%m')}).")

    return subreddits, range_start, range_end


# ==========================================
# FETCHING
# ==========================================
def fetch_subreddit_month_comments(subreddit, month_start, seen_ids):
    """Pages through one subreddit for one calendar month in ascending order.
    Returns (raw_comments_fetched, kept_rows). Failing pages after both
    attempts abandon paging but keep what was collected so far."""
    label = f"r/{subreddit} {month_start.strftime('%Y-%m')}"
    after = int(month_start.timestamp())
    before = int((month_start + relativedelta(months=1)).timestamp())

    print(f"--- Fetching {label} ---", flush=True)
    rows = []
    raw_fetched = 0
    page_count = 0
    current_after = after

    while True:
        page_count += 1
        params = {
            "subreddit": subreddit,
            "after": current_after,
            "before": before,
            "limit": 100,
            "sort": "asc"
        }

        try:
            data, elapsed = fetch_page_with_retries(params)
            comments = data.get("data", [])
            raw_fetched += len(comments)

            if not comments:
                print(f"    {label} page {page_count}: 0 new comments (end of range), "
                      f"req took {elapsed:.1f}s", flush=True)
                break

            kept = 0
            for comment in comments:
                body = comment.get("body", "")
                if not body or body in ("[removed]", "[deleted]"):
                    continue
                cid = comment.get("id")
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                kept += 1
                rows.append({
                    "id": cid,
                    "body": body,
                    "created_utc": comment.get("created_utc"),
                    "subreddit": subreddit,
                    "score": comment.get("score"),
                    "controversiality": comment.get("controversiality"),
                    "collapsed_reason_code": comment.get("collapsed_reason_code")
                })

            print(f"    {label} page {page_count}: fetched {len(comments)}, kept {kept} "
                  f"(running total {len(rows)}), req took {elapsed:.1f}s", flush=True)

            new_after = comments[-1]["created_utc"]
            if new_after == current_after:
                print(f"    [!] Pagination cursor stuck at {new_after} on {label} "
                      f"(page {page_count}). Nudging cursor forward by 1s.", flush=True)
                new_after += 1
            current_after = new_after
            time.sleep(PAGE_SLEEP_SECONDS)

        except Exception as e:
            print(f"  [!] Giving up on {label} at page {page_count} after {MAX_ATTEMPTS} failed "
                  f"attempts: {type(e).__name__}: {e}. Moving on with what was collected so far.", flush=True)
            break

    print(f"Collected {len(rows)} kept rows from {label} "
          f"({page_count} pages, {raw_fetched} raw comments)", flush=True)
    return raw_fetched, rows


# ==========================================
# UPLOADING
# ==========================================
def upload_rows(rows, label):
    """Serialize kept rows to a local parquet file with the pinned SCHEMA and
    upload it to HF via HfApi.upload_file, up to MAX_UPLOAD_ATTEMPTS with
    jittered, growing backoff. The local file is always deleted after the
    attempt block, regardless of success. Returns (succeeded, path_in_repo)."""
    df = pd.DataFrame(rows).drop_duplicates(subset=["id"])
    local_name = f"train_append_{str(uuid.uuid4())[:8]}.parquet"
    path_in_repo = f"data/{local_name}"

    try:
        dataset = Dataset.from_pandas(df, features=SCHEMA, preserve_index=False)
        dataset.to_parquet(local_name)
    except Exception as e:
        print(f"  [upload:{label}] Failed to serialize {len(df)} rows to {local_name}: "
              f"{type(e).__name__}: {e}", flush=True)
        if os.path.exists(local_name):
            os.remove(local_name)
        return False, None

    file_size_mb = os.path.getsize(local_name) / (1024 * 1024)
    print(f"  [upload:{label}] Serialized {len(df)} rows to {local_name} "
          f"({file_size_mb:.2f} MB)", flush=True)

    api = HfApi()
    succeeded = False
    for attempt in range(1, MAX_UPLOAD_ATTEMPTS + 1):
        try:
            print(f"  [upload:{label}] Uploading {len(df)} rows to "
                  f"{HF_DATASET_REPO}/{path_in_repo} (attempt {attempt}/{MAX_UPLOAD_ATTEMPTS})...", flush=True)
            api.upload_file(
                path_or_fileobj=local_name,
                path_in_repo=path_in_repo,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                token=HF_TOKEN
            )
            print(f"  [upload:{label}] Upload complete: {path_in_repo} "
                  f"({len(df)} rows, {file_size_mb:.2f} MB)", flush=True)
            succeeded = True
            break
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            if attempt < MAX_UPLOAD_ATTEMPTS:
                wait = random.uniform(3, 10) * attempt  # jittered, growing backoff
                print(f"  [upload:{label}] Upload failed (attempt {attempt}/{MAX_UPLOAD_ATTEMPTS}) -- "
                      f"{reason}. Retrying in {wait:.1f}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"  [upload:{label}] Upload failed after {attempt} attempt(s): {reason}", flush=True)

    if os.path.exists(local_name):
        os.remove(local_name)
        print(f"  [upload:{label}] Deleted local file {local_name}", flush=True)

    return succeeded, (path_in_repo if succeeded else None)


# ==========================================
# MAIN
# ==========================================
def main():
    print("\n" + "=" * 70)
    print("SUBREDDIT RANGE FETCH TO HF")
    print("=" * 70 + "\n", flush=True)

    subreddits, range_start, range_end = parse_inputs()

    # Build the inclusive month list once.
    months = []
    cursor = range_start
    while cursor <= range_end:
        months.append(cursor)
        cursor += relativedelta(months=1)

    print(f"Inputs:", flush=True)
    print(f"  HF_TOKEN: <set>", flush=True)
    print(f"  SUBREDDITS ({len(subreddits)}): {subreddits}", flush=True)
    print(f"  Range: {range_start.strftime('%Y-%m')} through "
          f"{range_end.strftime('%Y-%m')} inclusive "
          f"({len(months)} month(s): {[m.strftime('%Y-%m') for m in months]})", flush=True)

    seen_ids = set()
    outcomes = []
    upload_failures = 0
    total_rows_uploaded = 0
    total_files_uploaded = 0
    job_start_time = time.time()

    for subreddit in subreddits:
        for month_start in months:
            month_label = f"{subreddit} {month_start.strftime('%Y-%m')}"
            outcome = {
                "subreddit": subreddit,
                "month": month_start.strftime('%Y-%m'),
                "fetched": 0,
                "kept": 0,
                "file": None,
                "status": None
            }

            raw_fetched, rows = fetch_subreddit_month_comments(subreddit, month_start, seen_ids)
            outcome["fetched"] = raw_fetched
            outcome["kept"] = len(rows)

            if not rows:
                print(f"No comments found for r/{subreddit} in {month_start.strftime('%Y-%m')}; "
                      f"skipping (nothing to upload)", flush=True)
                outcome["status"] = "no data"
            else:
                ok, path_in_repo = upload_rows(rows, month_label)
                if ok:
                    total_rows_uploaded += len(rows)
                    total_files_uploaded += 1
                    outcome["file"] = path_in_repo
                    outcome["status"] = "uploaded"
                else:
                    upload_failures += 1
                    outcome["status"] = "upload failed"

            outcomes.append(outcome)

    elapsed = time.time() - job_start_time

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70, flush=True)
    print(f"  Inputs: SUBREDDITS={subreddits}", flush=True)
    print(f"  Range: {range_start.strftime('%Y-%m')} through "
          f"{range_end.strftime('%Y-%m')} inclusive", flush=True)
    for o in outcomes:
        file_note = f", file={o['file']}" if o["file"] else ""
        print(f"  r/{o['subreddit']} {o['month']}: rows_fetched={o['fetched']}, "
              f"rows_kept={o['kept']}, status={o['status']}{file_note}", flush=True)
    print(f"  Total rows uploaded: {total_rows_uploaded}", flush=True)
    print(f"  Total files uploaded: {total_files_uploaded}", flush=True)
    print(f"  Upload failures: {upload_failures}", flush=True)
    print(f"  Elapsed time: {elapsed:.0f}s", flush=True)

    if upload_failures:
        print(f"\n[FATAL] {upload_failures} non-empty subreddit-month dataset(s) failed "
              f"to upload after all {MAX_UPLOAD_ATTEMPTS} attempts. See summary above.", flush=True)
        sys.exit(1)

    print("\nDone.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
