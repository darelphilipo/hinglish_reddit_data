import requests
import pandas as pd
import time
import os
import random
import json
from datetime import datetime, timezone
from datasets import Dataset, Features, Value
from huggingface_hub.errors import HfHubHTTPError
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Explicit schema -- pinned so a batch where a column happens to be all-null
# (e.g. no [removed]/[deleted] comments in this particular slice) doesn't get
# its type inferred as Arrow 'null' and clash with other splits where that
# same column has real string values. 
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
CHECKPOINT_EVERY = 2
BATCH_NAME = os.getenv("BATCH_NAME", "heavy_1")
JOB_TIME_BUDGET_MINUTES = 120
JOB_TIME_BUDGET_SECONDS = JOB_TIME_BUDGET_MINUTES * 60
MIN_SUBREDDIT_SECONDS = 45  
ARCTIC_SHIFT_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"

# ---- Time-range / coverage-index settings ----
UTC = timezone.utc

def epoch(year, month=1, day=1):
    return int(datetime(year, month, day, tzinfo=UTC).timestamp())

def fmt_ts(t):
    return datetime.fromtimestamp(int(t), UTC).strftime('%Y-%m-%d %H:%M:%S')

EARLIEST_YEAR = int(os.getenv("EARLIEST_YEAR", "2020"))
EARLIEST_EPOCH = epoch(EARLIEST_YEAR)

# Never look at "now": the upper bound is always in the PAST. The small lag
# gives Arctic Shift time to ingest recent comments so we don't mark a range
# as covered before it is complete.
INGEST_LAG_SECONDS = int(os.getenv("INGEST_LAG_HOURS", "6")) * 3600
CEILING_EPOCH = int(time.time()) - INGEST_LAG_SECONDS
CEILING_YEAR = datetime.fromtimestamp(CEILING_EPOCH, UTC).year

MIN_GAP_SECONDS = 3600          # uncovered gaps shorter than this are ignored
PAGE_LIMIT = 100
MAX_CONSECUTIVE_EMPTY = 6       # empty random windows in a row -> give up on sub this run
MAX_CONSECUTIVE_FAILURES = 2    # failed segments in a row -> give up on sub this run
MAX_SEGMENTS_PER_SUB = 200
DEAD_RECHECK_SECONDS = 14 * 86400   # re-probe a sub with zero comments after 14 days
HISTORY_KEEP = 50
STATE_VERSION = 2

# ==========================================
# COVERAGE INDEX (per-subreddit ledger)
# ==========================================
# File format (prompt/checkpoint_<batch>.json):
# {
#   "_version": 2,
#   "subs": {
#     "<sub>": {
#        "earliest": <epoch of the sub's first comment >= EARLIEST_YEAR, or null>,
#        "dead_checked_at": <epoch, only if the sub has no comments at all>,
#        "intervals": [[start, end], ...],   # merged, sorted, already-fetched time ranges
#        "rows_total": <rows kept so far>,
#        "history": [{start, end, start_iso, end_iso, rows, run}, ...]  # last 50 segments
#     }
#   }
# }
os.makedirs("prompt", exist_ok=True)
CHECKPOINT_FILE = f"prompt/checkpoint_{BATCH_NAME}.json"

def load_state():
    fresh = {"_version": STATE_VERSION, "subs": {}}
    if not os.path.exists(CHECKPOINT_FILE):
        return fresh
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        print("  [!] Checkpoint file unreadable. Starting with an empty coverage index.", flush=True)
        return fresh
    if not isinstance(data, dict) or data.get("_version") != STATE_VERSION:
        print("  [!] Legacy (v1) checkpoint format detected -- its cursors are ignored "
              "(they were single timestamps, not ranges). Starting a fresh coverage index. "
              "Overlap with already-stored rows is removed by consolidate's global dedup.", flush=True)
        return fresh
    data.setdefault("subs", {})
    return data

def save_state(state):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, CHECKPOINT_FILE)

def get_sub_state(state, subreddit):
    st = state["subs"].setdefault(subreddit, {})
    st.setdefault("earliest", None)
    st.setdefault("intervals", [])
    st.setdefault("rows_total", 0)
    st.setdefault("history", [])
    return st

def merge_intervals(intervals, tol=1):
    cleaned = sorted([int(a), int(b)] for a, b in intervals if b >= a)
    merged = []
    for s, e in cleaned:
        if merged and s <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged

def find_gaps(lo, hi, intervals, min_gap):
    """Uncovered (start, end) pieces of [lo, hi] that are at least min_gap long."""
    gaps = []
    cur = lo
    for s, e in intervals:
        if e <= cur:
            continue
        if s >= hi:
            break
        if s > cur and (s - cur) >= min_gap:
            gaps.append((cur, s))
        cur = max(cur, e)
        if cur >= hi:
            break
    if hi - cur >= min_gap:
        gaps.append((cur, hi))
    return gaps

def pick_target(intervals, earliest):
    """
    Newest year first (2026, then 2025, ...). Inside the first year that still has
    uncovered time, choose a gap (weighted by its length) and a uniformly random
    start inside it. Returns (year, seg_start, seg_end) or None if everything is covered.
    """
    lower_bound = max(EARLIEST_EPOCH, earliest) if earliest else EARLIEST_EPOCH
    for year in range(CEILING_YEAR, EARLIEST_YEAR - 1, -1):
        lo = max(epoch(year), lower_bound)
        hi = min(epoch(year + 1), CEILING_EPOCH)
        if hi - lo < MIN_GAP_SECONDS:
            continue
        gaps = find_gaps(lo, hi, intervals, MIN_GAP_SECONDS)
        if not gaps:
            continue
        gap = random.choices(gaps, weights=[g[1] - g[0] for g in gaps])[0]
        seg_start = random.randint(gap[0], gap[1] - MIN_GAP_SECONDS)
        return year, seg_start, gap[1]
    return None

def commit_pending(state, pending):
    """Fold segments into the persistent ledger. Called ONLY after their rows were pushed to HF."""
    now = int(time.time())
    for sub, segs in pending.items():
        if not segs:
            continue
        st = get_sub_state(state, sub)
        for s, e, rows in segs:
            st["rows_total"] += rows
            st["history"].append({"start": s, "end": e, "start_iso": fmt_ts(s),
                                  "end_iso": fmt_ts(e), "rows": rows, "run": now})
        st["intervals"] = merge_intervals(st["intervals"] + [[s, e] for s, e, _ in segs])
        st["history"] = st["history"][-HISTORY_KEEP:]
    pending.clear()
    save_state(state)

# ==========================================
# DYNAMIC SUBREDDIT FETCHING & BUCKETING
# ==========================================
def get_dynamic_batches():
    url = "https://raw.githubusercontent.com/darelphilipo/hinglish_reddit_data/main/prompt/subreddits.json"
    print(f"Fetching dynamic subreddits from {url}...")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch subreddits.json: {e}")

    config = data.get("config", {})
    categories = data.get("categories", {})
    
    active_subs = set()
    # Obey the JSON config: Only add subreddits if the category config == 1
    for cat, subs in categories.items():
        if config.get(cat, 0) == 1:
            active_subs.update(subs)
            
    # Reference pools to maintain the historical heavy/medium grouping sizes
    KNOWN_HEAVY = {"IndiaSpeaks", "india", "indiameme", "funnyIndia", "IndianDankMemes", "CarryMinati", "ipl", "IndianGaming", "bollywood", "developersIndia", "UPSC", "IndianStockMarket", "JEENEETards", "Btechtards", "StartUpIndia", "AskIndia"}
    KNOWN_MEDIUM = {"indianews", "indiadiscussion", "CriticalThinkingIndia", "unitedstatesofindia", "bihar", "uttarpradesh", "delhi", "karnataka", "TamilNadu", "Maharashtra", "gujarat", "Rajasthan", "bangalore", "mumbai", "chennai", "hyderabad", "kolkata", "pune", "ahmedabad", "lucknow", "Arrangedmarriage", "RelationshipIndia", "TwoXIndia", "AskIndianWomen", "AskIndianMen", "OffMyChestIndia", "TeenIndia", "IndianTeenagers", "Indiangirlsontinder", "DesiWeddings", "TwentiesIndia", "CricketShitpost", "IndiaCricket", "IndianFootball", "indiansports", "RCB", "csk", "chessindia", "SaimanSays", "ShahRukhKhan", "SamayRaina", "thugeshh", "beastboyshub", "sunraybee", "FingMemes", "dankrishu", "ViratKohli", "BollyBlindsNGossip", "InstaCelebsGossip", "bollywoodmemes", "BollywoodFashion", "sharktankindia", "biggboss", "IndianTellyTalk", "DHHMemes", "punjabimusic", "kollywood", "tollywood", "IndianCinema", "BollywoodRealism", "IndianOTTbestof", "AnimeMirchi", "animeindian", "BollywoodMusic", "MalayalamMovies", "IndianHipHopHeads", "IndianStreetBets", "IndiaInvestments", "personalfinanceindia", "CreditCardsIndia", "CryptoIndia", "mutualfunds", "IndiaTax", "BitcoinIndia", "StockMarketIndia", "FIREIndia", "FatFIREIndia", "IndianStocks", "beermoneyindia", "Frugal_Ind", "CATpreparation", "Indian_Academia", "JEE", "IndiaCareers", "BITSPilani", "Indians_StudyAbroad", "IndianWorkplace", "ICSE", "CharteredAccountants", "IndiaBusiness", "smallbusinessindia", "CBSE", "indianmedschool", "CarsIndia", "indianrailways", "indianbikes", "AirTravelIndia", "Indianbooks", "IndianArtAndThinking", "indiafood", "IndianArtAI", "hindi", "IndianFoodPhotos", "IndiaCoffee", "PhotographyIndia", "IndiansRead", "IndiaTech", "GadgetsIndia", "Indiangamers", "XboxIndia", "IndiaPS5", "DesiVideoMemes", "indianmemer", "IndianMeyMeys", "IndianMemeTemplates", "desimemes"}
    
    heavy_pool = sorted([s for s in active_subs if s in KNOWN_HEAVY])
    medium_pool = sorted([s for s in active_subs if s in KNOWN_MEDIUM])
    tiny_pool = sorted([s for s in active_subs if s not in KNOWN_HEAVY and s not in KNOWN_MEDIUM])
    
    def chunker(seq, size):
        return [seq[pos:pos + size] for pos in range(0, len(seq), size)]
        
    batches = {}
    for i, b in enumerate(chunker(heavy_pool, 8), 1): batches[f"heavy_{i}"] = b
    for i, b in enumerate(chunker(medium_pool, 17), 1): batches[f"medium_{i}"] = b
    for i, b in enumerate(chunker(tiny_pool, 30), 1): batches[f"tiny_{i}"] = b
    
    return batches

BATCH_DEFINITIONS = get_dynamic_batches()

if BATCH_NAME not in BATCH_DEFINITIONS:
    raise ValueError(f"Unknown BATCH_NAME '{BATCH_NAME}'. Valid options: {list(BATCH_DEFINITIONS.keys())}")

SUBREDDITS = BATCH_DEFINITIONS[BATCH_NAME]

# ==========================================
# PARSE OVERRIDES FROM ENVIRONMENT
# ==========================================
def parse_env_int(key):
    val = os.getenv(key)
    return int(val) if val and val.strip() else None

max_rows_per_sub = parse_env_int("MAX_ROWS_PER_SUB")
SPLIT_NAME = f"tmp_batch_{BATCH_NAME}"

MAX_ATTEMPTS = 2               
HARD_REQUEST_TIMEOUT = 15      
_executor = ThreadPoolExecutor(max_workers=1)

def get_secure_session():
    return requests.Session()

session = get_secure_session()

def _do_request(params):
    response = session.get(ARCTIC_SHIFT_URL, params=params, timeout=HARD_REQUEST_TIMEOUT)
    return response

def _log_response_headers(response, context):
    interesting = {}
    for key in ("Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"):
        if key in response.headers:
            interesting[key] = response.headers[key]
    if interesting:
        print(f"    [headers:{context}] status={response.status_code} {interesting}", flush=True)
    else:
        print(f"    [headers:{context}] status={response.status_code} (no rate-limit headers present)", flush=True)

def fetch_page_with_retries(params):
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
            print(f"    [!] attempt {attempt}/{MAX_ATTEMPTS} failed after {elapsed:.1f}s: {e}", flush=True)
            if attempt == MAX_ATTEMPTS:
                raise


# ==========================================
# FETCHING
# ==========================================
def probe_earliest(subreddit, sub_state):
    """
    One cheap request that finds the sub's first comment on/after EARLIEST_YEAR.
    Everything before it is treated as 'sub did not exist yet' and is never
    scheduled. Returns False if the sub has no comments at all (skip it).
    """
    if sub_state.get("earliest") is not None:
        return True
    checked = sub_state.get("dead_checked_at")
    if checked and (time.time() - checked) < DEAD_RECHECK_SECONDS:
        print(f"  [skip] r/{subreddit} had no comments when last probed "
              f"({fmt_ts(checked)}). Skipping.", flush=True)
        return False

    params = {"subreddit": subreddit, "after": EARLIEST_EPOCH, "before": CEILING_EPOCH,
              "limit": 1, "sort": "asc"}
    data, elapsed = fetch_page_with_retries(params)
    comments = data.get("data", [])
    if not comments:
        sub_state["dead_checked_at"] = int(time.time())
        print(f"  [skip] r/{subreddit} has NO comments between {fmt_ts(EARLIEST_EPOCH)} and "
              f"{fmt_ts(CEILING_EPOCH)}. Skipping (probe {elapsed:.1f}s).", flush=True)
        return False

    sub_state["earliest"] = int(comments[0]["created_utc"])
    sub_state.pop("dead_checked_at", None)
    print(f"  [probe] r/{subreddit} first comment in range: {fmt_ts(sub_state['earliest'])} "
          f"(nothing before this will be scheduled)", flush=True)
    return True

def fetch_segment(subreddit, seg_start, seg_end, deadline, max_rows, all_comments):
    """
    Fetch comments ascending from seg_start, never past seg_end (which is the start of
    the next already-covered range / year end / ceiling).
    Returns (status, covered_end). Everything in [seg_start, covered_end] is fully fetched.
    status: complete | budget | row_limit | error
    """
    cursor = seg_start
    last_ts = None
    pages = 0
    status = "complete"

    while True:
        if time.time() > deadline:
            status = "budget"
            break
        if max_rows and len(all_comments) >= max_rows:
            status = "row_limit"
            break

        pages += 1
        params = {"subreddit": subreddit, "after": cursor, "before": seg_end,
                  "limit": PAGE_LIMIT, "sort": "asc"}
        try:
            data, elapsed = fetch_page_with_retries(params)
        except Exception as e:
            print(f"  [!] r/{subreddit}: giving up on this segment at page {pages} after "
                  f"{MAX_ATTEMPTS} failed attempts: {e}", flush=True)
            status = "error"
            break

        comments = data.get("data", [])
        if not comments:
            print(f"    r/{subreddit} page {pages}: 0 comments -> range confirmed complete "
                  f"up to {fmt_ts(seg_end)} (req {elapsed:.1f}s)", flush=True)
            cursor = seg_end
            status = "complete"
            break

        kept = 0
        truncated = False
        for comment in comments:
            if max_rows and len(all_comments) >= max_rows:
                truncated = True
                break
            last_ts = int(comment.get("created_utc"))
            body = comment.get("body", "")
            if body and body not in ["[removed]", "[deleted]"]:
                kept += 1
                all_comments.append({
                    "id": comment.get("id"),
                    "body": body,
                    "created_utc": comment.get("created_utc"),
                    "subreddit": subreddit,
                    "score": comment.get("score"),
                    "controversiality": comment.get("controversiality"),
                    "collapsed_reason_code": comment.get("collapsed_reason_code")
                })

        print(f"    r/{subreddit} page {pages}: fetched {len(comments)}, kept {kept} "
              f"(running total {len(all_comments)}), at {fmt_ts(last_ts or cursor)}, "
              f"req {elapsed:.1f}s", flush=True)

        if truncated:
            # Only claim coverage up to the last comment we actually processed.
            if last_ts is not None:
                cursor = last_ts
            status = "row_limit"
            break

        new_after = int(comments[-1]["created_utc"])
        if new_after == cursor:
            print(f"    [!] Pagination cursor stuck at {new_after} on r/{subreddit}. "
                  f"Nudging cursor forward by 1s.", flush=True)
            new_after += 1
        cursor = new_after
        time.sleep(1.0)

    covered_end = cursor if cursor > seg_start else None
    return status, covered_end

def fetch_subreddit_comments(subreddit, sub_state, time_budget_seconds, max_rows):
    """
    Repeatedly: pick a random uncovered window (newest year first), fetch it, and record the
    range. Returns (comments, segments) where segments = [(start, end, rows_kept), ...].
    The ledger is NOT updated here; main() commits segments only after the rows are pushed.
    """
    print(f"--- Fetching r/{subreddit} (time budget: {time_budget_seconds:.0f}s) ---", flush=True)
    sub_start_time = time.time()
    deadline = sub_start_time + time_budget_seconds
    all_comments = []
    segments = []

    try:
        if not probe_earliest(subreddit, sub_state):
            return all_comments, segments
    except Exception as e:
        print(f"  [!] Probe failed for r/{subreddit}: {e}. Skipping this sub for now.", flush=True)
        return all_comments, segments

    consecutive_empty = 0
    consecutive_fail = 0

    for seg_no in range(1, MAX_SEGMENTS_PER_SUB + 1):
        if time.time() > deadline:
            print(f"  [!] r/{subreddit} hit its {time_budget_seconds:.0f}s budget. Moving on.", flush=True)
            break
        if max_rows and len(all_comments) >= max_rows:
            print(f"  [!] r/{subreddit} reached max requested rows ({max_rows}). Moving on.", flush=True)
            break

        local_iv = merge_intervals(sub_state["intervals"] + [[s, e] for s, e, _ in segments])
        target = pick_target(local_iv, sub_state.get("earliest"))
        if target is None:
            print(f"  [✓] r/{subreddit} is fully covered from {fmt_ts(max(EARLIEST_EPOCH, sub_state.get('earliest') or 0))} "
                  f"to {fmt_ts(CEILING_EPOCH)}. Nothing left to fetch.", flush=True)
            break

        year, seg_start, seg_end = target
        print(f"  [🎲] segment {seg_no}: year {year}, random start {fmt_ts(seg_start)} "
              f"(window ends {fmt_ts(seg_end)})", flush=True)

        before = len(all_comments)
        status, covered_end = fetch_segment(subreddit, seg_start, seg_end, deadline, max_rows, all_comments)
        rows = len(all_comments) - before

        if covered_end is not None:
            segments.append((seg_start, covered_end, rows))
            print(f"  [📒] segment {seg_no} covered {fmt_ts(seg_start)} -> {fmt_ts(covered_end)} "
                  f"({rows} rows, status={status})", flush=True)

        if status == "error":
            consecutive_fail += 1
            if consecutive_fail >= MAX_CONSECUTIVE_FAILURES:
                print(f"  [!] {consecutive_fail} failed segments in a row on r/{subreddit}. Moving on.", flush=True)
                break
            continue
        consecutive_fail = 0

        if status == "complete" and rows == 0:
            consecutive_empty += 1
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                print(f"  [skip] {consecutive_empty} empty windows in a row on r/{subreddit} "
                      f"(sub likely inactive in those periods). Moving on.", flush=True)
                break
        else:
            consecutive_empty = 0

        if status in ("budget", "row_limit"):
            break

    sub_elapsed = time.time() - sub_start_time
    print(f"Collected {len(all_comments)} comments from r/{subreddit} in {len(segments)} segment(s) "
          f"({sub_elapsed:.0f}s)", flush=True)
    return all_comments, segments

def push_checkpoint(master_dataset, split_name, label):
    """Returns True when the rows are safely on HF (or there was nothing to push). Raises on failure."""
    if not master_dataset:
        print(f"  [checkpoint:{label}] Nothing to push yet, skipping.", flush=True)
        return True

    df_chunk = pd.DataFrame(master_dataset).drop_duplicates(subset=["id"])
    dataset = Dataset.from_pandas(df_chunk, features=SCHEMA, preserve_index=False)

    max_push_attempts = 5
    for attempt in range(1, max_push_attempts + 1):
        try:
            print(f"  [checkpoint:{label}] Pushing {len(df_chunk)} rows to split '{split_name}' "
                  f"(attempt {attempt}/{max_push_attempts})...", flush=True)
            dataset.push_to_hub(repo_id=HF_DATASET_REPO, split=split_name, private=True)
            print(f"  [checkpoint:{label}] Push complete.", flush=True)
            return True
        except Exception as e:
            is_conflict = "412" in str(e) or "Precondition Failed" in str(e)
            if attempt < max_push_attempts:
                wait = random.uniform(3, 10) * attempt 
                reason = "branch conflict from a concurrent job's push" if is_conflict else \
                          f"transient error ({type(e).__name__}: {e})"
                print(f"  [checkpoint:{label}] Push failed -- {reason}. "
                      f"Retrying in {wait:.1f}s...", flush=True)
                time.sleep(wait)
                continue
            print(f"  [checkpoint:{label}] Push failed after {attempt} attempt(s): {e}", flush=True)
            raise

def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set!")

    if not SUBREDDITS:
        print(f"Batch '{BATCH_NAME}' is empty. Nothing to do.", flush=True)
        return

    print(f"Batch '{BATCH_NAME}' covers {len(SUBREDDITS)} subreddits. Target split: '{SPLIT_NAME}'", flush=True)
    print(f"Time window: {fmt_ts(EARLIEST_EPOCH)} -> {fmt_ts(CEILING_EPOCH)} UTC "
          f"(newest year first, random start inside uncovered gaps)", flush=True)
    print(f"Job time budget: {JOB_TIME_BUDGET_SECONDS}s across {len(SUBREDDITS)} subreddits "
          f"(adaptive per-subreddit allocation, {MIN_SUBREDDIT_SECONDS}s floor)", flush=True)
    if max_rows_per_sub:
        print(f"Row limit override active: Fetching up to {max_rows_per_sub} comments per subreddit.", flush=True)

    state = load_state()
    pending = {}          # segments fetched this run, not yet confirmed pushed
    master_dataset = []
    last_pushed_rows = 0
    job_start_time = time.time()

    for i, sub in enumerate(SUBREDDITS, start=1):
        elapsed_job = time.time() - job_start_time
        remaining_total = JOB_TIME_BUDGET_SECONDS - elapsed_job
        remaining_subs = len(SUBREDDITS) - i + 1

        if remaining_total <= MIN_SUBREDDIT_SECONDS:
            skipped = SUBREDDITS[i - 1:]
            print(f"[!] Job time budget nearly exhausted after {i - 1}/{len(SUBREDDITS)} subreddits "
                  f"({remaining_total:.0f}s left). Skipping remaining {len(skipped)} subs: {skipped}", flush=True)
            break

        fair_share = max(remaining_total / remaining_subs, MIN_SUBREDDIT_SECONDS)
        print(f"[time budget] {remaining_total:.0f}s left for {remaining_subs} subs remaining "
              f"-> allocating up to {fair_share:.0f}s to r/{sub}", flush=True)

        sub_state = get_sub_state(state, sub)
        sub_comments, segments = fetch_subreddit_comments(
            subreddit=sub,
            sub_state=sub_state,
            time_budget_seconds=fair_share,
            max_rows=max_rows_per_sub
        )
        master_dataset.extend(sub_comments)
        pending[sub] = segments
        print(f"[batch progress] {i}/{len(SUBREDDITS)} subreddits done, "
              f"{len(master_dataset)} total rows collected so far in this batch\n", flush=True)

        if i % CHECKPOINT_EVERY == 0 or i == len(SUBREDDITS):
            if push_checkpoint(master_dataset, SPLIT_NAME, label=f"{i}/{len(SUBREDDITS)} subs done"):
                last_pushed_rows = len(master_dataset)
                commit_pending(state, pending)   # ledger only advances after a successful push

        time.sleep(1.0) 

    if len(master_dataset) != last_pushed_rows:
        push_checkpoint(master_dataset, SPLIT_NAME, label="final")
    commit_pending(state, pending)
    save_state(state)   # also persists probe results (earliest / dead markers)
    print(f"\nBatch '{BATCH_NAME}' finished. Final size: {len(master_dataset)} comments.", flush=True)

if __name__ == "__main__":
    main()
