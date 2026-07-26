import requests
import pandas as pd
import time
import os
import random
from datetime import datetime
from dateutil.relativedelta import relativedelta
from datasets import Dataset, Features, Value
from huggingface_hub import login
from huggingface_hub.errors import HfHubHTTPError
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Explicit schema -- pinned so a batch where a column happens to be all-null
# (e.g. no [removed]/[deleted] comments in this particular slice) doesn't get
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
HF_DATASET_REPO = "darelphilip/reddit_indian_subs"  # CHANGE THIS
HF_TOKEN = os.getenv("HF_TOKEN")

# Push a checkpoint split to HF every N subreddits *within* this batch.
CHECKPOINT_EVERY = 2

# Which named batch this job processes (set via the workflow matrix)
BATCH_NAME = os.getenv("BATCH_NAME", "heavy_1")

# Total wall-clock budget this job gives itself for scraping, leaving a
# safety margin inside the step's own timeout-minutes for the final
# checkpoint push and log flush. Tune JOB_TIME_BUDGET_MINUTES to comfortably
# fit inside your workflow's timeout-minutes (e.g. 55 for a 60-min step).
JOB_TIME_BUDGET_MINUTES = 55
JOB_TIME_BUDGET_SECONDS = JOB_TIME_BUDGET_MINUTES * 60
MIN_SUBREDDIT_SECONDS = 45  # floor so a late subreddit still gets *some* real time

# ==========================================
# SUBREDDIT BATCHES -- grouped by (a) how likely the sub is to yield hate
# speech / toxic content, ordered highest-likelihood-first within each tier,
# and (b) traffic volume, which drives batch size:
#   heavy_*  : 8 large/high-traffic subs per batch  (few subs, each can hog time)
#   medium_* : ~17 medium-traffic subs per batch
#   tiny_*   : 30+ small/niche subs per batch (many subs, each is quick)
# ==========================================
BATCH_DEFINITIONS = {
    "heavy_1": ["IndiaSpeaks", "india", "indiameme", "funnyIndia", "IndianDankMemes", "CarryMinati", "ipl", "IndianGaming"],
    "heavy_2": ["bollywood", "developersIndia", "UPSC", "IndianStockMarket", "JEENEETards", "Btechtards", "StartUpIndia", "AskIndia"],

    "medium_1": ["indianews", "indiadiscussion", "CriticalThinkingIndia", "unitedstatesofindia", "bihar", "uttarpradesh", "delhi", "karnataka", "TamilNadu", "Maharashtra", "gujarat", "Rajasthan", "bangalore", "mumbai", "chennai", "hyderabad", "kolkata"],
    "medium_2": ["pune", "ahmedabad", "lucknow", "Arrangedmarriage", "RelationshipIndia", "TwoXIndia", "AskIndianWomen", "AskIndianMen", "OffMyChestIndia", "TeenIndia", "IndianTeenagers", "Indiangirlsontinder", "DesiWeddings", "TwentiesIndia", "CricketShitpost", "IndiaCricket", "IndianFootball"],
    "medium_3": ["indiansports", "RCB", "csk", "chessindia", "SaimanSays", "ShahRukhKhan", "SamayRaina", "thugeshh", "beastboyshub", "sunraybee", "FingMemes", "dankrishu", "ViratKohli", "BollyBlindsNGossip", "InstaCelebsGossip", "bollywoodmemes", "BollywoodFashion"],
    "medium_4": ["sharktankindia", "biggboss", "IndianTellyTalk", "DHHMemes", "punjabimusic", "kollywood", "tollywood", "IndianCinema", "BollywoodRealism", "IndianOTTbestof", "AnimeMirchi", "animeindian", "BollywoodMusic", "MalayalamMovies", "IndianHipHopHeads", "IndianStreetBets", "IndiaInvestments"],
    "medium_5": ["personalfinanceindia", "CreditCardsIndia", "CryptoIndia", "mutualfunds", "IndiaTax", "BitcoinIndia", "StockMarketIndia", "FIREIndia", "FatFIREIndia", "IndianStocks", "beermoneyindia", "Frugal_Ind", "CATpreparation", "Indian_Academia", "JEE", "IndiaCareers", "BITSPilani"],
    "medium_6": ["Indians_StudyAbroad", "IndianWorkplace", "ICSE", "CharteredAccountants", "IndiaBusiness", "smallbusinessindia", "CBSE", "indianmedschool", "CarsIndia", "indianrailways", "indianbikes", "AirTravelIndia", "Indianbooks", "IndianArtAndThinking", "indiafood", "IndianArtAI", "hindi"],
    "medium_7": ["IndianFoodPhotos", "IndiaCoffee", "PhotographyIndia", "IndiansRead", "IndiaTech", "GadgetsIndia", "Indiangamers", "XboxIndia", "IndiaPS5", "DesiVideoMemes", "indianmemer", "IndianMeyMeys", "IndianMemeTemplates", "desimemes"],

    "tiny_1": ["AajMaineJana", "Chandigarh", "DesiFragranceAddicts", "DesiKeto", "Fitness_India", "Goa", "HimachalPradesh", "IncredibleIndia", "IndiaNostalgia", "IndiaTrending", "IndianFashionAddicts", "IndianHistory", "IndianMakeupAddicts", "IndianSkincareAddicts", "Indian_flex", "Kerala", "Northeastindia", "Odisha", "SneakersIndia", "SoloTravel_India", "Uttarakhand", "ZyadaKuchNai", "assam", "desitravellers", "gurgaon", "india_tourism", "indiafitcheck", "indianbeautyhauls", "indianfitness", "indiasocial", "watchesindia"],
}

if BATCH_NAME not in BATCH_DEFINITIONS:
    raise ValueError(f"Unknown BATCH_NAME '{BATCH_NAME}'. Valid options: {list(BATCH_DEFINITIONS.keys())}")

SUBREDDITS = BATCH_DEFINITIONS[BATCH_NAME]

ARCTIC_SHIFT_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"

# ==========================================
# TARGET MONTH SELECTION
# ==========================================
target_year = os.getenv("TARGET_YEAR")
target_month = os.getenv("TARGET_MONTH")

if target_year and target_month:
    target_year = int(target_year)
    target_month = int(target_month)
    if not (1 <= target_month <= 12):
        raise ValueError(f"TARGET_MONTH must be 1-12, got {target_month}")
    first_day_last_month = datetime(target_year, target_month, 1)
    first_day_this_month = first_day_last_month + relativedelta(months=1)
    print(f"Using manually specified target month: {first_day_last_month.strftime('%Y-%m')}", flush=True)
else:
    today = datetime.now()
    first_day_this_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    first_day_last_month = first_day_this_month - relativedelta(months=1)
    print(f"No TARGET_YEAR/TARGET_MONTH set -- defaulting to previous calendar month: "
          f"{first_day_last_month.strftime('%Y-%m')}", flush=True)

BEFORE_EPOCH = int(first_day_this_month.timestamp())
AFTER_EPOCH = int(first_day_last_month.timestamp())

# Fixed scratch-space split name (no month in it) -- overwritten every run.
# The consolidation job reads these and folds them into the permanent 'train' split.
SPLIT_NAME = f"tmp_batch_{BATCH_NAME}"

MAX_ATTEMPTS = 2               # one try, one retry -- then move on
HARD_REQUEST_TIMEOUT = 15      # true wall-clock cap per attempt, regardless of trickling data

_executor = ThreadPoolExecutor(max_workers=1)


def get_secure_session():
    return requests.Session()


session = get_secure_session()


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
    caller can abandon this subreddit and move on."""
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


def fetch_subreddit_comments(subreddit, after, before, time_budget_seconds):
    print(f"--- Fetching r/{subreddit} (time budget: {time_budget_seconds:.0f}s) ---", flush=True)
    sub_start_time = time.time()
    all_comments = []
    current_after = after
    page_count = 0

    while True:
        elapsed_this_sub = time.time() - sub_start_time
        if elapsed_this_sub > time_budget_seconds:
            print(f"  [!] r/{subreddit} hit its {time_budget_seconds:.0f}s allocated budget at page "
                  f"{page_count} ({len(all_comments)} collected). Moving on.", flush=True)
            break

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

            if not comments:
                print(f"    r/{subreddit} page {page_count}: 0 new comments (end of range), req took {elapsed:.1f}s", flush=True)
                break

            kept = 0
            for comment in comments:
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

            print(f"    r/{subreddit} page {page_count}: fetched {len(comments)}, kept {kept} "
                  f"(running total {len(all_comments)}), req took {elapsed:.1f}s", flush=True)

            new_after = comments[-1]["created_utc"]
            if new_after == current_after:
                print(f"    [!] Pagination cursor stuck at {new_after} on r/{subreddit} "
                      f"(page {page_count}). Nudging cursor forward by 1s.", flush=True)
                new_after += 1
            current_after = new_after
            time.sleep(1.0)  # reduced wait between pages

        except Exception as e:
            print(f"  [!] Giving up on r/{subreddit} at page {page_count} after {MAX_ATTEMPTS} failed "
                  f"attempts: {e}. Moving on with what was collected so far.", flush=True)
            break

    sub_elapsed = time.time() - sub_start_time
    print(f"Collected {len(all_comments)} comments from r/{subreddit} "
          f"({page_count} pages, {sub_elapsed:.0f}s)", flush=True)
    return all_comments


def push_checkpoint(master_dataset, split_name, label):
    """Pushes to HF with retries. push_to_hub commits to the repo's shared
    main branch -- with multiple matrix jobs pushing concurrently (different
    splits, same branch), two commits can race and one gets rejected with a
    412 Precondition Failed ('branch was updated since you opened this
    page'). That's an expected collision under parallelism, not a real
    error -- retrying against the now-current HEAD resolves it."""
    if not master_dataset:
        print(f"  [checkpoint:{label}] Nothing to push yet, skipping.", flush=True)
        return

    df_chunk = pd.DataFrame(master_dataset).drop_duplicates(subset=["id"])
    dataset = Dataset.from_pandas(df_chunk, features=SCHEMA, preserve_index=False)

    max_push_attempts = 5
    for attempt in range(1, max_push_attempts + 1):
        try:
            print(f"  [checkpoint:{label}] Pushing {len(df_chunk)} rows to split '{split_name}' "
                  f"(attempt {attempt}/{max_push_attempts})...", flush=True)
            dataset.push_to_hub(repo_id=HF_DATASET_REPO, split=split_name, private=True)
            print(f"  [checkpoint:{label}] Push complete.", flush=True)
            return
        except Exception as e:
            # Catches HfHubHTTPError (e.g. 412 branch conflicts) AND lower-level
            # transport errors (httpx.RemoteProtocolError, ConnectError, etc.)
            # that HF's own request layer can throw on a dropped connection --
            # those aren't HfHubHTTPError subclasses, so a narrower except
            # clause would let them crash the job uncaught, as happened here.
            is_conflict = "412" in str(e) or "Precondition Failed" in str(e)
            if attempt < max_push_attempts:
                wait = random.uniform(3, 10) * attempt  # jittered, growing backoff
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
    login(token=HF_TOKEN)

    if not SUBREDDITS:
        print(f"Batch '{BATCH_NAME}' is empty. Nothing to do.", flush=True)
        return

    print(f"Batch '{BATCH_NAME}' covers {len(SUBREDDITS)} subreddits. Target split: '{SPLIT_NAME}'", flush=True)
    print(f"Job time budget: {JOB_TIME_BUDGET_SECONDS}s across {len(SUBREDDITS)} subreddits "
          f"(adaptive per-subreddit allocation, {MIN_SUBREDDIT_SECONDS}s floor)", flush=True)

    master_dataset = []
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

        # Adaptive fair share: divide what's left evenly across what's left,
        # so an early subreddit hogging time automatically shrinks the
        # allowance for the rest, rather than the last subreddit getting zero.
        fair_share = max(remaining_total / remaining_subs, MIN_SUBREDDIT_SECONDS)
        print(f"[time budget] {remaining_total:.0f}s left for {remaining_subs} subs remaining "
              f"-> allocating up to {fair_share:.0f}s to r/{sub}", flush=True)

        sub_comments = fetch_subreddit_comments(sub, AFTER_EPOCH, BEFORE_EPOCH, time_budget_seconds=fair_share)
        master_dataset.extend(sub_comments)
        print(f"[batch progress] {i}/{len(SUBREDDITS)} subreddits done, "
              f"{len(master_dataset)} total rows collected so far in this batch\n", flush=True)

        if i % CHECKPOINT_EVERY == 0 or i == len(SUBREDDITS):
            push_checkpoint(master_dataset, SPLIT_NAME, label=f"{i}/{len(SUBREDDITS)} subs done")

        time.sleep(1.0)  # reduced wait between subreddits

    # Final safety-net push in case the loop broke early due to time budget
    push_checkpoint(master_dataset, SPLIT_NAME, label="final")

    print(f"\nBatch '{BATCH_NAME}' finished. Final size: {len(master_dataset)} comments.", flush=True)


if __name__ == "__main__":
    main()
