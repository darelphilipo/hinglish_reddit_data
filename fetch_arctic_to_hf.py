import requests
import pandas as pd
import time
import os
import random
from datetime import datetime
from dateutil.relativedelta import relativedelta
from datasets import Dataset, Features, Value
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
CHECKPOINT_EVERY = 2
BATCH_NAME = os.getenv("BATCH_NAME", "heavy_1")
JOB_TIME_BUDGET_MINUTES = 55
JOB_TIME_BUDGET_SECONDS = JOB_TIME_BUDGET_MINUTES * 60
MIN_SUBREDDIT_SECONDS = 45  
ARCTIC_SHIFT_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"

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
    # Apply historical density sizes per batch
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

target_year_start = parse_env_int("TARGET_YEAR_START")
target_year_end = parse_env_int("TARGET_YEAR_END")
max_rows_per_sub = parse_env_int("MAX_ROWS_PER_SUB")

# ==========================================
# TARGET DATE SELECTION
# ==========================================
if target_year_start and target_year_end:
    # Manual Year Override applied
    first_day_start = datetime(target_year_start, 1, 1)
    first_day_end = datetime(target_year_end + 1, 1, 1) # Capture up to the end of the specified year
    BEFORE_EPOCH = int(first_day_end.timestamp())
    AFTER_EPOCH = int(first_day_start.timestamp())
    print(f"Using manually specified date range override: {target_year_start} to {target_year_end}", flush=True)
else:
    # Standard Month Logic fallback
    target_year = parse_env_int("TARGET_YEAR")
    target_month = parse_env_int("TARGET_MONTH")
    if target_year and target_month:
        if not (1 <= target_month <= 12):
            raise ValueError(f"TARGET_MONTH must be 1-12, got {target_month}")
        first_day_last_month = datetime(target_year, target_month, 1)
        first_day_this_month = first_day_last_month + relativedelta(months=1)
        print(f"Using environment variable target month: {first_day_last_month.strftime('%Y-%m')}", flush=True)
    else:
        today = datetime.now()
        first_day_this_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        first_day_last_month = first_day_this_month - relativedelta(months=1)
        print(f"No date overrides set -- defaulting to previous calendar month: "
              f"{first_day_last_month.strftime('%Y-%m')}", flush=True)
              
    BEFORE_EPOCH = int(first_day_this_month.timestamp())
    AFTER_EPOCH = int(first_day_last_month.timestamp())

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

def fetch_subreddit_comments(subreddit, after, before, time_budget_seconds, max_rows):
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
            
        if max_rows and len(all_comments) >= max_rows:
            print(f"  [!] r/{subreddit} reached maximum requested rows ({max_rows}). Moving on.", flush=True)
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
                # Stop processing if max_rows boundary is hit mid-page
                if max_rows and len(all_comments) >= max_rows:
                    break
                    
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
            time.sleep(1.0) 

        except Exception as e:
            print(f"  [!] Giving up on r/{subreddit} at page {page_count} after {MAX_ATTEMPTS} failed "
                  f"attempts: {e}. Moving on with what was collected so far.", flush=True)
            break

    sub_elapsed = time.time() - sub_start_time
    print(f"Collected {len(all_comments)} comments from r/{subreddit} "
          f"({page_count} pages, {sub_elapsed:.0f}s)", flush=True)
    return all_comments

def push_checkpoint(master_dataset, split_name, label):
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
    print(f"Job time budget: {JOB_TIME_BUDGET_SECONDS}s across {len(SUBREDDITS)} subreddits "
          f"(adaptive per-subreddit allocation, {MIN_SUBREDDIT_SECONDS}s floor)", flush=True)
    if max_rows_per_sub:
        print(f"Row limit override active: Fetching up to {max_rows_per_sub} comments per subreddit.", flush=True)

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

        fair_share = max(remaining_total / remaining_subs, MIN_SUBREDDIT_SECONDS)
        print(f"[time budget] {remaining_total:.0f}s left for {remaining_subs} subs remaining "
              f"-> allocating up to {fair_share:.0f}s to r/{sub}", flush=True)

        sub_comments = fetch_subreddit_comments(
            subreddit=sub, 
            after=AFTER_EPOCH, 
            before=BEFORE_EPOCH, 
            time_budget_seconds=fair_share, 
            max_rows=max_rows_per_sub
        )
        master_dataset.extend(sub_comments)
        print(f"[batch progress] {i}/{len(SUBREDDITS)} subreddits done, "
              f"{len(master_dataset)} total rows collected so far in this batch\n", flush=True)

        if i % CHECKPOINT_EVERY == 0 or i == len(SUBREDDITS):
            push_checkpoint(master_dataset, SPLIT_NAME, label=f"{i}/{len(SUBREDDITS)} subs done")

        time.sleep(1.0) 

    push_checkpoint(master_dataset, SPLIT_NAME, label="final")
    print(f"\nBatch '{BATCH_NAME}' finished. Final size: {len(master_dataset)} comments.", flush=True)

if __name__ == "__main__":
    main()
