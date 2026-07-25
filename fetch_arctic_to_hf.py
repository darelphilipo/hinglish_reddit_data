import requests
import pandas as pd
import time
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from datasets import Dataset
from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# ==========================================
# CONFIGURATION
# ==========================================
HF_DATASET_REPO = "darelphilip/reddit_indian_subs"  # CHANGE THIS
HF_TOKEN = os.getenv("HF_TOKEN")

# Push a checkpoint split to HF every N subreddits *within* this batch,
# so a failure only loses the in-flight chunk, not the whole batch.
# Kept small since failures now move on quickly rather than retrying at length.
CHECKPOINT_EVERY = 3

# Which slice of SUBREDDITS this job processes. Defaults to the full list
# so the script still works fine for local/manual full runs.
BATCH_START = int(os.getenv("BATCH_START", "0"))
BATCH_END = int(os.getenv("BATCH_END", "999999"))

SUBREDDITS_FULL = [
    # National / General
    "india", "indiasocial", "AskIndia", "IndiaNostalgia", "IndiaTrending", "IncredibleIndia", "IndianHistory", "AajMaineJana", "ZyadaKuchNai",
    # Cities / States
    "delhi", "bangalore", "mumbai", "chennai", "hyderabad", "Kerala", "kolkata", "TamilNadu", "pune", "Maharashtra", "bihar", "ahmedabad", "lucknow", "Goa", "Uttarakhand", "assam", "gurgaon", "karnataka", "Rajasthan", "HimachalPradesh", "Chandigarh", "gujarat", "Odisha", "uttarpradesh", "Northeastindia",
    # Political
    "IndiaSpeaks", "unitedstatesofindia", "indianews", "indiadiscussion", "CriticalThinkingIndia",
    # Entertainment
    "BollyBlindsNGossip", "bollywood", "InstaCelebsGossip", "bollywoodmemes", "BollywoodFashion", "AnimeMirchi", "animeindian", "BollywoodMusic", "IndianOTTbestof", "MalayalamMovies", "kollywood", "IndianHipHopHeads", "tollywood", "BollywoodRealism", "IndianCinema", "sharktankindia", "biggboss", "IndianTellyTalk", "DHHMemes", "punjabimusic",
    # Sports
    "IndiaCricket", "ipl", "CricketShitpost", "indiansports", "IndianFootball", "RCB", "csk", "chessindia",
    # Social / Demographic
    "RelationshipIndia", "TwoXIndia", "TeenIndia", "IndianTeenagers", "Indiangirlsontinder", "AskIndianWomen", "DesiWeddings", "TwentiesIndia", "OffMyChestIndia", "Arrangedmarriage", "AskIndianMen",
    # Youtubers
    "SaimanSays", "CarryMinati", "ShahRukhKhan", "SamayRaina", "thugeshh", "beastboyshub", "sunraybee", "FingMemes", "dankrishu", "ViratKohli",
    # Finance
    "IndianStockMarket", "IndiaInvestments", "personalfinanceindia", "IndianStreetBets", "beermoneyindia", "CreditCardsIndia", "CryptoIndia", "FIREIndia", "StockMarketIndia", "IndiaTax", "mutualfunds", "BitcoinIndia", "FatFIREIndia", "IndianStocks", "Frugal_Ind",
    # Career / Academics
    "developersIndia", "JEENEETards", "UPSC", "StartUpIndia", "CBSE", "Btechtards", "indianmedschool", "CATpreparation", "Indian_Academia", "JEE", "IndiaCareers", "BITSPilani", "Indians_StudyAbroad", "IndianWorkplace", "ICSE", "CharteredAccountants", "IndiaBusiness", "smallbusinessindia",
    # Lifestyle / Travel
    "IndianFashionAddicts", "IndianSkincareAddicts", "DesiFragranceAddicts", "desitravellers", "Fitness_India", "watchesindia", "india_tourism", "IndianMakeupAddicts", "indianbeautyhauls", "SneakersIndia", "Indian_flex", "DesiKeto", "SoloTravel_India", "indiafitcheck", "indianfitness",
    # Auto / Art / Tech / Memes
    "CarsIndia", "indianrailways", "indianbikes", "AirTravelIndia", "Indianbooks", "IndianArtAndThinking", "indiafood", "IndianArtAI", "hindi", "IndianFoodPhotos", "IndiaCoffee", "PhotographyIndia", "IndiansRead", "IndiaTech", "IndianGaming", "GadgetsIndia", "Indiangamers", "XboxIndia", "IndiaPS5", "indiameme", "funnyIndia", "IndianDankMemes", "DesiVideoMemes", "indianmemer", "IndianMeyMeys", "IndianMemeTemplates", "desimemes"
]

# The slice this specific job/batch is responsible for
SUBREDDITS = SUBREDDITS_FULL[BATCH_START:BATCH_END]

ARCTIC_SHIFT_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"

# ==========================================
# TARGET MONTH SELECTION
# ==========================================
# Defaults to the previous calendar month (for the normal scheduled run).
# Override by setting TARGET_YEAR and TARGET_MONTH env vars, e.g. to backfill
# a specific historical month (TARGET_YEAR=2025, TARGET_MONTH=3 for March 2025).
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

# Fixed scratch-space split name (no month in it) -- this gets overwritten
# every run. It's purely a staging area the consolidation job reads from
# and folds into the single permanent 'train' split. This also means we
# never need to delete old splits: each month's run just overwrites the
# same 9 scratch splits.
SPLIT_NAME = f"tmp_batch_{BATCH_START:03d}_{min(BATCH_END, len(SUBREDDITS_FULL)):03d}"


def get_secure_session():
    """Returns a plain requests Session. Retries are now handled manually
    inside fetch_subreddit_comments so every attempt is logged -- urllib3's
    built-in Retry sleeps silently between attempts, which is what was
    causing long stretches with no visible output."""
    session = requests.Session()
    return session


session = get_secure_session()

MAX_ATTEMPTS = 2               # one try, one retry -- then move on, don't chase it
HARD_REQUEST_TIMEOUT = 15      # true wall-clock cap per attempt, regardless of trickling data

_executor = ThreadPoolExecutor(max_workers=1)


def _do_request(params):
    """The actual blocking network call, run in a worker thread so we can
    enforce a true total-duration timeout around it. requests' own `timeout`
    kwarg only resets on each new byte received, so a slow-trickling
    response can hang far longer than the value you pass it."""
    response = session.get(ARCTIC_SHIFT_URL, params=params, timeout=HARD_REQUEST_TIMEOUT)
    response.raise_for_status()
    return response


def fetch_page_with_retries(params):
    """Fetches a single page. One attempt, capped at HARD_REQUEST_TIMEOUT
    seconds. On failure, one retry with the same cap. If that also fails,
    raises so the caller can abandon this subreddit and move on -- data
    collection matters more than squeezing every last page out of a
    struggling subreddit."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time()
        try:
            future = _executor.submit(_do_request, params)
            response = future.result(timeout=HARD_REQUEST_TIMEOUT)
            elapsed = time.time() - t0
            return response.json(), elapsed
        except FutureTimeoutError:
            elapsed = time.time() - t0
            future.cancel()  # best-effort; underlying request may keep running in the worker thread
            print(f"    [!] attempt {attempt}/{MAX_ATTEMPTS} timed out after {elapsed:.1f}s "
                  f"(no response within {HARD_REQUEST_TIMEOUT}s).", flush=True)
            if attempt == MAX_ATTEMPTS:
                raise
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    [!] attempt {attempt}/{MAX_ATTEMPTS} failed after {elapsed:.1f}s: {e}", flush=True)
            if attempt == MAX_ATTEMPTS:
                raise


def fetch_subreddit_comments(subreddit, after, before):
    print(f"--- Fetching r/{subreddit} ---", flush=True)
    sub_start_time = time.time()
    all_comments = []
    current_after = after
    page_count = 0

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

            if not comments:
                print(f"    r/{subreddit} page {page_count}: 0 new comments (end of range), req took {elapsed:.1f}s", flush=True)
                break

            kept = 0
            for comment in comments:
                body = comment.get("body", "")

                # Skip strictly deleted/removed content to save DB space
                if body and body not in ["[removed]", "[deleted]"]:
                    kept += 1
                    # Keep exactly the columns requested
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
                # Pagination cursor didn't advance -- would loop forever otherwise.
                print(f"    [!] Pagination cursor stuck at {new_after} on r/{subreddit} "
                      f"(page {page_count}). Nudging cursor forward by 1s to avoid an infinite loop.", flush=True)
                new_after += 1
            current_after = new_after
            time.sleep(1.2)  # Crucial sleep to avoid IP bans from Arctic Shift

        except Exception as e:
            print(f"  [!] Giving up on r/{subreddit} at page {page_count} after {MAX_ATTEMPTS} failed "
                  f"attempts: {e}. Moving on with what was collected so far.", flush=True)
            break

    sub_elapsed = time.time() - sub_start_time
    print(f"Collected {len(all_comments)} comments from r/{subreddit} "
          f"({page_count} pages, {sub_elapsed:.0f}s)", flush=True)
    return all_comments


def push_checkpoint(master_dataset, split_name, label):
    """Push whatever has been collected so far under this batch's split name.
    Safe to call repeatedly -- each call re-pushes the full in-memory
    dataset for this batch, deduped, so a later checkpoint simply
    supersedes an earlier one for the same split."""
    if not master_dataset:
        print(f"  [checkpoint:{label}] Nothing to push yet, skipping.", flush=True)
        return

    df_chunk = pd.DataFrame(master_dataset).drop_duplicates(subset=["id"])
    print(f"  [checkpoint:{label}] Pushing {len(df_chunk)} rows to split '{split_name}'...", flush=True)
    dataset = Dataset.from_pandas(df_chunk, preserve_index=False)
    dataset.push_to_hub(repo_id=HF_DATASET_REPO, split=split_name, private=True)
    print(f"  [checkpoint:{label}] Push complete.", flush=True)


def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set!")
    login(token=HF_TOKEN)

    if not SUBREDDITS:
        print(f"Batch range [{BATCH_START}:{BATCH_END}] is empty against a list of "
              f"{len(SUBREDDITS_FULL)} subreddits. Nothing to do.", flush=True)
        return

    print(f"Batch covers {len(SUBREDDITS)} subreddits (indices {BATCH_START}:{BATCH_END} "
          f"of {len(SUBREDDITS_FULL)} total). Target split: '{SPLIT_NAME}'", flush=True)

    master_dataset = []

    for i, sub in enumerate(SUBREDDITS, start=1):
        sub_comments = fetch_subreddit_comments(sub, AFTER_EPOCH, BEFORE_EPOCH)
        master_dataset.extend(sub_comments)
        print(f"[batch progress] {i}/{len(SUBREDDITS)} subreddits done, "
              f"{len(master_dataset)} total rows collected so far in this batch\n", flush=True)

        should_checkpoint = (i % CHECKPOINT_EVERY == 0) or (i == len(SUBREDDITS))
        if should_checkpoint:
            push_checkpoint(master_dataset, SPLIT_NAME, label=f"{i}/{len(SUBREDDITS)} subs done")

        time.sleep(1.5)  # Pause between subreddits

    print(f"\nBatch finished. Final size for this batch: {len(master_dataset)} comments "
          f"across {len(SUBREDDITS)} subreddits.", flush=True)


if __name__ == "__main__":
    main()
