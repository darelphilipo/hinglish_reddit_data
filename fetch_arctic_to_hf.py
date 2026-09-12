import requests
import pandas as pd
import time
import os
import random
import json
from datetime import datetime
from datasets import Dataset, Features, Value
from huggingface_hub.errors import HfHubHTTPError
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

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

# Global bounds for randomization and cutoff (2020 to end of 2026)
START_EPOCH_2020 = int(datetime(2020, 1, 1).timestamp())
END_EPOCH_2026 = int(datetime(2026, 12, 31, 23, 59, 59).timestamp())

# ==========================================
# CHECKPOINT MANAGEMENT
# ==========================================
os.makedirs("prompt", exist_ok=True)
CHECKPOINT_FILE = f"prompt/checkpoint_{BATCH_NAME}.json"

def load_checkpoints():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {}

def save_checkpoint(checkpoints_dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoints_dict, f, indent=4)

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
    for cat, subs in categories.items():
        if config.get(cat, 0) == 1:
            active_subs.update(subs)
            
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
    raise ValueError(f"Unknown BATCH_NAME '{BATCH_NAME}'.")
SUBREDDITS = BATCH_DEFINITIONS[BATCH_NAME]
SPLIT_NAME = f"tmp_batch_{BATCH_NAME}"

# Parse environment overrides
max_rows_per_sub = os.getenv("MAX_ROWS_PER_SUB")
max_rows_per_sub = int(max_rows_per_sub) if max_rows_per_sub and max_rows_per_sub.strip() else None

# Execution variables
MAX_ATTEMPTS = 2               
HARD_REQUEST_TIMEOUT = 15      
_executor = ThreadPoolExecutor(max_workers=1)
session = requests.Session()

def _do_request(params):
    return session.get(ARCTIC_SHIFT_URL, params=params, timeout=HARD_REQUEST_TIMEOUT)

def fetch_page_with_retries(params):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time()
        try:
            future = _executor.submit(_do_request, params)
            response = future.result(timeout=HARD_REQUEST_TIMEOUT)
            elapsed = time.time() - t0
            if response.status_code >= 400:
                response.raise_for_status()
            return response.json(), elapsed
        except FutureTimeoutError:
            future.cancel()
            if attempt == MAX_ATTEMPTS: raise
        except Exception as e:
            if attempt == MAX_ATTEMPTS: raise

def fetch_subreddit_comments(subreddit, state_dict, time_budget_seconds, max_rows):
    print(f"--- Fetching r/{subreddit} (time budget: {time_budget_seconds:.0f}s) ---", flush=True)
    sub_start_time = time.time()
    all_comments = []
    page_count = 0
    
    # 1. State Memory & Randomization
    if subreddit in state_dict:
        current_after = state_dict[subreddit]
        resume_date = datetime.fromtimestamp(current_after).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [📍] Historical checkpoint found. Resuming strictly from: {resume_date}", flush=True)
    else:
        # If no checkpoint exists, drop into a random month between 2020 and 2026
        # Leaving a 30-day buffer from the end of 2026
        current_after = random.randint(START_EPOCH_2020, END_EPOCH_2026 - (86400 * 30))
        random_date = datetime.fromtimestamp(current_after).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [🎲] No checkpoint exists. Selecting random start: {random_date}", flush=True)

    while True:
        if (time.time() - sub_start_time) > time_budget_seconds:
            print(f"  [!] Time budget exhausted. Saving cursor.", flush=True)
            break
        if max_rows and len(all_comments) >= max_rows:
            break

        page_count += 1
        params = {
            "subreddit": subreddit,
            "after": current_after,
            "before": END_EPOCH_2026,
            "limit": 100,
            "sort": "asc"
        }

        try:
            data, elapsed = fetch_page_with_retries(params)
            comments = data.get("data", [])

            if not comments:
                print(f"    r/{subreddit}: 0 new comments (Reached end of 2026)", flush=True)
                break

            kept = 0
            for comment in comments:
                if max_rows and len(all_comments) >= max_rows: break
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

            new_after = comments[-1]["created_utc"]
            if new_after == current_after:
                new_after += 1
            current_after = new_after
            time.sleep(1.0) 

        except Exception as e:
            print(f"  [!] Error at page {page_count}: {e}. Halting sub and saving cursor.", flush=True)
            break

    # 2. Update and save checkpoint instantly
    state_dict[subreddit] = current_after
    save_checkpoint(state_dict)
    
    print(f"Collected {len(all_comments)} comments from r/{subreddit} in {time.time() - sub_start_time:.0f}s", flush=True)
    return all_comments

def push_checkpoint(master_dataset, split_name, label):
    if not master_dataset: return
    df_chunk = pd.DataFrame(master_dataset).drop_duplicates(subset=["id"])
    dataset = Dataset.from_pandas(df_chunk, features=SCHEMA, preserve_index=False)
    for attempt in range(1, 6):
        try:
            dataset.push_to_hub(repo_id=HF_DATASET_REPO, split=split_name, private=True)
            return
        except Exception:
            if attempt < 5:
                time.sleep(random.uniform(3, 10) * attempt)
            else:
                raise

def main():
    if not HF_TOKEN: raise ValueError("HF_TOKEN environment variable is not set!")
    if not SUBREDDITS: return

    print(f"Batch '{BATCH_NAME}' initializing...", flush=True)
    state_dict = load_checkpoints()
    master_dataset = []
    job_start_time = time.time()

    for i, sub in enumerate(SUBREDDITS, start=1):
        remaining_total = JOB_TIME_BUDGET_SECONDS - (time.time() - job_start_time)
        remaining_subs = len(SUBREDDITS) - i + 1

        if remaining_total <= MIN_SUBREDDIT_SECONDS: break
        fair_share = max(remaining_total / remaining_subs, MIN_SUBREDDIT_SECONDS)

        sub_comments = fetch_subreddit_comments(
            subreddit=sub, 
            state_dict=state_dict,
            time_budget_seconds=fair_share, 
            max_rows=max_rows_per_sub
        )
        master_dataset.extend(sub_comments)

        if i % CHECKPOINT_EVERY == 0 or i == len(SUBREDDITS):
            push_checkpoint(master_dataset, SPLIT_NAME, label=f"{i}/{len(SUBREDDITS)} subs done")
        time.sleep(1.0) 

    push_checkpoint(master_dataset, SPLIT_NAME, label="final")
    print(f"Batch '{BATCH_NAME}' finished.", flush=True)

if __name__ == "__main__":
    main()
