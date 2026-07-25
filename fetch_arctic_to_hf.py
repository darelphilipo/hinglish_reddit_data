import requests
import pandas as pd
import time
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from datasets import Dataset
from huggingface_hub import login
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================
# CONFIGURATION
# ==========================================
HF_DATASET_REPO = "your-hf-username/hinglish-moderation-raw" # CHANGE THIS
HF_TOKEN = os.getenv("HF_TOKEN")

SUBREDDITS = [
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

ARCTIC_SHIFT_URL = "https://arctic-shift.xk.io/api/comments/search"

# Calculate timestamps for the previous calendar month
today = datetime.now()
first_day_this_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
first_day_last_month = first_day_this_month - relativedelta(months=1)
BEFORE_EPOCH = int(first_day_this_month.timestamp())
AFTER_EPOCH = int(first_day_last_month.timestamp())

def get_secure_session():
    """Returns a requests Session with built-in retry logic to survive API hiccups."""
    session = requests.Session()
    retry = Retry(
        total=10,
        read=10,
        connect=10,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

session = get_secure_session()

def fetch_subreddit_comments(subreddit, after, before):
    print(f"--- Fetching r/{subreddit} ---")
    all_comments = []
    current_after = after
    
    while True:
        params = {
            "subreddit": subreddit,
            "after": current_after,
            "before": before,
            "limit": 100,
            "sort": "asc"
        }
        
        try:
            response = session.get(ARCTIC_SHIFT_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            comments = data.get("data", [])
            
            if not comments:
                break
                
            for comment in comments:
                body = comment.get("body", "")
                
                # Skip strictly deleted/removed content to save DB space
                if body and body not in ["[removed]", "[deleted]"]:
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
            
            current_after = comments[-1]["created_utc"]
            time.sleep(1.2) # Crucial sleep to avoid IP bans from Arctic Shift
            
        except Exception as e:
            print(f"  [!] Error on r/{subreddit}: {e}. Skipping remainder of this sub.")
            break 
            
    print(f"Collected {len(all_comments)} comments from r/{subreddit}")
    return all_comments

def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set!")
    login(token=HF_TOKEN)
    
    master_dataset = []
    
    # Loop through all subreddits
    for sub in SUBREDDITS:
        sub_comments = fetch_subreddit_comments(sub, AFTER_EPOCH, BEFORE_EPOCH)
        master_dataset.extend(sub_comments)
        time.sleep(1.5) # Pause between subreddits
        
    if not master_dataset:
        print("No valid data fetched across any subreddits. Exiting.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(master_dataset)
    df = df.drop_duplicates(subset=["id"])
    print(f"\nFinal dataset size: {len(df)} total comments across {len(SUBREDDITS)} subreddits.")
    
    # Push to Hugging Face Hub
    dataset = Dataset.from_pandas(df)
    
    # Naming convention: raw_YYYY_MM
    split_name = f"raw_{first_day_last_month.strftime('%Y_%m')}"
    
    print(f"Pushing dataset to Hugging Face as split: '{split_name}'...")
    dataset.push_to_hub(repo_id=HF_DATASET_REPO, split=split_name, private=True)
    print("✅ Successfully pushed to Hugging Face Hub!")

if __name__ == "__main__":
    main()
