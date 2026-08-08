# @title 🎯 Hunter-Seeker MLOps Pipeline (Targeting Caste & Misogyny)
!pip install -q pandas tqdm openai duckdb huggingface_hub

import pandas as pd
import duckdb
import os
import time
import json
import threading
import html
import re
from huggingface_hub import HfApi
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm
from openai import OpenAI

try:
    from google.colab import userdata, drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ==========================================
# 1. HUNTER CONFIGURATION
# ==========================================
if IN_COLAB:
    drive.mount('/content/drive', force_remount=False)

OUTPUT_DIR = '/content/drive/MyDrive/Hinglish_Classifier_Project/experiments/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Separate output file so we don't overwrite your baseline!
FINAL_CSV_PATH = os.path.join(OUTPUT_DIR, 'hunter_caste_misogyny_5k.csv')
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'hunter_checkpoint.csv')

TARGET_SUBREDDITS = [
    'chodi', 'bakchodi', 'sham_sharma_show', 'desimeta',
    'indiandankmemes', 'dankinindia', 'saimansays', 'librandu',
    'unitedstatesofindia', 'indiadiscussion', 'canconfirmiamindian',
    'arrangedmarriage', 'bollyblindsngossip'
]

START_EPOCH = 1483228800 # 2017
END_EPOCH = 1593561599   # Mid-2020
HUNTER_LIMIT = 5000      # How many highly-toxic rows we want to extract
MAX_WORKERS = 10 

if IN_COLAB:
    try: OPENCODE_KEY = userdata.get("opencode")
    except: OPENCODE_KEY = userdata.get("OPENCODE_KEY")
else:
    OPENCODE_KEY = os.getenv("OPENCODE_KEY")

client = OpenAI(api_key=OPENCODE_KEY, base_url="https://opencode.ai/zen/go/v1")
MODEL_NAME = "deepseek-v4-flash"

# ==========================================
# 2. DUCKDB SURGICAL EXTRACTION & TELEMETRY
# ==========================================
import psutil

print("\n🦆 Initializing DuckDB Hunter-Seeker...")
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

# 🛡️ Enhanced Memory & Concurrency Safeguards
con.execute("PRAGMA memory_limit='4GB';") 
con.execute("PRAGMA threads=2;") # Throttle parquet decompression overhead
con.execute("PRAGMA preserve_insertion_order=False;") # Saves RAM during massive scans
con.execute("PRAGMA enable_progress_bar;")
con.execute("PRAGMA enable_print_progress_bar;")

if IN_COLAB:
    try:
        HF_TOKEN = userdata.get('HF_TOKEN')
        con.execute(f"CREATE SECRET hf_auth (TYPE HUGGINGFACE, TOKEN '{HF_TOKEN}');")
    except Exception as e:
        print(f"⚠️ Could not load HF_TOKEN: {e}")

api = HfApi()
try:
    all_files = api.list_repo_files("open-index/arctic", repo_type="dataset")
    target_era_files = [
        f for f in all_files if f.endswith('.parquet') and 
        ('data/comments/2017' in f or 'data/comments/2018' in f or 
         'data/comments/2019' in f or 'data/comments/2020' in f)
    ]
    random.seed(99) 
    selected_shards = random.sample(target_era_files, min(150, len(target_era_files)))
    hf_urls = [f"hf://datasets/open-index/arctic/{f}" for f in selected_shards]
except Exception as e:
    raise RuntimeError(f"❌ HF API Error: {e}")

subs_formatted = ", ".join([f"'{s.lower()}'" for s in TARGET_SUBREDDITS])

# 🚨 THE HUNTER QUERY 🚨
query = f"""
SELECT 
    id,
    body,
    LOWER(subreddit) as subreddit,
    created_utc
FROM read_parquet({hf_urls})
WHERE LOWER(subreddit) IN ({subs_formatted})
  AND created_utc >= {START_EPOCH}
  AND created_utc <= {END_EPOCH}
  AND body NOT IN ('[deleted]', '[removed]', '')
  AND length(body) BETWEEN 10 AND 1000
  AND (
      body ILIKE '%dalit%' OR body ILIKE '%brahmin%' OR body ILIKE '%chamar%' OR 
      body ILIKE '%bhangi%' OR body ILIKE '%neech%' OR body ILIKE '%chura%' OR 
      body ILIKE '%quota%' OR body ILIKE '%untouchable%' OR
      body ILIKE '%randi%' OR body ILIKE '%chhakka%' OR body ILIKE '%hijra%' OR 
      body ILIKE '%feminazi%' OR body ILIKE '%slut%' OR body ILIKE '%whore%' OR 
      body ILIKE '%faggot%' OR body ILIKE '%gay%' OR
      body ILIKE '%bihari%' OR body ILIKE '%bimaru%' OR body ILIKE '%nigger%' OR 
      body ILIKE '%madrasi%' OR body ILIKE '%pajeet%'
  )
LIMIT {HUNTER_LIMIT}
"""

# 📊 Background Telemetry Monitor
def monitor_system(stop_event):
    while not stop_event.is_set():
        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        print(f"   [⚙️ System Status] RAM: {ram.used / (1024**3):.2f}GB / {ram.total / (1024**3):.2f}GB ({ram.percent}%) | CPU: {cpu}%")
        time.sleep(5)

stop_monitor = threading.Event()
monitor_thread = threading.Thread(target=monitor_system, args=(stop_monitor,))

try:
    print(f"⏳ Hunting explicit Caste, Misogyny, and Regional keywords...")
    # Start the live telemetry output
    monitor_thread.start()
    
    # Execute the heavy query
    raw_df = con.query(query).to_df()
    
    # Stop the telemetry once DuckDB survives the conversion
    stop_monitor.set()
    monitor_thread.join()
    
    print(f"📦 Pulled {len(raw_df)} high-probability toxic comments.")
except Exception as e:
    stop_monitor.set()
    monitor_thread.join()
    raise RuntimeError(f"❌ DuckDB Extraction failed: {e}")

df = raw_df.copy()

import gc
del raw_df
gc.collect()

# ==========================================
# 3. TEXT SANITIZATION
# ==========================================
def sanitize_text(text):
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u200b', '')
    return text.strip()

if not df.empty:
    print("\n🧹 Sanitizing text to optimize tokens...")
    tqdm.pandas(desc="Sanitization")
    df['body_clean'] = df['body'].progress_apply(sanitize_text)
    df = df[df['body_clean'].str.len() > 0].reset_index(drop=True)
    df['temp_id'] = df.index.astype(str)

# ==========================================
# 4. TEACHER MODEL PROMPT & ENGINE
# ==========================================
# (Keep your exact Version 6.5 Prompt string here. I have collapsed it for brevity, but paste the full v6.5 string in your actual notebook)
SYSTEM_PROMPT = """# Hinglish & English Content Moderation — Teacher Model Labeling Prompt
Version: 6.5 (Batch Array + Full v6.2 English Coverage Restored) | Hindi-English code-mixed and English, Roman script 

Philosophy: When in doubt on harassment/hate speech, output 0. Silencing legitimate speech is worse than missing an edge case. Never flag opinions, criticism, or political disagreement as harassment or hate speech.

Design note on pv (profanity_vulgarity): this flag is a detection signal, not a moderation verdict. It fires whenever explicit profanity is present, regardless of target.

--- 

## OUTPUT FORMAT 
Return ONLY a valid JSON object containing an array under the key "results", no preamble.
CRITICAL: You MUST put 'analysis' first to force logical evaluation before scoring.
Use exactly these short keys for every object to save tokens:

{
  "results": [
    {"id": "<id>", "analysis": "<1 brief sentence analyzing target and intent>", "pv": 0, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0}
  ]
}

Key Legend:
pv = profanity_vulgarity | tah = targeted_abuse_harassment | dhs = discriminatory_hate_speech
cst = caste | cr = communal_religious | rx = regional_xenophobic | mg = misogyny_gender
`dhs` = 1 iff any dimension (cst, cr, rx, mg) = 1. If 0, all dimensions = 0.
The three top-level flags are independent — any combination is valid, including all three at once.

--- 

## THE DECISION PROCEDURE 

Step 1 — Is explicit profanity/vulgarity present anywhere in the text? 
Check for curse words, obscene terms, sexual slurs, or vulgar swearing — Hindi/Hinglish (chutiya, bsdk, mc, bc, gaand, lund, randi, madarchod, behenchod, bhosadike, gandu, harami, kamine) and English (fuck, shit, bitch, cunt, asshole, bastard, motherfucker, whore, slut, dick, prick, retard/retarded) — including obfuscated variants. 
- Present anywhere — even filler, self-directed, or aimed at an abstract target → pv: 1. Otherwise 0.

Step 2 — Is there targeted personal abuse/harassment OR an identity-based stereotype/objectification/hierarchy statement present? 
- Aimed at an abstract entity (government, system, company, policy) → tah: 0 
- Self-directed or reported speech → 0 
- Public figure reverse test: Personal (appearance/character) → 1. Professional (policy) → 0. 
- A specific person insulted, threatened, or personally degraded → tah: 1 

Step 3 — Is an identity dimension (caste/religion/region/gender) load-bearing? 
- Yes → dhs: 1 + correct dimension(s). Applies even with NO profanity present — covert othering, rhetorical questions, mocking religious practices, and conspiracy framing in Hindi or English (e.g. "go back to Pakistan").
- No, generic insult → identity dimensions stay 0. 

Step 4 — Political terms (sanghi/bhakt/libtard) are NOT an identity dimension on their own.

--- 

## CATEGORY DEFINITIONS 

- Caste (cst): chamar, bhangi, neech jaati, dedh, chura, pallan, kanjar, dhobi, mehtar, "quota-wallah", "lower caste" (as insult).
- Communal (cr): katwa, mulla/mulle, kafir, malaun, pajeet, "cow piss drinker", "dung worshipper", "sanghi/bhakt" (only when dehumanizing), "terrorist"/"jihadi" (blanket-smear).
- Regional (rx): bihari/madrasi (pejorative), bimaru, chinki, ghati, bhaiyya, "Porki", "Paki", "chink", "bloody Bihari/Madrasi", "these North/South Indians" (dismissively).
- Gendered (mg): chhakka/hijra, "maal"/"item", "randi khana", "feminazi", "slut", "whore", "asking for it", "women should stay home".

--- 

## KNOWN TRAPS

1. Homonyms: chakka = jackfruit vs slur. BC = date vs profanity.
2. English Homonyms: "dick" or "cock" only trigger pv:1 when used as anatomical vulgarity/insults. Proper names (Dick) get pv:0.
3. India-Pakistan: criticizing Pakistani govt = 0. Blanket-dehumanizing Pakistanis or Indian Muslims = 1 on dhs.
4. Code-switching: Evaluate multi-lingual comments as a single unit.

## ANCHOR EXAMPLES 
[
  {"id": "ex1", "analysis": "English profanity directed at institutional target.", "pv": 1, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex2", "analysis": "Hinglish hate speech targeting religion.", "pv": 1, "tah": 1, "dhs": 1, "cst": 0, "cr": 1, "rx": 0, "mg": 0},
  {"id": "ex3", "analysis": "English xenophobic regional trope.", "pv": 0, "tah": 0, "dhs": 1, "cst": 0, "cr": 0, "rx": 1, "mg": 0},
  {"id": "ex4", "analysis": "English gender hierarchy statement.", "pv": 0, "tah": 0, "dhs": 1, "cst": 0, "cr": 0, "rx": 0, "mg": 1}
]
"""

def label_batch(comments_batch, attempt=1):
    numbered = "\n".join(f'ID: {cid} | Comment: {body}' for cid, body in comments_batch)
    user_prompt = f"Label these {len(comments_batch)} comments strictly following the JSON schema:\n{numbered}"

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            temperature=0.1,
            max_tokens=2500,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}}
        )
        
        raw_content = res.choices[0].message.content.strip()
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
            raw_content = re.sub(r"\n?```$", "", raw_content).strip()

        try:
            parsed_data = json.loads(raw_content)
        except json.JSONDecodeError:
            match = re.search(r'\[\s*\{.*?\}\s*\]', raw_content, re.DOTALL)
            if match: parsed_data = json.loads(match.group(0))
            else: raise ValueError("JSON parse error")
        
        if isinstance(parsed_data, list): parsed_array = parsed_data
        elif isinstance(parsed_data, dict):
            parsed_array = parsed_data.get("results", [])
            if not parsed_array:
                for val in parsed_data.values():
                    if isinstance(val, list):
                        parsed_array = val
                        break
        else: parsed_array = []
        
        if len(parsed_array) == len(comments_batch):
            for idx, item in enumerate(parsed_array):
                item["temp_id"] = str(comments_batch[idx][0])
            return parsed_array
            
        raise ValueError("Batch mismatch")
        
    except Exception:
        if attempt <= 4:
            time.sleep(min(2 ** attempt, 10))
            return label_batch(comments_batch, attempt + 1)
        return []

# ==========================================
# 5. PIPELINE EXECUTION
# ==========================================
def run_experiment():
    if df.empty:
        print("❌ No data available to label. Stopping pipeline.")
        return

    batches = [list(zip(df["temp_id"], df["body_clean"]))[i:i + 20] for i in range(0, len(df), 20)]
    all_labels = []
    lock = threading.Lock()

    print(f"\n🚀 Running Hunter Inference on {len(df)} rows...")
    
    with tqdm(total=len(batches), desc="Labeling Targets") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(label_batch, batch) for batch in batches]
            for future in as_completed(futures):
                with lock:
                    all_labels.extend(future.result())
                pbar.update(1)

    labels_df = pd.DataFrame(all_labels)
    if not labels_df.empty:
        labels_df["temp_id"] = labels_df["temp_id"].astype(str)
        KEY_MAPPING = {
            "analysis": "step_by_step_analysis", "pv": "profanity_vulgarity", "tah": "targeted_abuse_harassment",
            "dhs": "discriminatory_hate_speech", "cst": "caste", "cr": "communal_religious", 
            "rx": "regional_xenophobic", "mg": "misogyny_gender"
        }
        labels_df.rename(columns=KEY_MAPPING, inplace=True)
        final_df = df.merge(labels_df, on="temp_id", how="left").drop(columns=["temp_id", "body_clean", "pv", "tah", "dhs", "cst", "cr", "rx", "mg"], errors='ignore')
        final_df.to_csv(FINAL_CSV_PATH, index=False)
        print(f"\n✅ HUNTER COMPLETE! Saved to: {FINAL_CSV_PATH}")
        analyze_distribution(final_df)

def analyze_distribution(rdf):
    print("\n--- 🎯 Top-Level & Sub-Category Breakdowns ---")
    total = len(rdf)
    flags = [
        ('Caste (cst)', 'caste'),
        ('Misogyny (mg)', 'misogyny_gender'),
        ('Regional (rx)', 'regional_xenophobic')
    ]
    for name, col in flags:
        if col in rdf.columns:
            cnt = rdf[col].sum()
            print(f"{name:<24}: {cnt:>4} ({(cnt/total)*100:>5.1f}%)")

run_experiment()
