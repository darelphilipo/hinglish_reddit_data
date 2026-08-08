# @title
!pip install -q duckdb pandas tqdm huggingface_hub openai

import duckdb
import pandas as pd
import os
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm
from huggingface_hub import HfApi
from openai import OpenAI

try:
    from google.colab import userdata, drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ==========================================
# 1. SETUP & API CONFIG (Synchronous)
# ==========================================
if IN_COLAB:
    print("🔗 Mounting Google Drive...")
    drive.mount('/content/drive', force_remount=False)
    WORK_DIR = '/content/drive/MyDrive/Hinglish_Classifier_Project/chodi_test'
else:
    WORK_DIR = './chodi_test'

os.makedirs(WORK_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(WORK_DIR, "chodi_labeled.csv")
CHECKPOINT_FILE = os.path.join(WORK_DIR, "chodi_checkpoint.csv")

TARGET_SUBREDDIT = "chodi"
TOTAL_ROWS_TO_LABEL = 500

print("🔌 Connecting to OpenCode Go API (Sync Mode)...")
if IN_COLAB:
    try: OPENCODE_KEY = userdata.get("opencode")
    except: OPENCODE_KEY = userdata.get("OPENCODE_KEY")
else:
    OPENCODE_KEY = os.getenv("OPENCODE_KEY")

# Standard Synchronous OpenAI Client
client = OpenAI(api_key=OPENCODE_KEY, base_url="https://opencode.ai/zen/go/v1")
MODEL_NAME = "deepseek-v4-flash"

# ==========================================
# 2. DUCKDB LIGHTNING EXTRACTION (Anti-Hang Fix)
# ==========================================
print(f"\n🕵️ Fetching exact file paths via HF API to prevent metadata hang...")
api = HfApi()

try:
    import random
    all_files = api.list_repo_files("open-index/arctic", repo_type="dataset")
    target_era_files = [
        f for f in all_files 
        if f.endswith('.parquet') and ('data/comments/2020' in f or 'data/comments/2021' in f)
    ]
    random.seed(42)
    selected_shards = random.sample(target_era_files, min(50, len(target_era_files)))
    hf_urls = [f"hf://datasets/open-index/arctic/{f}" for f in selected_shards]
    print(f"✅ Locked onto {len(hf_urls)} explicit data shards. Booting DuckDB...")
except Exception as e:
    raise RuntimeError(f"❌ Failed to communicate with Hugging Face: {e}")

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

fetch_start = time.time()
query = f"""
SELECT 
    subreddit, body, score,
    0 AS pv, 0 AS tah, 0 AS dhs, 0 AS cst, 0 AS cr, 0 AS rx, 0 AS mg
FROM read_parquet({hf_urls})
WHERE LOWER(subreddit) = '{TARGET_SUBREDDIT}'
  AND body NOT IN ('[deleted]', '[removed]', '')
  AND body IS NOT NULL
  AND length(body) BETWEEN 30 AND 500
  AND score >= 2
LIMIT {TOTAL_ROWS_TO_LABEL};
"""

try:
    df = con.query(query).to_df()
    if df.empty: raise ValueError(f"🚨 Found 0 rows. Try increasing shard sample size.")
    print(f"✅ Fast-fetch complete! Downloaded {len(df)} rows in {time.time() - fetch_start:.2f}s")
except Exception as e:
    raise RuntimeError(f"❌ DuckDB Extraction failed: {e}")

import html
import re

def sanitize_reddit_text(text):
    if not isinstance(text, str):
        return ""

    # 1. Decode HTML entities (&gt; becomes >, &amp; becomes &)
    text = html.unescape(text)

    # 2. Extract text from Markdown links [Click Here](http://...) -> Click Here
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # 3. Strip all remaining raw URLs (http://..., https://..., www...)
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # 4. Remove excessive line breaks and multiple spaces (squash to single space)
    text = re.sub(r'\s+', ' ', text)

    # 5. Remove Zero-Width spaces (often used by trolls to bypass auto-mods)
    text = text.replace('\u200b', '')

    # 6. Strip leading/trailing whitespace
    return text.strip()

# Apply the sanitization to your dataframe
print("🧹 Sanitizing Reddit text to optimize tokens...")
df['body'] = df['body'].apply(sanitize_reddit_text)

# Drop any rows that became completely empty after removing URLs
df = df[df['body'].str.len() > 0].reset_index(drop=True)

# ==========================================
# 3. TEACHER MODEL PROMPT & SYNC FUNCTION
# ==========================================
SYSTEM_PROMPT = """Hinglish Content Moderation — Teacher Model Labeling Prompt
Version: 6.1 (Decoupled 3-Flag Architecture, Token Optimized) | Hindi-English code-mixed, Roman script | Reddit comments

Philosophy: When in doubt on harassment/hate speech, output 0. Silencing legitimate speech is worse than missing an edge case. Never flag opinions, criticism, or political disagreement as harassment or hate speech.

Design note on pv (profanity_vulgarity): this flag is a detection signal, not a moderation verdict. It fires whenever explicit profanity is present, regardless of target — including profanity aimed at a government, institution, or policy.

OUTPUT FORMAT
Return ONLY a JSON array, no preamble.
CRITICAL: Do NOT include the original text. You MUST put 'analysis' first to force logical evaluation before scoring.
Use exactly these short keys for every object to save tokens:
[{"id": "<id>", "analysis": "<1 brief sentence analyzing target and intent>", "pv": 0, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0}]

Key Legend:
pv = profanity_vulgarity | tah = targeted_abuse_harassment | dhs = discriminatory_hate_speech
cst = caste | cr = communal_religious | rx = regional_xenophobic | mg = misogyny_gender

`dhs` = 1 iff any dimension (cst, cr, rx, mg) = 1. If 0, all dimensions = 0.
The three top-level flags are independent — any combination is valid.

THE DECISION PROCEDURE
Step 1 — Is explicit profanity/vulgarity present anywhere in the text?
Check for curse words/vulgar swearing (chutiya, bsdk, mc, bc, gaand, lund, randi, fuck, madarchod, behenchod, bhosadike, gandu, harami, kamine), including obfuscated variants.
- Present anywhere (even filler/abstract target) → pv: 1. Otherwise 0.

Step 2 — Is there targeted personal abuse/harassment OR an identity-based stereotype/objectification/hierarchy statement present?
- Aimed at abstract entity (government, system, company, policy, game, title-alone) → tah: 0
- Self-directed ("main chutiya hoon") or reported speech → 0
- Filler/exclamation not directed at anyone ("bc yaar") → 0
- Public figure reverse test: Personal (appearance/character) → 1. Professional (policy) → 0.
- A specific person insulted, threatened, or personally degraded → tah: 1

Step 3 — Is an identity dimension (caste/religion/region/gender) load-bearing?
Test: if you removed the identity reference, would the sentence lose its hostile punch?
- Yes → dhs: 1 + correct dimension(s). Applies even with NO profanity present — covert othering, rhetorical questions, sarcastic praise-as-mockery, objectification, gender-hierarchy statements, mocking religious practices, and conspiracy framing count.
- No, insult/target is generic and applies regardless of identity ("abe randi" to a male gamer; animal metaphor as complaint) → identity dimensions stay 0.

Step 4 — Political-affiliation terms (sanghi/bhakt/libtard/chamcha) are NOT an identity dimension on their own. Only flag `cr` if hostility is tied to actual religious/communal identity.

Confidence check: if a comment hinges on one ambiguous/borderline term with no corroborating signal, prefer 0 on `tah` and `dhs`.

CATEGORY DEFINITIONS (Vocabulary)
- Caste slurs: chamar, bhangi, neech jaati, dedh, chura, pallan, kanjar, dhobi, mehtar, harijan, "aarakshanjeevi"/"quota khane wale" (only when demeaning people)
- Communal slurs: katwa, mulla/mulle, kafir, malaun, pajeet, "cow piss drinker", "dung worshipper", "sanghi/bhakt" (only when dehumanizing)
- Regional slurs: bihari/madrasi (pejorative), bimaru, chinki, ghati, bhaiyya, "Porki"/"Porkistan"
- Gendered: chhakka/hijra, "maal"/"item", "randi khana"/"randi rona", "feminazi", gender-hierarchy statements

KNOWN TRAPS (Read Carefully)
1. Homonyms: chakka = jackfruit vs. slur. BC = date vs. profanity. bhakt = devotee vs. slur. Context decides.
2. Idioms: ke chakkar mein, chakkar aana, chakka jam — benign idioms.
3. Meta-commentary: "Randia/Randians" as subreddit nickname is not a slur.
4. India-Pakistan: criticizing Pakistani govt/army = 0. Nationalist boasts = 0. Blanket-dehumanizing "Pakistani people" or Indian Muslims as a class (loyalty-questioning, "gaddar") = 1 on dhs.
5. Body-shaming/mental-health (motu, paagal, mental, psycho) = tah: 1 when directed at a person; NOT dhs unless identity-based. Neutral discussion = 0.
6. Obfuscation: letter repetition, character substitution (ch0@tiya), spacing, emoji substitution count for pv.
7. Topic discussion vs. endorsement: discussing controversy as news = 0. Asserting conspiracy as fact with hostile intent = 1.
8. Humor without animus: referencing a region/group with no demeaning framing = 0.
9. Icon-mocking: hostile mockery of cultural icons to humiliate the community = dhs: 1.

ANCHOR EXAMPLES
[
  {"id": "ex1", "analysis": "Profanity present, but institutional target.", "pv": 1, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex2", "analysis": "Directed personal insult using standard profanity.", "pv": 1, "tah": 1, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex3", "analysis": "Hostility and slur targeting communal identity.", "pv": 1, "tah": 1, "dhs": 1, "cst": 0, "cr": 1, "rx": 0, "mg": 0},
  {"id": "ex4", "analysis": "Caste slur used to demean a group, no profanity.", "pv": 0, "tah": 0, "dhs": 1, "cst": 1, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex5", "analysis": "Caste and communal overlap, no profanity.", "pv": 0, "tah": 0, "dhs": 1, "cst": 1, "cr": 1, "rx": 0, "mg": 0},
  {"id": "ex6", "analysis": "Xenophobic stereotyping of a region.", "pv": 0, "tah": 0, "dhs": 1, "cst": 0, "cr": 0, "rx": 1, "mg": 0},
  {"id": "ex7", "analysis": "Misogynistic statement enforcing gender hierarchy.", "pv": 0, "tah": 0, "dhs": 1, "cst": 0, "cr": 0, "rx": 0, "mg": 1},
  {"id": "ex8", "analysis": "Objectification, no profanity (kya maal hai).", "pv": 0, "tah": 0, "dhs": 1, "cst": 0, "cr": 0, "rx": 0, "mg": 1},
  {"id": "ex9", "analysis": "Covert othering based on religion.", "pv": 0, "tah": 0, "dhs": 1, "cst": 0, "cr": 1, "rx": 0, "mg": 0},
  {"id": "ex10", "analysis": "Sarcastic praise as mockery in soldier-death thread.", "pv": 0, "tah": 0, "dhs": 1, "cst": 0, "cr": 1, "rx": 0, "mg": 0},
  {"id": "ex11", "analysis": "Violence call against a religious group.", "pv": 0, "tah": 0, "dhs": 1, "cst": 0, "cr": 1, "rx": 0, "mg": 0},
  {"id": "ex12", "analysis": "Severe combined threat, graphic but no swear word.", "pv": 0, "tah": 1, "dhs": 1, "cst": 0, "cr": 0, "rx": 0, "mg": 1},
  {"id": "ex13", "analysis": "Profanity present in professional criticism.", "pv": 1, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex14", "analysis": "Profanity used for a personal attack on a public figure.", "pv": 1, "tah": 1, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex15", "analysis": "Nationalist boast, not a slur.", "pv": 0, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex16", "analysis": "Topic discussion, not endorsement.", "pv": 0, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex17", "analysis": "Animal metaphor as complaint, not insult.", "pv": 0, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex18", "analysis": "Profanity used as filler.", "pv": 1, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex19", "analysis": "Self-directed profanity.", "pv": 1, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex20", "analysis": "Generic profanity toward male gamer, no gender hate.", "pv": 1, "tah": 1, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex21", "analysis": "Mental-health insult, personal, no profanity.", "pv": 0, "tah": 1, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex22", "analysis": "Food homonym context, benign.", "pv": 0, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex23", "analysis": "Policy debate regarding reservation.", "pv": 0, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0},
  {"id": "ex24", "analysis": "Political critique lacking specific religious dehumanization.", "pv": 0, "tah": 0, "dhs": 0, "cst": 0, "cr": 0, "rx": 0, "mg": 0}
]
"""

def label_batch(comments_batch, attempt=1):
    numbered = "\n".join(f'ID: {cid} | Comment: {body}' for cid, body in comments_batch)
    user_prompt = f"Label these {len(comments_batch)} comments strictly following the JSON schema:\n{numbered}"

    try:
        start = time.time()
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            temperature=0.1,
            max_tokens=2500,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}} # Thinking OFF for fast-hack speed
        )
        duration = time.time() - start
        
        # 🛡️ DEFENSIVE PARSING: Handles both [...] and {"results": [...]}
        raw_content = res.choices[0].message.content
        parsed_data = json.loads(raw_content)
        
        if isinstance(parsed_data, list):
            parsed_array = parsed_data
        elif isinstance(parsed_data, dict):
            parsed_array = parsed_data.get("results", [])
            # Fallback in case it names the key something weird
            if not parsed_array:
                for val in parsed_data.values():
                    if isinstance(val, list):
                        parsed_array = val
                        break
        else:
            parsed_array = []
        
        if len(parsed_array) == len(comments_batch):
            for idx, item in enumerate(parsed_array): 
                item["id"] = str(comments_batch[idx][0])
                
            tqdm.write(f"  [⚡ inference] Time: {duration:.2f}s | Out tkns: {res.usage.completion_tokens}")
            return parsed_array
            
        raise ValueError(f"Dropped items. Expected {len(comments_batch)}, got {len(parsed_array)}")
        
    except Exception as e:
        if attempt <= 4:
            wait_time = min(2 ** attempt, 10)
            tqdm.write(f"    [!] Retry {attempt}/4 (Error: {e}). Waiting {wait_time}s...")
            time.sleep(wait_time)
            return label_batch(comments_batch, attempt + 1)
        tqdm.write(f"    [❌] Batch permanently failed.")
        return []

# ==========================================
# 4. LABELING PIPELINE (Thread Pool)
# ==========================================
df["id"] = df.index.astype(str)
batches = [list(zip(df["id"], df["body"]))[i:i + 20] for i in range(0, len(df), 20)]
all_labels = []

print(f"\n🚀 Starting Inference on {len(df)} rows with 10 Workers...")
lock = threading.Lock()

with tqdm(total=len(batches), desc="Labeling Batches") as pbar:
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Using executor.map for simple synchronous multi-threading
        for labels in executor.map(label_batch, batches):
            with lock:
                all_labels.extend(labels)
                if len(all_labels) % 100 < 20 and len(all_labels) > 0:
                    pd.DataFrame(all_labels).to_csv(CHECKPOINT_FILE, index=False)
            pbar.update(1)

labels_df = pd.DataFrame(all_labels)
if not labels_df.empty:
    labels_df["id"] = labels_df["id"].astype(str)

    KEY_MAPPING = {
        "analysis": "step_by_step_analysis", "pv": "profanity_vulgarity", "tah": "targeted_abuse_harassment",
        "dhs": "discriminatory_hate_speech", "cst": "caste", "cr": "communal_religious", 
        "rx": "regional_xenophobic", "mg": "misogyny_gender"
    }
    labels_df.rename(columns=KEY_MAPPING, inplace=True)
    
    final_df = df.merge(labels_df, on="id", how="left").drop(columns=["id", "pv", "tah", "dhs", "cst", "cr", "rx", "mg"], errors='ignore')
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ SUCCESS! Saved {len(final_df)} labeled r/{TARGET_SUBREDDIT} rows to: {OUTPUT_FILE}")
else:
    print("\n❌ Failed to generate labels.")
