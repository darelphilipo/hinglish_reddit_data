# @title
import os
import time
import json
import re
import datetime
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from openai import OpenAI

# 🔥 NEW: Ultra-fast PyArrow imports
from huggingface_hub import HfFileSystem
import pyarrow.dataset as pyds

try:
    from google.colab import userdata
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ==========================================
# 🎛️ CONFIGURATION & FOLDERS
# ==========================================
WORK_DIR = "/content/drive/MyDrive/Hinglish_Classifier_Project/testing_my_data"
OUTPUT_FILE = os.path.join(WORK_DIR, "test_labeled.csv")
CHECKPOINT_FILE = os.path.join(WORK_DIR, "checkpoint_labeled.csv")

os.makedirs(WORK_DIR, exist_ok=True)

# HF Data Configuration
HF_DATASET_REPO = "darelphilip/reddit_indian_subs"
TOTAL_ROWS_TO_LABEL = 1000

# 📅 NEW: Configurable Date Range (Format: DD-MM-YY)
START_DATE = "01-01-26"
END_DATE = "01-06-26"

# API Configuration
MODEL_NAME = "deepseek-v4-flash"
ENABLE_MINIMAL_THINKING = False

print("🔌 Connecting to OpenCode Go API...")
if IN_COLAB:
    try:
        try:
            OPENCODE_KEY = userdata.get("opencode")
        except userdata.SecretNotFoundError:
            OPENCODE_KEY = userdata.get("OPENCODE_KEY")
    except Exception:
        OPENCODE_KEY = None
else:
    OPENCODE_KEY = os.getenv("OPENCODE_KEY")

if not OPENCODE_KEY:
    raise ValueError("🚨 Missing OpenCode key. Set it in Colab Secrets or as an env variable 'OPENCODE_KEY'.")

client = OpenAI(
    api_key=OPENCODE_KEY,
    base_url="https://opencode.ai/zen/go/v1"
)

# ==========================================
# 🧠 SYSTEM PROMPT (Version 6.1 - Token Optimized)
# ==========================================
SYSTEM_PROMPT = """"Hinglish Content Moderation — Teacher Model Labeling Prompt
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


# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================
def parse_labels(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None

def label_batch(comments_batch):
    numbered = "\n".join(f'ID: {cid} | Comment: {body}' for cid, body in comments_batch)
    user_prompt = f"Label these {len(comments_batch)} comments strictly following the JSON schema:\n{numbered}"

    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            start_time = time.time()
            api_kwargs = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2500
            }

            if ENABLE_MINIMAL_THINKING:
                api_kwargs["reasoning_effort"] = "low"
                api_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                api_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            response = client.chat.completions.create(**api_kwargs)
            duration = time.time() - start_time

            usage = response.usage
            if usage:
                in_tkns, out_tkns = usage.prompt_tokens, usage.completion_tokens
                tps = out_tkns / duration if duration > 0 else 0
                mode_str = "🧠 Low-Think" if ENABLE_MINIMAL_THINKING else "⚡ Fast-Hack"
                tqdm.write(f"  [{mode_str}] Time: {duration:.2f}s | Out: {out_tkns} tkns | Speed: {tps:.1f} tkns/sec")

            response_text = response.choices[0].message.content
            parsed = parse_labels(response_text)

            if parsed:
                if len(parsed) == len(comments_batch):
                    input_ids = [str(cid) for cid, _ in comments_batch]
                    output_ids = [str(item.get("id", "")) for item in parsed]
                    if set(input_ids) != set(output_ids):
                         for idx, item in enumerate(parsed):
                             item["id"] = input_ids[idx]
                    return parsed
                else:
                    raise ValueError(f"Dropped item! Sent {len(comments_batch)}, but got {len(parsed)}.")
            raise ValueError("Unparseable JSON response.")

        except Exception as e:
            wait = min(2 ** attempt, 15)
            tqdm.write(f"    [!] Error on attempt {attempt}: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    return []

# ==========================================
# 🚀 MAIN PIPELINE
# ==========================================
def main():
    # Convert dates to Unix Epoch timestamps
    start_dt = datetime.datetime.strptime(START_DATE, "%d-%m-%y")
    end_dt = datetime.datetime.strptime(END_DATE, "%d-%m-%y").replace(hour=23, minute=59, second=59)
    start_epoch = int(start_dt.timestamp())
    end_epoch = int(end_dt.timestamp())

    print(f"📥 Querying dataset '{HF_DATASET_REPO}' directly via PyArrow Predicate Pushdown...")
    print(f"📅 Date Filter: {start_dt.strftime('%d-%b-%Y')} to {end_dt.strftime('%d-%b-%Y')}")
    print(f"🔢 Epoch Range: {start_epoch} to {end_epoch}")

    fs = HfFileSystem()
    fetch_start = time.time()

    # 🔍 Fetch ONLY completed train files, completely ignoring broken tmp_batch files
    valid_files = fs.glob(f"datasets/{HF_DATASET_REPO}/data/train*.parquet")
    print(f"📂 Found {len(valid_files)} valid Parquet files. Bypassing corrupted temp files...")

    # Connect directly to the repository's valid Parquet files
    dataset = pyds.dataset(valid_files, filesystem=fs, format="parquet")
    # MAGIC HAPPENS HERE: PyArrow pushes the date filter to Hugging Face, completely skipping
    # any row chunks outside your specified date range. It downloads ONLY what matches.
    filtered_table = dataset.to_table(
        columns=["body", "subreddit", "created_utc"],
        filter=(pyds.field("created_utc") >= start_epoch) & (pyds.field("created_utc") <= end_epoch)
    )

    df = filtered_table.to_pandas()

    if df.empty:
        raise ValueError(f"🚨 No rows found between {START_DATE} and {END_DATE}. Check your date ranges!")

    print(f"✅ Fast-fetch complete! Downloaded {len(df):,} matching rows in {time.time() - fetch_start:.2f}s")

    text_col = "body"

    # Perform Balanced Index Sampling directly on the heavily reduced dataframe
    print("⚖️ Balancing dataset equally across subreddits...")
    subs = df['subreddit'].unique()
    limit_per_sub = max(1, TOTAL_ROWS_TO_LABEL // len(subs))

    # Since the dataframe is now small, Pandas native group handling is instant
    df = df.groupby('subreddit', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), limit_per_sub), random_state=42)
    ).reset_index(drop=True)

    # Generate strict sequential IDs for processing
    df["id"] = df.index.astype(str)

    BATCH_SIZE = 20
    MAX_WORKERS = 10

    all_labels = []
    comments_list = list(zip(df["id"], df[text_col]))
    batches = [comments_list[i:i + BATCH_SIZE] for i in range(0, len(comments_list), BATCH_SIZE)]

    mode_print = "Minimal Thinking ON" if ENABLE_MINIMAL_THINKING else "Thinking OFF (Analysis-First)"
    print(f"\n🚀 Starting Inference | Mode: {mode_print} | Total balanced rows: {len(df)}")

    lock = threading.Lock()

    with tqdm(total=len(batches), desc="Labeling Batches") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(label_batch, batch) for batch in batches]

            for future in as_completed(futures):
                labels = future.result()

                with lock:
                    all_labels.extend(labels)
                    if len(all_labels) > 0 and len(all_labels) % 100 < BATCH_SIZE:
                        pd.DataFrame(all_labels).to_csv(CHECKPOINT_FILE, index=False)

                pbar.update(1)

    labels_df = pd.DataFrame(all_labels)
    labels_df["id"] = labels_df["id"].astype(str)

    # 🗺️ Expand Short Keys to Full Target Column Names
    KEY_MAPPING = {
        "analysis": "step_by_step_analysis",
        "pv": "profanity_vulgarity",
        "tah": "targeted_abuse_harassment",
        "dhs": "discriminatory_hate_speech",
        "cst": "caste",
        "cr": "communal_religious",
        "rx": "regional_xenophobic",
        "mg": "misogyny_gender"
    }
    labels_df = labels_df.rename(columns=KEY_MAPPING)

    # Merge on the temporary ID
    final_df = df.merge(labels_df, on="id", how="left")
    final_df = final_df.drop(columns=["id", "created_utc"])

    # Ensure column order matches exactly what you requested
    target_cols = [
        text_col, "step_by_step_analysis", "profanity_vulgarity", "targeted_abuse_harassment",
        "discriminatory_hate_speech", "caste", "communal_religious", "regional_xenophobic", "misogyny_gender"
    ]

    existing_target_cols = [c for c in target_cols if c in final_df.columns]
    other_cols = [c for c in final_df.columns if c not in existing_target_cols]
    final_df = final_df[existing_target_cols + other_cols]

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ SUCCESS! Saved {len(final_df):,} labeled rows perfectly balanced by subreddit to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
