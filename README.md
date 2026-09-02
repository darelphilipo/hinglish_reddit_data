# Building a Hinglish Hate-Speech Classifier — A Journey

*A data-engineering, prompt-engineering, and model-training case study — from a MuRIL baseline benchmark, through a 196k-row real-Reddit training set, to a fine-tuned hing-roberta classifier, with an mmBERT upgrade already on the roadmap.*

> Everything in this document is reconstructed from a **three-part AI-conversation transcript — 487 pages (part 1), a 359-page continuation (part 2), and a 552-page part 3** — of the actual build: the decisions, the dead ends, the debug sessions, and the wins. It is a documentation and educational showcase, **not a product**, and all metrics shown are as recorded during the build.

---

## Scorecard

| Metric | Value |
|---|---|
| MuRIL v3 baseline Macro F1 | **0.8398** — the benchmark every decision chased |
| Labeling prompt version | **v6.5** — the "Goldilocks" decoupled 3-flag design (`pv`/`tah`/`dhs`) |
| Tokens saved per batch | **~400**, via ultra-short JSON keys |
| Comments per API batch | **20** — sweet spot for throughput and parse reliability |
| Parallel year-workers | **4** (2017–2020) in the GitHub Actions matrix, `fail-fast: false` |
| DuckDB memory cap | **6GB** (`PRAGMA memory_limit`) |
| Parquet shards scanned | **200** production / **10** test — dynamic shard scaling |
| First baseline run | **210 rows labeled**, 24.3% overall toxicity rate |
| Prompt-cache hit rate | **86%** — static system prompt served from cache |
| Master dataset row target | **50,000**, grown entirely from real Reddit data |
| HUNTER_LIMIT cap | `min(shortfall × 2, 8,000)` rows per hunt |
| Minimum rows per minority class | **1,000** (caste, misogyny, regional, communal) |
| LLM-label spot-check sample | **10%** — manual review gate before shipping |
| Profanity sanity cap | **<30%** per class in the final distribution report |
| Unique rows after dedup | **196,700** — real size of the master dataset after the Cartesian-join fix |
| Deployed classifier Macro F1 | **0.55** — normal for imbalanced multi-label; edge cases documented |
| Model parameters | **278M** (`XLMRobertaForSequenceClassification`) |
| Fine-tuning set | **~110k rows** — 90/10 split, 3 epochs, seed 42 |
| Fine-tune time | **~1h05m** on a free Colab T4 at ~77 examples/s |
| mmBERT-base vocabulary | **256k** (Gemma-2 tokenizer), trained on 3T+ tokens over 1,800+ languages |

---

## Table of Contents

1. [The Baseline Model: MuRIL v3](#01--the-baseline-model-muril-v3)
2. [Data Collection Architecture](#02--data-collection-architecture)
3. [The Labeling Prompt: v1 → v6.x "Goldilocks"](#03--the-labeling-prompt-v1--v6x-goldilocks)
4. [The CI/CD Pipeline: GitHub Actions](#04--the-cicd-pipeline-github-actions)
5. [The Bug-Fixing Saga](#05--the-bug-fixing-saga)
6. [First Results & The Class-Imbalance Problem](#06--first-results--the-class-imbalance-problem)
7. [The Harvester: Filling the Gaps (No Synthetic Data)](#07--the-harvester-filling-the-gaps-no-synthetic-data)
8. [Engineering Judgement: When to Say No](#08--engineering-judgement-when-to-say-no)
9. [The 4-Step Closed Loop: Hunt, Harvest, Rebalance](#09--the-4-step-closed-loop-hunt-harvest-rebalance)
10. [The v6.5 Decision Procedure, Sharper](#10--the-v65-decision-procedure-sharper)
11. [Sanitization: Every Parameter Is a Trade-Off](#11--sanitization-every-parameter-is-a-trade-off)
12. [Validation Guardrails](#12--validation-guardrails--before-it-all-hangs-together)
13. [The Harvester Goes Tiered and Self-Tuning](#13--the-harvester-goes-tiered-and-self-tuning)
14. [From GitHub to Hugging Face — and a Cartesian Bug](#14--from-github-to-hugging-face--and-a-cartesian-bug)
15. [Model Selection: MuRIL Was the Benchmark, RoBERTa Got the Job](#15--model-selection-muril-was-the-benchmark-roberta-got-the-job)
16. [The Roadmap: mmBERT-base](#16--the-roadmap-mmbert-base)
17. [The Data Pipeline, End to End](#the-data-pipeline-end-to-end)
18. [Lessons Learned](#lessons-learned)

The project is told as sixteen stages spanning all three parts of the build transcript: **part 1 covers stages 1–8**, **part 2 covers stages 9–12** (closing the loop with an autonomous, self-healing pipeline), and **part 3 covers stages 13–16**, which take the pipeline into storage migration, model training, and the mmBERT roadmap.

---

## 01 · The Baseline Model: MuRIL v3

*baseline · fine-tuning*

Every serious ML project starts with a baseline. This one began with **MuRIL** — Google's Multilingual Representations for Indian Languages transformer — fine-tuned for the toxicity task. The **MuRIL v3** classifier reached a **Macro F1 of 0.8398**, the number every later decision had to beat.

**The core problem:** Hinglish (Hindi-English code-mixing) is *sparse, obfuscated, and domain-specific*. Speakers deliberately misspell words, write romanized Hindi ("`log hume hii kyun chhedte hai`"), and lean on subreddit-specific slang. Generic, off-the-shelf toxicity models fail on it because they never saw that distribution in training.

To source data of varying toxicity density, a **4-tier subreddit taxonomy** was designed — from high-priority toxic subs like **r/Chodi** and **r/bakchodi** down to clean "hard negative control" subs like **r/developersIndia**, **r/CarsIndia**, and **r/TwoXIndia**:

```python
# 4-tier taxonomy: tune the toxicity density per tier
TIERS = {
    "t1_high_tox": ["Chodi", "bakchodi"],   # high-priority toxic
    # ...intermediate tiers 2 & 3 fill the middle density...
    "t4_hard_neg": ["developersIndia", "CarsIndia",
                     "TwoXIndia"],           # clean controls
}
```

---

## 02 · Data Collection Architecture

*data collection*

Reddit data was sourced from the **Arctic Shift API** and the **historical Pushshift archives** hosted on Hugging Face — for example the `open-index/arctic` dataset and `sentence-transformers/reddit-title-body` for banned subreddits like **r/Chodi** that no longer exist as live subreddits.

Mid-project, labeling moved from **Gemini** to the **OpenCode Go API** — model `deepseek-v4-flash`, base URL `https://opencode.ai/zen/go/v1`, key supplied from **Colab secrets**.

Several early decisions shaped everything after:

- **Sanitize before sending** — HTML unescape, strip tags with the regex `<[^>]+>`, strip markdown links, and scrub `/u/` handles. Cheap tokens, but they add up across 20-comment batches.
- **Deduplicate aggressively** on a normalized string core so no comment appears twice.
- **Checkpoint sampled rows** — a `seen_ids_ledger.txt` so re-runs never label the same comment twice.

```python
import html, re

def sanitize(text):
    text = html.unescape(text)                    # &amp; &lt; &gt;
    text = re.sub(r"<[^>]+>", "", text)            # strip HTML tags
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)     # markdown links
    text = re.sub(r"/u/[\w-]+", "", text)          # scrub handles
    return " ".join(text.split())
```

```python
# never label a comment twice
pending = [c for c in sampled if c.id not in seen_ids]
with open("seen_ids_ledger.txt", "a") as f:
    for cid in batch_ids:
        f.write(f"{cid}\n")
```

---

## 03 · The Labeling Prompt: v1 → v6.x "Goldilocks"

*prompt engineering*

The labeling prompt went through many versions. The **v6.x "Goldilocks"** redesign is the centerpiece of the project — not too coarse to be useless, not too fine to be unreachable. It is built on a **decoupled 3-flag architecture**:

- `pv` — **profanity_vulgarity**
- `tah` — **targeted_abuse_harassment**
- `dhs` — **discriminatory_hate_speech** — **1 if and only if ANY hate dimension is 1**: `cst` (caste), `cr` (communal/religious), `rx` (regional/xenophobic), `mg` (misogyny/gender)

The **ultra-short JSON keys** (`pv`/`tah`/`dhs`/`cst`/`cr`/`rx`/`mg`) saved roughly **~400 tokens per batch** — a deliberate 1-to-1 trade of JSON formatting tokens for reasoning runway, reinvested in a mandatory one-sentence `analysis` field on every comment.

```
# Step 1  explicit profanity / vulgarity ......... pv  = 1
#         kept even for filler / abstract targets
# Step 2  targeted personal abuse / harassment .. tah = 1
#         identity-based stereotyping .......... dhs = 1
#         abstract entity / self-directed /
#         reported speech / filler ............. 0
# Step 3  "When in doubt, output 0."
```

> **Philosophy:** "When in doubt, output 0." `pv` is a *detection signal*, not a moderation verdict — profanity alone doesn't make a comment hate speech, and the flags must stay orthogonal.

**v6.5** restored **full English coverage** (handling English hate and abuse without deviating from the main Hinglish objective) and wrapped batches in `{"results": [...]}`. A **"Human Reviewer Guidelines" SOP** was written so human reviewers and the AI labeler judge by the same standard — exact category definitions, a "Load-Bearing Test" for hate speech, and a list of common Hinglish traps.

Inference was tuned hard. **Async labeling** was tried (thinking ON, 12–18 s/batch) but **reverted to synchronous batch calls** — thinking OFF via `extra_body={"thinking": {"type": "disabled"}}`, `temperature 0.1`, `max_tokens 2500`, `response_format={"type": "json_object"}`, **4 retries with backoff**, **20 comments per batch**, thread pool:

```python
resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    temperature=0.1,
    max_tokens=2500,
    response_format={"type": "json_object"},
    extra_body={"thinking": {"type": "disabled"}},
    messages=[{"role": "user", "content": batch_prompt}],
)
# 20 comments/batch · 4 retries w/ backoff · thread pool
```

---

## 04 · The CI/CD Pipeline: GitHub Actions

*ci/cd · github actions*

The full pipeline runs on **GitHub Actions** free-tier VMs, with a **matrix of 4 parallel year-workers** (2017, 2018, 2019, 2020) and `fail-fast: false` so one bad year doesn't kill the rest. **DuckDB with httpfs** reads Parquet shards directly from Hugging Face — `PRAGMA memory_limit='6GB'`, threads bumped 2→8 for parallel network pipes, `hf://datasets/open-index/arctic/…`.

```sql
SET memory_limit = '6GB';
SET threads = 8;            -- parallel network pipes
SET httpfs_enable = 1;

SELECT id, body, subreddit, created_utc
FROM read_parquet('hf://datasets/open-index/arctic/**/*.parquet')
WHERE subreddit IN (tier_subs)   -- 10 shards test / 200 prod
LIMIT 5000;
```

Two classic production mistakes got engineered away:

- **Dynamic shard scaling** — 10 shards for test runs (<1000 rows), 200 for production. The original mistake was scanning **200 shards (~100GB)** just to test on 50 rows.
- **Dynamic seed from `GITHUB_RUN_ID % 100000`** — a fixed `random.seed(42)` silently pulled *identical rows* on every run.

```python
seed = int(os.environ["GITHUB_RUN_ID"]) % 100000
random.seed(seed)   # never a hard-coded 42

df = df.groupby("bucket").sample(n, random_state=seed)
df = backfill_shortfalls(df, seed)   # stratify subreddit × month
```

**Stratified balancing** samples across subreddit × month buckets (with backfill) so every run samples evenly. Telemetry was added because CI is blind: a **psutil resource-monitor daemon** prints RAM/CPU every 30s, a **DuckDB "heartbeat" thread** prints elapsed time and network stats every 15s (DuckDB's native progress bar is swallowed by headless CI logs), and `PYTHONUNBUFFERED=1` defeats Python's log buffering that made the workflow look frozen.

The **system prompt is fetched at runtime from a raw GitHub URL** — prompt changes need no code change. Output is chunked to `labelled_output/chunks/` per year, then **Job 2** (`needs:` chain) downloads all artifacts, merges them into `master_baseline_tier1.csv`, prints a distribution report, and auto-commits/pushes via `github-actions[bot]` (`permissions: contents: write`). A `step3_harvester.py` (**Job 3**) closes the loop: read the master file → find category shortfalls → generate targeted keywords → surgically extract matching real comments → label and merge.

---

## 05 · The Bug-Fixing Saga

*debugging saga*

A CI pipeline is a debugging surface, and this one produced five memorable fights with the machine:

- **`id_x`/`id_y` merge collision** — both source data and the LLM JSON contained an `id` column, so pandas renamed them on merge. Fix: force-drop the LLM's `id` column before merging.
- **`groupby().sample()` index/data-loss bug** — the balancing logic silently dropped columns. Rewritten with a temporary `bucket` column (`subreddit_year_month`), direct index extraction, and explicit backfill — "bulletproof stratified balancing".
- **clean-text API drift** — `no_html=True` was never a valid argument; the library version silently changed and crashed. Fix: strip HTML with a manual regex before calling `clean()` — version-proof. The AI assistant *admitted its own mistake twice* here.
- **The missing rename line** — a "final" script accidentally deleted the column-rename step, so chunks were saved with short keys (`pv`, etc.) and the merge script crashed with `KeyError`. Salvage fix: a `rename_mapping` dictionary in the merge script expands abbreviations back to full column names.
- **Fresh-VM gotcha** — every GitHub Actions job boots a blank VM; Job 2 crashed with `ModuleNotFoundError: pandas`. Fix: add its own `pip install pandas` step.

```python
RENAME = {
    "pv":  "profanity_vulgarity",
    "tah": "targeted_abuse_harassment",
    "dhs": "discriminatory_hate_speech",
    "cst": "caste",  "cr": "communal_religious",
    "rx":  "regional_xenophobic",
    "mg":  "misogyny_gender",
}
df = df.drop(columns=["id"], errors="ignore")   # id_x/id_y fix
df = df.rename(columns=RENAME)
```

---

## 06 · First Results & The Class-Imbalance Problem

*first results*

The first baseline run produced **210 rows at a 24.3% overall toxicity rate**. But the distribution report showed **severe shortfalls** versus targets — exactly why random stratified sampling alone is economically wasteful:

| Category | Target | Found |
|---|---|---|
| Caste (`cst`) | 1,000 | 3 |
| Misogyny / Gender (`mg`) | 1,000 | 2 |
| Communal / Religious (`cr`) | 1,200 | 9 |
| Regional / Xenophobic (`rx`) | 1,000 | 10 |

Random sampling burns *thousands of rows and tokens* to scrape a handful of caste/misogyny positives out of a sparse corpus. That insight drove the entire **Stage 7** harvester design.

One bright spot: the first full job run observed an **86% prompt-cache hit rate** — the static system prompt was served from cache, slashing label costs.

---

## 07 · The Harvester: Filling the Gaps (No Synthetic Data)

*targeted extraction*

**Step 3 "Harvester"** is a targeted extraction engine. It reads the master CSV, computes shortfalls per category, picks the priority category, prompts the LLM to generate **~50 highly specific Hinglish slurs/phrases/text patterns**, builds a `WHERE body LIKE '%keyword%'` DuckDB query, and **surgically harvests real matching comments** (`LIMIT 5000`) — which are then labeled and merged back into the master file.

```python
clause = " OR ".join(
    f"LOWER(body) LIKE '%{k}%'" for k in keywords)
query = f"""
SELECT id, body, subreddit, created_utc
FROM read_parquet(shards)
WHERE ({clause}) AND subreddit IN ({tier_subs})
LIMIT 5000
"""
```

> **Critical clarification:** This is **NOT synthetic data**. The LLM is used only as a *lexicon / dictionary builder* — every harvested comment is a **real Reddit comment** pulled from the Arctic dataset.

**Data-informed keyword discovery (few-shot) beats zero-shot**: feed the LLM the few real positive examples already found, so the generated keywords match real-world typing habits and automod-obfuscation — not dictionary definitions.

The earlier **"Hunter" approach** was revisited: **Differential Word Frequency Analysis (TF-IDF)** — deterministic, no LLM. Split the corpus into the target category vs. a background corpus, tokenize, strip stopwords (English: *the/and/is/of/to*; Hinglish: *hai/ki/aur/mein/se*), and score words by TF-IDF. Words frequent in the target but rare in the background spike and become candidate search keywords:

```python
# target corpus  vs  background corpus
t = tokenize(target_corpus);   t = minus_stopwords(t)   # en + hinglish
b = tokenize(background);      b = minus_stopwords(b)   # hai/ki/aur/mein/se
score = lambda w: tfidf(w, t, b)   # freq in target, rare in bg
keywords = top_n(rank(score), 50)
```

---

## 08 · Engineering Judgement: When to Say No

*engineering judgement*

Library upgrades were considered for *every* pipeline stage — **Loguru** for telemetry, **Neattext + ftfy** for sanitization, **scikit-learn `train_test_split(stratify=…)`** for balancing, **AsyncOpenAI + Instructor** for inference, **Rich** for logs. The verdict:

> Don't refactor a working pipeline. If it isn't broken, don't fix it.
> — *the guiding principle of the entire build*

- **sklearn's `train_test_split` fatally errors** when any bucket has exactly 1 row — the custom backfill logic handles it.
- **scikit-learn bloats VM boot time** and eats free GitHub Actions minutes.
- **Async rewrites risk silently swallowing API errors** — synchronous, retry-with-backoff calls were battle-tested.

Save the shiny libraries for **v2.0**. The pipeline that shipped is boring on purpose — and it works.

---

## 09 · The 4-Step Closed Loop: Hunt, Harvest, Rebalance

*part 2 · closed loop*

Part 2 opens by locking the whole pipeline into a formal 4-step feedback loop:

1. **Parallel Extraction** — a 4-VM GitHub Actions matrix pulls and labels baseline data (12,500 rows per year).
2. **Analysis & Deficiency Detection** (`merge_and_analyze.py`) — merges the yearly CSVs, compares each minority category against targets (1,000 minimum rows each), and writes a `deficiencies.json`.
3. **Smart Hunter** (`hunter_worker.py`) — reads the deficiencies, surgically queries DuckDB with keyword vectors, filters out anything already labeled (no wasted API tokens), labels the new rows, and saves `hunter_supplement.csv`.
4. **Final Rebalance** (`final_rebalance.py`) — merges baseline + hunter, deduplicates, shuffles, prints the distribution report, exports the master CSV.

> The pipeline no longer just records what it lacks — it goes out and finds it.

```json
{
  "caste": { "shortfall": 997, "keywords": [
      "dalit", "chamar", "bhangi",
      "neech", "quota", "untouchable"
  ]},
  "misogyny_gender": { "shortfall": 998, "...": "..." }
}
```

**Spread cap:** the hunter caps its own appetite: `HUNTER_LIMIT = min(total_shortfall * 2, 8000)` — enough to cover the gap with redundancy, never enough to blow up API spend on a runaway search.

---

## 10 · The v6.5 Decision Procedure, Sharper

*part 2 · prompt precision*

Part 2 also pins down the exact v6.5 decision procedure the teacher model follows — the detail that makes labels consistent at scale:

1. Explicit profanity/vulgarity anywhere → `pv:1`, even toward filler or abstract targets.
2. Targeted personal abuse/harassment → `tah:1` — with the "public figure reverse test": professional critique of a politician, journalist, or actor is NOT abuse; a personal attack on a private individual IS.
3. `dhs:1` if and only if any identity dimension (`cst`/`cr`/`rx`/`mg`) fires.
4. Political labels (sanghi, bhakt, libtard, snowflake) are NOT identity dimensions on their own — they only become hate speech when paired with a dimension or targeted abuse.

```
# 1  explicit profanity / vulgarity anywhere .... pv  = 1
#    kept even for filler / abstract targets
# 2  targeted personal abuse / harassment ....... tah = 1
#    public figure, professional capacity = critique, not abuse
# 3  dhs = 1 if and only if a dimension fires
#    cst / cr / rx / mg
# 4  political labels alone .................... 0
#    sanghi / bhakt / libtard / snowflake need a dimension
```

> **Politics ≠ identity:** Rule 4 exists because politics ≠ identity: calling someone a "sanghi" is criticism; claiming their religion makes them subhuman is hate. The prompt must be able to tell the difference.

The full v6.5 prompt — including per-category Hinglish slur lists, homonym traps (`chakka` = jackfruit vs slur; `BC` = date vs profanity), the India-Pakistan government-vs-people rule, and code-switching guidance — is fetched at runtime from a raw URL in the GitHub repo, so every pipeline run labels with the newest prompt.

---

## 11 · Sanitization: Every Parameter Is a Trade-Off

*part 2 · sanitization*

Part 2 turns text scrubbing into a studied discipline. PII shredders were added — `no_emails=True`, `no_phone_numbers=True` — so doxxing data never poisons the training set. But NOT `no_numbers`: in regional-xenophobia detection, numbers carry meaning ("Tier 2", "2002").

- **`no_html=True` — high risk:** clever slang like "absolute `<idiot>`" looks like markup and gets deleted.
- **`no_urls=True` — medium risk:** sometimes the URL IS the context (a link to an offensive meme).
- **`to_ascii=False` — critical for Hinglish:** forcing ASCII would destroy Devanagari.
- **`lower=False`:** keeps the ALL-CAPS shouting signal intact.

```python
text = html.unescape(text)
text = strip_markdown_links(text)      # drop [...](url) text
text = scrub_u_handles(text)           # /u/... gone
text = clean(text, no_html=True, no_urls=True,
             to_ascii=False, lower=False)
key = dedup_key(text)   # lower + strip punctuation → core
```

The `dedup_key` trick became a token saver: normalize the string to its core before comparing, and near-duplicate spam collapses to a single paid label.

---

## 12 · Validation Guardrails — Before It All Hangs Together

*part 2 · validation*

Before declaring the dataset done, part 2 adds lean, explicit guardrails at every step of the pipeline:

1. **Extraction** — validate source completeness: row counts per shard and per year, explicit coverage check.
2. **Labeling** — take a 10% sample and manually spot-check the LLM's output for hallucinated or flipped flags.
3. **Hunt** — assert the detected deficiencies after each hunt cycle.
4. **Finalize** — assert minimums (>= 1,000 per minority class) and distribution sanity (> 1% per class, < 30% profanity).

> **Keep it lean:** Full custom validation is overkill at this stage — the rule is 10% spot-checks and three hard asserts, not a test suite. Don't over-engineer; the model will do the heavy validation later.

Along the way, part 2 collected its share of operational scars: GitHub's "100 entries per page" log pagination looks like a hang but isn't; fresh VMs mean every job needs its own `pip install pandas`; interleaved logs can lie — "Worker 2018 loaded 9,664 rows from the 2020 CSV" is an output artifact; and `python: can't open file` usually means the script isn't in the runner's working directory.

---

## 13 · The Harvester Goes Tiered and Self-Tuning

*part 3 · self-tuning*

Part 3 opens by making the pipeline self-aware. Hardcoded targets are gone — `step2` now writes a `pipeline_targets.json` blueprint that both the harvester and the merger read, so every run hunts against the real, current deficit.

- **Stateful memory:** keywords accumulate in `prompt/master_lexicon.json` — the harvester gets "smarter" with every run. Clearing it is a valid reset, but leave `{}`, never a blank file (JSON parse errors).
- **Corpus-specific thresholding:** a dynamic thresholding engine purges any word appearing in more than 15% of the corpus before the TF-IDF math — the fix for the harvester "hemorrhaging tokens on useless English."
- **Tiered search:** Tier 1 = legacy echo chambers (small weight), Tier 2 = exhaustive Indian subreddits, Tier 3 = live data (the Arctic API, later a private `reddit_indian_subs` dataset with 160+ subs across 2022–2026). Diversity caps stop Tier 1 from vacuuming the whole quota.
- **Concurrency locks:** editing `master_lexicon.json` on GitHub while the bot auto-commits is a guaranteed race condition — write-collision guards added.

> **The 15% ceiling:** "Just drop words that appear too often" is the simple-sounding idea that fixed token waste: a max-15% document-frequency purge before keyword math. The pipeline learned to stop paying for obvious English.

---

## 14 · From GitHub to Hugging Face — and a Cartesian Bug

*part 3 · storage migration*

The GitHub repo had quietly become a database. Part 3 migrates all data storage to the Hugging Face dataset `darelphilip/hinglish-toxicity` — ChatML rows, parquet shards, a clean student-vs-teacher prompt split — leaving GitHub to host code and orchestration only.

- **One-off migration script** with a 10% progress bar, HF rate-limit awareness, and resume-on-failure.
- **The Cartesian product bug:** duplicate rows traced to a cross-join in a harvester path — the same reddit id + text repeated across 7+ rows. A one-off Colab cleanup collapses the dataset to **196,700 unique rows**.
- **The HF caching gotcha:** freshly uploaded 2026 shards stayed invisible in stats runs — the datasets library served a cached snapshot. Fix: a raw `data/**/*.parquet` file-tree scan fallback in the auditor.

> **GitHub is not a database:** Storing data in a git repo worked until the dataset outgrew it. The move to Hugging Face Datasets is what made a 200k-row training run realistic.

---

## 15 · Model Selection: MuRIL Was the Benchmark, RoBERTa Got the Job

*part 3 · training*

Here is the honest arc: **MuRIL v3 (0.8398)** was the starting point and the benchmark every decision chased — but when it came time to actually train, the candidate showdown rejected MuRIL. The verdict: *"Google MuRIL — downgrade."* Standard BERT architecture, trained on formal Hindi, never optimized for chaotic, slang-heavy internet Hinglish.

The model actually fine-tuned was **`l3cube-pune/hing-roberta`** — an XLM-RoBERTa backbone further pre-trained (DAPT) on the **L3Cube-HingCorpus**: 52.9M sentences and 1.04B tokens of real Romanized Hinglish from Twitter. With 95%+ of Indian Reddit written in Romanized Hinglish, it is the "perfect weapon" for the job.

- **Training environment:** Google Colab on a T4 — GitHub Actions runners (2 vCPU / 7 GB RAM) are useless for GPU backprop.
- **Recipe:** ~110k rows, 90/10 split, seed 42, 3 epochs, BCE loss + class weights for imbalance, macro-F1 evaluation; 278M parameters; ~1h05m at ~77 examples/s.
- **Ship:** pushed to `darelphilip/hinglish-toxicity-classifier`, deployed as a free Gradio Space (`darelphilip/hinglish_toxicity`), with a model card and launch posts.
- **Reality check:** deployed macro-F1 lands around **0.55** — expected for imbalanced multi-label. All-zero outputs on "`Tu pyaar hai mere chamar I love you`" and "`Mulle bahut hi mast log hote hain…`" expose the two killers: sentiment-slur conflicts and sarcastic/masked hate. More data won't fix an encoder ceiling.

> MuRIL measured the problem at 0.8398. RoBERTa was what actually learned to solve it.

---

## 16 · The Roadmap: mmBERT-base

*part 3 · roadmap*

Part 3 ends with the next chapter planned: fine-tuning **`jhu-clsp/mmBERT-base`**. It is ModernBERT on a Gemma-2 tokenizer with a 256k vocabulary, trained on 3T+ tokens across 1,800+ languages, with Flash Attention 2 and unpadding — 2–4x faster inference and 25–40% faster training than RoBERTa-class models.

- **Expected cost:** ~35–45 minutes for a 3-epoch run on the ~226k deduplicated rows, free Colab T4.
- **Still the 2026 pick:** recent encoder releases optimize for retrieval (bi-encoders), not discriminative sequence classification — mmBERT-base wins for this task.
- **Discipline:** vocabulary trimming and 4-bit quantization are real-time-API optimizations — both explicitly deferred for the MVP (4-bit measurably hurts quality; trimming is "textbook premature optimization" right now).
- **Ready to go:** the trial Colab script targets `darelphilip/mmbert-hinglish-mvp`.

> **The loop keeps closing:** Data pipeline → storage → training → serving → evaluation → next model. The project stopped being a dataset project the day the first classifier shipped.

---

## The Data Pipeline, End to End

Every stage above collapses into this single flow. Nothing here is synthetic — raw Reddit comments in, a balanced master dataset out.

```
01 · source     Arctic / Pushshift data
                Raw Reddit comments from the Arctic Shift API and Pushshift archives on Hugging Face.
      ↓
02 · extract    DuckDB extraction
                httpfs reads Parquet shards straight from hf:// — 200 shards in production, 10 for tests.
      ↓
03 · clean      Sanitization & dedup
                HTML/markdown stripped, /u/ handles scrubbed, normalized-core dedup, ledger checkpoint.
      ↓
04 · balance    Stratified balancing
                subreddit × month buckets, dynamic run-ID seed, explicit backfill for shortfall categories.
      ↓
05 · label      DeepSeek V4 Flash labeling
                20-comment batches, thinking off, temperature 0.1, 4 retries with backoff, short-key JSON.
      ↓
06 · validate   JSON parse
                {"results":[...]} responses parsed, validated, keys expanded back to full column names.
      ↓
07 · commit     Merge & commit
                Job 2 merges year chunks into the master file, prints the distribution report, auto-commits.
      ↓
08 · ship       Master dataset
                master_baseline_tier1.csv — the growing, 50k-row training target, ready for MuRIL v3.
```

Job 3 (`step3_harvester.py`) feeds shortfalls back to step 05 — the loop closes: harvest, label, merge, repeat.

---

## Lessons Learned

Fourteen lessons distilled for the next project:

1. **[Reproducibility] Fixed seeds are a silent trap.** Derive the seed from the run ID (`GITHUB_RUN_ID % 100000`) and checkpoint everything — the ledger file *will* save you.
2. **[Performance] "Big data" is a networking problem before it is a compute problem.** Most of a 5–7 minute DuckDB extract was waiting on Parquet footers; parallel threads plus fewer shards for tests fixed it.
3. **[Prompt design] Design the prompt for the failure mode.** Short keys free tokens for reasoning, and "when in doubt, output 0" keeps precision honest.
4. **[CI/CD] A CI pipeline is also a debugging surface.** Log buffering, fresh VMs, and API drift will bite you — make every job self-contained and logs live (`PYTHONUNBUFFERED=1`).
5. **[Sampling] For rare classes, random sampling is economically insane.** Use statistical differential analysis (TF-IDF) and LLM-generated targeted queries to harvest real data.
6. **[Data ethics] Never generate synthetic data to fix class imbalance** when real data exists and can be found with better search.
7. **[Judgement] Refactoring a working, battle-tested script for the sake of "modern libraries" is a risk, not an improvement.** If it isn't broken, don't fix it.
8. **[Validation] Assert before you train.** Minimums per class, >1% per class, <30% profanity, and a 10% human spot-check — lean guardrails beat a test suite.
9. **[Prompt precision] Codify the decision procedure.** "Public figure, professional capacity" and "political term ≠ identity dimension" are the rules that keep labels consistent at 50k rows.
10. **[CI/CD] Logs are evidence, not truth.** "100 entries per page" isn't a hang, every job is a fresh VM, and interleaved output can misattribute rows to years.
11. **[Models] Benchmark ≠ deployed model.** MuRIL's 0.8398 was the starting point and measuring stick; what actually shipped was `l3cube-pune/hing-roberta`, an XLM-RoBERTa DAPT'd on Romanized Hinglish.
12. **[Models] Domain-adaptive pretraining beats newer architecture.** MuRIL lost because it wasn't optimized for internet Hinglish; the HingCorpus-DAPT'd RoBERTa won despite an older backbone.
13. **[Models] The encoder ceiling is real.** Sarcastic/masked hate and sentiment-slur conflicts defeat encoders no matter how much data you add — the mmBERT roadmap is the answer, not more rows.
14. **[Storage] GitHub is not a database.** Move to HF Datasets when the corpus outgrows git, dedupe properly (watch for Cartesian-join bugs), and remember client-side caching hides your freshest shards.

---

## Disclaimer

This document is a documentation and educational showcase reconstructed from a three-part AI-conversation transcript (487 + 359 + 552 pages) of building a Hinglish hate-speech classifier. It is not affiliated with, endorsed by, or a product of any of the tools, models, or platforms mentioned (MuRIL, DeepSeek, Gemini, DuckDB, GitHub Actions, Hugging Face, Arctic, Pushshift, Reddit). All metrics shown are as recorded during the build.
