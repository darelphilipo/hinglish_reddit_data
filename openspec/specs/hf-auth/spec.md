# hf-auth Specification

## Purpose
TBD - created by archiving change fix-remove-login-from-scripts. Update Purpose after archive.
## Requirements
### Requirement: No whoami call at startup

Neither `fetch_arctic_to_hf.py` nor `consolidate_to_hf.py` SHALL call
`huggingface_hub`'s `login()` (or otherwise hit the `/api/whoami-v2` endpoint)
at startup.

#### Scenario: Fetch script performs no token validation request

- **WHEN** `fetch_arctic_to_hf.py` starts with `HF_TOKEN` set and valid inputs
- **THEN** it performs no whoami/`login()` request and begins fetching pages
  immediately after input validation

#### Scenario: Consolidate script performs no token validation request

- **WHEN** `consolidate_to_hf.py` starts with `HF_TOKEN` set and valid inputs
- **THEN** it performs no whoami/`login()` request and proceeds directly to
  listing repository files and consolidating batches

#### Scenario: Rate-limited whoami endpoint cannot block either run

- **WHEN** `/api/whoami-v2` would return HTTP 429 (rate limited) at startup
- **THEN** neither script depends on that endpoint and both still proceed to
  fetch, consolidate, and upload normally

### Requirement: Authentication via HF_TOKEN env var

Both scripts SHALL remain authenticated without an explicit `login()` call,
because `huggingface_hub` auto-reads the `HF_TOKEN` env var that the workflow
exports.

#### Scenario: Fetch script uploads authenticated by env var

- **WHEN** `fetch_arctic_to_hf.py` pushes a checkpoint via
  `dataset.push_to_hub(repo_id=HF_DATASET_REPO, split=split_name, private=True)`
- **THEN** the push is authenticated using the `HF_TOKEN` env var with no
  explicit token argument and no `login()` call

#### Scenario: Consolidate script uploads and commits authenticated by env var

- **WHEN** `consolidate_to_hf.py` creates `api = HfApi()` and then calls
  `api.upload_file(...)` and `api.create_commit(...)`
- **THEN** the `HfApi()` instance auto-reads `HF_TOKEN` and uploads/commits
  remain authenticated with no explicit token argument and no `login()` call

### Requirement: Upload and commit behavior unchanged

Removing `login()` SHALL NOT change how either script uploads or commits; the
calls to `push_to_hub`, `upload_file`, and `create_commit` SHALL remain
identical to before.

#### Scenario: Fetch script uploads unchanged

- **WHEN** `fetch_arctic_to_hf.py` is run with `HF_TOKEN`, subreddits, and a
  batch name
- **THEN** it fetches and pushes checkpoints to the same `HF_DATASET_REPO`
  splits exactly as before, with `login` neither called nor imported

#### Scenario: Consolidate script uploads and commits unchanged

- **WHEN** `consolidate_to_hf.py` is run with `HF_TOKEN`
- **THEN** it uploads final files and creates commits to `HF_DATASET_REPO`
  exactly as before, with `login` neither called nor imported

### Requirement: Scripts run without login import

Removing `login()` SHALL NOT change either script's end-to-end behavior; both
SHALL still run successfully without the `login` import present.

#### Scenario: End-to-end runs unaffected

- **WHEN** both scripts are executed with `HF_TOKEN` and valid inputs
- **THEN** they fetch, consolidate, upload, and commit exactly as before, with
  `login` neither called nor imported in either script

