# fetch-script Specification

## Purpose
TBD - created by archiving change fix-fetch-login-rate-limit. Update Purpose after archive.
## Requirements
### Requirement: No whoami call at startup

The script SHALL NOT call `huggingface_hub`'s `login()` (or otherwise hit the
`/api/whoami-v2` endpoint) at startup, since the token is passed explicitly to
each `HfApi.upload_file` call instead.

#### Scenario: Startup performs no token validation request

- **WHEN** the script starts with `HF_TOKEN` set and valid inputs
- **THEN** it performs no whoami/`login()` request and begins fetching pages
  immediately after input validation

#### Scenario: Rate-limited whoami endpoint cannot block the run

- **WHEN** `/api/whoami-v2` would return HTTP 429 (rate limited) at startup
- **THEN** the script does not depend on that endpoint and still proceeds to
  fetch and upload normally

### Requirement: Uploads still authenticate with the token

The script SHALL continue to pass `token=HF_TOKEN` explicitly to every
`HfApi.upload_file` call so uploads remain authenticated without `login()`.

#### Scenario: Token passed to every upload

- **WHEN** the script uploads a non-empty subreddit-month dataset via
  `HfApi.upload_file`
- **THEN** the call includes `token=HF_TOKEN` and uploads to
  `darelphilip/reddit_indian_subs` exactly as before

### Requirement: Script runs without login

Removing `login()` SHALL NOT change the script's fetch, filter, deduplicate,
upload, or exit-code behavior; it SHALL still run end-to-end without the
`login` import.

#### Scenario: End-to-end run unaffected

- **WHEN** the script is run with `SUBREDDITS`, a month range, and `HF_TOKEN`
- **THEN** it fetches, filters, deduplicates, and uploads results exactly as
  before, with `login` neither called nor imported

