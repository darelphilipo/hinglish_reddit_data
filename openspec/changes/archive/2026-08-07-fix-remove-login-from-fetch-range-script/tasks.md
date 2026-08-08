# Tasks

- [x] Remove the `login(token=HF_TOKEN)` call from `main()` in
      `fetch_subreddit_range_to_hf.py`.
- [x] Remove `login` from the `from huggingface_hub import login, HfApi`
      line so only `HfApi` remains imported.
- [x] Confirm `HfApi.upload_file` still receives `token=HF_TOKEN` on every
      upload call.
- [x] Confirm the script still starts fetching pages immediately after input
      validation without any whoami call.
- [x] Validate the OpenSpec change with
      `openspec validate fix-remove-login-from-fetch-range-script`.
