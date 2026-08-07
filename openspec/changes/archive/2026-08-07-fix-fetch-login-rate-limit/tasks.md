# Tasks

- [ ] Remove the `login(token=HF_TOKEN)` call from `main()` in
      `fetch_subreddit_range_to_hf.py`.
- [ ] Remove `login` from the `from huggingface_hub import login, HfApi`
      line so only `HfApi` remains imported.
- [ ] Confirm `HfApi.upload_file` still receives `token=HF_TOKEN` on every
      upload call.
- [ ] Confirm the script still starts fetching pages immediately after input
      validation without any whoami call.
- [ ] Validate the OpenSpec change with
      `openspec validate fix-fetch-login-rate-limit`.
