# Tasks

- [x] Remove the `login(token=HF_TOKEN)` call in `fetch_arctic_to_hf.py`.
- [x] Remove `login` from the `from huggingface_hub import login` line in
      `fetch_arctic_to_hf.py`.
- [x] Remove the `login(token=HF_TOKEN)` call in `consolidate_to_hf.py`.
- [x] Remove `login` from the `from huggingface_hub import login, HfApi,
      CommitOperationDelete` line in `consolidate_to_hf.py` so only `HfApi`
      and `CommitOperationDelete` remain imported.
- [x] Confirm `fetch_arctic_to_hf.py` still calls
      `dataset.push_to_hub(repo_id=HF_DATASET_REPO, split=split_name,
      private=True)` unchanged, with `HF_TOKEN` auto-read from the env var.
- [x] Confirm `consolidate_to_hf.py` still calls `api.upload_file(...)` and
      `api.create_commit(...)` unchanged via `HfApi()`, with `HF_TOKEN`
      auto-read from the env var.
- [x] Confirm neither script performs a whoami/`login()` request at startup.
- [x] Validate the OpenSpec change with
      `openspec validate fix-remove-login-from-scripts`.
