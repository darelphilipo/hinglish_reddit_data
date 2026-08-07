# Tasks

- [x] Add a standalone GitHub Actions workflow with a `workflow_dispatch`
      trigger only.
- [x] Configure the workflow to check out the repository and set up Python
      3.10.
- [x] Install the dependencies required by `consolidate_to_hf.py`.
- [x] Run `consolidate_to_hf.py` with `HF_TOKEN` from GitHub Actions secrets
      and unbuffered Python output.
- [x] Confirm the existing monthly workflow and application files are
      unchanged.
- [x] Validate the OpenSpec change with
      `openspec validate e1-standalone-consolidation-workflow`.
