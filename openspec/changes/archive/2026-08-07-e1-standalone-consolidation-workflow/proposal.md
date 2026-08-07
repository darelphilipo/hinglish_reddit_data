**Priority:** 3/5

# Proposal: Standalone Consolidation Workflow

Add a separate, manually triggered GitHub Actions workflow that runs
`consolidate_to_hf.py` independently of the monthly scrape workflow.

This provides an operator-controlled way to consolidate pending temporary
batch files on Hugging Face without starting a scrape or waiting for the
monthly workflow.

## Scope

- Add a new workflow dedicated to consolidation.
- Trigger it only through `workflow_dispatch`.
- Reuse the repository's existing Python setup, dependencies, and `HF_TOKEN`
  secret needed by `consolidate_to_hf.py`.
- Leave the existing monthly workflow and application code unchanged.

## Out Of Scope

- Changes to `consolidate_to_hf.py` behavior.
- Changes to the monthly scrape schedule or job dependencies.
- New workflow inputs or consolidation options.
