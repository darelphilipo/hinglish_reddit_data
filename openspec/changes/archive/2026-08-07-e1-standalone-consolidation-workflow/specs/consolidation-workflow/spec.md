# Consolidation Workflow Specification

## ADDED Requirements

### Requirement: Manual standalone consolidation workflow

The repository SHALL provide a separate GitHub Actions workflow that runs
`consolidate_to_hf.py` without running the scrape workflow or depending on
any scrape job.

#### Scenario: Operator starts standalone consolidation

- **WHEN** an authorized operator manually dispatches the standalone
  consolidation workflow
- **THEN** GitHub Actions checks out the repository, installs Python 3.10
  and the consolidation dependencies, and runs `consolidate_to_hf.py`

#### Scenario: Workflow is not started automatically

- **WHEN** the monthly schedule occurs or another job in the monthly
  workflow runs
- **THEN** the standalone consolidation workflow is not triggered

### Requirement: Consolidation credentials are provided securely

The workflow SHALL provide the repository's `HF_TOKEN` GitHub Actions secret
to `consolidate_to_hf.py` through the `HF_TOKEN` environment variable.

#### Scenario: Consolidation accesses Hugging Face

- **WHEN** the manually dispatched workflow runs the consolidation step
- **THEN** `HF_TOKEN` is available to the script from
  `${{ secrets.HF_TOKEN }}` without hard-coding the token

### Requirement: Script result determines workflow result

The workflow SHALL preserve the exit status of `consolidate_to_hf.py`.

#### Scenario: Consolidation succeeds with pending files

- **WHEN** the script completes consolidation successfully
- **THEN** the workflow run succeeds

#### Scenario: No temporary files are pending

- **WHEN** the script finds no temporary batch files
- **THEN** the script exits gracefully and the workflow run succeeds

#### Scenario: Consolidation fails

- **WHEN** the script exits with a failure status
- **THEN** the workflow run is marked as failed
