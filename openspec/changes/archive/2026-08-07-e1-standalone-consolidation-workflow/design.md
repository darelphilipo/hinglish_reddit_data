# Design: Standalone Consolidation Workflow

## Workflow Shape

Create one new workflow under `.github/workflows/` with a descriptive name
such as `Standalone Consolidation to Hugging Face`. Its only event is
`workflow_dispatch`, so it cannot run from the monthly schedule or as a
side-effect of another workflow.

The workflow contains one job on `ubuntu-latest` with a bounded timeout. The
job checks out the repository, installs Python 3.10, upgrades pip, and
installs `pandas`, `datasets`, and `huggingface_hub`. The final step runs:

```yaml
python consolidate_to_hf.py
```

That step exposes `secrets.HF_TOKEN` as `HF_TOKEN` and sets
`PYTHONUNBUFFERED=1`, matching the consolidation steps already present in
the monthly workflow.

## Independence

The new workflow has no `needs`, reusable-workflow call, matrix, scrape job,
or schedule. It therefore performs only consolidation when manually started.
The existing monthly workflow remains responsible for its current cleanup,
scrape, and post-scrape consolidation behavior.

## Failure And Empty-Queue Behavior

The workflow propagates the script's exit status. A missing `HF_TOKEN` or
processing failure makes the workflow fail. When no temporary batch files are
present, the script's existing graceful exit makes the manually triggered
workflow succeed without making application changes.

## Security

The Hugging Face token is passed through the job environment from the GitHub
Actions secret and is not written to logs or repository files.
