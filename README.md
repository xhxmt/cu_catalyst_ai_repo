# Cu Catalyst AI

A pure-Python research workflow for Cu-catalyst screening with an ML-first roadmap and DFT-ready extension points.

## What this repository includes

- Demo data generation so the pipeline runs immediately
- Cleaning, split assignment, and feature engineering
- Baseline ML training for linear, random forest, xgboost, and GPR models
- Cross-validation metrics, parity plots, learning curves, and feature-importance summaries
- Placeholder DFT modules for later semi-automation

## Quick start

```bash
uv sync
uv run python -m cu_catalyst_ai.cli task=fetch
uv run python -m cu_catalyst_ai.cli task=clean
uv run python -m cu_catalyst_ai.cli task=featurize
uv run python -m cu_catalyst_ai.cli task=train model=rf
uv run python -m cu_catalyst_ai.cli task=explain model=rf
uv run python -m cu_catalyst_ai.cli task=report model=rf
uv run pytest
```

## Main commands

```bash
uv run python -m cu_catalyst_ai.cli task=fetch data=demo
uv run python -m cu_catalyst_ai.cli task=train model=linear
uv run python -m cu_catalyst_ai.cli task=train model=rf
uv run python -m cu_catalyst_ai.cli task=train model=xgb
uv run python -m cu_catalyst_ai.cli task=train model=gpr
```

## Repository layout

- `configs/` contains Hydra configuration
- `src/cu_catalyst_ai/` contains production code
- `tests/` contains unit tests
- `data/` stores raw, interim, and processed artifacts
- `reports/` stores plots, tables, and summary bundles

## Notes

The default data source is `demo`, which generates a synthetic Cu-catalyst dataset so the workflow is runnable without API keys.
For Materials Project access, set `MP_API_KEY` and extend `src/cu_catalyst_ai/dataio/mp_fetch.py` as needed.
