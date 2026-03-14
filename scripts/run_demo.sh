#!/usr/bin/env bash
set -euo pipefail
uv run python -m cu_catalyst_ai.cli task=fetch
uv run python -m cu_catalyst_ai.cli task=clean
uv run python -m cu_catalyst_ai.cli task=featurize
uv run python -m cu_catalyst_ai.cli task=train model=rf
uv run python -m cu_catalyst_ai.cli task=explain model=rf
uv run python -m cu_catalyst_ai.cli task=report model=rf
