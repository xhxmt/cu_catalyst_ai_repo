#!/usr/bin/env bash
set -euo pipefail
uv sync
uv run ruff check . || true
uv run pytest || true
