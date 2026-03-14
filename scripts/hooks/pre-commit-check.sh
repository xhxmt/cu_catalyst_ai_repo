#!/usr/bin/env bash
set -euo pipefail

if command -v ruff >/dev/null 2>&1; then
  ruff check . || true
fi

if command -v mypy >/dev/null 2>&1; then
  mypy . || true
fi
