#!/usr/bin/env bash
set -euo pipefail

if command -v prettier >/dev/null 2>&1; then
  prettier --write . >/dev/null 2>&1 || true
fi

if command -v black >/dev/null 2>&1; then
  black . >/dev/null 2>&1 || true
fi
