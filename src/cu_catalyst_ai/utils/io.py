from __future__ import annotations

from pathlib import Path

import pandas as pd

from cu_catalyst_ai.utils.paths import resolve_path

PARQUET_SUFFIXES = {".parquet", ".pq"}


def ensure_parent(path: str | Path) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_table(path: str | Path) -> pd.DataFrame:
    p = resolve_path(path)
    if p.suffix.lower() in PARQUET_SUFFIXES:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    p = ensure_parent(path)
    if p.suffix.lower() in PARQUET_SUFFIXES:
        df.to_parquet(p, index=index)
    else:
        df.to_csv(p, index=index)
    return p
