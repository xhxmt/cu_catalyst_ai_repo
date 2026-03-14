from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ProjectPaths(BaseModel):
    raw_dir: Path = Field(default=Path("data/raw"))
    interim_dir: Path = Field(default=Path("data/interim"))
    processed_dir: Path = Field(default=Path("data/processed"))
    reports_dir: Path = Field(default=Path("reports"))
    figures_dir: Path = Field(default=Path("reports/figures"))
    tables_dir: Path = Field(default=Path("reports/tables"))
    models_dir: Path = Field(default=Path("reports/models"))
    dft_dir: Path = Field(default=Path("data/external/dft"))
