from __future__ import annotations

import json
from pathlib import Path

from cu_catalyst_ai.schemas.dft_result import DFTResult


def parse_placeholder_output(path: str | Path) -> DFTResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DFTResult(**payload)
