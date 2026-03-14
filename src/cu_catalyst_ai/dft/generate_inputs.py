from __future__ import annotations

import json
from pathlib import Path

from cu_catalyst_ai.utils.io import ensure_parent


def write_placeholder_input(catalyst_id: str, output_dir: str) -> Path:
    payload = {
        "catalyst_id": catalyst_id,
        "code": "VASP",
        "status": "placeholder_input_written",
    }
    path = ensure_parent(Path(output_dir) / catalyst_id / "input.json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
