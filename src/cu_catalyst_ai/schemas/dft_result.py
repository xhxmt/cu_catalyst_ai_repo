from __future__ import annotations

from pydantic import BaseModel


class DFTResult(BaseModel):
    catalyst_id: str
    converged: bool
    total_energy: float
    adsorption_energy: float | None = None
    code: str = "VASP"
