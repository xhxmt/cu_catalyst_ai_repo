from __future__ import annotations

from cu_catalyst_ai.schemas.dft_result import DFTResult


def basic_sanity_check(result: DFTResult) -> bool:
    return result.converged and abs(result.total_energy) < 1e6
