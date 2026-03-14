from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CatalystRecord(BaseModel):
    catalyst_id: str
    element: Literal["Cu"] = "Cu"
    facet: str
    adsorbate: str = Field(default="CO")
    coordination_number: float
    avg_neighbor_distance: float
    electronegativity: float
    d_band_center: float
    surface_energy: float
    adsorption_energy: float
    provenance: str
    unit_adsorption_energy: str = "eV"
