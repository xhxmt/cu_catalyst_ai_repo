from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError


class CatalystRecord(BaseModel):
    """Pydantic schema for a single catalyst surface record.

    The ``target_definition`` field records which registered target definition
    was active when this record was ingested.  It is used only in the cleaning
    governance layer and is not included in the ML feature table.
    """

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
    target_definition: str | None = None


def validate_schema_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Validate each row against :class:`CatalystRecord`; flag failures for review.

    Rows that fail Pydantic validation are annotated with ``review_reason``
    (prefixed ``"schema_validation: ..."``) and ``review_stage = "schema_validation"``
    via the same :func:`~cu_catalyst_ai.clean.governance.flag_rows` mechanism used
    by all other governance checks.  Already-flagged rows are skipped (first-check-wins).

    Args:
        df: DataFrame that has passed all prior cleaning layers.

    Returns:
        DataFrame with ``review_reason`` / ``review_stage`` columns updated for
        any rows that fail schema validation.
    """
    from cu_catalyst_ai.clean.governance import REVIEW_REASON_COL, REVIEW_STAGE_COL  # noqa: PLC0415

    out = df.copy()

    # Ensure governance columns exist so we can check them inline.
    if REVIEW_REASON_COL not in out.columns:
        out[REVIEW_REASON_COL] = pd.NA
    if REVIEW_STAGE_COL not in out.columns:
        out[REVIEW_STAGE_COL] = pd.NA

    for idx, row in out.iterrows():
        # Skip rows already flagged by an earlier governance stage.
        if pd.notna(out.at[idx, REVIEW_REASON_COL]):
            continue
        try:
            CatalystRecord(**row.to_dict())
        except ValidationError as exc:
            first_msg = exc.errors()[0]["msg"]
            out.at[idx, REVIEW_REASON_COL] = f"schema_validation: {first_msg}"
            out.at[idx, REVIEW_STAGE_COL] = "schema_validation"
        except Exception as exc:  # noqa: BLE001
            out.at[idx, REVIEW_REASON_COL] = f"schema_validation: {exc!s:.120}"
            out.at[idx, REVIEW_STAGE_COL] = "schema_validation"

    return out
