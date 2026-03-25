"""MamunHighT2019 dataset loader.

Reads O and OH adsorption energy data from the local ``full_dataset.csv``
file (MamunHighT2019; DOI 10.1021/acscatal.0c01752) and outputs a
DataFrame in the project's canonical raw-table schema.

Key decisions:
- Only pure d-metals are included; sp-metals (Al, Ga, Zn, etc.) are
  excluded because they lack a meaningful d-band centre descriptor.
- All records are FCC (111) slabs → ``facet = "111"`` is hard-coded.
- All records use BEEF-vdW → ``dft_functional = "BEEF-vdW"`` is hard-coded.
- Target definition is set to ``"adsorption_energy_ev_multi_v1"`` which
  accepts multiple adsorbates (required_adsorbate = null).

Schema alignment
----------------
The output DataFrame has the same canonical columns as
:func:`cu_catalyst_ai.dataio.cathub_fetch.parse_cathub_response`, enabling
a clean ``pd.concat`` with the CatHub CO data.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# sp-metals in the Mamun dataset that lack a d-band centre and should be
# excluded from ML training.
SP_METALS: frozenset[str] = frozenset(
    {"Al", "Ga", "Bi", "Cd", "Hg", "In", "Sn", "Tl", "Zn", "Pb", "Ge"}
)

_TARGET_ADSORBATES: frozenset[str] = frozenset({"O", "OH"})

_TARGET_DEFINITION = "adsorption_energy_ev_multi_v1"
_PROVENANCE = "MamunHighT2019|10.1021/acscatal.0c01752|2020"

# Canonical column order — must match cathub_fetch._CANONICAL_COLUMNS.
_CANONICAL_COLUMNS = [
    "catalyst_id",
    "element",
    "facet",
    "adsorbate",
    "coordination_number",
    "avg_neighbor_distance",
    "electronegativity",
    "d_band_center",
    "surface_energy",
    "adsorption_energy",
    "provenance",
    "unit_adsorption_energy",
    "target_definition",
    "dft_functional",
]


def _is_pure_metal(formula: str) -> bool:
    """Return True if *formula* is a single element symbol (e.g. ``"Cu"``)."""
    return bool(re.match(r"^[A-Z][a-z]?$", str(formula).strip()))


def load_mamun_ooh(csv_path: str | Path) -> pd.DataFrame:
    """Load O and OH adsorption energy rows from the Mamun dataset CSV.

    Reads ``full_dataset.csv``, filters to O/OH adsorbates on pure d-metals,
    and returns a DataFrame aligned to the project's canonical raw-table schema
    so it can be concatenated with CatHub CO data.

    Args:
        csv_path: Path to ``full_dataset.csv``.

    Returns:
        ``pd.DataFrame`` with canonical project columns.  Only ``element``,
        ``facet``, ``adsorbate``, ``adsorption_energy``, ``dft_functional``,
        ``target_definition``, and ``provenance`` are populated; the remaining
        columns (``d_band_center``, ``electronegativity``, etc.) are NaN and
        are filled by the downstream featurisation stage.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        KeyError: If expected columns (``reaction_energy``, ``surface_composition``,
            ``ads_1``, ``site_1``) are missing from the CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Mamun dataset not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    logger.info("Mamun CSV loaded: %d rows, columns=%s", len(raw), raw.columns.tolist())

    required_cols = {"reaction_energy", "surface_composition", "ads_1", "site_1"}
    missing = required_cols - set(raw.columns)
    if missing:
        raise KeyError(f"Missing expected columns in Mamun CSV: {missing}")

    # --- Filter: target adsorbates only ---
    mask_ads = raw["ads_1"].isin(_TARGET_ADSORBATES)
    logger.info(
        "Mamun: %d/%d rows have ads_1 in %s",
        mask_ads.sum(),
        len(raw),
        _TARGET_ADSORBATES,
    )
    df = raw[mask_ads].copy()

    # --- Filter: pure metals only ---
    mask_pure = df["surface_composition"].apply(_is_pure_metal)
    df = df[mask_pure].copy()

    # --- Filter: d-metals only (exclude sp-metals) ---
    mask_sp = df["surface_composition"].isin(SP_METALS)
    n_sp = mask_sp.sum()
    if n_sp:
        sp_elements = sorted(df.loc[mask_sp, "surface_composition"].unique())
        logger.info(
            "Mamun: excluding %d rows from sp-metals %s (no d-band centre)",
            n_sp,
            sp_elements,
        )
    df = df[~mask_sp].copy()

    logger.info(
        "Mamun: %d rows after filtering (O: %d, OH: %d)",
        len(df),
        (df["ads_1"] == "O").sum(),
        (df["ads_1"] == "OH").sum(),
    )

    if df.empty:
        logger.warning("Mamun loader returned 0 rows after all filters.")
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)

    # --- Build canonical rows ---
    rows = []
    for idx, row in df.iterrows():
        element = str(row["surface_composition"]).strip()
        adsorbate = str(row["ads_1"]).strip()
        site_1 = str(row.get("site_1", "")).strip()
        # Build a deterministic catalyst_id
        catalyst_id = re.sub(r"[^\w]", "_", f"mamun_{element}_{adsorbate}_{site_1}_{idx}")

        rows.append(
            {
                "catalyst_id": catalyst_id,
                "element": element,
                "facet": "111",  # MamunHighT2019 is FCC 111
                "adsorbate": adsorbate,
                "coordination_number": float("nan"),
                "avg_neighbor_distance": float("nan"),
                "electronegativity": float("nan"),  # filled by enrich_with_element_features
                "d_band_center": float("nan"),  # filled by enrich_with_element_features
                "surface_energy": float("nan"),
                "adsorption_energy": row["reaction_energy"],
                "provenance": _PROVENANCE,
                "unit_adsorption_energy": "eV",
                "target_definition": _TARGET_DEFINITION,
                "dft_functional": "BEEF-vdW",
            }
        )

    out = pd.DataFrame(rows, columns=_CANONICAL_COLUMNS)
    out["adsorption_energy"] = pd.to_numeric(out["adsorption_energy"], errors="coerce")

    n_nan = out["adsorption_energy"].isna().sum()
    if n_nan:
        logger.warning("Mamun: dropping %d rows with non-numeric adsorption_energy", n_nan)
        out = out.dropna(subset=["adsorption_energy"])

    logger.info(
        "Mamun loader complete: %d rows, elements=%s",
        len(out),
        sorted(out["element"].unique()),
    )
    return out
