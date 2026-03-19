"""Element-level physical-chemistry feature lookup table.

Provides a static, offline mapping from transition-metal element symbols to
five key descriptors used in heterogeneous-catalysis ML models:

d_band_center (eV)
    Bulk d-band centre relative to the Fermi level.
    Source: Ruban A.V., Hammer B., Stoltze P., Skriver H.L., Nørskov J.K.,
    *J. Mol. Catal. A*, **115**, 421–429 (1997), Table 1.

work_function (eV)
    Polycrystalline average work function.
    Source: CRC Handbook of Chemistry and Physics, 97th ed. (2016), §12-114.

electronegativity (Pauling scale)
    Source: Allred A.L., *J. Inorg. Nucl. Chem.*, **17**, 215 (1961).

atomic_radius_pm (pm)
    Covalent atomic radius.
    Source: Alvarez S. et al., *Dalton Trans.*, 2832 (2008).

d_electron_count (int-as-float)
    Formal d-electron count from ground-state electron configuration.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static lookup table
# Keys: canonical element symbols (str).
# Values: dict with the five descriptors; None = not available.
# ---------------------------------------------------------------------------

_ELEMENT_DATA: dict[str, dict[str, float | None]] = {
    # d_band_center: Ruban et al. 1997, J. Mol. Catal. A 115, 421 (Table 1)
    # work_function: CRC Handbook 97th ed.; electronegativity: Allred 1961;
    # atomic_radius_pm: Alvarez 2008; d_electron_count: NIST ground state config
    "Sc": {
        "d_band_center": -2.13,
        "work_function": 3.50,
        "electronegativity": 1.36,
        "atomic_radius_pm": 170,
        "d_electron_count": 1,
    },
    "Ti": {
        "d_band_center": -2.76,
        "work_function": 4.33,
        "electronegativity": 1.54,
        "atomic_radius_pm": 160,
        "d_electron_count": 2,
    },
    "V": {
        "d_band_center": -2.65,
        "work_function": 4.30,
        "electronegativity": 1.63,
        "atomic_radius_pm": 153,
        "d_electron_count": 3,
    },
    "Cr": {
        "d_band_center": -2.55,
        "work_function": 4.50,
        "electronegativity": 1.66,
        "atomic_radius_pm": 139,
        "d_electron_count": 5,
    },
    "Mn": {
        "d_band_center": -2.18,
        "work_function": 4.10,
        "electronegativity": 1.55,
        "atomic_radius_pm": 150,
        "d_electron_count": 5,
    },
    "Fe": {
        "d_band_center": -2.29,
        "work_function": 4.67,
        "electronegativity": 1.83,
        "atomic_radius_pm": 142,
        "d_electron_count": 6,
    },
    "Co": {
        "d_band_center": -2.35,
        "work_function": 5.00,
        "electronegativity": 1.88,
        "atomic_radius_pm": 138,
        "d_electron_count": 7,
    },
    "Ni": {
        "d_band_center": -1.29,
        "work_function": 5.15,
        "electronegativity": 1.91,
        "atomic_radius_pm": 124,
        "d_electron_count": 8,
    },
    "Cu": {
        "d_band_center": -2.67,
        "work_function": 4.65,
        "electronegativity": 1.90,
        "atomic_radius_pm": 138,
        "d_electron_count": 10,
    },
    "Zn": {
        "d_band_center": -6.53,
        "work_function": 3.63,
        "electronegativity": 1.65,
        "atomic_radius_pm": 131,
        "d_electron_count": 10,
    },
    "Nb": {
        "d_band_center": -3.20,
        "work_function": 4.02,
        "electronegativity": 1.60,
        "atomic_radius_pm": 164,
        "d_electron_count": 4,
    },
    "Mo": {
        "d_band_center": -2.95,
        "work_function": 4.60,
        "electronegativity": 2.16,
        "atomic_radius_pm": 154,
        "d_electron_count": 5,
    },
    "Tc": {
        "d_band_center": -2.72,
        "work_function": 5.00,
        "electronegativity": 1.90,
        "atomic_radius_pm": 147,
        "d_electron_count": 6,
    },
    "Ru": {
        "d_band_center": -2.04,
        "work_function": 4.71,
        "electronegativity": 2.20,
        "atomic_radius_pm": 146,
        "d_electron_count": 7,
    },
    "Rh": {
        "d_band_center": -1.73,
        "work_function": 4.98,
        "electronegativity": 2.28,
        "atomic_radius_pm": 142,
        "d_electron_count": 8,
    },
    "Pd": {
        "d_band_center": -1.83,
        "work_function": 5.12,
        "electronegativity": 2.20,
        "atomic_radius_pm": 139,
        "d_electron_count": 10,
    },
    "Ag": {
        "d_band_center": -4.30,
        "work_function": 4.26,
        "electronegativity": 1.93,
        "atomic_radius_pm": 145,
        "d_electron_count": 10,
    },
    "Ta": {
        "d_band_center": -3.25,
        "work_function": 4.25,
        "electronegativity": 1.50,
        "atomic_radius_pm": 170,
        "d_electron_count": 3,
    },
    "W": {
        "d_band_center": -2.86,
        "work_function": 4.55,
        "electronegativity": 2.36,
        "atomic_radius_pm": 162,
        "d_electron_count": 4,
    },
    "Re": {
        "d_band_center": -2.58,
        "work_function": 4.96,
        "electronegativity": 1.90,
        "atomic_radius_pm": 151,
        "d_electron_count": 5,
    },
    "Os": {
        "d_band_center": -2.22,
        "work_function": 5.93,
        "electronegativity": 2.20,
        "atomic_radius_pm": 144,
        "d_electron_count": 6,
    },
    "Ir": {
        "d_band_center": -2.11,
        "work_function": 5.27,
        "electronegativity": 2.20,
        "atomic_radius_pm": 141,
        "d_electron_count": 7,
    },
    "Pt": {
        "d_band_center": -2.25,
        "work_function": 5.65,
        "electronegativity": 2.28,
        "atomic_radius_pm": 136,
        "d_electron_count": 9,
    },
    "Au": {
        "d_band_center": -3.56,
        "work_function": 5.10,
        "electronegativity": 2.54,
        "atomic_radius_pm": 136,
        "d_electron_count": 10,
    },
}

_FEATURE_KEYS: list[str] = [
    "d_band_center",
    "work_function",
    "electronegativity",
    "atomic_radius_pm",
    "d_electron_count",
]


def get_element_features(element: str) -> dict[str, float | None]:
    """Return physical-chemistry descriptors for *element*.

    Args:
        element: Chemical element symbol, e.g. ``"Cu"``, ``"Pt"``.

    Returns:
        Dict with keys ``d_band_center``, ``work_function``,
        ``electronegativity``, ``atomic_radius_pm``, ``d_electron_count``.
        All values are ``float`` or ``None`` when the element is not in the
        lookup table.  ``None`` values represent missing data — they are
        **never faked**.
    """
    data = _ELEMENT_DATA.get(element)
    if data is None:
        logger.warning(
            "Element %r not found in element feature lookup table. "
            "All descriptors will be NaN for this element. "
            "Add it to element_features._ELEMENT_DATA to suppress this warning.",
            element,
        )
        return {k: None for k in _FEATURE_KEYS}
    return dict(data)


def enrich_with_element_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join element descriptors onto *df* by the ``element`` column.

    For each row the five descriptors are looked up from the static table and
    injected as new columns.  Existing columns with the same name are
    **overwritten** — the table values are the authoritative source for these
    descriptors.

    Args:
        df: DataFrame that must contain an ``element`` column with chemical
            element symbols as strings.

    Returns:
        A copy of *df* with the five element-feature columns added/updated.
        Rows whose element is not found receive ``NaN`` for all five columns.

    Raises:
        KeyError: If ``df`` does not contain an ``element`` column.
    """
    if "element" not in df.columns:
        raise KeyError("DataFrame must contain an 'element' column.")

    out = df.copy()

    # Build a lookup DataFrame aligned to df.index.
    feature_rows = [get_element_features(str(el)) for el in out["element"]]
    feature_df = pd.DataFrame(feature_rows, index=out.index)

    for col in _FEATURE_KEYS:
        out[col] = pd.to_numeric(feature_df[col], errors="coerce")

    missing_elements = out.loc[out["d_band_center"].isna(), "element"].unique()
    if len(missing_elements) > 0:
        logger.warning(
            "Element feature enrichment: %d elements missing from table: %s",
            len(missing_elements),
            list(missing_elements),
        )

    logger.info(
        "Element feature enrichment complete. d_band_center present: %d/%d rows.",
        out["d_band_center"].notna().sum(),
        len(out),
    )
    return out
