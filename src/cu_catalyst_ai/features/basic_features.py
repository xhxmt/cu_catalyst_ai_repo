from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generalised Coordination Number (GCN) mapping
# Source: Calle-Vallejo et al., Nat. Chem. 2015, DOI 10.1038/nchem.2226
# Values are GCN approximations for FCC metal surface sites, rounded to 1 d.p.
# ---------------------------------------------------------------------------
_GCN_MAP: dict[str, float] = {
    "111": 7.5,  # close-packed terrace
    "0001": 7.5,  # HCP close-packed (mapped as equivalent)
    "100": 6.7,  # square lattice
    "001": 6.7,
    "110": 6.0,  # ridge / groove
    "211": 5.3,  # step-edge
    "311": 5.3,
    "332": 5.8,  # wide terrace step
    "511": 5.8,
    "310": 4.4,  # kinked / high-index
    "210": 4.4,
    "321": 4.4,
}
_GCN_DEFAULT: float = 6.0  # neutral mid-range (tentative; may underestimate dense faces)


def _facet_to_gcn(facet: object) -> float:
    """Convert a raw facet label to its GCN approximation.

    Extracts the *leading digit sequence* from *facet* (e.g. ``"211-(2x1)"``
    yields key ``"211"``), so that common surface-science suffix notations are
    handled correctly.  Returns ``_GCN_DEFAULT`` for NaN / missing / unknown.
    """
    if facet is None or (isinstance(facet, float) and __import__("math").isnan(facet)):
        return _GCN_DEFAULT
    s = str(facet).strip()
    m = re.match(r"^(\d+)", s)
    if m:
        key = m.group(1)
        return _GCN_MAP.get(key, _GCN_DEFAULT)
    return _GCN_DEFAULT


def add_proxy_cn(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``proxy_cn`` column encoding the Generalised Coordination Number.

    The GCN approximations are taken from:
        Calle-Vallejo, Martínez, García-Lastra, Sautet & Waquier,
        *Nature Chemistry* **7**, 403–410 (2015).
        DOI: 10.1038/nchem.2226

    Parameters
    ----------
    df:
        Input DataFrame.  Must contain a ``"facet"`` column to produce
        ``proxy_cn``; returns *df* unchanged when the column is absent
        (noop — safe for all data sources).

    Returns
    -------
    pd.DataFrame
        *df* with an additional ``proxy_cn`` column (float64), or *df*
        unmodified if ``"facet"`` is absent.
    """
    if "facet" not in df.columns:
        return df
    df = df.copy()
    df["proxy_cn"] = df["facet"].apply(_facet_to_gcn)
    return df


def add_gcn(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``gcn`` column with the Generalised Coordination Number.

    Functionally identical to :func:`add_proxy_cn` but writes to the
    independent column ``gcn``.  This allows G-group experiments to reference
    ``gcn`` without touching the ``proxy_cn`` values used by A–F experiments,
    preserving full reproducibility of the historical ablation chain.

    Parameters
    ----------
    df:
        Input DataFrame.  Requires a ``"facet"`` column; returns *df*
        unchanged when the column is absent.

    Returns
    -------
    pd.DataFrame
        *df* with an additional ``gcn`` column (float64).
    """
    if "facet" not in df.columns:
        return df
    df = df.copy()
    df["gcn"] = df["facet"].apply(_facet_to_gcn)
    return df


def add_adsorbate_ohe(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot encoded adsorbate indicator columns.

    Adds three binary columns ``is_CO``, ``is_O``, ``is_OH`` derived from
    the ``adsorbate`` column.  Values are 1 when the row's adsorbate matches
    and 0 otherwise.

    This is a **noop** when the ``adsorbate`` column is absent, so calling it
    on CO-only CatHub data (which may not carry the column through the entire
    pipeline) is safe and does not break A–H experiments.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``is_CO``, ``is_O``, ``is_OH`` columns added,
        or *df* unchanged when ``adsorbate`` column is absent.
    """
    if "adsorbate" not in df.columns:
        return df
    df = df.copy()
    for ads in ("CO", "O", "OH"):
        df[f"is_{ads}"] = (df["adsorbate"] == ads).astype(int)
    return df


# ---------------------------------------------------------------------------
# Surface-level d-band centre lookup: (element, facet) → value (eV)
# ---------------------------------------------------------------------------
# Value categories (annotated inline):
#   [L]  Exact literature value
#   [E]  Extrapolated from literature using CN-based shift model
#
# Cu(111): Ruban A.V. et al., J. Mol. Catal. A 115, 421 (1997) – Table 1.  [L]
# Cu(100), Cu(211): relative offsets from ACS Omega 2022 WGS DFT study.   [E]
# Cu(310), Cu(511): trend extrapolation following step-site sequence.       [E]
#
# Non-Cu metals, 111 face: taken directly from element_features._ELEMENT_DATA
#   (Ruban 1997 bulk close-packed values — same as 111 face for FCC metals). [L]
# Non-Cu metals, 211 face: 111 value + 0.07 eV uniform shift,
#   derived from the Cu(111)→Cu(211) shift (+0.07 eV) as a first-order
#   coordination-number-based approximation across metals.                   [E]
# Ru(001): HCP(0001) close-packed; treated as equivalent to FCC(111).       [L]
#
# 111-(NxN) supercell labels (e.g. "111-(1x1)") are identical sites to
# 111 — mapped to the same value.
# ---------------------------------------------------------------------------
_SURFACE_DBAND_MAP: dict[tuple[str, str], float] = {
    # ---- Cu (exact literature + controlled extrapolation) ------------------
    ("Cu", "111"): -2.67,  # [L] Ruban 1997
    ("Cu", "100"): -2.64,  # [E] ACS Omega 2022 WGS DFT
    ("Cu", "211"): -2.60,  # [E] ACS Omega 2022 WGS DFT
    ("Cu", "310"): -2.58,  # [E] trend extrapolation
    ("Cu", "511"): -2.56,  # [E] trend extrapolation
    # ---- Ag ----------------------------------------------------------------
    ("Ag", "111"): -4.30,  # [L] Ruban 1997
    ("Ag", "211"): -4.23,  # [E] Cu Δ(211-111) = +0.07 eV
    # ---- Au ----------------------------------------------------------------
    ("Au", "111"): -3.56,  # [L] Ruban 1997
    ("Au", "211"): -3.49,  # [E] +0.07 eV
    # ---- Co ----------------------------------------------------------------
    ("Co", "111"): -1.17,  # [L] Ruban 1997
    ("Co", "211"): -1.10,  # [E] +0.07 eV
    # ---- Fe ----------------------------------------------------------------
    ("Fe", "111"): -0.92,  # [L] Ruban 1997
    ("Fe", "211"): -0.85,  # [E] +0.07 eV
    # ---- Ir ----------------------------------------------------------------
    ("Ir", "111"): -2.11,  # [L] Ruban 1997
    ("Ir", "211"): -2.04,  # [E] +0.07 eV
    # ---- Ni ----------------------------------------------------------------
    ("Ni", "111"): -1.29,  # [L] Ruban 1997
    ("Ni", "211"): -1.22,  # [E] +0.07 eV
    # ---- Pd ----------------------------------------------------------------
    ("Pd", "111"): -1.83,  # [L] Ruban 1997
    ("Pd", "211"): -1.76,  # [E] +0.07 eV
    # ---- Pt ----------------------------------------------------------------
    ("Pt", "111"): -2.25,  # [L] Ruban 1997
    ("Pt", "211"): -2.18,  # [E] +0.07 eV
    # ---- Rh ----------------------------------------------------------------
    ("Rh", "111"): -1.73,  # [L] Ruban 1997
    ("Rh", "211"): -1.66,  # [E] +0.07 eV
    # ---- Ru (HCP; 001 = 0001 close-packed) ---------------------------------
    ("Ru", "001"): -1.41,  # [L] Ruban 1997 (HCP 0001 ≈ FCC 111)
    # ---- Additional d-metals from MamunHighT2019 (FCC 111 face) -----------
    # Values = Ruban 1997 bulk d_band_center (same scale as _ELEMENT_DATA).
    # Adding explicit (element,"111") entries lets add_surface_dband() resolve
    # these without falling back to the d_band_center column, ensuring
    # d_band_center and surface_d_band_center are consistent for 111 data.  [L]
    ("Sc", "111"): -2.13,
    ("Ti", "111"): -2.76,
    ("V", "111"): -2.65,
    ("Cr", "111"): -2.55,
    ("Mn", "111"): -2.18,
    ("Zr", "111"): -3.26,
    ("Nb", "111"): -3.20,
    ("Mo", "111"): -2.95,
    ("Tc", "111"): -2.72,
    ("Hf", "111"): -3.05,
    ("Ta", "111"): -3.25,
    ("W", "111"): -2.86,
    ("Re", "111"): -2.58,
    ("Os", "111"): -2.22,
    ("Y", "111"): -2.45,
}

# Supercell aliases: "111-(NxN)" labels map to the same sites as "111".
_SUPERCELL_ALIASES: dict[str, str] = {
    "111-(1x1)": "111",
    "111-(2x2)": "111",
    "111-(4x4)": "111",
    "0001": "111",
}


def add_surface_dband(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``surface_d_band_center`` column using ``(element, facet)`` lookup.

    Values sourced from:
    - Close-packed faces: Ruban et al., J. Mol. Catal. A 115, 421 (1997).
    - Cu open faces: ACS Omega 2022 WGS DFT study (relative shifts).
    - Cu(310/511) and non-Cu 211 faces: trend extrapolation using the
      Cu Δ(211−111) = +0.07 eV shift as a first-order CN-based approximation.

    Fallback logic (in priority order):
    1. ``(element, facet)`` found in ``_SURFACE_DBAND_MAP`` → table value.
    2. ``(element, canonical_facet)`` after resolving supercell alias → table value.
    3. ``(element, "111")`` → bulk close-packed value (if available in table).
    4. Row's existing ``d_band_center`` value (from ``element_features``).
    5. ``float("nan")`` if all else fails.

    Does **NOT** modify the existing ``d_band_center`` column.
    Safe to call when ``element`` or ``facet`` columns are absent (noop).

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an additional ``surface_d_band_center`` column.
    """
    if "element" not in df.columns or "facet" not in df.columns:
        return df

    df = df.copy()

    def _lookup(row: pd.Series) -> float:  # type: ignore[type-arg]
        element = str(row["element"])
        raw_facet = str(row["facet"]) if row["facet"] is not None else ""
        facet = _SUPERCELL_ALIASES.get(raw_facet, raw_facet)

        # 1. Direct lookup
        val = _SURFACE_DBAND_MAP.get((element, facet))
        if val is not None:
            return val

        # 2. Facet-stripping: extract leading digit sequence (e.g. "211-(2x1)" → "211")
        import re as _re  # noqa: PLC0415

        m = _re.match(r"^(\d+)", facet)
        if m:
            stripped = m.group(1)
            val = _SURFACE_DBAND_MAP.get((element, stripped))
            if val is not None:
                return val

        # 3. Fallback to the element's 111 (closest-packed) entry in the table.
        val = _SURFACE_DBAND_MAP.get((element, "111"))
        if val is not None:
            return val

        # 4. Fallback to existing d_band_center column.
        if "d_band_center" in row.index:
            dbc = row["d_band_center"]
            if dbc is not None and not (isinstance(dbc, float) and __import__("math").isnan(dbc)):
                return float(dbc)

        return float("nan")

    df["surface_d_band_center"] = df.apply(_lookup, axis=1)
    logger.info(
        "surface_d_band_center: %d/%d rows filled from table, %d fallback, %d NaN.",
        df["surface_d_band_center"].notna().sum(),
        len(df),
        (
            df.apply(
                lambda r: (
                    _SURFACE_DBAND_MAP.get(
                        (
                            str(r["element"]),
                            _SUPERCELL_ALIASES.get(str(r["facet"]), str(r["facet"])),
                        )
                    )
                    is None
                ),
                axis=1,
            ).sum()
        ),
        df["surface_d_band_center"].isna().sum(),
    )
    return df


def build_feature_table(
    df: pd.DataFrame, use_columns: list[str], categorical_columns: list[str]
) -> pd.DataFrame:
    """Build the ML-ready feature table from *df*.

    Drops any requested column that is entirely NaN (e.g. structural features
    absent from CatHub API records) so that linear models receive clean input
    and RF impurity scores are not distorted.

    The ``element`` column (if present) is always forwarded as a metadata
    column alongside ``catalyst_id``, ``adsorption_energy``, and ``split``.
    It is **not** treated as a feature — it is used downstream by
    :func:`~cu_catalyst_ai.models.train.train_model` to compute per-element
    inverse-frequency sample weights.
    """
    available = [c for c in use_columns if c in df.columns]
    all_nan = [c for c in available if df[c].isna().all()]
    if all_nan:
        logger.info("Dropping all-NaN columns from feature table: %s", all_nan)
    keep = [c for c in available if c not in all_nan]
    # Always retain element as a metadata column (needed for sample_weight in train.py).
    meta_cols = ["catalyst_id", "adsorption_energy", "split"]
    if "element" in df.columns:
        meta_cols.append("element")
    out = df[meta_cols + keep].copy()
    # Only one-hot encode categorical columns that are still present.
    cat_keep = [c for c in categorical_columns if c in keep]
    if cat_keep:
        out = pd.get_dummies(out, columns=cat_keep, drop_first=False)
    return out
