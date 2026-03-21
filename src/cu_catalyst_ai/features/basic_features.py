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
