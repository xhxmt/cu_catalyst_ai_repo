"""Tests for assign_splits: per-element downsampling and stratified splitting."""

from __future__ import annotations

import pandas as pd

from cu_catalyst_ai.clean.split_registry import assign_splits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n_cu: int = 100, n_pt: int = 20, n_pd: int = 10) -> pd.DataFrame:
    """Build a synthetic DataFrame with Cu, Pt, Pd elements."""
    rows = []
    for i in range(n_cu):
        rows.append({"catalyst_id": f"Cu_{i}", "element": "Cu", "adsorption_energy": -0.5})
    for i in range(n_pt):
        rows.append({"catalyst_id": f"Pt_{i}", "element": "Pt", "adsorption_energy": -0.8})
    for i in range(n_pd):
        rows.append({"catalyst_id": f"Pd_{i}", "element": "Pd", "adsorption_energy": -0.7})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def test_assign_splits_downsamples_dominant_element() -> None:
    """Cu (100 rows) should be capped at max_samples_per_element."""
    df = _make_df(n_cu=100, n_pt=20, n_pd=10)
    out = assign_splits(df, seed=42, max_samples_per_element=30)
    cu_count = (out["element"] == "Cu").sum()
    assert cu_count <= 30, f"Expected Cu ≤ 30, got {cu_count}"


def test_assign_splits_minority_elements_preserved() -> None:
    """Elements with fewer rows than the cap should not be increased."""
    df = _make_df(n_cu=100, n_pt=20, n_pd=10)
    out = assign_splits(df, seed=42, max_samples_per_element=50)
    assert (out["element"] == "Pt").sum() == 20
    assert (out["element"] == "Pd").sum() == 10


def test_assign_splits_no_downsampling_when_none() -> None:
    """max_samples_per_element=None preserves original row count."""
    df = _make_df(n_cu=100, n_pt=20, n_pd=10)
    out = assign_splits(df, seed=42, max_samples_per_element=None)
    assert len(out) == 130


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


def test_assign_splits_stratified_all_elements_in_test() -> None:
    """Stratified split should put every element into both train and test."""
    df = _make_df(n_cu=100, n_pt=20, n_pd=10)
    out = assign_splits(df, seed=42, max_samples_per_element=50)
    test_df = out[out["split"] == "test"]
    train_df = out[out["split"] == "train"]
    for elem in ("Cu", "Pt", "Pd"):
        assert elem in test_df["element"].values, f"{elem} missing from test set"
        assert elem in train_df["element"].values, f"{elem} missing from train set"


def test_assign_splits_test_size_respected() -> None:
    """Test fraction should be close to the requested test_size."""
    df = _make_df(n_cu=100, n_pt=20, n_pd=10)
    out = assign_splits(df, seed=42, test_size=0.2, max_samples_per_element=None)
    test_frac = (out["split"] == "test").mean()
    assert abs(test_frac - 0.2) < 0.05, f"Test fraction {test_frac:.3f} too far from 0.2"


def test_assign_splits_reproducible() -> None:
    """Same seed must always produce identical splits."""
    df = _make_df()
    out1 = assign_splits(df, seed=42)
    out2 = assign_splits(df, seed=42)
    assert (out1["split"] == out2["split"]).all()


def test_assign_splits_split_column_values() -> None:
    """'split' column must only contain 'train' and 'test'."""
    df = _make_df()
    out = assign_splits(df, seed=42)
    assert set(out["split"].unique()).issubset({"train", "test"})


# ---------------------------------------------------------------------------
# Sample weight logic (unit-tested independently)
# ---------------------------------------------------------------------------


def test_sample_weight_sums_to_n() -> None:
    """Inverse-frequency weights must sum to N (total sample count)."""
    elements = ["Cu"] * 100 + ["Pt"] * 20
    train_df = pd.DataFrame({"element": elements})
    element_counts = train_df["element"].value_counts()
    N = len(train_df)
    K = len(element_counts)
    weights = train_df["element"].map(lambda x: N / (K * element_counts[x]))
    assert abs(weights.sum() - N) < 1e-6, f"Weights sum {weights.sum():.4f} ≠ {N}"


def test_sample_weight_minority_higher() -> None:
    """Minority element must receive strictly higher weight than majority."""
    elements = ["Cu"] * 100 + ["Pt"] * 20
    train_df = pd.DataFrame({"element": elements})
    element_counts = train_df["element"].value_counts()
    N = len(train_df)
    K = len(element_counts)
    weights = train_df["element"].map(lambda x: N / (K * element_counts[x]))
    w_cu = weights[train_df["element"] == "Cu"].iloc[0]
    w_pt = weights[train_df["element"] == "Pt"].iloc[0]
    assert w_pt > w_cu, f"Expected w_Pt ({w_pt:.3f}) > w_Cu ({w_cu:.3f})"


def test_sample_weight_inversely_proportional() -> None:
    """Weight × count must be equal for all elements (each element contributes equally)."""
    elements = ["Cu"] * 100 + ["Pt"] * 20 + ["Pd"] * 10
    train_df = pd.DataFrame({"element": elements})
    element_counts = train_df["element"].value_counts()
    N = len(train_df)
    K = len(element_counts)
    weights = train_df["element"].map(lambda x: N / (K * element_counts[x]))
    # Each element's total weight contribution = N/K (a constant)
    for elem in ("Cu", "Pt", "Pd"):
        mask = train_df["element"] == elem
        total_w = weights[mask].sum()
        expected = N / K
        assert abs(total_w - expected) < 1e-6, (
            f"Element {elem}: total_weight {total_w:.4f} ≠ {expected:.4f}"
        )
