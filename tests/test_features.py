import pandas as pd

from cu_catalyst_ai.clean.split_registry import assign_splits
from cu_catalyst_ai.dataio.mp_fetch import generate_demo_dataset
from cu_catalyst_ai.features.basic_features import add_gcn, add_proxy_cn, build_feature_table
from cu_catalyst_ai.features.structural_features import add_structural_ratios


def test_feature_table_contains_expected_columns() -> None:
    df = generate_demo_dataset(n_samples=20, seed=42)
    df = assign_splits(df, seed=42)
    df = add_structural_ratios(df)
    features = build_feature_table(
        df,
        use_columns=[
            "coordination_number",
            "avg_neighbor_distance",
            "electronegativity",
            "d_band_center",
            "surface_energy",
            "coordination_to_distance",
            "facet",
        ],
        categorical_columns=["facet"],
    )
    assert "facet_111" in features.columns
    assert "coordination_to_distance" in features.columns


def test_add_proxy_cn_maps_known_facets() -> None:
    """GCN values follow Calle-Vallejo 2015 for known facets; NaN and unknown use default."""
    df = pd.DataFrame({"facet": ["111", "310", None, "unknown"]})
    out = add_proxy_cn(df)
    assert "proxy_cn" in out.columns
    assert list(out["proxy_cn"]) == [7.5, 4.4, 6.0, 6.0], out["proxy_cn"].tolist()


def test_add_proxy_cn_noop_without_facet() -> None:
    """add_proxy_cn is a noop when the 'facet' column is absent."""
    df = pd.DataFrame({"electronegativity": [1.9, 2.0]})
    out = add_proxy_cn(df)
    assert "proxy_cn" not in out.columns
    # Original data unchanged
    assert list(out.columns) == ["electronegativity"]


def test_add_gcn_maps_known_facets() -> None:
    """add_gcn() returns Calle-Vallejo 2015 float values for known facets."""
    df = pd.DataFrame({"facet": ["111", "100", "211", "110", "310", None, "unknown"]})
    out = add_gcn(df)
    assert "gcn" in out.columns, "gcn column must be present"
    expected = [7.5, 6.7, 5.3, 6.0, 4.4, 6.0, 6.0]
    assert list(out["gcn"]) == expected, f"Got {out['gcn'].tolist()}"


def test_add_gcn_independent_of_proxy_cn() -> None:
    """add_gcn() writes to 'gcn' without touching 'proxy_cn'."""
    df = pd.DataFrame({"facet": ["111", "211"]})
    df = add_proxy_cn(df)
    df = add_gcn(df)
    assert "proxy_cn" in df.columns
    assert "gcn" in df.columns
    assert list(df["proxy_cn"]) == list(df["gcn"])


def test_add_gcn_noop_without_facet() -> None:
    """add_gcn() is a noop when 'facet' column is absent."""
    df = pd.DataFrame({"electronegativity": [1.8, 2.2]})
    out = add_gcn(df)
    assert "gcn" not in out.columns
    assert list(out.columns) == ["electronegativity"]
