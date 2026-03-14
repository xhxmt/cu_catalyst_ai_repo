from cu_catalyst_ai.clean.split_registry import assign_splits
from cu_catalyst_ai.dataio.mp_fetch import generate_demo_dataset
from cu_catalyst_ai.features.basic_features import build_feature_table
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
