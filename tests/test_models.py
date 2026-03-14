from pathlib import Path

from cu_catalyst_ai.clean.split_registry import assign_splits
from cu_catalyst_ai.dataio.mp_fetch import generate_demo_dataset
from cu_catalyst_ai.features.basic_features import build_feature_table
from cu_catalyst_ai.features.structural_features import add_structural_ratios
from cu_catalyst_ai.models.train import train_model


def test_training_produces_metrics_and_predictions(tmp_path: Path) -> None:
    df = generate_demo_dataset(n_samples=80, seed=42)
    df = assign_splits(df, seed=42)
    df = add_structural_ratios(df)
    feature_df = build_feature_table(
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

    out = train_model(
        df=feature_df,
        model_name="rf",
        random_state=42,
        params={"n_estimators": 50, "max_depth": 4},
        target_col="adsorption_energy",
        n_splits=5,
        shuffle=True,
        cv_random_state=42,
        metrics_output=str(tmp_path / "metrics.csv"),
        model_output=str(tmp_path / "model.joblib"),
        predictions_output=str(tmp_path / "preds.csv"),
    )

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "preds.csv").exists()
    assert not out["metrics"].empty
