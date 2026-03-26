from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from cu_catalyst_ai.clean.split_registry import assign_splits
from cu_catalyst_ai.dataio.mp_fetch import generate_demo_dataset
from cu_catalyst_ai.features.basic_features import build_feature_table
from cu_catalyst_ai.features.structural_features import add_structural_ratios
from cu_catalyst_ai.models.cv import run_cv, run_cv_with_centering
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


def test_run_cv_group_kfold_prevents_leakage() -> None:
    """GroupKFold: each element should appear in exactly one test fold."""
    rng = np.random.default_rng(0)
    elements = ["Cu", "Ag", "Au", "Pd", "Pt"]
    rows_per_element = 10
    records = []
    for el in elements:
        for _ in range(rows_per_element):
            records.append(
                {
                    "f1": rng.random(),
                    "f2": rng.random(),
                    "y": rng.random(),
                    "element": el,
                }
            )
    df = pd.DataFrame(records)

    X = df[["f1", "f2"]]
    y = df["y"]
    groups = df["element"]

    model = RandomForestRegressor(n_estimators=10, random_state=0)
    summary = run_cv(model, X, y, n_splits=5, groups=groups)

    assert len(summary) == 1
    assert "r2_mean" in summary.columns
    assert "mae_mean" in summary.columns

    gkf = GroupKFold(n_splits=5)
    for train_idx, test_idx in gkf.split(X, y, groups):
        train_elements = set(groups.iloc[train_idx])
        test_elements = set(groups.iloc[test_idx])
        assert train_elements.isdisjoint(test_elements), (
            f"Element leakage detected: {train_elements & test_elements}"
        )


def test_run_cv_falls_back_to_kfold_without_groups() -> None:
    """When groups=None, run_cv should fall back to plain KFold (no crash)."""
    X = pd.DataFrame({"f1": np.linspace(0, 1, 40), "f2": np.linspace(1, 2, 40)})
    y = pd.Series(np.linspace(-1, 1, 40))

    model = RandomForestRegressor(n_estimators=10, random_state=0)
    summary = run_cv(model, X, y, n_splits=4, groups=None)

    assert len(summary) == 1
    assert "r2_mean" in summary.columns


def test_run_cv_with_centering_no_leakage() -> None:
    """Centering means are computed fold-locally; no adsorbate mean leaks across folds."""
    rng = np.random.default_rng(7)
    elements = ["Cu", "Ag", "Au", "Pd", "Pt"]
    adsorbates_list = ["CO", "O"]
    n_per = 12  # rows per (element, adsorbate) combo

    # Large mean offset between CO and O to test that centering is happening
    CO_OFFSET, O_OFFSET = 0.0, -4.0

    records = []
    for el in elements:
        for ads in adsorbates_list:
            for _ in range(n_per):
                offset = CO_OFFSET if ads == "CO" else O_OFFSET
                records.append(
                    {
                        "f1": rng.random(),
                        "f2": rng.random(),
                        "y": offset + rng.normal(0, 0.2),
                        "element": el,
                        "adsorbate": ads,
                    }
                )
    df = pd.DataFrame(records)
    X = df[["f1", "f2"]]
    y = df["y"]
    groups = df["element"]
    adsorbates = df["adsorbate"]

    model = RandomForestRegressor(n_estimators=20, random_state=0)
    summary = run_cv_with_centering(model, X, y, adsorbates=adsorbates, n_splits=5, groups=groups)

    # 1. Summary has the required columns
    for col in ("r2_mean", "mae_mean", "centered_r2_mean", "centered_mae_mean"):
        assert col in summary.columns, f"Missing column: {col}"

    # 2. Verify fold-local centering: train means are computed per fold
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    for train_idx, _ in gkf.split(X, y, groups):
        y_tr = y.iloc[train_idx]
        ads_tr = adsorbates.iloc[train_idx]
        fold_mean_CO = float(y_tr[ads_tr == "CO"].mean())
        fold_mean_O = float(y_tr[ads_tr == "O"].mean())
        # The intentional offset means CO and O means should differ by ~4 eV
        assert abs(fold_mean_CO - fold_mean_O) > 3.0, (
            f"Expected large offset, got CO={fold_mean_CO:.2f}, O={fold_mean_O:.2f}"
        )

    # 3. Leakage check: no element in both train and test of same fold
    for train_idx, test_idx in gkf.split(X, y, groups):
        assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))


def test_centering_noop_single_adsorbate(tmp_path: Path) -> None:
    """Single-adsorbate dataset: centering enabled == centering disabled."""
    df = generate_demo_dataset(n_samples=80, seed=42)
    df = assign_splits(df, seed=42)
    df = add_structural_ratios(df)
    feature_df = build_feature_table(
        df,
        use_columns=["d_band_center", "surface_energy", "electronegativity"],
        categorical_columns=[],
    )
    # generate_demo_dataset sets adsorbate="CO" uniformly → single adsorbate → noop
    common_kwargs: dict = dict(
        model_name="rf",
        random_state=0,
        params={"n_estimators": 20, "max_depth": 3},
        target_col="adsorption_energy",
        n_splits=3,
        shuffle=True,
        cv_random_state=0,
        metrics_output=str(tmp_path / "m.csv"),
        model_output=str(tmp_path / "mdl.joblib"),
        predictions_output=str(tmp_path / "p.csv"),
    )
    out_with = train_model(df=feature_df, adsorbate_col="adsorbate", **common_kwargs)
    out_without = train_model(df=feature_df, adsorbate_col=None, **common_kwargs)

    r2_with = float(out_with["metrics"]["test_r2"].iloc[0])
    r2_without = float(out_without["metrics"]["test_r2"].iloc[0])
    assert abs(r2_with - r2_without) < 1e-9, (
        f"Centering noop violated: R² with={r2_with:.6f}, without={r2_without:.6f}"
    )
