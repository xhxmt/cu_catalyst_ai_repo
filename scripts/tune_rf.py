"""Hyperparameter tuning for RandomForestRegressor on the G-group feature set.

Usage
-----
    uv run python scripts/tune_rf.py \\
        --features configs/features/cathub_gcn.yaml \\
        [--data   configs/data/cathub.yaml]          # optional, auto-detected
        [--n-iter 100]                               # RandomizedSearchCV iterations

Protocol
--------
1. Load the processed feature parquet for the given feature config.
2. Use only rows with split=="train" as the tuning pool.
3. From that pool, cut a stratified **hold-out** (15 %, stratified by element)
   that never participates in CV.
4. Run RandomizedSearchCV (n_iter=100, 5-fold, stratified if possible) on the
   remaining ~85 % of training data.
5. Refit the best estimator on all remaining-train rows.
6. Evaluate on the hold-out set — this is the final reported metric.
7. Emit a warning and fall back to no-hold-out if CV R² std > 0.08.
8. Write results to configs/model/rf_tuned.yaml and
   reports/tables/G_rf_tuned_metrics.csv.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split

logging.basicConfig(format="%(levelname)s  %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

PARAM_DISTRIBUTIONS: dict[str, list] = {
    "n_estimators": [100, 150, 200, 250, 300, 350, 400, 450, 500],
    "max_depth": [5, 8, 10, 12, 15, 20, None],
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf": [1, 2, 3, 4, 5],
    "max_features": ["sqrt", "log2", 0.5, 0.6, 0.8],
}

# Rollback threshold: if 5-fold CV R² std exceeds this, skip hold-out cut.
CV_STD_ROLLBACK_THRESHOLD = 0.08

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_processed_parquet(features_cfg_path: Path) -> Path:
    """Locate the processed parquet for this feature config.

    Scans common Hydra-generated run directories.  Falls back to the most
    recently modified parquet under data/processed/ or outputs/.
    """
    candidates: list[Path] = []
    for root in [Path("runs"), Path("outputs"), Path("data/processed")]:
        candidates.extend(root.rglob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            "No processed parquet found under runs/, outputs/, or data/processed/. "
            "Run the featurize stage first:\n"
            "  uv run python -m cu_catalyst_ai.cli task=featurize features=cathub_gcn"
        )
    # Return most-recently modified
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparameter tuning for RF (G-group experiment)")
    p.add_argument(
        "--features",
        default="configs/features/cathub_gcn.yaml",
        help="Feature config YAML path (default: configs/features/cathub_gcn.yaml)",
    )
    p.add_argument(
        "--parquet",
        default=None,
        help="Explicit path to processed feature parquet (auto-detected if omitted)",
    )
    p.add_argument(
        "--n-iter",
        type=int,
        default=100,
        help="Number of RandomizedSearchCV iterations (default: 100)",
    )
    p.add_argument(
        "--holdout-frac",
        type=float,
        default=0.15,
        help="Fraction of train split to reserve as hold-out (default: 0.15)",
    )
    p.add_argument(
        "--output-yaml",
        default="configs/model/rf_tuned.yaml",
        help="Destination for best-params YAML (default: configs/model/rf_tuned.yaml)",
    )
    p.add_argument(
        "--output-csv",
        default="reports/tables/G_rf_tuned_holdout_metrics.csv",
        help="Hold-out metrics CSV path",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # --- Load feature config ---
    feat_cfg = yaml.safe_load(Path(args.features).read_text(encoding="utf-8"))
    feature_cols_requested: list[str] = feat_cfg.get("use_columns", [])
    log.info("Feature config: %s", args.features)
    log.info("Requested features: %s", feature_cols_requested)

    # --- Load processed parquet ---
    parquet_path = (
        Path(args.parquet) if args.parquet else _find_processed_parquet(Path(args.features))
    )
    log.info("Loading parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    # --- Identify feature columns actually present in data ---
    # Expand categoricals: e.g. dft_functional generates dft_functional_BEEF-vdW etc.
    all_feature_cols = [
        c for c in df.columns if c not in ("catalyst_id", "adsorption_energy", "split", "element")
    ]
    log.info("Available feature columns in parquet: %s", all_feature_cols)

    # --- Filter to train split ---
    train_df = df[df["split"] == "train"].copy()
    log.info("Train rows: %d", len(train_df))

    X_all_train = train_df[all_feature_cols]
    y_all_train = train_df["adsorption_energy"]

    # --- Cut hold-out with element stratified split ---
    if "element" in train_df.columns and train_df["element"].nunique() > 1:
        stratify_col = train_df["element"]
    else:
        stratify_col = None
        log.warning("No multi-element 'element' column found; hold-out split is not stratified.")

    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X_all_train,
        y_all_train,
        test_size=args.holdout_frac,
        random_state=RANDOM_STATE,
        stratify=stratify_col,
    )
    log.info("Dev (tuning pool): %d rows | Hold-out: %d rows", len(X_dev), len(X_holdout))

    # --- RandomizedSearchCV ---
    base_rf = RandomForestRegressor(random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_rf,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=args.n_iter,
        scoring="r2",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    log.info("Starting RandomizedSearchCV (n_iter=%d) …", args.n_iter)
    search.fit(X_dev, y_dev)

    best_params = search.best_params_
    cv_r2_mean = search.best_score_
    log.info("Best CV R² (on dev set): %.4f", cv_r2_mean)
    log.info("Best params: %s", best_params)

    # --- Rollback check ---
    cv_results = pd.DataFrame(search.cv_results_)
    best_row = cv_results.loc[cv_results["mean_test_score"].idxmax()]
    cv_r2_std = float(best_row["std_test_score"])
    log.info("CV R² std of best param set: %.4f", cv_r2_std)

    use_holdout = True
    if cv_r2_std > CV_STD_ROLLBACK_THRESHOLD:
        warnings.warn(
            f"CV R² std ({cv_r2_std:.4f}) > rollback threshold ({CV_STD_ROLLBACK_THRESHOLD}). "
            "Data is too small for a clean hold-out evaluation. "
            "Reporting CV-based metrics instead of hold-out metrics.",
            stacklevel=2,
        )
        use_holdout = False

    # --- Refit on full dev set, evaluate on hold-out ---
    best_rf = RandomForestRegressor(random_state=RANDOM_STATE, **best_params)
    best_rf.fit(X_dev, y_dev)

    if use_holdout:
        preds = best_rf.predict(X_holdout)
        metrics = _regression_metrics(y_holdout.values, preds)
        eval_label = "hold_out"
        log.info(
            "Hold-out R²=%.4f  MAE=%.4f  RMSE=%.4f", metrics["r2"], metrics["mae"], metrics["rmse"]
        )
    else:
        metrics = {
            "r2": cv_r2_mean,
            "mae": float(best_row["mean_test_score"]),
            "rmse": float("nan"),
        }
        eval_label = "cv_fallback"
        log.info("CV-fallback R²=%.4f", cv_r2_mean)

    # --- Write best params yaml ---
    out_yaml = Path(args.output_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    best_params_serializable = {
        k: (
            None
            if v is None
            else (
                int(v)
                if isinstance(v, (np.integer,))
                else (float(v) if isinstance(v, (np.floating,)) else v)
            )
        )
        for k, v in best_params.items()
    }
    yaml_content = {
        "name": "rf",
        "random_state": RANDOM_STATE,
        "params": best_params_serializable,
    }
    out_yaml.write_text(yaml.dump(yaml_content, sort_keys=False))
    log.info("Wrote best params → %s", out_yaml)

    # --- Write metrics CSV ---
    metrics_row = {
        "eval_type": eval_label,
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "test_r2": metrics["r2"],
        "test_mae": metrics["mae"],
        "test_rmse": metrics["rmse"],
        "model_name": "rf_tuned",
        **{f"param_{k}": v for k, v in best_params_serializable.items()},
    }
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics_row]).to_csv(out_csv, index=False)
    log.info("Wrote hold-out metrics → %s", out_csv)


if __name__ == "__main__":
    main(sys.argv[1:])
