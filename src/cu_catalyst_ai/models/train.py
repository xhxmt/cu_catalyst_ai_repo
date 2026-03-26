from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cu_catalyst_ai.features.feature_selection import get_feature_columns
from cu_catalyst_ai.models.cv import run_cv, run_cv_with_centering
from cu_catalyst_ai.models.metrics import regression_metrics
from cu_catalyst_ai.models.registry import build_model
from cu_catalyst_ai.utils.io import ensure_parent

logger = logging.getLogger(__name__)


def _supports_sample_weight(model: object) -> bool:
    """Return True when *model* accepts ``sample_weight`` in ``.fit()``.

    Pipeline-wrapped GPR does not forward ``sample_weight`` by default;
    attempting to pass it raises a TypeError.  Tree models (RF, XGBoost)
    accept it directly.
    """
    if isinstance(model, Pipeline):
        # Only the final estimator receives fit kwargs; check its signature.
        import inspect  # noqa: PLC0415

        final = model.steps[-1][1]
        sig = inspect.signature(final.fit)
        return "sample_weight" in sig.parameters
    import inspect  # noqa: PLC0415

    sig = inspect.signature(model.fit)
    return "sample_weight" in sig.parameters


def train_model(
    df: pd.DataFrame,
    model_name: str,
    random_state: int,
    params: dict,
    target_col: str,
    n_splits: int,
    shuffle: bool,
    cv_random_state: int,
    metrics_output: str,
    model_output: str,
    predictions_output: str,
    adsorbate_col: str | None = "adsorbate",
) -> dict:
    """Train model with optional per-adsorbate target centering in CV.

    When *adsorbate_col* references a column present in *df* that contains
    ≥2 distinct adsorbate labels, the cross-validation uses
    :func:`~cu_catalyst_ai.models.cv.run_cv_with_centering` so that each
    training fold's adsorbate means (computed from that fold's training rows
    only) are subtracted before fitting and added back before evaluation.
    This prevents the absolute energy scale of O/OH from dominating MSE.

    The final model is always fitted on the full training set.  When centering
    is active, the train set adsorbate means are stored in the joblib bundle
    under ``"ads_mean_map"`` so they can be applied at inference time.

    Args:
        df: Processed feature table with ``split`` column.
        model_name: Registered model identifier (e.g. ``"rf"``, ``"gpr"``).
        random_state: Seed for the model's internal random state.
        params: Extra parameters forwarded to the model builder.
        target_col: Name of the target column in *df*.
        n_splits: Number of CV folds.
        shuffle: Whether to shuffle folds (KFold only).
        cv_random_state: Random seed for KFold shuffle.
        metrics_output: Path for the metrics CSV.
        model_output: Path for the joblib bundle.
        predictions_output: Path for the hold-out predictions CSV.
        adsorbate_col: Column name for adsorbate labels used in centering.
            Set to ``None`` to disable centering unconditionally.

    Returns:
        Dict with keys ``feature_columns``, ``test_df``, ``pred_df``,
        ``metrics``, ``model``.
    """
    feature_cols = get_feature_columns(df)
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = build_model(model_name=model_name, random_state=random_state, params=params)

    # --- Determine whether to use per-adsorbate centering -------------------
    use_centering = (
        adsorbate_col is not None
        and adsorbate_col in train_df.columns
        and train_df[adsorbate_col].nunique() >= 2
    )

    # Use GroupKFold (grouped by element) when available
    cv_groups = train_df["element"] if "element" in train_df.columns else None

    if use_centering:
        logger.info(
            "Per-adsorbate target centering enabled (%d unique adsorbates: %s).",
            train_df[adsorbate_col].nunique(),
            sorted(train_df[adsorbate_col].unique().tolist()),
        )
        cv_summary = run_cv_with_centering(
            model,
            X_train,
            y_train,
            adsorbates=train_df[adsorbate_col],
            n_splits=n_splits,
            groups=cv_groups,
        )
        # Compute train-set adsorbate means for hold-out centering and inference
        ads_mean_map: dict[str, float] = y_train.groupby(train_df[adsorbate_col]).mean().to_dict()
        global_mean = float(np.mean(list(ads_mean_map.values())))

        # Fit final model on centered targets (full train set)
        y_train_c = y_train - train_df[adsorbate_col].map(ads_mean_map)
        if "element" in train_df.columns and _supports_sample_weight(model):
            element_counts = train_df["element"].value_counts()
            n_total = len(train_df)
            n_elements = len(element_counts)
            sample_weight = (
                train_df["element"].map(lambda x: n_total / (n_elements * element_counts[x])).values
            )
            model.fit(X_train, y_train_c, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train_c)

        # Hold-out evaluation: restore predictions before computing metrics
        preds_c = model.predict(X_test)
        if adsorbate_col in test_df.columns:
            restore = test_df[adsorbate_col].map(ads_mean_map).fillna(global_mean).values
        else:
            restore = global_mean
        preds = preds_c + restore
    else:
        ads_mean_map = {}
        cv_summary = run_cv(
            model,
            X_train,
            y_train,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=cv_random_state,
            groups=cv_groups,
        )
        # --- Inverse-frequency sample weights (balances Cu-dominant distribution) ---
        if "element" in train_df.columns and _supports_sample_weight(model):
            element_counts = train_df["element"].value_counts()
            n_total = len(train_df)
            n_elements = len(element_counts)
            sample_weight = (
                train_df["element"].map(lambda x: n_total / (n_elements * element_counts[x])).values
            )
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        preds = model.predict(X_test)

    test_metrics = regression_metrics(y_test, preds)

    metrics = cv_summary.copy()
    metrics["test_mae"] = test_metrics["mae"]
    metrics["test_rmse"] = test_metrics["rmse"]
    metrics["test_r2"] = test_metrics["r2"]
    metrics["model_name"] = model_name
    metrics.to_csv(ensure_parent(metrics_output), index=False)

    pred_df = test_df[["catalyst_id", target_col]].copy()
    pred_df["prediction"] = preds
    pred_df.to_csv(ensure_parent(predictions_output), index=False)

    joblib.dump(
        {"model": model, "feature_columns": feature_cols, "ads_mean_map": ads_mean_map},
        ensure_parent(model_output),
    )

    return {
        "feature_columns": feature_cols,
        "test_df": test_df,
        "pred_df": pred_df,
        "metrics": metrics,
        "model": model,
    }
