from __future__ import annotations

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from cu_catalyst_ai.features.feature_selection import get_feature_columns
from cu_catalyst_ai.models.cv import run_cv
from cu_catalyst_ai.models.metrics import regression_metrics
from cu_catalyst_ai.models.registry import build_model
from cu_catalyst_ai.utils.io import ensure_parent


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
) -> dict:
    feature_cols = get_feature_columns(df)
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = build_model(model_name=model_name, random_state=random_state, params=params)
    cv_summary = run_cv(
        model, X_train, y_train, n_splits=n_splits, shuffle=shuffle, random_state=cv_random_state
    )

    # --- Inverse-frequency sample weights (balances Cu-dominant distribution) ---
    # w_i = N / (K * count_i), where N = total train samples, K = number of elements.
    # Each element's total weight contribution equals N/K — they all matter equally.
    # Falls back to unweighted fit when 'element' metadata column is absent.
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

    joblib.dump({"model": model, "feature_columns": feature_cols}, ensure_parent(model_output))

    return {
        "feature_columns": feature_cols,
        "test_df": test_df,
        "pred_df": pred_df,
        "metrics": metrics,
        "model": model,
    }
