from __future__ import annotations

import joblib
import pandas as pd

from cu_catalyst_ai.features.feature_selection import get_feature_columns
from cu_catalyst_ai.models.cv import run_cv
from cu_catalyst_ai.models.metrics import regression_metrics
from cu_catalyst_ai.models.registry import build_model
from cu_catalyst_ai.utils.io import ensure_parent


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
