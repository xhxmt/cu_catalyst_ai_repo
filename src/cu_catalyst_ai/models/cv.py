from __future__ import annotations

import pandas as pd
from sklearn.model_selection import KFold, cross_validate


def run_cv(
    model, X, y, n_splits: int = 5, shuffle: bool = True, random_state: int = 42
) -> pd.DataFrame:
    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        return_train_score=False,
    )
    df = pd.DataFrame(scores)
    for col in ["test_mae", "test_rmse"]:
        df[col] = -df[col]
    summary = {
        "mae_mean": float(df["test_mae"].mean()),
        "mae_std": float(df["test_mae"].std(ddof=0)),
        "rmse_mean": float(df["test_rmse"].mean()),
        "rmse_std": float(df["test_rmse"].std(ddof=0)),
        "r2_mean": float(df["test_r2"].mean()),
        "r2_std": float(df["test_r2"].std(ddof=0)),
    }
    return pd.DataFrame([summary])
