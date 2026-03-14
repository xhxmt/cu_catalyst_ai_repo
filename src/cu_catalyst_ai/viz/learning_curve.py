from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, learning_curve

from cu_catalyst_ai.features.feature_selection import get_feature_columns
from cu_catalyst_ai.utils.io import ensure_parent


def save_learning_curve(
    model, df: pd.DataFrame, target_col: str, output_path: str, n_splits: int = 5
) -> None:
    train_df = df[df["split"] == "train"].copy()
    X = train_df[get_feature_columns(train_df)]
    y = train_df[target_col]
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_sizes, train_scores, valid_scores = learning_curve(model, X, y, cv=cv, scoring="r2")
    train_mean = np.mean(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, label="Train R2")
    plt.plot(train_sizes, valid_mean, label="Validation R2")
    plt.xlabel("Training examples")
    plt.ylabel("R2")
    plt.title("Learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ensure_parent(output_path), dpi=200)
    plt.close()
