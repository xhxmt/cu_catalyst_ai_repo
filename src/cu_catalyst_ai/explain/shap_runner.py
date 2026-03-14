from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from cu_catalyst_ai.features.feature_selection import get_feature_columns
from cu_catalyst_ai.utils.io import ensure_parent


def explain_model(
    model, df: pd.DataFrame, target_col: str, output_path: str, random_state: int = 42
) -> pd.DataFrame:
    feature_cols = get_feature_columns(df)
    test_df = df[df["split"] == "test"].copy()
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    try:
        import shap

        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        importance = np.abs(shap_values.values).mean(axis=0)
    except Exception:
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=random_state
        )
        importance = result.importances_mean

    explanation_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    explanation_df = explanation_df.sort_values("importance", ascending=False).reset_index(
        drop=True
    )
    explanation_df.to_csv(ensure_parent(output_path), index=False)
    return explanation_df
