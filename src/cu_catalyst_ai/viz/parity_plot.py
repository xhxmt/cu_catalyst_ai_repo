from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

from cu_catalyst_ai.utils.io import ensure_parent


def save_parity_plot(pred_df: pd.DataFrame, target_col: str, output_path: str) -> None:
    """Save a parity plot with R² annotation and labelled ideal diagonal."""
    y_true = pred_df[target_col]
    y_pred = pred_df["prediction"]
    r2 = r2_score(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.75, edgecolors="none", label="Test samples")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Ideal")
    plt.xlabel("Observed (eV)")
    plt.ylabel("Predicted (eV)")
    plt.title("Parity plot")
    plt.legend(loc="upper left", framealpha=0.8)
    plt.text(
        0.05,
        0.88,
        f"$R^2 = {r2:.3f}$",
        transform=plt.gca().transAxes,
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
    )
    plt.tight_layout()
    plt.savefig(ensure_parent(output_path), dpi=200)
    plt.close()
