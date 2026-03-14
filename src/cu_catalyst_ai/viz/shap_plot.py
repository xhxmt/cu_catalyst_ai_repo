from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from cu_catalyst_ai.utils.io import ensure_parent


def save_importance_plot(explanation_df: pd.DataFrame, output_path: str, top_n: int = 12) -> None:
    plot_df = explanation_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature importance summary")
    plt.tight_layout()
    plt.savefig(ensure_parent(output_path), dpi=200)
    plt.close()
