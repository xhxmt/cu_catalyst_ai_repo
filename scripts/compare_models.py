"""Multi-model comparison script: RF (default) vs XGBoost (default) vs GPR.

Reads per-model metrics CSVs already produced by the training stage and
generates a consolidated comparison table + a 3-panel parity plot.

Fair-comparison design
----------------------
* All three models use **default** hyper-parameters for the algorithm
  comparison (pure algorithm signal, no confounding from tuning).
* RF-tuned is shown as a *separate* row labelled "rf_tuned" so the reader
  can distinguish tuning gain from algorithm gain.
* All models must have been trained on the **same** processed feature parquet
  (same split, same seed).

Usage
-----
    uv run python scripts/compare_models.py \\
        [--tables-dir reports/tables] \\
        [--figures-dir reports/figures] \\
        [--output-table G_model_comparison.csv] \\
        [--output-figure G_parity_comparison.png]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
logging.basicConfig(format="%(levelname)s  %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry for this comparison
# ---------------------------------------------------------------------------

#   key     → (metrics_csv, predictions_csv, display_label)
MODEL_REGISTRY: dict[str, tuple[str, str, str]] = {
    "G_rf": ("G_rf_metrics.csv", "G_rf_predictions.csv", "RF (default)"),
    "G_xgb": ("G_xgb_metrics.csv", "G_xgb_predictions.csv", "XGBoost (default)"),
    "G_gpr": ("G_gpr_metrics.csv", "G_gpr_predictions.csv", "GPR (RBF+White)"),
    "G_rf_tuned": ("G_rf_tuned_metrics.csv", "G_rf_tuned_predictions.csv", "RF (tuned)"),
}

METRIC_COLS = ["test_r2", "test_mae", "test_rmse", "r2_mean", "mae_mean"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_metrics(tables_dir: Path, csv_name: str) -> pd.Series | None:
    p = tables_dir / csv_name
    if not p.exists():
        log.warning("Metrics file not found (skipping): %s", p)
        return None
    df = pd.read_csv(p)
    return df.iloc[0]


def _load_predictions(tables_dir: Path, csv_name: str) -> pd.DataFrame | None:
    p = tables_dir / csv_name
    if not p.exists():
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _make_parity_panel(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    r2: float,
    color: str,
) -> None:
    ax.scatter(y_true, y_pred, s=18, alpha=0.55, color=color, edgecolors="none")
    lims = [
        min(y_true.min(), y_pred.min()) - 0.1,
        max(y_true.max(), y_pred.max()) + 0.1,
    ]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("DFT (eV)", fontsize=9)
    ax.set_ylabel("Predicted (eV)", fontsize=9)
    ax.set_title(f"{label}\nR²={r2:.3f}", fontsize=10)
    ax.set_aspect("equal", adjustable="box")


PALETTE = ["#2D6A4F", "#1565C0", "#BF360C", "#6A1B9A"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-model comparison (RF / XGBoost / GPR)")
    p.add_argument("--tables-dir", default="reports/tables", help="Directory with metrics CSVs")
    p.add_argument("--figures-dir", default="reports/figures", help="Output directory for figures")
    p.add_argument(
        "--output-table", default="G_model_comparison.csv", help="Output comparison CSV filename"
    )
    p.add_argument(
        "--output-figure",
        default="G_parity_comparison.png",
        help="Output parity comparison figure filename",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict] = []
    parity_data: list[tuple[str, np.ndarray, np.ndarray, float]] = []

    for key, (metrics_csv, preds_csv, label) in MODEL_REGISTRY.items():
        m = _load_metrics(tables_dir, metrics_csv)
        if m is None:
            continue

        row: dict = {"model": key, "label": label}
        for col in METRIC_COLS:
            row[col] = m.get(col, float("nan"))
        comparison_rows.append(row)
        log.info(
            "%-20s  test_r2=%.4f  test_mae=%.4f",
            label,
            row.get("test_r2", float("nan")),
            row.get("test_mae", float("nan")),
        )

        preds_df = _load_predictions(tables_dir, preds_csv)
        if (
            preds_df is not None
            and "adsorption_energy" in preds_df.columns
            and "prediction" in preds_df.columns
        ):
            parity_data.append(
                (
                    label,
                    preds_df["adsorption_energy"].values,
                    preds_df["prediction"].values,
                    row.get("test_r2", float("nan")),
                )
            )

    if not comparison_rows:
        log.error("No metrics files found in %s — run training stages first.", tables_dir)
        sys.exit(1)

    # --- Write comparison table ---
    comp_df = pd.DataFrame(comparison_rows).sort_values("test_r2", ascending=False)
    out_table = tables_dir / args.output_table
    comp_df.to_csv(out_table, index=False)
    log.info("Comparison table → %s", out_table)
    print("\n" + comp_df.to_string(index=False))

    # --- Parity comparison figure ---
    if not parity_data:
        log.warning("No prediction files found; skipping parity figure.")
        return

    n = len(parity_data)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (label, y_true, y_pred, r2), color in zip(axes, parity_data, PALETTE, strict=False):  # type: ignore[arg-type]
        _make_parity_panel(ax, y_true, y_pred, label, r2, color)

    fig.suptitle("G-group: Model Comparison — CO Adsorption Energy (eV)", fontsize=11, y=1.02)
    out_fig = figures_dir / args.output_figure
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Parity comparison figure → %s", out_fig)


if __name__ == "__main__":
    main(sys.argv[1:])
