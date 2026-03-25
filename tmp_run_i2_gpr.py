"""Run I-2 GPR experiment directly, bypassing Hydra config group resolution issues.

Reads the already-processed cathub_multi model table and trains a GPR model,
saving metrics to reports/tables/I2_gpr_metrics.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from sklearn.metrics import r2_score

from cu_catalyst_ai.models.train import train_model
from cu_catalyst_ai.utils.io import read_table
from cu_catalyst_ai.viz.learning_curve import save_learning_curve
from cu_catalyst_ai.viz.parity_plot import save_parity_plot

ROOT = Path(__file__).parent

# Paths
processed_path = ROOT / "data/processed/cathub_multi_model_table.parquet"
metrics_output = str(ROOT / "reports/tables/I2_gpr_metrics.csv")
predictions_output = str(ROOT / "reports/tables/I2_gpr_predictions.csv")
model_output = str(ROOT / "reports/models/I2_gpr.joblib")
parity_output = str(ROOT / "reports/figures/I2_gpr_parity.png")
learning_curve_output = str(ROOT / "reports/figures/I2_gpr_learning_curve.png")

print(f"Loading processed data from: {processed_path}")
df = read_table(str(processed_path))
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Split distribution:\n{df['split'].value_counts()}")

# Show adsorbate distribution if present
if "is_CO" in df.columns:
    print(
        f"\nAdsorbate OHE columns: is_CO={df['is_CO'].sum()}, "
        f"is_O={df.get('is_O', pd.Series([0])).sum()}, "
        f"is_OH={df.get('is_OH', pd.Series([0])).sum()}"
    )

print("\nRunning GPR training...")
result = train_model(
    df=df,
    model_name="gpr",
    random_state=42,
    params={},
    target_col="adsorption_energy",
    n_splits=5,
    shuffle=True,
    cv_random_state=42,
    metrics_output=metrics_output,
    model_output=model_output,
    predictions_output=predictions_output,
)

print("\n=== I-2 GPR Results ===")
metrics = result["metrics"]
print(metrics.to_string())

# Save visualizations
save_parity_plot(result["pred_df"], "adsorption_energy", parity_output)
save_learning_curve(result["model"], df, "adsorption_energy", learning_curve_output, 5)
print(f"\nParity plot saved to: {parity_output}")
print(f"Learning curve saved to: {learning_curve_output}")

# Compute CO subset R2
pred_df = result["pred_df"]
test_df = result["test_df"]

if "is_CO" in test_df.columns:
    co_mask = test_df["is_CO"] == 1
    if co_mask.sum() > 0:
        co_test = test_df[co_mask]
        # Match predictions by index
        co_pred = (
            pred_df.loc[co_test.index, "prediction"] if co_test.index[0] in pred_df.index else None
        )
        if co_pred is None:
            # Try matching via catalyst_id
            co_preds_df = pred_df[pred_df["catalyst_id"].isin(co_test["catalyst_id"])]
            if len(co_preds_df) > 0:
                merged = co_test[["catalyst_id", "adsorption_energy"]].merge(
                    co_preds_df[["catalyst_id", "prediction"]], on="catalyst_id"
                )
                co_r2 = r2_score(merged["adsorption_energy"], merged["prediction"])
                print(f"\nCO subset test R² = {co_r2:.4f} (n={len(merged)})")
        else:
            co_r2 = r2_score(co_test["adsorption_energy"], co_pred)
            print(f"\nCO subset test R² = {co_r2:.4f} (n={co_mask.sum()})")
    else:
        print("No CO rows in test set")
else:
    print("is_CO column not found in test_df — single-adsorbate data?")

print("\nDone!")
