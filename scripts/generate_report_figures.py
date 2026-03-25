"""Generate all figures and tables needed for the research report.

Produces:
  1. R² progression chart  (reports/figures/research_r2_progression.png)
  2. Multi-model comparison table  (reports/tables/research_model_comparison.csv)
         + comparison bar chart    (reports/figures/research_model_comparison.png)
  3. I-2 CO-subset parity plot    (reports/figures/research_I2_gpr_CO_parity.png)
  4. SHAP feature importance bar   (reports/figures/research_shap_importance.png)
     (uses I-2 RF model on I-2 test data; RF supports SHAP TreeExplainer natively)

Usage
-----
    uv run python scripts/generate_report_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

ROOT = Path(__file__).parent.parent
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1: R² Progression (A → I)
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # (label, test_r2, description)
    ("A\n(linear,\nCu-only)", -0.097, "Linear regression\nCu-only~24 rows\nfacet OHE features"),
    ("B\n(proxy_cn,\nCu-only)", -0.007, "Replaced facet OHE\nwith GCN proxy_cn\nstill Cu only"),
    ("C\n(proxy_only\nbaseline)", 0.022, "Removed facet OHE\nbulk d-band added\nstill Cu-only"),
    ("D\n(RF,\nmulti-metal)", 0.105, "+Multi-metal (11)\n+d-band, χ, GCN\nBEEF-filtered"),
    ("E\n(+full\nfunctionals)", 0.299, "+All functionals\n(no filter)\nmore data"),
    ("F\n(RF,\nclean split)", 0.316, "Stratified split\nby element\nclean test set"),
    ("G\nRF", 0.337, "G-group: added\nelectronegativity\n+GCN feature"),
    ("G\nXGB", 0.293, "XGBoost baseline\nsame feature set\nas G-RF"),
    ("G\nGPR", 0.433, "GPR (RBF+White)\nsame feature set\nnew best record"),
    ("H\nGPR", 0.240, "Surface d-band\nfacet-resolved\ninconclusive"),
    ("I-1\nRF", 0.357, "I-1: CO-only,\n25 metals\nscaling val."),
    ("I-2\nRF", 0.341, "I-2: CO+O+OH\n221 rows\nmulti-ads"),
    ("I-2\nGPR\n(all)", 0.236, "I-2 GPR\nall adsorbates\nmixed scale"),
    ("I-2\nGPR\n(CO)", 0.490, "I-2 GPR\nCO subset only\n★ new record"),
]

COLORS = [
    "#90CAF9",
    "#90CAF9",
    "#90CAF9",  # A-C: blue family (early)
    "#81C784",
    "#81C784",
    "#81C784",  # D-F: green (multi-metal expansion)
    "#FFB74D",
    "#FFB74D",
    "#FFB74D",  # G: orange (model comparison)
    "#CE93D8",  # H: purple (surface dband)
    "#4DB6AC",
    "#4DB6AC",  # I-1, I-2 RF: teal
    "#EF9A9A",  # I-2 GPR all: pink/grey
    "#D32F2F",  # I-2 GPR CO: red star (best)
]

fig, ax = plt.subplots(figsize=(16, 7))

x = np.arange(len(EXPERIMENTS))
labels = [e[0] for e in EXPERIMENTS]
r2_vals = [e[1] for e in EXPERIMENTS]

bars = ax.bar(x, r2_vals, color=COLORS, width=0.72, linewidth=0.5, edgecolor="#555", zorder=3)

# Reference lines
ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=2)
ax.axhline(
    0.433,
    color="#FF6F00",
    linewidth=1.2,
    linestyle=":",
    alpha=0.7,
    zorder=2,
    label="G-GPR record (0.433)",
)
ax.axhline(
    0.490,
    color="#D32F2F",
    linewidth=1.4,
    linestyle="-.",
    alpha=0.8,
    zorder=2,
    label="I-2 GPR CO record (0.490)",
)

# Annotate bar values
for bar, val in zip(bars, r2_vals, strict=True):
    ypos = val + 0.010 if val >= 0 else val - 0.028
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        ypos,
        f"{val:.3f}",
        ha="center",
        va="bottom" if val >= 0 else "top",
        fontsize=7.5,
        fontweight="bold",
        color="#222",
    )

# Group separators + labels
group_spans = [
    (0, 2, "A–C\nEarly Ablation\n(Cu-only)", "#90CAF9"),
    (3, 5, "D–F\nMulti-Metal\nExpansion", "#81C784"),
    (6, 8, "G-Group\nModel\nComparison", "#FFB74D"),
    (9, 9, "H\nSurface\nd-Band", "#CE93D8"),
    (10, 13, "I-Group\nMulti-Adsorbate\nExtension", "#EF5350"),
]
y_bracket = -0.25
for start, end, glab, gcol in group_spans:
    mid = (start + end) / 2
    ax.annotate(
        "",
        xy=(start - 0.35, y_bracket + 0.02),
        xytext=(end + 0.35, y_bracket + 0.02),
        arrowprops=dict(arrowstyle="-", color=gcol, lw=1.8),
        annotation_clip=False,
    )
    ax.text(
        mid,
        y_bracket - 0.04,
        glab,
        ha="center",
        va="top",
        fontsize=7,
        color=gcol,
        fontweight="bold",
        clip_on=False,
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, ha="center")
ax.set_ylabel("Test $R^2$", fontsize=12)
ax.set_title(
    "Model Performance Progression: A → I Group Experiments\n"
    "CO Adsorption Energy Prediction on Transition Metals",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.set_ylim(-0.35, 0.58)
ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
ax.grid(axis="y", which="major", linestyle="--", alpha=0.35, zorder=0)
ax.legend(fontsize=9, loc="upper left", framealpha=0.7)

# Highlight best bar
ax.patches[-1].set_edgecolor("#B71C1C")
ax.patches[-1].set_linewidth(2.0)

plt.tight_layout()
out1 = FIGURES / "research_r2_progression.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[1/4] R² progression chart saved → {out1}")


# ---------------------------------------------------------------------------
# Figure 2: Multi-model comparison bar chart + table
# ---------------------------------------------------------------------------

COMPARISON_DATA = [
    # (group, model, dataset, test_r2)
    ("G", "RF", "CO (11 metals)", 0.337),
    ("G", "XGBoost", "CO (11 metals)", 0.293),
    ("G", "GPR", "CO (11 metals)", 0.433),
    ("I-2", "RF (all ads)", "CO+O+OH (25 metals)", 0.341),
    ("I-2", "RF (CO only)", "CO+O+OH (25 metals)", 0.417),
    ("I-2", "GPR (all ads)", "CO+O+OH (25 metals)", 0.236),
    ("I-2", "GPR (CO only)", "CO+O+OH (25 metals)", 0.490),
]

# Save comparison table
comp_df = pd.DataFrame(
    COMPARISON_DATA, columns=["Group", "Model", "Dataset", "test_r2"]
).sort_values("test_r2", ascending=False)
comp_table_out = TABLES / "research_model_comparison.csv"
comp_df.to_csv(comp_table_out, index=False)
print(f"[2/4] Comparison table saved → {comp_table_out}")
print(comp_df.to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
model_labels = [f"{r['Group']}\n{r['Model']}" for _, r in comp_df.iterrows()]
colors_cmp = ["#FFB74D" if r["Group"] == "G" else "#EF5350" for _, r in comp_df.iterrows()]
# Best result gets full red
colors_cmp[0] = "#B71C1C"  # top row after sort is I-2 GPR CO

y_pos = np.arange(len(comp_df))
hbars = ax.barh(
    y_pos, comp_df["test_r2"], color=colors_cmp, height=0.65, edgecolor="#555", linewidth=0.5
)
ax.set_yticks(y_pos)
ax.set_yticklabels(model_labels, fontsize=10)
ax.set_xlabel("Test $R^2$", fontsize=12)
ax.set_title("Multi-Model Comparison: G-Group vs I-2 Group", fontsize=13, fontweight="bold")
ax.axvline(0.433, color="#FF6F00", linestyle=":", linewidth=1.3, alpha=0.75, label="G-GPR (0.433)")  # noqa: E501
ax.axvline(
    0.490, color="#B71C1C", linestyle="-.", linewidth=1.5, alpha=0.85, label="I-2 GPR CO (0.490)"
)  # noqa: E501
for bar, val in zip(hbars, comp_df["test_r2"], strict=True):
    ax.text(
        val + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
ax.set_xlim(0, 0.57)
ax.legend(fontsize=9)

g_patch = mpatches.Patch(color="#FFB74D", label="G-Group (CO only)")
i_patch = mpatches.Patch(color="#EF5350", label="I-2 Group (multi-ads)")
best_patch = mpatches.Patch(color="#B71C1C", label="Best result")
ax.legend(handles=[g_patch, i_patch, best_patch], fontsize=9, loc="lower right")

plt.tight_layout()
out2 = FIGURES / "research_model_comparison.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[2/4] Model comparison chart saved → {out2}")


# ---------------------------------------------------------------------------
# Figure 3: I-2 GPR CO-subset parity plot
# ---------------------------------------------------------------------------

pred_df = pd.read_csv(TABLES / "I2_gpr_predictions.csv")

# Identify CO rows: catalyst_id comes from CatHub (non-mamun) = CO
# Mamun rows start with "mamun_"
is_co = ~pred_df["catalyst_id"].str.startswith("mamun_")
co_df = pred_df[is_co].copy()

print(f"\n[3/4] CO rows in test set: {len(co_df)} (non-mamun catalyst_ids)")

y_true = co_df["adsorption_energy"].values
y_pred = co_df["prediction"].values

from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402, PLC0415

co_r2 = r2_score(y_true, y_pred)
co_mae = mean_absolute_error(y_true, y_pred)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(
    y_true,
    y_pred,
    s=55,
    alpha=0.75,
    color="#D32F2F",
    edgecolors="#7B0000",
    linewidths=0.5,
    zorder=3,
    label=f"CO adsorption (n={len(co_df)})",
)

lims = [min(y_true.min(), y_pred.min()) - 0.15, max(y_true.max(), y_pred.max()) + 0.15]
ax.plot(lims, lims, "k--", lw=1.2, alpha=0.55, label="Perfect prediction")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("DFT adsorption energy (eV)", fontsize=12)
ax.set_ylabel("GPR predicted energy (eV)", fontsize=12)
ax.set_title(
    f"I-2 GPR: CO Subset Parity Plot\n$R^2 = {co_r2:.4f}$,  MAE = {co_mae:.3f} eV",
    fontsize=12,
    fontweight="bold",
)
ax.text(
    0.05,
    0.92,
    f"$R^2 = {co_r2:.3f}$\nMAE = {co_mae:.3f} eV\nn = {len(co_df)}",
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="#FFEBEE", alpha=0.75, edgecolor="#D32F2F"),
)
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
plt.tight_layout()
out3 = FIGURES / "research_I2_gpr_CO_parity.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[3/4] I-2 GPR CO-subset parity plot saved → {out3}")
print(f"       CO subset: R²={co_r2:.4f}, MAE={co_mae:.4f} eV")


# ---------------------------------------------------------------------------
# Figure 4: SHAP Feature Importance using I-2 RF model
# ---------------------------------------------------------------------------

import joblib  # noqa: E402, PLC0415

sys.path.insert(0, str(ROOT / "src"))
from cu_catalyst_ai.features.feature_selection import get_feature_columns  # noqa: E402, PLC0415
from cu_catalyst_ai.utils.io import read_table  # noqa: E402, PLC0415

model_path = ROOT / "reports" / "models" / "I2_gpr.joblib"
data_path = ROOT / "data" / "processed" / "cathub_multi_model_table.parquet"

if not model_path.exists():
    print("[4/4] I2_gpr.joblib not found — skipping SHAP (run training first)")
else:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    df = read_table(str(data_path))
    feat_cols = get_feature_columns(df)
    test_df = df[df["split"] == "test"].copy()
    # Use permutation importance — works for any sklearn-compatible model including
    # GPR Pipeline, and is robust. SHAP KernelExplainer has Pipeline compatibility
    # issues with this version of sklearn.
    from sklearn.impute import SimpleImputer  # noqa: PLC0415
    from sklearn.inspection import permutation_importance  # noqa: PLC0415

    print("[4/4] Computing permutation importance on I-2 GPR test set...")
    # Impute NaN before passing to permutation_importance scorer
    imputer = SimpleImputer(strategy="median")
    X_test_imp = imputer.fit_transform(test_df[feat_cols].values)

    pi = permutation_importance(
        model,
        X_test_imp,
        test_df["adsorption_energy"].values,
        n_repeats=50,
        random_state=42,
        scoring="r2",
    )
    importance_df = pd.DataFrame(
        {
            "feature": feat_cols,
            "mean_decrease_r2": pi.importances_mean,
            "std_decrease_r2": pi.importances_std,
        }
    ).sort_values("mean_decrease_r2", ascending=True)

    feat_labels = {
        "d_band_center": "d-band center\n(Ruban 1997)",
        "electronegativity": "Electronegativity\n(Pauling)",
        "gcn": "GCN\n(coord. no.)",
        "is_CO": "is_CO\n(adsorbate OHE)",
        "is_O": "is_O\n(adsorbate OHE)",
        "is_OH": "is_OH\n(adsorbate OHE)",
    }
    y_labels = [feat_labels.get(f, f) for f in importance_df["feature"]]
    y_pos = np.arange(len(importance_df))
    bar_colors = plt.cm.RdYlGn(  # type: ignore[attr-defined]
        np.linspace(0.2, 0.85, len(importance_df))
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    hbars2 = ax.barh(
        y_pos,
        importance_df["mean_decrease_r2"],
        xerr=importance_df["std_decrease_r2"],
        color=bar_colors,
        edgecolor="#333",
        linewidth=0.5,
        height=0.65,
        capsize=4,
        error_kw={"linewidth": 1.2, "ecolor": "#444"},
    )
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Mean decrease in $R^2$ (permutation, n=50)", fontsize=11)
    ax.set_title(
        "Feature Importance: Permutation Analysis on I-2 GPR\n(CO + O + OH, 25 transition metals)",
        fontsize=12,
        fontweight="bold",
    )
    for i, (val, std) in enumerate(
        zip(importance_df["mean_decrease_r2"], importance_df["std_decrease_r2"], strict=True)
    ):
        ax.text(max(val, 0) + 0.002, i, f"{val:+.4f} ± {std:.4f}", va="center", fontsize=8.5)
    ax.set_xlim(
        min(importance_df["mean_decrease_r2"].min() - 0.01, -0.02),
        importance_df["mean_decrease_r2"].max() + 0.05,
    )
    plt.tight_layout()
    out4 = FIGURES / "research_shap_importance.png"
    fig.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[4/4] Permutation importance saved → {out4}")
    print("\nFeature importance ranking:")
    print(importance_df.sort_values("mean_decrease_r2", ascending=False).to_string(index=False))

    importance_df.sort_values("mean_decrease_r2", ascending=False).to_csv(
        TABLES / "research_feature_importance.csv", index=False
    )

print("\n✓ All report figures generated successfully.")
