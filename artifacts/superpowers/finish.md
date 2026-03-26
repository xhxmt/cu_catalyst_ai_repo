# J-Group: Per-Adsorbate Target Centering — Results

## Changes made

| File | Change |
|------|--------|
| `features/basic_features.py` | `adsorbate` added to `meta_cols` (forwarded through feature table, not a model input) |
| `features/feature_selection.py` | `adsorbate` added to `NON_FEATURE_COLUMNS` (excluded from X) |
| `models/cv.py` | New function `run_cv_with_centering`: per-fold fold-isolated centering + dual restored/centered metrics |
| `models/train.py` | `adsorbate_col` param; auto-detect multi-adsorbate; centering in CV + hold-out; `ads_mean_map` saved in joblib bundle |
| `tests/test_models.py` | 2 new tests: fold-isolation leakage test + single-adsorbate noop test |

## J-group GPR metrics (centering enabled)

| Metric | I2 GPR (no centering) | J GPR (centering) | Δ |
|--------|----------------------|-------------------|---|
| LOEO CV R² | +0.229 | **+0.220** | −0.009 |
| LOEO CV MAE | 1.903 eV | **1.905 eV** | +0.002 |
| Hold-out test R² | +0.169 | **+0.233** | **+0.064** ✅ |
| CO subset hold-out R² | 0.484 | **0.484** | ~0 |
| `centered_r2_mean` | — | 0.196 | — |

## Acceptance criteria

| Criterion | Target | Result |
|-----------|--------|--------|
| LOEO all-ads R² ≥ 0.28 | 0.28 | **0.220** ⚠️ not met |
| CO subset LOEO R² ≥ 0.10 | 0.10 | *n/a (LOEO not broken out per adsorbate here)* |
| Hold-out test R² improvement | > I2 | **+0.064** ✅ |
| `ads_mean_map` in joblib | yes | ✅ CO=−1.27, O=−2.34, OH=−0.87 eV |
| 107 tests pass | ✅ | ✅ |
| ruff clean | ✅ | ✅ |

## Interpretation

- **Hold-out test R² improved** from 0.169 → 0.233 (+0.064): centering removes the O energy scale offset and the held-out test benefits.
- **LOEO CV R² did NOT improve** (0.229 → 0.220): consistent with what the user predicted — the LOEO difficulty is cross-metal extrapolation (Ti, V, Nb, W are hard), not the O offset. Centering helps test-set metrics but not across-metal generalisation.
- **`centered_r2_mean` = 0.196**: the model is learning relative trends within each adsorbate class fairly consistently; the issue is restoring accuracy across metals.
- The CO subset hold-out R² (0.484) is preserved — O/OH data does not harm CO predictions.

## Follow-up items

- Consider stricter feature engineering for early d-metals (Ti, V, Nb, W) which dominate LOEO errors
- Run RF equivalent (J_rf) for comparison
- Consider LOEO-aware evaluation split per adsorbate for cleaner diagnostic
