# Target Centering Implementation Plan (J-Group Experiment)

## Goal

实现 per-adsorbate target centering，使多吸附质混合模型（I-2/J 组）的 LOEO CV
R² 从当前 0.229 提升到 ≥ 0.28，方法是在 training fold 内减去吸附质均值、
预测后还原，避免 O 的绝对值域（−6～−2 eV）主导 MSE。

## Assumptions

- `cathub_multi_model_table.parquet` 中 `adsorbate` 列已被 `build_feature_table`
  丢弃（只保留了 `is_CO`/`is_O`/`is_OH` OHE 列）；需修复为显式保留
- `adsorbate` 列不作为模型输入特征（OHE 列才是），仅作 centering 用途
- GroupKFold 按 element 分组；centering 均值每折只从该折的训练子集计算 → 无信息泄露
- CO-only 实验（`adsorbate` 列缺失或 unique < 2）自动跳过 centering → 原有实验全部不变

---

## Plan

### Step 1 — `build_feature_table`：将 `adsorbate` 加入 metadata 列（保留不进 X）

**Files**: `src/cu_catalyst_ai/features/basic_features.py` (line 333–336)

**Change**:
```python
# Before
meta_cols = ["catalyst_id", "adsorption_energy", "split"]
if "element" in df.columns:
    meta_cols.append("element")

# After
meta_cols = ["catalyst_id", "adsorption_energy", "split"]
for col in ("element", "adsorbate"):
    if col in df.columns:
        meta_cols.append(col)
```

**Verify**:
```
uv run python -c "
import pandas as pd; from cu_catalyst_ai.features.basic_features import build_feature_table
df = pd.read_parquet('data/processed/cathub_multi_model_table.parquet')
# Re-run featurize is not needed for testing; just check basic_features logic:
print('Test: adsorbate added to meta')
"
uv run pytest tests/test_features.py -v
```

---

### Step 2 — `feature_selection.py`：将 `adsorbate` 加入 NON_FEATURE_COLUMNS

**Files**: `src/cu_catalyst_ai/features/feature_selection.py` (line 5)

**Change**:
```python
# Before
NON_FEATURE_COLUMNS = {"catalyst_id", "adsorption_energy", "split", "element"}

# After
NON_FEATURE_COLUMNS = {"catalyst_id", "adsorption_energy", "split", "element", "adsorbate"}
```

**Verify**: 同 Step 1，`get_feature_columns` 不再把 `adsorbate` 列当特征。

---

### Step 3 — `cv.py`：新增 `run_cv_with_centering`，手动 fold 循环

**Files**: `src/cu_catalyst_ai/models/cv.py`（在现有 `run_cv` 之后新增函数）

**Change**（新增函数，不改现有 `run_cv`）:
```python
def run_cv_with_centering(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    adsorbates: pd.Series,
    n_splits: int = 5,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """GroupKFold CV with per-fold per-adsorbate target centering.

    每折内只用训练子集的 adsorbate 均值做 centering，预测后还原，
    避免 O 绝对值域主导 MSE。返回 restored R²/MAE 和 centered R²/MAE
    两套指标（后者用于诊断 train_means 估计质量）。
    """
    ...
    # 返回 pd.DataFrame 格式同 run_cv (一行 summary),
    # 额外包含 centered_r2_mean/std, centered_mae_mean/std 列
```

返回结构（一行 DataFrame）：

| 列 | 说明 |
|----|------|
| `mae_mean` / `mae_std` | 还原后 MAE（主要指标） |
| `rmse_mean` / `rmse_std` | 还原后 RMSE |
| `r2_mean` / `r2_std` | 还原后 R²（主要指标） |
| `centered_r2_mean` / `centered_r2_std` | centered 空间 R²（诊断用） |
| `centered_mae_mean` / `centered_mae_std` | centered 空间 MAE（诊断用） |

**Verify**:
```
uv run pytest tests/test_models.py -v
```

---

### Step 4 — `train.py`：路由到 `run_cv_with_centering`；hold-out 也做 centering；保存 `ads_mean_map`

**Files**: `src/cu_catalyst_ai/models/train.py`

**Change**:
1. `train_model` 签名加 `adsorbate_col: str | None = "adsorbate"` 参数
2. 检测到多 adsorbate（`df[adsorbate_col].nunique() >= 2`）时路由到 `run_cv_with_centering`
3. hold-out 测试：用全量 `train_df` 的 adsorbate 均值 centering `y_train`，训练 final model；predict + 还原后计算 test metrics
4. 把 `ads_mean_map` 写入 joblib bundle：`{"model": ..., "feature_columns": ..., "ads_mean_map": {...}}`
5. 同时在 `metrics` DataFrame 里记录 `centered_r2_mean` / `centered_mae_mean`（透传）

**Warning guard**（低优先级实现，顺手加）：
```python
for ads, cnt in train_df.groupby(adsorbate_col).size().items():
    if cnt < 30:
        logger.warning("Fold centering: adsorbate '%s' has only %d train samples; "
                       "mean estimate may be unstable.", ads, cnt)
```

**Verify**:
```
uv run pytest tests/test_models.py -v
```

---

### Step 5 — `tests/test_models.py`：新增 centering 专项测试

**Files**: `tests/test_models.py`

**新增两个 test**:

**`test_run_cv_with_centering_no_leakage`**  
构造含 CO / O 两种 adsorbate + 5 种金属的合成 DataFrame，验证：
- 每折内，centering 均值只从该折的训练行计算（通过比较手动算的 fold mean vs 函数内部使用的 mean）
- 还原后 y_pred 在原始量纲上与 y_true 相差合理

**`test_centering_noop_single_adsorbate`**  
构造只有 CO 一种 adsorbate 的 DataFrame，调用 `train_model(adsorbate_col="adsorbate")`，
验证输出与不传 `adsorbate_col` 时完全一致（MAE/R² 相同）。

**Verify**:
```
uv run pytest tests/test_models.py::test_run_cv_with_centering_no_leakage -v
uv run pytest tests/test_models.py::test_centering_noop_single_adsorbate -v
```

---

### Step 6 — 重新 featurize I-2 数据并运行 J-group GPR 实验

**Files**: 无新文件，只是重新跑 pipeline

```bash
# 重跑 featurize（因为 Step 1 改了 build_feature_table，需要更新 parquet）
uv run python -m cu_catalyst_ai.cli task=featurize \
    data=cathub_multi features=cathub_multi_ads model=gpr \
    data.cleaned_output=data/interim/cathub_multi_clean.parquet \
    data.processed_output=data/processed/cathub_multi_model_table.parquet

# 验证 adsorbate 列现在在 parquet 里
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/processed/cathub_multi_model_table.parquet')
assert 'adsorbate' in df.columns, 'adsorbate missing!'
print('OK — adsorbate in columns:', df['adsorbate'].value_counts().to_dict())
"

# 运行 J-group 训练（重用 I-2 的 cleaned data）
uv run python -m cu_catalyst_ai.cli task=train \
    data=cathub_multi features=cathub_multi_ads model=gpr \
    data.processed_output=data/processed/cathub_multi_model_table.parquet \
    paths.metrics_output=reports/tables/J_gpr_metrics.csv \
    paths.model_output=reports/models/J_gpr.joblib \
    paths.predictions_output=reports/tables/J_gpr_predictions.csv \
    paths.parity_output=reports/figures/J_gpr_parity.png \
    paths.learning_curve_output=reports/figures/J_gpr_learning_curve.png
```

**Verify**:
- `reports/tables/J_gpr_metrics.csv` 存在且 `r2_mean >= 0.28`
- `reports/tables/J_gpr_predictions.csv` 存在
- `reports/models/J_gpr.joblib` 的 bundle 包含 `ads_mean_map` 键

---

### Step 7 — 全套检查

```bash
uv run ruff check src/cu_catalyst_ai/features/basic_features.py \
    src/cu_catalyst_ai/features/feature_selection.py \
    src/cu_catalyst_ai/models/cv.py \
    src/cu_catalyst_ai/models/train.py
uv run ruff format src/cu_catalyst_ai/features/basic_features.py \
    src/cu_catalyst_ai/features/feature_selection.py \
    src/cu_catalyst_ai/models/cv.py \
    src/cu_catalyst_ai/models/train.py
uv run pytest --ignore=tests/test_cathub_fetch.py -q
```

期望：全部绿，105+ passed。

---

## Risks & mitigations

| 风险 | 缓解 |
|------|------|
| `adsorbate` 加入 `meta_cols` 后，CO-only 实验 featurize 输出 `adsorbate` 列，`get_feature_columns` 因 Step 2 的 `NON_FEATURE_COLUMNS` 已屏蔽它 → 安全 | Step 2 同步做，不单独做 |
| 旧的 `cathub_multi_model_table.parquet` 没有 `adsorbate` 列，需要重跑 featurize | Step 6 第一步是 re-featurize |
| `train_model` 新增 `adsorbate_col` 参数是接口变更，需同步更新 CLI 调用（`_run_train`）| `adsorbate_col` 默认值为 `"adsorbate"`，不传时行为不变；CLI 无需改 |
| 某折某 adsorbate 训练样本极少（< 30）导致 train_mean 不稳定 | `logger.warning` 告警，不阻塞 |
| Acceptance R² ≥ 0.28 万一 centering 只对某些折有效，均值仍未达标 | 主要验收仍以 CO 子集 LOEO R² 回正（≥ 0.10）为分析基准 |

## Rollback plan

所有改动都在独立代码路径上（`adsorbate_col` 默认触发，单 adsorbate → noop）；
若需回滚：
1. 在 `feature_selection.py` 从 `NON_FEATURE_COLUMNS` 移除 `adsorbate`
2. 在 `basic_features.py` 从 `meta_cols` 移除 `adsorbate` 分支
3. 在 `train.py` 移除 centering 路径（或设 `adsorbate_col=None`）
4. 恢复 `cathub_multi_model_table.parquet`（从已保存的 I-2 joblib / predictions 可重建）

历史 I-2 实验产物（`I2_gpr_metrics.csv`、`I2_gpr.joblib`）不会被覆盖，因为 J-group 输出到 `J_gpr_*` 路径。
