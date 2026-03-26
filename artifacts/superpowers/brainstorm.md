## Goal

实现 per-adsorbate target centering：训练前将每种吸附质的吸附能减去该吸附质在**训练集**上的均值，使不同吸附质的 y 分布对齐，从而避免 MSE 被 O 绝对值域主导，提升多吸附质混合模型（I-2/I-3）的 R² 和 MAE。

## Constraints

- Centering 均值**只能从训练折的训练部分**计算（GroupKFold 每折内各自算），防止数据泄露
- 预测时必须**还原**（+ train_means[adsorbate]）才能与原始 y 对比评估
- 现有 `run_cv` 的 `cross_validate` 框架不支持 fold-aware target transform，需要改为手动 fold 循环
- `train_model` 的最终 hold-out 测试也需做同样的 centering（全量训练集算 mean，然后 apply 到 test_df）
- CO-only 实验（没有 `adsorbate` 列或只有 1 种吸附质）不受影响，行为不变（noop）
- 不能修改公共接口的输出列名（`adsorption_energy` 仍是主要列）
- `adsorbate` 列必须在特征表中保留（需要核查）

## Known context

- **数据集**：I-2 = CatHub CO（300 train）+ Mamun O/OH（176 train），`adsorbate` ∈ {CO, O, OH}
- **当前问题**：I-2 全集 LOEO R² = 0.229（GPR），CO-only H 组 LOEO R² ≈ −0.05；O 的绝对值（−6～−2 eV）相比 CO（−2～+1 eV）高 4× 范围，MSE 被 O 主导
- **模型**：H_rf / I2_gpr；GPR 的 `clone()+fit()` 在手动 fold 循环中可用
- **当前 CV 路径**：`cv.py::run_cv → GroupKFold → cross_validate`（sklearn 封装），不支持 fold-aware y 变换
- **保存的模型**：`train_model` 最终做全量训练集 fit，`ads_mean_map` 需写入 joblib bundle

## Risks

| 风险 | 等级 | 应对 |
|------|------|------|
| test_df 中有训练集未见的 adsorbate → map 返回 NaN | 中 | 对未知 adsorbate 用全局均值 fallback，并警告 |
| `adsorbate` 列在 feature_df 中被丢弃 → centering 无法做 | 高 | 在 `train_model` 入参的 `df` 里检查；`feature_df` 必须携带 metadata 列 |
| CO-only 实验误触发 centering（只有 1 种 adsorbate，无意义） | 低 | guard：unique adsorbate < 2 时 skip |
| hold-out test 用全量训练集 mean vs CV 用各折训练子集 mean，微小不一致 | 低 | 正确做法，文档说明即可 |
| 改变 y 后 SHAP 值单位随之改变 | 低 | explain 阶段用还原后的 model，说明单位 |

## Options

### Option A — 仅改 `cv.py`，手动 fold 循环替代 `cross_validate`
- ✅ 改动范围最小，CO-only 路径照旧
- ❌ `run_cv` 签名变复杂（需传入原始 df / adsorbate_col），hold-out test 仍需单独处理

### Option B — 引入 `AdsorbateCenteringWrapper`（sklearn TransformerMixin）
- ✅ 原生支持 `cross_validate`
- ❌ sklearn 标准接口不支持 X 和 y 的 adsorbate 耦合，实现复杂，测试成本高

### Option C — 改 `train_model` + `cv.py`，传入 adsorbate Series，手动 fold 循环（**推荐**）

扩展 `train_model` 签名加 `adsorbate_col: str | None = "adsorbate"`：

1. 检测 `adsorbate_col` 是否存在且 unique ≥ 2
2. 路由到新的 `run_cv_with_centering`（手动 fold 循环）
3. 每折内算 `train_means`，centering `y_train / y_test`，训练预测，还原后算 R²/MAE
4. hold-out test 同理（全量训练集算 mean）
5. `ads_mean_map` 存进 joblib bundle，供推断时调用
6. CO-only / 单 adsorbate → fallback 到原有 `run_cv`

- ✅ 改动聚焦，语义清晰，向下兼容
- ❌ run_cv 和 run_cv_with_centering 两套路径需各自测试

### Option D — 在 clean/featurize 阶段预先写死 centered target 列
- ❌ 全局 mean → 信息泄露，不符合正确 LOEO CV 语义

## Recommendation

**选 Option C**。

实现步骤：
1. `cv.py` 新增 `run_cv_with_centering(model, X, y, adsorbates, groups, n_splits)` — 手动 fold 循环，per-fold centering，还原后返回 summary + fold_preds
2. `train.py::train_model` 新增 `adsorbate_col` 参数（默认 `"adsorbate"`），自动检测多 adsorbate 时路由到新函数；最终 hold-out 也做 centering；`ads_mean_map` 写入 joblib bundle
3. 新增测试：`test_run_cv_with_centering_no_leakage` 验证 fold-aware mean；`test_centering_noop_single_adsorbate`
4. 运行 I-2 GPR，对比 LOEO R²（预期 0.23 → 0.35–0.45）

## Acceptance criteria

- [ ] `run_cv_with_centering` 通过测试，验证 fold isolation（每折只用训练子集算 mean）
- [ ] 测试验证：单 adsorbate 时行为与 `run_cv` 完全一致
- [ ] I-2 GPR 全集 LOEO Test R² ≥ 0.35（相对当前 0.229 有提升）
- [ ] CO 子集 R² 不退步超过 0.05
- [ ] `ads_mean_map` 保存在 joblib bundle，推断时可还原
- [ ] ruff check + format 通过；pytest 全部 pass
