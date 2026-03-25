## Goal

在现有 ML 工作流中新增**表面（面指数）级 d 带中心**特征 `surface_d_band_center`，作为 H-Group 实验，与 G-Group（纯体相 `d_band_center`）做单变量纯净对比，验证晶面精度是否提升 CO 吸附能预测性能。

---

## Constraints

- **绝对不动历史实验**：A–G 组使用的任何文件（`element_features.py` 的 `_ELEMENT_DATA`、`basic_features.py` 的 `add_gcn`/`add_proxy_cn`）均只能新增，不能修改。
- **`d_band_center` 列保持不变**：新列名为 `surface_d_band_center`，两列并存于 processed Parquet 文件中，由配置文件决定哪列进入模型。
- **数据覆盖率现实**：数据集中 94.5% 是 Cu，Cu 有 100/111/211/310/511 五个晶面数据；其他 10 种金属几乎只有 111 或 211，且每种最多 6–11 条。

---

## Known Context

### 代码现状（读代码确认）

- **`d_band_center` 的实际位置**：`src/cu_catalyst_ai/features/element_features.py` 的 `_ELEMENT_DATA` 字典（Ruban et al. 1997，体相最密排面值），通过 `enrich_with_element_features()` 注入 DataFrame。  
  ⚠️ **用户方案说"在 `basic_features.py` 中新增 `_SURFACE_DBAND_MAP`"——这个位置是正确的**，与 `_GCN_MAP` 并列是最自然的风格；`element_features.py` 只负责一维体相值，不应扩展为二维。
- **`basic_features.py`**：现有 `_GCN_MAP`（facet→float），`add_proxy_cn()`，`add_gcn()`，`build_feature_table()`。拟新增的 `_SURFACE_DBAND_MAP`（`(element, facet)→float`）风格完全吻合。
- **`cli.py` featurize 顺序**：`enrich_with_element_features()` → `add_proxy_cn()` → `add_gcn()`，新函数在 `add_gcn()` 之后插入，天然可以访问已注入的 `d_band_center` 列作为 fallback 来源。
- **G 组配置**：`configs/features/cathub_gcn.yaml` 使用 `d_band_center + gcn + electronegativity`，H 组唯一区别是替换第一项。
- **现有测试**：`tests/test_features.py` 已覆盖 `add_gcn`/`add_proxy_cn`，测试风格可直接复用。

### 数据分布（影响外推可靠性）

| 金属 | 晶面 | 样本量 |
|------|------|--------|
| Cu | 100 / 111 / 211 / 310 / 511 | ~1110 |
| Ag, Pd, Rh | 111 / 211 | 6–11 |
| Au, Ir, Pt | 111 / 211 | 6–9 |
| Co, Fe, Ni | 111 | 2–3 |
| Ru | 001 | 1 |

**结论**：晶面级精度的实际受益者几乎只有 Cu。非 Cu 金属的 fallback 到体相值影响的样本极少，不会伤害模型。

---

## Risks

| 风险 | 等级 | 应对 |
|------|------|------|
| Cu(310)/Cu(511) 无文献精确值 | 中 | 用 CN-based 偏移量外推，注释注明来源；fallback 到 `d_band_center` 也可接受 |
| 外推偏移量在非 Cu/Pt 金属上不准确 | 低 | 这些金属样本量 <6，对模型权重影响可忽略 |
| `surface_d_band_center` 与 `d_band_center` 共线性 | 低 | H 组配置 **只用** `surface_d_band_center` 不同时保留两者，无共线问题 |
| H 组 R² 提升不显著（预期 0–0.05） | 中 | 即便不提升，结论本身（晶面信息已被 gcn 充分编码）也是有价值的实验结论，可写入报告 |

---

## Options

### Option A（推荐）：Conservative fallback — Cu 精确值 + 其他金属 fallback 体相值

`_SURFACE_DBAND_MAP` 只包含 Cu（和可选的 Pt）的多晶面精确/外推值，所有其他 `(metal, facet)` 组合 fallback 到该金属的 `d_band_center` 体相值（读自已注入的列）。

**优势**：Cu 精度提升，非 Cu 等价于不变，零噪声引入；实现简单；rollback 干净。

### Option B：全金属外推偏移量

对所有 11 种金属的 111/100/211/其他晶面都用 Cu 偏移规律外推，使每个 `(metal, facet)` 都有唯一值。

**分析**：理论上信息更丰富，但非 Cu 样本量极少（<6/金属），外推误差对模型的实际影响难以评估，且答辩时文献"追溯链"更长。收益极小，风险边际更高。

### Option C：仅 Cu 精确值，其他 NaN

非 Cu 行的 `surface_d_band_center` 填 NaN，`build_feature_table` 的 all-NaN 逻辑按列处理（不按行），所以不会自动丢弃。GPR 和部分 sklearn 模型不接受 NaN 输入，需要额外 imputer，实现复杂度显著上升。

---

## Recommendation

**采用 Option A**，理由：

1. Cu 占 94.5%，精确晶面值覆盖了数据集的几乎全部"有效信号区"
2. 其他金属 fallback 等价于"什么都没改，只是维持 G 组精度"，不引入噪声
3. 实现量最小（1 个字典 + 1 个函数），与 `_GCN_MAP` 风格完全一致
4. 答辩追问"这些数值哪来的"：Cu 的文献链最清晰（Mavrikakis/Hammer 体系），注释写清楚即可
5. 结果无论好坏都是干净的对照实验

**Cu 的查表值推荐**（供实现参考）：

| 晶面 | 推荐值 (eV) | 来源分类 |
|------|-------------|----------|
| Cu(111) | −2.67 | 精确文献值（Ruban 1997 / Hammer-Nørskov） |
| Cu(100) | −2.80 | 外推（111 + CN 偏移 ~−0.13 eV） |
| Cu(211) | −2.27 | 外推（step-edge, +0.40 eV vs 111） |
| Cu(310) | −2.27 | 外推（同 211，高指数 step site） |
| Cu(511) | −2.40 | 外推（宽台阶 step, +0.27 eV vs 111） |

---

## Acceptance Criteria

1. `basic_features.py` 新增 `_SURFACE_DBAND_MAP: dict[tuple[str,str], float]` 和 `add_surface_dband(df)` 函数，不修改任何现有函数
2. `_SURFACE_DBAND_MAP` 上方注释明确区分"精确文献值"与"CN-based 外推值"
3. Fallback 逻辑：`(metal, facet)` 不在表中 → 读取该行 `d_band_center` 列；连 `d_band_center` 也为 NaN → 返回 NaN
4. `cli.py` featurize 阶段调用 `add_surface_dband()`，processed Parquet 文件包含 `surface_d_band_center` 列
5. `configs/features/cathub_surface_dband.yaml` 新建，与 G 组唯一区别是 `d_band_center → surface_d_band_center`
6. `tests/test_features.py` 新增至少 5 个测试：Cu+111 精确值、Cu+211 外推值近似、unknown 金属→NaN、unknown 晶面→fallback 体相值、无 facet 列→noop
7. `uv run pytest` 全绿，`uv run ruff check src/ tests/` 无新 error
8. 用 `model=rf`、`model=rf_tuned`、`model=gpr` 分别跑 H 组实验，指标保存在 `reports/tables/`，与 G 组 R² 对比写入 CHANGELOG
