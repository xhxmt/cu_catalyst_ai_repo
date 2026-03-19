# Brainstorm: 多金属 CatHub 拉取 + 元素特征查表

## Goal

扩展 CatHub 数据获取范围至所有过渡金属（非仅 Cu），并在特征化阶段引入元素级别的物理化学特征查表（d 带中心、功函数、电负性、原子半径、d 电子数），以便训练出 R² 显著优于当前接近零的随机森林模型。

---

## Constraints

- 不得修改现有 schema 列名（`adsorption_energy`, `facet`, `element` 等）
- 不得覆盖原始 raw 数据（append-only 规则）
- 新引入的元素特征表必须带来源文献标注（Ruban 1997 / Nørskov 教材）
- 特征查表必须是纯 Python，无需网络请求
- 不能静默修改 `target_definition` 或数据划分逻辑
- 所有实验必须能被 `uv run pytest` 通过
- DFT 模块不应阻塞 ML 主流程

---

## Known context

- 当前 CatHub 拉取配置固定 `surface_composition=Cu`、`reactants=CO`
- `cathub_fetch.py` 已支持 `surface_composition` 参数化，只需修改 `cathub.yaml`
- `CatalystRecord.element` 目前硬编码为 `Literal["Cu"]`，需要放宽
- `cathub_fetch.py` 中 `element` 字段硬编码 `"Cu"`，需要从 API 响应推断
- 特征文件 `configs/features/cathub_minimal.yaml` 当前只用 `facet` + 3 个结构列（均为 NaN）
- `src/cu_catalyst_ai/features/` 目前只有 `basic_features.py`、`structural_features.py`、`feature_selection.py`
- d 带中心参考值：Ruban A.V. et al., *J. Mol. Catal. A*, **115**, 421–429 (1997) Table 1，覆盖 27 种过渡金属

---

## Risks

| 风险 | 严重程度 | 缓解 |
|---|---|---|
| `element` 列 schema 放宽后，非 Cu 元素混入 Cu 专项分析 | 中 | `target_definition` 不改变，保持版本隔离 |
| CatHub 不同 pubId 对同一体系用不同计算设置 | 高 | 用 `dftCode`/`dftFunctional` 字段做 provenance 过滤 |
| 查表值是文献单点值，与 DFT 实际值存在偏差 | 中 | 在 provenance 中注明数据来源版本，不参与 target 计算 |
| 多金属数据量大（可能 5k~20k 行），内存/时间增加 | 低 | 分批分元素拉取，每批写入独立 parquet |
| `Literal["Cu"]` schema 校验失败（其他元素被硬拒） | 高 | 修改 `CatalystRecord.element` 为 `str` |
| 元素查表有缺失（稀有金属）| 低 | 缺失值保持 NaN，不伪造；在日志中警告 |

---

## Options

### Option A：仅换元素、保留 CatHub 架构（最小改动）
- **摘要**：修改 `cathub.yaml` 的 `surface_composition` 为空（拉全部），在 `cathub_fetch.py` 中从 API 推断 `element`，放宽 schema，新增 `element_features.py` 查表模块
- **优点**：改动集中，测试边界清晰；查表数据完全离线
- **缺点**：不过滤 DFT 方法差异；element 推断可能有误（合金表面）
- **复杂度/风险**：中/中

### Option B：分元素分批拉取 + 严格 provenance 过滤（推荐）
- **摘要**：为每个目标元素（Fe, Co, Ni, Cu, Ru, Rh, Pd, Ag, Ir, Pt, Au 等）单独查询 CatHub，只保留 `dftFunctional=BEEF-vdW` 的记录（最常用、可比），再 concat + 去重；查表模块独立
- **优点**：DFT 设置一致性好；数据来源清晰；可单元素重拉
- **缺点**：需要多次 API 请求；BEEF-vdW 过滤会减少数据量
- **复杂度/风险**：中/低

### Option C：引入 OpenCatalyst / OC20 数据集（大数据路线）
- **摘要**：下载 OC20/OC22 开放数据集，包含百万量级 DFT 计算，含结构特征
- **优点**：数据量充足；已有 d 带中心等特征
- **缺点**：数据集数 GB，本地环境可能有限；需要大量预处理
- **复杂度/风险**：高/中

---

## Recommendation

**选 Option B**：分元素拉取 + BEEF-vdW provenance 过滤 + 离线元素特征查表。

理由：
1. 数据质量优先——统一 DFT 泛函是 ML 可比性的基础
2. 元素查表（Ruban 1997）覆盖全部主流过渡金属，完全离线，无依赖
3. 改动边界清晰：`cathub.yaml`、`cathub_fetch.py`（element 推断）、新增 `element_features.py`、`configs/features/transition_metals.yaml`、schema 放宽
4. 风险最低：不改 target_definition，不改数据划分，不改现有 Cu 流水线

预期效果：d 带中心加入后，文献中类似任务 R² 通常可达 0.6~0.9。

---

## Acceptance Criteria

- [ ] `task=fetch data=cathub` 能拉取多种过渡金属（至少 5 种）的 CO 吸附数据
- [ ] `element` 列正确填充金属元素名称（非硬编码 Cu）
- [ ] `element_features.py` 查表函数：输入元素符号 → 返回 d 带中心、功函数、电负性、原子半径、d 电子数
- [ ] 所有查表值标注文献出处（Ruban 1997 / CRC Handbook）
- [ ] 缺失查表值保持 NaN，不伪造，在日志中警告
- [ ] `configs/features/transition_metals.yaml` 新增特征配置，包含元素特征列
- [ ] `task=baseline data=cathub features=transition_metals` 端到端跑通
- [ ] 新模型 R² > 0.5（对测试集）
- [ ] `uv run pytest` 全部通过
- [ ] `uv run ruff check .` 无报错
