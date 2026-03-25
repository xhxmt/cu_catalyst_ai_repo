# Changelog

本文件记录项目所有重要变更，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

每次合并功能分支后请在对应版本节（或 `[Unreleased]`）下补充条目，格式如下：
- **Added**：新增功能
- **Changed**：对现有功能的变更
- **Fixed**：错误修复
- **Removed**：已移除的功能
- **Deprecated**：即将废弃的功能
- **Security**：安全相关修复

---

## [Unreleased]

### Added
- `src/cu_catalyst_ai/features/basic_features.py`：新增 `_SURFACE_DBAND_MAP`（(element, facet)→eV）和 `add_surface_dband()` 函数，实现表面级 d 带中心查表，不修改任何现有函数
- `configs/features/cathub_surface_dband.yaml`：H-Group 实验配置（与 G-Group 唯一区别：`surface_d_band_center` 替换 `d_band_center`）
- `tests/test_features.py`：新增 7 个 `add_surface_dband` 测试（精确值、超胞别名、fallback、NaN、noop、不覆盖已有列）

### Changed
- `src/cu_catalyst_ai/cli.py`：featurize 阶段在 `add_gcn()` 之后追加 `add_surface_dband()` 调用

### H-Group 实验结果（vs G-Group）

| 模型 | G-Group test_R² | H-Group test_R² | Δ |
|------|:---------:|:---------:|:--:|
| RF | — | 0.358 | — |
| RF tuned | — | 0.362 | — |
| GPR | 0.433 | 0.240 | **−0.193** |

**结论**：H-Group GPR R² 低于 G-Group（0.240 vs 0.433），表明外推的表面级 d 带中心值在当前数据量下引入了噪声，晶面信息已经被 `proxy_cn`（GCN）充分编码。根据预设决策逻辑，保留 `surface_d_band_center` 列以备查，但后续默认特征配置维持 G-Group（`cathub_gcn.yaml`）。

_（此处记录尚未发版的变更）_

---

## [0.5.0] - 2026-03-21

### Added
- `scripts/compare_models.py`：统一评估协议下的多模型对比脚本，支持生成对比报告和可视化图表
- G-Group 实验：新增电负性（`electronegativity`）作为独立特征维度，完成对应特征配置

### Changed
- `scripts/tune_rf.py`：随机森林超参数调优改为使用独立 hold-out 测试集进行最终评估，增加 CV 不稳定时的回滚机制
- `configs/model/rf_tuned.yaml`：更新为本次调优搜索得到的最优超参数

---

## [0.4.0] - 2026-03-20

### Added
- `configs/features/cathub_physics_stratified.yaml`：实验 F，按金属元素分层采样的特征配置
- `configs/features/cathub_physics_functional.yaml`：实验 E，完整物理描述符集
- `configs/features/cathub_physics_clean.yaml`：实验 D，消融对比用精简特征集
- 数据清洗阶段新增分层划分支持（`assign_splits` 扩展）

### Changed
- `src/cu_catalyst_ai/clean/split_registry.py`：支持按 `element` 列做分层分组划分

---

## [0.3.0] - 2026-03-20

### Added
- `src/cu_catalyst_ai/features/basic_features.py`：新增 `add_proxy_cn` 函数，将离散 facet 字符串映射为连续 GCN 坐标数（来源：Calle-Vallejo et al., *Nat. Chem.* 2015）
- `configs/features/cathub_proxy.yaml`：使用 proxy CN 替代 One-Hot facet 编码的特征配置
- `configs/features/cathub_proxy_only.yaml`：仅保留 proxy CN 的消融实验配置

### Fixed
- 修复 CatHub 数据中 `coordination_number` / `avg_neighbor_distance` 列全 NaN 时误伤随机森林特征重要性排序的问题（all-NaN 列现在自动排除）

---

## [0.2.0] - 2026-03-19

### Added
- `src/cu_catalyst_ai/dataio/cathub_fetch.py`：接入 Catalysis-Hub GraphQL API，支持多金属批量分页拉取、BEEF-vdW 泛函过滤
- `src/cu_catalyst_ai/features/element_features.py`：离线元素描述符查找表（d-band center、功函数、电负性、原子半径等）
- `configs/features/transition_metals.yaml`：多金属实验特征配置
- `configs/features/cathub_gcn.yaml`：含 GCN/proxy CN 的完整特征配置
- `src/cu_catalyst_ai/models/registry.py`：支持 XGBoost（`xgb`）和高斯过程回归（`gpr`，自动 StandardScaler 包裹）
- `scripts/tune_rf.py`：随机森林 RandomizedSearchCV 超参数调优脚本
- `configs/data/cathub.yaml`：CatHub 数据源配置（支持 `target_elements` 列表、`dft_functional_filter`）
- 数据清洗层新增 503 重试逻辑（指数退避）

### Changed
- `configs/config.yaml`：默认数据源切换为 `cathub`，默认模型仍为 `rf`
- Pydantic schema 放宽 `element` 字段约束，支持多种过渡金属
- `src/cu_catalyst_ai/clean/governance.py`：CatHub 来源跳过结构字段 NaN 检查

### Fixed
- 修复 CatHub API 返回 HTTP 400 的 GraphQL 查询格式错误
- 修复偶发 503 请求失败导致流程中断的问题

---

## [0.1.0] - 2026-03-18

### Added
- 项目初始化，上传首批代码
- `src/cu_catalyst_ai/cli.py`：Hydra 驱动的 CLI 入口，支持 `baseline`（全流程）及各阶段独立运行（`fetch / clean / featurize / train / explain / report`）
- `src/cu_catalyst_ai/dataio/mp_fetch.py`：Materials Project 数据获取（需 `MP_API_KEY`）；内置合成演示数据生成（`data=demo`）
- `src/cu_catalyst_ai/clean/`：数据清洗全套模块（单位归一化、重复值去除、target definition 校验、provenance 校验、Pydantic schema 验证、governance 分层）
- `src/cu_catalyst_ai/features/basic_features.py`：基础特征工程（`add_gcn`、`build_feature_table`、One-Hot 编码）
- `src/cu_catalyst_ai/features/structural_features.py`：结构比率特征
- `src/cu_catalyst_ai/models/`：线性回归与随机森林基线训练，5 折交叉验证，MAE/RMSE/R² 指标
- `src/cu_catalyst_ai/explain/shap_runner.py`：SHAP 特征重要性计算
- `src/cu_catalyst_ai/viz/`：parity plot、learning curve、特征重要性图、Markdown 摘要报告
- `configs/`：Hydra 配置体系（data / features / model / cv / target 分组）
- `tests/`：数据清洗、schema 验证、特征工程单元测试
- `scripts/hooks/`：pre-commit 检查脚本（ruff、mypy、secret scan、test gate）
- `.agents/` / `.agent/`：Codex 技能与 Superpowers 工作流框架

---

[Unreleased]: https://github.com/xhxmt/cu_catalyst_ai_repo/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/xhxmt/cu_catalyst_ai_repo/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/xhxmt/cu_catalyst_ai_repo/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/xhxmt/cu_catalyst_ai_repo/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/xhxmt/cu_catalyst_ai_repo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/xhxmt/cu_catalyst_ai_repo/releases/tag/v0.1.0
