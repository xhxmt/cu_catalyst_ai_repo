# Cu Catalyst AI

一个面向过渡金属催化剂筛选的纯 Python 研究工作流，以 ML 为核心，并为后续 DFT 扩展预留接口。

## 仓库当前包含的内容

- **数据获取**：支持合成演示数据（`demo`）、Catalysis-Hub API（`cathub`，多金属批量拉取）、自定义实验表格（`real_table`）三种数据源
- **数据清洗**：单位归一化、重复值去除、多层 governance 过滤（Pydantic schema 验证、target definition 校验、provenance 校验）、训练/测试集自动划分
- **特征工程**：元素描述符（d-band center、功函数、电负性等）、GCN/proxy CN 配位数、结构比率；支持 9 种特征配置（`baseline`、`cathub_gcn`、`cathub_proxy`、`transition_metals` 等）
- **模型训练**：线性回归（`linear`）、随机森林（`rf`）、随机森林调参版（`rf_tuned`）、XGBoost（`xgb`）、高斯过程回归（`gpr`）
- **模型解释**：SHAP 特征重要性计算与可视化
- **可视化报告**：parity plot、learning curve、特征重要性图、Markdown 摘要报告
- **超参数调优**：`scripts/tune_rf.py` — RandomizedSearchCV + hold-out 测试集评估，结果保存在 `reports/`
- **多模型对比**：`scripts/compare_models.py` — 统一协议下比较多个模型，生成对比报告与图表
- **DFT 占位模块**：供后续半自动化扩展使用

## 环境要求

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) 包管理器

## 快速开始

```bash
# 安装依赖
uv sync

# 一键跑通完整流程（默认使用 cathub 数据 + 随机森林）
uv run python -m cu_catalyst_ai.cli

# 或分步执行
uv run python -m cu_catalyst_ai.cli task=fetch
uv run python -m cu_catalyst_ai.cli task=clean
uv run python -m cu_catalyst_ai.cli task=featurize
uv run python -m cu_catalyst_ai.cli task=train model=rf
uv run python -m cu_catalyst_ai.cli task=explain model=rf
uv run python -m cu_catalyst_ai.cli task=report model=rf

# 运行测试
uv run pytest
```

## 常用命令

```bash
# 使用合成演示数据（无需 API，快速验证流程）
uv run python -m cu_catalyst_ai.cli data=demo

# 切换模型
uv run python -m cu_catalyst_ai.cli task=train model=linear
uv run python -m cu_catalyst_ai.cli task=train model=rf
uv run python -m cu_catalyst_ai.cli task=train model=xgb
uv run python -m cu_catalyst_ai.cli task=train model=gpr
uv run python -m cu_catalyst_ai.cli task=train model=rf_tuned

# 切换特征配置
uv run python -m cu_catalyst_ai.cli features=cathub_gcn
uv run python -m cu_catalyst_ai.cli features=cathub_proxy
uv run python -m cu_catalyst_ai.cli features=transition_metals

# 超参数调优（随机森林）
uv run python scripts/tune_rf.py

# 多模型对比
uv run python scripts/compare_models.py
```

## 数据源说明

| 数据源 | 配置 | 说明 |
|--------|------|------|
| `demo` | `data=demo` | 合成 Cu 催化剂数据集，无需任何 API Key，适合快速验证 |
| `cathub` | `data=cathub`（默认） | 从 [Catalysis-Hub](https://www.catalysis-hub.org/) GraphQL API 拉取 CO 吸附能数据，支持多金属批量查询 |
| `real_table` | `data=real_table` | 导入本地实验/DFT 表格，支持列名映射与默认值填充 |

## 模型说明

| 模型 | 配置键 | 说明 |
|------|--------|------|
| 线性回归 | `model=linear` | 基线参考模型 |
| 随机森林 | `model=rf` | 默认模型，5 折交叉验证 |
| 随机森林（调参） | `model=rf_tuned` | 使用 `tune_rf.py` 搜索得到的最优超参数 |
| XGBoost | `model=xgb` | 梯度提升树 |
| 高斯过程回归 | `model=gpr` | 自带不确定性估计，特征自动标准化 |

## 输出产物

所有产物路径均由 `configs/config.yaml` 中的 `paths:` 节点控制，按模型名称前缀区分：

| 类型 | 路径模式 |
|------|----------|
| 训练指标 | `reports/tables/{model}_metrics.csv` |
| 预测结果 | `reports/tables/{model}_predictions.csv` |
| 特征重要性 | `reports/tables/{model}_feature_importance.csv` |
| Parity plot | `reports/figures/{model}_parity.png` |
| Learning curve | `reports/figures/{model}_learning_curve.png` |
| 特征重要性图 | `reports/figures/{model}_importance.png` |
| 模型文件 | `reports/models/{model}.joblib` |
| 摘要报告 | `reports/{model}_summary.md` |

## 仓库结构

```
.
├── configs/                  # Hydra 配置
│   ├── config.yaml           # 主配置（入口）
│   ├── data/                 # 数据源配置（demo / cathub / real_table）
│   ├── features/             # 特征组配置（9 种）
│   ├── model/                # 模型配置（5 种）
│   ├── cv/                   # 交叉验证配置
│   └── target/               # 目标列定义
├── src/cu_catalyst_ai/       # 生产代码
│   ├── clean/                # 数据清洗与 governance
│   ├── dataio/               # 数据获取（demo、CatHub、MP、real_table）
│   ├── features/             # 特征工程
│   ├── models/               # 模型注册与训练
│   ├── explain/              # SHAP 解释
│   ├── viz/                  # 可视化（parity plot、learning curve 等）
│   ├── schemas/              # Pydantic 数据 schema
│   ├── utils/                # 通用工具（IO、日志、随机种子）
│   ├── dft/                  # DFT 占位模块（扩展用）
│   └── cli.py                # CLI 入口
├── scripts/
│   ├── tune_rf.py            # 随机森林超参数调优
│   └── compare_models.py     # 多模型对比
├── tests/                    # 单元测试
├── data/                     # 数据产物（raw / interim / processed）
└── reports/                  # 报告产物（figures / tables / models）
```

## 注意事项

- **Catalysis-Hub**：默认以 BEEF-vdW 泛函过滤，可在 `configs/data/cathub.yaml` 中调整 `dft_functional_filter` 和 `target_elements`。
- **Materials Project**：如需接入 MP，请设置环境变量 `MP_API_KEY`，并参考 [`src/cu_catalyst_ai/dataio/mp_fetch.py`](src/cu_catalyst_ai/dataio/mp_fetch.py)。
- **可复现性**：所有随机操作均通过 `project.seed`（默认 42）统一控制；数据划分由 `configs/cv/` 决定。
- **数据治理**：原始数据只追加不覆盖；被 governance 层隔离的行写入 `data/interim/*_review.parquet` 供人工审查。

## 变更日志

所有版本的功能变更、修复和新增内容均记录在 [CHANGELOG.md](CHANGELOG.md)。

每次提交包含生产代码变更时（`src/` 或 `scripts/`），请同步更新 `CHANGELOG.md` 中的 `[Unreleased]` 节。

## 贡献指南

### 更新变更日志

在 `CHANGELOG.md` 的 `[Unreleased]` 节下追加条目，格式说明：

```markdown
## [Unreleased]

### Added
- 新增 xxx 功能（文件：`src/cu_catalyst_ai/xxx.py`）

### Changed
- 修改 xxx 行为（原因：...）

### Fixed
- 修复 xxx 问题（相关 issue / 实验编号）
```

发版时，将 `[Unreleased]` 重命名为 `[x.y.z] - YYYY-MM-DD`，并在文件底部添加对应的比较链接。

### 安装 pre-commit CHANGELOG 检查

```bash
# 将 hook 脚本写入 git hook 目录
cp scripts/hooks/pre-commit-changelog.sh .git/hooks/pre-commit-changelog
chmod +x .git/hooks/pre-commit-changelog

# 在已有的 pre-commit hook 末尾追加调用（如 .git/hooks/pre-commit 存在）
echo 'bash "$(git rev-parse --show-toplevel)/scripts/hooks/pre-commit-changelog.sh"' >> .git/hooks/pre-commit
```

安装后，每次 `git commit` 时若暂存了 `src/` 或 `scripts/` 下的文件但未更新 `CHANGELOG.md`，将收到警告并阻止提交。
紧急情况下可使用 `git commit --no-verify` 跳过检查。
