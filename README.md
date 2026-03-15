# Cu Catalyst AI

一个面向 Cu 催化剂筛选的纯 Python 研究工作流，采用 ML 优先路线，并为后续 DFT 扩展预留接口。

## 仓库当前包含的内容

- 可直接运行的演示数据生成
- 数据清洗、数据集划分与特征工程
- 线性回归、随机森林、XGBoost、GPR 基线模型训练
- 交叉验证指标、parity plot、learning curve 与特征重要性摘要
- 供后续半自动化使用的 DFT 占位模块

## 快速开始

```bash
uv sync
uv run python -m cu_catalyst_ai.cli task=fetch
uv run python -m cu_catalyst_ai.cli task=clean
uv run python -m cu_catalyst_ai.cli task=featurize
uv run python -m cu_catalyst_ai.cli task=train model=rf
uv run python -m cu_catalyst_ai.cli task=explain model=rf
uv run python -m cu_catalyst_ai.cli task=report model=rf
uv run pytest
```

## 常用命令

```bash
uv run python -m cu_catalyst_ai.cli task=fetch data=demo
uv run python -m cu_catalyst_ai.cli task=train model=linear
uv run python -m cu_catalyst_ai.cli task=train model=rf
uv run python -m cu_catalyst_ai.cli task=train model=xgb
uv run python -m cu_catalyst_ai.cli task=train model=gpr
```

## 仓库结构

- `configs/`：Hydra 配置
- `src/cu_catalyst_ai/`：生产代码
- `tests/`：单元测试
- `data/`：原始、中间和处理后的数据产物
- `reports/`：图表、表格和摘要报告

## 说明

默认数据源为 `demo`，会生成一份合成 Cu 催化剂数据集，因此即使没有 API Key 也能直接跑通流程。

如果要接入 Materials Project，请设置 `MP_API_KEY`，并按项目需求扩展 [src/cu_catalyst_ai/dataio/mp_fetch.py](D:/dachuang/code/src/cu_catalyst_ai/dataio/mp_fetch.py)。
