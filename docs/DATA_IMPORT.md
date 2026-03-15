# 数据导入说明

本文档汇总当前项目中真实数据的导入方式。

## 最简单的导入路径

不修改代码时，最简单的做法是：

1. 将数据文件放到 `data/raw/real/input.csv`
2. 使用项目标准列名
3. 运行一条命令

```bash
uv run python -m cu_catalyst_ai.cli task=baseline data=real_table data.input_path=data/raw/real/input.csv
```

## 标准 CSV 模板

将下面内容保存为 `data/raw/real/input.csv`：

```csv
catalyst_id,facet,adsorbate,coordination_number,avg_neighbor_distance,electronegativity,d_band_center,surface_energy,adsorption_energy,provenance,unit_adsorption_energy
cu_0001,111,CO,8.0,2.55,1.90,-1.60,1.55,-0.62,my_lab_v1,eV
cu_0002,100,CO,7.5,2.60,1.90,-1.48,1.42,-0.41,my_lab_v1,eV
cu_0003,110,CO,8.7,2.49,1.90,-1.72,1.63,-0.83,my_lab_v1,eV
```

然后执行：

```bash
uv run python -m cu_catalyst_ai.cli task=baseline data=real_table data.input_path=data/raw/real/input.csv
```

## 精简 CSV 模板

如果你希望文件更短，可以省略固定元数据列，再通过命令行补默认值。

CSV 内容：

```csv
catalyst_id,facet,coordination_number,avg_neighbor_distance,electronegativity,d_band_center,surface_energy,adsorption_energy
cu_0001,111,8.0,2.55,1.90,-1.60,1.55,-0.62
cu_0002,100,7.5,2.60,1.90,-1.48,1.42,-0.41
cu_0003,110,8.7,2.49,1.90,-1.72,1.63,-0.83
```

命令：

```bash
uv run python -m cu_catalyst_ai.cli task=baseline data=real_table data.input_path=data/raw/real/input.csv data.fill_defaults.adsorbate=CO data.fill_defaults.provenance=my_lab_v1 data.fill_defaults.unit_adsorption_energy=eV data.fill_defaults.element=Cu
```

## 当前要求的标准列

项目当前在真实数据清洗路径上要求这些标准列：

- `catalyst_id`
- `facet`
- `adsorbate`
- `coordination_number`
- `avg_neighbor_distance`
- `electronegativity`
- `d_band_center`
- `surface_energy`
- `adsorption_energy`
- `provenance`
- `unit_adsorption_energy`

其中有些列可以通过 `data.fill_defaults.*` 自动补齐，尤其是：

- `element`
- `adsorbate`
- `provenance`
- `unit_adsorption_energy`

## 如果你的列名不同

最简单的方式是先把 CSV 表头改成项目标准列名，再导入。

如果你希望保留原始列名，可以通过 Hydra 传入映射。示例：

```bash
uv run python -m cu_catalyst_ai.cli task=fetch data=real_table data.input_path=data/raw/my_data.csv data.fill_defaults.provenance=my_lab_v1 data.column_mapping.ID=catalyst_id data.column_mapping.AdsEnergy=adsorption_energy
```

## Excel 填表说明

在 Excel 第一行中使用这些列名：

```text
catalyst_id | facet | adsorbate | coordination_number | avg_neighbor_distance | electronegativity | d_band_center | surface_energy | adsorption_energy | provenance | unit_adsorption_energy
```

字段说明如下：

| 列名 | 是否必填 | 示例 | 含义 | 建议填写 |
|---|---|---|---|---|
| `catalyst_id` | 是 | `cu_0001` | 样本唯一编号 | 不要重复 |
| `facet` | 是 | `111` | 催化剂表面晶面 | 常见值如 `111`、`100`、`110`、`211` |
| `adsorbate` | 是 | `CO` | 吸附物种 | 当前目标建议统一填 `CO` |
| `coordination_number` | 是 | `8.0` | 配位数 | 填数字 |
| `avg_neighbor_distance` | 是 | `2.55` | 平均近邻距离 | 填数字，必须大于 0 |
| `electronegativity` | 是 | `1.90` | 电负性 | 填数字 |
| `d_band_center` | 是 | `-1.60` | d-band center | 填数字 |
| `surface_energy` | 是 | `1.55` | 表面能 | 填数字，建议大于 0 |
| `adsorption_energy` | 是 | `-0.62` | 吸附能 | 填数字 |
| `provenance` | 是 | `my_lab_v1` | 数据来源标识 | 建议写数据批次或来源名称 |
| `unit_adsorption_energy` | 是 | `eV` | 吸附能单位 | 推荐填 `eV`；当前支持 `eV`、`meV`、`kJ/mol` |

示例数据行：

```text
cu_0001 | 111 | CO | 8.0 | 2.55 | 1.90 | -1.60 | 1.55 | -0.62 | my_lab_v1 | eV
cu_0002 | 100 | CO | 7.5 | 2.60 | 1.90 | -1.48 | 1.42 | -0.41 | my_lab_v1 | eV
cu_0003 | 110 | CO | 8.7 | 2.49 | 1.90 | -1.72 | 1.63 | -0.83 | my_lab_v1 | eV
```

## 主要配置入口

真实数据导入配置位于：

- `configs/data/real_table.yaml`

最常用的配置项有：

- `data.input_path`
- `data.column_mapping`
- `data.fill_defaults.*`
- `data.target_definition`

## 输出文件

在 `real_table` 路径下，流程会写出：

- `data/raw/real/` 下的标准化原始表
- `data/interim/` 下的清洗结果
- `data/interim/` 下的 review 表

典型输出如下：

- `data/raw/real/cu_real_raw.parquet`
- `data/interim/cu_real_cleaned.parquet`
- `data/interim/cu_real_review.parquet`
