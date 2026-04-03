# PM2.5 预测项目说明

## 项目概述

本项目围绕结构化表格数据上的 PM2.5 浓度预测任务，逐步实现了多条可对比、可复现的建模路径，包括：

1. 单模型 XGBoost 回归
2. 单模型 CatBoost 回归
3. 线性趋势 + 树模型残差修正的两阶段模型
4. XGBoost / CatBoost 残差对比与加权集成
5. 区域先验特征、`COUNTY` 消融实验、CatBoost 专家模型等扩展路线

当前最核心的训练脚本是：

- `train_pm25_trend_residual_xgboost.py`

该脚本支持以下残差建模模式：

- `xgboost`
- `catboost`
- `catboost_experts`
- `ensemble`

项目中还包含两个实验辅助脚本：

- `compare_rmse_paths.py`：在同一切分下比较多条路径
- `sweep_blended_weights.py`：扫描集成权重配置


## 目录结构

```text
.
|-- data.csv
|-- train_pm25_xgboost.py
|-- train_pm25_catboost.py
|-- train_pm25_trend_residual_xgboost.py
|-- compare_rmse_paths.py
|-- sweep_blended_weights.py
|-- cache/
|-- .diag/
```

目录约定：

- `cache/`：保存训练好的模型和序列化产物
- `.diag/`：保存诊断图、对比表、实验结果等分析文件


## 数据集说明

当前训练数据为 `data.csv`。

基础统计：

- 样本数：`2676`
- 总列数：`13`
- 缺失值：`0`
- 目标列：`PM2.5`

字段构成：

- 疑似标识列：`OBJECTID_1`
- 地区列：`PROVINCE`、`CITY`、`COUNTY`
- 数值特征：`AET`、`ppt`、`tem`、`wind`、`NOX`、`SO2`、`fertilzier`、`manure`
- 目标列：`PM2.5`

当前数据的几个关键特征：

- `COUNTY` 唯一值数为 `2676`，几乎每条样本一个值
- `CITY` 唯一值数为 `338`
- `PROVINCE` 唯一值数为 `31`
- `PM2.5` 为中等程度右偏
- `NOX`、`SO2` 偏态很强

这些特征直接影响了后续建模设计，例如：

- 是否需要 `log1p` 特征扩展
- 是否优先采用 CatBoost 处理高基数类别
- 是否引入区域层级先验
- 是否需要专门做 `COUNTY` 消融实验


## 环境依赖

推荐 Python 版本：

- `Python 3.10+`

安装依赖：

```bash
pip install joblib scikit-learn xgboost catboost pandas numpy matplotlib
```


## 快速开始

### 1. 训练单模型 XGBoost

```bash
python train_pm25_xgboost.py
```

关闭调参的快速版本：

```bash
python train_pm25_xgboost.py --no-tune
```


### 2. 训练单模型 CatBoost

```bash
python train_pm25_catboost.py
```

关闭调参的快速版本：

```bash
python train_pm25_catboost.py --no-tune
```


### 3. 训练两阶段趋势 + 残差模型

线性趋势 + CatBoost 残差：

```bash
python train_pm25_trend_residual_xgboost.py --residual-model catboost
```

线性趋势 + XGBoost 残差：

```bash
python train_pm25_trend_residual_xgboost.py --residual-model xgboost
```

CatBoost 专家模型：

```bash
python train_pm25_trend_residual_xgboost.py --residual-model catboost_experts
```

双残差模型集成：

```bash
python train_pm25_trend_residual_xgboost.py --residual-model ensemble
```


### 4. 统一比较多条路径

```bash
python compare_rmse_paths.py
```

只比较关键 no-log 路径：

```bash
python compare_rmse_paths.py --paths xgb_complementary,catboost,catboost_experts --no-log-features
```


### 5. 扫描集成权重

```bash
python sweep_blended_weights.py
```


## 核心脚本：`train_pm25_trend_residual_xgboost.py`

### 功能说明

该脚本实现的是“两阶段回归”：

1. 使用线性模型学习 PM2.5 的全局趋势；
2. 使用树模型学习“真实值 - 趋势预测值”的残差；
3. 最终预测值为：

```text
最终预测 = 趋势预测 + 残差预测
```

### 常用命令

当前较稳妥的 CatBoost 主线路径：

```bash
python train_pm25_trend_residual_xgboost.py --residual-model catboost --no-log-features --no-region-priors
```

跨城市分组验证：

```bash
python train_pm25_trend_residual_xgboost.py --residual-model catboost --validation-mode group --group-col CITY
```

CatBoost 专家模型：

```bash
python train_pm25_trend_residual_xgboost.py --residual-model catboost_experts --no-log-features --no-region-priors
```

GPU 训练：

```bash
python train_pm25_trend_residual_xgboost.py --residual-model catboost --device gpu --gpu-id 0
```


### 参数详解

#### 1. 基础输入输出参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---:|---|
| `--data` | str | `data.csv` | 训练数据路径 |
| `--target` | str | `PM2.5` | 目标列名 |
| `--test-size` | float | `0.2` | 测试集比例 |
| `--random-state` | int | `42` | 随机种子 |
| `--model-out` | str | 自动 | 模型输出路径 |
| `--plot-out` | str | 自动 | 诊断图输出路径 |
| `--run-name` | str | 自动 | 实验名，用于组织输出目录 |
| `--no-plot` | flag | 关闭 | 不生成诊断图 |
| `--no-save` | flag | 关闭 | 不保存模型 |

默认输出规则：

- 模型保存到 `cache/<run-name>/`
- 图表和表格保存到 `.diag/<run-name>/`


#### 2. 验证与调参参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---:|---|
| `--validation-mode` | `group/random/within_group` | `random` | 验证方式 |
| `--group-col` | str | `CITY` | 分组验证使用的列 |
| `--trend-cv-folds` | int | `5` | 趋势 OOF 折数 |
| `--residual-cv-folds` | int | `5` | 残差调参 / OOF 折数 |
| `--n-iter` | int | `20` | 随机搜索迭代次数 |
| `--no-tune` | flag | 关闭 | 关闭残差模型调参 |

建议：

- 如果当前目标是优化全局 RMSE，优先从 `--validation-mode random` 开始
- 如果更关心跨城市泛化，使用 `--validation-mode group --group-col CITY`
- 如果你希望测试集样本所属城市在训练集中也出现过，使用 `--validation-mode within_group --group-col CITY`


#### 3. 趋势模型参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---:|---|
| `--trend-model` | `ridge/linear` | `ridge` | 趋势模型类型 |
| `--trend-feature-view` | `all/numeric_only` | `all` | 趋势模型可见的特征视图 |

说明：

- `ridge` 是当前推荐默认值
- `linear` 适合做基线对照
- `all` 允许趋势模型看到数值和类别特征
- `numeric_only` 强制趋势模型只看数值特征，让地区类别信息只对残差模型可见


#### 4. 残差模型参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---:|---|
| `--residual-model` | enum | `xgboost` | 残差模型类型 |

可选值：

- `xgboost`：线性趋势 + XGBoost 残差
- `catboost`：线性趋势 + CatBoost 残差
- `catboost_experts`：线性趋势 + CatBoost 分 regime 专家模型
- `ensemble`：线性趋势 + XGBoost/CatBoost 残差集成


#### 5. XGBoost 残差相关参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---:|---|
| `--xgb-feature-view` | `auto/all/complementary` | `auto` | XGBoost 特征视图 |
| `--xgb-complementary-drop-cols` | str | `CITY,COUNTY` | complementary 视图下移除的高基数列 |
| `--no-xgb-complementary-interactions` | flag | 关闭 | 关闭 complementary 交互特征 |
| `--no-xgb-all-interactions` | flag | 关闭 | 关闭 all 视图的精选交互特征 |

说明：

- `all`：保留所有可用特征
- `complementary`：弱化高基数地区列，强调数值交互
- `auto`：在 `ensemble` 模式下自动使用 `complementary`，其他情况使用 `all`


#### 6. CatBoost 专家模型参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---:|---|
| `--catboost-expert-regime` | enum | `trend_high_or_low_wind` | 特殊 regime 的定义方式 |
| `--catboost-expert-trend-quantile` | float | `0.75` | 高趋势样本分位数阈值 |
| `--catboost-expert-low-wind-quantile` | float | `0.35` | 低风速样本分位数阈值 |
| `--catboost-expert-special-weight` | float | `0.75` | 特殊 regime 内特殊专家的融合权重 |
| `--catboost-expert-min-samples` | int | `120` | 启用特殊专家所需的最少样本数 |
| `--catboost-early-stopping-rounds` | int | `80` | CatBoost 早停轮数 |

支持的 regime：

- `trend_high`
- `low_wind`
- `trend_high_or_low_wind`


#### 7. 集成模型参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---:|---|
| `--ensemble-weight-xgb` | float | 自动 | 手动指定 XGBoost 权重 |
| `--ensemble-weight-objective` | `overall/high_pm25/blended` | `blended` | 自动定权目标 |
| `--ensemble-blended-overall-weight` | float | `0.7` | blended 目标中整体 RMSE 的权重 |
| `--ensemble-weight-grid-step` | float | `0.02` | 自动搜索权重的步长 |

目标说明：

- `overall`：优先最小化整体 RMSE
- `high_pm25`：优先最小化高污染子集 RMSE
- `blended`：同时兼顾整体 RMSE 和高污染子集 RMSE


#### 8. 特征工程与区域先验参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---:|---|
| `--log-feature-cols` | str | `NOX,SO2,fertilzier,manure` | `log1p` 候选列 |
| `--no-log-features` | flag | 关闭 | 关闭 `log1p` 扩展 |
| `--region-prior-cols` | str | `PROVINCE,CITY` | 区域先验使用的层级列 |
| `--region-prior-smoothing` | float | `15.0` | 区域先验平滑强度 |
| `--no-region-priors` | flag | 关闭 | 关闭区域先验 |

区域先验会构造：

- 地区支持度特征
- 目标先验特征
- 残差先验特征
- 相邻层级差值特征


#### 9. 样本加权与高污染评估参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---:|---|
| `--sample-weight-mode` | `none/high_pm25` | `none` | 样本加权方式 |
| `--high-pm25-quantile` | float | `0.75` | 高 PM2.5 加权阈值分位数 |
| `--high-pm25-weight` | float | `2.0` | 高污染样本权重倍数 |
| `--eval-high-pm25-quantile` | float | `0.75` | 高污染评估阈值分位数 |
| `--no-high-pm25-report` | flag | 关闭 | 不输出高污染子集指标 |


#### 10. 设备与数据清理参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---:|---|
| `--device` | `cpu/gpu` | `cpu` | 残差模型训练设备 |
| `--gpu-id` | int | `0` | GPU 编号 |
| `--keep-id-cols` | flag | 关闭 | 保留疑似标识列 |

默认情况下，脚本会自动尝试剔除类似 `OBJECTID_1` 这样的疑似 ID 列。


## `train_pm25_xgboost.py`

### 功能

训练单模型 XGBoost 回归器，包含：

- 预处理流水线
- 随机搜索调参
- 模型保存
- 诊断图输出

### 主要参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--data` | `data.csv` | 训练数据路径 |
| `--target` | `PM2.5` | 目标列 |
| `--test-size` | `0.2` | 测试集比例 |
| `--cv-folds` | `5` | 交叉验证折数 |
| `--n-iter` | `30` | 随机搜索迭代次数 |
| `--device` | `cpu` | CPU 或 GPU |
| `--gpu-id` | `0` | GPU 编号 |
| `--no-tune` | 关闭 | 关闭调参 |
| `--model-out` | `cache/pm25_xgboost_pipeline.joblib` | 模型保存路径 |
| `--plot-out` | `.diag/pm25_xgboost_diagnostics.png` | 诊断图路径 |
| `--keep-id-cols` | 关闭 | 保留疑似 ID 列 |
| `--no-plot` | 关闭 | 不生成诊断图 |
| `--no-save` | 关闭 | 不保存模型 |


## `train_pm25_catboost.py`

### 功能

训练单模型 CatBoost 回归器，利用 CatBoost 对类别变量的原生支持进行 PM2.5 预测。

### 主要参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--data` | `data.csv` | 训练数据路径 |
| `--target` | `PM2.5` | 目标列 |
| `--test-size` | `0.2` | 测试集比例 |
| `--cv-folds` | `5` | 交叉验证折数 |
| `--n-iter` | `20` | 随机参数采样次数 |
| `--early-stopping-rounds` | `80` | 早停轮数 |
| `--device` | `cpu` | CPU 或 GPU |
| `--gpu-devices` | `0` | GPU 设备字符串 |
| `--no-tune` | 关闭 | 关闭调参 |
| `--keep-id-cols` | 关闭 | 保留疑似 ID 列 |
| `--model-out` | `cache/pm25_catboost_model.cbm` | 模型路径 |
| `--meta-out` | `cache/pm25_catboost_meta.json` | 元信息路径 |
| `--no-save` | 关闭 | 不保存模型与元信息 |


## `compare_rmse_paths.py`

### 功能

在同一份训练/测试切分下，对多条建模路径进行统一比较，并输出：

- 对比 CSV 表
- 路径对比图

### 支持的路径名

- `xgb_all`
- `xgb_complementary`
- `catboost`
- `catboost_experts`
- `ensemble_overall_all`
- `ensemble_overall_complementary`
- `ensemble_blended_complementary`

### 常用命令

比较主路径：

```bash
python compare_rmse_paths.py --no-log-features
```

只比较 CatBoost 与专家模型：

```bash
python compare_rmse_paths.py --paths catboost,catboost_experts --no-log-features --no-region-priors
```

比较不同区域先验平滑强度：

```bash
python compare_rmse_paths.py --paths catboost --no-log-features --region-prior-smoothing-values off,5,10,15,20
```

比较 `COUNTY` 保留与删除：

```bash
python compare_rmse_paths.py --paths catboost --no-log-features --county-ablation-modes keep,drop --region-prior-smoothing-values off
```

### 关键参数

除继承主训练脚本的大部分参数外，还新增：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--paths` | 默认全路径列表 | 需要比较的路径 |
| `--out` | 自动 | CSV 输出路径 |
| `--plot-out` | 自动 | 图片输出路径 |
| `--no-plot` | 关闭 | 不生成对比图 |
| `--region-prior-smoothing-values` | 空 | 在一次运行中比较多个区域先验设置 |
| `--county-ablation-modes` | `keep` | 比较 `COUNTY` 的 keep / drop 模式 |


## `sweep_blended_weights.py`

### 功能

批量扫描 `blended` 集成目标下的权重配置，并输出结果表，便于分析不同 `overall_weight` 对整体 RMSE 与高污染子集 RMSE 的影响。

### 示例

```bash
python sweep_blended_weights.py --include-baselines
```


## 诊断输出说明

项目会根据不同脚本输出以下诊断信息：

- 整体 RMSE、MAE、R2
- 高污染子集 RMSE、MAE、R2
- 测试集预测样例
- 特征重要性
- 真实值与预测值拟合图
- 残差分布图
- 多路径对比 CSV 表
- 多路径柱状图 / 对比图


## 当前实验阶段的经验结论

以下结论来自当前项目中已有的 no-tune 或轻量实验结果。

### 1. 当前 no-log 整体优于 log

代表性结果：

| 路径 | no-log RMSE | log RMSE |
|---|---:|---:|
| `catboost` | `9.6773` | `9.8813` |
| `xgb_complementary` | `9.7896` | `9.8963` |
| `xgb_all` | `9.9612` | `10.1545` |

### 2. 当前 no-tune 主线中 CatBoost 最强

在 no-log 条件下：

- `catboost` 的整体 RMSE 最优
- ensemble 路线在 no-tune 情况下经常退化为纯 CatBoost

### 3. 区域先验目前更像实验方向，而不是稳定增益项

在当前 no-tune CatBoost 对比中：

- `region prior off`: `RMSE = 9.6773`
- `s20`: `RMSE = 10.3864`
- `s15`: `RMSE = 10.4637`
- `s5`: `RMSE = 10.4806`
- `s10`: `RMSE = 10.4958`

说明区域先验在当前参数下尚未调顺。

### 4. `COUNTY` 的作用仍需要继续验证

一组后续 no-tune 对比显示：

- `keep COUNTY`: `RMSE = 9.6773`
- `drop COUNTY`: `RMSE = 9.7231`

因此，`COUNTY` 目前更适合作为待调设计点，而不是直接永久删除。

### 5. CatBoost 专家模型已经可用，但还在探索阶段

一组轻量对比结果：

- `catboost`: `RMSE = 9.7858`
- `catboost_experts`: `RMSE = 9.8648`

说明专家模型已经具备完整训练和评估能力，但还需要进一步围绕 regime 和融合权重做专项调参。


## 当前推荐起点

### 如果你的目标是全局 RMSE

```bash
python train_pm25_trend_residual_xgboost.py --residual-model catboost --no-log-features --no-region-priors
```

### 如果你的目标是统一比较多条路径

```bash
python compare_rmse_paths.py --paths xgb_complementary,catboost,catboost_experts --no-log-features --region-prior-smoothing-values off
```

### 如果你的目标是跨城市泛化

```bash
python train_pm25_trend_residual_xgboost.py --residual-model catboost --validation-mode group --group-col CITY --no-log-features --no-region-priors
```


## 常见问题

### 1. XGBoost GPU 出现 device mismatch 警告

如果出现类似：

```text
Falling back to prediction using DMatrix due to mismatched devices
```

通常意味着：

- booster 在 GPU 上
- 但经过 sklearn 预处理后的输入仍在 CPU 上

这不一定代表训练失败，但会让预测变慢。

### 2. CatBoost GPU 报错

如果 GPU 训练失败，可先切回 CPU：

```bash
--device cpu
```

再检查本地 CUDA 与驱动是否匹配。


## 后续建议

如果你还要继续沿着这条路线优化，比较自然的下一步是：

1. 对当前最优路径做多随机种子复核
2. 对 `catboost_experts` 单独做 regime 和权重扫描
3. 联合评估区域先验与 `COUNTY` 消融
4. 在 `random` 与 `group` 两种验证模式下分别复核结论
