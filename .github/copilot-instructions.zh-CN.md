# PM2.5 空气质量回归预测项目

本仓库包含使用多种回归方法预测 PM2.5 空气质量水平的机器学习脚本，包括 XGBoost、CatBoost 以及基于趋势-残差分解的集成方法。

## 运行脚本

所有训练脚本都是独立的 Python 文件，可以直接运行：

```bash
# 基础 XGBoost 模型
python train_pm25_xgboost.py --data data.csv --target PM2.5

# CatBoost 模型
python train_pm25_catboost.py --data data.csv --target PM2.5

# 趋势 + 残差模型（主要方法）
python train_pm25_trend_residual_xgboost.py --data data.csv --target PM2.5

# 比较多种建模路径
python compare_rmse_paths.py --data data.csv

# 扫描集成权重
python sweep_blended_weights.py --data data.csv
```

### 通用命令行参数

所有训练脚本共享以下参数：
- `--data`: 训练数据 CSV 文件路径（默认：`data.csv`）
- `--target`: 目标列名（默认：`PM2.5`）
- `--test-size`: 测试集比例（默认：`0.2`）
- `--random-state`: 随机种子（默认：`42`）
- `--device`: 训练设备，可选 `cpu` 或 `gpu`（默认：`cpu`）
- `--no-tune`: 跳过超参数调优，仅使用默认参数（速度更快）
- `--no-save`: 不保存训练好的模型
- `--keep-id-cols`: 保留 ID 列（默认会自动移除）

### GPU 训练

启用 GPU 加速：

```bash
# XGBoost GPU 训练
python train_pm25_xgboost.py --device gpu --gpu-id 0

# CatBoost GPU 训练
python train_pm25_catboost.py --device gpu --gpu-devices 0

# 趋势-残差模型 GPU 训练
python train_pm25_trend_residual_xgboost.py --device gpu --gpu-id 0
```

## 架构概览

### 三种建模方法

1. **直接 XGBoost 建模** (`train_pm25_xgboost.py`)
   - 单一 XGBoost 模型直接预测 PM2.5
   - Pipeline 包括：预处理 → 超参数调优 → 模型训练
   - 保存到 `cache/pm25_xgboost_pipeline.joblib`

2. **直接 CatBoost 建模** (`train_pm25_catboost.py`)
   - 单一 CatBoost 模型直接预测 PM2.5
   - 原生支持类别特征（无需手动 one-hot 编码）
   - 保存到 `cache/pm25_catboost_model.cbm` + 元数据 JSON

3. **趋势-残差分解建模** (`train_pm25_trend_residual_xgboost.py`) **[主要方法]**
   - 两阶段建模：线性趋势模型 + 树模型残差修正
   - 第一阶段：线性模型（LinearRegression 或 RidgeCV）学习全局趋势
   - 第二阶段：树模型（XGBoost、CatBoost 或集成）修正残差
   - 最终预测 = 趋势预测 + 残差预测
   - 支持三种残差模式：
     - `xgb`: XGBoost 残差模型
     - `catboost`: CatBoost 残差模型
     - `ensemble`: XGBoost + CatBoost 残差加权组合
   - 保存到 `cache/pm25_trend_residual_*.joblib`

### 趋势-残差特征工程

趋势-残差方法包含专门的特征工程：

- **区域先验特征**：从 PROVINCE/CITY 列计算平滑的区域统计量（均值、标准差、计数），使用带平滑的目标编码防止过拟合
- **对数变换特征**：对指定特征自动应用 log(1+x) 变换（默认对 NOX、SO2、fertilizer、manure）
- **交互特征**：工程化特征如 NOX×SO2、NOX×wind、temperature×precipitation
- **趋势预测作为特征**：线性趋势预测值以 `__linear_trend_pred__` 的形式传递给残差模型

### 数据划分策略

- **随机划分**（默认）：标准随机训练/测试集划分
- **分组划分**（`--group-col`）：确保同一组的所有样本保持在一起（例如按 COUNTY 划分以防止数据泄漏）

### 交叉验证与调参

- **RandomizedSearchCV**（XGBoost）：K 折交叉验证 + 随机超参数搜索
- **手动 CV 循环**（CatBoost）：自定义 K 折循环，每折内使用早停
- **Out-of-Fold (OOF) 预测**：用于集成权重优化，防止过拟合

## 关键约定

### 文件组织

- **训练脚本**：所有 `train_*.py` 文件位于根目录
- **实验脚本**：`compare_*.py` 和 `sweep_*.py` 用于模型对比实验
- **模型产物**：保存到 `cache/` 目录（已添加到 .gitignore）
- **诊断图表**：保存到 `.diag/` 目录（已添加到 .gitignore）
- **数据**：默认期望输入 CSV 为 `data.csv`

### 代码风格

- **中文注释**：所有文档字符串和主要注释使用中文
- **类型提示**：所有函数使用类型提示（`from __future__ import annotations`）
- **参数解析**：每个脚本使用 `argparse` 处理命令行参数
- **错误处理**：依赖导入包装在 try-except 中，提供友好的错误信息
- **Pipeline 模式**：使用 scikit-learn Pipeline 进行预处理 + 模型（便于序列化）

### 命名规范

- **脚本命名**：`train_pm25_<方法>.py` 用于训练脚本
- **模型文件**：`pm25_<方法>_pipeline.joblib` 或 `.cbm`（CatBoost）
- **诊断图表**：`pm25_<方法>_diagnostics.png`
- **函数命名**：snake_case，描述性命名（如 `build_residual_pipeline`、`tune_hyperparameters`）
- **常量命名**：UPPER_SNAKE_CASE 表示默认值（如 `DEFAULT_RIDGE_ALPHAS`、`TREND_FEATURE_NAME`）

### ID 列处理

默认情况下，所有脚本会自动检测并移除 ID 列（列名中包含 "id"，不区分大小写）以防止过拟合。这包括 `OBJECTID_1`、`sample_id` 等列。使用 `--keep-id-cols` 可保留它们。

### 模型保存

- **XGBoost 模型**：作为 scikit-learn Pipeline 对象使用 `joblib` 保存
- **CatBoost 模型**：使用原生 `.save_model()` 保存，附带单独的 JSON 元数据
- **集成模型**：保存的产物包含所有子模型和权重信息

### 诊断输出

所有训练脚本生成：
1. **指标**：测试集上的 RMSE、MAE、R²（调参期间的 CV RMSE）
2. **样本预测**：前 10 条测试集预测结果打印到控制台
3. **特征重要性**：Top 20 最重要特征
4. **诊断图表**（除非使用 `--no-plot`）：
   - 左图：实际值 vs 预测值曲线（按实际值排序）
   - 右图：残差分布直方图

### 高污染评估

趋势-残差脚本支持专门评估高污染样本的性能：
- 使用 `--high-pm25-threshold` 设置阈值（例如 `75`）
- 报告阈值以上样本的单独 RMSE、MAE、R²
- 适用于评估模型在关键空气质量事件上的表现

## 依赖包

必需的包（通过 pip 安装）：
```bash
pip install joblib scikit-learn xgboost catboost pandas numpy matplotlib
```

GPU 支持需要安装 CUDA 兼容版本：
```bash
pip install xgboost[gpu] catboost[gpu]
```

## 集成权重策略

在趋势-残差脚本中使用 `--residual-model ensemble` 时：

1. **等权重**（`--ensemble-weights equal`）：XGBoost 和 CatBoost 各 50%
2. **整体优化**（`--ensemble-weights overall`）：优化整体测试集 RMSE 最小
3. **高 PM2.5 优化**（`--ensemble-weights high_pm25`）：优化高污染子集 RMSE 最小
4. **混合优化**（`--ensemble-weights blended`）：平衡整体和高污染 RMSE 的联合目标（由 `--blended-overall-weight` 控制）

## 常见陷阱：特征视图选择

在趋势-残差 XGBoost 路径中，使用 `--xgb-feature-view` 控制残差模型使用的特征：

- `all`（默认）：使用所有原始特征 + 工程特征 + 趋势预测
- `complementary`：丢弃 CITY/COUNTY 列（已被区域先验捕获），使用 NOX、SO2、气象变量和工程特征

`complementary` 视图通常表现更好，因为减少了冗余，聚焦于物理上有意义的预测因子。
