"""
基于“线性趋势 + 树模型残差修正”的 PM2.5 回归预测脚本
=========================================================================
建模思路：
1. 先使用线性模型学习数据中的整体趋势；
2. 再使用树模型学习“真实值 - 趋势预测值”的残差；
3. 最终预测值 = 趋势预测值 + 残差预测值。

本脚本支持三种残差学习模式：
1. XGBoost 残差：结构成熟、速度快，是稳健基线；
2. CatBoost 残差：对原生类别特征更友好，适合地区类字段较多的场景；
3. 双模型集成：同时训练 XGBoost 残差与 CatBoost 残差，再按验证表现加权融合。

这种两阶段建模方式的好处是：
1. 趋势模型负责解释稳定、平滑、长期的全局关系；
2. 树模型负责补充非线性、交互项和局部复杂模式；
3. 集成模式可以综合两类树模型的优势，提高稳健性与泛化能力。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# 依赖检查：若缺少必要包，则直接停止并给出安装提示。
try:
    import joblib
    import matplotlib
    import numpy as np
    import pandas as pd
    from catboost import CatBoostError, CatBoostRegressor, Pool
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import (
        GroupKFold,
        GroupShuffleSplit,
        KFold,
        ParameterSampler,
        RandomizedSearchCV,
        train_test_split,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBRegressor
    from xgboost.core import XGBoostError
except ImportError as exc:
    missing_pkg = str(exc).split("'")[1] if "'" in str(exc) else "未知包"
    raise SystemExit(
        f"缺少依赖包：{missing_pkg}\n"
        "请先安装后再运行，例如：\n"
        "pip install joblib scikit-learn xgboost catboost pandas numpy"
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 将线性趋势预测值作为一个额外特征传给残差模型。
TREND_FEATURE_NAME = "__linear_trend_pred__"
DEFAULT_RIDGE_ALPHAS = np.logspace(-3, 3, 13)
DEFAULT_LOG_FEATURE_CANDIDATES = ("NOX", "SO2", "fertilzier", "manure")


def parse_args() -> argparse.Namespace:
    """解析命令行参数，便于在不同配置下复用脚本。"""
    parser = argparse.ArgumentParser(
        description="使用线性趋势 + 树模型残差修正模型预测 PM2.5"
    )
    parser.add_argument(
        "--data", default="data.csv", help="训练数据 CSV 路径，默认：data.csv"
    )
    parser.add_argument("--target", default="PM2.5", help="目标列名，默认：PM2.5")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="测试集比例，默认：0.2"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="随机种子，默认：42"
    )
    parser.add_argument(
        "--sample-weight-mode",
        choices=["none", "high_pm25"],
        default="none",
        help="样本加权方式：none=不加权，high_pm25=对高 PM2.5 样本加权，默认：none",
    )
    parser.add_argument(
        "--high-pm25-quantile",
        type=float,
        default=0.75,
        help="高 PM2.5 样本阈值分位数，仅在 --sample-weight-mode high_pm25 时生效，默认：0.75",
    )
    parser.add_argument(
        "--high-pm25-weight",
        type=float,
        default=2.0,
        help="高 PM2.5 样本的权重倍数，仅在 --sample-weight-mode high_pm25 时生效，默认：2.0",
    )
    parser.add_argument(
        "--eval-high-pm25-quantile",
        type=float,
        default=0.75,
        help="高污染子集评估阈值分位数，基于训练集目标值计算，默认：0.75",
    )
    parser.add_argument(
        "--no-high-pm25-report",
        action="store_true",
        help="不输出高污染子集评估指标",
    )
    parser.add_argument(
        "--log-feature-cols",
        default="NOX,SO2,fertilzier,manure",
        help="需要生成 log1p 特征的列名，逗号分隔。默认：NOX,SO2,fertilzier,manure",
    )
    parser.add_argument(
        "--no-log-features",
        action="store_true",
        help="关闭 log1p 数值特征扩展",
    )
    parser.add_argument(
        "--trend-model",
        choices=["ridge", "linear"],
        default="ridge",
        help="趋势模型类型：ridge=RidgeCV（默认，推荐），linear=普通线性回归",
    )
    parser.add_argument(
        "--residual-model",
        choices=["xgboost", "catboost", "ensemble"],
        default="xgboost",
        help=(
            "残差模型类型：xgboost=仅用 XGBoost，"
            "catboost=仅用 CatBoost，ensemble=双模型加权集成；默认：xgboost"
        ),
    )
    parser.add_argument(
        "--ensemble-weight-xgb",
        type=float,
        default=None,
        help=(
            "当 --residual-model ensemble 时，手动指定 XGBoost 残差权重，范围 [0, 1]。"
            "若不指定，则会基于训练集 OOF 预测自动搜索权重；默认：自动"
        ),
    )
    parser.add_argument(
        "--ensemble-weight-objective",
        choices=["overall", "high_pm25", "blended"],
        default="blended",
        help=(
            "双模型集成自动定权的优化目标：overall=按整体 RMSE 搜索权重，"
            "high_pm25=按高污染子集 RMSE 搜索权重，"
            "blended=同时兼顾整体 RMSE 与高污染子集 RMSE；默认：blended"
        ),
    )
    parser.add_argument(
        "--ensemble-blended-overall-weight",
        type=float,
        default=0.7,
        help=(
            "当 --ensemble-weight-objective blended 时，整体 RMSE 在联合目标中的权重，"
            "范围 [0, 1]，默认：0.7"
        ),
    )
    parser.add_argument(
        "--ensemble-weight-grid-step",
        type=float,
        default=0.02,
        help="双模型集成自动搜索权重时的步长，范围 (0, 1]，默认：0.02",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["group", "random"],
        default="random",
        help="验证方式：group=按地区分组验证，random=随机切分与随机交叉验证，默认：random",
    )
    parser.add_argument(
        "--group-col",
        default="CITY",
        help="分组验证使用的列名，默认：CITY。仅在 --validation-mode group 时生效",
    )
    parser.add_argument(
        "--trend-cv-folds",
        type=int,
        default=5,
        help="线性趋势阶段生成 OOF 预测时使用的折数，默认：5",
    )
    parser.add_argument(
        "--residual-cv-folds",
        type=int,
        default=5,
        help="残差模型调参使用的交叉验证折数，默认：5",
    )
    parser.add_argument(
        "--n-iter", type=int, default=20, help="随机搜索迭代次数，默认：20"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="残差模型训练设备：cpu 或 gpu，对 XGBoost 与 CatBoost 同时生效，默认：cpu",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU 编号。XGBoost 使用 cuda:<gpu-id>，CatBoost 使用 devices=<gpu-id>",
    )
    parser.add_argument(
        "--catboost-early-stopping-rounds",
        type=int,
        default=80,
        help="CatBoost 残差模型早停轮数，默认：80",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="关闭残差模型调参，仅使用默认参数训练",
    )
    parser.add_argument(
        "--keep-id-cols",
        action="store_true",
        help="保留疑似 ID 列（默认会自动剔除，如 OBJECTID_1）",
    )
    parser.add_argument(
        "--model-out",
        default=None,
        help="模型输出文件路径；若不指定，会根据残差模型类型自动命名",
    )
    parser.add_argument(
        "--plot-out",
        default=None,
        help="诊断图输出路径；若不指定，会根据残差模型类型自动命名",
    )
    parser.add_argument("--no-plot", action="store_true", help="不绘制并保存诊断图")
    parser.add_argument("--no-save", action="store_true", help="不保存训练好的模型")
    return parser.parse_args()


def resolve_output_paths(
    residual_model: str,
    model_out_arg: str | None,
    plot_out_arg: str | None,
) -> tuple[Path, Path]:
    """根据残差模型类型自动决定输出文件名。"""
    default_outputs = {
        "xgboost": (
            "pm25_trend_residual_xgboost.joblib",
            "pm25_trend_residual_xgboost_diagnostics.png",
        ),
        "catboost": (
            "pm25_trend_residual_catboost.joblib",
            "pm25_trend_residual_catboost_diagnostics.png",
        ),
        "ensemble": (
            "pm25_trend_residual_ensemble.joblib",
            "pm25_trend_residual_ensemble_diagnostics.png",
        ),
    }
    default_model_out, default_plot_out = default_outputs[residual_model]
    return Path(model_out_arg or default_model_out), Path(plot_out_arg or default_plot_out)


def load_data(data_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """读取 CSV 并对目标列做基础校验。"""
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件：{data_path}")

    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在，当前列：{list(df.columns)}")

    y_all = pd.to_numeric(df[target_col], errors="coerce")
    x_all = df.drop(columns=[target_col])

    valid_mask = y_all.notna()
    if valid_mask.sum() == 0:
        raise ValueError("目标列全部为空或非数值，无法训练模型。")

    if valid_mask.sum() < len(df):
        dropped = len(df) - int(valid_mask.sum())
        print(f"[提示] 目标列存在 {dropped} 条非数值/缺失记录，训练前已自动剔除。")

    x = x_all.loc[valid_mask].copy()
    y = y_all.loc[valid_mask].copy()
    return x, y


def remove_identifier_columns(x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    自动移除疑似 ID 列，避免模型记住样本编号而不是学习规律。

    当前策略比较保守：只移除列名中包含 id 的字段。
    """
    drop_cols = [col for col in x.columns if "id" in col.lower()]
    if not drop_cols:
        return x, []
    return x.drop(columns=drop_cols), drop_cols


def parse_feature_list(raw_text: str) -> list[str]:
    """将逗号分隔的参数解析为列名列表。"""
    if not raw_text.strip():
        return list(DEFAULT_LOG_FEATURE_CANDIDATES)
    return [item.strip() for item in raw_text.split(",") if item.strip()]


def add_log_transformed_features(
    x: pd.DataFrame,
    candidate_cols: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    为偏态数值特征新增 log1p 版本。

    这样做的目的：
    1. 缓和长尾分布带来的极端值影响；
    2. 让线性趋势模型更容易捕捉近似线性关系；
    3. 给残差模型提供更平滑的数值表达。

    仅当列满足以下条件时才新增：
    - 列存在；
    - 列是数值型；
    - 非空值全部 >= 0（因为 log1p 适合非负数据）。
    """
    x_new = x.copy()
    added_cols: list[str] = []
    skipped_cols: list[str] = []

    for col in candidate_cols:
        if col not in x_new.columns:
            skipped_cols.append(f"{col}(列不存在)")
            continue

        series = pd.to_numeric(x_new[col], errors="coerce")
        non_missing = series.dropna()
        if non_missing.empty:
            skipped_cols.append(f"{col}(无有效数值)")
            continue

        if (non_missing < 0).any():
            skipped_cols.append(f"{col}(存在负值)")
            continue

        log_col = f"{col}_log1p"
        x_new[log_col] = np.log1p(series)
        added_cols.append(log_col)

    return x_new, added_cols, skipped_cols


def build_sample_weight_config(
    y_train: pd.Series,
    mode: str,
    quantile: float,
    high_weight: float,
) -> dict[str, float] | None:
    """
    根据训练集目标值构造样本加权配置。

    当前提供的实用策略是 high_pm25：
    - 先用训练集 PM2.5 的分位数计算阈值；
    - 高于该阈值的样本给更高权重；
    - 这样模型会更关注高污染区间的误差。
    """
    if mode == "none":
        return None

    if not 0.0 < quantile < 1.0:
        raise ValueError("high-pm25-quantile 必须在 (0, 1) 之间。")

    if high_weight <= 1.0:
        raise ValueError("high-pm25-weight 必须大于 1.0。")

    threshold = float(np.quantile(y_train.to_numpy(dtype=float), quantile))
    return {
        "mode": mode,
        "threshold": threshold,
        "quantile": float(quantile),
        "high_weight": float(high_weight),
    }


def compute_sample_weights(
    y_values: pd.Series | np.ndarray,
    weight_config: dict[str, float] | None,
) -> np.ndarray | None:
    """根据配置生成样本权重。"""
    if weight_config is None:
        return None

    y_array = np.asarray(y_values, dtype=float)
    weights = np.ones(len(y_array), dtype=float)

    if weight_config["mode"] == "high_pm25":
        high_mask = y_array >= weight_config["threshold"]
        weights[high_mask] = weight_config["high_weight"]
        return weights

    raise ValueError(f"不支持的样本加权方式：{weight_config['mode']}")


def describe_sample_weighting(
    weight_config: dict[str, float] | None,
    y_train: pd.Series,
) -> str:
    """生成当前加权策略的说明文本。"""
    if weight_config is None:
        return "none"

    if weight_config["mode"] == "high_pm25":
        high_count = int(
            (y_train.to_numpy(dtype=float) >= weight_config["threshold"]).sum()
        )
        return (
            f"high_pm25(quantile={weight_config['quantile']:.2f}, "
            f"threshold={weight_config['threshold']:.4f}, "
            f"weight={weight_config['high_weight']:.2f}, "
            f"high_samples={high_count})"
        )

    return weight_config["mode"]


def build_high_pm25_eval_config(
    y_train: pd.Series, quantile: float
) -> dict[str, float]:
    """
    构造高污染子集评估配置。

    阈值严格基于训练集目标值计算，而不是测试集，
    这样可以避免在评估阶段“偷看”测试集分布。
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("eval-high-pm25-quantile 必须在 (0, 1) 之间。")

    threshold = float(np.quantile(y_train.to_numpy(dtype=float), quantile))
    return {"quantile": float(quantile), "threshold": threshold}


def resolve_groups(x: pd.DataFrame, group_col: str) -> pd.Series:
    """
    解析并校验分组列。

    分组列的作用不是作为标签，而是控制“哪些样本可以同时出现在训练和验证中”。
    对 PM2.5 这种存在明显空间相关性的数据，按地区分组能更真实地评估泛化能力。
    """
    if group_col not in x.columns:
        raise ValueError(f"分组列 '{group_col}' 不存在，当前可用列：{list(x.columns)}")

    groups = x[group_col].fillna("__MISSING_GROUP__").astype(str)
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError(
            f"分组列 '{group_col}' 的不同取值不足 2 个，无法进行分组切分。"
        )
    return groups


def validate_group_folds(groups: pd.Series, n_splits: int, stage_name: str) -> None:
    """校验当前分组数量是否足以支撑 GroupKFold。"""
    unique_groups = groups.nunique()
    if unique_groups < n_splits:
        raise ValueError(
            f"{stage_name} 需要至少 {n_splits} 个不同分组，但当前只有 {unique_groups} 个。"
        )


def build_cv_split_indices(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    groups_data: pd.Series | None,
    cv_folds: int,
    random_state: int,
    validation_mode: str,
    stage_name: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    统一构造交叉验证的索引切分。

    这样做有两个好处：
    1. 趋势模型、XGBoost 残差、CatBoost 残差都能复用同一套切分逻辑；
    2. 后续如果要加入更多模型，验证策略仍然保持一致，便于公平比较。
    """
    if validation_mode == "group":
        if groups_data is None:
            raise ValueError(f"{stage_name} 在分组验证模式下需要 groups_data。")
        validate_group_folds(groups_data, cv_folds, stage_name)
        splitter = GroupKFold(n_splits=cv_folds)
        return list(splitter.split(x_data, y_data, groups_data))

    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return list(splitter.split(x_data, y_data))


def split_train_test_by_group(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    按组进行训练集/测试集切分。

    与普通随机切分不同，这里保证同一个 group 不会同时出现在训练集和测试集。
    这样更接近真实场景：
    模型需要对“没见过的地区”进行泛化，而不是仅仅记住相邻样本。
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(x, y, groups))

    x_train = x.iloc[train_idx].copy()
    x_test = x.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()
    groups_train = groups.iloc[train_idx].copy()
    groups_test = groups.iloc[test_idx].copy()
    return x_train, x_test, y_train, y_test, groups_train, groups_test


def split_train_test_random(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """随机划分训练集和测试集，便于和传统 baseline 做可比实验。"""
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return x_train, x_test, y_train, y_test


def build_linear_trend_pipeline(x: pd.DataFrame, trend_model: str) -> Pipeline:
    """
    构建线性趋势模型。

    这里支持两种线性趋势模型：
    1. 数值特征：缺失值用中位数填补，再做标准化；
    2. 类别特征：缺失值用众数填补，再做 One-Hot 编码；
    3. 最终由线性模型学习整体趋势。

    默认使用 RidgeCV：
    - 它本质上仍然是线性模型；
    - 但会自动选择正则强度 alpha；
    - 在高维 one-hot 特征场景下通常比普通线性回归更稳。
    """
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = x.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    if trend_model == "ridge":
        trend_estimator = RidgeCV(alphas=DEFAULT_RIDGE_ALPHAS)
    else:
        trend_estimator = LinearRegression()

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", trend_estimator),
        ]
    )


def prepare_catboost_features(x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    将特征整理成适合 CatBoost 直接使用的格式。

    处理规则：
    1. 非数值列视为类别特征，缺失值填成固定字符串，并统一转为 str；
    2. 数值列强制转数值，非法值转为 NaN，由 CatBoost 原生处理；
    3. 返回处理后的 DataFrame 与类别列名列表。
    """
    x_new = x.copy()
    categorical_cols = x_new.select_dtypes(exclude=["number"]).columns.tolist()

    for col in categorical_cols:
        x_new[col] = x_new[col].where(x_new[col].notna(), "__MISSING__").astype(str)

    numeric_cols = [col for col in x_new.columns if col not in categorical_cols]
    for col in numeric_cols:
        x_new[col] = pd.to_numeric(x_new[col], errors="coerce")

    return x_new, categorical_cols


def build_residual_pipeline(
    x: pd.DataFrame,
    random_state: int,
    device: str,
    gpu_id: int,
) -> Pipeline:
    """
    构建残差预测模型。

    残差模型依旧使用原始特征，同时额外加入一列线性趋势预测值，
    让 XGBoost 明确知道“线性部分已经预测到了哪里”，从而更专注于学习偏差。
    """
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = x.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    model_kwargs: dict[str, Any] = {
        "objective": "reg:squarederror",
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": random_state,
    }

    if device == "gpu":
        model_kwargs.update(
            {
                "tree_method": "hist",
                "device": f"cuda:{gpu_id}",
                "n_jobs": 1,
            }
        )
    else:
        model_kwargs.update(
            {
                "tree_method": "hist",
                "n_jobs": -1,
            }
        )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", XGBRegressor(**model_kwargs)),
        ]
    )


def default_catboost_residual_params(
    random_state: int,
    device: str,
    gpu_id: int,
) -> dict[str, Any]:
    """
    构建一组适合作为残差学习起点的 CatBoost 参数。

    这里优先追求稳健：
    - 学习率适中，减少过拟合风险；
    - 迭代轮数偏大，再配合 early stopping 自动截断；
    - 保留 CatBoost 对类别特征的原生处理优势。
    """
    params: dict[str, Any] = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
        "iterations": 1200,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
    }

    if device == "gpu":
        params.update({"task_type": "GPU", "devices": str(gpu_id)})
    else:
        params.update({"task_type": "CPU", "thread_count": -1})

    return params


def calculate_metrics(
    y_true: pd.Series | np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """统一计算常用回归指标。"""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    if len(y_true_array) == 0:
        return {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}

    if len(y_true_array) < 2:
        r2_value = float("nan")
    else:
        r2_value = float(r2_score(y_true_array, y_pred_array))

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "MAE": float(mean_absolute_error(y_true_array, y_pred_array)),
        "R2": r2_value,
    }


def calculate_high_pm25_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], int]:
    """
    只在高污染子集上计算指标。

    高污染子集定义为：真实 PM2.5 >= threshold。
    这里使用真实值筛选，是为了回答“模型在高污染样本上表现如何”。
    """
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    high_mask = y_true_array >= threshold
    high_count = int(high_mask.sum())

    if high_count == 0:
        return {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}, 0

    metrics = calculate_metrics(y_true_array[high_mask], y_pred_array[high_mask])
    return metrics, high_count


def save_diagnostic_plot(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    plot_out: Path,
    model_name: str,
) -> None:
    """
    保存回归诊断图：
    1) 左图：按真实值排序后的真实曲线与预测曲线；
    2) 右图：残差分布直方图。
    """
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    residuals = y_true_array - y_pred_array

    sort_idx = np.argsort(y_true_array)
    y_true_sorted = y_true_array[sort_idx]
    y_pred_sorted = y_pred_array[sort_idx]

    plot_out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(y_true_sorted, label="Actual", linewidth=2.0, color="#1f77b4")
    axes[0].plot(
        y_pred_sorted, label="Predicted", linewidth=2.0, color="#ff7f0e", alpha=0.85
    )
    axes[0].set_title(f"{model_name}: Actual vs Predicted")
    axes[0].set_xlabel("Samples sorted by actual value")
    axes[0].set_ylabel("PM2.5")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].hist(residuals, bins=30, color="#4c78a8", alpha=0.85, edgecolor="white")
    axes[1].axvline(0.0, color="#d62728", linestyle="--", linewidth=1.5)
    axes[1].set_title(f"{model_name}: Residual distribution")
    axes[1].set_xlabel("Residual (actual - predicted)")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(plot_out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_augmented_features(x: pd.DataFrame, trend_pred: np.ndarray) -> pd.DataFrame:
    """
    将线性趋势预测值拼接回原始特征，供残差模型使用。

    这样做的直观含义是：
    “在已有线性趋势判断的基础上，再决定还需要修正多少。”
    """
    x_aug = x.copy()
    x_aug[TREND_FEATURE_NAME] = trend_pred
    return x_aug


def generate_oof_trend_predictions(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series | None,
    cv_folds: int,
    random_state: int,
    validation_mode: str,
    trend_model: str,
    weight_config: dict[str, float] | None,
) -> np.ndarray:
    """
    生成线性趋势模型的 OOF（Out-Of-Fold）预测。

    为什么需要 OOF 预测：
    1. 如果直接用“全训练集拟合后的线性模型”给训练集做预测，
       那么每一条样本都被模型见过，残差会偏乐观；
    2. 用 OOF 预测后，每个样本的趋势预测值都来自“没见过它的模型”，
       这样残差更真实，第二阶段模型更不容易发生信息泄漏。

    当 validation_mode="group" 时，进一步使用按组的 GroupKFold：
    同一个地区不会同时出现在当前 fold 的训练子集和验证子集中，
    因而得到的是“跨地区泛化”的趋势预测。
    """
    split_indices = build_cv_split_indices(
        x_data=x_train,
        y_data=y_train,
        groups_data=groups_train,
        cv_folds=cv_folds,
        random_state=random_state,
        validation_mode=validation_mode,
        stage_name="趋势模型交叉验证",
    )

    oof_pred = np.zeros(len(x_train), dtype=float)

    for fold_no, (tr_idx, val_idx) in enumerate(split_indices, start=1):
        x_tr = x_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        x_val = x_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]
        fold_weights = compute_sample_weights(y_tr, weight_config)

        trend_pipeline = build_linear_trend_pipeline(x_tr, trend_model)
        if fold_weights is None:
            trend_pipeline.fit(x_tr, y_tr)
        else:
            trend_pipeline.fit(x_tr, y_tr, model__sample_weight=fold_weights)
        fold_pred = trend_pipeline.predict(x_val)
        oof_pred[val_idx] = fold_pred

        fold_metrics = calculate_metrics(y_val, fold_pred)
        print(
            f"[趋势 OOF] fold {fold_no}/{cv_folds} "
            f"RMSE={fold_metrics['RMSE']:.4f}, R2={fold_metrics['R2']:.4f}"
        )

    return oof_pred


def fit_final_trend_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    trend_model: str,
    sample_weight: np.ndarray | None,
) -> Pipeline:
    """在完整训练集上拟合最终线性趋势模型。"""
    trend_pipeline = build_linear_trend_pipeline(x_train, trend_model)
    if sample_weight is None:
        trend_pipeline.fit(x_train, y_train)
    else:
        trend_pipeline.fit(x_train, y_train, model__sample_weight=sample_weight)
    return trend_pipeline


def describe_trend_model(model: Pipeline) -> str:
    """
    返回趋势模型的简要说明，便于在训练结果中直接看到当前使用的是哪种趋势层。
    """
    trend_estimator = model.named_steps["model"]
    if isinstance(trend_estimator, RidgeCV):
        return f"RidgeCV(alpha={trend_estimator.alpha_:.6g})"
    if isinstance(trend_estimator, LinearRegression):
        return "LinearRegression"
    return trend_estimator.__class__.__name__


def tune_residual_model(
    pipeline: Pipeline,
    x_train_aug: pd.DataFrame,
    residual_target: pd.Series,
    groups_train: pd.Series | None,
    sample_weight: np.ndarray | None,
    cv_folds: int,
    n_iter: int,
    random_state: int,
    device: str,
    validation_mode: str,
) -> tuple[dict[str, Any], float]:
    """对 XGBoost 残差模型做随机搜索调参。"""
    param_distributions: dict[str, Any] = {
        "model__n_estimators": [300, 500, 700, 900],
        "model__learning_rate": [0.02, 0.03, 0.05, 0.08],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__min_child_weight": [1, 2, 3, 5, 8],
        "model__subsample": [0.7, 0.8, 0.85, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.85, 0.9, 1.0],
        "model__gamma": [0.0, 0.1, 0.2, 0.5],
        "model__reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    if validation_mode == "group":
        if groups_train is None:
            raise ValueError("分组验证模式下，groups_train 不能为空。")
        validate_group_folds(groups_train, cv_folds, "残差模型 GroupKFold")
        cv = GroupKFold(n_splits=cv_folds)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search_n_jobs = 1 if device == "gpu" else -1

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=search_n_jobs,
        refit=False,
    )
    fit_params: dict[str, Any] = {}
    if sample_weight is not None:
        fit_params["model__sample_weight"] = sample_weight

    if validation_mode == "group":
        search.fit(x_train_aug, residual_target, groups=groups_train, **fit_params)
    else:
        search.fit(x_train_aug, residual_target, **fit_params)
    return search.best_params_, -float(search.best_score_)


def tune_catboost_residual_model(
    x_train_aug: pd.DataFrame,
    residual_target: pd.Series,
    groups_train: pd.Series | None,
    sample_weight: np.ndarray | None,
    cv_folds: int,
    n_iter: int,
    early_stopping_rounds: int,
    random_state: int,
    device: str,
    gpu_id: int,
    validation_mode: str,
) -> tuple[dict[str, Any], float]:
    """
    对 CatBoost 残差模型做随机参数采样调参。

    说明：
    1. 这里不用 RandomizedSearchCV，是因为 CatBoost 原生类别特征更适合手工控制训练流程；
    2. 每个候选参数都会做 K 折 / GroupKFold 评估；
    3. 每个折内都启用 early stopping，降低无效迭代开销。
    """
    x_train_cb, categorical_cols = prepare_catboost_features(x_train_aug)
    base_params = default_catboost_residual_params(random_state, device, gpu_id)
    param_space: dict[str, list[Any]] = {
        "iterations": [600, 900, 1200, 1500],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "depth": [4, 5, 6, 8, 10],
        "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0, 10.0],
        "random_strength": [0.0, 0.5, 1.0, 2.0, 5.0],
        "bagging_temperature": [0.0, 0.5, 1.0, 2.0],
    }
    candidates = list(
        ParameterSampler(param_space, n_iter=n_iter, random_state=random_state)
    )

    split_indices = build_cv_split_indices(
        x_data=x_train_cb,
        y_data=residual_target,
        groups_data=groups_train,
        cv_folds=cv_folds,
        random_state=random_state,
        validation_mode=validation_mode,
        stage_name="CatBoost 残差模型交叉验证",
    )

    best_params: dict[str, Any] | None = None
    best_cv_rmse = float("inf")

    for candidate_no, sampled_params in enumerate(candidates, start=1):
        fold_rmses: list[float] = []
        params = {**base_params, **sampled_params}

        for fold_no, (tr_idx, val_idx) in enumerate(split_indices, start=1):
            x_tr = x_train_cb.iloc[tr_idx]
            y_tr = residual_target.iloc[tr_idx]
            x_val = x_train_cb.iloc[val_idx]
            y_val = residual_target.iloc[val_idx]
            fold_weight = sample_weight[tr_idx] if sample_weight is not None else None

            train_pool = Pool(
                data=x_tr,
                label=y_tr,
                cat_features=categorical_cols,
                weight=fold_weight,
            )
            valid_pool = Pool(data=x_val, label=y_val, cat_features=categorical_cols)

            model = CatBoostRegressor(**params)
            model.fit(
                train_pool,
                eval_set=valid_pool,
                use_best_model=True,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )

            fold_pred = model.predict(valid_pool)
            fold_rmse = calculate_metrics(y_val, fold_pred)["RMSE"]
            fold_rmses.append(fold_rmse)
            print(
                f"[CatBoost 调参] 候选 {candidate_no}/{len(candidates)} - "
                f"fold {fold_no}/{cv_folds} RMSE={fold_rmse:.4f}"
            )

        mean_rmse = float(np.mean(fold_rmses))
        print(
            f"[CatBoost 调参] 候选 {candidate_no}/{len(candidates)} "
            f"平均 RMSE={mean_rmse:.4f}"
        )
        if mean_rmse < best_cv_rmse:
            best_cv_rmse = mean_rmse
            best_params = sampled_params

    if best_params is None:
        raise RuntimeError("CatBoost 调参失败：未找到有效参数。")

    return best_params, best_cv_rmse


def fit_final_residual_model(
    x_train_aug: pd.DataFrame,
    residual_target: pd.Series,
    sample_weight: np.ndarray | None,
    random_state: int,
    device: str,
    gpu_id: int,
    best_params: dict[str, Any] | None,
) -> Pipeline:
    """使用完整训练集上的残差重新训练最终 XGBoost 模型。"""
    residual_model = build_residual_pipeline(x_train_aug, random_state, device, gpu_id)
    if best_params:
        residual_model.set_params(**best_params)
    if sample_weight is None:
        residual_model.fit(x_train_aug, residual_target)
    else:
        residual_model.fit(
            x_train_aug, residual_target, model__sample_weight=sample_weight
        )
    return residual_model


def fit_final_catboost_residual_model(
    x_train_aug: pd.DataFrame,
    residual_target: pd.Series,
    groups_train: pd.Series | None,
    sample_weight: np.ndarray | None,
    random_state: int,
    device: str,
    gpu_id: int,
    best_params: dict[str, Any] | None,
    early_stopping_rounds: int,
    validation_mode: str,
) -> tuple[CatBoostRegressor, list[str], list[str]]:
    """
    在完整训练集上训练最终 CatBoost 残差模型。

    为了兼顾效率和泛化，这里会在训练集内部再切一小部分验证集用于 early stopping。
    如果分组模式可用，则优先使用按组切分，避免同城样本同时进入内部训练与验证。
    """
    x_train_cb, categorical_cols = prepare_catboost_features(x_train_aug)
    feature_cols = x_train_cb.columns.tolist()
    final_params = default_catboost_residual_params(random_state, device, gpu_id)
    if best_params:
        final_params.update(best_params)

    if (
        validation_mode == "group"
        and groups_train is not None
        and groups_train.nunique() >= 2
    ):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.15,
            random_state=random_state,
        )
        fit_idx, val_idx = next(
            splitter.split(x_train_cb, residual_target, groups_train)
        )
    else:
        fit_idx, val_idx = train_test_split(
            np.arange(len(x_train_cb)),
            test_size=0.15,
            random_state=random_state,
        )

    x_fit = x_train_cb.iloc[fit_idx]
    y_fit = residual_target.iloc[fit_idx]
    x_val = x_train_cb.iloc[val_idx]
    y_val = residual_target.iloc[val_idx]
    fit_weight = sample_weight[fit_idx] if sample_weight is not None else None

    train_pool = Pool(
        data=x_fit,
        label=y_fit,
        cat_features=categorical_cols,
        weight=fit_weight,
    )
    valid_pool = Pool(data=x_val, label=y_val, cat_features=categorical_cols)

    model = CatBoostRegressor(**final_params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )
    return model, categorical_cols, feature_cols


def predict_catboost_residual_model(
    model: CatBoostRegressor,
    x_aug: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> np.ndarray:
    """使用训练好的 CatBoost 残差模型对新样本做预测。"""
    x_cb = x_aug.reindex(columns=feature_cols).copy()
    x_cb, _ = prepare_catboost_features(x_cb)
    pred_pool = Pool(data=x_cb, cat_features=categorical_cols)
    return np.asarray(model.predict(pred_pool), dtype=float)


def generate_oof_xgboost_residual_predictions(
    x_train_aug: pd.DataFrame,
    residual_target: pd.Series,
    groups_train: pd.Series | None,
    sample_weight: np.ndarray | None,
    cv_folds: int,
    random_state: int,
    device: str,
    gpu_id: int,
    best_params: dict[str, Any] | None,
    validation_mode: str,
) -> np.ndarray:
    """
    生成 XGBoost 残差模型的 OOF 预测。

    这些 OOF 残差预测不会直接用于最终上线预测，
    而是专门用于“给双模型集成搜索权重”。
    这样定权时看到的是更接近真实泛化场景的预测，而不是训练集拟合值。
    """
    split_indices = build_cv_split_indices(
        x_data=x_train_aug,
        y_data=residual_target,
        groups_data=groups_train,
        cv_folds=cv_folds,
        random_state=random_state,
        validation_mode=validation_mode,
        stage_name="XGBoost 残差 OOF",
    )
    oof_pred = np.zeros(len(x_train_aug), dtype=float)

    for fold_no, (tr_idx, val_idx) in enumerate(split_indices, start=1):
        x_tr = x_train_aug.iloc[tr_idx]
        y_tr = residual_target.iloc[tr_idx]
        x_val = x_train_aug.iloc[val_idx]
        fold_weight = sample_weight[tr_idx] if sample_weight is not None else None

        model = build_residual_pipeline(x_tr, random_state, device, gpu_id)
        if best_params:
            model.set_params(**best_params)

        if fold_weight is None:
            model.fit(x_tr, y_tr)
        else:
            model.fit(x_tr, y_tr, model__sample_weight=fold_weight)

        fold_pred = model.predict(x_val)
        oof_pred[val_idx] = fold_pred
        fold_rmse = calculate_metrics(residual_target.iloc[val_idx], fold_pred)["RMSE"]
        print(
            f"[XGBoost OOF] fold {fold_no}/{cv_folds} residual RMSE={fold_rmse:.4f}"
        )

    return oof_pred


def generate_oof_catboost_residual_predictions(
    x_train_aug: pd.DataFrame,
    residual_target: pd.Series,
    groups_train: pd.Series | None,
    sample_weight: np.ndarray | None,
    cv_folds: int,
    random_state: int,
    device: str,
    gpu_id: int,
    best_params: dict[str, Any] | None,
    validation_mode: str,
) -> np.ndarray:
    """
    生成 CatBoost 残差模型的 OOF 预测。

    这里使用固定参数直接训练每一折，
    不在 OOF 阶段再做 early stopping，避免把验证折既当“评估”又当“调停止”的轻微偏乐观问题。
    """
    x_train_cb, categorical_cols = prepare_catboost_features(x_train_aug)
    split_indices = build_cv_split_indices(
        x_data=x_train_cb,
        y_data=residual_target,
        groups_data=groups_train,
        cv_folds=cv_folds,
        random_state=random_state,
        validation_mode=validation_mode,
        stage_name="CatBoost 残差 OOF",
    )
    oof_pred = np.zeros(len(x_train_cb), dtype=float)
    params = default_catboost_residual_params(random_state, device, gpu_id)
    if best_params:
        params.update(best_params)

    for fold_no, (tr_idx, val_idx) in enumerate(split_indices, start=1):
        x_tr = x_train_cb.iloc[tr_idx]
        y_tr = residual_target.iloc[tr_idx]
        x_val = x_train_cb.iloc[val_idx]
        fold_weight = sample_weight[tr_idx] if sample_weight is not None else None

        train_pool = Pool(
            data=x_tr,
            label=y_tr,
            cat_features=categorical_cols,
            weight=fold_weight,
        )
        valid_pool = Pool(data=x_val, cat_features=categorical_cols)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, verbose=False)

        fold_pred = np.asarray(model.predict(valid_pool), dtype=float)
        oof_pred[val_idx] = fold_pred
        fold_rmse = calculate_metrics(residual_target.iloc[val_idx], fold_pred)["RMSE"]
        print(
            f"[CatBoost OOF] fold {fold_no}/{cv_folds} residual RMSE={fold_rmse:.4f}"
        )

    return oof_pred


def search_ensemble_weights_by_oof(
    y_true: pd.Series,
    trend_oof_pred: np.ndarray,
    xgb_residual_oof_pred: np.ndarray,
    cat_residual_oof_pred: np.ndarray,
    objective: str,
    high_pm25_threshold: float,
    grid_step: float,
    blended_overall_weight: float,
) -> dict[str, Any]:
    """
    基于训练集 OOF 预测搜索双模型集成权重。

    搜索目标有三种：
    1. overall：让整体 OOF RMSE 最小；
    2. high_pm25：让高污染子集 OOF RMSE 最小。
    3. blended：同时兼顾整体 RMSE 和高污染 RMSE。

    这里直接对权重做网格搜索，而不是只用单模型 RMSE 反推权重，
    好处是能显式利用两个模型之间的互补性。
    """
    if not 0.0 < grid_step <= 1.0:
        raise ValueError("ensemble-weight-grid-step 必须在 (0, 1] 之间。")
    if not 0.0 <= blended_overall_weight <= 1.0:
        raise ValueError("ensemble-blended-overall-weight 必须在 [0, 1] 之间。")

    y_true_array = np.asarray(y_true, dtype=float)
    trend_overall_rmse = calculate_metrics(y_true_array, trend_oof_pred)["RMSE"]
    trend_high_metrics, trend_high_count = calculate_high_pm25_metrics(
        y_true=y_true_array,
        y_pred=trend_oof_pred,
        threshold=high_pm25_threshold,
    )
    trend_high_rmse = trend_high_metrics["RMSE"]
    trend_overall_scale = max(float(trend_overall_rmse), 1e-12)
    trend_high_scale = (
        max(float(trend_high_rmse), 1e-12)
        if np.isfinite(trend_high_rmse)
        else 1.0
    )
    best_result: dict[str, Any] | None = None
    overall_sample_count = len(y_true_array)
    weight_candidates = np.arange(0.0, 1.0 + grid_step / 2.0, grid_step)

    for xgb_weight in weight_candidates:
        cat_weight = 1.0 - float(xgb_weight)
        residual_pred = (
            float(xgb_weight) * xgb_residual_oof_pred
            + cat_weight * cat_residual_oof_pred
        )
        final_pred = trend_oof_pred + residual_pred

        overall_metrics = calculate_metrics(y_true_array, final_pred)
        high_metrics, high_sample_count = calculate_high_pm25_metrics(
            y_true=y_true_array,
            y_pred=final_pred,
            threshold=high_pm25_threshold,
        )
        overall_rmse = float(overall_metrics["RMSE"])
        high_rmse = float(high_metrics["RMSE"])

        if objective == "high_pm25":
            objective_score = high_rmse
            objective_metric = "RMSE"
            objective_sample_count = high_sample_count
        elif objective == "blended":
            # 使用相对趋势基线的归一化 RMSE，避免高污染 RMSE 数值更大而天然主导目标。
            normalized_overall = overall_rmse / trend_overall_scale
            normalized_high = high_rmse / trend_high_scale
            blended_high_weight = 1.0 - float(blended_overall_weight)
            objective_score = (
                float(blended_overall_weight) * normalized_overall
                + blended_high_weight * normalized_high
            )
            objective_metric = "weighted_normalized_rmse"
            objective_sample_count = overall_sample_count
        else:
            objective_score = overall_rmse
            objective_metric = "RMSE"
            objective_sample_count = overall_sample_count

        candidate = {
            "strategy": f"oof_grid_search_{objective}",
            "objective": objective,
            "objective_metric": objective_metric,
            "objective_score": float(objective_score),
            "objective_sample_count": int(objective_sample_count),
            "xgb_weight": float(xgb_weight),
            "catboost_weight": float(cat_weight),
            "grid_step": float(grid_step),
            "overall_rmse": overall_rmse,
            "high_pm25_rmse": high_rmse,
            "overall_sample_count": int(overall_sample_count),
            "high_pm25_sample_count": int(high_sample_count),
            "trend_overall_rmse": float(trend_overall_rmse),
            "trend_high_pm25_rmse": float(trend_high_rmse),
            "trend_high_pm25_sample_count": int(trend_high_count),
        }

        if objective == "blended":
            candidate["blended_overall_weight"] = float(blended_overall_weight)
            candidate["blended_high_weight"] = 1.0 - float(blended_overall_weight)
            candidate["normalized_overall_rmse"] = overall_rmse / trend_overall_scale
            candidate["normalized_high_pm25_rmse"] = high_rmse / trend_high_scale

        if (
            best_result is None
            or candidate["objective_score"] < best_result["objective_score"]
        ):
            best_result = candidate

    if best_result is None:
        raise RuntimeError("双模型集成权重搜索失败：未产生有效候选。")
    return best_result


def resolve_ensemble_weights(
    residual_model: str,
    manual_weight_xgb: float | None,
    ensemble_weight_objective: str,
    ensemble_weight_grid_step: float,
    ensemble_blended_overall_weight: float,
    y_train: pd.Series,
    trend_train_oof_pred: np.ndarray,
    xgb_residual_oof_pred: np.ndarray | None,
    cat_residual_oof_pred: np.ndarray | None,
    high_pm25_threshold: float,
    xgb_cv_rmse: float | None,
    cat_cv_rmse: float | None,
) -> dict[str, Any]:
    """
    解析双模型集成的权重配置。

    权重策略：
    1. 如果用户显式给出 XGBoost 权重，则直接使用；
    2. 否则，优先使用 OOF 预测直接搜索最优权重；
    3. 如果 OOF 预测不可用，再回退到基于单模型 CV RMSE 的倒数加权；
    4. 如果仍不可用，则退回等权平均。
    """
    if residual_model != "ensemble":
        return {
            "strategy": "single_model",
            "objective": "single_model",
            "xgb_weight": 1.0 if residual_model == "xgboost" else 0.0,
            "catboost_weight": 1.0 if residual_model == "catboost" else 0.0,
        }

    if manual_weight_xgb is not None:
        if not 0.0 <= manual_weight_xgb <= 1.0:
            raise ValueError("ensemble-weight-xgb 必须在 [0, 1] 之间。")
        xgb_weight = float(manual_weight_xgb)
        return {
            "strategy": "manual",
            "objective": "manual",
            "xgb_weight": xgb_weight,
            "catboost_weight": 1.0 - xgb_weight,
        }

    if xgb_residual_oof_pred is not None and cat_residual_oof_pred is not None:
        return search_ensemble_weights_by_oof(
            y_true=y_train,
            trend_oof_pred=trend_train_oof_pred,
            xgb_residual_oof_pred=xgb_residual_oof_pred,
            cat_residual_oof_pred=cat_residual_oof_pred,
            objective=ensemble_weight_objective,
            high_pm25_threshold=high_pm25_threshold,
            grid_step=ensemble_weight_grid_step,
            blended_overall_weight=ensemble_blended_overall_weight,
        )

    if (
        xgb_cv_rmse is not None
        and cat_cv_rmse is not None
        and np.isfinite(xgb_cv_rmse)
        and np.isfinite(cat_cv_rmse)
        and xgb_cv_rmse > 0.0
        and cat_cv_rmse > 0.0
    ):
        inv_xgb = 1.0 / xgb_cv_rmse
        inv_cat = 1.0 / cat_cv_rmse
        total = inv_xgb + inv_cat
        return {
            "strategy": "inverse_cv_rmse",
            "objective": "overall",
            "xgb_weight": inv_xgb / total,
            "catboost_weight": inv_cat / total,
        }

    return {
        "strategy": "equal_fallback",
        "objective": "overall",
        "xgb_weight": 0.5,
        "catboost_weight": 0.5,
    }


def print_top_linear_coefficients(model: Pipeline, top_n: int = 15) -> None:
    """打印线性趋势模型中绝对值最大的若干个系数。"""
    preprocess = model.named_steps["preprocess"]
    linear_model = model.named_steps["model"]

    try:
        feature_names = preprocess.get_feature_names_out()
        coefficients = linear_model.coef_
    except Exception:
        print("\n[提示] 当前线性趋势模型无法提取系数。")
        return

    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).head(top_n)

    print(f"\n线性趋势模型 Top {top_n} 系数：")
    print(coef_df[["feature", "coefficient"]].to_string(index=False))


def print_top_residual_importances(model: Pipeline, top_n: int = 20) -> None:
    """打印残差模型的特征重要性。"""
    preprocess = model.named_steps["preprocess"]
    regressor = model.named_steps["model"]

    try:
        feature_names = preprocess.get_feature_names_out()
        importances = regressor.feature_importances_
    except Exception:
        print("\n[提示] 当前残差模型无法提取特征重要性。")
        return

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)

    print(f"\n残差模型 Top {top_n} 特征重要性：")
    print(importance_df.to_string(index=False))


def print_top_catboost_residual_importances(
    model: CatBoostRegressor, top_n: int = 20
) -> None:
    """打印 CatBoost 残差模型的特征重要性。"""
    try:
        importances = model.get_feature_importance(type="FeatureImportance")
        feature_names = model.feature_names_
    except Exception:
        print("\n[提示] 当前 CatBoost 残差模型无法提取特征重要性。")
        return

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    importance_df = importance_df.head(top_n)

    print(f"\nCatBoost 残差模型 Top {top_n} 特征重要性：")
    print(importance_df.to_string(index=False))


def save_artifact(
    trend_model: Pipeline,
    residual_model_mode: str,
    xgboost_residual_model: Pipeline | None,
    catboost_residual_model: CatBoostRegressor | None,
    catboost_feature_cols: list[str],
    catboost_categorical_cols: list[str],
    ensemble_config: dict[str, Any],
    model_out: Path,
    target_col: str,
    dropped_cols: list[str],
    validation_mode: str,
    group_col: str,
    log_feature_cols: list[str],
    sample_weight_config: dict[str, float] | None,
    high_pm25_eval_config: dict[str, float] | None,
) -> None:
    """保存最终模型对象，便于后续直接加载使用。"""
    split_strategy = (
        "GroupShuffleSplit + GroupKFold"
        if validation_mode == "group"
        else "train_test_split + KFold"
    )
    artifact = {
        "trend_model": trend_model,
        "residual_model": (
            xgboost_residual_model
            if residual_model_mode == "xgboost"
            else catboost_residual_model
            if residual_model_mode == "catboost"
            else None
        ),
        "residual_model_mode": residual_model_mode,
        "residual_models": {
            "xgboost": xgboost_residual_model,
            "catboost": catboost_residual_model,
        },
        "catboost_feature_cols": catboost_feature_cols,
        "catboost_categorical_cols": catboost_categorical_cols,
        "ensemble_config": ensemble_config,
        "target_col": target_col,
        "trend_feature_name": TREND_FEATURE_NAME,
        "dropped_identifier_cols": dropped_cols,
        "validation_mode": validation_mode,
        "group_col": group_col,
        "split_strategy": split_strategy,
        "trend_model_name": describe_trend_model(trend_model),
        "log_feature_cols": log_feature_cols,
        "sample_weight_config": sample_weight_config,
        "high_pm25_eval_config": high_pm25_eval_config,
        "model_type": f"linear_trend_plus_{residual_model_mode}_residual",
    }
    joblib.dump(artifact, model_out)
    print(f"\n模型已保存到：{model_out}")


def main() -> None:
    """主流程：读取数据 -> 线性趋势 -> 残差模型 -> 评估 -> 保存。"""
    args = parse_args()

    data_path = Path(args.data)
    model_out, plot_out = resolve_output_paths(
        residual_model=args.residual_model,
        model_out_arg=args.model_out,
        plot_out_arg=args.plot_out,
    )

    # 1) 读取并清洗数据
    x, y = load_data(data_path, args.target)

    dropped_cols: list[str] = []
    if not args.keep_id_cols:
        x, dropped_cols = remove_identifier_columns(x)
        if dropped_cols:
            print(f"[信息] 已自动剔除疑似 ID 列：{dropped_cols}")

    added_log_cols: list[str] = []
    skipped_log_cols: list[str] = []
    if args.no_log_features:
        print("[信息] 已关闭 log1p 特征扩展。")
    else:
        x, added_log_cols, skipped_log_cols = add_log_transformed_features(
            x=x,
            candidate_cols=parse_feature_list(args.log_feature_cols),
        )
        if added_log_cols:
            print(f"[信息] 已新增 log1p 特征：{added_log_cols}")
        if skipped_log_cols:
            print(f"[信息] 跳过的 log1p 特征：{skipped_log_cols}")

    groups: pd.Series | None = None
    groups_train: pd.Series | None = None
    groups_test: pd.Series | None = None

    # 2) 按配置选择验证方式
    if args.validation_mode == "group":
        groups = resolve_groups(x, args.group_col)
        x_train, x_test, y_train, y_test, groups_train, groups_test = (
            split_train_test_by_group(
                x=x,
                y=y,
                groups=groups,
                test_size=args.test_size,
                random_state=args.random_state,
            )
        )
    else:
        x_train, x_test, y_train, y_test = split_train_test_random(
            x=x,
            y=y,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    print(f"[信息] 当前训练设备：{args.device.upper()}")
    if args.device == "gpu":
        print(f"[信息] 使用 GPU 编号：{args.gpu_id}")
    print(f"[信息] 当前趋势模型：{args.trend_model}")
    print(f"[信息] 当前残差模型：{args.residual_model}")
    print(f"[信息] 当前验证方式：{args.validation_mode}")
    if args.validation_mode == "group":
        print(f"[信息] 当前分组列：{args.group_col}")
        print(
            f"[信息] 分组统计：总组数={groups.nunique()}，"
            f"训练组数={groups_train.nunique()}，测试组数={groups_test.nunique()}"
        )

    sample_weight_config = build_sample_weight_config(
        y_train=y_train,
        mode=args.sample_weight_mode,
        quantile=args.high_pm25_quantile,
        high_weight=args.high_pm25_weight,
    )
    high_pm25_eval_config = build_high_pm25_eval_config(
        y_train=y_train,
        quantile=args.eval_high_pm25_quantile,
    )
    trend_sample_weight = compute_sample_weights(y_train, sample_weight_config)
    residual_sample_weight = compute_sample_weights(y_train, sample_weight_config)
    print(
        f"[信息] 当前样本加权：{describe_sample_weighting(sample_weight_config, y_train)}"
    )
    print(
        "[信息] 高污染子集评估阈值："
        f"quantile={high_pm25_eval_config['quantile']:.2f}, "
        f"threshold={high_pm25_eval_config['threshold']:.4f}"
    )

    # 3) 使用按组 OOF 方式获得训练集的线性趋势预测，避免第二阶段信息泄漏
    print(f"[信息] 开始生成线性趋势 OOF 预测：trend_cv_folds={args.trend_cv_folds}")
    trend_train_oof_pred = generate_oof_trend_predictions(
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        cv_folds=args.trend_cv_folds,
        random_state=args.random_state,
        validation_mode=args.validation_mode,
        trend_model=args.trend_model,
        weight_config=sample_weight_config,
    )

    # 4) 在完整训练集上拟合最终线性趋势模型，并对训练/测试集做趋势预测
    trend_model = fit_final_trend_model(
        x_train=x_train,
        y_train=y_train,
        trend_model=args.trend_model,
        sample_weight=trend_sample_weight,
    )
    trend_test_pred = trend_model.predict(x_test)

    # 5) 训练残差模型时，使用 OOF 趋势预测构造更真实的残差标签
    residual_train_oof = y_train - trend_train_oof_pred
    x_train_aug_for_tune = make_augmented_features(x_train, trend_train_oof_pred)

    # 6) 最终训练残差模型时，仍然使用 OOF 残差作为监督信号。
    #
    # 原因：
    # 如果直接使用“完整线性趋势模型在训练集上的拟合残差”，
    # 对于表达能力较强的线性模型，训练残差可能被压得非常小，
    # 第二阶段模型就会学不到真正有价值的修正规律。
    #
    # 因此这里保留 OOF 残差作为最终训练目标，让残差模型学习更真实、更可泛化的偏差。
    x_train_aug_for_fit = x_train_aug_for_tune.copy()
    x_test_aug = make_augmented_features(x_test, trend_test_pred)

    xgb_best_params: dict[str, Any] | None = None
    xgb_best_cv_rmse: float | None = None
    cat_best_params: dict[str, Any] | None = None
    cat_best_cv_rmse: float | None = None
    xgb_residual_model: Pipeline | None = None
    catboost_residual_model: CatBoostRegressor | None = None
    xgb_residual_train_oof_pred: np.ndarray | None = None
    cat_residual_train_oof_pred: np.ndarray | None = None
    catboost_feature_cols: list[str] = []
    catboost_categorical_cols: list[str] = []

    if args.residual_model in ("xgboost", "ensemble"):
        try:
            if args.no_tune:
                print("[信息] 已关闭 XGBoost 残差调参，将使用默认参数训练。")
            else:
                print(
                    f"[信息] 开始 XGBoost 残差调参：n_iter={args.n_iter}, "
                    f"residual_cv_folds={args.residual_cv_folds}"
                )
                residual_pipeline_for_tune = build_residual_pipeline(
                    x_train_aug_for_tune,
                    args.random_state,
                    args.device,
                    args.gpu_id,
                )
                xgb_best_params, xgb_best_cv_rmse = tune_residual_model(
                    pipeline=residual_pipeline_for_tune,
                    x_train_aug=x_train_aug_for_tune,
                    residual_target=residual_train_oof,
                    groups_train=groups_train,
                    sample_weight=residual_sample_weight,
                    cv_folds=args.residual_cv_folds,
                    n_iter=args.n_iter,
                    random_state=args.random_state,
                    device=args.device,
                    validation_mode=args.validation_mode,
                )

            xgb_residual_model = fit_final_residual_model(
                x_train_aug=x_train_aug_for_fit,
                residual_target=residual_train_oof,
                sample_weight=residual_sample_weight,
                random_state=args.random_state,
                device=args.device,
                gpu_id=args.gpu_id,
                best_params=xgb_best_params,
            )
        except XGBoostError as exc:
            if args.device == "gpu":
                raise SystemExit(
                    "XGBoost GPU 训练失败。请确认 CUDA 驱动/显卡可用，或改用 --device cpu。\n"
                    f"原始错误：{exc}"
                ) from exc
            raise

    if args.residual_model in ("catboost", "ensemble"):
        try:
            if args.no_tune:
                print("[信息] 已关闭 CatBoost 残差调参，将使用默认参数训练。")
            else:
                print(
                    f"[信息] 开始 CatBoost 残差调参：n_iter={args.n_iter}, "
                    f"residual_cv_folds={args.residual_cv_folds}"
                )
                cat_best_params, cat_best_cv_rmse = tune_catboost_residual_model(
                    x_train_aug=x_train_aug_for_tune,
                    residual_target=residual_train_oof,
                    groups_train=groups_train,
                    sample_weight=residual_sample_weight,
                    cv_folds=args.residual_cv_folds,
                    n_iter=args.n_iter,
                    early_stopping_rounds=args.catboost_early_stopping_rounds,
                    random_state=args.random_state,
                    device=args.device,
                    gpu_id=args.gpu_id,
                    validation_mode=args.validation_mode,
                )

            (
                catboost_residual_model,
                catboost_categorical_cols,
                catboost_feature_cols,
            ) = fit_final_catboost_residual_model(
                x_train_aug=x_train_aug_for_fit,
                residual_target=residual_train_oof,
                groups_train=groups_train,
                sample_weight=residual_sample_weight,
                random_state=args.random_state,
                device=args.device,
                gpu_id=args.gpu_id,
                best_params=cat_best_params,
                early_stopping_rounds=args.catboost_early_stopping_rounds,
                validation_mode=args.validation_mode,
            )
        except CatBoostError as exc:
            if args.device == "gpu":
                raise SystemExit(
                    "CatBoost GPU 训练失败。请确认 CUDA 驱动/显卡可用，或改用 --device cpu。\n"
                    f"原始错误：{exc}"
                ) from exc
            raise

    if args.residual_model == "ensemble" and args.ensemble_weight_xgb is None:
        print(
            "[信息] 开始基于 OOF 预测搜索双模型集成权重："
            f"objective={args.ensemble_weight_objective}, "
            f"grid_step={args.ensemble_weight_grid_step}, "
            f"blended_overall_weight={args.ensemble_blended_overall_weight:.2f}"
        )
        if xgb_residual_model is None or catboost_residual_model is None:
            raise RuntimeError("自动搜索集成权重前，必须先成功训练 XGBoost 和 CatBoost 残差模型。")

        try:
            xgb_residual_train_oof_pred = generate_oof_xgboost_residual_predictions(
                x_train_aug=x_train_aug_for_tune,
                residual_target=residual_train_oof,
                groups_train=groups_train,
                sample_weight=residual_sample_weight,
                cv_folds=args.residual_cv_folds,
                random_state=args.random_state,
                device=args.device,
                gpu_id=args.gpu_id,
                best_params=xgb_best_params,
                validation_mode=args.validation_mode,
            )
        except XGBoostError as exc:
            if args.device == "gpu":
                raise SystemExit(
                    "XGBoost GPU 训练失败。请确认 CUDA 驱动/显卡可用，或改用 --device cpu。\n"
                    f"原始错误：{exc}"
                ) from exc
            raise

        try:
            cat_residual_train_oof_pred = generate_oof_catboost_residual_predictions(
                x_train_aug=x_train_aug_for_tune,
                residual_target=residual_train_oof,
                groups_train=groups_train,
                sample_weight=residual_sample_weight,
                cv_folds=args.residual_cv_folds,
                random_state=args.random_state,
                device=args.device,
                gpu_id=args.gpu_id,
                best_params=cat_best_params,
                validation_mode=args.validation_mode,
            )
        except CatBoostError as exc:
            if args.device == "gpu":
                raise SystemExit(
                    "CatBoost GPU 训练失败。请确认 CUDA 驱动/显卡可用，或改用 --device cpu。\n"
                    f"原始错误：{exc}"
                ) from exc
            raise

    ensemble_config = resolve_ensemble_weights(
        residual_model=args.residual_model,
        manual_weight_xgb=args.ensemble_weight_xgb,
        ensemble_weight_objective=args.ensemble_weight_objective,
        ensemble_weight_grid_step=args.ensemble_weight_grid_step,
        ensemble_blended_overall_weight=args.ensemble_blended_overall_weight,
        y_train=y_train,
        trend_train_oof_pred=trend_train_oof_pred,
        xgb_residual_oof_pred=xgb_residual_train_oof_pred,
        cat_residual_oof_pred=cat_residual_train_oof_pred,
        high_pm25_threshold=high_pm25_eval_config["threshold"],
        xgb_cv_rmse=xgb_best_cv_rmse,
        cat_cv_rmse=cat_best_cv_rmse,
    )

    # 8) 评估：先看线性趋势单独表现，再看单模型/双模型集成后的最终表现
    trend_metrics = calculate_metrics(y_test, trend_test_pred)
    xgb_residual_test_pred: np.ndarray | None = None
    cat_residual_test_pred: np.ndarray | None = None
    xgb_final_test_pred: np.ndarray | None = None
    cat_final_test_pred: np.ndarray | None = None
    xgb_hybrid_metrics: dict[str, float] | None = None
    cat_hybrid_metrics: dict[str, float] | None = None

    if xgb_residual_model is not None:
        xgb_residual_test_pred = xgb_residual_model.predict(x_test_aug)
        xgb_final_test_pred = trend_test_pred + xgb_residual_test_pred
        xgb_hybrid_metrics = calculate_metrics(y_test, xgb_final_test_pred)

    if catboost_residual_model is not None:
        cat_residual_test_pred = predict_catboost_residual_model(
            model=catboost_residual_model,
            x_aug=x_test_aug,
            feature_cols=catboost_feature_cols,
            categorical_cols=catboost_categorical_cols,
        )
        cat_final_test_pred = trend_test_pred + cat_residual_test_pred
        cat_hybrid_metrics = calculate_metrics(y_test, cat_final_test_pred)

    if args.residual_model == "xgboost":
        residual_test_pred = xgb_residual_test_pred
        final_test_pred = xgb_final_test_pred
        hybrid_metrics = xgb_hybrid_metrics
        final_model_name = "Trend + XGBoost Residual"
    elif args.residual_model == "catboost":
        residual_test_pred = cat_residual_test_pred
        final_test_pred = cat_final_test_pred
        hybrid_metrics = cat_hybrid_metrics
        final_model_name = "Trend + CatBoost Residual"
    else:
        if xgb_residual_test_pred is None or cat_residual_test_pred is None:
            raise RuntimeError("双模型集成需要同时得到 XGBoost 与 CatBoost 残差预测。")
        residual_test_pred = (
            ensemble_config["xgb_weight"] * xgb_residual_test_pred
            + ensemble_config["catboost_weight"] * cat_residual_test_pred
        )
        final_test_pred = trend_test_pred + residual_test_pred
        hybrid_metrics = calculate_metrics(y_test, final_test_pred)
        final_model_name = "Trend + XGBoost/CatBoost Residual Ensemble"

    if residual_test_pred is None or final_test_pred is None or hybrid_metrics is None:
        raise RuntimeError("最终预测未正确生成，请检查残差模型配置。")

    trend_high_metrics, high_sample_count = calculate_high_pm25_metrics(
        y_true=y_test,
        y_pred=trend_test_pred,
        threshold=high_pm25_eval_config["threshold"],
    )
    hybrid_high_metrics, _ = calculate_high_pm25_metrics(
        y_true=y_test,
        y_pred=final_test_pred,
        threshold=high_pm25_eval_config["threshold"],
    )

    rmse_gain = trend_metrics["RMSE"] - hybrid_metrics["RMSE"]
    mae_gain = trend_metrics["MAE"] - hybrid_metrics["MAE"]
    r2_gain = hybrid_metrics["R2"] - trend_metrics["R2"]

    result_title = {
        "xgboost": "PM2.5 线性趋势 + XGBoost 残差结果",
        "catboost": "PM2.5 线性趋势 + CatBoost 残差结果",
        "ensemble": "PM2.5 线性趋势 + 双残差模型集成结果",
    }[args.residual_model]

    print(f"\n=== {result_title} ===")
    print(f"数据文件                    : {data_path}")
    print(f"样本数（总/训练/测试）       : {len(x)} / {len(x_train)} / {len(x_test)}")
    print(f"趋势模型                    : {describe_trend_model(trend_model)}")
    print(f"残差模型模式                : {args.residual_model}")
    print(f"验证方式                    : {args.validation_mode}")
    if args.validation_mode == "group":
        print(f"分组列                      : {args.group_col}")
    print(f"目标列                      : {args.target}")
    if xgb_best_cv_rmse is not None:
        print(f"XGBoost 残差 CV 最优 RMSE   : {xgb_best_cv_rmse:.4f}")
    if cat_best_cv_rmse is not None:
        print(f"CatBoost 残差 CV 最优 RMSE  : {cat_best_cv_rmse:.4f}")
    if args.residual_model == "ensemble":
        print(
            "集成权重策略                : "
            f"{ensemble_config['strategy']} "
            f"(xgb={ensemble_config['xgb_weight']:.4f}, "
            f"cat={ensemble_config['catboost_weight']:.4f})"
        )
        print(f"集成权重目标                : {ensemble_config.get('objective', 'overall')}")
        if "objective_score" in ensemble_config:
            print(
                "集成目标分数                : "
                f"{ensemble_config['objective_metric']}={ensemble_config['objective_score']:.4f} "
                f"(samples={ensemble_config['objective_sample_count']})"
            )
        if ensemble_config.get("objective") == "blended":
            print(
                "联合目标权重                : "
                f"overall={ensemble_config['blended_overall_weight']:.2f}, "
                f"high_pm25={ensemble_config['blended_high_weight']:.2f}"
            )
            print(
                "OOF 融合候选指标            : "
                f"overall_RMSE={ensemble_config['overall_rmse']:.4f}, "
                f"high_pm25_RMSE={ensemble_config['high_pm25_rmse']:.4f}"
            )
    print(f"趋势模型 测试集 RMSE        : {trend_metrics['RMSE']:.4f}")
    print(f"趋势模型 测试集 MAE         : {trend_metrics['MAE']:.4f}")
    print(f"趋势模型 测试集 R^2         : {trend_metrics['R2']:.4f}")
    if xgb_hybrid_metrics is not None and args.residual_model == "ensemble":
        print(f"XGBoost 融合 测试集 RMSE    : {xgb_hybrid_metrics['RMSE']:.4f}")
        print(f"XGBoost 融合 测试集 MAE     : {xgb_hybrid_metrics['MAE']:.4f}")
        print(f"XGBoost 融合 测试集 R^2     : {xgb_hybrid_metrics['R2']:.4f}")
    if cat_hybrid_metrics is not None and args.residual_model == "ensemble":
        print(f"CatBoost 融合 测试集 RMSE   : {cat_hybrid_metrics['RMSE']:.4f}")
        print(f"CatBoost 融合 测试集 MAE    : {cat_hybrid_metrics['MAE']:.4f}")
        print(f"CatBoost 融合 测试集 R^2    : {cat_hybrid_metrics['R2']:.4f}")
    print(f"融合模型 测试集 RMSE        : {hybrid_metrics['RMSE']:.4f}")
    print(f"融合模型 测试集 MAE         : {hybrid_metrics['MAE']:.4f}")
    print(f"融合模型 测试集 R^2         : {hybrid_metrics['R2']:.4f}")
    print(f"RMSE 改善                   : {rmse_gain:.4f}")
    print(f"MAE 改善                    : {mae_gain:.4f}")
    print(f"R^2 提升                    : {r2_gain:.4f}")
    if args.residual_model in ("xgboost", "ensemble"):
        print(
            "XGBoost 最优参数            : "
            f"{xgb_best_params if xgb_best_params else '默认参数（未调参）'}"
        )
    if args.residual_model in ("catboost", "ensemble"):
        print(
            "CatBoost 最优参数           : "
            f"{cat_best_params if cat_best_params else '默认参数（未调参）'}"
        )

    if not args.no_high_pm25_report:
        print(
            f"高污染子集阈值              : {high_pm25_eval_config['threshold']:.4f} "
            f"(训练集 {high_pm25_eval_config['quantile']:.2f} 分位)"
        )
        print(f"高污染测试样本数            : {high_sample_count}")
        print(f"趋势模型 高污染 RMSE        : {trend_high_metrics['RMSE']:.4f}")
        print(f"趋势模型 高污染 MAE         : {trend_high_metrics['MAE']:.4f}")
        print(f"趋势模型 高污染 R^2         : {trend_high_metrics['R2']:.4f}")
        print(f"融合模型 高污染 RMSE        : {hybrid_high_metrics['RMSE']:.4f}")
        print(f"融合模型 高污染 MAE         : {hybrid_high_metrics['MAE']:.4f}")
        print(f"融合模型 高污染 R^2         : {hybrid_high_metrics['R2']:.4f}")

    sample_dict: dict[str, Any] = {
        "actual": y_test.values[:10],
        "trend_pred": trend_test_pred[:10],
    }
    if args.residual_model == "ensemble":
        sample_dict["xgb_residual_pred"] = xgb_residual_test_pred[:10]
        sample_dict["cat_residual_pred"] = cat_residual_test_pred[:10]
        sample_dict["ensemble_residual_pred"] = residual_test_pred[:10]
    else:
        sample_dict["residual_pred"] = residual_test_pred[:10]
    sample_dict["final_pred"] = final_test_pred[:10]
    sample = pd.DataFrame(sample_dict)
    print("\n测试集前 10 条预测：")
    print(sample.to_string(index=False))

    if not args.no_plot:
        save_diagnostic_plot(
            y_true=y_test,
            y_pred=final_test_pred,
            plot_out=plot_out,
            model_name=final_model_name,
        )
        print(f"\n诊断图已保存到：{plot_out}")

    print_top_linear_coefficients(trend_model, top_n=15)
    if xgb_residual_model is not None:
        print_top_residual_importances(xgb_residual_model, top_n=20)
    if catboost_residual_model is not None:
        print_top_catboost_residual_importances(catboost_residual_model, top_n=20)

    # 9) 按需保存模型
    if not args.no_save:
        save_artifact(
            trend_model=trend_model,
            residual_model_mode=args.residual_model,
            xgboost_residual_model=xgb_residual_model,
            catboost_residual_model=catboost_residual_model,
            catboost_feature_cols=catboost_feature_cols,
            catboost_categorical_cols=catboost_categorical_cols,
            ensemble_config=ensemble_config,
            model_out=model_out,
            target_col=args.target,
            dropped_cols=dropped_cols,
            validation_mode=args.validation_mode,
            group_col=args.group_col,
            log_feature_cols=added_log_cols,
            sample_weight_config=sample_weight_config,
            high_pm25_eval_config=high_pm25_eval_config,
        )


if __name__ == "__main__":
    main()
