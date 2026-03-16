"""
基于“线性趋势 + XGBoost 残差修正”的 PM2.5 回归预测脚本
=========================================================================
建模思路：
1. 先使用线性回归学习数据中的整体线性趋势；
2. 再使用 XGBoost 学习“真实值 - 线性趋势预测值”的残差；
3. 最终预测值 = 线性趋势预测值 + 残差预测值。

这种两阶段建模方式的好处是：
1. 线性模型负责解释稳定、平滑、长期的线性关系；
2. 树模型负责补充非线性、交互项和局部复杂模式；
3. 相比单一模型，更容易解释每一部分在做什么。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# 依赖检查：若缺少必要包，则直接停止并给出安装提示。
try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBRegressor
    from xgboost.core import XGBoostError
except ImportError as exc:
    missing_pkg = str(exc).split("'")[1] if "'" in str(exc) else "未知包"
    raise SystemExit(
        f"缺少依赖包：{missing_pkg}\n"
        "请先安装后再运行，例如：\n"
        "pip install joblib scikit-learn xgboost pandas numpy"
    ) from exc


# 将线性趋势预测值作为一个额外特征传给残差模型。
TREND_FEATURE_NAME = "__linear_trend_pred__"


def parse_args() -> argparse.Namespace:
    """解析命令行参数，便于在不同配置下复用脚本。"""
    parser = argparse.ArgumentParser(
        description="使用线性趋势 + XGBoost 残差修正模型预测 PM2.5"
    )
    parser.add_argument("--data", default="data.csv", help="训练数据 CSV 路径，默认：data.csv")
    parser.add_argument("--target", default="PM2.5", help="目标列名，默认：PM2.5")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例，默认：0.2")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子，默认：42")
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
    parser.add_argument("--n-iter", type=int, default=20, help="随机搜索迭代次数，默认：20")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="XGBoost 残差模型训练设备：cpu 或 gpu，默认：cpu",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU 编号（仅在 --device gpu 时生效）")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="关闭 XGBoost 残差模型调参，仅使用默认参数训练",
    )
    parser.add_argument(
        "--keep-id-cols",
        action="store_true",
        help="保留疑似 ID 列（默认会自动剔除，如 OBJECTID_1）",
    )
    parser.add_argument(
        "--model-out",
        default="pm25_trend_residual_xgboost.joblib",
        help="模型输出文件路径，默认：pm25_trend_residual_xgboost.joblib",
    )
    parser.add_argument("--no-save", action="store_true", help="不保存训练好的模型")
    return parser.parse_args()


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


def build_linear_trend_pipeline(x: pd.DataFrame) -> Pipeline:
    """
    构建线性趋势模型。

    这里使用普通线性回归：
    1. 数值特征：缺失值用中位数填补，再做标准化；
    2. 类别特征：缺失值用众数填补，再做 One-Hot 编码；
    3. 最终由 LinearRegression 学习整体线性趋势。
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

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LinearRegression()),
        ]
    )


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


def calculate_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """统一计算常用回归指标。"""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


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
    cv_folds: int,
    random_state: int,
) -> np.ndarray:
    """
    生成线性趋势模型的 OOF（Out-Of-Fold）预测。

    为什么需要 OOF 预测：
    1. 如果直接用“全训练集拟合后的线性模型”给训练集做预测，
       那么每一条样本都被模型见过，残差会偏乐观；
    2. 用 OOF 预测后，每个样本的趋势预测值都来自“没见过它的模型”，
       这样残差更真实，第二阶段模型更不容易发生信息泄漏。
    """
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(x_train), dtype=float)

    for fold_no, (tr_idx, val_idx) in enumerate(kfold.split(x_train), start=1):
        x_tr = x_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        x_val = x_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        trend_model = build_linear_trend_pipeline(x_tr)
        trend_model.fit(x_tr, y_tr)
        fold_pred = trend_model.predict(x_val)
        oof_pred[val_idx] = fold_pred

        fold_metrics = calculate_metrics(y_val, fold_pred)
        print(
            f"[趋势 OOF] fold {fold_no}/{cv_folds} "
            f"RMSE={fold_metrics['RMSE']:.4f}, R2={fold_metrics['R2']:.4f}"
        )

    return oof_pred


def fit_final_trend_model(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """在完整训练集上拟合最终线性趋势模型。"""
    trend_model = build_linear_trend_pipeline(x_train)
    trend_model.fit(x_train, y_train)
    return trend_model


def tune_residual_model(
    pipeline: Pipeline,
    x_train_aug: pd.DataFrame,
    residual_target: pd.Series,
    cv_folds: int,
    n_iter: int,
    random_state: int,
    device: str,
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
    search.fit(x_train_aug, residual_target)
    return search.best_params_, -float(search.best_score_)


def fit_final_residual_model(
    x_train_aug: pd.DataFrame,
    residual_target: pd.Series,
    random_state: int,
    device: str,
    gpu_id: int,
    best_params: dict[str, Any] | None,
) -> Pipeline:
    """使用完整训练集上的残差重新训练最终 XGBoost 模型。"""
    residual_model = build_residual_pipeline(x_train_aug, random_state, device, gpu_id)
    if best_params:
        residual_model.set_params(**best_params)
    residual_model.fit(x_train_aug, residual_target)
    return residual_model


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


def save_artifact(
    trend_model: Pipeline,
    residual_model: Pipeline,
    model_out: Path,
    target_col: str,
    dropped_cols: list[str],
) -> None:
    """保存最终模型对象，便于后续直接加载使用。"""
    artifact = {
        "trend_model": trend_model,
        "residual_model": residual_model,
        "target_col": target_col,
        "trend_feature_name": TREND_FEATURE_NAME,
        "dropped_identifier_cols": dropped_cols,
        "model_type": "linear_trend_plus_xgboost_residual",
    }
    joblib.dump(artifact, model_out)
    print(f"\n模型已保存到：{model_out}")


def main() -> None:
    """主流程：读取数据 -> 线性趋势 -> 残差模型 -> 评估 -> 保存。"""
    args = parse_args()

    data_path = Path(args.data)
    model_out = Path(args.model_out)

    # 1) 读取并清洗数据
    x, y = load_data(data_path, args.target)

    dropped_cols: list[str] = []
    if not args.keep_id_cols:
        x, dropped_cols = remove_identifier_columns(x)
        if dropped_cols:
            print(f"[信息] 已自动剔除疑似 ID 列：{dropped_cols}")

    # 2) 划分训练集 / 测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"[信息] 当前训练设备：{args.device.upper()}")
    if args.device == "gpu":
        print(f"[信息] 使用 GPU 编号：{args.gpu_id}")

    # 3) 使用 OOF 方式获得训练集的线性趋势预测，避免第二阶段信息泄漏
    print(
        f"[信息] 开始生成线性趋势 OOF 预测：trend_cv_folds={args.trend_cv_folds}"
    )
    trend_train_oof_pred = generate_oof_trend_predictions(
        x_train=x_train,
        y_train=y_train,
        cv_folds=args.trend_cv_folds,
        random_state=args.random_state,
    )

    # 4) 在完整训练集上拟合最终线性趋势模型，并对训练/测试集做趋势预测
    trend_model = fit_final_trend_model(x_train, y_train)
    trend_train_full_pred = trend_model.predict(x_train)
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

    best_params: dict[str, Any] | None = None
    best_cv_rmse: float | None = None

    try:
        if args.no_tune:
            print("[信息] 已关闭残差模型调参，将使用默认参数训练。")
        else:
            print(
                f"[信息] 开始调参：n_iter={args.n_iter}, "
                f"residual_cv_folds={args.residual_cv_folds}"
            )
            residual_pipeline_for_tune = build_residual_pipeline(
                x_train_aug_for_tune,
                args.random_state,
                args.device,
                args.gpu_id,
            )
            best_params, best_cv_rmse = tune_residual_model(
                pipeline=residual_pipeline_for_tune,
                x_train_aug=x_train_aug_for_tune,
                residual_target=residual_train_oof,
                cv_folds=args.residual_cv_folds,
                n_iter=args.n_iter,
                random_state=args.random_state,
                device=args.device,
            )
    except XGBoostError as exc:
        if args.device == "gpu":
            raise SystemExit(
                "XGBoost GPU 训练失败。请确认 CUDA 驱动/显卡可用，或改用 --device cpu。\n"
                f"原始错误：{exc}"
            ) from exc
        raise

    # 7) 用完整训练集上的残差重新训练最终残差模型
    try:
        residual_model = fit_final_residual_model(
            x_train_aug=x_train_aug_for_fit,
            residual_target=residual_train_oof,
            random_state=args.random_state,
            device=args.device,
            gpu_id=args.gpu_id,
            best_params=best_params,
        )
    except XGBoostError as exc:
        if args.device == "gpu":
            raise SystemExit(
                "XGBoost GPU 训练失败。请确认 CUDA 驱动/显卡可用，或改用 --device cpu。\n"
                f"原始错误：{exc}"
            ) from exc
        raise

    # 8) 评估：先看线性趋势单独表现，再看两阶段模型最终表现
    trend_metrics = calculate_metrics(y_test, trend_test_pred)
    residual_test_pred = residual_model.predict(x_test_aug)
    final_test_pred = trend_test_pred + residual_test_pred
    hybrid_metrics = calculate_metrics(y_test, final_test_pred)

    rmse_gain = trend_metrics["RMSE"] - hybrid_metrics["RMSE"]
    mae_gain = trend_metrics["MAE"] - hybrid_metrics["MAE"]
    r2_gain = hybrid_metrics["R2"] - trend_metrics["R2"]

    print("\n=== PM2.5 线性趋势 + XGBoost 残差结果 ===")
    print(f"数据文件                    : {data_path}")
    print(f"样本数（总/训练/测试）       : {len(x)} / {len(x_train)} / {len(x_test)}")
    print(f"目标列                      : {args.target}")
    if best_cv_rmse is not None:
        print(f"残差模型 CV 最优 RMSE       : {best_cv_rmse:.4f}")
    print(f"趋势模型 测试集 RMSE        : {trend_metrics['RMSE']:.4f}")
    print(f"趋势模型 测试集 MAE         : {trend_metrics['MAE']:.4f}")
    print(f"趋势模型 测试集 R^2         : {trend_metrics['R2']:.4f}")
    print(f"融合模型 测试集 RMSE        : {hybrid_metrics['RMSE']:.4f}")
    print(f"融合模型 测试集 MAE         : {hybrid_metrics['MAE']:.4f}")
    print(f"融合模型 测试集 R^2         : {hybrid_metrics['R2']:.4f}")
    print(f"RMSE 改善                   : {rmse_gain:.4f}")
    print(f"MAE 改善                    : {mae_gain:.4f}")
    print(f"R^2 提升                    : {r2_gain:.4f}")
    print(f"残差模型最优参数            : {best_params if best_params else '默认参数（未调参）'}")

    sample = pd.DataFrame(
        {
            "actual": y_test.values[:10],
            "trend_pred": trend_test_pred[:10],
            "residual_pred": residual_test_pred[:10],
            "final_pred": final_test_pred[:10],
        }
    )
    print("\n测试集前 10 条预测：")
    print(sample.to_string(index=False))

    print_top_linear_coefficients(trend_model, top_n=15)
    print_top_residual_importances(residual_model, top_n=20)

    # 9) 按需保存模型
    if not args.no_save:
        save_artifact(
            trend_model=trend_model,
            residual_model=residual_model,
            model_out=model_out,
            target_col=args.target,
            dropped_cols=dropped_cols,
        )


if __name__ == "__main__":
    main()
