"""
基于 XGBoost 的 PM2.5 回归预测脚本（含交叉验证调参与完整中文注释）
=========================================================================
核心目标：
1. 在可读性较高的前提下，构建稳定、可复用的数据建模流程；
2. 通过随机搜索 + K 折交叉验证，提升模型泛化性能；
3. 支持直接保存训练好的 Pipeline，便于后续加载与部署。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# 依赖检查：如果缺少必要包，直接给出安装提示并停止运行。
try:
    import joblib
    import matplotlib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBRegressor
    from xgboost.core import XGBoostError
except ImportError as exc:
    missing_pkg = str(exc).split("'")[1] if "'" in str(exc) else "未知包"
    raise SystemExit(
        f"缺少依赖包：{missing_pkg}\n"
        "请先安装后再运行，例如：\n"
        "pip install joblib scikit-learn xgboost pandas numpy"
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """解析命令行参数，便于在不同数据和配置下复用脚本。"""
    parser = argparse.ArgumentParser(
        description="使用 XGBoost 训练 PM2.5 回归模型（支持交叉验证调参）"
    )
    parser.add_argument("--data", default="data.csv", help="训练数据 CSV 路径，默认：data.csv")
    parser.add_argument("--target", default="PM2.5", help="目标列名，默认：PM2.5")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例，默认：0.2")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子，默认：42")
    parser.add_argument("--cv-folds", type=int, default=5, help="交叉验证折数，默认：5")
    parser.add_argument("--n-iter", type=int, default=30, help="随机搜索迭代次数，默认：30")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="训练设备：cpu 或 gpu，默认：cpu",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU 编号（仅在 --device gpu 时生效）")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="关闭随机搜索调参，仅使用默认参数训练（速度更快）",
    )
    parser.add_argument(
        "--model-out",
        default="cache/pm25_xgboost_pipeline.joblib",
        help="模型输出文件路径，默认：cache/pm25_xgboost_pipeline.joblib",
    )
    parser.add_argument(
        "--plot-out",
        default=".diag/pm25_xgboost_diagnostics.png",
        help="诊断图输出路径，默认：.diag/pm25_xgboost_diagnostics.png",
    )
    parser.add_argument(
        "--keep-id-cols",
        action="store_true",
        help="保留疑似 ID 列（默认会自动剔除，如 OBJECTID_1）",
    )
    parser.add_argument("--no-plot", action="store_true", help="不绘制并保存诊断图")
    parser.add_argument("--no-save", action="store_true", help="不保存训练好的模型")
    return parser.parse_args()


def load_data(data_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """读取 CSV 并做基础校验，返回特征矩阵 X 和目标变量 y。"""
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件：{data_path}")

    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在，当前列：{list(df.columns)}")

    # 目标列强制转数值。若出现非法值则记为 NaN，后续统一剔除。
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
    自动移除疑似 ID 列，降低“记住样本编号”导致的过拟合风险。

    判定规则：
    1) 列名包含 id/ID（例如 OBJECTID_1）。

    说明：此前若按“高唯一值数值列”做规则，会误删 NOX、SO2 等真实有效特征，
    反而降低性能，因此这里采用保守策略，只按列名判定。
    """
    drop_cols: list[str] = []
    for col in x.columns:
        col_lower = col.lower()
        is_id_name = "id" in col_lower

        if is_id_name:
            drop_cols.append(col)

    if not drop_cols:
        return x, []

    return x.drop(columns=drop_cols), drop_cols


def build_pipeline(x: pd.DataFrame, random_state: int, device: str, gpu_id: int) -> Pipeline:
    """
    构建完整训练流水线：
    1) 数值特征：中位数填补缺失；
    2) 类别特征：众数填补 + One-Hot 编码；
    3) 模型：XGBRegressor 回归器。
    """
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = x.select_dtypes(exclude=["number"]).columns.tolist()

    # 数值列预处理流程
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # 类别列预处理流程
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 将不同类型特征的预处理整合到同一 ColumnTransformer
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # 先给一个较稳健的初始参数；若启用调参，后续会被 RandomizedSearchCV 搜索优化。
    model_kwargs: dict[str, Any] = {
        "objective": "reg:squarederror",
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": random_state,
    }

    # GPU 模式说明：
    # 1) 使用 gpu_hist 可显著加速树构建；
    # 2) 调参与训练默认单并发，避免多任务同时占用显存导致速度下降或 OOM。
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

    model = XGBRegressor(
        **model_kwargs,
    )

    # Pipeline 的好处：训练、预测、保存、加载都只对一个对象操作，不易出错。
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def tune_hyperparameters(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    n_iter: int,
    random_state: int,
    device: str,
) -> RandomizedSearchCV:
    """对 XGBoost 参数做随机搜索，使用 K 折交叉验证选择最优配置。"""
    # 使用列表形式定义搜索空间，避免额外依赖 scipy。
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

    # KFold 用于回归任务的交叉验证，shuffle=True 可降低数据顺序带来的偏差。
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # GPU 调参时如果并发过高，多个候选会争用同一块显存，反而变慢或直接 OOM。
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
        refit=True,
    )
    search.fit(x_train, y_train)
    return search


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """在测试集上评估模型性能，返回常用回归指标。"""
    y_pred = model.predict(x_test)
    metrics = {
        # 兼容旧版 scikit-learn：不使用 squared=False 参数，统一手动开方。
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred)),
    }
    return metrics


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
    axes[0].plot(y_pred_sorted, label="Predicted", linewidth=2.0, color="#ff7f0e", alpha=0.85)
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


def print_top_importances(model: Pipeline, top_n: int = 20) -> None:
    """打印前 N 个最重要特征，帮助理解模型关注点。"""
    preprocess = model.named_steps["preprocess"]
    regressor = model.named_steps["model"]

    try:
        feature_names = preprocess.get_feature_names_out()
        importances = regressor.feature_importances_
    except Exception:
        print("\n[提示] 当前模型无法提取特征重要性。")
        return

    if len(feature_names) != len(importances):
        print("\n[提示] 特征名数量与重要性数量不一致，跳过重要性展示。")
        return

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)

    print(f"\nTop {top_n} 特征重要性：")
    print(importance_df.to_string(index=False))


def main() -> None:
    """脚本主流程：读取数据 -> 划分数据 -> 调参/训练 -> 评估 -> 保存。"""
    args = parse_args()
    model_out = Path(args.model_out)
    plot_out = Path(args.plot_out)

    # 第一步：读取并校验数据
    data_path = Path(args.data)
    x, y = load_data(data_path, args.target)

    # 可选：自动剔除疑似 ID 字段，提升泛化能力。
    if not args.keep_id_cols:
        x, dropped_cols = remove_identifier_columns(x)
        if dropped_cols:
            print(f"[信息] 已自动剔除疑似 ID 列：{dropped_cols}")

    # 第二步：划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"[信息] 当前训练设备：{args.device.upper()}")
    if args.device == "gpu":
        print(f"[信息] 使用 GPU 编号：{args.gpu_id}")

    # 第三步：构建基线 Pipeline
    pipeline = build_pipeline(x_train, args.random_state, args.device, args.gpu_id)

    # 第四步：可选的随机搜索调参
    try:
        if args.no_tune:
            print("[信息] 已关闭调参，将使用默认参数直接训练。")
            pipeline.fit(x_train, y_train)
            best_model = pipeline
            best_params = "默认参数（未调参）"
            best_cv_rmse = None
        else:
            print(
                f"[信息] 开始随机搜索调参：n_iter={args.n_iter}, cv_folds={args.cv_folds}，"
                "评分指标=RMSE（越小越好）"
            )
            search = tune_hyperparameters(
                pipeline=pipeline,
                x_train=x_train,
                y_train=y_train,
                cv_folds=args.cv_folds,
                n_iter=args.n_iter,
                random_state=args.random_state,
                device=args.device,
            )
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_cv_rmse = -float(search.best_score_)
    except XGBoostError as exc:
        if args.device == "gpu":
            raise SystemExit(
                "XGBoost GPU 训练失败。请确认 CUDA 驱动/显卡可用，或改用 --device cpu。\n"
                f"原始错误：{exc}"
            ) from exc
        raise

    # 第五步：在独立测试集上评估
    metrics = evaluate_model(best_model, x_test, y_test)

    print("\n=== PM2.5 XGBoost 回归结果 ===")
    print(f"数据文件                : {data_path}")
    print(f"样本数（总/训练/测试）   : {len(x)} / {len(x_train)} / {len(x_test)}")
    print(f"目标列                  : {args.target}")
    if best_cv_rmse is not None:
        print(f"CV 最优 RMSE            : {best_cv_rmse:.4f}")
    print(f"测试集 RMSE             : {metrics['RMSE']:.4f}")
    print(f"测试集 MAE              : {metrics['MAE']:.4f}")
    print(f"测试集 R^2              : {metrics['R2']:.4f}")
    print(f"最优参数                : {best_params}")

    # 打印测试集前 10 条预测结果，便于快速人工检查
    y_pred = best_model.predict(x_test)
    sample = pd.DataFrame({"actual": y_test.values[:10], "predicted": y_pred[:10]})
    print("\n测试集前 10 条预测：")
    print(sample.to_string(index=False))

    if not args.no_plot:
        save_diagnostic_plot(
            y_true=y_test,
            y_pred=y_pred,
            plot_out=plot_out,
            model_name="XGBoost Regression",
        )
        print(f"\n诊断图已保存到：{plot_out}")

    # 展示主要特征重要性，提升可解释性
    print_top_importances(best_model, top_n=20)

    # 第六步：按需保存模型
    if not args.no_save:
        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_out)
        print(f"\n模型已保存到：{model_out}")


if __name__ == "__main__":
    main()
