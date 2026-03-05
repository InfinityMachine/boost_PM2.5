"""
基于 CatBoost 的 PM2.5 回归预测脚本（含调参与详细中文注释）
=========================================================================
脚本设计目标：
1. 预测性能：使用随机参数搜索 + K 折交叉验证，选择更优参数；
2. 易读性：函数化拆分训练流程，关键步骤提供中文注释；
3. 易复用：支持命令行参数、模型保存、元信息保存。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# 依赖检查：若缺包则直接停止，并给出安装提示（符合你的要求）。
try:
    import numpy as np
    import pandas as pd
    from catboost import CatBoostError, CatBoostRegressor, Pool
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, ParameterSampler, train_test_split
except ImportError as exc:
    missing_pkg = str(exc).split("'")[1] if "'" in str(exc) else "未知包"
    raise SystemExit(
        f"缺少依赖包：{missing_pkg}\n"
        "请先安装后再运行，例如：\n"
        "pip install catboost scikit-learn pandas numpy"
    ) from exc


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用 CatBoost 训练 PM2.5 回归模型（支持随机搜索调参）"
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
    parser.add_argument("--cv-folds", type=int, default=5, help="交叉验证折数，默认：5")
    parser.add_argument(
        "--n-iter", type=int, default=20, help="随机参数采样次数，默认：20"
    )
    parser.add_argument(
        "--early-stopping-rounds", type=int, default=80, help="早停轮数，默认：80"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="训练设备：cpu 或 gpu，默认：cpu",
    )
    parser.add_argument(
        "--gpu-devices",
        default="0",
        help="GPU 设备编号字符串（仅在 --device gpu 时生效），默认：0",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="关闭调参，仅使用默认参数训练（速度更快）",
    )
    parser.add_argument(
        "--keep-id-cols",
        action="store_true",
        help="保留疑似 ID 列（默认会自动剔除）",
    )
    parser.add_argument(
        "--model-out",
        default="pm25_catboost_model.cbm",
        help="CatBoost 模型输出路径，默认：pm25_catboost_model.cbm",
    )
    parser.add_argument(
        "--meta-out",
        default="pm25_catboost_meta.json",
        help="模型元信息输出路径，默认：pm25_catboost_meta.json",
    )
    parser.add_argument("--no-save", action="store_true", help="不保存模型和元信息")
    return parser.parse_args()


def load_data(data_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """读取数据并完成目标列校验。"""
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
        print(f"[提示] 目标列中有 {dropped} 条无效记录，已自动剔除。")

    x = x_all.loc[valid_mask].copy()
    y = y_all.loc[valid_mask].copy()
    return x, y


def remove_identifier_columns(x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    自动移除疑似 ID 列，降低过拟合风险。

    规则：列名包含 'id'（大小写不敏感）即视为 ID。
    """
    drop_cols = [col for col in x.columns if "id" in col.lower()]
    if not drop_cols:
        return x, []
    return x.drop(columns=drop_cols), drop_cols


def prepare_feature_types(x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    统一特征类型，返回：
    1) 处理后的特征 DataFrame；
    2) 类别特征列名列表（供 CatBoost 使用）。
    """
    x_new = x.copy()
    categorical_cols = x_new.select_dtypes(exclude=["number"]).columns.tolist()

    # 类别列：缺失值填充为固定字符串，并转为 str，避免类型混杂。
    for col in categorical_cols:
        x_new[col] = x_new[col].where(x_new[col].notna(), "__MISSING__").astype(str)

    # 数值列：强制转数值，非法值转 NaN；CatBoost 可自动处理 NaN。
    numeric_cols = x_new.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        x_new[col] = pd.to_numeric(x_new[col], errors="coerce")

    return x_new, categorical_cols


def evaluate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 RMSE（手动开方以兼容旧版本 sklearn）。"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def default_catboost_params(
    random_state: int, device: str, gpu_devices: str
) -> dict[str, Any]:
    """返回一组稳健的默认参数。"""
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

    # GPU 模式下启用 CatBoost 原生 GPU 训练；CPU 模式使用多线程。
    if device == "gpu":
        params.update({"task_type": "GPU", "devices": gpu_devices})
    else:
        params.update({"task_type": "CPU", "thread_count": -1})
    return params


def tune_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_cols: list[str],
    cv_folds: int,
    n_iter: int,
    early_stopping_rounds: int,
    random_state: int,
    device: str,
    gpu_devices: str,
) -> tuple[dict[str, Any], float]:
    """
    通过随机参数采样 + K 折交叉验证选择最优参数。

    说明：
    - 搜索目标是最小化平均 RMSE；
    - 每个折内都使用 early stopping，提高搜索效率并降低过拟合。
    """
    # 参数搜索空间：覆盖模型复杂度、学习率、正则项、采样策略等关键维度。
    param_space: dict[str, list[Any]] = {
        "iterations": [600, 900, 1200, 1500],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "depth": [4, 5, 6, 8, 10],
        "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0, 10.0],
        "random_strength": [0.0, 0.5, 1.0, 2.0, 5.0],
        "bagging_temperature": [0.0, 0.5, 1.0, 2.0],
    }

    base_params = default_catboost_params(random_state, device, gpu_devices)
    candidates = list(
        ParameterSampler(param_space, n_iter=n_iter, random_state=random_state)
    )
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    best_params: dict[str, Any] | None = None
    best_cv_rmse = float("inf")

    for idx, sampled_params in enumerate(candidates, start=1):
        fold_rmses: list[float] = []
        params = {**base_params, **sampled_params}

        for fold_no, (tr_idx, val_idx) in enumerate(kfold.split(x_train), start=1):
            x_tr = x_train.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            x_val = x_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            train_pool = Pool(x_tr, y_tr, cat_features=categorical_cols)
            valid_pool = Pool(x_val, y_val, cat_features=categorical_cols)

            model = CatBoostRegressor(**params)
            model.fit(
                train_pool,
                eval_set=valid_pool,
                use_best_model=True,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )

            pred_val = model.predict(valid_pool)
            fold_rmse = evaluate_rmse(y_val.to_numpy(), pred_val)
            fold_rmses.append(fold_rmse)

            print(
                f"[调参] 候选 {idx}/{len(candidates)} - fold {fold_no}/{cv_folds} "
                f"RMSE={fold_rmse:.4f}"
            )

        mean_rmse = float(np.mean(fold_rmses))
        print(f"[调参] 候选 {idx}/{len(candidates)} 平均 RMSE={mean_rmse:.4f}")

        if mean_rmse < best_cv_rmse:
            best_cv_rmse = mean_rmse
            best_params = sampled_params

    if best_params is None:
        raise RuntimeError("调参失败：未找到有效参数组合。")

    return best_params, best_cv_rmse


def train_final_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_cols: list[str],
    params: dict[str, Any],
    early_stopping_rounds: int,
    random_state: int,
) -> CatBoostRegressor:
    """
    用最优参数训练最终模型。

    训练策略：
    - 在训练集内再划分一小部分验证集，仅用于 early stopping；
    - 最终评估仍在外部测试集上进行，避免信息泄漏。
    """
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.15,
        random_state=random_state,
    )

    train_pool = Pool(x_fit, y_fit, cat_features=categorical_cols)
    valid_pool = Pool(x_val, y_val, cat_features=categorical_cols)

    model = CatBoostRegressor(**params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )
    return model


def evaluate_model(
    model: CatBoostRegressor,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: list[str],
) -> tuple[dict[str, float], np.ndarray]:
    """在测试集上计算 RMSE、MAE、R2。"""
    test_pool = Pool(x_test, y_test, cat_features=categorical_cols)
    y_pred = model.predict(test_pool)

    metrics = {
        "RMSE": evaluate_rmse(y_test.to_numpy(), y_pred),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred)),
    }
    return metrics, y_pred


def print_top_importance(model: CatBoostRegressor, top_n: int = 20) -> None:
    """打印前 N 个特征重要性。"""
    importances = model.get_feature_importance(type="FeatureImportance")
    feature_names = model.feature_names_

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    importance_df = importance_df.head(top_n)

    print(f"\nTop {top_n} 特征重要性：")
    print(importance_df.to_string(index=False))


def save_artifacts(
    model: CatBoostRegressor,
    model_out: Path,
    meta_out: Path,
    feature_cols: list[str],
    categorical_cols: list[str],
    dropped_cols: list[str],
    target_col: str,
) -> None:
    """保存模型文件与元信息文件。"""
    model.save_model(str(model_out))

    meta = {
        "target_col": target_col,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "dropped_identifier_cols": dropped_cols,
        "model_format": "catboost_cbm",
        "model_file": str(model_out),
    }
    meta_out.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n模型已保存：{model_out}")
    print(f"元信息已保存：{meta_out}")


def main() -> None:
    """主流程：读取数据 -> 清洗/分割 -> 调参 -> 训练 -> 评估 -> 保存。"""
    args = parse_args()

    data_path = Path(args.data)
    model_out = Path(args.model_out)
    meta_out = Path(args.meta_out)

    # 1) 读取数据
    x, y = load_data(data_path, args.target)

    # 2) 移除疑似 ID 列（可关闭）
    dropped_cols: list[str] = []
    if not args.keep_id_cols:
        x, dropped_cols = remove_identifier_columns(x)
        if dropped_cols:
            print(f"[信息] 已自动剔除疑似 ID 列：{dropped_cols}")

    # 3) 统一特征类型，识别类别列
    x, categorical_cols = prepare_feature_types(x)
    feature_cols = x.columns.tolist()

    # 4) 外部分割：训练集 / 测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"[信息] 当前训练设备：{args.device.upper()}")
    if args.device == "gpu":
        print(f"[信息] 使用 GPU 设备：{args.gpu_devices}")

    # 5) 调参（可选）
    base_params = default_catboost_params(
        args.random_state, args.device, args.gpu_devices
    )
    try:
        if args.no_tune:
            print("[信息] 已关闭调参，使用默认参数训练。")
            final_params = base_params
            best_cv_rmse = None
        else:
            print(
                f"[信息] 开始调参：n_iter={args.n_iter}, cv_folds={args.cv_folds}, "
                "评分指标=RMSE（越小越好）"
            )
            best_params, best_cv_rmse = tune_hyperparameters(
                x_train=x_train,
                y_train=y_train,
                categorical_cols=categorical_cols,
                cv_folds=args.cv_folds,
                n_iter=args.n_iter,
                early_stopping_rounds=args.early_stopping_rounds,
                random_state=args.random_state,
                device=args.device,
                gpu_devices=args.gpu_devices,
            )
            final_params = {**base_params, **best_params}

        # 6) 最终训练
        model = train_final_model(
            x_train=x_train,
            y_train=y_train,
            categorical_cols=categorical_cols,
            params=final_params,
            early_stopping_rounds=args.early_stopping_rounds,
            random_state=args.random_state,
        )
    except CatBoostError as exc:
        if args.device == "gpu":
            raise SystemExit(
                "CatBoost GPU 训练失败。请确认 CUDA 驱动/显卡可用，或改用 --device cpu。\n"
                f"原始错误：{exc}"
            ) from exc
        raise

    # 7) 测试集评估
    metrics, y_pred = evaluate_model(
        model=model,
        x_test=x_test,
        y_test=y_test,
        categorical_cols=categorical_cols,
    )

    print("\n=== PM2.5 CatBoost 回归结果 ===")
    print(f"数据文件                : {data_path}")
    print(f"样本数（总/训练/测试）   : {len(x)} / {len(x_train)} / {len(x_test)}")
    print(f"目标列                  : {args.target}")
    if best_cv_rmse is not None:
        print(f"CV 最优 RMSE            : {best_cv_rmse:.4f}")
    print(f"测试集 RMSE             : {metrics['RMSE']:.4f}")
    print(f"测试集 MAE              : {metrics['MAE']:.4f}")
    print(f"测试集 R^2              : {metrics['R2']:.4f}")
    print(f"最终训练参数            : {final_params}")

    sample = pd.DataFrame({"actual": y_test.values[:10], "predicted": y_pred[:10]})
    print("\n测试集前 10 条预测：")
    print(sample.to_string(index=False))

    print_top_importance(model, top_n=20)

    # 8) 保存模型与元信息
    if not args.no_save:
        save_artifacts(
            model=model,
            model_out=model_out,
            meta_out=meta_out,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            dropped_cols=dropped_cols,
            target_col=args.target,
        )


if __name__ == "__main__":
    main()
