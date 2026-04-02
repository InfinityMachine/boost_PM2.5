"""
批量扫描 blended 联合目标权重的实验脚本
=========================================================================
用途：
1. 固定同一份训练/测试切分；
2. 只训练一次趋势模型、XGBoost 残差模型、CatBoost 残差模型；
3. 在此基础上批量扫描 blended 联合目标中的 overall 权重；
4. 输出每个候选权重对应的测试集整体指标与高污染子集指标。

这样可以更高效地回答一个关键问题：
“在兼顾整体 RMSE 和高污染子集 RMSE 时，联合目标应该把多少权重给整体样本？”
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    from catboost import CatBoostError
    from xgboost.core import XGBoostError
except ImportError as exc:
    missing_pkg = str(exc).split("'")[1] if "'" in str(exc) else "未知包"
    raise SystemExit(
        f"缺少依赖包：{missing_pkg}\n"
        "请先安装后再运行，例如：\n"
        "pip install joblib scikit-learn xgboost catboost pandas numpy matplotlib"
    ) from exc

from train_pm25_trend_residual_xgboost import (
    add_log_transformed_features,
    build_high_pm25_eval_config,
    build_residual_pipeline,
    build_sample_weight_config,
    calculate_high_pm25_metrics,
    calculate_metrics,
    compute_sample_weights,
    describe_sample_weighting,
    describe_trend_model,
    fit_final_catboost_residual_model,
    fit_final_residual_model,
    fit_final_trend_model,
    generate_oof_catboost_residual_predictions,
    generate_oof_trend_predictions,
    generate_oof_xgboost_residual_predictions,
    load_data,
    make_augmented_features,
    parse_feature_list,
    predict_catboost_residual_model,
    remove_identifier_columns,
    resolve_ensemble_weights,
    resolve_groups,
    split_train_test_by_group,
    split_train_test_random,
    tune_catboost_residual_model,
    tune_residual_model,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="批量扫描 blended 联合目标权重，并输出 PM2.5 集成结果表"
    )
    parser.add_argument("--data", default="data.csv", help="训练数据 CSV 路径，默认：data.csv")
    parser.add_argument("--target", default="PM2.5", help="目标列名，默认：PM2.5")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例，默认：0.2")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子，默认：42")
    parser.add_argument(
        "--validation-mode",
        choices=["group", "random"],
        default="random",
        help="验证方式：group=按地区分组验证，random=随机切分，默认：random",
    )
    parser.add_argument(
        "--group-col",
        default="CITY",
        help="分组验证使用的列名，默认：CITY。仅在 --validation-mode group 时生效",
    )
    parser.add_argument(
        "--trend-model",
        choices=["ridge", "linear"],
        default="ridge",
        help="趋势模型类型，默认：ridge",
    )
    parser.add_argument(
        "--trend-cv-folds",
        type=int,
        default=5,
        help="趋势模型生成 OOF 预测时使用的折数，默认：5",
    )
    parser.add_argument(
        "--residual-cv-folds",
        type=int,
        default=5,
        help="残差模型调参/OOF 预测时使用的折数，默认：5",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="残差模型随机搜索调参迭代次数，默认：20",
    )
    parser.add_argument(
        "--catboost-early-stopping-rounds",
        type=int,
        default=80,
        help="CatBoost 残差模型早停轮数，默认：80",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="残差模型训练设备：cpu 或 gpu，默认：cpu",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU 编号。XGBoost 使用 cuda:<gpu-id>，CatBoost 使用 devices=<gpu-id>",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="关闭残差模型调参，仅使用默认参数训练",
    )
    parser.add_argument(
        "--sample-weight-mode",
        choices=["none", "high_pm25"],
        default="none",
        help="样本加权方式，默认：none",
    )
    parser.add_argument(
        "--high-pm25-quantile",
        type=float,
        default=0.75,
        help="高 PM2.5 样本阈值分位数，仅在加权模式为 high_pm25 时生效，默认：0.75",
    )
    parser.add_argument(
        "--high-pm25-weight",
        type=float,
        default=2.0,
        help="高 PM2.5 样本权重倍数，仅在加权模式为 high_pm25 时生效，默认：2.0",
    )
    parser.add_argument(
        "--eval-high-pm25-quantile",
        type=float,
        default=0.75,
        help="高污染子集评估阈值分位数，默认：0.75",
    )
    parser.add_argument(
        "--log-feature-cols",
        default="NOX,SO2,fertilzier,manure",
        help="需要生成 log1p 特征的列名，逗号分隔",
    )
    parser.add_argument(
        "--no-log-features",
        action="store_true",
        help="关闭 log1p 数值特征扩展",
    )
    parser.add_argument(
        "--keep-id-cols",
        action="store_true",
        help="保留疑似 ID 列（默认会自动剔除）",
    )
    parser.add_argument(
        "--weights",
        default="0.55,0.60,0.65,0.70,0.75,0.80",
        help=(
            "需要扫描的 blended overall 权重列表，逗号分隔。"
            "例如：0.55,0.60,0.65,0.70,0.75,0.80"
        ),
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="额外输出 overall/high_pm25 两种自动定权基线行，便于对照",
    )
    parser.add_argument(
        "--ensemble-weight-grid-step",
        type=float,
        default=0.02,
        help="自动搜索 XGBoost/CatBoost 集成权重时的网格步长，默认：0.02",
    )
    parser.add_argument(
        "--out",
        default="pm25_blended_weight_sweep.csv",
        help="实验结果输出 CSV 路径，默认：pm25_blended_weight_sweep.csv",
    )
    return parser.parse_args()


def parse_weight_candidates(raw_text: str) -> list[float]:
    """解析 blended overall 权重列表。"""
    weights: list[float] = []
    for item in raw_text.split(","):
        text = item.strip()
        if not text:
            continue
        value = float(text)
        if not 0.0 <= value <= 1.0:
            raise ValueError("weights 中的每个权重都必须在 [0, 1] 之间。")
        weights.append(value)

    if not weights:
        raise ValueError("weights 不能为空。")

    # 去重并排序，便于实验结果更稳定、可读。
    return sorted(set(weights))


def build_result_row(
    row_name: str,
    objective: str,
    ensemble_config: dict[str, Any],
    y_test: pd.Series,
    final_test_pred,
    high_pm25_threshold: float,
) -> dict[str, Any]:
    """统一组装一行实验结果。"""
    metrics = calculate_metrics(y_test, final_test_pred)
    high_metrics, high_count = calculate_high_pm25_metrics(
        y_true=y_test,
        y_pred=final_test_pred,
        threshold=high_pm25_threshold,
    )

    row: dict[str, Any] = {
        "row_name": row_name,
        "objective": objective,
        "xgb_weight": float(ensemble_config["xgb_weight"]),
        "catboost_weight": float(ensemble_config["catboost_weight"]),
        "weight_strategy": ensemble_config["strategy"],
        "objective_metric": ensemble_config.get("objective_metric", ""),
        "objective_score": ensemble_config.get("objective_score"),
        "test_rmse": float(metrics["RMSE"]),
        "test_mae": float(metrics["MAE"]),
        "test_r2": float(metrics["R2"]),
        "high_pm25_rmse": float(high_metrics["RMSE"]),
        "high_pm25_mae": float(high_metrics["MAE"]),
        "high_pm25_r2": float(high_metrics["R2"]),
        "high_pm25_sample_count": int(high_count),
    }

    for key in [
        "grid_step",
        "blended_overall_weight",
        "blended_high_weight",
        "overall_rmse",
        "normalized_overall_rmse",
        "normalized_high_pm25_rmse",
    ]:
        if key in ensemble_config:
            row[key] = ensemble_config[key]

    return row


def main() -> None:
    """主流程：训练一次，扫描多组 blended 权重，输出对比表。"""
    args = parse_args()
    weight_candidates = parse_weight_candidates(args.weights)
    out_path = Path(args.out)

    # 1) 读取与基础特征清洗
    x, y = load_data(Path(args.data), args.target)

    dropped_cols: list[str] = []
    if not args.keep_id_cols:
        x, dropped_cols = remove_identifier_columns(x)
        if dropped_cols:
            print(f"[信息] 已自动剔除疑似 ID 列：{dropped_cols}")

    if args.no_log_features:
        print("[信息] 已关闭 log1p 特征扩展。")
        added_log_cols: list[str] = []
    else:
        x, added_log_cols, skipped_log_cols = add_log_transformed_features(
            x=x,
            candidate_cols=parse_feature_list(args.log_feature_cols),
        )
        if added_log_cols:
            print(f"[信息] 已新增 log1p 特征：{added_log_cols}")
        if skipped_log_cols:
            print(f"[信息] 跳过的 log1p 特征：{skipped_log_cols}")

    # 2) 固定一次数据切分，确保所有候选权重都在同一条件下比较
    groups = None
    groups_train = None
    groups_test = None
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
    print(f"[信息] 当前趋势模型：{args.trend_model}")
    print(f"[信息] 当前验证方式：{args.validation_mode}")
    print(f"[信息] blended 权重候选：{weight_candidates}")
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

    # 3) 趋势层只训练一次
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
    trend_model = fit_final_trend_model(
        x_train=x_train,
        y_train=y_train,
        trend_model=args.trend_model,
        sample_weight=trend_sample_weight,
    )
    trend_test_pred = trend_model.predict(x_test)
    trend_metrics = calculate_metrics(y_test, trend_test_pred)
    trend_high_metrics, trend_high_count = calculate_high_pm25_metrics(
        y_true=y_test,
        y_pred=trend_test_pred,
        threshold=high_pm25_eval_config["threshold"],
    )

    # 4) 残差训练目标与增强特征
    residual_train_oof = y_train - trend_train_oof_pred
    x_train_aug_for_tune = make_augmented_features(x_train, trend_train_oof_pred)
    x_train_aug_for_fit = x_train_aug_for_tune.copy()
    x_test_aug = make_augmented_features(x_test, trend_test_pred)

    # 5) 训练 XGBoost 残差
    xgb_best_params = None
    xgb_best_cv_rmse = None
    try:
        if args.no_tune:
            print("[信息] 已关闭 XGBoost 残差调参，将使用默认参数训练。")
        else:
            print(
                f"[信息] 开始 XGBoost 残差调参：n_iter={args.n_iter}, "
                f"residual_cv_folds={args.residual_cv_folds}"
            )
            xgb_tune_pipeline = build_residual_pipeline(
                x_train_aug_for_tune,
                args.random_state,
                args.device,
                args.gpu_id,
            )
            xgb_best_params, xgb_best_cv_rmse = tune_residual_model(
                pipeline=xgb_tune_pipeline,
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

    # 6) 训练 CatBoost 残差
    cat_best_params = None
    cat_best_cv_rmse = None
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

    # 7) 生成两类残差模型的测试集预测与 OOF 预测
    xgb_residual_test_pred = xgb_residual_model.predict(x_test_aug)
    cat_residual_test_pred = predict_catboost_residual_model(
        model=catboost_residual_model,
        x_aug=x_test_aug,
        feature_cols=catboost_feature_cols,
        categorical_cols=catboost_categorical_cols,
    )

    print("[信息] 开始生成 XGBoost / CatBoost 残差 OOF 预测，用于扫描集成权重。")
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

    # 8) 先记录单模型结果，便于判断集成是否真的带来收益
    rows: list[dict[str, Any]] = []
    xgb_single_pred = trend_test_pred + xgb_residual_test_pred
    cat_single_pred = trend_test_pred + cat_residual_test_pred
    rows.append(
        build_result_row(
            row_name="single_xgboost",
            objective="single",
            ensemble_config={
                "strategy": "single_model",
                "xgb_weight": 1.0,
                "catboost_weight": 0.0,
            },
            y_test=y_test,
            final_test_pred=xgb_single_pred,
            high_pm25_threshold=high_pm25_eval_config["threshold"],
        )
    )
    rows.append(
        build_result_row(
            row_name="single_catboost",
            objective="single",
            ensemble_config={
                "strategy": "single_model",
                "xgb_weight": 0.0,
                "catboost_weight": 1.0,
            },
            y_test=y_test,
            final_test_pred=cat_single_pred,
            high_pm25_threshold=high_pm25_eval_config["threshold"],
        )
    )

    # 9) 可选：补充整体导向 / 高污染导向的自动定权基线
    if args.include_baselines:
        for objective in ["overall", "high_pm25"]:
            ensemble_config = resolve_ensemble_weights(
                residual_model="ensemble",
                manual_weight_xgb=None,
                ensemble_weight_objective=objective,
                ensemble_weight_grid_step=args.ensemble_weight_grid_step,
                ensemble_blended_overall_weight=0.7,
                y_train=y_train,
                trend_train_oof_pred=trend_train_oof_pred,
                xgb_residual_oof_pred=xgb_residual_train_oof_pred,
                cat_residual_oof_pred=cat_residual_train_oof_pred,
                high_pm25_threshold=high_pm25_eval_config["threshold"],
                xgb_cv_rmse=xgb_best_cv_rmse,
                cat_cv_rmse=cat_best_cv_rmse,
            )
            final_test_pred = trend_test_pred + (
                ensemble_config["xgb_weight"] * xgb_residual_test_pred
                + ensemble_config["catboost_weight"] * cat_residual_test_pred
            )
            rows.append(
                build_result_row(
                    row_name=f"baseline_{objective}",
                    objective=objective,
                    ensemble_config=ensemble_config,
                    y_test=y_test,
                    final_test_pred=final_test_pred,
                    high_pm25_threshold=high_pm25_eval_config["threshold"],
                )
            )

    # 10) 正式扫描 blended 联合目标中的 overall 权重
    for blended_weight in weight_candidates:
        ensemble_config = resolve_ensemble_weights(
            residual_model="ensemble",
            manual_weight_xgb=None,
            ensemble_weight_objective="blended",
            ensemble_weight_grid_step=args.ensemble_weight_grid_step,
            ensemble_blended_overall_weight=blended_weight,
            y_train=y_train,
            trend_train_oof_pred=trend_train_oof_pred,
            xgb_residual_oof_pred=xgb_residual_train_oof_pred,
            cat_residual_oof_pred=cat_residual_train_oof_pred,
            high_pm25_threshold=high_pm25_eval_config["threshold"],
            xgb_cv_rmse=xgb_best_cv_rmse,
            cat_cv_rmse=cat_best_cv_rmse,
        )
        final_test_pred = trend_test_pred + (
            ensemble_config["xgb_weight"] * xgb_residual_test_pred
            + ensemble_config["catboost_weight"] * cat_residual_test_pred
        )
        rows.append(
            build_result_row(
                row_name=f"blended_{blended_weight:.2f}",
                objective="blended",
                ensemble_config=ensemble_config,
                y_test=y_test,
                final_test_pred=final_test_pred,
                high_pm25_threshold=high_pm25_eval_config["threshold"],
            )
        )

    results_df = pd.DataFrame(rows)
    results_df["rmse_rank"] = results_df["test_rmse"].rank(method="min")
    results_df["high_pm25_rmse_rank"] = results_df["high_pm25_rmse"].rank(method="min")
    results_df["combined_rank_sum"] = (
        results_df["rmse_rank"] + results_df["high_pm25_rmse_rank"]
    )
    results_df = results_df.sort_values(
        ["combined_rank_sum", "test_rmse", "high_pm25_rmse", "row_name"]
    ).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== 实验配置摘要 ===")
    print(f"趋势模型                : {describe_trend_model(trend_model)}")
    print(f"训练设备                : {args.device}")
    print(f"验证方式                : {args.validation_mode}")
    print(f"样本数（总/训练/测试）   : {len(x)} / {len(x_train)} / {len(x_test)}")
    print(f"趋势模型测试集 RMSE      : {trend_metrics['RMSE']:.4f}")
    print(f"趋势模型高污染 RMSE      : {trend_high_metrics['RMSE']:.4f}")
    if xgb_best_cv_rmse is not None:
        print(f"XGBoost 残差 CV RMSE    : {xgb_best_cv_rmse:.4f}")
    if cat_best_cv_rmse is not None:
        print(f"CatBoost 残差 CV RMSE   : {cat_best_cv_rmse:.4f}")

    print("\n=== Top 结果（按 combined_rank_sum 排序） ===")
    display_cols = [
        "row_name",
        "objective",
        "xgb_weight",
        "catboost_weight",
        "test_rmse",
        "high_pm25_rmse",
        "test_mae",
        "high_pm25_mae",
        "test_r2",
        "high_pm25_r2",
        "combined_rank_sum",
    ]
    existing_display_cols = [col for col in display_cols if col in results_df.columns]
    print(results_df[existing_display_cols].head(10).to_string(index=False))
    print(f"\n实验结果已保存到：{out_path}")


if __name__ == "__main__":
    main()
