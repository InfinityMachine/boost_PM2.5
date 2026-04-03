"""
自动比较不同建模路径 RMSE 表现的实验脚本
=========================================================================
用途：
1. 固定同一份训练/测试切分；
2. 共享同一个趋势模型；
3. 自动训练并比较多条残差建模路径；
4. 输出整体 RMSE、高污染子集 RMSE 等指标到一张对比表。

默认会比较这些常见路线：
1. trend + xgboost(all)
2. trend + xgboost(complementary)
3. trend + catboost
4. trend + catboost_experts
5. trend + ensemble(overall, all)
6. trend + ensemble(overall, complementary)
7. trend + ensemble(blended, complementary)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import matplotlib
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["axes.unicode_minus"] = False

from train_pm25_trend_residual_xgboost import (
    DEFAULT_REGION_PRIOR_COLS,
    add_log_transformed_features,
    append_region_prior_features,
    build_high_pm25_eval_config,
    build_residual_pipeline,
    build_sample_weight_config,
    calculate_high_pm25_metrics,
    calculate_metrics,
    compute_sample_weights,
    default_xgb_residual_params,
    describe_sample_weighting,
    describe_trend_model,
    fit_final_catboost_residual_model,
    fit_catboost_regime_expert_bundle,
    fit_final_residual_model,
    fit_final_trend_model,
    generate_region_prior_features,
    generate_oof_catboost_residual_predictions,
    generate_oof_trend_predictions,
    generate_oof_xgboost_residual_predictions,
    load_data,
    make_augmented_features,
    parse_feature_list,
    predict_catboost_regime_expert_bundle,
    predict_catboost_residual_model,
    prepare_xgb_residual_features,
    remove_identifier_columns,
    resolve_ensemble_weights,
    resolve_groups,
    split_train_test_by_group,
    split_train_test_random,
    tune_catboost_residual_model,
    tune_residual_model,
)


DEFAULT_PATHS = (
    "xgb_all",
    "xgb_complementary",
    "catboost",
    "catboost_experts",
    "ensemble_overall_all",
    "ensemble_overall_complementary",
    "ensemble_blended_complementary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自动比较多条 PM2.5 模型路径的 RMSE 表现"
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
    parser.add_argument("--trend-cv-folds", type=int, default=5, help="趋势 OOF 折数，默认：5")
    parser.add_argument("--residual-cv-folds", type=int, default=5, help="残差 OOF/调参折数，默认：5")
    parser.add_argument("--n-iter", type=int, default=20, help="随机搜索迭代次数，默认：20")
    parser.add_argument(
        "--catboost-early-stopping-rounds",
        type=int,
        default=80,
        help="CatBoost 残差早停轮数，默认：80",
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
    parser.add_argument("--no-log-features", action="store_true", help="关闭 log1p 特征扩展")
    parser.add_argument(
        "--catboost-expert-regime",
        choices=["trend_high", "low_wind", "trend_high_or_low_wind"],
        default="trend_high_or_low_wind",
        help=(
            "CatBoost 专家模型的特殊 regime。"
            "trend_high=趋势预测较高的样本，"
            "low_wind=低风速样本，"
            "trend_high_or_low_wind=两者取并集，默认：trend_high_or_low_wind"
        ),
    )
    parser.add_argument(
        "--catboost-expert-trend-quantile",
        type=float,
        default=0.75,
        help="使用趋势预测划分高污染 regime 的分位数阈值，默认：0.75",
    )
    parser.add_argument(
        "--catboost-expert-low-wind-quantile",
        type=float,
        default=0.35,
        help="使用 wind 划分低风速 regime 的分位数阈值，默认：0.35",
    )
    parser.add_argument(
        "--catboost-expert-special-weight",
        type=float,
        default=0.75,
        help="特殊 regime 内特殊专家预测的融合权重，默认：0.75",
    )
    parser.add_argument(
        "--catboost-expert-min-samples",
        type=int,
        default=120,
        help="特殊 regime 最少训练样本数，小于该值则退化为普通 CatBoost，默认：120",
    )
    parser.add_argument(
        "--region-prior-cols",
        default="PROVINCE,CITY",
        help="区域层级先验特征使用的地区列，按层级从粗到细填写，默认：PROVINCE,CITY",
    )
    parser.add_argument(
        "--region-prior-smoothing",
        type=float,
        default=15.0,
        help="区域层级先验的平滑强度，默认：15.0",
    )
    parser.add_argument(
        "--region-prior-smoothing-values",
        default="",
        help=(
            "需要批量比较的区域层级先验平滑强度，逗号分隔。"
            "可写数值，也可写 off/none 表示关闭区域先验。"
            "例如：5,10,15,20,off。默认：空（只使用 --region-prior-smoothing）"
        ),
    )
    parser.add_argument(
        "--no-region-priors",
        action="store_true",
        help="关闭区域层级先验特征",
    )
    parser.add_argument(
        "--county-col",
        default="COUNTY",
        help="需要做消融实验的县级列名，默认：COUNTY",
    )
    parser.add_argument(
        "--county-ablation-modes",
        default="keep",
        help=(
            "COUNTY 消融模式，逗号分隔。支持：keep, drop。"
            "例如：keep,drop。默认：keep"
        ),
    )
    parser.add_argument("--keep-id-cols", action="store_true", help="保留疑似 ID 列")
    parser.add_argument(
        "--xgb-complementary-drop-cols",
        default="CITY,COUNTY",
        help="XGBoost complementary 视图中需要移除的高基数列，逗号分隔",
    )
    parser.add_argument(
        "--no-xgb-complementary-interactions",
        action="store_true",
        help="关闭 XGBoost complementary 视图下的默认数值交互特征",
    )
    parser.add_argument(
        "--no-xgb-all-interactions",
        action="store_true",
        help="关闭 XGBoost all 视图下的默认精选交互特征",
    )
    parser.add_argument(
        "--ensemble-weight-grid-step",
        type=float,
        default=0.02,
        help="自动搜索 XGBoost/CatBoost 集成权重时的网格步长，默认：0.02",
    )
    parser.add_argument(
        "--ensemble-blended-overall-weight",
        type=float,
        default=0.7,
        help="blended 联合目标中整体 RMSE 的权重，默认：0.7",
    )
    parser.add_argument(
        "--paths",
        default=",".join(DEFAULT_PATHS),
        help=(
            "需要比较的路径列表，逗号分隔。默认："
            + ",".join(DEFAULT_PATHS)
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="比较结果输出 CSV 路径；若不指定，将自动保存到 .diag/<run-name>/ 下",
    )
    parser.add_argument(
        "--plot-out",
        default=None,
        help="路径对比图输出 PNG 路径；若不指定，将自动保存到 .diag/<run-name>/ 下",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不生成路径对比图",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="实验运行名称。若不指定，将自动生成带时间戳的目录名",
    )
    return parser.parse_args()


def sanitize_run_name(run_name: str) -> str:
    sanitized = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_name.strip()
    ).strip("_")
    return sanitized or "run"


def build_default_run_name(prefix: str, run_name: str | None) -> str:
    if run_name:
        return sanitize_run_name(run_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{sanitize_run_name(prefix)}_{timestamp}"


def resolve_output_path(out_arg: str | None, run_name: str) -> Path:
    if out_arg:
        return Path(out_arg)
    return Path(f".diag/{run_name}/pm25_path_compare.csv")


def resolve_plot_path(plot_arg: str | None, run_name: str) -> Path:
    if plot_arg:
        return Path(plot_arg)
    return Path(f".diag/{run_name}/pm25_path_compare.png")


def parse_path_candidates(raw_text: str) -> list[str]:
    candidates = [item.strip() for item in raw_text.split(",") if item.strip()]
    if not candidates:
        raise ValueError("paths 不能为空。")
    return candidates


def parse_region_prior_setting_candidates(
    raw_text: str,
    default_smoothing: float,
    no_region_priors: bool,
) -> list[dict[str, Any]]:
    """
    解析区域层级先验的对比设置。

    支持两种使用方式：
    1. 不传 `--region-prior-smoothing-values`，则只跑当前单一设置；
    2. 传入多个值时，脚本会在同一次运行中逐个比较。
    """
    if not raw_text.strip():
        if no_region_priors:
            return [
                {
                    "enabled": False,
                    "smoothing": None,
                    "label": "off",
                }
            ]
        return [
            {
                "enabled": True,
                "smoothing": float(default_smoothing),
                "label": f"s{default_smoothing:g}",
            }
        ]

    candidates: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for item in [part.strip() for part in raw_text.split(",") if part.strip()]:
        item_lower = item.lower()
        if item_lower in {"off", "none", "disable", "disabled"}:
            label = "off"
            candidate = {"enabled": False, "smoothing": None, "label": label}
        else:
            try:
                smoothing = float(item)
            except ValueError as exc:
                raise ValueError(
                    f"无法解析 region-prior-smoothing-values 中的值：{item}"
                ) from exc
            if smoothing <= 0.0:
                raise ValueError("区域层级先验平滑强度必须大于 0。")
            label = f"s{smoothing:g}"
            candidate = {"enabled": True, "smoothing": smoothing, "label": label}

        if label not in seen_labels:
            candidates.append(candidate)
            seen_labels.add(label)

    if not candidates:
        raise ValueError("region-prior-smoothing-values 不能为空。")
    return candidates


def parse_county_ablation_candidates(
    raw_text: str,
) -> list[dict[str, Any]]:
    """解析 COUNTY 消融实验配置。"""
    candidates: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    raw_items = [item.strip() for item in raw_text.split(",") if item.strip()]
    if not raw_items:
        raw_items = ["keep"]

    for item in raw_items:
        item_lower = item.lower()
        if item_lower in {"keep", "on", "full"}:
            label = "keep"
            drop_county = False
        elif item_lower in {"drop", "drop_county", "no_county", "off"}:
            label = "drop"
            drop_county = True
        else:
            raise ValueError(
                "county-ablation-modes 只支持 keep 和 drop。"
            )

        if label not in seen_labels:
            candidates.append(
                {
                    "label": label,
                    "drop_county": drop_county,
                }
            )
            seen_labels.add(label)

    return candidates


def apply_county_ablation(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    county_col: str,
    county_setting: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """根据设定保留或移除 COUNTY 列。"""
    x_train_new = x_train.copy()
    x_test_new = x_test.copy()
    removed_cols: list[str] = []

    if county_setting["drop_county"] and county_col in x_train_new.columns:
        x_train_new = x_train_new.drop(columns=[county_col])
        if county_col in x_test_new.columns:
            x_test_new = x_test_new.drop(columns=[county_col])
        removed_cols.append(county_col)

    return x_train_new, x_test_new, {
        "label": county_setting["label"],
        "drop_county": bool(county_setting["drop_county"]),
        "county_col": county_col,
        "removed_cols": removed_cols,
    }


def build_result_row(
    path_name: str,
    path_type: str,
    xgb_view: str,
    objective: str,
    ensemble_config: dict[str, Any],
    y_test: pd.Series,
    final_test_pred,
    high_pm25_threshold: float,
    region_prior_enabled: bool,
    region_prior_smoothing: float | None,
    region_prior_label: str,
    county_ablation_label: str,
    county_removed_cols: list[str],
    expert_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = calculate_metrics(y_test, final_test_pred)
    high_metrics, high_count = calculate_high_pm25_metrics(
        y_true=y_test,
        y_pred=final_test_pred,
        threshold=high_pm25_threshold,
    )
    expert_config = expert_config or {}
    return {
        "path_name": path_name,
        "path_type": path_type,
        "xgb_view": xgb_view,
        "objective": objective,
        "xgb_weight": float(ensemble_config.get("xgb_weight", 0.0)),
        "catboost_weight": float(ensemble_config.get("catboost_weight", 0.0)),
        "weight_strategy": ensemble_config.get("strategy", ""),
        "objective_metric": ensemble_config.get("objective_metric", ""),
        "objective_score": ensemble_config.get("objective_score"),
        "county_ablation_label": county_ablation_label,
        "county_removed_cols": ",".join(county_removed_cols),
        "region_prior_enabled": bool(region_prior_enabled),
        "region_prior_smoothing": region_prior_smoothing,
        "region_prior_label": region_prior_label,
        "expert_regime_requested": expert_config.get("requested_regime_mode", ""),
        "expert_regime_effective": expert_config.get("effective_regime_mode", ""),
        "expert_special_enabled": bool(expert_config.get("special_expert_enabled", False)),
        "expert_special_weight": expert_config.get("special_weight"),
        "expert_special_train_count": expert_config.get("special_train_count"),
        "expert_special_train_fraction": expert_config.get("special_train_fraction"),
        "expert_special_test_count": expert_config.get("special_test_count"),
        "test_rmse": float(metrics["RMSE"]),
        "test_mae": float(metrics["MAE"]),
        "test_r2": float(metrics["R2"]),
        "high_pm25_rmse": float(high_metrics["RMSE"]),
        "high_pm25_mae": float(high_metrics["MAE"]),
        "high_pm25_r2": float(high_metrics["R2"]),
        "high_pm25_sample_count": int(high_count),
    }


def plot_path_comparison_chart(results_df: pd.DataFrame, plot_path: Path) -> None:
    """绘制各建模路径在整体 RMSE 与高污染子集 RMSE 上的对比图。"""
    has_multiple_county_modes = (
        "county_ablation_label" in results_df.columns
        and results_df["county_ablation_label"].nunique() > 1
    )
    has_multiple_region_settings = (
        "region_prior_label" in results_df.columns
        and results_df["region_prior_label"].nunique() > 1
    )
    if has_multiple_region_settings:
        plot_df = results_df.assign(
            region_prior_smoothing_sort=results_df["region_prior_smoothing"].fillna(-1.0)
        ).sort_values(
            [
                "path_name",
                "county_ablation_label",
                "region_prior_enabled",
                "region_prior_smoothing_sort",
            ]
        ).reset_index(drop=True)
        labels = []
        for _, row in plot_df.iterrows():
            suffix_parts: list[str] = []
            if has_multiple_county_modes:
                suffix_parts.append(f"county={row['county_ablation_label']}")
            suffix_parts.append(f"region={row['region_prior_label']}")
            labels.append(f"{row['path_name']}\n[" + ", ".join(suffix_parts) + "]")
    else:
        plot_df = results_df.sort_values(
            ["test_rmse", "high_pm25_rmse", "path_name", "county_ablation_label"]
        ).reset_index(drop=True)
        if has_multiple_county_modes:
            labels = [
                f"{path_name}\n[county={county_label}]"
                for path_name, county_label in zip(
                    plot_df["path_name"], plot_df["county_ablation_label"]
                )
            ]
        else:
            labels = plot_df["path_name"].tolist()

    x_positions = list(range(len(plot_df)))
    test_rmse = plot_df["test_rmse"].tolist()
    high_rmse = plot_df["high_pm25_rmse"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(plot_df) * 2.2), 5.8))

    bar_width = 0.38
    axes[0].bar(
        [pos - bar_width / 2 for pos in x_positions],
        test_rmse,
        width=bar_width,
        color="#2F6BFF",
        label="Overall RMSE",
    )
    axes[0].bar(
        [pos + bar_width / 2 for pos in x_positions],
        high_rmse,
        width=bar_width,
        color="#FF7A45",
        label="High-PM2.5 RMSE",
    )
    axes[0].set_title(
        "RMSE by Modeling Path"
        if not has_multiple_region_settings and not has_multiple_county_modes
        else "RMSE by Path / County Ablation / Region Prior"
    )
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend()

    rmse_gap = [high - overall for overall, high in zip(test_rmse, high_rmse)]
    gap_colors = ["#5B8C00" if gap <= 0 else "#BFBFBF" for gap in rmse_gap]
    axes[1].bar(x_positions, rmse_gap, color=gap_colors)
    axes[1].axhline(0.0, color="#595959", linewidth=1.0)
    axes[1].set_title(
        "High-PM2.5 vs Overall RMSE Gap"
        if not has_multiple_region_settings and not has_multiple_county_modes
        else "RMSE Gap by County Ablation / Region Prior"
    )
    axes[1].set_ylabel("High-PM2.5 RMSE - Overall RMSE")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    best_overall_idx = int(plot_df["test_rmse"].idxmin())
    best_high_idx = int(plot_df["high_pm25_rmse"].idxmin())
    axes[0].text(
        x=plot_df.index.get_loc(best_overall_idx) - bar_width / 2,
        y=plot_df.loc[best_overall_idx, "test_rmse"],
        s="  Best",
        va="bottom",
        ha="left",
        fontsize=9,
        color="#1D39C4",
    )
    axes[0].text(
        x=plot_df.index.get_loc(best_high_idx) + bar_width / 2,
        y=plot_df.loc[best_high_idx, "high_pm25_rmse"],
        s="  Best",
        va="bottom",
        ha="left",
        fontsize=9,
        color="#D4380D",
    )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.suptitle("PM2.5 Modeling Path Comparison", fontsize=14)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_xgb_path(
    x_train_aug_for_tune: pd.DataFrame,
    x_train_aug_for_fit: pd.DataFrame,
    x_test_aug: pd.DataFrame,
    residual_train_oof: pd.Series,
    groups_train,
    residual_sample_weight,
    args: argparse.Namespace,
    xgb_feature_view: str,
) -> dict[str, Any]:
    xgb_train_aug_for_tune, xgb_feature_view_info = prepare_xgb_residual_features(
        x=x_train_aug_for_tune,
        residual_model="ensemble",
        xgb_feature_view=xgb_feature_view,
        complementary_drop_cols=parse_feature_list(args.xgb_complementary_drop_cols),
        add_all_interactions=not args.no_xgb_all_interactions,
        add_complementary_interactions=not args.no_xgb_complementary_interactions,
    )
    xgb_train_aug_for_fit, _ = prepare_xgb_residual_features(
        x=x_train_aug_for_fit,
        residual_model="ensemble",
        xgb_feature_view=xgb_feature_view,
        complementary_drop_cols=parse_feature_list(args.xgb_complementary_drop_cols),
        add_all_interactions=not args.no_xgb_all_interactions,
        add_complementary_interactions=not args.no_xgb_complementary_interactions,
    )
    xgb_test_aug, _ = prepare_xgb_residual_features(
        x=x_test_aug,
        residual_model="ensemble",
        xgb_feature_view=xgb_feature_view,
        complementary_drop_cols=parse_feature_list(args.xgb_complementary_drop_cols),
        add_all_interactions=not args.no_xgb_all_interactions,
        add_complementary_interactions=not args.no_xgb_complementary_interactions,
    )

    xgb_best_params = None
    xgb_best_cv_rmse = None
    if args.no_tune:
        print(
            f"[信息] 已关闭 XGBoost({xgb_feature_view_info['effective_view']}) 调参，将使用默认参数训练。"
        )
    else:
        print(
            f"[信息] 开始 XGBoost({xgb_feature_view_info['effective_view']}) 调参："
            f"n_iter={args.n_iter}, residual_cv_folds={args.residual_cv_folds}"
        )
        xgb_tune_pipeline = build_residual_pipeline(
            xgb_train_aug_for_tune,
            args.random_state,
            args.device,
            args.gpu_id,
            xgb_feature_view_info["effective_view"],
        )
        xgb_best_params, xgb_best_cv_rmse = tune_residual_model(
            pipeline=xgb_tune_pipeline,
            x_train_aug=xgb_train_aug_for_tune,
            residual_target=residual_train_oof,
            groups_train=groups_train,
            sample_weight=residual_sample_weight,
            cv_folds=args.residual_cv_folds,
            n_iter=args.n_iter,
            random_state=args.random_state,
            device=args.device,
            validation_mode=args.validation_mode,
            xgb_effective_view=xgb_feature_view_info["effective_view"],
        )

    xgb_model = fit_final_residual_model(
        x_train_aug=xgb_train_aug_for_fit,
        residual_target=residual_train_oof,
        sample_weight=residual_sample_weight,
        random_state=args.random_state,
        device=args.device,
        gpu_id=args.gpu_id,
        xgb_effective_view=xgb_feature_view_info["effective_view"],
        best_params=xgb_best_params,
    )
    xgb_test_pred = xgb_model.predict(xgb_test_aug)
    xgb_oof_pred = generate_oof_xgboost_residual_predictions(
        x_train_aug=xgb_train_aug_for_tune,
        residual_target=residual_train_oof,
        groups_train=groups_train,
        sample_weight=residual_sample_weight,
        cv_folds=args.residual_cv_folds,
        random_state=args.random_state,
        device=args.device,
        gpu_id=args.gpu_id,
        xgb_effective_view=xgb_feature_view_info["effective_view"],
        best_params=xgb_best_params,
        validation_mode=args.validation_mode,
    )
    return {
        "feature_view_info": xgb_feature_view_info,
        "best_params": xgb_best_params,
        "best_cv_rmse": xgb_best_cv_rmse,
        "model": xgb_model,
        "test_pred": xgb_test_pred,
        "oof_pred": xgb_oof_pred,
    }


def evaluate_paths_for_region_prior_setting(
    args: argparse.Namespace,
    path_candidates: list[str],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    groups_train,
    trend_train_oof_pred: np.ndarray,
    trend_test_pred: np.ndarray,
    residual_train_oof: pd.Series,
    residual_sample_weight,
    high_pm25_eval_config: dict[str, float],
    region_prior_setting: dict[str, Any],
    county_setting_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """在给定区域先验配置下，训练并评估多条建模路径。"""
    region_prior_label = str(region_prior_setting["label"])
    x_train_aug_for_tune = make_augmented_features(x_train, trend_train_oof_pred)
    x_train_aug_for_fit = x_train_aug_for_tune.copy()
    x_test_aug = make_augmented_features(x_test, trend_test_pred)

    if not region_prior_setting["enabled"]:
        print(f"[信息] 区域层级先验设置：{region_prior_label}（关闭）")
    else:
        region_prior_cols = parse_feature_list(
            args.region_prior_cols,
            DEFAULT_REGION_PRIOR_COLS,
        )
        region_train_prior_df, region_test_prior_df, region_prior_config = (
            generate_region_prior_features(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                residual_train_oof=residual_train_oof,
                groups_train=groups_train,
                cv_folds=args.residual_cv_folds,
                random_state=args.random_state,
                validation_mode=args.validation_mode,
                region_cols=region_prior_cols,
                smoothing=float(region_prior_setting["smoothing"]),
            )
        )
        if region_prior_config["enabled"]:
            x_train_aug_for_tune = append_region_prior_features(
                x_train_aug_for_tune,
                region_train_prior_df,
            )
            x_train_aug_for_fit = append_region_prior_features(
                x_train_aug_for_fit,
                region_train_prior_df,
            )
            x_test_aug = append_region_prior_features(
                x_test_aug,
                region_test_prior_df,
            )
            print(
                "[信息] 区域层级先验设置："
                f"{region_prior_label} -> cols={region_prior_config['region_cols']}, "
                f"smoothing={region_prior_config['smoothing']:.2f}, "
                f"features={len(region_prior_config['feature_names'])}"
            )
        else:
            print(
                "[信息] 区域层级先验设置："
                f"{region_prior_label} -> 未生成特征（未找到可用地区列）"
            )

    path_results: list[dict[str, Any]] = []
    xgb_bundles: dict[str, dict[str, Any]] = {}
    need_xgb_all = any("xgb_all" in path or path.endswith("_all") for path in path_candidates)
    need_xgb_comp = any("complementary" in path for path in path_candidates)

    if need_xgb_all:
        print("\n[信息] 训练路径：trend + XGBoost(all)")
        try:
            xgb_bundles["all"] = train_xgb_path(
                x_train_aug_for_tune=x_train_aug_for_tune,
                x_train_aug_for_fit=x_train_aug_for_fit,
                x_test_aug=x_test_aug,
                residual_train_oof=residual_train_oof,
                groups_train=groups_train,
                residual_sample_weight=residual_sample_weight,
                args=args,
                xgb_feature_view="all",
            )
        except XGBoostError as exc:
            if args.device == "gpu":
                raise SystemExit(
                    "XGBoost GPU 训练失败。请确认 CUDA 驱动/显卡可用，或改用 --device cpu。\n"
                    f"原始错误：{exc}"
                ) from exc
            raise

    if need_xgb_comp:
        print("\n[信息] 训练路径：trend + XGBoost(complementary)")
        try:
            xgb_bundles["complementary"] = train_xgb_path(
                x_train_aug_for_tune=x_train_aug_for_tune,
                x_train_aug_for_fit=x_train_aug_for_fit,
                x_test_aug=x_test_aug,
                residual_train_oof=residual_train_oof,
                groups_train=groups_train,
                residual_sample_weight=residual_sample_weight,
                args=args,
                xgb_feature_view="complementary",
            )
        except XGBoostError as exc:
            if args.device == "gpu":
                raise SystemExit(
                    "XGBoost GPU 训练失败。请确认 CUDA 驱动/显卡可用，或改用 --device cpu。\n"
                    f"原始错误：{exc}"
                ) from exc
            raise

    need_cat = any("catboost" in path or "ensemble" in path for path in path_candidates)
    cat_bundle: dict[str, Any] | None = None
    cat_expert_bundle: dict[str, Any] | None = None
    if need_cat:
        print("\n[信息] 训练路径：trend + CatBoost")
        cat_best_params = None
        cat_best_cv_rmse = None
        try:
            if args.no_tune:
                print("[信息] 已关闭 CatBoost 调参，将使用默认参数训练。")
            else:
                print(
                    f"[信息] 开始 CatBoost 调参：n_iter={args.n_iter}, "
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
                cat_model,
                cat_categorical_cols,
                cat_feature_cols,
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
            cat_test_pred = predict_catboost_residual_model(
                model=cat_model,
                x_aug=x_test_aug,
                feature_cols=cat_feature_cols,
                categorical_cols=cat_categorical_cols,
            )
            cat_oof_pred = generate_oof_catboost_residual_predictions(
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
            cat_bundle = {
                "best_params": cat_best_params,
                "best_cv_rmse": cat_best_cv_rmse,
                "model": cat_model,
                "feature_cols": cat_feature_cols,
                "categorical_cols": cat_categorical_cols,
                "test_pred": cat_test_pred,
                "oof_pred": cat_oof_pred,
            }

            if "catboost_experts" in path_candidates:
                print("\n[信息] 训练路径：trend + CatBoost regime experts")
                cat_expert_bundle = fit_catboost_regime_expert_bundle(
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
                    regime_mode=args.catboost_expert_regime,
                    trend_quantile=args.catboost_expert_trend_quantile,
                    low_wind_quantile=args.catboost_expert_low_wind_quantile,
                    special_weight=args.catboost_expert_special_weight,
                    min_samples=args.catboost_expert_min_samples,
                    general_model=cat_model,
                    general_feature_cols=cat_feature_cols,
                    general_categorical_cols=cat_categorical_cols,
                )
        except CatBoostError as exc:
            if args.device == "gpu":
                raise SystemExit(
                    "CatBoost GPU 训练失败。请确认 CUDA 驱动/显卡可用，或改用 --device cpu。\n"
                    f"原始错误：{exc}"
                ) from exc
            raise

    for path_name in path_candidates:
        if path_name == "xgb_all":
            bundle = xgb_bundles["all"]
            final_pred = trend_test_pred + bundle["test_pred"]
            path_results.append(
                build_result_row(
                    path_name=path_name,
                    path_type="single",
                    xgb_view="all",
                    objective="single",
                    ensemble_config={
                        "strategy": "single_model",
                        "xgb_weight": 1.0,
                        "catboost_weight": 0.0,
                    },
                    y_test=y_test,
                    final_test_pred=final_pred,
                    high_pm25_threshold=high_pm25_eval_config["threshold"],
                    region_prior_enabled=bool(region_prior_setting["enabled"]),
                    region_prior_smoothing=region_prior_setting["smoothing"],
                    region_prior_label=region_prior_label,
                    county_ablation_label=county_setting_info["label"],
                    county_removed_cols=county_setting_info["removed_cols"],
                )
            )
        elif path_name == "xgb_complementary":
            bundle = xgb_bundles["complementary"]
            final_pred = trend_test_pred + bundle["test_pred"]
            path_results.append(
                build_result_row(
                    path_name=path_name,
                    path_type="single",
                    xgb_view="complementary",
                    objective="single",
                    ensemble_config={
                        "strategy": "single_model",
                        "xgb_weight": 1.0,
                        "catboost_weight": 0.0,
                    },
                    y_test=y_test,
                    final_test_pred=final_pred,
                    high_pm25_threshold=high_pm25_eval_config["threshold"],
                    region_prior_enabled=bool(region_prior_setting["enabled"]),
                    region_prior_smoothing=region_prior_setting["smoothing"],
                    region_prior_label=region_prior_label,
                    county_ablation_label=county_setting_info["label"],
                    county_removed_cols=county_setting_info["removed_cols"],
                )
            )
        elif path_name == "catboost":
            if cat_bundle is None:
                raise RuntimeError("catboost 路径需要 CatBoost 结果。")
            final_pred = trend_test_pred + cat_bundle["test_pred"]
            path_results.append(
                build_result_row(
                    path_name=path_name,
                    path_type="single",
                    xgb_view="none",
                    objective="single",
                    ensemble_config={
                        "strategy": "single_model",
                        "xgb_weight": 0.0,
                        "catboost_weight": 1.0,
                    },
                    y_test=y_test,
                    final_test_pred=final_pred,
                    high_pm25_threshold=high_pm25_eval_config["threshold"],
                    region_prior_enabled=bool(region_prior_setting["enabled"]),
                    region_prior_smoothing=region_prior_setting["smoothing"],
                    region_prior_label=region_prior_label,
                    county_ablation_label=county_setting_info["label"],
                    county_removed_cols=county_setting_info["removed_cols"],
                )
            )
        elif path_name == "catboost_experts":
            if cat_expert_bundle is None:
                raise RuntimeError("catboost_experts 路径需要 CatBoost 专家模型结果。")
            cat_expert_pred_bundle = predict_catboost_regime_expert_bundle(
                expert_bundle=cat_expert_bundle,
                x_aug=x_test_aug,
            )
            final_pred = trend_test_pred + cat_expert_pred_bundle["final_pred"]
            expert_regime_config = dict(cat_expert_bundle["regime_config"])
            expert_regime_config["special_test_count"] = int(
                cat_expert_pred_bundle["special_count"]
            )
            path_results.append(
                build_result_row(
                    path_name=path_name,
                    path_type="expert",
                    xgb_view="none",
                    objective="single",
                    ensemble_config={
                        "strategy": "single_model",
                        "xgb_weight": 0.0,
                        "catboost_weight": 1.0,
                    },
                    y_test=y_test,
                    final_test_pred=final_pred,
                    high_pm25_threshold=high_pm25_eval_config["threshold"],
                    region_prior_enabled=bool(region_prior_setting["enabled"]),
                    region_prior_smoothing=region_prior_setting["smoothing"],
                    region_prior_label=region_prior_label,
                    county_ablation_label=county_setting_info["label"],
                    county_removed_cols=county_setting_info["removed_cols"],
                    expert_config=expert_regime_config,
                )
            )
        elif path_name.startswith("ensemble_"):
            if cat_bundle is None:
                raise RuntimeError(f"{path_name} 需要 CatBoost 结果。")
            if path_name.endswith("_all"):
                bundle = xgb_bundles["all"]
                xgb_view = "all"
            else:
                bundle = xgb_bundles["complementary"]
                xgb_view = "complementary"

            objective = "overall" if "overall" in path_name else "blended"
            ensemble_config = resolve_ensemble_weights(
                residual_model="ensemble",
                manual_weight_xgb=None,
                ensemble_weight_objective=objective,
                ensemble_weight_grid_step=args.ensemble_weight_grid_step,
                ensemble_blended_overall_weight=args.ensemble_blended_overall_weight,
                y_train=y_train,
                trend_train_oof_pred=trend_train_oof_pred,
                xgb_residual_oof_pred=bundle["oof_pred"],
                cat_residual_oof_pred=cat_bundle["oof_pred"],
                high_pm25_threshold=high_pm25_eval_config["threshold"],
                xgb_cv_rmse=bundle["best_cv_rmse"],
                cat_cv_rmse=cat_bundle["best_cv_rmse"],
            )
            final_pred = trend_test_pred + (
                ensemble_config["xgb_weight"] * bundle["test_pred"]
                + ensemble_config["catboost_weight"] * cat_bundle["test_pred"]
            )
            path_results.append(
                build_result_row(
                    path_name=path_name,
                    path_type="ensemble",
                    xgb_view=xgb_view,
                    objective=objective,
                    ensemble_config=ensemble_config,
                    y_test=y_test,
                    final_test_pred=final_pred,
                    high_pm25_threshold=high_pm25_eval_config["threshold"],
                    region_prior_enabled=bool(region_prior_setting["enabled"]),
                    region_prior_smoothing=region_prior_setting["smoothing"],
                    region_prior_label=region_prior_label,
                    county_ablation_label=county_setting_info["label"],
                    county_removed_cols=county_setting_info["removed_cols"],
                )
            )
        else:
            raise ValueError(f"不支持的路径名称：{path_name}")

    return path_results


def main() -> None:
    args = parse_args()
    run_name = build_default_run_name("compare_rmse_paths", args.run_name)
    out_path = resolve_output_path(args.out, run_name)
    plot_path = resolve_plot_path(args.plot_out, run_name)
    path_candidates = parse_path_candidates(args.paths)
    county_ablation_settings = parse_county_ablation_candidates(
        args.county_ablation_modes
    )
    region_prior_settings = parse_region_prior_setting_candidates(
        raw_text=args.region_prior_smoothing_values,
        default_smoothing=args.region_prior_smoothing,
        no_region_priors=args.no_region_priors,
    )

    x, y = load_data(Path(args.data), args.target)
    if not args.keep_id_cols:
        x, dropped_cols = remove_identifier_columns(x)
        if dropped_cols:
            print(f"[信息] 已自动剔除疑似 ID 列：{dropped_cols}")

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

    print(f"[信息] 当前运行目录：{run_name}")
    print(f"[信息] 当前训练设备：{args.device.upper()}")
    print(f"[信息] 当前验证方式：{args.validation_mode}")
    print(f"[信息] 当前比较路径：{path_candidates}")
    print(
        "[信息] 当前 COUNTY 消融设置："
        f"{[setting['label'] for setting in county_ablation_settings]}"
    )
    print(
        "[信息] 当前区域层级先验设置："
        f"{[setting['label'] for setting in region_prior_settings]}"
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

    path_results: list[dict[str, Any]] = []
    for county_setting in county_ablation_settings:
        print("\n" + "=" * 72)
        x_train_mode, x_test_mode, county_setting_info = apply_county_ablation(
            x_train=x_train,
            x_test=x_test,
            county_col=args.county_col,
            county_setting=county_setting,
        )
        print(
            "[信息] 当前 COUNTY 消融模式："
            f"{county_setting_info['label']}, "
            f"removed_cols={county_setting_info['removed_cols']}"
        )
        print(
            f"[信息] 开始生成线性趋势 OOF 预测：trend_cv_folds={args.trend_cv_folds}"
        )
        trend_train_oof_pred = generate_oof_trend_predictions(
            x_train=x_train_mode,
            y_train=y_train,
            groups_train=groups_train,
            cv_folds=args.trend_cv_folds,
            random_state=args.random_state,
            validation_mode=args.validation_mode,
            trend_model=args.trend_model,
            weight_config=sample_weight_config,
        )
        trend_model = fit_final_trend_model(
            x_train=x_train_mode,
            y_train=y_train,
            trend_model=args.trend_model,
            sample_weight=trend_sample_weight,
        )
        trend_test_pred = trend_model.predict(x_test_mode)
        residual_train_oof = y_train - trend_train_oof_pred

        for region_setting in region_prior_settings:
            print("\n" + "-" * 72)
            path_results.extend(
                evaluate_paths_for_region_prior_setting(
                    args=args,
                    path_candidates=path_candidates,
                    x_train=x_train_mode,
                    x_test=x_test_mode,
                    y_train=y_train,
                    y_test=y_test,
                    groups_train=groups_train,
                    trend_train_oof_pred=trend_train_oof_pred,
                    trend_test_pred=trend_test_pred,
                    residual_train_oof=residual_train_oof,
                    residual_sample_weight=residual_sample_weight,
                    high_pm25_eval_config=high_pm25_eval_config,
                    region_prior_setting=region_setting,
                    county_setting_info=county_setting_info,
                )
            )

    results_df = pd.DataFrame(path_results)
    results_df["test_rmse_rank"] = results_df["test_rmse"].rank(method="min")
    results_df["high_pm25_rmse_rank"] = results_df["high_pm25_rmse"].rank(method="min")
    results_df["combined_rank_sum"] = (
        results_df["test_rmse_rank"] + results_df["high_pm25_rmse_rank"]
    )
    results_df["test_rmse_rank_within_path"] = results_df.groupby("path_name")[
        "test_rmse"
    ].rank(method="min")
    results_df["high_pm25_rmse_rank_within_path"] = results_df.groupby("path_name")[
        "high_pm25_rmse"
    ].rank(method="min")

    has_multiple_region_settings = results_df["region_prior_label"].nunique() > 1
    if has_multiple_region_settings:
        results_df = results_df.assign(
            region_prior_smoothing_sort=results_df["region_prior_smoothing"].fillna(-1.0)
        ).sort_values(
            [
                "path_name",
                "county_ablation_label",
                "test_rmse",
                "high_pm25_rmse",
                "region_prior_enabled",
                "region_prior_smoothing_sort",
            ]
        ).drop(columns=["region_prior_smoothing_sort"]).reset_index(drop=True)
    else:
        results_df = results_df.sort_values(
            ["test_rmse", "high_pm25_rmse", "path_name"]
        ).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    if not args.no_plot:
        plot_path_comparison_chart(results_df, plot_path)

    print("\n=== 路径对比结果（按 test_rmse 排序） ===")
    display_cols = [
        "path_name",
        "county_ablation_label",
        "region_prior_label",
        "region_prior_smoothing",
        "path_type",
        "xgb_view",
        "objective",
        "xgb_weight",
        "catboost_weight",
        "test_rmse",
        "high_pm25_rmse",
        "test_rmse_rank_within_path",
        "test_mae",
        "high_pm25_mae",
        "test_r2",
        "high_pm25_r2",
    ]
    if "catboost_experts" in set(results_df["path_name"]):
        expert_display_cols = [
            "expert_regime_effective",
            "expert_special_enabled",
            "expert_special_train_count",
            "expert_special_test_count",
            "expert_special_weight",
        ]
        insert_at = display_cols.index("test_rmse")
        display_cols[insert_at:insert_at] = expert_display_cols
    print(results_df[display_cols].to_string(index=False))
    print(f"\n对比结果已保存到：{out_path}")
    if not args.no_plot:
        print(f"对比图已保存到：{plot_path}")


if __name__ == "__main__":
    main()
