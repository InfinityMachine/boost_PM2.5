# PM2.5 Air Quality Regression Project

This repository contains machine learning scripts for predicting PM2.5 air quality levels using various regression approaches including XGBoost, CatBoost, and ensemble methods with trend-residual decomposition.

## Running Scripts

All training scripts are standalone Python files that can be run directly:

```bash
# Basic XGBoost model
python train_pm25_xgboost.py --data data.csv --target PM2.5

# CatBoost model
python train_pm25_catboost.py --data data.csv --target PM2.5

# Trend + Residual model (primary approach)
python train_pm25_trend_residual_xgboost.py --data data.csv --target PM2.5

# Compare multiple modeling paths
python compare_rmse_paths.py --data data.csv

# Sweep ensemble weights
python sweep_blended_weights.py --data data.csv
```

### Common Command-Line Arguments

All training scripts share these common arguments:
- `--data`: Path to training CSV file (default: `data.csv`)
- `--target`: Target column name (default: `PM2.5`)
- `--test-size`: Test set proportion (default: `0.2`)
- `--random-state`: Random seed (default: `42`)
- `--device`: Training device, either `cpu` or `gpu` (default: `cpu`)
- `--no-tune`: Skip hyperparameter tuning, use defaults only (faster)
- `--no-save`: Don't save trained models
- `--keep-id-cols`: Preserve ID columns (by default they are automatically removed)

### GPU Training

To enable GPU acceleration:

```bash
# XGBoost on GPU
python train_pm25_xgboost.py --device gpu --gpu-id 0

# CatBoost on GPU
python train_pm25_catboost.py --device gpu --gpu-devices 0

# Trend-residual on GPU
python train_pm25_trend_residual_xgboost.py --device gpu --gpu-id 0
```

## Architecture Overview

### Three Modeling Approaches

1. **Direct XGBoost** (`train_pm25_xgboost.py`)
   - Single XGBoost model directly predicting PM2.5
   - Pipeline includes: preprocessing → hyperparameter tuning → model training
   - Saves to `cache/pm25_xgboost_pipeline.joblib`

2. **Direct CatBoost** (`train_pm25_catboost.py`)
   - Single CatBoost model directly predicting PM2.5
   - Native categorical feature handling (no manual one-hot encoding needed)
   - Saves to `cache/pm25_catboost_model.cbm` + metadata JSON

3. **Trend-Residual Decomposition** (`train_pm25_trend_residual_xgboost.py`) **[Primary Approach]**
   - Two-stage modeling: Linear trend model + Tree-based residual model
   - Stage 1: Linear model (LinearRegression or RidgeCV) learns global trends
   - Stage 2: Tree model (XGBoost, CatBoost, or ensemble) corrects residuals
   - Final prediction = trend_prediction + residual_prediction
   - Supports three residual modes:
     - `xgb`: XGBoost residual model
     - `catboost`: CatBoost residual model
     - `ensemble`: Weighted combination of XGBoost + CatBoost residuals
   - Saves to `cache/pm25_trend_residual_*.joblib`

### Trend-Residual Feature Engineering

The trend-residual approach includes specialized feature engineering:

- **Region Prior Features**: Smoothed regional statistics (mean, std, count) computed from PROVINCE/CITY columns using target encoding with smoothing to prevent overfitting
- **Log-Transformed Features**: Automatic log(1+x) transforms for specified features (NOX, SO2, fertilizer, manure by default)
- **Interaction Features**: Engineered features like NOX×SO2, NOX×wind, temperature×precipitation
- **Trend Predictions as Features**: The linear trend prediction is passed as `__linear_trend_pred__` to the residual model

### Data Split Strategies

- **Random Split** (default): Standard random train/test split
- **Group-Based Split** (`--group-col`): Ensures all samples from the same group stay together (e.g., split by COUNTY to prevent data leakage)

### Cross-Validation and Tuning

- **RandomizedSearchCV** (XGBoost): K-fold cross-validation with random hyperparameter search
- **Manual CV Loop** (CatBoost): Custom K-fold loop with early stopping per fold
- **Out-of-Fold (OOF) Predictions**: Used for ensemble weight optimization to prevent overfitting

## Key Conventions

### File Organization

- **Training Scripts**: All `train_*.py` files in root directory
- **Experimental Scripts**: `compare_*.py` and `sweep_*.py` for model comparison experiments
- **Model Artifacts**: Saved to `cache/` directory (gitignored)
- **Diagnostic Plots**: Saved to `.diag/` directory (gitignored)
- **Data**: Input CSV expected at `data.csv` by default

### Code Style

- **Chinese Comments**: All docstrings and major comments are in Chinese
- **Type Hints**: All functions use type hints (`from __future__ import annotations`)
- **Argument Parsing**: Every script uses `argparse` for CLI parameters
- **Error Handling**: Dependency imports wrapped in try-except with helpful error messages
- **Pipeline Pattern**: Use scikit-learn Pipeline for preprocessing + model (easier serialization)

### Naming Conventions

- **Scripts**: `train_pm25_<method>.py` for training scripts
- **Model Files**: `pm25_<method>_pipeline.joblib` or `.cbm` for CatBoost
- **Diagnostic Plots**: `pm25_<method>_diagnostics.png`
- **Functions**: Snake_case with descriptive names (e.g., `build_residual_pipeline`, `tune_hyperparameters`)
- **Constants**: UPPER_SNAKE_CASE for defaults (e.g., `DEFAULT_RIDGE_ALPHAS`, `TREND_FEATURE_NAME`)

### ID Column Handling

By default, all scripts automatically detect and remove ID columns (columns with "id" in the name, case-insensitive) to prevent overfitting. This includes columns like `OBJECTID_1`, `sample_id`, etc. Use `--keep-id-cols` to preserve them.

### Model Saving

- **XGBoost Models**: Saved as scikit-learn Pipeline objects using `joblib`
- **CatBoost Models**: Saved using native `.save_model()` with separate JSON metadata
- **Ensemble Models**: Include all sub-models and weight information in saved artifacts

### Diagnostic Outputs

All training scripts generate:
1. **Metrics**: RMSE, MAE, R² on test set (and CV RMSE during tuning)
2. **Sample Predictions**: First 10 test set predictions printed to console
3. **Feature Importance**: Top 20 most important features
4. **Diagnostic Plots** (unless `--no-plot`):
   - Left: Actual vs Predicted curves (sorted by actual value)
   - Right: Residual distribution histogram

### High Pollution Evaluation

The trend-residual script supports evaluating performance specifically on high-pollution samples:
- Use `--high-pm25-threshold` to set a threshold (e.g., `75`)
- Reports separate RMSE, MAE, R² for samples above threshold
- Useful for assessing model performance on critical air quality events

## Dependencies

Required packages (install via pip):
```bash
pip install joblib scikit-learn xgboost catboost pandas numpy matplotlib
```

For GPU support, ensure CUDA-compatible versions:
```bash
pip install xgboost[gpu] catboost[gpu]
```

## Ensemble Weight Strategies

When using `--residual-model ensemble` in the trend-residual script:

1. **Equal Weights** (`--ensemble-weights equal`): 50/50 XGBoost and CatBoost
2. **Overall Optimization** (`--ensemble-weights overall`): Optimize for lowest overall test RMSE
3. **High-PM2.5 Optimization** (`--ensemble-weights high_pm25`): Optimize for lowest RMSE on high-pollution subset
4. **Blended Optimization** (`--ensemble-weights blended`): Joint objective balancing overall and high-pollution RMSE (controlled by `--blended-overall-weight`)

## Common Pitfall: Feature View Selection

In the trend-residual XGBoost path, use `--xgb-feature-view` to control which features the residual model sees:

- `all` (default): Use all original features + engineered features + trend prediction
- `complementary`: Drop CITY/COUNTY columns (already captured by region priors), use NOX, SO2, meteorological vars, and engineered features

The `complementary` view often performs better by reducing redundancy and focusing on physically meaningful predictors.
