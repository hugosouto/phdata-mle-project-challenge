"""Train three models for the Sound Realty enhanced API.

1. Basic model     — KNN (original): 7 house features + demographics
2. Enhanced model  — LightGBM: 10 best features + demographics
3. Optimized model — LightGBM: 10 best features only (no demographics)

Outputs:
  model/basic_model.pkl         + model/basic_model_features.json
  model/enhanced_model.pkl      + model/enhanced_model_features.json
  model/optimized_model.pkl     + model/optimized_model_features.json
"""

import json
import pathlib
import pickle
from typing import List

import lightgbm as lgb
import numpy as np
import pandas
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
OUTPUT_DIR = "model"

# Original 7 house features used by the basic KNN model
BASIC_HOUSE_COLS = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement',
]

# 10 best features for LightGBM models (zipcode kept as integer feature)
OPTIMIZED_HOUSE_COLS = [
    'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement',
    'grade', 'lat', 'long', 'bathrooms', 'waterfront', 'zipcode',
]


def load_with_demographics(sales_path: str, demographics_path: str, house_cols: List[str]):
    """Load sales data, merge demographics, drop zipcode. Used by KNN baseline."""
    cols_to_load = list(set(['price', 'zipcode'] + house_cols))
    data = pandas.read_csv(sales_path, usecols=cols_to_load, dtype={'zipcode': str})
    demographics = pandas.read_csv(demographics_path, dtype={'zipcode': str})

    merged = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    y = merged.pop('price')
    return merged, y


def load_without_demographics(sales_path: str, house_cols: List[str]):
    """Load sales data with only house features + zipcode as int. No demographics join."""
    cols_to_load = list(set(['price', 'zipcode'] + house_cols))
    data = pandas.read_csv(sales_path, usecols=cols_to_load, dtype={'zipcode': str})
    data['zipcode'] = data['zipcode'].astype(int)
    y = data.pop('price')
    return data, y


def train_basic(x_train, y_train):
    """Train the original KNN pipeline (RobustScaler + KNeighborsRegressor)."""
    model = make_pipeline(RobustScaler(), KNeighborsRegressor())
    model.fit(x_train, y_train)
    return model


def train_lightgbm(x_train, y_train):
    """Train a LightGBM regressor with tuned hyperparameters."""
    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    return model


def save_model(model, features, output_dir, name, metrics=None):
    """Save model pickle, feature list JSON, and optional metrics JSON."""
    pickle.dump(model, open(output_dir / f"{name}.pkl", 'wb'))
    json.dump(list(features), open(output_dir / f"{name}_features.json", 'w'))
    if metrics is not None:
        json.dump(metrics, open(output_dir / f"{name}_metrics.json", 'w'))


def evaluate(model, x_test, y_test, label):
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    print(f"  {label}: R²={r2:.4f}  MAE=${mae:,.0f}  MAPE={mape:.2f}%  RMSE=${rmse:,.0f}")
    return {"r2_score": round(r2, 4), "mae": round(mae, 2), "mape": round(mape, 2), "rmse": round(rmse, 2)}


def main():
    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # --- Basic model (KNN with 7 features + demographics) ---
    print("Training basic model (KNN)...")
    x_basic, y_basic = load_with_demographics(SALES_PATH, DEMOGRAPHICS_PATH, BASIC_HOUSE_COLS)
    xb_train, xb_test, yb_train, yb_test = train_test_split(
        x_basic, y_basic, random_state=42
    )
    basic_model = train_basic(xb_train, yb_train)
    basic_metrics = evaluate(basic_model, xb_test, yb_test, "Basic KNN (test)")
    save_model(basic_model, xb_train.columns, output_dir, "basic_model", basic_metrics)
    print(f"  Saved: {output_dir}/basic_model.pkl ({len(xb_train.columns)} features)\n")

    # --- Enhanced model (LightGBM with 10 features + demographics) ---
    print("Training enhanced model (LightGBM + demographics)...")
    x_enh, y_enh = load_with_demographics(SALES_PATH, DEMOGRAPHICS_PATH, OPTIMIZED_HOUSE_COLS)
    # Keep zipcode as int in the merged result
    # Re-load with demographics but keep zipcode
    cols_to_load = list(set(['price', 'zipcode'] + OPTIMIZED_HOUSE_COLS))
    data_enh = pandas.read_csv(SALES_PATH, usecols=cols_to_load, dtype={'zipcode': str})
    demographics = pandas.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})
    merged_enh = data_enh.merge(demographics, how="left", on="zipcode")
    merged_enh['zipcode'] = merged_enh['zipcode'].astype(int)
    y_enh = merged_enh.pop('price')
    x_enh = merged_enh

    xe_train, xe_test, ye_train, ye_test = train_test_split(
        x_enh, y_enh, random_state=42
    )
    enhanced_model = train_lightgbm(xe_train, ye_train)
    enhanced_metrics = evaluate(enhanced_model, xe_test, ye_test, "Enhanced LightGBM (test)")
    save_model(enhanced_model, xe_train.columns, output_dir, "enhanced_model", enhanced_metrics)
    print(f"  Saved: {output_dir}/enhanced_model.pkl ({len(xe_train.columns)} features)\n")

    # --- Optimized model (LightGBM with 10 features only, NO demographics) ---
    print("Training optimized model (LightGBM, no demographics)...")
    x_opt, y_opt = load_without_demographics(SALES_PATH, OPTIMIZED_HOUSE_COLS)
    xo_train, xo_test, yo_train, yo_test = train_test_split(
        x_opt, y_opt, random_state=42
    )
    optimized_model = train_lightgbm(xo_train, yo_train)
    optimized_metrics = evaluate(optimized_model, xo_test, yo_test, "Optimized LightGBM (test)")
    save_model(optimized_model, xo_train.columns, output_dir, "optimized_model", optimized_metrics)
    print(f"  Saved: {output_dir}/optimized_model.pkl ({len(xo_train.columns)} features)\n")

    # Legacy model.pkl for backwards compat with test_api.py
    save_model(basic_model, xb_train.columns, output_dir, "model")
    print("Legacy model.pkl saved (alias for basic_model).")


if __name__ == "__main__":
    main()