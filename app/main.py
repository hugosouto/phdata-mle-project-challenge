"""Sound Realty housing price prediction API (Enhanced).

Four inference endpoints backed by three models:
  /predict/basic     — KNN model, all 18 input columns (same as baseline-api)
  /predict/minimal   — KNN model, only the 8 columns it actually needs
  /predict/enhanced  — LightGBM model, 10 best features + demographics
  /predict/optimized — LightGBM model, 10 best features only (no demographics)

Plus monitoring dashboard, retraining module, and health/metrics endpoints.
"""

import io
import json
import logging
import pickle
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
BASIC_MODEL_PATH = BASE_DIR / "model" / "basic_model.pkl"
BASIC_FEATURES_PATH = BASE_DIR / "model" / "basic_model_features.json"
BASIC_METRICS_PATH = BASE_DIR / "model" / "basic_model_metrics.json"
ENHANCED_MODEL_PATH = BASE_DIR / "model" / "enhanced_model.pkl"
ENHANCED_FEATURES_PATH = BASE_DIR / "model" / "enhanced_model_features.json"
ENHANCED_METRICS_PATH = BASE_DIR / "model" / "enhanced_model_metrics.json"
OPTIMIZED_MODEL_PATH = BASE_DIR / "model" / "optimized_model.pkl"
OPTIMIZED_FEATURES_PATH = BASE_DIR / "model" / "optimized_model_features.json"
OPTIMIZED_METRICS_PATH = BASE_DIR / "model" / "optimized_model_metrics.json"
DEMOGRAPHICS_PATH = BASE_DIR / "data" / "zipcode_demographics.csv"
UNSEEN_EXAMPLES_PATH = BASE_DIR / "data" / "future_unseen_examples.csv"

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class BasicHouseFeatures(BaseModel):
    """Full input schema matching future_unseen_examples.csv (18 columns)."""
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: str
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "bedrooms": 4, "bathrooms": 1.0, "sqft_living": 1680,
                "sqft_lot": 5043, "floors": 1.5, "waterfront": 0, "view": 0,
                "condition": 4, "grade": 6, "sqft_above": 1680,
                "sqft_basement": 0, "yr_built": 1911, "yr_renovated": 0,
                "zipcode": "98118", "lat": 47.5354, "long": -122.273,
                "sqft_living15": 1560, "sqft_lot15": 5765,
            }]
        }
    }


class MinimalHouseFeatures(BaseModel):
    """Only the 7 features the KNN model needs + zipcode for demographics join."""
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    zipcode: str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "bedrooms": 4, "bathrooms": 1.0, "sqft_living": 1680,
                "sqft_lot": 5043, "floors": 1.5, "sqft_above": 1680,
                "sqft_basement": 0, "zipcode": "98118",
            }]
        }
    }


class EnhancedHouseFeatures(BaseModel):
    """10 best features for LightGBM + demographics."""
    sqft_living: int
    sqft_lot: int
    sqft_above: int
    sqft_basement: int
    grade: int
    lat: float
    long: float
    bathrooms: float
    waterfront: int
    zipcode: str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "sqft_living": 1680, "sqft_lot": 5043, "sqft_above": 1680,
                "sqft_basement": 0, "grade": 6, "lat": 47.5354,
                "long": -122.273, "bathrooms": 1.0, "waterfront": 0,
                "zipcode": "98118",
            }]
        }
    }


# Optimized uses the same input schema as Enhanced (same 10 fields)
OptimizedHouseFeatures = EnhancedHouseFeatures


class PredictionResponse(BaseModel):
    predicted_price: float
    model_type: str
    model_version: str
    provider: str
    r2_score: float
    mae: float
    mape: float
    rmse: float
    n_features: int
    uses_demographics: bool
    warning: Optional[str] = None


class RetrainResponse(BaseModel):
    status: str
    candidate_model: dict
    reference_metrics: dict
    improved: bool
    auto_deployed: bool
    message: str


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


def _load_artifact(path, loader, label):
    try:
        result = loader(path)
        logger.info("%s loaded from %s", label, path)
        return result
    except Exception as exc:
        logger.error("FATAL: Could not load %s — %s", path, exc)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models, feature orders, and demographics at startup."""
    app.state.basic_model = _load_artifact(
        BASIC_MODEL_PATH, lambda p: pickle.load(open(p, "rb")), "Basic model"
    )
    app.state.basic_features = _load_artifact(
        BASIC_FEATURES_PATH, lambda p: json.load(open(p)), "Basic features"
    )
    app.state.enhanced_model = _load_artifact(
        ENHANCED_MODEL_PATH, lambda p: pickle.load(open(p, "rb")), "Enhanced model"
    )
    app.state.enhanced_features = _load_artifact(
        ENHANCED_FEATURES_PATH, lambda p: json.load(open(p)), "Enhanced features"
    )
    app.state.optimized_model = _load_artifact(
        OPTIMIZED_MODEL_PATH, lambda p: pickle.load(open(p, "rb")), "Optimized model"
    )
    app.state.optimized_features = _load_artifact(
        OPTIMIZED_FEATURES_PATH, lambda p: json.load(open(p)), "Optimized features"
    )
    app.state.demographics = _load_artifact(
        DEMOGRAPHICS_PATH,
        lambda p: pd.read_csv(p, dtype={"zipcode": str}),
        "Demographics",
    )

    app.state.model_version = "2.0.0"
    app.state.retrain_history = []

    # Load model performance metrics from training evaluation files
    def _load_metrics(path, n_features, uses_demographics):
        m = _load_artifact(path, lambda p: json.load(open(p)), "Metrics")
        m["n_features"] = n_features
        m["uses_demographics"] = uses_demographics
        return m

    app.state.model_metrics = {
        "basic": _load_metrics(BASIC_METRICS_PATH, len(app.state.basic_features), True),
        "enhanced": _load_metrics(ENHANCED_METRICS_PATH, len(app.state.enhanced_features), True),
        "optimized": _load_metrics(OPTIMIZED_METRICS_PATH, len(app.state.optimized_features), False),
    }

    yield


app = FastAPI(
    title="Sound Realty Price Prediction API (Enhanced)",
    description=(
        "Four prediction endpoints: /predict/basic (KNN, all inputs), "
        "/predict/minimal (KNN, subset), /predict/enhanced (LightGBM + demographics), "
        "/predict/optimized (LightGBM, 10 features only). "
        "Plus monitoring dashboard and retraining module."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Metrics middleware
# ---------------------------------------------------------------------------
from app.metrics import metrics_middleware, store  # noqa: E402

app.middleware("http")(metrics_middleware)

DASHBOARD_HTML_PATH = Path(__file__).resolve().parent / "templates" / "dashboard.html"
RETRAIN_HTML_PATH = Path(__file__).resolve().parent / "templates" / "retrain.html"

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

BASIC_HOUSE_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "sqft_above", "sqft_basement",
]

# 10 best features (used by both enhanced and optimized, with/without demographics)
BEST_HOUSE_FEATURES = [
    "sqft_living", "sqft_lot", "sqft_above", "sqft_basement",
    "grade", "lat", "long", "bathrooms", "waterfront", "zipcode",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_zipcode(zipcode: str, demographics_df: pd.DataFrame):
    if zipcode not in demographics_df["zipcode"].values:
        raise HTTPException(
            status_code=422,
            detail=f"Zipcode {zipcode} not found in demographics data",
        )


def _build_features_with_demographics(
    house_data: dict, zipcode: str, house_cols: list,
    demographics_df: pd.DataFrame, feature_order: list,
    keep_zipcode: bool = False,
) -> pd.DataFrame:
    """Build features by joining demographics. Used by basic, minimal, enhanced."""
    _validate_zipcode(zipcode, demographics_df)

    row = {col: house_data[col] for col in house_cols if col != "zipcode"}
    row["zipcode"] = zipcode
    house_df = pd.DataFrame([row])

    merged = house_df.merge(demographics_df, how="inner", on="zipcode")
    if keep_zipcode:
        merged["zipcode"] = merged["zipcode"].astype(int)
    else:
        merged = merged.drop(columns="zipcode")
    merged = merged[feature_order]
    return merged


def _build_features_no_demographics(
    house_data: dict, house_cols: list, feature_order: list,
) -> pd.DataFrame:
    """Build features without demographics join. Used by optimized."""
    row = {col: house_data[col] for col in house_cols if col != "zipcode"}
    if "zipcode" in house_cols:
        row["zipcode"] = int(house_data["zipcode"])
    df = pd.DataFrame([row])
    df = df[feature_order]
    return df


def _make_prediction(model, features_df, model_type, model_version, metrics):
    prediction = float(model.predict(features_df)[0])
    warning = None
    if prediction <= 0:
        warning = "Model returned non-positive price — prediction may be unreliable"
    return PredictionResponse(
        predicted_price=prediction,
        model_type=model_type,
        model_version=model_version,
        provider="Sound Realty AI",
        r2_score=metrics["r2_score"],
        mae=metrics["mae"],
        mape=metrics["mape"],
        rmse=metrics["rmse"],
        n_features=metrics["n_features"],
        uses_demographics=metrics["uses_demographics"],
        warning=warning,
    )


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------


@app.post("/predict/basic", response_model=PredictionResponse)
async def predict_basic(house: BasicHouseFeatures):
    """Predict using the original KNN model. Accepts all 18 property columns.

    The API joins 26 demographic features from zipcode_demographics.csv on the backend.
    **Total model features: 33** (7 house + 26 demographics).

    House features used: bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement.

    Demographics joined on zipcode: ppltn_qty, urbn_ppltn_qty, sbrbn_ppltn_qty, farm_ppltn_qty,
    non_farm_qty, medn_hshld_incm_amt, medn_incm_per_prsn_amt, hous_val_amt,
    edctn_less_than_9_qty, edctn_9_12_qty, edctn_high_schl_qty, edctn_some_clg_qty,
    edctn_assoc_dgre_qty, edctn_bchlr_dgre_qty, edctn_prfsnl_qty,
    per_urbn, per_sbrbn, per_farm, per_non_farm,
    per_less_than_9, per_9_to_12, per_hsd, per_some_clg, per_assoc, per_bchlr, per_prfsnl.
    """
    features_df = _build_features_with_demographics(
        house.model_dump(), house.zipcode, BASIC_HOUSE_FEATURES,
        app.state.demographics, app.state.basic_features,
    )
    return _make_prediction(
        app.state.basic_model, features_df, "KNN (basic)", app.state.model_version,
        app.state.model_metrics["basic"],
    )


@app.post("/predict/minimal", response_model=PredictionResponse)
async def predict_minimal(house: MinimalHouseFeatures):
    """Predict using the KNN model with only the 8 required input columns.

    Same model as /predict/basic — only 8 input fields needed, but the API joins
    26 demographic features on the backend. **Total model features: 33** (7 house + 26 demographics).

    House features used: bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement.

    Demographics joined on zipcode: ppltn_qty, urbn_ppltn_qty, sbrbn_ppltn_qty, farm_ppltn_qty,
    non_farm_qty, medn_hshld_incm_amt, medn_incm_per_prsn_amt, hous_val_amt,
    edctn_less_than_9_qty, edctn_9_12_qty, edctn_high_schl_qty, edctn_some_clg_qty,
    edctn_assoc_dgre_qty, edctn_bchlr_dgre_qty, edctn_prfsnl_qty,
    per_urbn, per_sbrbn, per_farm, per_non_farm,
    per_less_than_9, per_9_to_12, per_hsd, per_some_clg, per_assoc, per_bchlr, per_prfsnl.
    """
    features_df = _build_features_with_demographics(
        house.model_dump(), house.zipcode, BASIC_HOUSE_FEATURES,
        app.state.demographics, app.state.basic_features,
    )
    return _make_prediction(
        app.state.basic_model, features_df, "KNN (minimal)", app.state.model_version,
        app.state.model_metrics["basic"],
    )


@app.post("/predict/enhanced", response_model=PredictionResponse)
async def predict_enhanced(house: EnhancedHouseFeatures):
    """Predict using LightGBM with 10 best features + demographics.

    The API joins 26 demographic features on the backend.
    **Total model features: 36** (10 house + 26 demographics).

    House features used: sqft_living, sqft_lot, sqft_above, sqft_basement,
    grade, lat, long, bathrooms, waterfront, zipcode.

    Demographics joined on zipcode: ppltn_qty, urbn_ppltn_qty, sbrbn_ppltn_qty, farm_ppltn_qty,
    non_farm_qty, medn_hshld_incm_amt, medn_incm_per_prsn_amt, hous_val_amt,
    edctn_less_than_9_qty, edctn_9_12_qty, edctn_high_schl_qty, edctn_some_clg_qty,
    edctn_assoc_dgre_qty, edctn_bchlr_dgre_qty, edctn_prfsnl_qty,
    per_urbn, per_sbrbn, per_farm, per_non_farm,
    per_less_than_9, per_9_to_12, per_hsd, per_some_clg, per_assoc, per_bchlr, per_prfsnl.
    """
    features_df = _build_features_with_demographics(
        house.model_dump(), house.zipcode, BEST_HOUSE_FEATURES,
        app.state.demographics, app.state.enhanced_features,
        keep_zipcode=True,
    )
    return _make_prediction(
        app.state.enhanced_model, features_df, "LightGBM (enhanced)", app.state.model_version,
        app.state.model_metrics["enhanced"],
    )


@app.post("/predict/optimized", response_model=PredictionResponse)
async def predict_optimized(house: EnhancedHouseFeatures):
    """Predict using LightGBM with 10 best features only (no demographics).

    Best cost-benefit model: no external data dependency, yet comparable accuracy.
    **Total model features: 10** (no demographics join).

    Features used: sqft_living, sqft_lot, sqft_above, sqft_basement,
    grade, lat, long, bathrooms, waterfront, zipcode.
    """
    features_df = _build_features_no_demographics(
        house.model_dump(), BEST_HOUSE_FEATURES, app.state.optimized_features,
    )
    return _make_prediction(
        app.state.optimized_model, features_df, "LightGBM (optimized)", app.state.model_version,
        app.state.model_metrics["optimized"],
    )


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": {
            "basic": app.state.basic_model is not None,
            "enhanced": app.state.enhanced_model is not None,
            "optimized": app.state.optimized_model is not None,
        },
        "model_version": app.state.model_version,
    }


@app.get("/metrics")
async def metrics():
    """Return current API metrics as JSON."""
    return store.snapshot()


@app.post("/metrics/reset")
async def metrics_reset():
    """Reset all API metrics to zero."""
    store.reset()
    return {"status": "reset"}


@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring():
    """Serve the monitoring dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML_PATH.read_text())


@app.get("/retrain", response_class=HTMLResponse)
async def retrain_dashboard():
    """Serve the retraining dashboard."""
    return HTMLResponse(content=RETRAIN_HTML_PATH.read_text())


@app.get("/model-info")
async def model_info():
    """Return performance metrics for all models."""
    return app.state.model_metrics


@app.get("/test-data")
async def test_data():
    """Return future_unseen_examples.csv rows as JSON for the stress test panel."""
    df = pd.read_csv(UNSEEN_EXAMPLES_PATH, dtype={"zipcode": str})
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Retraining module (retrains the optimized model — best production model)
# ---------------------------------------------------------------------------


@app.get("/retrain/history")
async def retrain_history():
    """Return the history of retraining attempts."""
    return app.state.retrain_history


@app.post("/retrain/run", response_model=RetrainResponse)
async def retrain_run(
    file: UploadFile = File(...),
    train_ratio: float = Form(0.8),
    auto_deploy: bool = Form(False),
    model_type: str = Form("optimized"),
):
    """Retrain a model with new CSV data.

    - Receives a CSV with at least: date, price, zipcode, and relevant house columns.
    - The backend joins demographics automatically for basic/enhanced models.
    - Splits chronologically (before/after cutoff based on train_ratio).
    - Trains a candidate model and compares with the current one.
    - If auto_deploy=True and candidate is better, hot-swaps the model in memory.

    model_type: "basic" (KNN), "enhanced" (LightGBM + demographics), or "optimized" (LightGBM only).
    """
    valid_types = {"basic", "enhanced", "optimized"}
    if model_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"model_type must be one of {valid_types}")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), dtype={'zipcode': str})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if len(df) < 10:
        raise HTTPException(status_code=400, detail="CSV must have at least 10 rows")
    if train_ratio < 0.1 or train_ratio > 0.95:
        raise HTTPException(status_code=400, detail="train_ratio must be between 0.1 and 0.95")

    # Determine which house features to use
    if model_type == "basic":
        house_cols = BASIC_HOUSE_FEATURES
        uses_demographics = True
    else:
        house_cols = BEST_HOUSE_FEATURES
        uses_demographics = (model_type == "enhanced")

    required_cols = {'date', 'price', 'zipcode'} | set(house_cols)
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV missing columns: {missing}")

    # Parse and sort by date for chronological split
    df = df.copy()
    df['_date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S', errors='coerce')
    if df['_date'].isna().all():
        df['_date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('_date').reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    cutoff_date = df.iloc[split_idx]['_date']

    # Prepare features based on model type
    demographics_df = app.state.demographics

    def prepare(subset):
        if uses_demographics:
            feature_cols = [c for c in house_cols if c != 'zipcode']
            sub = subset[['price', 'zipcode'] + feature_cols].copy()
            merged = sub.merge(demographics_df, how="left", on="zipcode")
            if model_type == "enhanced":
                merged['zipcode'] = merged['zipcode'].astype(int)
            else:
                merged = merged.drop(columns='zipcode')
            y = merged.pop('price')
            return merged, y
        else:
            feature_cols = [c for c in house_cols if c != 'zipcode']
            sub = subset[['price', 'zipcode'] + feature_cols].copy()
            sub['zipcode'] = sub['zipcode'].astype(int)
            y = sub.pop('price')
            return sub, y

    X_train, y_train = prepare(train_df)
    X_test, y_test = prepare(test_df)

    # Train candidate
    if model_type == "basic":
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler
        candidate = make_pipeline(RobustScaler(), KNeighborsRegressor())
    else:
        candidate = lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
        )
    candidate.fit(X_train, y_train)
    candidate_metrics = _eval_metrics(candidate, X_test, y_test)

    # Compare candidate against deployed model's reference metrics
    ref = app.state.model_metrics[model_type]
    improved = candidate_metrics['r2'] > ref['r2_score']

    # Deploy paths per model type
    model_paths = {
        "basic": (BASIC_MODEL_PATH, BASIC_FEATURES_PATH),
        "enhanced": (ENHANCED_MODEL_PATH, ENHANCED_FEATURES_PATH),
        "optimized": (OPTIMIZED_MODEL_PATH, OPTIMIZED_FEATURES_PATH),
    }
    model_state_keys = {
        "basic": ("basic_model", "basic_features"),
        "enhanced": ("enhanced_model", "enhanced_features"),
        "optimized": ("optimized_model", "optimized_features"),
    }

    auto_deployed = False
    if improved and auto_deploy:
        model_key, features_key = model_state_keys[model_type]
        setattr(app.state, model_key, candidate)
        setattr(app.state, features_key, list(X_train.columns))
        app.state.model_version = f"2.1.{len(app.state.retrain_history)}"

        model_path, features_path = model_paths[model_type]
        pickle.dump(candidate, open(model_path, 'wb'))
        json.dump(list(X_train.columns), open(features_path, 'w'))

        # Update model metrics
        app.state.model_metrics[model_type] = {
            "r2_score": candidate_metrics['r2'],
            "mae": candidate_metrics['mae'],
            "mape": candidate_metrics['mape'],
            "rmse": candidate_metrics['rmse'],
            "n_features": len(X_train.columns),
            "uses_demographics": uses_demographics,
        }

        auto_deployed = True
        logger.info("%s model auto-deployed: v%s", model_type, app.state.model_version)

    entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'filename': file.filename,
        'model_type': model_type,
        'rows': len(df),
        'train_size': len(train_df),
        'test_size': len(test_df),
        'cutoff_date': str(cutoff_date.date()) if pd.notna(cutoff_date) else 'unknown',
        'train_ratio': train_ratio,
        'candidate_metrics': candidate_metrics,
        'improved': improved,
        'auto_deployed': auto_deployed,
    }
    app.state.retrain_history.append(entry)

    if improved:
        msg = "Candidate model is BETTER."
        msg += f" Auto-deployed as v{app.state.model_version}." if auto_deployed else " Auto-deploy was OFF — model NOT replaced."
    else:
        msg = "Candidate model did NOT improve over current. No changes made."

    # Stored training metrics (from original training, for reference)
    ref = app.state.model_metrics[model_type]
    reference = {
        'r2': ref['r2_score'], 'mae': ref['mae'],
        'mape': ref['mape'], 'rmse': ref['rmse'],
        'note': 'Original training metrics (random 75/25 split)',
    }

    return RetrainResponse(
        status="completed",
        candidate_model=candidate_metrics,
        reference_metrics=reference,
        improved=improved,
        auto_deployed=auto_deployed,
        message=msg,
    )


def _eval_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        'r2': round(float(r2_score(y_test, y_pred)), 4),
        'mae': round(float(mean_absolute_error(y_test, y_pred)), 2),
        'mape': round(float(mean_absolute_percentage_error(y_test, y_pred) * 100), 2),
        'rmse': round(float(np.sqrt(np.mean((y_test - y_pred) ** 2))), 2),
    }
