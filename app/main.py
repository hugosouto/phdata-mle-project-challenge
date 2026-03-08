"""Sound Realty housing price prediction API.

Serves a KNeighborsRegressor model trained on King County house sales data.
Accepts property features via POST /predict, joins demographic data on zipcode
in the backend, and returns a price prediction with metadata.
"""

import json
import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
FEATURES_PATH = BASE_DIR / "model" / "model_features.json"
DEMOGRAPHICS_PATH = BASE_DIR / "data" / "zipcode_demographics.csv"
UNSEEN_EXAMPLES_PATH = BASE_DIR / "data" / "future_unseen_examples.csv"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class HouseFeatures(BaseModel):
    """Input schema matching future_unseen_examples.csv.

    Client sends all 18 property columns. The API joins demographic data on
    zipcode automatically — clients never send demographics.
    """

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
            "examples": [
                {
                    "bedrooms": 4,
                    "bathrooms": 1.0,
                    "sqft_living": 1680,
                    "sqft_lot": 5043,
                    "floors": 1.5,
                    "waterfront": 0,
                    "view": 0,
                    "condition": 4,
                    "grade": 6,
                    "sqft_above": 1680,
                    "sqft_basement": 0,
                    "yr_built": 1911,
                    "yr_renovated": 0,
                    "zipcode": "98118",
                    "lat": 47.5354,
                    "long": -122.273,
                    "sqft_living15": 1560,
                    "sqft_lot15": 5765,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response payload for a price prediction.

    predicted_price: the model's estimated sale price in USD.
    model_version: semantic version of the deployed model.
    provider: branding identifier.
    warning: optional note when the prediction may be unreliable.
    """

    predicted_price: float
    model_version: str
    provider: str
    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, feature order, and demographics once at startup.

    The model (~4 MB) and demographics (~70 rows) are small enough to hold in
    memory. Fails fast if any artifact is missing so the container never starts
    in a broken state.
    """
    try:
        with open(MODEL_PATH, "rb") as f:
            app.state.model = pickle.load(f)
        logger.info("Model loaded from %s", MODEL_PATH)
    except Exception as exc:
        logger.error("FATAL: Could not load %s — %s", MODEL_PATH, exc)
        raise

    try:
        with open(FEATURES_PATH) as f:
            app.state.feature_order = json.load(f)
        logger.info("Feature order loaded (%d features)", len(app.state.feature_order))
    except Exception as exc:
        logger.error("FATAL: Could not load %s — %s", FEATURES_PATH, exc)
        raise

    try:
        app.state.demographics = pd.read_csv(
            DEMOGRAPHICS_PATH, dtype={"zipcode": str}
        )
        logger.info(
            "Demographics loaded (%d zipcodes)", len(app.state.demographics)
        )
    except Exception as exc:
        logger.error("FATAL: Could not load %s — %s", DEMOGRAPHICS_PATH, exc)
        raise

    yield  # application runs here


app = FastAPI(
    title="Sound Realty Price Prediction API",
    description="Housing price predictions for the Seattle metro area.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Metrics middleware (must be registered before endpoints)
# ---------------------------------------------------------------------------
from app.metrics import metrics_middleware, store  # noqa: E402

app.middleware("http")(metrics_middleware)

DASHBOARD_HTML_PATH = Path(__file__).resolve().parent / "templates" / "dashboard.html"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The 7 house features the model actually uses (subset of the 18 input fields).
MODEL_HOUSE_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
]


def prepare_features(
    house: HouseFeatures,
    demographics_df: pd.DataFrame,
    feature_order: list,
) -> pd.DataFrame:
    """Build a single-row DataFrame ready for model.predict().

    Mirrors the data pipeline from create_model.py: extract model features,
    join demographics on zipcode, drop zipcode, reorder to match training
    feature order. Validates zipcode before merge to fail fast with a clear
    error.
    """
    # 1. Validate zipcode exists in demographics
    if house.zipcode not in demographics_df["zipcode"].values:
        raise HTTPException(
            status_code=422,
            detail=f"Zipcode {house.zipcode} not found in demographics data",
        )

    # 2. Single-row DataFrame with 7 house features + zipcode (join key)
    row = {feat: getattr(house, feat) for feat in MODEL_HOUSE_FEATURES}
    row["zipcode"] = house.zipcode
    house_df = pd.DataFrame([row])

    # 3. Merge with demographics on zipcode
    merged = house_df.merge(demographics_df, how="inner", on="zipcode")

    # 4. Drop zipcode (join key, not a model feature)
    merged = merged.drop(columns="zipcode")

    # 5. Reorder to match exact training feature order
    merged = merged[feature_order]

    return merged


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResponse)
async def predict(house: HouseFeatures):
    """Predict the sale price of a property.

    Accepts 18 property columns, joins demographic data on zipcode, feeds
    the resulting 33-feature vector to the model, and returns a price estimate.
    """
    features_df = prepare_features(
        house, app.state.demographics, app.state.feature_order
    )

    prediction = app.state.model.predict(features_df)
    predicted_price = float(prediction[0])

    warning = None
    if predicted_price <= 0:
        logger.warning(
            "Model returned non-positive price %.2f for zipcode %s",
            predicted_price,
            house.zipcode,
        )
        warning = "Model returned non-positive price — prediction may be unreliable"

    logger.info("Prediction: zipcode=%s price=%.2f", house.zipcode, predicted_price)

    return PredictionResponse(
        predicted_price=predicted_price,
        model_version="1.0.0",
        provider="Sound Realty AI",
        warning=warning,
    )


@app.get("/health")
async def health():
    """Health check endpoint for container orchestration and load balancer probes."""
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    return {"status": "healthy", "model_loaded": model_loaded}


@app.get("/metrics")
async def metrics():
    """Return current API metrics as JSON."""
    return store.snapshot()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the monitoring dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML_PATH.read_text())


@app.get("/test-data")
async def test_data():
    """Return future_unseen_examples.csv rows as JSON for the stress test panel."""
    df = pd.read_csv(UNSEEN_EXAMPLES_PATH, dtype={"zipcode": str})
    return df.to_dict(orient="records")
