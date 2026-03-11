![phData Logo](phData.png "phData Logo")

# Sound Realty — Enhanced Price Prediction API

An enhanced REST API serving housing price predictions for the Seattle metro area, featuring four inference endpoints, a monitoring dashboard, and a model retraining module.

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-started/) installed
- Or: Python 3.9 with [Conda](https://docs.conda.io/en/latest/)

### 1. Train the Models

```sh
conda env create -f conda_environment.yml
conda activate housing
python create_model.py
```

This outputs three models:
- `model/basic_model.pkl` — KNN (7 house features + demographics, R² ≈ 0.73)
- `model/enhanced_model.pkl` — LightGBM (10 best features + demographics, R² ≈ 0.87)
- `model/optimized_model.pkl` — LightGBM (10 best features only, no demographics, R² ≈ 0.87)

### 2. Run the API

**With Docker (recommended):**

```sh
docker build -t sound-realty-enhanced .
docker run -p 8000:8000 sound-realty-enhanced
```

**Without Docker:**

```sh
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API is now available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Prediction Endpoints

### `POST /predict/basic` — KNN model, full input

Accepts all 18 property columns from `future_unseen_examples.csv`. Uses the original KNN model (same as baseline-api). The API joins demographic data on `zipcode` automatically.

```sh
curl -X POST http://localhost:8000/predict/basic \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 4, "bathrooms": 1.0, "sqft_living": 1680,
    "sqft_lot": 5043, "floors": 1.5, "waterfront": 0, "view": 0,
    "condition": 4, "grade": 6, "sqft_above": 1680,
    "sqft_basement": 0, "yr_built": 1911, "yr_renovated": 0,
    "zipcode": "98118", "lat": 47.5354, "long": -122.273,
    "sqft_living15": 1560, "sqft_lot15": 5765
  }'
```

### `POST /predict/minimal` — KNN model, minimal input (bonus)

Only the 8 columns the KNN model actually needs. Same model as `/predict/basic`, fewer required fields.

```sh
curl -X POST http://localhost:8000/predict/minimal \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 4, "bathrooms": 1.0, "sqft_living": 1680,
    "sqft_lot": 5043, "floors": 1.5, "sqft_above": 1680,
    "sqft_basement": 0, "zipcode": "98118"
  }'
```

### `POST /predict/enhanced` — LightGBM + demographics

Uses 10 best features + demographics joined on zipcode. Highest feature count but depends on external demographics data.

```sh
curl -X POST http://localhost:8000/predict/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "sqft_living": 1680, "sqft_lot": 5043, "sqft_above": 1680,
    "sqft_basement": 0, "grade": 6, "lat": 47.5354,
    "long": -122.273, "bathrooms": 1.0, "waterfront": 0,
    "zipcode": "98118"
  }'
```

### `POST /predict/optimized` — LightGBM, no demographics

Best cost-benefit model: uses only 10 house features (no demographics join), yet achieves comparable accuracy. No external data dependency.

```sh
curl -X POST http://localhost:8000/predict/optimized \
  -H "Content-Type: application/json" \
  -d '{
    "sqft_living": 1680, "sqft_lot": 5043, "sqft_above": 1680,
    "sqft_basement": 0, "grade": 6, "lat": 47.5354,
    "long": -122.273, "bathrooms": 1.0, "waterfront": 0,
    "zipcode": "98118"
  }'
```

### Response format (all four endpoints)

```json
{
  "predicted_price": 351694.46,
  "model_type": "LightGBM (optimized)",
  "model_version": "2.0.0",
  "provider": "Sound Realty AI",
  "warning": null
}
```

| Field             | Type           | Description                             |
| ----------------- | -------------- | --------------------------------------- |
| `predicted_price` | float          | Estimated sale price in USD             |
| `model_type`      | string         | Which model served this prediction      |
| `model_version`   | string         | Version of the deployed model           |
| `provider`        | string         | Branding identifier                     |
| `warning`         | string or null | Note if prediction may be unreliable    |

**Error:** `422` — Invalid input (missing fields, wrong types, or unknown zipcode)

---

## Monitoring Dashboard

`http://localhost:8000/dashboard`

- Real-time metrics: request count, error rate, average latency
- Live charts: latency timeline, status codes, price distribution, endpoint breakdown
- Stress test panel: fire requests from `future_unseen_examples.csv` and watch metrics update

| Endpoint | Description |
| --- | --- |
| `GET /dashboard` | Interactive monitoring UI |
| `GET /metrics` | Raw metrics as JSON |
| `GET /test-data` | Unseen examples as JSON (used by stress test panel) |

---

## Retraining Module

`http://localhost:8000/retrain`

A dashboard for retraining the optimized (LightGBM) model with new data, without restarting the service.

### How it works

1. **Upload a CSV** with columns: `date`, `price`, `zipcode`, and the 10 optimized feature columns
2. **Choose the train/test split ratio** — the split is always chronological (earlier dates = train, later dates = test)
3. **Optionally enable auto-deploy** — if the candidate model beats the current one (by R²), it hot-swaps in memory
4. **Compare results** — side-by-side metrics (R², MAE, MAPE, RMSE) for current vs. candidate

| Endpoint | Description |
| --- | --- |
| `GET /retrain` | Retraining dashboard UI |
| `POST /retrain/run` | Run retraining (multipart form: `file`, `train_ratio`, `auto_deploy`) |
| `GET /retrain/history` | History of all retraining attempts |

### Testing retraining with existing data

```sh
curl -X POST http://localhost:8000/retrain/run \
  -F "file=@data/kc_house_data.csv" \
  -F "train_ratio=0.8" \
  -F "auto_deploy=false"
```

---

## Other Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | GET | Health check — reports model load status and version |
| `/docs` | GET | Interactive Swagger UI (auto-generated by FastAPI) |

---

## Testing the API

```sh
python test_api.py
```

Sends sample properties to all four endpoints plus edge cases. Use `--url` for a different host:

```sh
python test_api.py --url http://my-server:8000
```

---

## Project Structure

```
.
├── app/
│   ├── main.py              # FastAPI application (4 endpoints + retraining)
│   ├── metrics.py           # In-memory metrics store & middleware
│   └── templates/
│       ├── dashboard.html   # Monitoring dashboard UI
│       └── retrain.html     # Retraining dashboard UI
├── data/
│   ├── kc_house_data.csv
│   ├── zipcode_demographics.csv
│   └── future_unseen_examples.csv
├── model/
│   ├── basic_model.pkl            # KNN model (generated)
│   ├── basic_model_features.json
│   ├── enhanced_model.pkl         # LightGBM + demographics (generated)
│   ├── enhanced_model_features.json
│   ├── optimized_model.pkl        # LightGBM, no demographics (generated)
│   └── optimized_model_features.json
├── create_model.py           # Trains all three models
├── test_api.py               # API test script (tests all 4 endpoints)
├── Dockerfile
├── requirements.txt
└── conda_environment.yml
```

## Architecture

- **Framework:** FastAPI + uvicorn
- **Models:** KNN (baseline) + LightGBM (enhanced, with demographics) + LightGBM (optimized, no demographics)
- **Container:** Docker with non-root user, port 8000
- **Retraining:** Chronological split, compare-and-swap, zero-downtime model updates
