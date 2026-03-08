![phData Logo](phData.png "phData Logo")

# Sound Realty — Housing Price Prediction API

A REST API that serves housing price predictions for the Seattle metro area, built for the phData MLE candidate project.

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-started/) installed
- Or: Python 3.9 with [Conda](https://docs.conda.io/en/latest/)

### 1. Train the Model

```sh
conda env create -f conda_environment.yml
conda activate housing
python create_model.py
```

This outputs `model/model.pkl` and `model/model_features.json`.

### 2. Run the API

**With Docker (recommended):**

```sh
docker build -t sound-realty-api .
docker run -p 8000:8000 sound-realty-api
```

**Without Docker:**

```sh
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API is now available at `http://localhost:8000`.

### 3. Verify It's Running

Open `http://localhost:8000/health` in your browser or run:

```sh
curl http://localhost:8000/health
```

Expected response:

```json
{ "status": "healthy", "model_loaded": true }
```

---

## API Endpoints

### `GET /health`

Health check for container orchestration and load balancer probes.

**Response:**

```json
{ "status": "healthy", "model_loaded": true }
```

### `POST /predict`

Predict the sale price of a property. Accepts 18 property columns as JSON. The API joins demographic data on `zipcode` automatically — clients never send demographics.

**Request body (all fields required):**

| Field            | Type    | Description                        |
| ---------------- | ------- | ---------------------------------- |
| `bedrooms`       | int     | Number of bedrooms                 |
| `bathrooms`      | float   | Number of bathrooms                |
| `sqft_living`    | int     | Living area in sq ft               |
| `sqft_lot`       | int     | Lot size in sq ft                  |
| `floors`         | float   | Number of floors                   |
| `waterfront`     | int     | Waterfront property (0 or 1)       |
| `view`           | int     | View rating (0–4)                  |
| `condition`      | int     | Condition rating (1–5)             |
| `grade`          | int     | Building grade (1–13)              |
| `sqft_above`     | int     | Above-ground sq ft                 |
| `sqft_basement`  | int     | Basement sq ft                     |
| `yr_built`       | int     | Year built                         |
| `yr_renovated`   | int     | Year renovated (0 if never)        |
| `zipcode`        | string  | 5-digit zipcode (must exist in demographics data) |
| `lat`            | float   | Latitude                           |
| `long`           | float   | Longitude                          |
| `sqft_living15`  | int     | Living area of nearest 15 neighbors |
| `sqft_lot15`     | int     | Lot size of nearest 15 neighbors   |

**Response:**

| Field             | Type           | Description                                  |
| ----------------- | -------------- | -------------------------------------------- |
| `predicted_price` | float          | Estimated sale price in USD                   |
| `model_version`   | string         | Semantic version of the deployed model        |
| `provider`        | string         | Branding identifier ("Sound Realty AI")       |
| `warning`         | string or null | Note if prediction may be unreliable          |

**Error responses:**

- `422` — Invalid input (missing fields, wrong types, or unknown zipcode)

---

## Testing the API

### Option 1: Test Script

```sh
python test_api.py
```

Sends 5 sample properties from `data/future_unseen_examples.csv` plus an edge case (invalid zipcode). Pass `--url` to target a different host:

```sh
python test_api.py --url http://my-server:8000
```

### Option 2: cURL

```sh
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "sqft_lot15": 5765
  }'
```

### Option 3: Postman

#### Setting Up

1. Open Postman and create a new **Collection** called `Sound Realty API`
2. Add requests as described below

#### Health Check Request

- **Method:** `GET`
- **URL:** `http://localhost:8000/health`
- Click **Send** — you should see `{ "status": "healthy", "model_loaded": true }`

#### Predict Request

1. Create a new request in the collection
2. **Method:** `POST`
3. **URL:** `http://localhost:8000/predict`
4. Go to the **Headers** tab and add:
   - Key: `Content-Type` — Value: `application/json`
5. Go to the **Body** tab, select **raw** and **JSON**, then paste:

```json
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
  "sqft_lot15": 5765
}
```

6. Click **Send**

**Expected response:**

```json
{
  "predicted_price": 386800.0,
  "model_version": "1.0.0",
  "provider": "Sound Realty AI",
  "warning": null
}
```

#### Testing Edge Cases in Postman

- **Invalid zipcode:** Change `"zipcode"` to `"00000"` — should return `422`
- **Missing field:** Remove any field from the body — should return `422`
- **Different properties:** Use values from `data/future_unseen_examples.csv` to test various homes

#### Valid Zipcodes for Testing

Some zipcodes present in the demographics data: `98001`, `98002`, `98003`, `98004`, `98005`, `98006`, `98007`, `98008`, `98010`, `98011`, `98014`, `98019`, `98022`, `98023`, `98024`, `98027`, `98028`, `98029`, `98030`, `98031`, `98032`, `98033`, `98034`, `98038`, `98039`, `98040`, `98042`, `98045`, `98052`, `98053`, `98055`, `98056`, `98058`, `98059`, `98065`, `98070`, `98072`, `98074`, `98075`, `98077`, `98092`, `98102`, `98103`, `98105`, `98106`, `98107`, `98108`, `98109`, `98112`, `98115`, `98116`, `98117`, `98118`, `98119`, `98122`, `98125`, `98126`, `98133`, `98136`, `98144`, `98146`, `98148`, `98155`, `98166`, `98168`, `98177`, `98178`, `98188`, `98198`, `98199`.

### Option 4: Interactive Docs (Swagger UI)

FastAPI auto-generates interactive docs at `http://localhost:8000/docs`. You can test endpoints directly from the browser.

---

## Project Structure

```
.
├── app/
│   └── main.py              # FastAPI application
├── data/
│   ├── kc_house_data.csv     # Training data (21,613 home sales)
│   ├── zipcode_demographics.csv  # US Census demographics per zipcode
│   └── future_unseen_examples.csv # Unseen homes for prediction
├── model/
│   ├── model.pkl             # Trained model (generated)
│   └── model_features.json   # Feature order (generated)
├── notebook/
│   ├── explore_data.ipynb    # EDA notebook
│   └── create_model.ipynb    # Model training notebook
├── create_model.py           # Model training script
├── test_api.py               # API test script
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
└── conda_environment.yml     # Conda environment
```

## Architecture

- **Framework:** FastAPI with uvicorn
- **Model:** `RobustScaler` + `KNeighborsRegressor` pipeline (scikit-learn)
- **Container:** Docker with non-root user, single exposed port (8000)
- **Data pipeline:** 7 house features + demographic join on zipcode = 33 model features
