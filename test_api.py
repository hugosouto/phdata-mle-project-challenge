"""Test script that sends example properties to all four prediction endpoints.

Reads properties from data/future_unseen_examples.csv, posts them to each
endpoint, and verifies responses. Also tests edge cases like invalid zipcodes.

Usage:
    python test_api.py [--url http://localhost:8000]
"""

import argparse
import sys

import pandas as pd
import requests

DEFAULT_URL = "http://localhost:8000"


def main():
    parser = argparse.ArgumentParser(description="Test the Sound Realty prediction API")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the API")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    passed = 0
    failed = 0

    # -----------------------------------------------------------------------
    # Health check
    # -----------------------------------------------------------------------
    print(f"Testing API at {base_url}\n")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
    except requests.ConnectionError:
        print(f"Could not connect to API at {base_url}. Is the server running?")
        sys.exit(1)

    try:
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
        health = resp.json()
        assert health["status"] == "healthy", f"Unhealthy: {health}"
        models = health["models_loaded"]
        assert models["basic"] is True, f"Basic model not loaded: {health}"
        assert models["enhanced"] is True, f"Enhanced model not loaded: {health}"
        assert models["optimized"] is True, f"Optimized model not loaded: {health}"
        print(f"Health check: {health}")
        passed += 1
    except AssertionError as exc:
        print(f"Health check: FAILED — {exc}")
        failed += 1

    # -----------------------------------------------------------------------
    # Load test data
    # -----------------------------------------------------------------------
    df = pd.read_csv("data/future_unseen_examples.csv", dtype={"zipcode": str})
    sample = df.head(3)

    # -----------------------------------------------------------------------
    # /predict/basic — KNN, all 18 columns
    # -----------------------------------------------------------------------
    print("\n--- /predict/basic (KNN, all 18 columns) ---")
    for i, row in sample.iterrows():
        payload = row.to_dict()
        resp = requests.post(f"{base_url}/predict/basic", json=payload, timeout=10)
        try:
            assert resp.status_code == 200, f"Row {i}: status {resp.status_code}"
            body = resp.json()
            assert body["predicted_price"] > 0, f"Row {i}: non-positive price"
            assert body["model_type"] == "KNN (basic)"
            print(
                f"  Row {i}: zipcode={payload['zipcode']}  "
                f"predicted=${body['predicted_price']:,.0f}  "
                f"model={body['model_type']}"
            )
            passed += 1
        except AssertionError as exc:
            print(f"  Row {i}: FAILED — {exc}")
            failed += 1

    # -----------------------------------------------------------------------
    # /predict/minimal — KNN, 8 columns only
    # -----------------------------------------------------------------------
    print("\n--- /predict/minimal (KNN, 8 columns) ---")
    minimal_cols = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot",
                    "floors", "sqft_above", "sqft_basement", "zipcode"]
    for i, row in sample.iterrows():
        payload = {col: row[col] for col in minimal_cols}
        resp = requests.post(f"{base_url}/predict/minimal", json=payload, timeout=10)
        try:
            assert resp.status_code == 200, f"Row {i}: status {resp.status_code}"
            body = resp.json()
            assert body["predicted_price"] > 0, f"Row {i}: non-positive price"
            assert body["model_type"] == "KNN (minimal)"
            print(
                f"  Row {i}: zipcode={payload['zipcode']}  "
                f"predicted=${body['predicted_price']:,.0f}  "
                f"model={body['model_type']}"
            )
            passed += 1
        except AssertionError as exc:
            print(f"  Row {i}: FAILED — {exc}")
            failed += 1

    # -----------------------------------------------------------------------
    # /predict/enhanced — LightGBM + demographics, 10 columns
    # -----------------------------------------------------------------------
    print("\n--- /predict/enhanced (LightGBM + demographics, 10 columns) ---")
    enhanced_cols = ["sqft_living", "sqft_lot", "sqft_above", "sqft_basement",
                     "grade", "lat", "long", "bathrooms", "waterfront", "zipcode"]
    for i, row in sample.iterrows():
        payload = {col: row[col] for col in enhanced_cols}
        resp = requests.post(f"{base_url}/predict/enhanced", json=payload, timeout=10)
        try:
            assert resp.status_code == 200, f"Row {i}: status {resp.status_code}"
            body = resp.json()
            assert body["predicted_price"] > 0, f"Row {i}: non-positive price"
            assert body["model_type"] == "LightGBM (enhanced)"
            print(
                f"  Row {i}: zipcode={payload['zipcode']}  "
                f"predicted=${body['predicted_price']:,.0f}  "
                f"model={body['model_type']}"
            )
            passed += 1
        except AssertionError as exc:
            print(f"  Row {i}: FAILED — {exc}")
            failed += 1

    # -----------------------------------------------------------------------
    # /predict/optimized — LightGBM, 10 columns, NO demographics
    # -----------------------------------------------------------------------
    print("\n--- /predict/optimized (LightGBM, 10 columns, no demographics) ---")
    for i, row in sample.iterrows():
        payload = {col: row[col] for col in enhanced_cols}
        resp = requests.post(f"{base_url}/predict/optimized", json=payload, timeout=10)
        try:
            assert resp.status_code == 200, f"Row {i}: status {resp.status_code}"
            body = resp.json()
            assert body["predicted_price"] > 0, f"Row {i}: non-positive price"
            assert body["model_type"] == "LightGBM (optimized)"
            print(
                f"  Row {i}: zipcode={payload['zipcode']}  "
                f"predicted=${body['predicted_price']:,.0f}  "
                f"model={body['model_type']}"
            )
            passed += 1
        except AssertionError as exc:
            print(f"  Row {i}: FAILED — {exc}")
            failed += 1

    # -----------------------------------------------------------------------
    # Edge case: invalid zipcode (test on all endpoints that validate zipcode)
    # -----------------------------------------------------------------------
    print("\n--- Edge cases ---")
    edge_payload = sample.iloc[0].to_dict()
    edge_payload["zipcode"] = "00000"

    for endpoint in ["/predict/basic", "/predict/minimal", "/predict/enhanced"]:
        if endpoint == "/predict/minimal":
            payload = {col: edge_payload[col] for col in minimal_cols}
        elif endpoint == "/predict/enhanced":
            payload = {col: edge_payload[col] for col in enhanced_cols}
        else:
            payload = edge_payload.copy()

        resp = requests.post(f"{base_url}{endpoint}", json=payload, timeout=10)
        try:
            assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
            print(f"  {endpoint} (zipcode='00000'): correctly returned 422")
            passed += 1
        except AssertionError as exc:
            print(f"  {endpoint} edge case: FAILED — {exc}")
            failed += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total = passed + failed
    print(f"\nTested {total} checks: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
