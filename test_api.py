"""Test script that sends example properties to the prediction API.

Reads properties from data/future_unseen_examples.csv, posts them to the
/predict endpoint, and verifies the responses. Also tests edge cases like
invalid zipcodes.

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

    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    health = resp.json()
    assert health["status"] == "healthy", f"Unhealthy: {health}"
    assert health["model_loaded"] is True, f"Model not loaded: {health}"
    print(f"Health check: {health}")
    passed += 1

    # -----------------------------------------------------------------------
    # Predict with real examples
    # -----------------------------------------------------------------------
    df = pd.read_csv("data/future_unseen_examples.csv", dtype={"zipcode": str})

    for i, row in df.head(5).iterrows():
        payload = row.to_dict()
        resp = requests.post(f"{base_url}/predict", json=payload, timeout=10)

        try:
            assert resp.status_code == 200, f"Row {i}: status {resp.status_code}"
            body = resp.json()
            assert body["predicted_price"] > 0, f"Row {i}: non-positive price"
            assert "model_version" in body, f"Row {i}: missing model_version"
            assert "provider" in body, f"Row {i}: missing provider"
            print(
                f"  Row {i}: zipcode={payload['zipcode']}  "
                f"predicted_price=${body['predicted_price']:,.2f}  "
                f"model_version={body['model_version']}"
            )
            passed += 1
        except AssertionError as exc:
            print(f"  Row {i}: FAILED — {exc}")
            failed += 1

    # -----------------------------------------------------------------------
    # Edge case: invalid zipcode
    # -----------------------------------------------------------------------
    edge_payload = df.iloc[0].to_dict()
    edge_payload["zipcode"] = "00000"
    resp = requests.post(f"{base_url}/predict", json=edge_payload, timeout=10)

    try:
        assert resp.status_code == 422, f"Edge case: expected 422, got {resp.status_code}"
        print(f"\n  Edge case (zipcode='00000'): correctly returned 422")
        passed += 1
    except AssertionError as exc:
        print(f"\n  Edge case: FAILED — {exc}")
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
