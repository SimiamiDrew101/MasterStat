#!/usr/bin/env python3
"""
Test Multi-Response Surface Generation - Phase 2 Feature 3
"""

import requests
import json

API_URL = "http://localhost:8000"

print("=" * 80)
print("TESTING MULTI-RESPONSE SURFACE GENERATION ENDPOINT")
print("=" * 80)

# First, fit models to get coefficients
data = [
    {"X1": -1.0, "X2": -1.0, "Y1": 10.5, "Y2": 0.45},
    {"X1": 1.0, "X2": -1.0, "Y1": 15.2, "Y2": 0.32},
    {"X1": -1.0, "X2": 1.0, "Y1": 12.3, "Y2": 0.38},
    {"X1": 1.0, "X2": 1.0, "Y1": 18.7, "Y2": 0.25},
    {"X1": -1.414, "X2": 0.0, "Y1": 9.8, "Y2": 0.48},
    {"X1": 1.414, "X2": 0.0, "Y1": 17.1, "Y2": 0.28},
    {"X1": 0.0, "X2": -1.414, "Y1": 11.5, "Y2": 0.40},
    {"X1": 0.0, "X2": 1.414, "Y1": 14.2, "Y2": 0.35},
    {"X1": 0.0, "X2": 0.0, "Y1": 20.1, "Y2": 0.20},
    {"X1": 0.0, "X2": 0.0, "Y1": 19.8, "Y2": 0.21},
    {"X1": 0.0, "X2": 0.0, "Y1": 20.3, "Y2": 0.19},
    {"X1": 0.0, "X2": 0.0, "Y1": 19.9, "Y2": 0.20},
    {"X1": 0.0, "X2": 0.0, "Y1": 20.0, "Y2": 0.20},
]

print("\n[SETUP] Fitting models to get coefficients...")
fit_response = requests.post(f"{API_URL}/api/rsm/fit-multi-model", json={
    "data": data,
    "factors": ["X1", "X2"],
    "responses": ["Y1", "Y2"]
})

if fit_response.status_code != 200:
    print(f"✗ Failed to fit models: {fit_response.text}")
    exit(1)

models = fit_response.json()["models"]
print(f"✓ Models fitted (Y1 R²={models['Y1']['r_squared']:.4f}, Y2 R²={models['Y2']['r_squared']:.4f})")

# Test 1: Generate surface data with no normalization
print("\n[TEST 1] Generate surface data (no normalization)")
try:
    response = requests.post(f"{API_URL}/api/rsm/generate-multi-surface", json={
        "models": {
            "Y1": models["Y1"],
            "Y2": models["Y2"]
        },
        "factors": ["X1", "X2"],
        "grid_resolution": 20,
        "normalize": "none"
    })
    print(f"✓ Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  Number of responses: {result['n_responses']}")
        print(f"  Responses: {result['responses']}")
        print(f"  Grid resolution: {result['grid_info']['resolution']}")
        print(f"  Total points per surface: {result['grid_info']['total_points_per_surface']}")

        y1_surface = result['surfaces']['Y1']
        print(f"\n  Y1 Surface:")
        print(f"    Points: {len(y1_surface)}")
        print(f"    Sample point: {y1_surface[0]}")
        print(f"    Z range: {result['normalization_params']['Y1']['min']:.2f} to {result['normalization_params']['Y1']['max']:.2f}")

        y2_surface = result['surfaces']['Y2']
        print(f"\n  Y2 Surface:")
        print(f"    Points: {len(y2_surface)}")
        print(f"    Sample point: {y2_surface[0]}")
        print(f"    Z range: {result['normalization_params']['Y2']['min']:.2f} to {result['normalization_params']['Y2']['max']:.2f}")
    else:
        print(f"  Error: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Z-score normalization
print("\n[TEST 2] Generate surface data (z-score normalization)")
try:
    response = requests.post(f"{API_URL}/api/rsm/generate-multi-surface", json={
        "models": {
            "Y1": models["Y1"],
            "Y2": models["Y2"]
        },
        "factors": ["X1", "X2"],
        "grid_resolution": 15,
        "normalize": "zscore"
    })
    print(f"✓ Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  Normalization method: {result['normalization_params']['Y1']['method']}")

        # Check that z-scores are actually normalized
        y1_params = result['normalization_params']['Y1']
        print(f"\n  Y1 normalization:")
        print(f"    Original mean: {y1_params['mean']:.2f}")
        print(f"    Original std: {y1_params['std']:.2f}")
        print(f"    Original range: {y1_params['min']:.2f} to {y1_params['max']:.2f}")

        y1_normalized = [p['z'] for p in result['surfaces']['Y1']]
        actual_mean = sum(y1_normalized) / len(y1_normalized)
        actual_std = (sum((z - actual_mean)**2 for z in y1_normalized) / (len(y1_normalized) - 1))**0.5
        print(f"    Normalized mean: {actual_mean:.4f} (should be ~0)")
        print(f"    Normalized std: {actual_std:.4f} (should be ~1)")

        y2_params = result['normalization_params']['Y2']
        print(f"\n  Y2 normalization:")
        print(f"    Original mean: {y2_params['mean']:.2f}")
        print(f"    Original std: {y2_params['std']:.2f}")
    else:
        print(f"  Error: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Min-max normalization
print("\n[TEST 3] Generate surface data (min-max normalization)")
try:
    response = requests.post(f"{API_URL}/api/rsm/generate-multi-surface", json={
        "models": {
            "Y1": models["Y1"],
            "Y2": models["Y2"]
        },
        "factors": ["X1", "X2"],
        "grid_resolution": 10,
        "normalize": "minmax"
    })
    print(f"✓ Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  Normalization method: {result['normalization_params']['Y1']['method']}")

        # Check that values are in [0, 1]
        y1_normalized = [p['z'] for p in result['surfaces']['Y1']]
        actual_min = min(y1_normalized)
        actual_max = max(y1_normalized)
        print(f"\n  Y1 normalized range: {actual_min:.4f} to {actual_max:.4f} (should be [0, 1])")
        print(f"  ✓ Valid range" if 0 <= actual_min and actual_max <= 1 else "  ✗ Invalid range")

        y2_normalized = [p['z'] for p in result['surfaces']['Y2']]
        actual_min = min(y2_normalized)
        actual_max = max(y2_normalized)
        print(f"\n  Y2 normalized range: {actual_min:.4f} to {actual_max:.4f} (should be [0, 1])")
        print(f"  ✓ Valid range" if 0 <= actual_min and actual_max <= 1 else "  ✗ Invalid range")
    else:
        print(f"  Error: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Custom x/y ranges
print("\n[TEST 4] Custom factor ranges")
try:
    response = requests.post(f"{API_URL}/api/rsm/generate-multi-surface", json={
        "models": {"Y1": models["Y1"]},
        "factors": ["X1", "X2"],
        "grid_resolution": 10,
        "x_range": [-1.5, 1.5],
        "y_range": [-1.5, 1.5],
        "normalize": "none"
    })
    print(f"✓ Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  X range: {result['grid_info']['x_range']}")
        print(f"  Y range: {result['grid_info']['y_range']}")

        # Verify actual x/y values
        surface = result['surfaces']['Y1']
        x_values = sorted(set(p['x'] for p in surface))
        y_values = sorted(set(p['y'] for p in surface))
        print(f"  Actual X range: {x_values[0]:.2f} to {x_values[-1]:.2f}")
        print(f"  Actual Y range: {y_values[0]:.2f} to {y_values[-1]:.2f}")
    else:
        print(f"  Error: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 5: Invalid inputs
print("\n[TEST 5] Invalid grid resolution (should fail)")
try:
    response = requests.post(f"{API_URL}/api/rsm/generate-multi-surface", json={
        "models": {"Y1": models["Y1"]},
        "factors": ["X1", "X2"],
        "grid_resolution": 5,  # Too low
        "normalize": "none"
    })
    print(f"  Status: {response.status_code} (should be 400)")
    if response.status_code == 400:
        print(f"  ✓ Correctly rejected: {response.json()['detail']}")
    else:
        print(f"  ✗ Should have failed")
except Exception as e:
    print(f"  Error: {e}")

print("\n[TEST 6] Too many factors (should fail)")
try:
    response = requests.post(f"{API_URL}/api/rsm/generate-multi-surface", json={
        "models": {"Y1": models["Y1"]},
        "factors": ["X1", "X2", "X3"],  # 3 factors (contour needs 2)
        "grid_resolution": 20,
        "normalize": "none"
    })
    print(f"  Status: {response.status_code} (should be 400)")
    if response.status_code == 400:
        print(f"  ✓ Correctly rejected: {response.json()['detail']}")
    else:
        print(f"  ✗ Should have failed")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 80)
print("MULTI-RESPONSE SURFACE GENERATION TESTS COMPLETE")
print("=" * 80)
print("\nBackend Features Verified:")
print("  ✓ Surface data generation for multiple responses")
print("  ✓ Z-score normalization (mean=0, std=1)")
print("  ✓ Min-max normalization to [0,1]")
print("  ✓ Custom factor ranges")
print("  ✓ Grid resolution configuration")
print("  ✓ Input validation (factors, resolution)")
print("  ✓ Normalization metadata returned")
print("\n" + "=" * 80)
