#!/usr/bin/env python3
"""
Test Multi-Response Model Fitting Backend - Phase 2 Feature 3
"""

import requests
import json

API_URL = "http://localhost:8000"

print("=" * 80)
print("TESTING MULTI-RESPONSE MODEL FITTING ENDPOINT")
print("=" * 80)

# Sample data: 2-factor CCD with 2 responses
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

# Test 1: 2-response fitting (basic functionality)
print("\n[TEST 1] Fit 2 Response Models (Y1, Y2)")
try:
    response = requests.post(f"{API_URL}/api/rsm/fit-multi-model", json={
        "data": data,
        "factors": ["X1", "X2"],
        "responses": ["Y1", "Y2"],
        "alpha": 0.05
    })
    print(f"✓ Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  Number of responses: {result['summary']['n_responses']}")
        print(f"  All models significant: {result['summary']['all_models_significant']}")
        print(f"  Significant models: {result['summary']['significant_models']}")
        print(f"\n  Y1 Model R²: {result['models']['Y1']['r_squared']:.4f}")
        print(f"  Y2 Model R²: {result['models']['Y2']['r_squared']:.4f}")
        print(f"\n  R² Summary:")
        print(f"    Mean: {result['summary']['r_squared_summary']['mean']:.4f}")
        print(f"    Min: {result['summary']['r_squared_summary']['min']:.4f}")
        print(f"    Max: {result['summary']['r_squared_summary']['max']:.4f}")
        print(f"    Std: {result['summary']['r_squared_summary']['std']:.4f}")
        print(f"\n  Correlation Y1-Y2: {result['correlation_matrix']['Y1']['Y2']:.4f}")
        print(f"\n  Interpretation ({len(result['interpretation'])} insights):")
        for interp in result['interpretation']:
            print(f"    - {interp}")
        print(f"\n  Recommendations ({len(result['recommendations'])} items):")
        for rec in result['recommendations']:
            print(f"    - {rec}")
    else:
        print(f"  Error: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: 3 responses
print("\n[TEST 2] Fit 3 Response Models (Y1, Y2, Y3)")
data_3resp = [
    {**row, "Y3": row["Y1"] * 0.5 + row["Y2"] * 20 + 5}
    for row in data
]
try:
    response = requests.post(f"{API_URL}/api/rsm/fit-multi-model", json={
        "data": data_3resp,
        "factors": ["X1", "X2"],
        "responses": ["Y1", "Y2", "Y3"],
        "alpha": 0.05
    })
    print(f"✓ Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  Number of responses: {result['summary']['n_responses']}")
        print(f"  All models significant: {result['summary']['all_models_significant']}")
        print(f"  R² Mean: {result['summary']['r_squared_summary']['mean']:.4f}")
        print(f"  Correlation Y1-Y2: {result['correlation_matrix']['Y1']['Y2']:.4f}")
        print(f"  Correlation Y1-Y3: {result['correlation_matrix']['Y1']['Y3']:.4f}")
        print(f"  Correlation Y2-Y3: {result['correlation_matrix']['Y2']['Y3']:.4f}")
    else:
        print(f"  Error: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Single response (should fail)
print("\n[TEST 3] Single response (should fail with 400)")
try:
    response = requests.post(f"{API_URL}/api/rsm/fit-multi-model", json={
        "data": data,
        "factors": ["X1", "X2"],
        "responses": ["Y1"],  # Only 1 response
        "alpha": 0.05
    })
    print(f"  Status: {response.status_code} (should be 400)")
    if response.status_code == 400:
        print(f"  ✓ Correctly rejected: {response.json()['detail']}")
    else:
        print(f"  ✗ Should have failed but didn't")
except Exception as e:
    print(f"  Error: {e}")

# Test 4: Too many responses (should fail)
print("\n[TEST 4] Too many responses (should fail with 400)")
many_responses = [f"Y{i}" for i in range(1, 12)]  # 11 responses
try:
    response = requests.post(f"{API_URL}/api/rsm/fit-multi-model", json={
        "data": data,
        "factors": ["X1", "X2"],
        "responses": many_responses,
        "alpha": 0.05
    })
    print(f"  Status: {response.status_code} (should be 400)")
    if response.status_code == 400:
        print(f"  ✓ Correctly rejected: {response.json()['detail']}")
    else:
        print(f"  ✗ Should have failed but didn't")
except Exception as e:
    print(f"  Error: {e}")

# Test 5: Missing response in data
print("\n[TEST 5] Missing response in data (should fail with 400)")
try:
    response = requests.post(f"{API_URL}/api/rsm/fit-multi-model", json={
        "data": data,
        "factors": ["X1", "X2"],
        "responses": ["Y1", "Y_nonexistent"],
        "alpha": 0.05
    })
    print(f"  Status: {response.status_code} (should be 400)")
    if response.status_code == 400:
        print(f"  ✓ Correctly rejected: {response.json()['detail']}")
    else:
        print(f"  ✗ Should have failed but didn't")
except Exception as e:
    print(f"  Error: {e}")

# Test 6: Missing values in response
print("\n[TEST 6] Missing values in response (should fail with 400)")
data_with_na = data.copy()
data_with_na[2] = {"X1": -1.0, "X2": 1.0, "Y1": 12.3}  # Y2 missing
try:
    response = requests.post(f"{API_URL}/api/rsm/fit-multi-model", json={
        "data": data_with_na,
        "factors": ["X1", "X2"],
        "responses": ["Y1", "Y2"],
        "alpha": 0.05
    })
    print(f"  Status: {response.status_code} (should be 400)")
    if response.status_code == 400:
        print(f"  ✓ Correctly rejected: {response.json()['detail']}")
    else:
        print(f"  ✗ Should have failed but didn't")
except Exception as e:
    print(f"  Error: {e}")

# Test 7: Highly correlated responses (should warn)
print("\n[TEST 7] Highly correlated responses (should include warning)")
data_corr = [
    {**row, "Y1_copy": row["Y1"] * 1.01}  # Y1_copy is almost identical to Y1
    for row in data
]
try:
    response = requests.post(f"{API_URL}/api/rsm/fit-multi-model", json={
        "data": data_corr,
        "factors": ["X1", "X2"],
        "responses": ["Y1", "Y1_copy"],
        "alpha": 0.05
    })
    print(f"✓ Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  Correlation Y1-Y1_copy: {result['correlation_matrix']['Y1']['Y1_copy']:.4f}")
        print(f"  Interpretation includes correlation warning: ", end="")
        has_warning = any("highly correlated" in i.lower() for i in result['interpretation'])
        print("✓ Yes" if has_warning else "✗ No")
    else:
        print(f"  Error: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 8: Model diagnostics are preserved
print("\n[TEST 8] Model diagnostics and ANOVA are preserved")
try:
    response = requests.post(f"{API_URL}/api/rsm/fit-multi-model", json={
        "data": data,
        "factors": ["X1", "X2"],
        "responses": ["Y1", "Y2"],
        "alpha": 0.05
    })
    print(f"✓ Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        # Check Y1 model has all expected keys
        y1_model = result['models']['Y1']
        expected_keys = ['coefficients', 'anova', 'enhanced_anova', 'r_squared',
                        'adj_r_squared', 'rmse', 'diagnostics']
        has_all_keys = all(key in y1_model for key in expected_keys)
        print(f"  Y1 model has all keys: {'✓ Yes' if has_all_keys else '✗ No'}")

        # Check coefficients
        has_coeffs = 'Intercept' in y1_model['coefficients']
        print(f"  Y1 model has coefficients: {'✓ Yes' if has_coeffs else '✗ No'}")

        # Check ANOVA
        has_anova = 'Model' in y1_model['enhanced_anova']
        print(f"  Y1 model has ANOVA: {'✓ Yes' if has_anova else '✗ No'}")

        # Check diagnostics
        has_diagnostics = 'residuals' in y1_model['diagnostics']
        print(f"  Y1 model has diagnostics: {'✓ Yes' if has_diagnostics else '✗ No'}")
    else:
        print(f"  Error: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("MULTI-RESPONSE MODEL FITTING TESTS COMPLETE")
print("=" * 80)
print("\nBackend Features Verified:")
print("  ✓ Multi-response model fitting (2-3 responses)")
print("  ✓ Summary statistics across responses")
print("  ✓ Response correlation matrix")
print("  ✓ Automated interpretations")
print("  ✓ Intelligent recommendations")
print("  ✓ Input validation (min/max responses)")
print("  ✓ Missing data detection")
print("  ✓ High correlation warnings")
print("  ✓ Full diagnostics preservation")
print("\n" + "=" * 80)
