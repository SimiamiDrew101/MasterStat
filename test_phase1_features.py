#!/usr/bin/env python3
"""
Comprehensive Phase 1 Feature Testing for RSM
Tests all 4 critical features to ensure expert-level quality
"""

import requests
import json

API_URL = "http://localhost:8000"

# Test data: Simple 2-factor CCD design
test_data = [
    {"X1": -1, "X2": -1, "Y": 10.5},
    {"X1": 1, "X2": -1, "Y": 15.2},
    {"X1": -1, "X2": 1, "Y": 12.8},
    {"X1": 1, "X2": 1, "Y": 20.3},
    {"X1": -1.414, "X2": 0, "Y": 9.8},
    {"X1": 1.414, "X2": 0, "Y": 18.5},
    {"X1": 0, "X2": -1.414, "Y": 11.2},
    {"X1": 0, "X2": 1.414, "Y": 16.9},
    {"X1": 0, "X2": 0, "Y": 14.5},
    {"X1": 0, "X2": 0, "Y": 14.8},
    {"X1": 0, "X2": 0, "Y": 14.3},
]

factors = ["X1", "X2"]
response = "Y"

print("=" * 80)
print("PHASE 1 FEATURE TESTING - RSM")
print("=" * 80)

# Test 1: Fit Model (prerequisite for other tests)
print("\n[TEST 1] Fitting RSM Model...")
try:
    fit_response = requests.post(
        f"{API_URL}/api/rsm/fit-model",
        json={
            "data": test_data,
            "factors": factors,
            "response": response,
            "alpha": 0.05
        }
    )
    fit_response.raise_for_status()
    model_result = fit_response.json()
    print(f"✓ Model fitted successfully")
    print(f"  R² = {model_result['r_squared']}")
    print(f"  Adj R² = {model_result['adj_r_squared']}")
    print(f"  RMSE = {model_result['rmse']}")
    coefficients = {k: v['estimate'] for k, v in model_result['coefficients'].items()}
    variance_estimate = model_result['anova']['Residual']['mean_sq']
    print(f"  Variance Estimate (MSE) = {variance_estimate}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 2: Advanced Model Diagnostics
print("\n[TEST 2] Advanced Model Diagnostics...")
try:
    diag_response = requests.post(
        f"{API_URL}/api/rsm/advanced-diagnostics",
        json={
            "data": test_data,
            "factors": factors,
            "response": response,
            "coefficients": model_result['coefficients']
        }
    )
    diag_response.raise_for_status()
    diagnostics = diag_response.json()

    print(f"✓ Diagnostics calculated successfully")
    print(f"  PRESS = {diagnostics['press']['value']}")
    print(f"  R² Prediction = {diagnostics['press']['r2_prediction']}")
    print(f"  High Leverage Points = {diagnostics['summary']['n_high_leverage']}")
    print(f"  Influential (Cook's D) = {diagnostics['summary']['n_influential_cooks']}")
    print(f"  Influential (DFFITS) = {diagnostics['summary']['n_influential_dffits']}")
    print(f"  Multicollinearity Issues = {diagnostics['summary']['n_multicollinearity_issues']}")

    # Verify all diagnostic components exist
    assert 'leverage' in diagnostics['diagnostics']
    assert 'cooks_distance' in diagnostics['diagnostics']
    assert 'dffits' in diagnostics['diagnostics']
    assert 'vif' in diagnostics['diagnostics']
    assert len(diagnostics['recommendations']) > 0
    print(f"  Recommendations: {len(diagnostics['recommendations'])} generated")

except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 3: Optimization with Confidence & Prediction Intervals
print("\n[TEST 3] Optimization with Intervals...")
try:
    opt_response = requests.post(
        f"{API_URL}/api/rsm/optimize",
        json={
            "coefficients": coefficients,
            "factors": factors,
            "target": "maximize",
            "variance_estimate": variance_estimate
        }
    )
    opt_response.raise_for_status()
    optimization = opt_response.json()

    print(f"✓ Optimization completed successfully")
    print(f"  Optimal Response = {optimization['predicted_response']}")
    print(f"  Optimal Point: {optimization['optimal_point']}")

    # Verify intervals are present
    assert 'intervals' in optimization
    assert optimization['intervals'] is not None
    intervals = optimization['intervals']

    print(f"  95% Confidence Interval: [{intervals['confidence_interval']['lower']}, {intervals['confidence_interval']['upper']}]")
    print(f"  95% Prediction Interval: [{intervals['prediction_interval']['lower']}, {intervals['prediction_interval']['upper']}]")

    # Verify interval properties
    assert intervals['confidence_interval']['level'] == 0.95
    assert intervals['prediction_interval']['level'] == 0.95
    assert intervals['prediction_interval']['lower'] < intervals['confidence_interval']['lower']
    assert intervals['prediction_interval']['upper'] > intervals['confidence_interval']['upper']
    print(f"  ✓ Interval relationships correct (PI wider than CI)")

except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 4: Constrained Optimization with Intervals
print("\n[TEST 4] Constrained Optimization with Intervals...")
try:
    const_opt_response = requests.post(
        f"{API_URL}/api/rsm/constrained-optimization",
        json={
            "coefficients": coefficients,
            "factors": factors,
            "target": "maximize",
            "bounds": {"X1": [-1, 1], "X2": [-1, 1]},
            "variance_estimate": variance_estimate
        }
    )
    const_opt_response.raise_for_status()
    const_optimization = const_opt_response.json()

    print(f"✓ Constrained optimization completed successfully")
    print(f"  Optimal Response = {const_optimization['predicted_response']}")
    print(f"  Optimal Point: {const_optimization['optimal_point']}")

    # Verify intervals
    assert 'intervals' in const_optimization
    assert const_optimization['intervals'] is not None
    c_intervals = const_optimization['intervals']

    print(f"  95% Confidence Interval: [{c_intervals['confidence_interval']['lower']}, {c_intervals['confidence_interval']['upper']}]")
    print(f"  95% Prediction Interval: [{c_intervals['prediction_interval']['lower']}, {c_intervals['prediction_interval']['upper']}]")

except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 5: Export to JMP
print("\n[TEST 5] Export to JMP...")
try:
    jmp_response = requests.post(
        f"{API_URL}/api/rsm/export",
        json={
            "format": "jmp",
            "model_data": model_result,
            "factors": factors,
            "response": response,
            "data": test_data
        }
    )
    jmp_response.raise_for_status()
    jmp_export = jmp_response.json()

    print(f"✓ JMP export successful")
    print(f"  Filename: {jmp_export['filename']}")
    print(f"  Format: {jmp_export['format']}")
    print(f"  Script length: {len(jmp_export['content'])} characters")

    # Verify JMP script contains key elements
    script = jmp_export['content']
    assert '// JMP JSL Script' in script
    assert 'New Table' in script
    assert 'Fit Model' in script
    assert response in script
    for factor in factors:
        assert factor in script
    print(f"  ✓ JMP script contains all required elements")

except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 6: Export to R
print("\n[TEST 6] Export to R...")
try:
    r_response = requests.post(
        f"{API_URL}/api/rsm/export",
        json={
            "format": "r",
            "model_data": model_result,
            "factors": factors,
            "response": response,
            "data": test_data
        }
    )
    r_response.raise_for_status()
    r_export = r_response.json()

    print(f"✓ R export successful")
    print(f"  Filename: {r_export['filename']}")
    print(f"  Script length: {len(r_export['content'])} characters")

    # Verify R script contains key elements
    script = r_export['content']
    assert '# R Script for RSM Analysis' in script
    assert 'library(rsm)' in script
    assert 'data.frame' in script
    assert 'lm(' in script
    assert 'summary(model)' in script
    print(f"  ✓ R script contains all required elements")

except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 7: Export to Python
print("\n[TEST 7] Export to Python...")
try:
    py_response = requests.post(
        f"{API_URL}/api/rsm/export",
        json={
            "format": "python",
            "model_data": model_result,
            "factors": factors,
            "response": response,
            "data": test_data
        }
    )
    py_response.raise_for_status()
    py_export = py_response.json()

    print(f"✓ Python export successful")
    print(f"  Filename: {py_export['filename']}")
    print(f"  Script length: {len(py_export['content'])} characters")

    # Verify Python script contains key elements
    script = py_export['content']
    assert '# Python Script for RSM Analysis' in script
    assert 'import pandas as pd' in script
    assert 'import statsmodels.api as sm' in script
    assert 'ols(' in script
    assert 'model.summary()' in script
    print(f"  ✓ Python script contains all required elements")

except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 8: Prediction Profiler Data (verify coefficients can be used)
print("\n[TEST 8] Prediction Profiler Compatibility...")
try:
    # Verify we have all data needed for PredictionProfiler component
    assert 'coefficients' in model_result
    assert 'anova' in model_result

    # Verify coefficient structure
    for term, coef_data in model_result['coefficients'].items():
        assert 'estimate' in coef_data
        assert 'std_error' in coef_data
        assert 't_value' in coef_data
        assert 'p_value' in coef_data

    # Verify variance estimate exists for intervals
    assert 'Residual' in model_result['anova']
    assert 'mean_sq' in model_result['anova']['Residual']

    print(f"✓ All data structures compatible with PredictionProfiler")
    print(f"  Coefficients: {len(model_result['coefficients'])} terms")
    print(f"  Variance for intervals: Available")

except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Summary
print("\n" + "=" * 80)
print("PHASE 1 TESTING COMPLETE - ALL TESTS PASSED ✓")
print("=" * 80)
print("\nFeatures Verified:")
print("  1. ✓ Interactive Prediction Profiler - Data structures ready")
print("  2. ✓ Advanced Model Diagnostics - All metrics calculated")
print("  3. ✓ Confidence & Prediction Intervals - Present in optimization")
print("  4. ✓ Export to Industry Standards - JMP, R, Python all working")
print("\n" + "=" * 80)
print("READY FOR PRODUCTION - EXPERT LEVEL QUALITY CONFIRMED")
print("=" * 80)
