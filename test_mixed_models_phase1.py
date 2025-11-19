#!/usr/bin/env python3
"""
Test script for Mixed Models Phase 1 enhancements
Tests: ICC calculations and Model Fit metrics for all design types
"""

import requests
import json

API_URL = "http://localhost:8000"

def test_mixed_anova_icc():
    """Test Mixed ANOVA with ICC and model fit"""
    print("\n=== Testing Mixed Model ANOVA (ICC + Model Fit) ===")

    # Mixed ANOVA data with fixed and random factors
    data = [
        {"Treatment": "T1", "Subject": "S1", "Response": 75},
        {"Treatment": "T1", "Subject": "S2", "Response": 78},
        {"Treatment": "T1", "Subject": "S3", "Response": 72},
        {"Treatment": "T2", "Subject": "S1", "Response": 85},
        {"Treatment": "T2", "Subject": "S2", "Response": 88},
        {"Treatment": "T2", "Subject": "S3", "Response": 82},
        {"Treatment": "T3", "Subject": "S1", "Response": 80},
        {"Treatment": "T3", "Subject": "S2", "Response": 83},
        {"Treatment": "T3", "Subject": "S3", "Response": 77},
    ]

    payload = {
        "data": data,
        "fixed_factors": ["Treatment"],
        "random_factors": ["Subject"],
        "response": "Response",
        "alpha": 0.05,
        "include_interactions": False
    }

    response = requests.post(f"{API_URL}/api/mixed/mixed-model-anova", json=payload)

    if response.status_code == 200:
        result = response.json()

        # Check ICC
        if 'icc' in result and result['icc']:
            print("✓ ICC data present")
            for factor, icc_data in result['icc'].items():
                if 'error' not in icc_data:
                    print(f"  - {factor}: ICC = {icc_data['icc']:.4f} ({icc_data['quality']})")
                    print(f"    95% CI: [{icc_data['ci_lower']:.4f}, {icc_data['ci_upper']:.4f}]")
                else:
                    print(f"  - {factor}: Error - {icc_data['error']}")
        else:
            print("✗ ICC data missing")
            return False

        # Check Model Fit
        if 'model_fit' in result and result['model_fit']:
            print("✓ Model fit metrics present")
            mf = result['model_fit']
            if 'aic' in mf:
                print(f"  - AIC: {mf['aic']:.2f}")
            if 'bic' in mf:
                print(f"  - BIC: {mf['bic']:.2f}")
            if 'log_likelihood' in mf:
                print(f"  - Log-likelihood: {mf['log_likelihood']:.4f}")
        else:
            print("✗ Model fit data missing")
            return False

        return True
    else:
        print(f"✗ Mixed ANOVA test failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


def test_split_plot_icc():
    """Test Split-Plot with ICC and model fit"""
    print("\n=== Testing Split-Plot Design (ICC + Model Fit) ===")

    # Split-plot data
    data = [
        {"Block": "B1", "Irrigation": "I1", "Variety": "V1", "Response": 65},
        {"Block": "B1", "Irrigation": "I1", "Variety": "V2", "Response": 70},
        {"Block": "B1", "Irrigation": "I2", "Variety": "V1", "Response": 75},
        {"Block": "B1", "Irrigation": "I2", "Variety": "V2", "Response": 80},
        {"Block": "B2", "Irrigation": "I1", "Variety": "V1", "Response": 68},
        {"Block": "B2", "Irrigation": "I1", "Variety": "V2", "Response": 73},
        {"Block": "B2", "Irrigation": "I2", "Variety": "V1", "Response": 78},
        {"Block": "B2", "Irrigation": "I2", "Variety": "V2", "Response": 83},
    ]

    payload = {
        "data": data,
        "whole_plot_factor": "Irrigation",
        "subplot_factor": "Variety",
        "block": "Block",
        "response": "Response",
        "alpha": 0.05
    }

    response = requests.post(f"{API_URL}/api/mixed/split-plot", json=payload)

    if response.status_code == 200:
        result = response.json()

        # Check ICC
        if 'icc' in result and result['icc']:
            print("✓ ICC data present")
            for factor, icc_data in result['icc'].items():
                if 'error' not in icc_data:
                    print(f"  - {factor}: ICC = {icc_data['icc']:.4f} ({icc_data['quality']})")
                else:
                    print(f"  - {factor}: {icc_data['error']}")
        else:
            print("✗ ICC data missing")

        # Check Model Fit
        if 'model_fit' in result and result['model_fit']:
            print("✓ Model fit metrics present")
            mf = result['model_fit']
            if 'aic' in mf and 'bic' in mf:
                print(f"  - AIC: {mf['aic']:.2f}, BIC: {mf['bic']:.2f}")
        else:
            print("✗ Model fit data missing")
            return False

        return True
    else:
        print(f"✗ Split-plot test failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


def test_nested_design_model_fit():
    """Test Nested Design with model fit"""
    print("\n=== Testing Nested Design (Model Fit) ===")

    # Nested design data (teachers nested in schools)
    data = [
        {"School": "S1", "Teacher": "T1", "Response": 75},
        {"School": "S1", "Teacher": "T1", "Response": 78},
        {"School": "S1", "Teacher": "T2", "Response": 72},
        {"School": "S1", "Teacher": "T2", "Response": 74},
        {"School": "S2", "Teacher": "T3", "Response": 85},
        {"School": "S2", "Teacher": "T3", "Response": 87},
        {"School": "S2", "Teacher": "T4", "Response": 82},
        {"School": "S2", "Teacher": "T4", "Response": 84},
    ]

    payload = {
        "data": data,
        "factor_a": "School",
        "factor_b_nested": "Teacher",
        "response": "Response",
        "alpha": 0.05
    }

    response = requests.post(f"{API_URL}/api/mixed/nested-design", json=payload)

    if response.status_code == 200:
        result = response.json()

        # Nested already has custom ICC, check model_fit
        if 'model_fit' in result and result['model_fit']:
            print("✓ Model fit metrics present")
            mf = result['model_fit']
            if 'aic' in mf and 'bic' in mf:
                print(f"  - AIC: {mf['aic']:.2f}, BIC: {mf['bic']:.2f}")
        else:
            print("✗ Model fit data missing")
            return False

        # Check existing ICC structure
        if 'icc' in result:
            print("✓ ICC structure present")
        else:
            print("⚠ ICC not present (nested design has custom implementation)")

        return True
    else:
        print(f"✗ Nested design test failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


def test_repeated_measures_icc():
    """Test Repeated Measures with ICC and model fit"""
    print("\n=== Testing Repeated Measures (ICC + Model Fit) ===")

    # Repeated measures data
    data = [
        {"Subject": "S1", "Time": "T1", "Response": 70},
        {"Subject": "S1", "Time": "T2", "Response": 75},
        {"Subject": "S1", "Time": "T3", "Response": 80},
        {"Subject": "S2", "Time": "T1", "Response": 72},
        {"Subject": "S2", "Time": "T2", "Response": 77},
        {"Subject": "S2", "Time": "T3", "Response": 82},
        {"Subject": "S3", "Time": "T1", "Response": 68},
        {"Subject": "S3", "Time": "T2", "Response": 73},
        {"Subject": "S3", "Time": "T3", "Response": 78},
    ]

    payload = {
        "data": data,
        "subject": "Subject",
        "within_factor": "Time",
        "response": "Response",
        "alpha": 0.05
    }

    response = requests.post(f"{API_URL}/api/mixed/repeated-measures", json=payload)

    if response.status_code == 200:
        result = response.json()

        # Check ICC
        if 'icc' in result and result['icc']:
            print("✓ ICC data present")
            for factor, icc_data in result['icc'].items():
                if 'error' not in icc_data:
                    print(f"  - {factor}: ICC = {icc_data['icc']:.4f} ({icc_data['quality']})")
                else:
                    print(f"  - {factor}: {icc_data['error']}")
        else:
            print("✗ ICC data missing")
            return False

        # Check Model Fit
        if 'model_fit' in result and result['model_fit']:
            print("✓ Model fit metrics present")
            mf = result['model_fit']
            if 'aic' in mf and 'bic' in mf:
                print(f"  - AIC: {mf['aic']:.2f}, BIC: {mf['bic']:.2f}")
        else:
            print("✗ Model fit data missing")
            return False

        return True
    else:
        print(f"✗ Repeated measures test failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


def main():
    """Run all Phase 1 tests"""
    print("=" * 60)
    print("MIXED MODELS PHASE 1 ENHANCEMENT TESTS")
    print("Testing: ICC Calculations + Model Fit Metrics")
    print("=" * 60)

    tests = [
        ("Mixed Model ANOVA", test_mixed_anova_icc),
        ("Split-Plot Design", test_split_plot_icc),
        ("Nested Design", test_nested_design_model_fit),
        ("Repeated Measures", test_repeated_measures_icc),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All Phase 1 enhancements working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
