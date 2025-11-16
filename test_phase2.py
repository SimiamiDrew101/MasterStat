#!/usr/bin/env python3
"""
Test script for Block Designs Phase 2 enhancements
Tests: ANCOVA, Missing Data Handling, Crossover Designs, Incomplete Block Designs
"""

import requests
import json

API_URL = "http://localhost:8000"

def test_ancova():
    """Test ANCOVA with covariate support"""
    print("\n=== Testing ANCOVA ===")

    # RCBD data with covariate
    data = [
        {"block": "B1", "treatment": "T1", "Baseline": 45, "Response": 78},
        {"block": "B1", "treatment": "T2", "Baseline": 52, "Response": 85},
        {"block": "B1", "treatment": "T3", "Baseline": 48, "Response": 80},
        {"block": "B2", "treatment": "T1", "Baseline": 50, "Response": 82},
        {"block": "B2", "treatment": "T2", "Baseline": 55, "Response": 90},
        {"block": "B2", "treatment": "T3", "Baseline": 47, "Response": 79},
        {"block": "B3", "treatment": "T1", "Baseline": 46, "Response": 77},
        {"block": "B3", "treatment": "T2", "Baseline": 53, "Response": 88},
        {"block": "B3", "treatment": "T3", "Baseline": 49, "Response": 81},
    ]

    payload = {
        "data": data,
        "treatment": "treatment",
        "block": "block",
        "response": "Response",
        "covariate": "Baseline",
        "alpha": 0.05,
        "random_blocks": False
    }

    response = requests.post(f"{API_URL}/api/block-designs/rcbd", json=payload)

    if response.status_code == 200:
        result = response.json()
        if 'ancova' in result:
            print("✓ ANCOVA analysis successful")
            print(f"  - Slopes homogeneous: {result['ancova']['slopes_homogeneous']}")
            print(f"  - Covariate coefficient: {result['ancova']['covariate_coefficient']:.4f}")
            print(f"  - Model R²: {result['ancova']['model_r_squared']:.4f}")
            return True
        else:
            print("✗ ANCOVA data missing from response")
            return False
    else:
        print(f"✗ ANCOVA test failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


def test_missing_data():
    """Test Missing Data Handling with imputation"""
    print("\n=== Testing Missing Data Handling ===")

    # RCBD data with missing values (using empty string or null)
    data = [
        {"block": "B1", "treatment": "T1", "Response": 78},
        {"block": "B1", "treatment": "T2", "Response": None},  # Missing
        {"block": "B1", "treatment": "T3", "Response": 80},
        {"block": "B2", "treatment": "T1", "Response": 82},
        {"block": "B2", "treatment": "T2", "Response": 90},
        {"block": "B2", "treatment": "T3", "Response": None},  # Missing
        {"block": "B3", "treatment": "T1", "Response": 77},
        {"block": "B3", "treatment": "T2", "Response": 88},
        {"block": "B3", "treatment": "T3", "Response": 81},
    ]

    payload = {
        "data": data,
        "treatment": "treatment",
        "block": "block",
        "response": "Response",
        "alpha": 0.05,
        "random_blocks": False,
        "imputation_method": "mean"
    }

    response = requests.post(f"{API_URL}/api/block-designs/rcbd", json=payload)

    if response.status_code == 200:
        result = response.json()
        if 'missing_data' in result:
            missing_data = result['missing_data']
            if missing_data['pattern']['has_missing']:
                print("✓ Missing data detected and handled")
                print(f"  - Missing observations: {missing_data['pattern']['n_missing']}")
                print(f"  - Imputation method: {missing_data.get('method_used', 'none')}")
                if missing_data.get('mcar_test'):
                    print(f"  - MCAR test p-value: {missing_data['mcar_test'].get('p_value', 'N/A')}")
                return True
            else:
                print("✗ Missing data not detected in test data")
                return False
        else:
            print("✗ Missing data analysis not present in response")
            return False
    else:
        print(f"✗ Missing data test failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


def test_crossover_generate():
    """Test Crossover Design generation"""
    print("\n=== Testing Crossover Design Generation ===")

    payload = {
        "n_subjects": 8,
        "n_treatments": 2,
        "design_type": "2x2"
    }

    response = requests.post(f"{API_URL}/api/block-designs/crossover/generate", json=payload)

    if response.status_code == 200:
        result = response.json()
        print("✓ Crossover design generated successfully")
        print(f"  - Design type: {result['design_type']}")
        print(f"  - Subjects: {result['n_subjects']}")
        print(f"  - Periods: {result['n_periods']}")
        print(f"  - Total runs: {result['n_runs']}")
        return True
    else:
        print(f"✗ Crossover generation failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


def test_crossover_analyze():
    """Test Crossover Design analysis"""
    print("\n=== Testing Crossover Design Analysis ===")

    # 2x2 crossover data
    data = [
        {"subject": "S1", "sequence": "AB", "period": "1", "treatment": "A", "Response": 75},
        {"subject": "S1", "sequence": "AB", "period": "2", "treatment": "B", "Response": 82},
        {"subject": "S2", "sequence": "AB", "period": "1", "treatment": "A", "Response": 78},
        {"subject": "S2", "sequence": "AB", "period": "2", "treatment": "B", "Response": 85},
        {"subject": "S3", "sequence": "BA", "period": "1", "treatment": "B", "Response": 80},
        {"subject": "S3", "sequence": "BA", "period": "2", "treatment": "A", "Response": 73},
        {"subject": "S4", "sequence": "BA", "period": "1", "treatment": "B", "Response": 83},
        {"subject": "S4", "sequence": "BA", "period": "2", "treatment": "A", "Response": 76},
    ]

    payload = {
        "data": data,
        "subject": "subject",
        "period": "period",
        "treatment": "treatment",
        "sequence": "sequence",
        "response": "Response",
        "alpha": 0.05
    }

    response = requests.post(f"{API_URL}/api/block-designs/crossover/analyze", json=payload)

    if response.status_code == 200:
        result = response.json()
        print("✓ Crossover analysis successful")
        print(f"  - Design type: {result['design_type']}")
        print(f"  - Treatment effect significant: {result['treatment_effect']['significant']}")
        print(f"  - Period effect significant: {result['period_effect']['significant']}")
        if 'carryover_effect' in result and not result['carryover_effect'].get('error'):
            print(f"  - Carryover effect significant: {result['carryover_effect']['significant']}")
        return True
    else:
        print(f"✗ Crossover analysis failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


def test_incomplete_bib_generate():
    """Test BIB design generation"""
    print("\n=== Testing BIB Design Generation ===")

    payload = {
        "n_treatments": 5,
        "block_size": 3
    }

    response = requests.post(f"{API_URL}/api/block-designs/incomplete/generate/bib", json=payload)

    if response.status_code == 200:
        result = response.json()
        print("✓ BIB design generated successfully")
        print(f"  - Treatments (v): {result['n_treatments']}")
        print(f"  - Blocks (b): {result['n_blocks']}")
        print(f"  - Block size (k): {result['block_size']}")
        print(f"  - Replications (r): {result['replications']}")
        print(f"  - Lambda (λ): {result['lambda']}")
        print(f"  - Efficiency: {result['efficiency']:.2%}")
        return True
    else:
        print(f"✗ BIB generation failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


def test_incomplete_youden_generate():
    """Test Youden Square generation"""
    print("\n=== Testing Youden Square Generation ===")

    payload = {
        "n_treatments": 5,
        "n_rows": 4,
        "n_columns": 3
    }

    response = requests.post(f"{API_URL}/api/block-designs/incomplete/generate/youden", json=payload)

    if response.status_code == 200:
        result = response.json()
        print("✓ Youden Square generated successfully")
        print(f"  - Treatments: {result['n_treatments']}")
        print(f"  - Rows: {result['n_rows']}")
        print(f"  - Columns: {result['n_columns']}")
        print(f"  - Total runs: {result['n_runs']}")
        return True
    else:
        print(f"✗ Youden generation failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


def test_incomplete_analyze():
    """Test Incomplete Block analysis"""
    print("\n=== Testing Incomplete Block Analysis ===")

    # BIB data (5 treatments, block size 3)
    data = [
        {"block": "B1", "treatment": "T1", "Response": 75},
        {"block": "B1", "treatment": "T2", "Response": 82},
        {"block": "B1", "treatment": "T3", "Response": 78},
        {"block": "B2", "treatment": "T1", "Response": 77},
        {"block": "B2", "treatment": "T4", "Response": 85},
        {"block": "B2", "treatment": "T5", "Response": 80},
        {"block": "B3", "treatment": "T2", "Response": 84},
        {"block": "B3", "treatment": "T3", "Response": 79},
        {"block": "B3", "treatment": "T4", "Response": 86},
        {"block": "B4", "treatment": "T1", "Response": 76},
        {"block": "B4", "treatment": "T3", "Response": 81},
        {"block": "B4", "treatment": "T5", "Response": 82},
    ]

    payload = {
        "data": data,
        "treatment": "treatment",
        "block": "block",
        "response": "Response",
        "alpha": 0.05
    }

    response = requests.post(f"{API_URL}/api/block-designs/incomplete/analyze", json=payload)

    if response.status_code == 200:
        result = response.json()
        print("✓ Incomplete block analysis successful")
        print(f"  - Design incomplete: {result['design_info']['is_incomplete']}")
        print(f"  - Design balanced: {result['design_info']['is_balanced']}")
        print(f"  - Treatment effect significant: {result['treatment_effect']['significant']}")
        print(f"  - Efficiency: {result['efficiency']:.2%}")
        return True
    else:
        print(f"✗ Incomplete block analysis failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


def main():
    """Run all Phase 2 tests"""
    print("=" * 60)
    print("BLOCK DESIGNS PHASE 2 FEATURE TESTS")
    print("=" * 60)

    tests = [
        ("ANCOVA", test_ancova),
        ("Missing Data Handling", test_missing_data),
        ("Crossover Generation", test_crossover_generate),
        ("Crossover Analysis", test_crossover_analyze),
        ("BIB Generation", test_incomplete_bib_generate),
        ("Youden Generation", test_incomplete_youden_generate),
        ("Incomplete Block Analysis", test_incomplete_analyze),
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
        print("\n🎉 All Phase 2 features working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
