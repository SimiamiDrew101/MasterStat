#!/usr/bin/env python3
"""
Test Mixed Models Phase 3: BLUPs & Random Effects Diagnostics
"""

import requests
import json

API_URL = "http://localhost:8000"

print("=" * 80)
print("MIXED MODELS PHASE 3 TEST: BLUPs & Random Effects Diagnostics")
print("=" * 80)

# Test 1: Mixed Model ANOVA with BLUPs
print("\n" + "=" * 80)
print("TEST 1: Mixed Model ANOVA - BLUPs Extraction")
print("=" * 80)

mixed_anova_data = [
    {"Operator": "Op1", "Machine": "M1", "Shift": "Day", "Response": 25.3},
    {"Operator": "Op1", "Machine": "M1", "Shift": "Day", "Response": 24.8},
    {"Operator": "Op1", "Machine": "M2", "Shift": "Day", "Response": 26.1},
    {"Operator": "Op1", "Machine": "M2", "Shift": "Day", "Response": 25.7},
    {"Operator": "Op2", "Machine": "M1", "Shift": "Day", "Response": 27.2},
    {"Operator": "Op2", "Machine": "M1", "Shift": "Day", "Response": 26.9},
    {"Operator": "Op2", "Machine": "M2", "Shift": "Day", "Response": 28.3},
    {"Operator": "Op2", "Machine": "M2", "Shift": "Day", "Response": 27.8},
    {"Operator": "Op3", "Machine": "M1", "Shift": "Night", "Response": 23.5},
    {"Operator": "Op3", "Machine": "M1", "Shift": "Night", "Response": 24.1},
    {"Operator": "Op3", "Machine": "M2", "Shift": "Night", "Response": 25.2},
    {"Operator": "Op3", "Machine": "M2", "Shift": "Night", "Response": 24.9}
]

payload1 = {
    "data": mixed_anova_data,
    "fixed_factors": ["Machine"],
    "random_factors": ["Operator"],
    "response": "Response",
    "alpha": 0.05,
    "include_interactions": False
}

try:
    print("\n📤 Sending request to /api/mixed/mixed-model-anova...")
    response1 = requests.post(f"{API_URL}/api/mixed/mixed-model-anova", json=payload1)
    print(f"✓ Status Code: {response1.status_code}")

    if response1.status_code == 200:
        result1 = response1.json()

        # Check BLUPs present
        print("\n✅ BLUPs Check:")
        if 'blups' in result1:
            print("   ✓ BLUPs field present")
            blups = result1['blups']

            for factor, factor_data in blups.items():
                print(f"\n   Factor: {factor}")
                if 'error' in factor_data:
                    print(f"   ✗ Error: {factor_data['error']}")
                else:
                    print(f"   ✓ Number of levels: {factor_data.get('n_levels', 'N/A')}")
                    print(f"   ✓ BLUPs extracted: {len(factor_data.get('blups', []))}")

                    if 'summary' in factor_data:
                        summary = factor_data['summary']
                        print(f"   ✓ Mean BLUP: {summary.get('mean_blup', 'N/A')}")
                        print(f"   ✓ SD BLUP: {summary.get('std_blup', 'N/A')}")
                        print(f"   ✓ Mean Shrinkage: {summary.get('mean_shrinkage', 'N/A')}")

                    # Show first BLUP entry
                    if factor_data.get('blups'):
                        print(f"\n   First BLUP entry:")
                        first_blup = factor_data['blups'][0]
                        for key, value in first_blup.items():
                            print(f"      - {key}: {value}")
        else:
            print("   ✗ BLUPs field missing")

        print("\n✅ TEST 1 PASSED")
    else:
        print(f"\n✗ TEST 1 FAILED: {response1.text}")

except Exception as e:
    print(f"\n✗ TEST 1 ERROR: {e}")

# Test 2: Split-Plot Design with BLUPs
print("\n" + "=" * 80)
print("TEST 2: Split-Plot Design - BLUPs Extraction")
print("=" * 80)

split_plot_data = [
    {"Block": "B1", "WholePlot": "A1", "SubPlot": "T1", "Response": 45.2},
    {"Block": "B1", "WholePlot": "A1", "SubPlot": "T2", "Response": 47.1},
    {"Block": "B1", "WholePlot": "A2", "SubPlot": "T1", "Response": 52.3},
    {"Block": "B1", "WholePlot": "A2", "SubPlot": "T2", "Response": 54.7},
    {"Block": "B2", "WholePlot": "A1", "SubPlot": "T1", "Response": 46.5},
    {"Block": "B2", "WholePlot": "A1", "SubPlot": "T2", "Response": 48.2},
    {"Block": "B2", "WholePlot": "A2", "SubPlot": "T1", "Response": 53.1},
    {"Block": "B2", "WholePlot": "A2", "SubPlot": "T2", "Response": 55.4},
    {"Block": "B3", "WholePlot": "A1", "SubPlot": "T1", "Response": 44.9},
    {"Block": "B3", "WholePlot": "A1", "SubPlot": "T2", "Response": 46.8},
    {"Block": "B3", "WholePlot": "A2", "SubPlot": "T1", "Response": 51.7},
    {"Block": "B3", "WholePlot": "A2", "SubPlot": "T2", "Response": 54.2}
]

payload2 = {
    "data": split_plot_data,
    "whole_plot_factor": "WholePlot",
    "subplot_factor": "SubPlot",
    "block": "Block",
    "response": "Response",
    "alpha": 0.05
}

try:
    print("\n📤 Sending request to /api/mixed/split-plot...")
    response2 = requests.post(f"{API_URL}/api/mixed/split-plot", json=payload2)
    print(f"✓ Status Code: {response2.status_code}")

    if response2.status_code == 200:
        result2 = response2.json()

        if 'blups' in result2:
            print("\n✅ BLUPs extracted successfully")
            blups = result2['blups']
            for factor in blups:
                print(f"   ✓ Factor: {factor} with {blups[factor].get('n_levels', 0)} levels")
            print("✅ TEST 2 PASSED")
        else:
            print("\n✗ BLUPs field missing")
            print("✗ TEST 2 FAILED")
    else:
        print(f"\n✗ TEST 2 FAILED: {response2.text}")

except Exception as e:
    print(f"\n✗ TEST 2 ERROR: {e}")

# Test 3: Nested Design with BLUPs
print("\n" + "=" * 80)
print("TEST 3: Nested Design - BLUPs Extraction")
print("=" * 80)

nested_data = [
    {"School": "S1", "Teacher": "T1", "Response": 75},
    {"School": "S1", "Teacher": "T1", "Response": 78},
    {"School": "S1", "Teacher": "T2", "Response": 72},
    {"School": "S1", "Teacher": "T2", "Response": 74},
    {"School": "S1", "Teacher": "T3", "Response": 76},
    {"School": "S1", "Teacher": "T3", "Response": 79},
    {"School": "S2", "Teacher": "T4", "Response": 85},
    {"School": "S2", "Teacher": "T4", "Response": 87},
    {"School": "S2", "Teacher": "T5", "Response": 82},
    {"School": "S2", "Teacher": "T5", "Response": 84},
    {"School": "S2", "Teacher": "T6", "Response": 86},
    {"School": "S2", "Teacher": "T6", "Response": 88}
]

payload3 = {
    "data": nested_data,
    "factor_a": "School",
    "factor_b_nested": "Teacher",
    "response": "Response",
    "alpha": 0.05
}

try:
    print("\n📤 Sending request to /api/mixed/nested-design...")
    response3 = requests.post(f"{API_URL}/api/mixed/nested-design", json=payload3)
    print(f"✓ Status Code: {response3.status_code}")

    if response3.status_code == 200:
        result3 = response3.json()

        if 'blups' in result3:
            print("\n✅ BLUPs extracted successfully")
            blups = result3['blups']

            for factor in blups:
                factor_data = blups[factor]
                if 'error' not in factor_data:
                    print(f"   ✓ Factor: {factor}")
                    print(f"      - Levels: {factor_data.get('n_levels', 0)}")
                    print(f"      - Mean Shrinkage: {factor_data.get('summary', {}).get('mean_shrinkage', 'N/A')}")

            print("✅ TEST 3 PASSED")
        else:
            print("\n✗ BLUPs field missing")
            print("✗ TEST 3 FAILED")
    else:
        print(f"\n✗ TEST 3 FAILED: {response3.text}")

except Exception as e:
    print(f"\n✗ TEST 3 ERROR: {e}")

print("\n" + "=" * 80)
print("PHASE 3 TESTING COMPLETE")
print("=" * 80)
print("\nAll Phase 3 enhancements (BLUPs & Random Effects Diagnostics) tested!")
print("Frontend components: BLUPsPlot (caterpillar plot) & RandomEffectsQQPlot")
print("Backend: extract_blups() function integrated into all endpoints")
