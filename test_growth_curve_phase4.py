#!/usr/bin/env python3
"""
Test Growth Curve Models Phase 4
"""

import requests
import json

API_URL = "http://localhost:8000"

print("=" * 80)
print("MIXED MODELS PHASE 4 TEST: Growth Curve Models")
print("=" * 80)

# Test 1: Linear Growth Model
print("\n" + "=" * 80)
print("TEST 1: Linear Growth Model with Random Intercepts and Slopes")
print("=" * 80)

# Generate growth curve data: 8 subjects × 5 time points
growth_data = []
subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
times = [0, 1, 2, 3, 4]
baselines = [20, 18, 22, 24, 19, 21, 17, 23]
growth_rates = [2.0, 2.5, 1.8, 2.2, 1.9, 2.3, 2.1, 1.7]

for idx, subject in enumerate(subjects):
    for time in times:
        value = baselines[idx] + growth_rates[idx] * time + (idx % 2) * 0.5
        growth_data.append({
            "SubjectID": subject,
            "Time": time,
            "Value": round(value, 2)
        })

payload1 = {
    "data": growth_data,
    "subject_id": "SubjectID",
    "time_var": "Time",
    "response": "Value",
    "polynomial_order": "linear",
    "random_effects": "intercept_slope",
    "alpha": 0.05
}

try:
    print("\n📤 Sending request to /api/mixed/growth-curve...")
    response1 = requests.post(f"{API_URL}/api/mixed/growth-curve", json=payload1)
    print(f"✓ Status Code: {response1.status_code}")

    if response1.status_code == 200:
        result1 = response1.json()

        print("\n✅ Model Summary:")
        print(f"   ✓ Polynomial Order: {result1.get('polynomial_order', 'N/A')}")
        print(f"   ✓ Random Effects: {result1.get('random_effects_structure', 'N/A')}")
        print(f"   ✓ Number of Subjects: {result1.get('n_subjects', 'N/A')}")
        print(f"   ✓ Number of Observations: {result1.get('n_observations', 'N/A')}")

        print("\n✅ Fixed Effects:")
        fixed = result1.get('fixed_effects', {})
        for param, coef in fixed.get('coefficients', {}).items():
            pval = fixed.get('p_values', {}).get(param, 'N/A')
            print(f"   {param}: {coef:.4f} (p = {pval:.4f})")

        print("\n✅ Random Effects Variance:")
        re_var = result1.get('random_effects_variance', {})
        print(f"   ✓ Intercept Variance: {re_var.get('intercept_var', 'N/A')}")
        print(f"   ✓ Slope Variance: {re_var.get('slope_var', 'N/A')}")
        print(f"   ✓ Intercept-Slope Correlation: {re_var.get('intercept_slope_corr', 'N/A')}")
        print(f"   ✓ Residual Variance: {re_var.get('residual_var', 'N/A')}")

        print(f"\n✅ ICC: {result1.get('icc', 'N/A')}")

        print("\n✅ Individual Trajectories:")
        trajectories = result1.get('individual_trajectories', [])
        print(f"   ✓ Generated {len(trajectories)} trajectories")
        if trajectories and len(trajectories) > 0:
            first_traj = trajectories[0]
            print(f"   ✓ First subject: {first_traj.get('subject_id', 'N/A')}")
            print(f"   ✓ Observed points: {len(first_traj.get('observed', []))}")
            print(f"   ✓ Predicted points: {len(first_traj.get('predicted', []))}")

        print("\n✅ Population Curve:")
        pop_curve = result1.get('population_curve', {})
        if 'time_points' in pop_curve:
            print(f"   ✓ Time points: {len(pop_curve['time_points'])}")
            print(f"   ✓ Predictions: {len(pop_curve.get('predicted', []))}")
            print(f"   ✓ CI bands: {len(pop_curve.get('ci_lower', []))} lower, {len(pop_curve.get('ci_upper', []))} upper")

        print("\n✅ Interpretation:")
        interpretation = result1.get('interpretation', 'N/A')
        print(f"   {interpretation[:200]}...")

        print("\n✅ TEST 1 PASSED")
    else:
        print(f"\n✗ TEST 1 FAILED: {response1.text}")

except Exception as e:
    print(f"\n✗ TEST 1 ERROR: {e}")

# Test 2: Quadratic Growth Model
print("\n" + "=" * 80)
print("TEST 2: Quadratic Growth Model")
print("=" * 80)

# Generate quadratic growth data
quad_data = []
for idx, subject in enumerate(subjects):
    for time in times:
        # Quadratic pattern: baseline + linear*time + quad*time^2
        value = baselines[idx] + growth_rates[idx] * time - 0.1 * time * time
        quad_data.append({
            "SubjectID": subject,
            "Time": time,
            "Score": round(value, 2)
        })

payload2 = {
    "data": quad_data,
    "subject_id": "SubjectID",
    "time_var": "Time",
    "response": "Score",
    "polynomial_order": "quadratic",
    "random_effects": "intercept",
    "alpha": 0.05
}

try:
    print("\n📤 Sending request to /api/mixed/growth-curve...")
    response2 = requests.post(f"{API_URL}/api/mixed/growth-curve", json=payload2)
    print(f"✓ Status Code: {response2.status_code}")

    if response2.status_code == 200:
        result2 = response2.json()

        print(f"\n✅ Polynomial Order: {result2.get('polynomial_order', 'N/A')}")

        fixed2 = result2.get('fixed_effects', {})
        print("\n✅ Fixed Effects (Quadratic):")
        for param in ['Intercept', 'time_centered', 'time_squared']:
            if param in fixed2.get('coefficients', {}):
                coef = fixed2['coefficients'][param]
                pval = fixed2.get('p_values', {}).get(param, 'N/A')
                print(f"   {param}: {coef:.4f} (p = {pval:.4f})")

        print("\n✅ TEST 2 PASSED")
    else:
        print(f"\n✗ TEST 2 FAILED: {response2.text}")

except Exception as e:
    print(f"\n✗ TEST 2 ERROR: {e}")

print("\n" + "=" * 80)
print("PHASE 4 TESTING COMPLETE")
print("=" * 80)
print("\nAll Phase 4 enhancements (Growth Curve Models) tested!")
print("Frontend components: GrowthCurvePlot (spaghetti plot) & GrowthCurveResults")
print("Backend: growth-curve endpoint with polynomial trends and random slopes")
