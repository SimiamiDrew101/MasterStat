from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import stats

router = APIRouter()

# ============================================================================
# Pydantic Models
# ============================================================================

class ControlChartRequest(BaseModel):
    data: List[float] = Field(..., description="Process measurements")
    chart_type: str = Field(..., description="Type of control chart (i-mr, xbar-r, xbar-s, p, c, u)")
    subgroup_size: Optional[int] = Field(1, description="Subgroup size for grouped charts")
    spec_limits: Optional[Dict[str, float]] = Field(None, description="Specification limits (lsl, usl, target)")

class CapabilityRequest(BaseModel):
    data: List[float] = Field(..., description="Process measurements")
    spec_limits: Dict[str, float] = Field(..., description="Specification limits (must have lsl and/or usl)")
    target: Optional[float] = Field(None, description="Target value for Cpm calculation")
    confidence_level: float = Field(0.95, description="Confidence level for intervals")

class CUSUMRequest(BaseModel):
    data: List[float] = Field(..., description="Process measurements")
    target: Optional[float] = Field(None, description="Target mean (defaults to sample mean)")
    k: float = Field(0.5, description="Slack value (allowance) in sigma units")
    h: float = Field(5.0, description="Decision interval (threshold) in sigma units")
    sigma: Optional[float] = Field(None, description="Known process std dev (estimated if not provided)")

class EWMARequest(BaseModel):
    data: List[float] = Field(..., description="Process measurements")
    lambda_: float = Field(0.2, description="Smoothing parameter (0 < lambda <= 1)")
    L: float = Field(3.0, description="Control limit width in sigma units")
    target: Optional[float] = Field(None, description="Target mean (defaults to sample mean)")
    sigma: Optional[float] = Field(None, description="Known process std dev (estimated if not provided)")

# ============================================================================
# Control Chart Constants (ASQ/ASTM)
# ============================================================================

# Constants for X-bar and R charts
D3_CONSTANTS = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
D4_CONSTANTS = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
A2_CONSTANTS = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}

# Constants for X-bar and S charts
B3_CONSTANTS = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0.029, 7: 0.113, 8: 0.179, 9: 0.232, 10: 0.276}
B4_CONSTANTS = {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716}
A3_CONSTANTS = {2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182, 8: 1.099, 9: 1.032, 10: 0.975}

# ============================================================================
# Control Chart Generation Functions
# ============================================================================

def generate_i_mr_chart(data: np.ndarray) -> Dict:
    """
    Individuals and Moving Range (I-MR) chart
    For individual measurements (subgroup size = 1)
    """
    n = len(data)

    # I chart (individuals)
    mean_i = np.mean(data)

    # Moving ranges (absolute differences between consecutive points)
    mr = np.abs(np.diff(data))
    mean_mr = np.mean(mr)

    # Control limits for I chart (using moving range)
    # UCL/LCL = mean ± 2.66 × MR-bar
    ucl_i = mean_i + 2.66 * mean_mr
    lcl_i = mean_i - 2.66 * mean_mr

    # Control limits for MR chart
    # UCL = 3.27 × MR-bar, LCL = 0
    ucl_mr = 3.27 * mean_mr
    lcl_mr = 0

    return {
        "chart_type": "i-mr",
        "i_chart": {
            "values": data.tolist(),
            "center_line": float(mean_i),
            "ucl": float(ucl_i),
            "lcl": float(lcl_i),
            "sigma": float(mean_mr / 1.128)  # Estimate of process sigma
        },
        "mr_chart": {
            "values": mr.tolist(),
            "center_line": float(mean_mr),
            "ucl": float(ucl_mr),
            "lcl": float(lcl_mr)
        }
    }

def generate_xbar_r_chart(data: np.ndarray, subgroup_size: int) -> Dict:
    """
    X-bar and R chart for subgrouped data
    Monitors process mean and variability using range
    """
    if subgroup_size < 2 or subgroup_size > 10:
        raise ValueError("Subgroup size must be between 2 and 10")

    # Reshape data into subgroups
    n_subgroups = len(data) // subgroup_size
    if n_subgroups < 2:
        raise ValueError("Need at least 2 subgroups")

    subgroups = data[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)

    # Calculate subgroup means and ranges
    xbar = np.mean(subgroups, axis=1)
    ranges = np.ptp(subgroups, axis=1)  # ptp = peak-to-peak (max - min)

    # Overall mean and average range
    xbar_bar = np.mean(xbar)
    r_bar = np.mean(ranges)

    # Get constants
    A2 = A2_CONSTANTS[subgroup_size]
    D3 = D3_CONSTANTS[subgroup_size]
    D4 = D4_CONSTANTS[subgroup_size]

    # X-bar chart limits
    ucl_xbar = xbar_bar + A2 * r_bar
    lcl_xbar = xbar_bar - A2 * r_bar

    # R chart limits
    ucl_r = D4 * r_bar
    lcl_r = D3 * r_bar

    # Estimate process sigma
    d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
    sigma = r_bar / d2[subgroup_size]

    return {
        "chart_type": "xbar-r",
        "xbar_chart": {
            "values": xbar.tolist(),
            "center_line": float(xbar_bar),
            "ucl": float(ucl_xbar),
            "lcl": float(lcl_xbar),
            "sigma": float(sigma)
        },
        "r_chart": {
            "values": ranges.tolist(),
            "center_line": float(r_bar),
            "ucl": float(ucl_r),
            "lcl": float(lcl_r)
        },
        "subgroup_size": subgroup_size,
        "n_subgroups": n_subgroups
    }

def generate_xbar_s_chart(data: np.ndarray, subgroup_size: int) -> Dict:
    """
    X-bar and S chart for subgrouped data
    Monitors process mean and variability using standard deviation
    Preferred over R chart for subgroup sizes > 10
    """
    if subgroup_size < 2 or subgroup_size > 10:
        raise ValueError("Subgroup size must be between 2 and 10")

    # Reshape data into subgroups
    n_subgroups = len(data) // subgroup_size
    if n_subgroups < 2:
        raise ValueError("Need at least 2 subgroups")

    subgroups = data[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)

    # Calculate subgroup means and standard deviations
    xbar = np.mean(subgroups, axis=1)
    s = np.std(subgroups, axis=1, ddof=1)  # Sample standard deviation

    # Overall mean and average standard deviation
    xbar_bar = np.mean(xbar)
    s_bar = np.mean(s)

    # Get constants
    A3 = A3_CONSTANTS[subgroup_size]
    B3 = B3_CONSTANTS[subgroup_size]
    B4 = B4_CONSTANTS[subgroup_size]

    # X-bar chart limits
    ucl_xbar = xbar_bar + A3 * s_bar
    lcl_xbar = xbar_bar - A3 * s_bar

    # S chart limits
    ucl_s = B4 * s_bar
    lcl_s = B3 * s_bar

    # Estimate process sigma
    c4 = np.sqrt(2/(subgroup_size-1)) * (np.math.gamma(subgroup_size/2) / np.math.gamma((subgroup_size-1)/2))
    sigma = s_bar / c4

    return {
        "chart_type": "xbar-s",
        "xbar_chart": {
            "values": xbar.tolist(),
            "center_line": float(xbar_bar),
            "ucl": float(ucl_xbar),
            "lcl": float(lcl_xbar),
            "sigma": float(sigma)
        },
        "s_chart": {
            "values": s.tolist(),
            "center_line": float(s_bar),
            "ucl": float(ucl_s),
            "lcl": float(lcl_s)
        },
        "subgroup_size": subgroup_size,
        "n_subgroups": n_subgroups
    }

def generate_p_chart(data: np.ndarray, subgroup_size: int) -> Dict:
    """
    P chart for proportion defective
    Data should be number of defects in each subgroup
    """
    if subgroup_size < 1:
        raise ValueError("Subgroup size must be >= 1")

    # Convert counts to proportions
    p = data / subgroup_size
    p_bar = np.mean(p)

    # Control limits (varying if subgroup sizes differ, constant if same)
    # UCL/LCL = p-bar ± 3 × sqrt(p-bar × (1 - p-bar) / n)
    std_p = np.sqrt(p_bar * (1 - p_bar) / subgroup_size)
    ucl_p = p_bar + 3 * std_p
    lcl_p = p_bar - 3 * std_p

    # LCL can't be negative
    lcl_p = max(0, lcl_p)

    # UCL can't exceed 1
    ucl_p = min(1, ucl_p)

    return {
        "chart_type": "p",
        "p_chart": {
            "values": p.tolist(),
            "center_line": float(p_bar),
            "ucl": float(ucl_p),
            "lcl": float(lcl_p),
            "subgroup_size": subgroup_size
        }
    }

def generate_c_chart(data: np.ndarray) -> Dict:
    """
    C chart for count of defects per unit
    Data should be number of defects in each inspection unit (constant sample size)
    """
    c_bar = np.mean(data)

    # Control limits
    # UCL/LCL = c-bar ± 3 × sqrt(c-bar)
    std_c = np.sqrt(c_bar)
    ucl_c = c_bar + 3 * std_c
    lcl_c = c_bar - 3 * std_c

    # LCL can't be negative
    lcl_c = max(0, lcl_c)

    return {
        "chart_type": "c",
        "c_chart": {
            "values": data.tolist(),
            "center_line": float(c_bar),
            "ucl": float(ucl_c),
            "lcl": float(lcl_c)
        }
    }

def generate_u_chart(data: np.ndarray, sample_sizes: np.ndarray) -> Dict:
    """
    U chart for defects per unit with varying sample sizes
    Data: number of defects per sample
    Sample sizes: number of units inspected per sample
    """
    # Calculate defect rates
    u = data / sample_sizes
    u_bar = np.sum(data) / np.sum(sample_sizes)

    # Control limits (vary by sample size)
    ucl = u_bar + 3 * np.sqrt(u_bar / sample_sizes)
    lcl = u_bar - 3 * np.sqrt(u_bar / sample_sizes)
    lcl = np.maximum(0, lcl)  # LCL can't be negative

    return {
        "chart_type": "u",
        "u_chart": {
            "values": u.tolist(),
            "center_line": float(u_bar),
            "ucl": ucl.tolist(),
            "lcl": lcl.tolist(),
            "sample_sizes": sample_sizes.tolist()
        }
    }

def generate_cusum_chart(data: np.ndarray, target: float, k: float, h: float, sigma: float) -> Dict:
    """
    CUSUM (Cumulative Sum) control chart
    Detects small shifts in process mean faster than Shewhart charts

    Parameters:
    - target: Target mean value
    - k: Slack value (allowance), typically 0.5 sigma
    - h: Decision interval (threshold), typically 4-5 sigma
    - sigma: Process standard deviation
    """
    n = len(data)

    # Standardize the data
    z = (data - target) / sigma

    # One-sided CUSUM - upper (detects upward shifts)
    cusum_upper = np.zeros(n)
    # One-sided CUSUM - lower (detects downward shifts)
    cusum_lower = np.zeros(n)

    for i in range(n):
        if i == 0:
            cusum_upper[i] = max(0, z[i] - k)
            cusum_lower[i] = min(0, z[i] + k)
        else:
            cusum_upper[i] = max(0, cusum_upper[i-1] + z[i] - k)
            cusum_lower[i] = min(0, cusum_lower[i-1] + z[i] + k)

    # Two-sided CUSUM (absolute values for lower)
    cusum_lower_abs = np.abs(cusum_lower)

    # Detect violations
    upper_violations = np.where(cusum_upper > h)[0].tolist()
    lower_violations = np.where(cusum_lower_abs > h)[0].tolist()

    # Decision threshold lines
    ucl = h
    lcl = -h

    return {
        "chart_type": "cusum",
        "cusum_chart": {
            "values": data.tolist(),
            "cusum_upper": cusum_upper.tolist(),
            "cusum_lower": cusum_lower.tolist(),
            "target": float(target),
            "k": float(k),
            "h": float(h),
            "sigma": float(sigma),
            "ucl": float(ucl),
            "lcl": float(lcl),
            "upper_violations": upper_violations,
            "lower_violations": lower_violations
        }
    }

def generate_ewma_chart(data: np.ndarray, lambda_: float, L: float, target: float, sigma: float) -> Dict:
    """
    EWMA (Exponentially Weighted Moving Average) control chart
    Weights recent observations more heavily, good for detecting small shifts

    Parameters:
    - lambda_: Smoothing parameter (0 < λ ≤ 1). Smaller values give more weight to history
    - L: Control limit width in sigma units (typically 2.5-3.0)
    - target: Target mean value
    - sigma: Process standard deviation
    """
    n = len(data)

    # Calculate EWMA statistic
    ewma = np.zeros(n)
    ewma[0] = lambda_ * data[0] + (1 - lambda_) * target
    for i in range(1, n):
        ewma[i] = lambda_ * data[i] + (1 - lambda_) * ewma[i-1]

    # Time-varying control limits (approach asymptotic limits)
    # UCL/LCL = target ± L × σ × sqrt(λ/(2-λ) × (1-(1-λ)^(2i)))
    i_vals = np.arange(1, n + 1)
    factor = np.sqrt((lambda_ / (2 - lambda_)) * (1 - (1 - lambda_) ** (2 * i_vals)))
    ucl = target + L * sigma * factor
    lcl = target - L * sigma * factor

    # Asymptotic limits (for reference)
    asymptotic_factor = np.sqrt(lambda_ / (2 - lambda_))
    ucl_asymptotic = target + L * sigma * asymptotic_factor
    lcl_asymptotic = target - L * sigma * asymptotic_factor

    # Detect violations
    violations = []
    for i in range(n):
        if ewma[i] > ucl[i] or ewma[i] < lcl[i]:
            violations.append(i)

    return {
        "chart_type": "ewma",
        "ewma_chart": {
            "values": data.tolist(),
            "ewma": ewma.tolist(),
            "target": float(target),
            "lambda": float(lambda_),
            "L": float(L),
            "sigma": float(sigma),
            "ucl": ucl.tolist(),
            "lcl": lcl.tolist(),
            "ucl_asymptotic": float(ucl_asymptotic),
            "lcl_asymptotic": float(lcl_asymptotic),
            "violations": violations
        }
    }

# ============================================================================
# Western Electric Rules (Control Chart Rule Violations)
# ============================================================================

def detect_rule_violations(values: List[float], cl: float, ucl: float, lcl: float) -> List[Dict]:
    """
    Detect Western Electric rules violations:
    Rule 1: One point beyond 3-sigma (outside control limits)
    Rule 2: Two of three consecutive points beyond 2-sigma on same side
    Rule 3: Four of five consecutive points beyond 1-sigma on same side
    Rule 4: Eight consecutive points on one side of center line
    Rule 5: Six points in a row steadily increasing or decreasing (trend)
    Rule 6: Fifteen points in a row within 1-sigma (both sides)
    """
    violations = []
    n = len(values)

    # Calculate sigma zones
    sigma = (ucl - cl) / 3
    zone_a_upper = cl + 2 * sigma  # 2-sigma
    zone_a_lower = cl - 2 * sigma
    zone_b_upper = cl + sigma      # 1-sigma
    zone_b_lower = cl - sigma

    for i in range(n):
        value = values[i]

        # Rule 1: Beyond 3-sigma (outside control limits)
        if value > ucl or value < lcl:
            violations.append({
                "point": i,
                "rule": 1,
                "description": "Point beyond control limits (3-sigma)",
                "severity": "critical"
            })

        # Rule 2: 2 of 3 beyond 2-sigma
        if i >= 2:
            last_3 = values[i-2:i+1]
            beyond_2sigma_upper = sum(1 for v in last_3 if v > zone_a_upper)
            beyond_2sigma_lower = sum(1 for v in last_3 if v < zone_a_lower)
            if beyond_2sigma_upper >= 2 or beyond_2sigma_lower >= 2:
                violations.append({
                    "point": i,
                    "rule": 2,
                    "description": "2 of 3 points beyond 2-sigma",
                    "severity": "warning"
                })

        # Rule 3: 4 of 5 beyond 1-sigma
        if i >= 4:
            last_5 = values[i-4:i+1]
            beyond_1sigma_upper = sum(1 for v in last_5 if v > zone_b_upper)
            beyond_1sigma_lower = sum(1 for v in last_5 if v < zone_b_lower)
            if beyond_1sigma_upper >= 4 or beyond_1sigma_lower >= 4:
                violations.append({
                    "point": i,
                    "rule": 3,
                    "description": "4 of 5 points beyond 1-sigma",
                    "severity": "warning"
                })

        # Rule 4: 8 consecutive on one side
        if i >= 7:
            last_8 = values[i-7:i+1]
            if all(v > cl for v in last_8) or all(v < cl for v in last_8):
                violations.append({
                    "point": i,
                    "rule": 4,
                    "description": "8 consecutive points on one side of center",
                    "severity": "warning"
                })

        # Rule 5: 6 points steadily increasing or decreasing
        if i >= 5:
            last_6 = values[i-5:i+1]
            increasing = all(last_6[j] < last_6[j+1] for j in range(5))
            decreasing = all(last_6[j] > last_6[j+1] for j in range(5))
            if increasing or decreasing:
                violations.append({
                    "point": i,
                    "rule": 5,
                    "description": "6 points in a row trending",
                    "severity": "info"
                })

        # Rule 6: 15 points within 1-sigma
        if i >= 14:
            last_15 = values[i-14:i+1]
            within_1sigma = all(zone_b_lower < v < zone_b_upper for v in last_15)
            if within_1sigma:
                violations.append({
                    "point": i,
                    "rule": 6,
                    "description": "15 points within 1-sigma (too good, suspicious)",
                    "severity": "info"
                })

    return violations

# ============================================================================
# Process Capability Analysis
# ============================================================================

def calculate_capability_confidence_intervals(n: int, cpk: float, confidence: float = 0.95) -> Dict:
    """
    Calculate confidence intervals for Cpk using the Chou et al. (1990) method
    Based on non-central chi-square distribution approximation
    """
    alpha = 1 - confidence
    z_alpha2 = stats.norm.ppf(1 - alpha / 2)

    # Approximate variance of Cpk
    # Var(Cpk) ≈ 1/(9n) + Cpk^2/(2(n-1))
    var_cpk = 1 / (9 * n) + (cpk ** 2) / (2 * (n - 1))
    se_cpk = np.sqrt(var_cpk)

    lower = cpk - z_alpha2 * se_cpk
    upper = cpk + z_alpha2 * se_cpk

    return {
        "lower": float(max(0, lower)),
        "upper": float(upper),
        "se": float(se_cpk)
    }

def calculate_cp_confidence_interval(n: int, cp: float, confidence: float = 0.95) -> Dict:
    """
    Calculate confidence interval for Cp using chi-square distribution
    """
    alpha = 1 - confidence

    # Cp follows sqrt(chi-square(n-1)/(n-1)) distribution
    chi2_lower = stats.chi2.ppf(alpha / 2, n - 1)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, n - 1)

    lower = cp * np.sqrt(chi2_lower / (n - 1))
    upper = cp * np.sqrt(chi2_upper / (n - 1))

    return {
        "lower": float(lower),
        "upper": float(upper)
    }

def calculate_capability_indices(data: np.ndarray, lsl: Optional[float], usl: Optional[float],
                                 target: Optional[float], confidence: float = 0.95) -> Dict:
    """
    Calculate process capability indices:
    - Cp: Potential capability (process spread vs. spec width)
    - Cpk: Actual capability (accounts for centering)
    - Pp: Long-term potential capability
    - Ppk: Long-term actual capability
    - Cpm: Taguchi capability index (accounts for target)
    """
    mean = np.mean(data)
    std_within = np.std(data, ddof=1)  # Within-subgroup variation (short-term)
    std_overall = np.std(data, ddof=0)  # Overall variation (long-term)

    results = {
        "process_mean": float(mean),
        "process_std_within": float(std_within),
        "process_std_overall": float(std_overall),
        "sample_size": len(data)
    }

    # Cp and Cpk (short-term capability)
    if lsl is not None and usl is not None:
        # Two-sided specification
        cp = (usl - lsl) / (6 * std_within)
        cpu = (usl - mean) / (3 * std_within)
        cpl = (mean - lsl) / (3 * std_within)
        cpk = min(cpu, cpl)

        results["cp"] = float(cp)
        results["cpk"] = float(cpk)
        results["cpu"] = float(cpu)
        results["cpl"] = float(cpl)

    elif usl is not None:
        # One-sided specification (upper)
        cpk = (usl - mean) / (3 * std_within)
        results["cpk"] = float(cpk)
        results["cp"] = None

    elif lsl is not None:
        # One-sided specification (lower)
        cpk = (mean - lsl) / (3 * std_within)
        results["cpk"] = float(cpk)
        results["cp"] = None

    else:
        results["cp"] = None
        results["cpk"] = None

    # Pp and Ppk (long-term capability)
    if lsl is not None and usl is not None:
        pp = (usl - lsl) / (6 * std_overall)
        ppu = (usl - mean) / (3 * std_overall)
        ppl = (mean - lsl) / (3 * std_overall)
        ppk = min(ppu, ppl)

        results["pp"] = float(pp)
        results["ppk"] = float(ppk)
        results["ppu"] = float(ppu)
        results["ppl"] = float(ppl)

    elif usl is not None:
        ppk = (usl - mean) / (3 * std_overall)
        results["ppk"] = float(ppk)
        results["pp"] = None

    elif lsl is not None:
        ppk = (mean - lsl) / (3 * std_overall)
        results["ppk"] = float(ppk)
        results["pp"] = None

    else:
        results["pp"] = None
        results["ppk"] = None

    # Cpm (Taguchi index - accounts for deviation from target)
    if target is not None and lsl is not None and usl is not None:
        tau = np.sqrt(std_within**2 + (mean - target)**2)
        cpm = (usl - lsl) / (6 * tau)
        results["cpm"] = float(cpm)
        results["target"] = target
    else:
        results["cpm"] = None

    # Interpretation
    if results.get("cpk") is not None:
        results["interpretation"] = interpret_capability(results["cpk"])

    # Estimated defect rates (PPM - parts per million)
    if results.get("cpk") is not None:
        z_min = results["cpk"] * 3  # Minimum z-score
        ppm = (1 - stats.norm.cdf(z_min)) * 1e6 * 2  # Two-tail
        results["estimated_ppm"] = float(ppm)

    # Confidence intervals
    n = len(data)
    results["confidence_level"] = confidence

    if results.get("cp") is not None:
        results["cp_ci"] = calculate_cp_confidence_interval(n, results["cp"], confidence)

    if results.get("cpk") is not None:
        results["cpk_ci"] = calculate_capability_confidence_intervals(n, results["cpk"], confidence)

    if results.get("pp") is not None:
        results["pp_ci"] = calculate_cp_confidence_interval(n, results["pp"], confidence)

    if results.get("ppk") is not None:
        results["ppk_ci"] = calculate_capability_confidence_intervals(n, results["ppk"], confidence)

    # Histogram data for visualization
    hist_counts, hist_edges = np.histogram(data, bins='auto')
    results["histogram"] = {
        "counts": hist_counts.tolist(),
        "edges": hist_edges.tolist(),
        "bin_centers": ((hist_edges[:-1] + hist_edges[1:]) / 2).tolist()
    }

    # Normal curve overlay data
    x_range = np.linspace(mean - 4 * std_within, mean + 4 * std_within, 100)
    normal_pdf = stats.norm.pdf(x_range, mean, std_within)
    # Scale to match histogram
    bin_width = hist_edges[1] - hist_edges[0]
    normal_scaled = normal_pdf * n * bin_width
    results["normal_curve"] = {
        "x": x_range.tolist(),
        "y": normal_scaled.tolist()
    }

    return results

def interpret_capability(cpk: float) -> str:
    """Interpret Cpk value"""
    if cpk >= 2.0:
        return "Excellent (6-sigma capability)"
    elif cpk >= 1.67:
        return "Very good (5-sigma capability)"
    elif cpk >= 1.33:
        return "Adequate (4-sigma capability)"
    elif cpk >= 1.0:
        return "Marginal (3-sigma capability)"
    else:
        return "Inadequate (<3-sigma, improvement needed)"

# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/control-chart")
async def generate_control_chart(request: ControlChartRequest):
    """
    Generate control chart with specified type
    Returns center line, control limits, and detected rule violations
    """
    try:
        data = np.array(request.data)

        if len(data) < 3:
            raise ValueError("Need at least 3 data points")

        # Generate appropriate chart
        if request.chart_type == "i-mr":
            chart_data = generate_i_mr_chart(data)

        elif request.chart_type == "xbar-r":
            if not request.subgroup_size or request.subgroup_size < 2:
                raise ValueError("Subgroup size must be >= 2 for Xbar-R chart")
            chart_data = generate_xbar_r_chart(data, request.subgroup_size)

        elif request.chart_type == "xbar-s":
            if not request.subgroup_size or request.subgroup_size < 2:
                raise ValueError("Subgroup size must be >= 2 for Xbar-S chart")
            chart_data = generate_xbar_s_chart(data, request.subgroup_size)

        elif request.chart_type == "p":
            if not request.subgroup_size or request.subgroup_size < 1:
                raise ValueError("Subgroup size must be >= 1 for P chart")
            chart_data = generate_p_chart(data, request.subgroup_size)

        elif request.chart_type == "c":
            chart_data = generate_c_chart(data)

        else:
            raise ValueError(f"Unknown chart type: {request.chart_type}. Supported: i-mr, xbar-r, xbar-s, p, c")

        # Detect rule violations for primary chart
        if request.chart_type == "i-mr":
            chart_values = chart_data["i_chart"]["values"]
            cl = chart_data["i_chart"]["center_line"]
            ucl = chart_data["i_chart"]["ucl"]
            lcl = chart_data["i_chart"]["lcl"]
        elif request.chart_type in ["xbar-r", "xbar-s"]:
            chart_values = chart_data["xbar_chart"]["values"]
            cl = chart_data["xbar_chart"]["center_line"]
            ucl = chart_data["xbar_chart"]["ucl"]
            lcl = chart_data["xbar_chart"]["lcl"]
        elif request.chart_type == "p":
            chart_values = chart_data["p_chart"]["values"]
            cl = chart_data["p_chart"]["center_line"]
            ucl = chart_data["p_chart"]["ucl"]
            lcl = chart_data["p_chart"]["lcl"]
        else:  # c chart
            chart_values = chart_data["c_chart"]["values"]
            cl = chart_data["c_chart"]["center_line"]
            ucl = chart_data["c_chart"]["ucl"]
            lcl = chart_data["c_chart"]["lcl"]

        violations = detect_rule_violations(chart_values, cl, ucl, lcl)
        chart_data["violations"] = violations

        # Add specification limits if provided
        if request.spec_limits:
            chart_data["spec_limits"] = request.spec_limits

        return chart_data

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/capability")
async def capability_analysis(request: CapabilityRequest):
    """
    Calculate process capability indices
    Requires specification limits (LSL and/or USL)
    """
    try:
        data = np.array(request.data)

        if len(data) < 3:
            raise ValueError("Need at least 3 data points")

        lsl = request.spec_limits.get("lsl")
        usl = request.spec_limits.get("usl")
        target = request.target

        if lsl is None and usl is None:
            raise ValueError("Must provide at least one specification limit (lsl or usl)")

        results = calculate_capability_indices(data, lsl, usl, target, request.confidence_level)
        results["spec_limits"] = request.spec_limits

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/cusum")
async def cusum_chart(request: CUSUMRequest):
    """
    Generate CUSUM (Cumulative Sum) control chart.
    More sensitive to small, persistent shifts in process mean than Shewhart charts.
    """
    try:
        data = np.array(request.data)

        if len(data) < 5:
            raise ValueError("Need at least 5 data points for CUSUM chart")

        # Use sample mean as target if not provided
        target = request.target if request.target is not None else float(np.mean(data))

        # Estimate sigma from moving range if not provided
        if request.sigma is not None:
            sigma = request.sigma
        else:
            mr = np.abs(np.diff(data))
            sigma = float(np.mean(mr) / 1.128)

        chart_data = generate_cusum_chart(data, target, request.k, request.h, sigma)

        return chart_data

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/ewma")
async def ewma_chart(request: EWMARequest):
    """
    Generate EWMA (Exponentially Weighted Moving Average) control chart.
    Weights recent observations more heavily, effective for detecting small shifts.
    """
    try:
        data = np.array(request.data)

        if len(data) < 5:
            raise ValueError("Need at least 5 data points for EWMA chart")

        if not (0 < request.lambda_ <= 1):
            raise ValueError("Lambda must be between 0 and 1")

        # Use sample mean as target if not provided
        target = request.target if request.target is not None else float(np.mean(data))

        # Estimate sigma from moving range if not provided
        if request.sigma is not None:
            sigma = request.sigma
        else:
            mr = np.abs(np.diff(data))
            sigma = float(np.mean(mr) / 1.128)

        chart_data = generate_ewma_chart(data, request.lambda_, request.L, target, sigma)

        return chart_data

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/info")
async def get_info():
    """
    Get information about control charts and capability indices
    """
    return {
        "control_charts": {
            "i-mr": {
                "name": "Individuals and Moving Range",
                "description": "For individual measurements (subgroup size = 1)",
                "use_case": "When only one measurement per time period is available",
                "data_type": "Continuous"
            },
            "xbar-r": {
                "name": "X-bar and R",
                "description": "For subgrouped data, uses range to estimate variation",
                "use_case": "Subgroup sizes 2-10, most common control chart",
                "data_type": "Continuous"
            },
            "xbar-s": {
                "name": "X-bar and S",
                "description": "For subgrouped data, uses standard deviation",
                "use_case": "Preferred when subgroup size > 10",
                "data_type": "Continuous"
            },
            "p": {
                "name": "P chart (Proportion)",
                "description": "For proportion of defective items",
                "use_case": "Count of defective items with varying sample sizes",
                "data_type": "Attribute"
            },
            "c": {
                "name": "C chart (Count)",
                "description": "For number of defects per unit",
                "use_case": "Count of defects with constant sample size",
                "data_type": "Attribute"
            },
            "u": {
                "name": "U chart (Defects per Unit)",
                "description": "For defects per unit with varying sample sizes",
                "use_case": "Number of defects when sample size varies",
                "data_type": "Attribute"
            },
            "cusum": {
                "name": "CUSUM (Cumulative Sum)",
                "description": "Cumulative sum of deviations from target",
                "use_case": "Detecting small, persistent shifts in process mean",
                "data_type": "Continuous"
            },
            "ewma": {
                "name": "EWMA (Exponentially Weighted Moving Average)",
                "description": "Weighted average giving more weight to recent data",
                "use_case": "Detecting small shifts, smooth response to changes",
                "data_type": "Continuous"
            }
        },
        "western_electric_rules": [
            "Rule 1: One point beyond 3-sigma (outside control limits)",
            "Rule 2: Two of three consecutive points beyond 2-sigma",
            "Rule 3: Four of five consecutive points beyond 1-sigma",
            "Rule 4: Eight consecutive points on one side of center",
            "Rule 5: Six points in a row trending",
            "Rule 6: Fifteen points within 1-sigma (too good)"
        ],
        "capability_indices": {
            "Cp": "Potential capability (spec width / process width)",
            "Cpk": "Actual capability (accounts for centering)",
            "Pp": "Long-term potential capability",
            "Ppk": "Long-term actual capability",
            "Cpm": "Taguchi index (penalizes deviation from target)"
        },
        "interpretation": {
            "Cpk >= 2.0": "Excellent (6-sigma)",
            "Cpk >= 1.67": "Very good (5-sigma)",
            "Cpk >= 1.33": "Adequate (4-sigma)",
            "Cpk >= 1.0": "Marginal (3-sigma)",
            "Cpk < 1.0": "Inadequate"
        }
    }
