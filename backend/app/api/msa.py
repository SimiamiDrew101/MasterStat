"""
Measurement Systems Analysis (MSA) Module
Gauge R&R studies and Attribute Agreement Analysis
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
from scipy import stats
from itertools import combinations

router = APIRouter(prefix="/api/msa", tags=["Measurement Systems Analysis"])

# ============================================================================
# Pydantic Models
# ============================================================================

class GaugeRRCrossedRequest(BaseModel):
    """
    Crossed Gauge R&R study data
    Each operator measures each part multiple times
    """
    measurements: List[List[List[float]]] = Field(
        ...,
        description="3D array: [operators][parts][replicates]"
    )
    part_names: Optional[List[str]] = Field(None, description="Part identifiers")
    operator_names: Optional[List[str]] = Field(None, description="Operator identifiers")
    tolerance: Optional[float] = Field(None, description="Part tolerance for %Tolerance calculation")
    confidence_level: float = Field(0.95, description="Confidence level for intervals")

class GaugeRRNestedRequest(BaseModel):
    """
    Nested Gauge R&R study data
    Each part is measured by only one operator (destructive testing)
    """
    measurements: List[List[List[float]]] = Field(
        ...,
        description="3D array: [operators][parts_per_operator][replicates]"
    )
    operator_names: Optional[List[str]] = Field(None, description="Operator identifiers")
    tolerance: Optional[float] = Field(None, description="Part tolerance for %Tolerance calculation")

class AttributeAgreementRequest(BaseModel):
    """
    Attribute Agreement Analysis data
    Each appraiser rates each sample multiple times
    """
    ratings: List[List[List[str]]] = Field(
        ...,
        description="3D array: [appraisers][samples][trials]"
    )
    reference_values: Optional[List[str]] = Field(None, description="Known correct values for each sample")
    appraiser_names: Optional[List[str]] = Field(None, description="Appraiser identifiers")

# ============================================================================
# Gauge R&R Functions
# ============================================================================

def calculate_gauge_rr_crossed_anova(data: np.ndarray, tolerance: Optional[float] = None) -> Dict:
    """
    Crossed Gauge R&R using ANOVA method

    Data shape: (n_operators, n_parts, n_replicates)

    Variance components:
    - Part-to-Part variation
    - Repeatability (within operator, same part)
    - Reproducibility (between operators)
    - Operator × Part interaction
    """
    n_operators, n_parts, n_replicates = data.shape
    n_total = n_operators * n_parts * n_replicates

    # Grand mean
    grand_mean = np.mean(data)

    # Part means
    part_means = np.mean(data, axis=(0, 2))

    # Operator means
    operator_means = np.mean(data, axis=(1, 2))

    # Operator × Part cell means
    cell_means = np.mean(data, axis=2)

    # Calculate Sum of Squares
    # SS Total
    ss_total = np.sum((data - grand_mean) ** 2)

    # SS Parts
    ss_parts = n_operators * n_replicates * np.sum((part_means - grand_mean) ** 2)

    # SS Operators
    ss_operators = n_parts * n_replicates * np.sum((operator_means - grand_mean) ** 2)

    # SS Interaction (Operator × Part)
    ss_interaction = n_replicates * np.sum((cell_means - part_means - operator_means[:, np.newaxis] + grand_mean) ** 2)

    # SS Repeatability (Error/Within)
    ss_repeatability = np.sum((data - cell_means[:, :, np.newaxis]) ** 2)

    # Degrees of freedom
    df_parts = n_parts - 1
    df_operators = n_operators - 1
    df_interaction = df_parts * df_operators
    df_repeatability = n_operators * n_parts * (n_replicates - 1)
    df_total = n_total - 1

    # Mean Squares
    ms_parts = ss_parts / df_parts if df_parts > 0 else 0
    ms_operators = ss_operators / df_operators if df_operators > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_repeatability = ss_repeatability / df_repeatability if df_repeatability > 0 else 0

    # Variance components
    # σ²_repeatability = MS_repeatability
    var_repeatability = ms_repeatability

    # σ²_interaction = (MS_interaction - MS_repeatability) / n_replicates
    var_interaction = max(0, (ms_interaction - ms_repeatability) / n_replicates)

    # σ²_operator = (MS_operator - MS_interaction) / (n_parts × n_replicates)
    var_operator = max(0, (ms_operators - ms_interaction) / (n_parts * n_replicates))

    # σ²_part = (MS_part - MS_interaction) / (n_operators × n_replicates)
    var_part = max(0, (ms_parts - ms_interaction) / (n_operators * n_replicates))

    # Reproducibility = σ²_operator + σ²_interaction
    var_reproducibility = var_operator + var_interaction

    # Gauge R&R = σ²_repeatability + σ²_reproducibility
    var_gauge_rr = var_repeatability + var_reproducibility

    # Total variation = σ²_gauge_rr + σ²_part
    var_total = var_gauge_rr + var_part

    # Standard deviations
    std_repeatability = np.sqrt(var_repeatability)
    std_reproducibility = np.sqrt(var_reproducibility)
    std_operator = np.sqrt(var_operator)
    std_interaction = np.sqrt(var_interaction)
    std_gauge_rr = np.sqrt(var_gauge_rr)
    std_part = np.sqrt(var_part)
    std_total = np.sqrt(var_total)

    # Study variation (5.15 × σ for 99% of normal distribution)
    # Some use 6 × σ for 99.73%
    k = 5.15
    sv_repeatability = k * std_repeatability
    sv_reproducibility = k * std_reproducibility
    sv_gauge_rr = k * std_gauge_rr
    sv_part = k * std_part
    sv_total = k * std_total

    # Percent of Study Variation
    pct_sv_repeatability = 100 * std_repeatability / std_total if std_total > 0 else 0
    pct_sv_reproducibility = 100 * std_reproducibility / std_total if std_total > 0 else 0
    pct_sv_gauge_rr = 100 * std_gauge_rr / std_total if std_total > 0 else 0
    pct_sv_part = 100 * std_part / std_total if std_total > 0 else 0

    # Percent of Tolerance (if tolerance provided)
    pct_tol_repeatability = None
    pct_tol_reproducibility = None
    pct_tol_gauge_rr = None
    if tolerance is not None and tolerance > 0:
        pct_tol_repeatability = 100 * sv_repeatability / tolerance
        pct_tol_reproducibility = 100 * sv_reproducibility / tolerance
        pct_tol_gauge_rr = 100 * sv_gauge_rr / tolerance

    # Number of distinct categories (ndc)
    # ndc = 1.41 × (σ_part / σ_gauge_rr)
    ndc = 1.41 * (std_part / std_gauge_rr) if std_gauge_rr > 0 else 0

    # F-tests for significance
    f_parts = ms_parts / ms_interaction if ms_interaction > 0 else 0
    f_operators = ms_operators / ms_interaction if ms_interaction > 0 else 0
    f_interaction = ms_interaction / ms_repeatability if ms_repeatability > 0 else 0

    p_parts = 1 - stats.f.cdf(f_parts, df_parts, df_interaction) if df_interaction > 0 else 1
    p_operators = 1 - stats.f.cdf(f_operators, df_operators, df_interaction) if df_interaction > 0 else 1
    p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_repeatability) if df_repeatability > 0 else 1

    # Interpretation
    if pct_sv_gauge_rr < 10:
        interpretation = "Excellent measurement system (<%10 GRR)"
    elif pct_sv_gauge_rr < 30:
        interpretation = "Acceptable measurement system (10-30% GRR)"
    else:
        interpretation = "Unacceptable measurement system (>30% GRR) - needs improvement"

    ndc_interpretation = "Adequate" if ndc >= 5 else "Inadequate (needs ndc ≥ 5)"

    return {
        "anova_table": {
            "sources": ["Part", "Operator", "Part × Operator", "Repeatability", "Total"],
            "df": [df_parts, df_operators, df_interaction, df_repeatability, df_total],
            "ss": [float(ss_parts), float(ss_operators), float(ss_interaction), float(ss_repeatability), float(ss_total)],
            "ms": [float(ms_parts), float(ms_operators), float(ms_interaction), float(ms_repeatability), None],
            "f_value": [float(f_parts), float(f_operators), float(f_interaction), None, None],
            "p_value": [float(p_parts), float(p_operators), float(p_interaction), None, None]
        },
        "variance_components": {
            "total_gauge_rr": float(var_gauge_rr),
            "repeatability": float(var_repeatability),
            "reproducibility": float(var_reproducibility),
            "operator": float(var_operator),
            "operator_part_interaction": float(var_interaction),
            "part_to_part": float(var_part),
            "total_variation": float(var_total)
        },
        "std_dev": {
            "total_gauge_rr": float(std_gauge_rr),
            "repeatability": float(std_repeatability),
            "reproducibility": float(std_reproducibility),
            "part_to_part": float(std_part),
            "total_variation": float(std_total)
        },
        "study_variation": {
            "total_gauge_rr": float(sv_gauge_rr),
            "repeatability": float(sv_repeatability),
            "reproducibility": float(sv_reproducibility),
            "part_to_part": float(sv_part),
            "total_variation": float(sv_total)
        },
        "percent_study_variation": {
            "total_gauge_rr": float(pct_sv_gauge_rr),
            "repeatability": float(pct_sv_repeatability),
            "reproducibility": float(pct_sv_reproducibility),
            "part_to_part": float(pct_sv_part)
        },
        "percent_tolerance": {
            "total_gauge_rr": float(pct_tol_gauge_rr) if pct_tol_gauge_rr is not None else None,
            "repeatability": float(pct_tol_repeatability) if pct_tol_repeatability is not None else None,
            "reproducibility": float(pct_tol_reproducibility) if pct_tol_reproducibility is not None else None
        },
        "number_distinct_categories": float(ndc),
        "ndc_interpretation": ndc_interpretation,
        "interpretation": interpretation,
        "study_info": {
            "n_operators": n_operators,
            "n_parts": n_parts,
            "n_replicates": n_replicates,
            "n_total": n_total,
            "grand_mean": float(grand_mean)
        }
    }

def calculate_gauge_rr_nested(data: np.ndarray, tolerance: Optional[float] = None) -> Dict:
    """
    Nested Gauge R&R for destructive testing
    Each part is measured by only one operator

    Data shape: (n_operators, n_parts_per_operator, n_replicates)
    """
    n_operators, n_parts_per_op, n_replicates = data.shape
    n_total = n_operators * n_parts_per_op * n_replicates

    # Grand mean
    grand_mean = np.mean(data)

    # Operator means
    operator_means = np.mean(data, axis=(1, 2))

    # Part(Operator) means - parts nested within operators
    part_means = np.mean(data, axis=2)  # Shape: (n_operators, n_parts_per_op)

    # Calculate Sum of Squares
    # SS Operators
    ss_operators = n_parts_per_op * n_replicates * np.sum((operator_means - grand_mean) ** 2)

    # SS Part(Operator) - parts nested within operators
    ss_parts_nested = n_replicates * np.sum((part_means - operator_means[:, np.newaxis]) ** 2)

    # SS Repeatability (Error)
    ss_repeatability = np.sum((data - part_means[:, :, np.newaxis]) ** 2)

    # SS Total
    ss_total = np.sum((data - grand_mean) ** 2)

    # Degrees of freedom
    df_operators = n_operators - 1
    df_parts_nested = n_operators * (n_parts_per_op - 1)
    df_repeatability = n_operators * n_parts_per_op * (n_replicates - 1)
    df_total = n_total - 1

    # Mean Squares
    ms_operators = ss_operators / df_operators if df_operators > 0 else 0
    ms_parts_nested = ss_parts_nested / df_parts_nested if df_parts_nested > 0 else 0
    ms_repeatability = ss_repeatability / df_repeatability if df_repeatability > 0 else 0

    # Variance components
    var_repeatability = ms_repeatability
    var_parts = max(0, (ms_parts_nested - ms_repeatability) / n_replicates)
    var_operator = max(0, (ms_operators - ms_parts_nested) / (n_parts_per_op * n_replicates))

    var_reproducibility = var_operator
    var_gauge_rr = var_repeatability + var_reproducibility
    var_total = var_gauge_rr + var_parts

    # Standard deviations
    std_repeatability = np.sqrt(var_repeatability)
    std_reproducibility = np.sqrt(var_reproducibility)
    std_gauge_rr = np.sqrt(var_gauge_rr)
    std_parts = np.sqrt(var_parts)
    std_total = np.sqrt(var_total)

    # Percent of Study Variation
    pct_sv_gauge_rr = 100 * std_gauge_rr / std_total if std_total > 0 else 0
    pct_sv_repeatability = 100 * std_repeatability / std_total if std_total > 0 else 0
    pct_sv_reproducibility = 100 * std_reproducibility / std_total if std_total > 0 else 0

    # Number of distinct categories
    ndc = 1.41 * (std_parts / std_gauge_rr) if std_gauge_rr > 0 else 0

    # F-tests
    f_operators = ms_operators / ms_parts_nested if ms_parts_nested > 0 else 0
    f_parts = ms_parts_nested / ms_repeatability if ms_repeatability > 0 else 0

    p_operators = 1 - stats.f.cdf(f_operators, df_operators, df_parts_nested) if df_parts_nested > 0 else 1
    p_parts = 1 - stats.f.cdf(f_parts, df_parts_nested, df_repeatability) if df_repeatability > 0 else 1

    # Interpretation
    if pct_sv_gauge_rr < 10:
        interpretation = "Excellent measurement system (<10% GRR)"
    elif pct_sv_gauge_rr < 30:
        interpretation = "Acceptable measurement system (10-30% GRR)"
    else:
        interpretation = "Unacceptable measurement system (>30% GRR) - needs improvement"

    return {
        "anova_table": {
            "sources": ["Operator", "Part(Operator)", "Repeatability", "Total"],
            "df": [df_operators, df_parts_nested, df_repeatability, df_total],
            "ss": [float(ss_operators), float(ss_parts_nested), float(ss_repeatability), float(ss_total)],
            "ms": [float(ms_operators), float(ms_parts_nested), float(ms_repeatability), None],
            "f_value": [float(f_operators), float(f_parts), None, None],
            "p_value": [float(p_operators), float(p_parts), None, None]
        },
        "variance_components": {
            "total_gauge_rr": float(var_gauge_rr),
            "repeatability": float(var_repeatability),
            "reproducibility": float(var_reproducibility),
            "part_to_part": float(var_parts),
            "total_variation": float(var_total)
        },
        "percent_study_variation": {
            "total_gauge_rr": float(pct_sv_gauge_rr),
            "repeatability": float(pct_sv_repeatability),
            "reproducibility": float(pct_sv_reproducibility)
        },
        "number_distinct_categories": float(ndc),
        "interpretation": interpretation,
        "study_info": {
            "n_operators": n_operators,
            "n_parts_per_operator": n_parts_per_op,
            "n_replicates": n_replicates,
            "n_total": n_total,
            "grand_mean": float(grand_mean)
        }
    }

# ============================================================================
# Attribute Agreement Analysis Functions
# ============================================================================

def calculate_cohens_kappa(ratings1: np.ndarray, ratings2: np.ndarray) -> Dict:
    """
    Calculate Cohen's Kappa for two raters
    """
    # Get unique categories
    categories = np.unique(np.concatenate([ratings1, ratings2]))
    n_categories = len(categories)
    n_samples = len(ratings1)

    # Create confusion matrix
    confusion = np.zeros((n_categories, n_categories))
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    for r1, r2 in zip(ratings1, ratings2):
        confusion[cat_to_idx[r1], cat_to_idx[r2]] += 1

    # Observed agreement
    p_o = np.trace(confusion) / n_samples

    # Expected agreement (by chance)
    row_marginals = confusion.sum(axis=1) / n_samples
    col_marginals = confusion.sum(axis=0) / n_samples
    p_e = np.sum(row_marginals * col_marginals)

    # Cohen's Kappa
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0

    # Standard error of Kappa
    # Approximate formula
    se_kappa = np.sqrt(p_o * (1 - p_o) / (n_samples * (1 - p_e) ** 2))

    # 95% CI
    z = 1.96
    kappa_lower = kappa - z * se_kappa
    kappa_upper = kappa + z * se_kappa

    return {
        "kappa": float(kappa),
        "se": float(se_kappa),
        "ci_lower": float(kappa_lower),
        "ci_upper": float(kappa_upper),
        "observed_agreement": float(p_o),
        "expected_agreement": float(p_e),
        "n_samples": n_samples
    }

def calculate_fleiss_kappa(ratings: np.ndarray) -> Dict:
    """
    Calculate Fleiss' Kappa for multiple raters
    ratings: 2D array (n_subjects × n_categories) with counts
    """
    n_subjects, n_categories = ratings.shape
    n_raters = ratings.sum(axis=1)[0]  # Assume same number of raters per subject

    # Proportion of raters in each category
    p_j = ratings.sum(axis=0) / (n_subjects * n_raters)

    # Agreement per subject
    P_i = (ratings.sum(axis=1) ** 2 - n_raters).sum() / (n_raters * (n_raters - 1))
    P_bar = P_i / n_subjects

    # Expected agreement
    P_e = np.sum(p_j ** 2)

    # Fleiss' Kappa
    kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1 else 0

    return {
        "kappa": float(kappa),
        "observed_agreement": float(P_bar),
        "expected_agreement": float(P_e),
        "n_subjects": n_subjects,
        "n_categories": n_categories
    }

def calculate_kendalls_w(rankings: np.ndarray) -> Dict:
    """
    Calculate Kendall's W (coefficient of concordance) for multiple raters
    rankings: 2D array (n_raters × n_items) with rank values
    """
    n_raters, n_items = rankings.shape

    # Sum of ranks for each item
    R_j = rankings.sum(axis=0)

    # Mean of rank sums
    R_bar = np.mean(R_j)

    # Sum of squared deviations
    S = np.sum((R_j - R_bar) ** 2)

    # Maximum possible S
    S_max = (n_raters ** 2 * (n_items ** 3 - n_items)) / 12

    # Kendall's W
    W = S / S_max if S_max > 0 else 0

    # Chi-square test for significance
    chi2 = n_raters * (n_items - 1) * W
    df = n_items - 1
    p_value = 1 - stats.chi2.cdf(chi2, df)

    return {
        "W": float(W),
        "chi_square": float(chi2),
        "df": df,
        "p_value": float(p_value),
        "n_raters": n_raters,
        "n_items": n_items,
        "interpretation": interpret_kendalls_w(W)
    }

def interpret_kendalls_w(W: float) -> str:
    """Interpret Kendall's W value"""
    if W >= 0.9:
        return "Very strong agreement"
    elif W >= 0.7:
        return "Strong agreement"
    elif W >= 0.5:
        return "Moderate agreement"
    elif W >= 0.3:
        return "Weak agreement"
    else:
        return "Very weak or no agreement"

def interpret_kappa(kappa: float) -> str:
    """Interpret Kappa value according to Landis & Koch (1977)"""
    if kappa >= 0.81:
        return "Almost perfect agreement"
    elif kappa >= 0.61:
        return "Substantial agreement"
    elif kappa >= 0.41:
        return "Moderate agreement"
    elif kappa >= 0.21:
        return "Fair agreement"
    elif kappa > 0:
        return "Slight agreement"
    else:
        return "Poor or no agreement"

def calculate_attribute_agreement(ratings: np.ndarray, reference: Optional[np.ndarray] = None) -> Dict:
    """
    Comprehensive Attribute Agreement Analysis

    ratings: 3D array (n_appraisers × n_samples × n_trials)
    reference: Optional 1D array of known correct values for each sample
    """
    n_appraisers, n_samples, n_trials = ratings.shape

    results = {
        "within_appraiser": [],
        "between_appraisers": {},
        "appraiser_vs_standard": [] if reference is not None else None,
        "overall": {}
    }

    # Within-appraiser agreement (repeatability)
    for a in range(n_appraisers):
        # Check if all trials for each sample agree
        agreements = []
        for s in range(n_samples):
            trials = ratings[a, s, :]
            agrees = len(set(trials)) == 1
            agreements.append(agrees)

        within_pct = 100 * sum(agreements) / n_samples

        # Kappa for within-appraiser (if n_trials >= 2)
        if n_trials >= 2:
            # Pairwise kappa between trials
            trial_kappas = []
            for t1, t2 in combinations(range(n_trials), 2):
                k = calculate_cohens_kappa(ratings[a, :, t1], ratings[a, :, t2])
                trial_kappas.append(k["kappa"])
            avg_kappa = np.mean(trial_kappas) if trial_kappas else None
        else:
            avg_kappa = None

        results["within_appraiser"].append({
            "appraiser": a,
            "percent_agreement": float(within_pct),
            "n_matches": sum(agreements),
            "n_samples": n_samples,
            "avg_kappa": float(avg_kappa) if avg_kappa is not None else None
        })

    # Between-appraiser agreement (reproducibility)
    # Use first trial from each appraiser
    if n_appraisers >= 2:
        all_pairwise_kappas = []
        pairwise_agreements = []

        for a1, a2 in combinations(range(n_appraisers), 2):
            ratings1 = ratings[a1, :, 0]  # First trial
            ratings2 = ratings[a2, :, 0]

            kappa_result = calculate_cohens_kappa(ratings1, ratings2)
            all_pairwise_kappas.append(kappa_result["kappa"])

            # Simple agreement
            agree = sum(r1 == r2 for r1, r2 in zip(ratings1, ratings2))
            pairwise_agreements.append(100 * agree / n_samples)

        results["between_appraisers"] = {
            "avg_kappa": float(np.mean(all_pairwise_kappas)),
            "avg_percent_agreement": float(np.mean(pairwise_agreements)),
            "interpretation": interpret_kappa(np.mean(all_pairwise_kappas))
        }

    # All appraisers agree (first trial)
    all_agree_count = 0
    for s in range(n_samples):
        first_trials = [ratings[a, s, 0] for a in range(n_appraisers)]
        if len(set(first_trials)) == 1:
            all_agree_count += 1

    results["overall"]["all_appraisers_agree"] = {
        "percent": float(100 * all_agree_count / n_samples),
        "count": all_agree_count,
        "n_samples": n_samples
    }

    # Appraiser vs Standard (if reference provided)
    if reference is not None:
        for a in range(n_appraisers):
            # Use first trial
            appraiser_ratings = ratings[a, :, 0]

            # Agreement with standard
            agree_count = sum(r == ref for r, ref in zip(appraiser_ratings, reference))
            agree_pct = 100 * agree_count / n_samples

            # Kappa vs standard
            kappa_result = calculate_cohens_kappa(appraiser_ratings, reference)

            results["appraiser_vs_standard"].append({
                "appraiser": a,
                "percent_agreement": float(agree_pct),
                "n_matches": agree_count,
                "kappa": kappa_result["kappa"],
                "interpretation": interpret_kappa(kappa_result["kappa"])
            })

        # All appraisers match standard
        all_match_standard = 0
        for s in range(n_samples):
            appraiser_first_trials = [ratings[a, s, 0] for a in range(n_appraisers)]
            if all(r == reference[s] for r in appraiser_first_trials):
                all_match_standard += 1

        results["overall"]["all_match_standard"] = {
            "percent": float(100 * all_match_standard / n_samples),
            "count": all_match_standard
        }

    return results

# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/gauge-rr/crossed")
async def gauge_rr_crossed(request: GaugeRRCrossedRequest):
    """
    Crossed Gauge R&R Study (ANOVA Method)

    Use when each operator measures each part multiple times.
    Standard study design for continuous measurement systems.
    """
    try:
        data = np.array(request.measurements)

        if data.ndim != 3:
            raise ValueError("Measurements must be a 3D array [operators][parts][replicates]")

        n_operators, n_parts, n_replicates = data.shape

        if n_operators < 2:
            raise ValueError("Need at least 2 operators")
        if n_parts < 2:
            raise ValueError("Need at least 2 parts")
        if n_replicates < 2:
            raise ValueError("Need at least 2 replicates")

        results = calculate_gauge_rr_crossed_anova(data, request.tolerance)

        # Add names if provided
        if request.operator_names:
            results["operator_names"] = request.operator_names
        if request.part_names:
            results["part_names"] = request.part_names

        # Add raw statistics for plotting
        results["operator_means"] = np.mean(data, axis=(1, 2)).tolist()
        results["part_means"] = np.mean(data, axis=(0, 2)).tolist()

        # Range chart data (for each operator, range across replicates for each part)
        ranges_by_operator = []
        for op in range(n_operators):
            ranges = []
            for part in range(n_parts):
                ranges.append(float(np.ptp(data[op, part, :])))
            ranges_by_operator.append(ranges)
        results["range_data"] = ranges_by_operator

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/gauge-rr/nested")
async def gauge_rr_nested(request: GaugeRRNestedRequest):
    """
    Nested Gauge R&R Study

    Use for destructive testing where each part can only be measured once.
    Parts are nested within operators.
    """
    try:
        data = np.array(request.measurements)

        if data.ndim != 3:
            raise ValueError("Measurements must be a 3D array [operators][parts_per_operator][replicates]")

        n_operators, n_parts_per_op, n_replicates = data.shape

        if n_operators < 2:
            raise ValueError("Need at least 2 operators")
        if n_parts_per_op < 2:
            raise ValueError("Need at least 2 parts per operator")

        results = calculate_gauge_rr_nested(data, request.tolerance)

        if request.operator_names:
            results["operator_names"] = request.operator_names

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/attribute-agreement")
async def attribute_agreement(request: AttributeAgreementRequest):
    """
    Attribute Agreement Analysis

    Evaluates consistency of categorical/attribute data ratings.
    Calculates within-appraiser, between-appraiser, and appraiser-vs-standard agreement.
    """
    try:
        # Convert string ratings to numpy array
        ratings = np.array(request.ratings)

        if ratings.ndim != 3:
            raise ValueError("Ratings must be a 3D array [appraisers][samples][trials]")

        n_appraisers, n_samples, n_trials = ratings.shape

        if n_appraisers < 1:
            raise ValueError("Need at least 1 appraiser")
        if n_samples < 2:
            raise ValueError("Need at least 2 samples")
        if n_trials < 1:
            raise ValueError("Need at least 1 trial")

        reference = np.array(request.reference_values) if request.reference_values else None

        if reference is not None and len(reference) != n_samples:
            raise ValueError("Reference values length must match number of samples")

        results = calculate_attribute_agreement(ratings, reference)

        if request.appraiser_names:
            results["appraiser_names"] = request.appraiser_names

        # Get unique categories
        results["categories"] = list(np.unique(ratings))

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/info")
async def msa_info():
    """
    Get information about MSA methods
    """
    return {
        "gauge_rr": {
            "crossed": {
                "description": "Standard Gauge R&R study where each operator measures each part",
                "requirements": "≥2 operators, ≥5 parts (recommended ≥10), ≥2 replicates",
                "outputs": ["ANOVA table", "Variance components", "% Study Variation", "% Tolerance", "ndc"]
            },
            "nested": {
                "description": "Gauge R&R for destructive testing where parts are consumed",
                "requirements": "≥2 operators, ≥2 parts per operator, ≥2 replicates",
                "outputs": ["ANOVA table", "Variance components", "% Study Variation"]
            }
        },
        "attribute_agreement": {
            "description": "Analysis for categorical/attribute measurement systems",
            "metrics": {
                "within_appraiser": "Repeatability - does same appraiser give same rating?",
                "between_appraiser": "Reproducibility - do different appraisers agree?",
                "vs_standard": "Accuracy - do ratings match known reference?"
            },
            "kappa_interpretation": {
                "0.81-1.00": "Almost perfect agreement",
                "0.61-0.80": "Substantial agreement",
                "0.41-0.60": "Moderate agreement",
                "0.21-0.40": "Fair agreement",
                "0.00-0.20": "Slight agreement",
                "<0": "Poor agreement"
            }
        },
        "acceptance_criteria": {
            "gauge_rr_percent": {
                "<10%": "Excellent - acceptable for all applications",
                "10-30%": "Acceptable - may be suitable depending on application",
                ">30%": "Unacceptable - measurement system needs improvement"
            },
            "ndc": "Number of distinct categories should be ≥5"
        }
    }
