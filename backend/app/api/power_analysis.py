from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower, TTestPower, FTestAnovaPower
import math

router = APIRouter()

class TTestSampleSizeRequest(BaseModel):
    test_type: Literal["one-sample", "two-sample", "paired"] = Field(..., description="Type of t-test")
    power: float = Field(0.8, ge=0.5, le=0.99, description="Desired statistical power (1-β)")
    alpha: float = Field(0.05, ge=0.001, le=0.5, description="Significance level (α)")
    alternative: Literal["two-sided", "greater", "less"] = Field("two-sided", description="Alternative hypothesis")

    # Effect size specification (use one of these methods)
    effect_size: Optional[float] = Field(None, description="Cohen's d effect size")
    mean_diff: Optional[float] = Field(None, description="Expected mean difference")
    std_dev: Optional[float] = Field(None, description="Standard deviation")

    # Two-sample specific
    ratio: float = Field(1.0, ge=0.1, le=10, description="Ratio of n2/n1 for unequal sample sizes")

    # Paired test specific
    correlation: Optional[float] = Field(0.5, ge=-0.99, le=0.99, description="Correlation for paired test")

@router.post("/sample-size/t-test")
def calculate_t_test_sample_size(request: TTestSampleSizeRequest):
    """
    Calculate required sample size for t-tests (one-sample, two-sample, paired)
    """
    try:
        # Calculate effect size if not provided
        if request.effect_size is None:
            if request.mean_diff is None or request.std_dev is None:
                raise HTTPException(
                    status_code=400,
                    detail="Either effect_size or both mean_diff and std_dev must be provided"
                )
            effect_size = abs(request.mean_diff) / request.std_dev
        else:
            effect_size = abs(request.effect_size)

        if effect_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Effect size cannot be zero"
            )

        # Determine number of tails
        if request.alternative == "two-sided":
            alternative = "two-sided"
        else:
            alternative = "larger"  # statsmodels uses 'larger' for one-sided

        # Calculate sample size based on test type
        if request.test_type == "one-sample":
            # One-sample t-test
            power_analysis = TTestPower()
            sample_size = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=request.alpha,
                power=request.power,
                alternative=alternative
            )
            sample_size = math.ceil(sample_size)

            result = {
                "test_type": "One-Sample t-Test",
                "sample_size": sample_size,
                "per_group": None,
                "total_sample_size": sample_size,
                "parameters": {
                    "effect_size": round(effect_size, 4),
                    "power": request.power,
                    "alpha": request.alpha,
                    "alternative": request.alternative
                }
            }

        elif request.test_type == "two-sample":
            # Two-sample t-test (independent samples)
            power_analysis = TTestIndPower()

            # Calculate for equal sample sizes first
            sample_size_per_group = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=request.alpha,
                power=request.power,
                ratio=request.ratio,
                alternative=alternative
            )
            sample_size_per_group = math.ceil(sample_size_per_group)

            # Calculate for unequal samples if ratio != 1
            if request.ratio != 1.0:
                n1 = sample_size_per_group
                n2 = math.ceil(sample_size_per_group * request.ratio)
                total = n1 + n2
            else:
                n1 = n2 = sample_size_per_group
                total = n1 + n2

            result = {
                "test_type": "Two-Sample t-Test (Independent)",
                "sample_size": None,
                "per_group": {
                    "group_1": n1,
                    "group_2": n2
                },
                "total_sample_size": total,
                "parameters": {
                    "effect_size": round(effect_size, 4),
                    "power": request.power,
                    "alpha": request.alpha,
                    "alternative": request.alternative,
                    "ratio": request.ratio
                }
            }

        else:  # paired
            # Paired t-test
            # For paired test, effect size is adjusted by correlation
            # Effective effect size = d / sqrt(2(1-r))
            r = request.correlation or 0.5
            adjusted_effect_size = effect_size / math.sqrt(2 * (1 - r))

            power_analysis = TTestPower()
            sample_size = power_analysis.solve_power(
                effect_size=adjusted_effect_size,
                alpha=request.alpha,
                power=request.power,
                alternative=alternative
            )
            sample_size = math.ceil(sample_size)

            result = {
                "test_type": "Paired t-Test",
                "sample_size": sample_size,
                "per_group": None,
                "total_sample_size": sample_size,
                "parameters": {
                    "effect_size": round(effect_size, 4),
                    "adjusted_effect_size": round(adjusted_effect_size, 4),
                    "correlation": r,
                    "power": request.power,
                    "alpha": request.alpha,
                    "alternative": request.alternative
                }
            }

        # Add interpretation
        result["interpretation"] = generate_interpretation(result, effect_size)

        # Add effect size classification
        result["effect_size_classification"] = classify_effect_size(effect_size)

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Calculation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

def classify_effect_size(d: float) -> dict:
    """Classify effect size according to Cohen's conventions"""
    abs_d = abs(d)
    if abs_d < 0.2:
        classification = "negligible"
        description = "very small effect, may not be practically significant"
    elif abs_d < 0.5:
        classification = "small"
        description = "small effect size"
    elif abs_d < 0.8:
        classification = "medium"
        description = "medium effect size"
    else:
        classification = "large"
        description = "large effect size"

    return {
        "classification": classification,
        "description": description,
        "cohens_d": round(abs_d, 4)
    }

def generate_interpretation(result: dict, effect_size: float) -> str:
    """Generate human-readable interpretation"""
    test_type = result["test_type"]
    power = result["parameters"]["power"]
    alpha = result["parameters"]["alpha"]

    if result["sample_size"]:
        n = result["sample_size"]
        interpretation = (
            f"To detect an effect size of {effect_size:.3f} with {power*100:.0f}% power "
            f"at α = {alpha}, you need a sample size of {n} "
        )
        if test_type == "Paired t-Test":
            interpretation += "pairs."
        else:
            interpretation += "observations."
    else:
        n1 = result["per_group"]["group_1"]
        n2 = result["per_group"]["group_2"]
        interpretation = (
            f"To detect an effect size of {effect_size:.3f} with {power*100:.0f}% power "
            f"at α = {alpha}, you need {n1} participants in group 1 and {n2} in group 2 "
            f"(total N = {result['total_sample_size']})."
        )

    return interpretation


class ANOVASampleSizeRequest(BaseModel):
    anova_type: Literal["one-way", "two-way"] = Field(..., description="Type of ANOVA")
    power: float = Field(0.8, ge=0.5, le=0.99, description="Desired statistical power (1-β)")
    alpha: float = Field(0.05, ge=0.001, le=0.5, description="Significance level (α)")

    # Effect size specification (use one of these methods)
    effect_size: Optional[float] = Field(None, description="Cohen's f effect size")
    eta_squared: Optional[float] = Field(None, description="Eta-squared (η²) effect size")

    # One-way ANOVA specific
    num_groups: Optional[int] = Field(None, ge=2, le=20, description="Number of groups")

    # Two-way ANOVA specific
    num_levels_a: Optional[int] = Field(None, ge=2, le=10, description="Number of levels for Factor A")
    num_levels_b: Optional[int] = Field(None, ge=2, le=10, description="Number of levels for Factor B")
    effect_of_interest: Optional[Literal["main_a", "main_b", "interaction"]] = Field(
        "main_a", description="Which effect to power for in two-way ANOVA"
    )


@router.post("/sample-size/anova")
def calculate_anova_sample_size(request: ANOVASampleSizeRequest):
    """
    Calculate required sample size for ANOVA (one-way and two-way)
    """
    try:
        # Calculate effect size if not provided
        if request.effect_size is None:
            if request.eta_squared is None:
                raise HTTPException(
                    status_code=400,
                    detail="Either effect_size (Cohen's f) or eta_squared must be provided"
                )
            # Convert eta-squared to Cohen's f: f = sqrt(η² / (1 - η²))
            if request.eta_squared >= 1.0:
                raise HTTPException(
                    status_code=400,
                    detail="Eta-squared must be less than 1.0"
                )
            effect_size = math.sqrt(request.eta_squared / (1 - request.eta_squared))
        else:
            effect_size = abs(request.effect_size)

        if effect_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Effect size cannot be zero"
            )

        power_analysis = FTestAnovaPower()

        if request.anova_type == "one-way":
            # One-way ANOVA
            if request.num_groups is None:
                raise HTTPException(
                    status_code=400,
                    detail="num_groups is required for one-way ANOVA"
                )

            # Calculate sample size per group
            # k = number of groups, df = k - 1
            k = request.num_groups
            df_between = k - 1

            sample_size_per_group = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=request.alpha,
                power=request.power,
                k_groups=k
            )
            sample_size_per_group = math.ceil(sample_size_per_group)
            total_n = sample_size_per_group * k

            result = {
                "test_type": "One-Way ANOVA",
                "anova_type": "one-way",
                "sample_size_per_group": sample_size_per_group,
                "num_groups": k,
                "total_sample_size": total_n,
                "parameters": {
                    "effect_size": round(effect_size, 4),
                    "power": request.power,
                    "alpha": request.alpha,
                    "num_groups": k,
                    "df_between": df_between
                }
            }

            # Add interpretation
            result["interpretation"] = (
                f"To detect an effect size of f = {effect_size:.3f} with {request.power*100:.0f}% power "
                f"at α = {request.alpha}, you need {sample_size_per_group} participants per group "
                f"across {k} groups (total N = {total_n})."
            )

        else:  # two-way ANOVA
            if request.num_levels_a is None or request.num_levels_b is None:
                raise HTTPException(
                    status_code=400,
                    detail="num_levels_a and num_levels_b are required for two-way ANOVA"
                )

            a = request.num_levels_a
            b = request.num_levels_b
            total_groups = a * b

            # Determine degrees of freedom based on effect of interest
            if request.effect_of_interest == "main_a":
                df_effect = a - 1
                effect_name = f"Main Effect of Factor A ({a} levels)"
            elif request.effect_of_interest == "main_b":
                df_effect = b - 1
                effect_name = f"Main Effect of Factor B ({b} levels)"
            else:  # interaction
                df_effect = (a - 1) * (b - 1)
                effect_name = f"Interaction Effect (A×B)"

            # Calculate sample size per cell
            sample_size_per_cell = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=request.alpha,
                power=request.power,
                k_groups=total_groups
            )
            sample_size_per_cell = math.ceil(sample_size_per_cell)
            total_n = sample_size_per_cell * total_groups

            result = {
                "test_type": "Two-Way ANOVA",
                "anova_type": "two-way",
                "sample_size_per_cell": sample_size_per_cell,
                "num_cells": total_groups,
                "total_sample_size": total_n,
                "design": {
                    "factor_a_levels": a,
                    "factor_b_levels": b,
                    "total_cells": total_groups
                },
                "parameters": {
                    "effect_size": round(effect_size, 4),
                    "power": request.power,
                    "alpha": request.alpha,
                    "effect_of_interest": effect_name,
                    "df_effect": df_effect
                }
            }

            # Add interpretation
            result["interpretation"] = (
                f"To detect an effect size of f = {effect_size:.3f} for {effect_name} "
                f"with {request.power*100:.0f}% power at α = {request.alpha}, "
                f"you need {sample_size_per_cell} participants per cell in a {a}×{b} design "
                f"(total N = {total_n})."
            )

        # Add effect size classification
        result["effect_size_classification"] = classify_anova_effect_size(effect_size)

        # Calculate eta-squared
        f_squared = effect_size ** 2
        eta_squared = f_squared / (1 + f_squared)
        result["eta_squared"] = round(eta_squared, 4)

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Calculation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def classify_anova_effect_size(f: float) -> dict:
    """Classify ANOVA effect size (Cohen's f) according to conventions"""
    abs_f = abs(f)
    if abs_f < 0.1:
        classification = "negligible"
        description = "very small effect, may not be practically significant"
    elif abs_f < 0.25:
        classification = "small"
        description = "small effect size"
    elif abs_f < 0.4:
        classification = "medium"
        description = "medium effect size"
    else:
        classification = "large"
        description = "large effect size"

    return {
        "classification": classification,
        "description": description,
        "cohens_f": round(abs_f, 4)
    }


class PowerCurveRequest(BaseModel):
    test_family: Literal["t-test", "anova"] = Field(..., description="Test family")
    test_type: Optional[str] = Field(None, description="Specific test type")
    alpha: float = Field(0.05, ge=0.001, le=0.5, description="Significance level")

    # Fixed parameters for the curve
    effect_size: Optional[float] = Field(None, description="Effect size (fixed when varying sample size)")
    sample_size: Optional[int] = Field(None, description="Sample size (fixed when varying effect size)")

    # Test-specific parameters
    alternative: Optional[str] = Field("two-sided", description="Alternative hypothesis for t-test")
    num_groups: Optional[int] = Field(3, description="Number of groups for ANOVA")
    num_levels_a: Optional[int] = Field(2, description="Factor A levels for two-way ANOVA")
    num_levels_b: Optional[int] = Field(2, description="Factor B levels for two-way ANOVA")
    effect_of_interest: Optional[str] = Field("main_a", description="Effect of interest for two-way ANOVA")
    ratio: Optional[float] = Field(1.0, description="Sample size ratio for two-sample t-test")
    correlation: Optional[float] = Field(0.5, description="Correlation for paired t-test")

    curve_type: Literal["power_vs_n", "power_vs_effect"] = Field("power_vs_n", description="Type of curve to generate")


@router.post("/power-curve")
def generate_power_curve(request: PowerCurveRequest):
    """
    Generate power curve data for visualization
    """
    try:
        curves_data = []

        if request.curve_type == "power_vs_n":
            # Power vs. Sample Size curve
            if request.effect_size is None:
                raise HTTPException(
                    status_code=400,
                    detail="effect_size is required for power_vs_n curve"
                )

            # Generate range of sample sizes
            if request.test_family == "t-test":
                # For t-tests, use reasonable range based on test type
                min_n = 5
                max_n = 200
                sample_sizes = list(range(min_n, min(50, max_n), 2)) + list(range(50, max_n + 1, 5))
            else:
                # For ANOVA, per-group sample sizes
                min_n = 5
                max_n = 100
                sample_sizes = list(range(min_n, min(30, max_n), 2)) + list(range(30, max_n + 1, 3))

            powers = []
            for n in sample_sizes:
                power = calculate_power_for_n(
                    n, request.effect_size, request.alpha,
                    request.test_family, request.test_type,
                    request.alternative, request.num_groups,
                    request.num_levels_a, request.num_levels_b,
                    request.effect_of_interest, request.ratio, request.correlation
                )
                powers.append(round(power, 4))

            curves_data.append({
                "name": f"Effect Size = {request.effect_size:.2f}",
                "x_values": sample_sizes,
                "y_values": powers,
                "x_label": "Sample Size (n)" if request.test_family == "t-test" and request.test_type != "two-sample" else "Sample Size per Group (n)",
                "y_label": "Statistical Power"
            })

        else:  # power_vs_effect
            # Power vs. Effect Size curve
            if request.sample_size is None:
                raise HTTPException(
                    status_code=400,
                    detail="sample_size is required for power_vs_effect curve"
                )

            # Generate range of effect sizes
            if request.test_family == "t-test":
                # Cohen's d range
                effect_sizes = [i * 0.05 for i in range(2, 41)]  # 0.1 to 2.0
            else:
                # Cohen's f range
                effect_sizes = [i * 0.02 for i in range(1, 51)]  # 0.02 to 1.0

            powers = []
            for es in effect_sizes:
                power = calculate_power_for_n(
                    request.sample_size, es, request.alpha,
                    request.test_family, request.test_type,
                    request.alternative, request.num_groups,
                    request.num_levels_a, request.num_levels_b,
                    request.effect_of_interest, request.ratio, request.correlation
                )
                powers.append(round(power, 4))

            es_label = "Cohen's d" if request.test_family == "t-test" else "Cohen's f"
            curves_data.append({
                "name": f"n = {request.sample_size}",
                "x_values": [round(es, 3) for es in effect_sizes],
                "y_values": powers,
                "x_label": f"Effect Size ({es_label})",
                "y_label": "Statistical Power"
            })

        return {
            "curve_type": request.curve_type,
            "curves": curves_data,
            "parameters": {
                "alpha": request.alpha,
                "test_family": request.test_family,
                "test_type": request.test_type
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating power curve: {str(e)}")


def calculate_power_for_n(n, effect_size, alpha, test_family, test_type,
                          alternative, num_groups, num_levels_a, num_levels_b,
                          effect_of_interest, ratio, correlation):
    """Calculate power for given sample size and effect size"""
    try:
        if test_family == "t-test":
            alt = "two-sided" if alternative == "two-sided" else "larger"

            if test_type == "one-sample":
                power_analysis = TTestPower()
                power = power_analysis.solve_power(
                    effect_size=effect_size,
                    nobs=n,
                    alpha=alpha,
                    alternative=alt
                )
            elif test_type == "two-sample":
                power_analysis = TTestIndPower()
                power = power_analysis.solve_power(
                    effect_size=effect_size,
                    nobs1=n,
                    alpha=alpha,
                    ratio=ratio,
                    alternative=alt
                )
            else:  # paired
                # Adjust effect size for correlation
                adjusted_es = effect_size / math.sqrt(2 * (1 - correlation))
                power_analysis = TTestPower()
                power = power_analysis.solve_power(
                    effect_size=adjusted_es,
                    nobs=n,
                    alpha=alpha,
                    alternative=alt
                )
        else:  # ANOVA
            power_analysis = FTestAnovaPower()

            if test_type == "one-way":
                k = num_groups
                power = power_analysis.solve_power(
                    effect_size=effect_size,
                    nobs=n,
                    alpha=alpha,
                    k_groups=k
                )
            else:  # two-way
                total_groups = num_levels_a * num_levels_b
                power = power_analysis.solve_power(
                    effect_size=effect_size,
                    nobs=n,
                    alpha=alpha,
                    k_groups=total_groups
                )

        return min(power, 0.9999)  # Cap at 0.9999 for display

    except Exception as e:
        return 0.0  # Return 0 if calculation fails


# ============================================================================
# EFFECT SIZE ESTIMATION ENDPOINTS
# ============================================================================

class MinimumEffectSizeRequest(BaseModel):
    """Request for calculating minimum detectable effect size"""
    test_family: Literal["t-test", "anova"] = Field(..., description="Type of test")
    test_type: str = Field(..., description="Specific test type")
    sample_size: int = Field(..., ge=2, description="Available sample size")
    power: float = Field(0.8, ge=0.5, le=0.99, description="Desired power")
    alpha: float = Field(0.05, ge=0.001, le=0.5, description="Significance level")

    # t-test specific
    alternative: Optional[str] = Field("two-sided", description="Alternative hypothesis")
    ratio: Optional[float] = Field(1.0, description="Ratio n2/n1 for two-sample test")
    correlation: Optional[float] = Field(0.5, description="Correlation for paired test")

    # ANOVA specific
    num_groups: Optional[int] = Field(None, description="Number of groups")
    num_levels_a: Optional[int] = Field(None, description="Levels of factor A")
    num_levels_b: Optional[int] = Field(None, description="Levels of factor B")


@router.post("/minimum-effect-size")
def calculate_minimum_effect_size(request: MinimumEffectSizeRequest):
    """
    Calculate the minimum detectable effect size given sample size, power, and alpha.
    This is the reverse calculation of sample size determination.
    """
    try:
        alternative = "two-sided" if request.alternative == "two-sided" else "larger"

        if request.test_family == "t-test":
            if request.test_type == "one-sample":
                power_analysis = TTestPower()
                effect_size = power_analysis.solve_power(
                    nobs=request.sample_size,
                    alpha=request.alpha,
                    power=request.power,
                    alternative=alternative
                )

                return {
                    "test_type": "One-Sample t-Test",
                    "minimum_effect_size": round(effect_size, 4),
                    "effect_size_metric": "Cohen's d",
                    "interpretation": get_cohens_d_interpretation(effect_size),
                    "parameters": {
                        "sample_size": request.sample_size,
                        "power": request.power,
                        "alpha": request.alpha,
                        "alternative": request.alternative
                    }
                }

            elif request.test_type == "two-sample":
                power_analysis = TTestIndPower()
                effect_size = power_analysis.solve_power(
                    nobs1=request.sample_size,
                    alpha=request.alpha,
                    power=request.power,
                    ratio=request.ratio or 1.0,
                    alternative=alternative
                )

                n2 = math.ceil(request.sample_size * (request.ratio or 1.0))

                return {
                    "test_type": "Two-Sample t-Test",
                    "minimum_effect_size": round(effect_size, 4),
                    "effect_size_metric": "Cohen's d",
                    "interpretation": get_cohens_d_interpretation(effect_size),
                    "sample_sizes": {
                        "group_1": request.sample_size,
                        "group_2": n2,
                        "total": request.sample_size + n2
                    },
                    "parameters": {
                        "power": request.power,
                        "alpha": request.alpha,
                        "alternative": request.alternative,
                        "ratio": request.ratio or 1.0
                    }
                }

            else:  # paired
                # For paired tests, adjust for correlation
                power_analysis = TTestPower()
                adjusted_es = power_analysis.solve_power(
                    nobs=request.sample_size,
                    alpha=request.alpha,
                    power=request.power,
                    alternative=alternative
                )
                # Convert back to unadjusted effect size
                effect_size = adjusted_es * math.sqrt(2 * (1 - (request.correlation or 0.5)))

                return {
                    "test_type": "Paired t-Test",
                    "minimum_effect_size": round(effect_size, 4),
                    "effect_size_metric": "Cohen's d",
                    "interpretation": get_cohens_d_interpretation(effect_size),
                    "parameters": {
                        "sample_size": request.sample_size,
                        "power": request.power,
                        "alpha": request.alpha,
                        "alternative": request.alternative,
                        "correlation": request.correlation or 0.5
                    }
                }

        else:  # ANOVA
            power_analysis = FTestAnovaPower()

            if request.test_type == "one-way":
                k = request.num_groups or 3
                effect_size = power_analysis.solve_power(
                    nobs=request.sample_size,
                    alpha=request.alpha,
                    power=request.power,
                    k_groups=k
                )

                # Convert to eta-squared
                eta_squared = (effect_size ** 2) / (1 + effect_size ** 2)

                return {
                    "test_type": "One-Way ANOVA",
                    "minimum_effect_size_f": round(effect_size, 4),
                    "minimum_effect_size_eta_squared": round(eta_squared, 4),
                    "interpretation_f": get_cohens_f_interpretation(effect_size),
                    "interpretation_eta_squared": get_eta_squared_interpretation(eta_squared),
                    "parameters": {
                        "sample_size_per_group": request.sample_size,
                        "num_groups": k,
                        "total_sample_size": request.sample_size * k,
                        "power": request.power,
                        "alpha": request.alpha
                    }
                }

            else:  # two-way
                total_groups = (request.num_levels_a or 2) * (request.num_levels_b or 2)
                effect_size = power_analysis.solve_power(
                    nobs=request.sample_size,
                    alpha=request.alpha,
                    power=request.power,
                    k_groups=total_groups
                )

                eta_squared = (effect_size ** 2) / (1 + effect_size ** 2)

                return {
                    "test_type": "Two-Way ANOVA",
                    "minimum_effect_size_f": round(effect_size, 4),
                    "minimum_effect_size_eta_squared": round(eta_squared, 4),
                    "interpretation_f": get_cohens_f_interpretation(effect_size),
                    "interpretation_eta_squared": get_eta_squared_interpretation(eta_squared),
                    "parameters": {
                        "sample_size_per_cell": request.sample_size,
                        "num_levels_a": request.num_levels_a or 2,
                        "num_levels_b": request.num_levels_b or 2,
                        "total_cells": total_groups,
                        "total_sample_size": request.sample_size * total_groups,
                        "power": request.power,
                        "alpha": request.alpha
                    }
                }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculating minimum effect size: {str(e)}")


class EffectSizeConversionRequest(BaseModel):
    """Request for converting between effect size metrics"""
    from_metric: Literal["cohens_d", "cohens_f", "eta_squared", "r", "odds_ratio"] = Field(..., description="Source metric")
    to_metric: Literal["cohens_d", "cohens_f", "eta_squared", "r", "odds_ratio"] = Field(..., description="Target metric")
    value: float = Field(..., description="Value to convert")


@router.post("/convert-effect-size")
def convert_effect_size(request: EffectSizeConversionRequest):
    """
    Convert between different effect size metrics.
    Supports: Cohen's d, Cohen's f, η², correlation r, odds ratio
    """
    try:
        value = abs(request.value)
        conversions = {}

        # First convert input to a standard metric (Cohen's d)
        if request.from_metric == "cohens_d":
            d = value
        elif request.from_metric == "cohens_f":
            # f to d: d = 2f
            d = 2 * value
        elif request.from_metric == "eta_squared":
            # η² to d: d = 2 * sqrt(η² / (1 - η²))
            if value >= 1:
                raise HTTPException(status_code=400, detail="Eta-squared must be less than 1")
            d = 2 * math.sqrt(value / (1 - value))
        elif request.from_metric == "r":
            # r to d: d = 2r / sqrt(1 - r²)
            if abs(value) >= 1:
                raise HTTPException(status_code=400, detail="Correlation must be between -1 and 1")
            d = (2 * value) / math.sqrt(1 - value ** 2)
        elif request.from_metric == "odds_ratio":
            # OR to d: d = ln(OR) * sqrt(3) / π
            if value <= 0:
                raise HTTPException(status_code=400, detail="Odds ratio must be positive")
            d = math.log(value) * math.sqrt(3) / math.pi

        # Now convert from d to target metric
        if request.to_metric == "cohens_d":
            result_value = d
        elif request.to_metric == "cohens_f":
            # d to f: f = d / 2
            result_value = d / 2
        elif request.to_metric == "eta_squared":
            # d to η²: η² = d² / (d² + 4)
            result_value = (d ** 2) / (d ** 2 + 4)
        elif request.to_metric == "r":
            # d to r: r = d / sqrt(d² + 4)
            result_value = d / math.sqrt(d ** 2 + 4)
        elif request.to_metric == "odds_ratio":
            # d to OR: OR = exp(d * π / sqrt(3))
            result_value = math.exp(d * math.pi / math.sqrt(3))

        # Get all conversions for reference
        conversions = {
            "cohens_d": round(d, 4),
            "cohens_f": round(d / 2, 4),
            "eta_squared": round((d ** 2) / (d ** 2 + 4), 4),
            "r": round(d / math.sqrt(d ** 2 + 4), 4),
            "odds_ratio": round(math.exp(d * math.pi / math.sqrt(3)), 4)
        }

        return {
            "original": {
                "metric": request.from_metric,
                "value": request.value
            },
            "converted": {
                "metric": request.to_metric,
                "value": round(result_value, 4)
            },
            "all_conversions": conversions,
            "interpretations": {
                "cohens_d": get_cohens_d_interpretation(conversions["cohens_d"]),
                "cohens_f": get_cohens_f_interpretation(conversions["cohens_f"]),
                "eta_squared": get_eta_squared_interpretation(conversions["eta_squared"])
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error converting effect size: {str(e)}")


class PilotDataRequest(BaseModel):
    """Request for calculating effect size from pilot study data"""
    test_type: Literal["independent", "paired", "anova"] = Field(..., description="Type of comparison")

    # For independent samples t-test
    mean1: Optional[float] = Field(None, description="Mean of group 1")
    sd1: Optional[float] = Field(None, description="SD of group 1")
    n1: Optional[int] = Field(None, description="Sample size group 1")
    mean2: Optional[float] = Field(None, description="Mean of group 2")
    sd2: Optional[float] = Field(None, description="SD of group 2")
    n2: Optional[int] = Field(None, description="Sample size group 2")

    # For paired samples
    mean_diff: Optional[float] = Field(None, description="Mean difference")
    sd_diff: Optional[float] = Field(None, description="SD of differences")

    # For ANOVA
    group_means: Optional[list[float]] = Field(None, description="List of group means")
    group_sds: Optional[list[float]] = Field(None, description="List of group SDs")
    group_ns: Optional[list[int]] = Field(None, description="List of group sample sizes")


@router.post("/effect-size-from-pilot")
def calculate_effect_size_from_pilot(request: PilotDataRequest):
    """
    Calculate effect size estimates from pilot study data.
    Provides Cohen's d for t-tests and Cohen's f / η² for ANOVA.
    """
    try:
        if request.test_type == "independent":
            if None in [request.mean1, request.sd1, request.n1, request.mean2, request.sd2, request.n2]:
                raise HTTPException(
                    status_code=400,
                    detail="For independent samples, provide mean1, sd1, n1, mean2, sd2, n2"
                )

            # Pooled standard deviation
            pooled_sd = math.sqrt(
                ((request.n1 - 1) * request.sd1 ** 2 + (request.n2 - 1) * request.sd2 ** 2) /
                (request.n1 + request.n2 - 2)
            )

            # Cohen's d
            cohens_d = abs(request.mean1 - request.mean2) / pooled_sd

            # Standard error of d
            se_d = math.sqrt(
                (request.n1 + request.n2) / (request.n1 * request.n2) +
                (cohens_d ** 2) / (2 * (request.n1 + request.n2))
            )

            # 95% CI
            ci_lower = cohens_d - 1.96 * se_d
            ci_upper = cohens_d + 1.96 * se_d

            return {
                "test_type": "Independent Samples t-Test",
                "cohens_d": round(cohens_d, 4),
                "interpretation": get_cohens_d_interpretation(cohens_d),
                "confidence_interval_95": {
                    "lower": round(max(0, ci_lower), 4),
                    "upper": round(ci_upper, 4)
                },
                "pilot_data": {
                    "group1": {"mean": request.mean1, "sd": request.sd1, "n": request.n1},
                    "group2": {"mean": request.mean2, "sd": request.sd2, "n": request.n2},
                    "pooled_sd": round(pooled_sd, 4),
                    "mean_difference": round(abs(request.mean1 - request.mean2), 4)
                },
                "note": "This is an estimate from pilot data. Use with caution for final sample size calculations."
            }

        elif request.test_type == "paired":
            if None in [request.mean_diff, request.sd_diff]:
                raise HTTPException(
                    status_code=400,
                    detail="For paired samples, provide mean_diff and sd_diff"
                )

            cohens_d = abs(request.mean_diff) / request.sd_diff

            return {
                "test_type": "Paired Samples t-Test",
                "cohens_d": round(cohens_d, 4),
                "interpretation": get_cohens_d_interpretation(cohens_d),
                "pilot_data": {
                    "mean_difference": request.mean_diff,
                    "sd_difference": request.sd_diff
                },
                "note": "This is an estimate from pilot data. Use with caution for final sample size calculations."
            }

        else:  # ANOVA
            if None in [request.group_means, request.group_sds, request.group_ns]:
                raise HTTPException(
                    status_code=400,
                    detail="For ANOVA, provide group_means, group_sds, and group_ns as lists"
                )

            k = len(request.group_means)
            if len(request.group_sds) != k or len(request.group_ns) != k:
                raise HTTPException(
                    status_code=400,
                    detail="All group lists must have the same length"
                )

            # Overall mean
            total_n = sum(request.group_ns)
            grand_mean = sum(m * n for m, n in zip(request.group_means, request.group_ns)) / total_n

            # Between-group variance (effect)
            ss_between = sum(n * (m - grand_mean) ** 2 for m, n, in zip(request.group_means, request.group_ns))
            ms_between = ss_between / (k - 1)

            # Within-group variance (pooled)
            ss_within = sum((n - 1) * sd ** 2 for n, sd in zip(request.group_ns, request.group_sds))
            ms_within = ss_within / (total_n - k)

            # Cohen's f
            cohens_f = math.sqrt(ms_between / ms_within) / math.sqrt(k)

            # Eta-squared
            eta_squared = ss_between / (ss_between + ss_within)

            return {
                "test_type": "One-Way ANOVA",
                "cohens_f": round(cohens_f, 4),
                "eta_squared": round(eta_squared, 4),
                "interpretation_f": get_cohens_f_interpretation(cohens_f),
                "interpretation_eta_squared": get_eta_squared_interpretation(eta_squared),
                "pilot_data": {
                    "num_groups": k,
                    "total_n": total_n,
                    "grand_mean": round(grand_mean, 4),
                    "group_means": request.group_means,
                    "group_sds": request.group_sds,
                    "group_ns": request.group_ns
                },
                "note": "This is an estimate from pilot data. Use with caution for final sample size calculations."
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculating effect size from pilot data: {str(e)}")


def get_cohens_d_interpretation(d: float) -> str:
    """Return interpretation of Cohen's d effect size"""
    d = abs(d)
    if d < 0.2:
        return "Negligible effect"
    elif d < 0.5:
        return "Small effect"
    elif d < 0.8:
        return "Medium effect"
    else:
        return "Large effect"


def get_cohens_f_interpretation(f: float) -> str:
    """Return interpretation of Cohen's f effect size"""
    f = abs(f)
    if f < 0.10:
        return "Negligible effect"
    elif f < 0.25:
        return "Small effect"
    elif f < 0.40:
        return "Medium effect"
    else:
        return "Large effect"


def get_eta_squared_interpretation(eta_sq: float) -> str:
    """Return interpretation of eta-squared effect size"""
    if eta_sq < 0.01:
        return "Negligible effect"
    elif eta_sq < 0.06:
        return "Small effect"
    elif eta_sq < 0.14:
        return "Medium effect"
    else:
        return "Large effect"


# ============================================================================
# PROPORTION TEST SAMPLE SIZE CALCULATIONS
# ============================================================================

class ProportionSampleSizeRequest(BaseModel):
    """Request for calculating sample size for proportion tests"""
    test_type: Literal["one-sample", "two-sample", "mcnemar"] = Field(..., description="Type of proportion test")
    power: float = Field(0.8, ge=0.5, le=0.99, description="Desired statistical power (1-β)")
    alpha: float = Field(0.05, ge=0.001, le=0.5, description="Significance level (α)")
    alternative: Literal["two-sided", "greater", "less"] = Field("two-sided", description="Alternative hypothesis")

    # One-sample test parameters
    p0: Optional[float] = Field(None, ge=0, le=1, description="Null hypothesis proportion")
    p1: Optional[float] = Field(None, ge=0, le=1, description="Alternative hypothesis proportion")

    # Two-sample test parameters
    p1_group1: Optional[float] = Field(None, ge=0, le=1, description="Proportion for group 1")
    p2_group2: Optional[float] = Field(None, ge=0, le=1, description="Proportion for group 2")
    ratio: float = Field(1.0, ge=0.1, le=10, description="Ratio of n2/n1 for unequal sample sizes")

    # McNemar test parameters
    p_discordant: Optional[float] = Field(None, ge=0, le=1, description="Proportion of discordant pairs")
    p_diff: Optional[float] = Field(None, description="Difference in proportions (p10 - p01)")


@router.post("/sample-size/proportion")
def calculate_proportion_sample_size(request: ProportionSampleSizeRequest):
    """
    Calculate required sample size for proportion tests
    """
    try:
        # Determine z-values based on alpha and alternative
        if request.alternative == "two-sided":
            z_alpha = stats.norm.ppf(1 - request.alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - request.alpha)

        z_beta = stats.norm.ppf(request.power)

        if request.test_type == "one-sample":
            # One-sample proportion test
            if request.p0 is None or request.p1 is None:
                raise HTTPException(
                    status_code=400,
                    detail="p0 and p1 are required for one-sample proportion test"
                )

            p0 = request.p0
            p1 = request.p1

            if abs(p1 - p0) < 0.0001:
                raise HTTPException(
                    status_code=400,
                    detail="p1 must be different from p0"
                )

            # Sample size formula for one proportion
            # Using normal approximation
            effect_size = abs(p1 - p0)

            # Under H0
            se0 = math.sqrt(p0 * (1 - p0))

            # Under H1
            se1 = math.sqrt(p1 * (1 - p1))

            # Sample size calculation
            n = ((z_alpha * se0 + z_beta * se1) / effect_size) ** 2
            n = math.ceil(n)

            # Cohen's h effect size for proportions
            cohens_h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p0)))

            result = {
                "test_type": "One-Sample Proportion Test",
                "sample_size": n,
                "total_sample_size": n,
                "parameters": {
                    "p0": p0,
                    "p1": p1,
                    "effect_size": round(abs(p1 - p0), 4),
                    "cohens_h": round(abs(cohens_h), 4),
                    "power": request.power,
                    "alpha": request.alpha,
                    "alternative": request.alternative
                },
                "interpretation": (
                    f"To detect a difference from {p0:.3f} to {p1:.3f} "
                    f"(effect size = {abs(p1-p0):.3f}) with {request.power*100:.0f}% power "
                    f"at α = {request.alpha}, you need {n} observations."
                )
            }

        elif request.test_type == "two-sample":
            # Two-sample proportion test (independent groups)
            if request.p1_group1 is None or request.p2_group2 is None:
                raise HTTPException(
                    status_code=400,
                    detail="p1_group1 and p2_group2 are required for two-sample proportion test"
                )

            p1 = request.p1_group1
            p2 = request.p2_group2

            if abs(p2 - p1) < 0.0001:
                raise HTTPException(
                    status_code=400,
                    detail="p2_group2 must be different from p1_group1"
                )

            effect_size = abs(p2 - p1)

            # Pooled proportion under H0
            p_pooled = (p1 + request.ratio * p2) / (1 + request.ratio)

            # Standard error under H0
            se0 = math.sqrt(p_pooled * (1 - p_pooled) * (1 + 1/request.ratio))

            # Standard error under H1
            se1 = math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / request.ratio)

            # Sample size for group 1
            n1 = ((z_alpha * se0 + z_beta * se1) / effect_size) ** 2
            n1 = math.ceil(n1)
            n2 = math.ceil(n1 * request.ratio)

            # Cohen's h for two proportions
            cohens_h = 2 * abs(math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

            result = {
                "test_type": "Two-Sample Proportion Test",
                "per_group": {
                    "group_1": n1,
                    "group_2": n2
                },
                "total_sample_size": n1 + n2,
                "parameters": {
                    "p1": p1,
                    "p2": p2,
                    "effect_size": round(effect_size, 4),
                    "cohens_h": round(cohens_h, 4),
                    "power": request.power,
                    "alpha": request.alpha,
                    "alternative": request.alternative,
                    "ratio": request.ratio
                },
                "interpretation": (
                    f"To detect a difference from {p1:.3f} to {p2:.3f} "
                    f"(effect size = {effect_size:.3f}) with {request.power*100:.0f}% power "
                    f"at α = {request.alpha}, you need {n1} participants in group 1 and "
                    f"{n2} in group 2 (total N = {n1 + n2})."
                )
            }

        else:  # mcnemar
            # McNemar's test for paired proportions
            if request.p_discordant is None or request.p_diff is None:
                raise HTTPException(
                    status_code=400,
                    detail="p_discordant and p_diff are required for McNemar's test"
                )

            p_disc = request.p_discordant
            p_diff = request.p_diff

            if p_disc <= 0 or p_disc > 1:
                raise HTTPException(
                    status_code=400,
                    detail="p_discordant must be between 0 and 1"
                )

            # For McNemar's test
            # p10 = proportion with outcome + at time 1, - at time 2
            # p01 = proportion with outcome - at time 1, + at time 2
            # p_diff = p10 - p01
            # p_disc = p10 + p01 (total discordant pairs)

            # Calculate p10 and p01
            p10 = (p_disc + p_diff) / 2
            p01 = (p_disc - p_diff) / 2

            if p10 < 0 or p10 > 1 or p01 < 0 or p01 > 1:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid combination of p_discordant and p_diff"
                )

            # Sample size for McNemar's test
            # Based on discordant pairs
            n = ((z_alpha * math.sqrt(p_disc) + z_beta * math.sqrt(p10 + p01 - (p10 - p01)**2)) / (p10 - p01)) ** 2
            n = math.ceil(n)

            # Effect size (odds ratio)
            if p01 > 0:
                odds_ratio = p10 / p01
            else:
                odds_ratio = float('inf')

            result = {
                "test_type": "McNemar's Test (Paired Proportions)",
                "sample_size": n,
                "total_sample_size": n,
                "parameters": {
                    "p_discordant": p_disc,
                    "p_diff": p_diff,
                    "p10": round(p10, 4),
                    "p01": round(p01, 4),
                    "odds_ratio": round(odds_ratio, 4) if odds_ratio != float('inf') else "inf",
                    "power": request.power,
                    "alpha": request.alpha,
                    "alternative": request.alternative
                },
                "interpretation": (
                    f"To detect a difference in proportions with {p_disc*100:.1f}% discordant pairs "
                    f"and a difference of {abs(p_diff)*100:.1f}% with {request.power*100:.0f}% power "
                    f"at α = {request.alpha}, you need {n} matched pairs."
                )
            }

        # Add effect size classification
        if "cohens_h" in result["parameters"]:
            result["effect_size_classification"] = classify_cohens_h(result["parameters"]["cohens_h"])

        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Calculation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def classify_cohens_h(h: float) -> dict:
    """Classify Cohen's h effect size for proportions"""
    abs_h = abs(h)
    if abs_h < 0.2:
        classification = "small"
        description = "small effect size"
    elif abs_h < 0.5:
        classification = "medium"
        description = "medium effect size"
    else:
        classification = "large"
        description = "large effect size"

    return {
        "classification": classification,
        "description": description,
        "cohens_h": round(abs_h, 4)
    }


class CostBenefitRequest(BaseModel):
    test_family: Literal["t-test", "anova", "proportion"] = Field(..., description="Test family")
    test_type: str = Field(..., description="Specific test type")
    effect_size: float = Field(..., description="Expected effect size")
    alpha: float = Field(0.05, description="Significance level")
    alternative: Optional[str] = Field("two-sided", description="Alternative hypothesis")
    cost_per_participant: float = Field(..., gt=0, description="Cost per participant")
    max_budget: Optional[float] = Field(None, gt=0, description="Maximum budget available")
    min_power: float = Field(0.5, ge=0.5, le=0.99, description="Minimum acceptable power")
    max_power: float = Field(0.95, ge=0.5, le=0.99, description="Maximum desired power")

    # Test-specific parameters
    ratio: Optional[float] = Field(1.0, description="Ratio for two-sample tests")
    correlation: Optional[float] = Field(0.5, description="Correlation for paired tests")
    num_groups: Optional[int] = Field(3, description="Number of groups for ANOVA")
    num_levels_a: Optional[int] = Field(2, description="Levels of factor A for two-way ANOVA")
    num_levels_b: Optional[int] = Field(2, description="Levels of factor B for two-way ANOVA")


@router.post("/cost-benefit-analysis")
def calculate_cost_benefit(request: CostBenefitRequest):
    """
    Calculate cost-benefit analysis for study design
    Shows total cost, cost per unit power, and optimal sample size given budget
    """
    try:
        # Generate power curve data for different sample sizes
        power_levels = np.linspace(request.min_power, request.max_power, 10)
        sample_sizes = []
        total_costs = []
        cost_per_power_unit = []

        for power in power_levels:
            # Calculate sample size for this power level
            if request.test_family == "t-test":
                power_obj = TTestIndPower() if request.test_type == "two-sample" else TTestPower()

                if request.test_type == "two-sample":
                    n = power_obj.solve_power(
                        effect_size=abs(request.effect_size),
                        alpha=request.alpha,
                        power=power,
                        ratio=request.ratio,
                        alternative=request.alternative or "two-sided"
                    )
                    total_n = int(np.ceil(n * (1 + request.ratio)))
                else:
                    n = power_obj.solve_power(
                        effect_size=abs(request.effect_size),
                        alpha=request.alpha,
                        power=power,
                        alternative=request.alternative or "two-sided"
                    )
                    total_n = int(np.ceil(n))

            elif request.test_family == "anova":
                power_obj = FTestAnovaPower()

                if request.test_type == "one-way":
                    k = request.num_groups
                else:
                    k = request.num_levels_a * request.num_levels_b

                n_per_group = power_obj.solve_power(
                    effect_size=abs(request.effect_size),
                    alpha=request.alpha,
                    power=power,
                    k_groups=k
                )
                total_n = int(np.ceil(n_per_group * k))
            else:
                # Proportions - simplified calculation
                from statsmodels.stats.power import zt_ind_solve_power
                n = zt_ind_solve_power(
                    effect_size=abs(request.effect_size),
                    alpha=request.alpha,
                    power=power,
                    ratio=request.ratio or 1.0,
                    alternative=request.alternative or "two-sided"
                )
                total_n = int(np.ceil(n * (1 + (request.ratio or 1.0))))

            sample_sizes.append(total_n)
            cost = total_n * request.cost_per_participant
            total_costs.append(cost)
            cost_per_power_unit.append(cost / power if power > 0 else 0)

        # Find optimal sample size given budget constraint
        optimal_sample_size = None
        optimal_power = None
        optimal_cost = None
        budget_exceeded = False

        if request.max_budget:
            max_affordable_n = int(request.max_budget / request.cost_per_participant)

            # Find the highest power achievable within budget
            for i, (n, power, cost) in enumerate(zip(sample_sizes, power_levels, total_costs)):
                if n <= max_affordable_n:
                    optimal_sample_size = n
                    optimal_power = power
                    optimal_cost = cost

            if optimal_sample_size is None:
                # Even minimum power exceeds budget
                budget_exceeded = True
                optimal_sample_size = sample_sizes[0]
                optimal_power = power_levels[0]
                optimal_cost = total_costs[0]
        else:
            # No budget constraint - use 80% power as default
            target_power = 0.8
            closest_idx = np.argmin([abs(p - target_power) for p in power_levels])
            optimal_sample_size = sample_sizes[closest_idx]
            optimal_power = power_levels[closest_idx]
            optimal_cost = total_costs[closest_idx]

        # Calculate cost efficiency metrics
        min_cost_idx = np.argmin(cost_per_power_unit[1:]) + 1  # Skip first to avoid div by small power
        most_efficient_n = sample_sizes[min_cost_idx]
        most_efficient_power = power_levels[min_cost_idx]
        most_efficient_cost = total_costs[min_cost_idx]

        return {
            "cost_curve": {
                "sample_sizes": [int(n) for n in sample_sizes],
                "power_levels": [round(float(p), 4) for p in power_levels],
                "total_costs": [round(float(c), 2) for c in total_costs],
                "cost_per_power_unit": [round(float(c), 2) for c in cost_per_power_unit]
            },
            "optimal_design": {
                "sample_size": int(optimal_sample_size),
                "power": round(float(optimal_power), 4),
                "total_cost": round(float(optimal_cost), 2),
                "cost_per_participant": request.cost_per_participant,
                "budget_exceeded": budget_exceeded
            },
            "most_efficient_design": {
                "sample_size": int(most_efficient_n),
                "power": round(float(most_efficient_power), 4),
                "total_cost": round(float(most_efficient_cost), 2),
                "cost_per_power_unit": round(float(cost_per_power_unit[min_cost_idx]), 2),
                "description": "Best balance of power vs. cost"
            },
            "budget_info": {
                "max_budget": request.max_budget,
                "has_constraint": request.max_budget is not None,
                "max_affordable_participants": int(request.max_budget / request.cost_per_participant) if request.max_budget else None
            },
            "recommendations": generate_cost_recommendations(
                optimal_sample_size,
                optimal_power,
                optimal_cost,
                request.max_budget,
                budget_exceeded,
                most_efficient_n,
                most_efficient_power
            )
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def generate_cost_recommendations(
    optimal_n, optimal_power, optimal_cost,
    max_budget, budget_exceeded,
    efficient_n, efficient_power
):
    """Generate recommendations for cost-benefit trade-offs"""
    recommendations = []

    if budget_exceeded:
        recommendations.append(
            f"⚠ Your budget constraint limits you to {optimal_power*100:.1f}% power. "
            f"Consider increasing budget or accepting lower effect size detection."
        )
    else:
        if max_budget and optimal_cost < max_budget * 0.7:
            recommendations.append(
                f"✓ You have {max_budget - optimal_cost:.2f} remaining in your budget. "
                f"You could increase power to {min(optimal_power + 0.1, 0.95)*100:.1f}% for better sensitivity."
            )

    if optimal_n != efficient_n:
        savings = optimal_cost - (efficient_n * (optimal_cost / optimal_n))
        recommendations.append(
            f"💡 Most cost-efficient design: n={efficient_n} ({efficient_power*100:.1f}% power) "
            f"could save ${abs(savings):.2f} while maintaining good statistical properties."
        )

    if optimal_power < 0.8:
        recommendations.append(
            "⚠ Power is below the conventional 80% threshold. "
            "Consider if this aligns with your tolerance for Type II errors."
        )

    return recommendations


class OptimalAllocationRequest(BaseModel):
    test_family: Literal["t-test", "proportion"] = Field(..., description="Test family (only t-test and proportion support unequal allocation)")
    effect_size: float = Field(..., description="Expected effect size")
    alpha: float = Field(0.05, description="Significance level")
    power: float = Field(0.8, ge=0.5, le=0.99, description="Desired power")
    alternative: str = Field("two-sided", description="Alternative hypothesis")

    # Cost parameters
    cost_group1: float = Field(..., gt=0, description="Cost per participant in group 1")
    cost_group2: float = Field(..., gt=0, description="Cost per participant in group 2")

    # Optional constraints
    max_budget: Optional[float] = Field(None, gt=0, description="Maximum total budget")
    max_n_group1: Optional[int] = Field(None, gt=0, description="Maximum participants in group 1")
    max_n_group2: Optional[int] = Field(None, gt=0, description="Maximum participants in group 2")


@router.post("/optimal-allocation")
def calculate_optimal_allocation(request: OptimalAllocationRequest):
    """
    Calculate optimal allocation of participants to groups when costs differ
    Uses Neyman allocation: n1/n2 = sqrt(c2/c1) for minimum variance given total cost
    """
    try:
        # Calculate optimal allocation ratio (minimizes variance for fixed cost)
        # For equal variances: n1/n2 = sqrt(c2/c1)
        optimal_ratio = np.sqrt(request.cost_group2 / request.cost_group1)

        # Calculate sample sizes with equal allocation (1:1 ratio)
        if request.test_family == "t-test":
            power_obj = TTestIndPower()
            n_equal = power_obj.solve_power(
                effect_size=abs(request.effect_size),
                alpha=request.alpha,
                power=request.power,
                ratio=1.0,
                alternative=request.alternative
            )
            n1_equal = int(np.ceil(n_equal))
            n2_equal = int(np.ceil(n_equal))
            total_n_equal = n1_equal + n2_equal
            cost_equal = n1_equal * request.cost_group1 + n2_equal * request.cost_group2

            # Calculate sample sizes with optimal allocation
            n_optimal = power_obj.solve_power(
                effect_size=abs(request.effect_size),
                alpha=request.alpha,
                power=request.power,
                ratio=optimal_ratio,
                alternative=request.alternative
            )
            n1_optimal = int(np.ceil(n_optimal))
            n2_optimal = int(np.ceil(n_optimal * optimal_ratio))
            total_n_optimal = n1_optimal + n2_optimal
            cost_optimal = n1_optimal * request.cost_group1 + n2_optimal * request.cost_group2

        else:  # proportion
            from statsmodels.stats.power import zt_ind_solve_power
            n_equal = zt_ind_solve_power(
                effect_size=abs(request.effect_size),
                alpha=request.alpha,
                power=request.power,
                ratio=1.0,
                alternative=request.alternative
            )
            n1_equal = int(np.ceil(n_equal))
            n2_equal = int(np.ceil(n_equal))
            total_n_equal = n1_equal + n2_equal
            cost_equal = n1_equal * request.cost_group1 + n2_equal * request.cost_group2

            # Optimal allocation
            n_optimal = zt_ind_solve_power(
                effect_size=abs(request.effect_size),
                alpha=request.alpha,
                power=request.power,
                ratio=optimal_ratio,
                alternative=request.alternative
            )
            n1_optimal = int(np.ceil(n_optimal))
            n2_optimal = int(np.ceil(n_optimal * optimal_ratio))
            total_n_optimal = n1_optimal + n2_optimal
            cost_optimal = n1_optimal * request.cost_group1 + n2_optimal * request.cost_group2

        # Calculate cost savings
        cost_savings = cost_equal - cost_optimal
        cost_savings_percent = (cost_savings / cost_equal) * 100 if cost_equal > 0 else 0

        # Check budget constraints
        budget_feasible = True
        budget_message = None
        if request.max_budget:
            if cost_optimal > request.max_budget:
                budget_feasible = False
                budget_message = f"Optimal design exceeds budget by ${cost_optimal - request.max_budget:.2f}"

        # Check sample size constraints
        constraints_met = True
        constraint_messages = []
        if request.max_n_group1 and n1_optimal > request.max_n_group1:
            constraints_met = False
            constraint_messages.append(f"Group 1 size ({n1_optimal}) exceeds maximum ({request.max_n_group1})")
        if request.max_n_group2 and n2_optimal > request.max_n_group2:
            constraints_met = False
            constraint_messages.append(f"Group 2 size ({n2_optimal}) exceeds maximum ({request.max_n_group2})")

        # Generate allocation comparison data for visualization
        ratios = np.linspace(0.5, 2.0, 20)
        costs_by_ratio = []
        power_by_ratio = []

        for ratio in ratios:
            try:
                if request.test_family == "t-test":
                    n = power_obj.solve_power(
                        effect_size=abs(request.effect_size),
                        alpha=request.alpha,
                        power=request.power,
                        ratio=ratio,
                        alternative=request.alternative
                    )
                else:
                    n = zt_ind_solve_power(
                        effect_size=abs(request.effect_size),
                        alpha=request.alpha,
                        power=request.power,
                        ratio=ratio,
                        alternative=request.alternative
                    )
                n1 = int(np.ceil(n))
                n2 = int(np.ceil(n * ratio))
                cost = n1 * request.cost_group1 + n2 * request.cost_group2
                costs_by_ratio.append(cost)
                power_by_ratio.append(request.power)
            except:
                costs_by_ratio.append(None)
                power_by_ratio.append(None)

        return {
            "equal_allocation": {
                "n_group1": n1_equal,
                "n_group2": n2_equal,
                "total_n": total_n_equal,
                "ratio": 1.0,
                "total_cost": round(cost_equal, 2),
                "cost_per_group1": request.cost_group1,
                "cost_per_group2": request.cost_group2
            },
            "optimal_allocation": {
                "n_group1": n1_optimal,
                "n_group2": n2_optimal,
                "total_n": total_n_optimal,
                "ratio": round(optimal_ratio, 4),
                "total_cost": round(cost_optimal, 2),
                "cost_per_group1": request.cost_group1,
                "cost_per_group2": request.cost_group2,
                "cost_savings": round(cost_savings, 2),
                "cost_savings_percent": round(cost_savings_percent, 2)
            },
            "comparison_curve": {
                "ratios": [round(float(r), 2) for r in ratios],
                "total_costs": [round(float(c), 2) if c is not None else None for c in costs_by_ratio],
                "power_levels": [round(float(p), 4) if p is not None else None for p in power_by_ratio]
            },
            "feasibility": {
                "budget_feasible": budget_feasible,
                "budget_message": budget_message,
                "constraints_met": constraints_met,
                "constraint_messages": constraint_messages
            },
            "interpretation": {
                "formula": f"Optimal ratio n₁/n₂ = √(c₂/c₁) = √({request.cost_group2}/{request.cost_group1}) = {optimal_ratio:.3f}",
                "explanation": (
                    f"When group 2 costs {request.cost_group2/request.cost_group1:.2f}x more than group 1, "
                    f"the optimal allocation is to have {optimal_ratio:.2f}x more participants in the cheaper group. "
                    f"This minimizes total cost while maintaining statistical power."
                ),
                "savings_summary": (
                    f"Optimal allocation saves ${cost_savings:.2f} ({cost_savings_percent:.1f}%) "
                    f"compared to equal allocation, while achieving the same {request.power*100:.0f}% power."
                )
            },
            "recommendations": generate_allocation_recommendations(
                n1_optimal, n2_optimal, cost_savings, cost_savings_percent,
                budget_feasible, constraints_met, optimal_ratio
            )
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def generate_allocation_recommendations(
    n1, n2, savings, savings_percent,
    budget_feasible, constraints_met, ratio
):
    """Generate recommendations for optimal allocation"""
    recommendations = []

    if savings_percent > 10:
        recommendations.append(
            f"💰 Significant cost savings: Using optimal allocation saves {savings_percent:.1f}% "
            f"(${savings:.2f}) compared to equal group sizes."
        )
    elif savings_percent > 0:
        recommendations.append(
            f"💡 Modest cost savings: Optimal allocation saves {savings_percent:.1f}% "
            f"(${savings:.2f}). Consider if the complexity of unequal groups is worth the savings."
        )
    else:
        recommendations.append(
            "✓ Equal allocation is already optimal when costs are similar."
        )

    if not budget_feasible:
        recommendations.append(
            "⚠ Budget constraint exceeded. Consider reducing power, accepting smaller effect size detection, "
            "or increasing budget."
        )

    if not constraints_met:
        recommendations.append(
            "⚠ Sample size constraints cannot be met. Consider relaxing constraints or adjusting power requirements."
        )

    if ratio > 1.5 or ratio < 0.67:
        recommendations.append(
            f"⚠ Large allocation imbalance (ratio = {ratio:.2f}). Ensure this is practical for your study design. "
            "Severely unequal groups may raise concerns about bias or generalizability."
        )

    recommendations.append(
        "📊 Use the comparison curve to explore trade-offs between different allocation ratios and total costs."
    )

    return recommendations
