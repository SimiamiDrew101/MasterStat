"""
Reliability and Survival Analysis Module

Provides comprehensive reliability analysis capabilities including:
- Life Distribution Fitting (Weibull, Lognormal, Exponential, etc.)
- Kaplan-Meier Survival Analysis with log-rank tests
- Cox Proportional Hazards Regression
- Accelerated Life Testing (ALT)
- Reliability Test Planning

Uses the lifelines library for survival analysis computations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import warnings

# Lifelines imports
try:
    from lifelines import (
        KaplanMeierFitter,
        WeibullFitter,
        LogNormalFitter,
        ExponentialFitter,
        LogLogisticFitter,
        WeibullAFTFitter,
        LogNormalAFTFitter,
        LogLogisticAFTFitter,
        CoxPHFitter
    )
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    from lifelines.utils import median_survival_times
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

router = APIRouter()


# =============================================================================
# Helper Functions
# =============================================================================

def safe_float(value: Any) -> Optional[float]:
    """Safely convert value to float, handling NaN, inf, and None."""
    if value is None:
        return None
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return None
        return round(val, 6)
    except (TypeError, ValueError):
        return None


def make_json_safe(obj: Any) -> Any:
    """Recursively make an object JSON-serializable."""
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return round(val, 6)
    if isinstance(obj, np.ndarray):
        return [make_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [make_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, (int, float, str, bool)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    return str(obj)


def calculate_reliability_metrics(
    times: np.ndarray,
    events: np.ndarray,
    time_points: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Calculate basic reliability metrics from survival data."""
    n = len(times)
    n_events = int(np.sum(events))
    n_censored = n - n_events

    # Time statistics
    observed_times = times[events == 1]
    censored_times = times[events == 0]

    metrics = {
        "n_observations": n,
        "n_events": n_events,
        "n_censored": n_censored,
        "censoring_rate": safe_float(n_censored / n * 100),
        "time_range": {
            "min": safe_float(np.min(times)),
            "max": safe_float(np.max(times)),
            "mean": safe_float(np.mean(times)),
            "median": safe_float(np.median(times))
        }
    }

    if len(observed_times) > 0:
        metrics["event_time_stats"] = {
            "min": safe_float(np.min(observed_times)),
            "max": safe_float(np.max(observed_times)),
            "mean": safe_float(np.mean(observed_times)),
            "median": safe_float(np.median(observed_times))
        }

    return metrics


# =============================================================================
# Request Models
# =============================================================================

class LifeDistributionRequest(BaseModel):
    """Request model for life distribution fitting."""
    times: List[float] = Field(..., description="Time-to-event data (failure times)")
    events: List[int] = Field(..., description="Event indicators (1=event/failure, 0=censored)")
    distributions: Optional[List[str]] = Field(
        default=["weibull", "lognormal", "exponential", "loglogistic"],
        description="Distributions to fit"
    )
    confidence_level: float = Field(default=0.95, description="Confidence level for intervals")
    time_points: Optional[List[float]] = Field(
        default=None,
        description="Specific time points for reliability estimates"
    )


class KaplanMeierRequest(BaseModel):
    """Request model for Kaplan-Meier survival analysis."""
    times: List[float] = Field(..., description="Time-to-event data")
    events: List[int] = Field(..., description="Event indicators (1=event, 0=censored)")
    groups: Optional[List[str]] = Field(
        default=None,
        description="Group labels for stratified analysis"
    )
    confidence_level: float = Field(default=0.95, description="Confidence level for intervals")
    time_points: Optional[List[float]] = Field(
        default=None,
        description="Specific time points for survival estimates"
    )


class CoxPHRequest(BaseModel):
    """Request model for Cox Proportional Hazards regression."""
    times: List[float] = Field(..., description="Time-to-event data")
    events: List[int] = Field(..., description="Event indicators (1=event, 0=censored)")
    covariates: Dict[str, List[Union[float, str]]] = Field(
        ...,
        description="Covariate data as column_name: values"
    )
    confidence_level: float = Field(default=0.95, description="Confidence level for intervals")
    penalizer: float = Field(default=0.0, description="L2 regularization penalty")
    strata: Optional[List[str]] = Field(
        default=None,
        description="Stratification variable values"
    )


class ALTRequest(BaseModel):
    """Request model for Accelerated Life Testing analysis."""
    times: List[float] = Field(..., description="Time-to-event data")
    events: List[int] = Field(..., description="Event indicators (1=event, 0=censored)")
    stress_variable: str = Field(..., description="Name of the stress variable")
    stress_values: List[float] = Field(..., description="Stress level values for each observation")
    model_type: str = Field(
        default="weibull",
        description="Life distribution model: 'weibull', 'lognormal', 'loglogistic'"
    )
    use_stress: Optional[float] = Field(
        default=None,
        description="Stress level for use conditions (for extrapolation)"
    )
    confidence_level: float = Field(default=0.95, description="Confidence level for intervals")


class TestPlanningRequest(BaseModel):
    """Request model for reliability test planning."""
    target_reliability: float = Field(
        ...,
        description="Target reliability (proportion surviving, 0-1)",
        ge=0.0,
        le=1.0
    )
    test_time: float = Field(..., description="Duration of the test")
    confidence_level: float = Field(default=0.95, description="Confidence level")
    test_type: str = Field(
        default="demonstration",
        description="Test type: 'demonstration', 'estimation', 'comparison'"
    )
    allowable_failures: int = Field(default=0, description="Maximum allowable failures")
    distribution: str = Field(default="exponential", description="Assumed life distribution")
    shape_parameter: Optional[float] = Field(
        default=None,
        description="Shape parameter for Weibull distribution"
    )
    comparison_reliability: Optional[float] = Field(
        default=None,
        description="Reliability to compare against (for comparison tests)"
    )
    power: float = Field(default=0.8, description="Statistical power for test planning")


# =============================================================================
# Life Distribution Fitting Endpoint
# =============================================================================

@router.post("/life-distribution")
async def fit_life_distributions(request: LifeDistributionRequest):
    """
    Fit parametric life distributions to time-to-event data.

    Fits multiple distributions (Weibull, Lognormal, Exponential, Log-Logistic)
    and compares them using AIC/BIC criteria. Returns parameter estimates,
    reliability metrics, and probability plot data.
    """
    if not LIFELINES_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="lifelines library not available. Install with: pip install lifelines"
        )

    try:
        # Validate input
        times = np.array(request.times, dtype=float)
        events = np.array(request.events, dtype=int)

        if len(times) != len(events):
            raise HTTPException(
                status_code=400,
                detail="times and events arrays must have the same length"
            )

        if len(times) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 observations required for distribution fitting"
            )

        if np.any(times <= 0):
            raise HTTPException(
                status_code=400,
                detail="All times must be positive"
            )

        # Calculate basic metrics
        basic_metrics = calculate_reliability_metrics(times, events)

        # Define fitter classes
        fitter_classes = {
            "weibull": WeibullFitter,
            "lognormal": LogNormalFitter,
            "exponential": ExponentialFitter,
            "loglogistic": LogLogisticFitter
        }

        # Fit each distribution
        results = {}
        comparison = []

        for dist_name in request.distributions:
            if dist_name.lower() not in fitter_classes:
                continue

            try:
                fitter_class = fitter_classes[dist_name.lower()]
                fitter = fitter_class()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitter.fit(times, events, alpha=1 - request.confidence_level)

                # Extract parameters based on distribution type
                params = {}
                if dist_name.lower() == "weibull":
                    params = {
                        "lambda": safe_float(fitter.lambda_),  # Scale
                        "rho": safe_float(fitter.rho_),  # Shape
                        "scale": safe_float(fitter.lambda_),
                        "shape": safe_float(fitter.rho_),
                        "mttf": safe_float(fitter.median_survival_time_)
                    }
                    # Weibull interpretation
                    shape = fitter.rho_
                    if shape < 1:
                        failure_pattern = "Decreasing failure rate (infant mortality)"
                    elif shape > 1:
                        failure_pattern = "Increasing failure rate (wear-out)"
                    else:
                        failure_pattern = "Constant failure rate (random failures)"
                    params["interpretation"] = failure_pattern

                elif dist_name.lower() == "lognormal":
                    params = {
                        "mu": safe_float(fitter.mu_),
                        "sigma": safe_float(fitter.sigma_),
                        "median": safe_float(np.exp(fitter.mu_)),
                        "mttf": safe_float(fitter.median_survival_time_)
                    }

                elif dist_name.lower() == "exponential":
                    params = {
                        "lambda": safe_float(fitter.lambda_),
                        "mttf": safe_float(1 / fitter.lambda_),
                        "failure_rate": safe_float(fitter.lambda_)
                    }

                elif dist_name.lower() == "loglogistic":
                    params = {
                        "alpha": safe_float(fitter.alpha_),
                        "beta": safe_float(fitter.beta_),
                        "scale": safe_float(fitter.alpha_),
                        "shape": safe_float(fitter.beta_),
                        "median": safe_float(fitter.median_survival_time_)
                    }

                # Get confidence intervals for parameters
                conf_int = fitter.confidence_interval_
                param_cis = {}
                if conf_int is not None and hasattr(conf_int, 'values'):
                    for i, param_name in enumerate(fitter.summary.index):
                        ci_lower = conf_int.iloc[i, 0] if i < len(conf_int) else None
                        ci_upper = conf_int.iloc[i, 1] if i < len(conf_int) else None
                        param_cis[str(param_name)] = {
                            "lower": safe_float(ci_lower),
                            "upper": safe_float(ci_upper)
                        }

                # Generate survival curve data
                time_range = np.linspace(0.001, np.max(times) * 1.5, 100)
                survival_probs = fitter.survival_function_at_times(time_range)

                # Calculate reliability at specific time points if requested
                reliability_at_times = {}
                if request.time_points:
                    for t in request.time_points:
                        if t > 0:
                            r = fitter.survival_function_at_times([t]).values[0]
                            ci = fitter.confidence_interval_survival_function_at_times([t])
                            reliability_at_times[str(t)] = {
                                "reliability": safe_float(r),
                                "ci_lower": safe_float(ci.iloc[0, 0]) if ci is not None else None,
                                "ci_upper": safe_float(ci.iloc[0, 1]) if ci is not None else None
                            }

                # Probability plot data (for Weibull plot)
                prob_plot_data = None
                if dist_name.lower() == "weibull":
                    # Median rank approximation for probability plotting
                    sorted_times = np.sort(times[events == 1])
                    n_failures = len(sorted_times)
                    if n_failures > 0:
                        median_ranks = [(i - 0.3) / (n_failures + 0.4) for i in range(1, n_failures + 1)]
                        prob_plot_data = {
                            "x": [safe_float(np.log(t)) for t in sorted_times],
                            "y": [safe_float(np.log(-np.log(1 - p))) for p in median_ranks],
                            "times": [safe_float(t) for t in sorted_times],
                            "probabilities": [safe_float(p) for p in median_ranks]
                        }

                # Model fit statistics
                log_likelihood = safe_float(fitter.log_likelihood_)
                n_params = len(fitter.summary)
                aic = safe_float(-2 * fitter.log_likelihood_ + 2 * n_params)
                bic = safe_float(-2 * fitter.log_likelihood_ + n_params * np.log(len(times)))

                results[dist_name] = {
                    "parameters": params,
                    "parameter_confidence_intervals": param_cis,
                    "fit_statistics": {
                        "log_likelihood": log_likelihood,
                        "aic": aic,
                        "bic": bic,
                        "n_parameters": n_params
                    },
                    "survival_curve": {
                        "times": [safe_float(t) for t in time_range],
                        "survival": [safe_float(s) for s in survival_probs.values]
                    },
                    "reliability_at_times": reliability_at_times,
                    "probability_plot": prob_plot_data,
                    "median_survival_time": safe_float(fitter.median_survival_time_),
                    "summary": fitter.summary.to_dict() if hasattr(fitter, 'summary') else None
                }

                comparison.append({
                    "distribution": dist_name,
                    "aic": aic,
                    "bic": bic,
                    "log_likelihood": log_likelihood,
                    "median_survival": safe_float(fitter.median_survival_time_)
                })

            except Exception as e:
                results[dist_name] = {
                    "error": str(e),
                    "message": f"Failed to fit {dist_name} distribution"
                }

        # Sort comparison by AIC
        comparison = sorted(comparison, key=lambda x: x["aic"] if x["aic"] is not None else float('inf'))

        # Identify best distribution
        best_distribution = comparison[0]["distribution"] if comparison else None

        return make_json_safe({
            "basic_metrics": basic_metrics,
            "distributions": results,
            "comparison": comparison,
            "best_distribution": best_distribution,
            "recommendation": f"Based on AIC, the {best_distribution} distribution provides the best fit for this data." if best_distribution else None,
            "message": "Life distribution analysis completed successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Life distribution analysis failed: {str(e)}")


# =============================================================================
# Kaplan-Meier Survival Analysis Endpoint
# =============================================================================

@router.post("/kaplan-meier")
async def kaplan_meier_analysis(request: KaplanMeierRequest):
    """
    Perform Kaplan-Meier survival analysis with optional group comparisons.

    Returns survival curves, confidence intervals, median survival times,
    and log-rank tests for comparing groups.
    """
    if not LIFELINES_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="lifelines library not available. Install with: pip install lifelines"
        )

    try:
        # Validate input
        times = np.array(request.times, dtype=float)
        events = np.array(request.events, dtype=int)

        if len(times) != len(events):
            raise HTTPException(
                status_code=400,
                detail="times and events arrays must have the same length"
            )

        if len(times) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 observations required"
            )

        # Calculate basic metrics
        basic_metrics = calculate_reliability_metrics(times, events)
        alpha = 1 - request.confidence_level

        results = {}

        if request.groups is None:
            # Single group analysis
            kmf = KaplanMeierFitter()
            kmf.fit(times, events, alpha=alpha, label="Overall")

            # Get survival function
            sf = kmf.survival_function_
            ci_lower = kmf.confidence_interval_survival_function_.iloc[:, 0]
            ci_upper = kmf.confidence_interval_survival_function_.iloc[:, 1]

            # Create time points for smooth curve
            time_points = sf.index.tolist()

            results["overall"] = {
                "survival_curve": {
                    "times": [safe_float(t) for t in time_points],
                    "survival": [safe_float(s) for s in sf["Overall"].values],
                    "ci_lower": [safe_float(l) for l in ci_lower.values],
                    "ci_upper": [safe_float(u) for u in ci_upper.values]
                },
                "median_survival_time": safe_float(kmf.median_survival_time_),
                "median_ci": {
                    "lower": safe_float(median_survival_times(kmf.confidence_interval_).iloc[0, 0]),
                    "upper": safe_float(median_survival_times(kmf.confidence_interval_).iloc[0, 1])
                } if kmf.confidence_interval_ is not None else None,
                "event_table": kmf.event_table.to_dict() if hasattr(kmf, 'event_table') else None,
                "n_observations": len(times),
                "n_events": int(np.sum(events)),
                "n_censored": int(np.sum(1 - events))
            }

            # Survival at specific time points
            if request.time_points:
                survival_at_times = {}
                for t in request.time_points:
                    if t > 0:
                        s = kmf.predict(t)
                        survival_at_times[str(t)] = {
                            "survival": safe_float(s),
                            "se": None  # SE not directly available from predict
                        }
                results["overall"]["survival_at_times"] = survival_at_times

        else:
            # Grouped analysis
            groups = np.array(request.groups)

            if len(groups) != len(times):
                raise HTTPException(
                    status_code=400,
                    detail="groups array must have the same length as times"
                )

            unique_groups = np.unique(groups)
            group_results = {}

            for group in unique_groups:
                mask = groups == group
                group_times = times[mask]
                group_events = events[mask]

                kmf = KaplanMeierFitter()
                kmf.fit(group_times, group_events, alpha=alpha, label=str(group))

                sf = kmf.survival_function_
                ci_lower = kmf.confidence_interval_survival_function_.iloc[:, 0]
                ci_upper = kmf.confidence_interval_survival_function_.iloc[:, 1]
                time_points = sf.index.tolist()

                group_results[str(group)] = {
                    "survival_curve": {
                        "times": [safe_float(t) for t in time_points],
                        "survival": [safe_float(s) for s in sf[str(group)].values],
                        "ci_lower": [safe_float(l) for l in ci_lower.values],
                        "ci_upper": [safe_float(u) for u in ci_upper.values]
                    },
                    "median_survival_time": safe_float(kmf.median_survival_time_),
                    "n_observations": int(np.sum(mask)),
                    "n_events": int(np.sum(group_events)),
                    "n_censored": int(np.sum(1 - group_events))
                }

            results["groups"] = group_results

            # Log-rank test for comparing groups
            if len(unique_groups) == 2:
                # Two-group comparison
                group1 = unique_groups[0]
                group2 = unique_groups[1]

                mask1 = groups == group1
                mask2 = groups == group2

                lr_result = logrank_test(
                    times[mask1], times[mask2],
                    events[mask1], events[mask2],
                    alpha=alpha
                )

                results["log_rank_test"] = {
                    "test_statistic": safe_float(lr_result.test_statistic),
                    "p_value": safe_float(lr_result.p_value),
                    "degrees_of_freedom": 1,
                    "significant": bool(lr_result.p_value < alpha),
                    "interpretation": (
                        f"Significant difference in survival between groups (p={safe_float(lr_result.p_value):.4f})"
                        if lr_result.p_value < alpha
                        else f"No significant difference in survival between groups (p={safe_float(lr_result.p_value):.4f})"
                    )
                }

            elif len(unique_groups) > 2:
                # Multi-group comparison
                lr_result = multivariate_logrank_test(times, groups, events)

                results["log_rank_test"] = {
                    "test_statistic": safe_float(lr_result.test_statistic),
                    "p_value": safe_float(lr_result.p_value),
                    "degrees_of_freedom": len(unique_groups) - 1,
                    "significant": bool(lr_result.p_value < alpha),
                    "interpretation": (
                        f"Significant difference in survival among groups (p={safe_float(lr_result.p_value):.4f})"
                        if lr_result.p_value < alpha
                        else f"No significant difference in survival among groups (p={safe_float(lr_result.p_value):.4f})"
                    )
                }

        return make_json_safe({
            "basic_metrics": basic_metrics,
            "results": results,
            "confidence_level": request.confidence_level,
            "message": "Kaplan-Meier analysis completed successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kaplan-Meier analysis failed: {str(e)}")


# =============================================================================
# Cox Proportional Hazards Regression Endpoint
# =============================================================================

@router.post("/cox-ph")
async def cox_proportional_hazards(request: CoxPHRequest):
    """
    Fit Cox Proportional Hazards regression model.

    Returns hazard ratios, confidence intervals, concordance index,
    and model diagnostics for assessing covariate effects on survival.
    """
    if not LIFELINES_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="lifelines library not available. Install with: pip install lifelines"
        )

    try:
        # Validate input
        times = np.array(request.times, dtype=float)
        events = np.array(request.events, dtype=int)

        if len(times) != len(events):
            raise HTTPException(
                status_code=400,
                detail="times and events arrays must have the same length"
            )

        # Build dataframe with covariates
        df = pd.DataFrame({
            "T": times,
            "E": events
        })

        # Add covariates
        for cov_name, cov_values in request.covariates.items():
            if len(cov_values) != len(times):
                raise HTTPException(
                    status_code=400,
                    detail=f"Covariate '{cov_name}' must have the same length as times"
                )
            df[cov_name] = cov_values

        # Handle categorical variables (convert strings to dummies)
        categorical_cols = []
        for col in df.columns:
            if col not in ["T", "E"] and df[col].dtype == object:
                categorical_cols.append(col)

        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Get covariate columns
        covariate_cols = [col for col in df.columns if col not in ["T", "E"]]

        if len(covariate_cols) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one covariate is required for Cox regression"
            )

        # Fit Cox model
        cph = CoxPHFitter(penalizer=request.penalizer)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(df, duration_col="T", event_col="E")

        alpha = 1 - request.confidence_level

        # Extract results
        summary = cph.summary

        # Hazard ratios and confidence intervals
        hazard_ratios = {}
        for covariate in summary.index:
            hr = np.exp(summary.loc[covariate, "coef"])
            hr_lower = np.exp(summary.loc[covariate, f"coef lower {int(request.confidence_level*100)}%"])
            hr_upper = np.exp(summary.loc[covariate, f"coef upper {int(request.confidence_level*100)}%"])

            hazard_ratios[str(covariate)] = {
                "coefficient": safe_float(summary.loc[covariate, "coef"]),
                "se": safe_float(summary.loc[covariate, "se(coef)"]),
                "hazard_ratio": safe_float(hr),
                "hr_ci_lower": safe_float(hr_lower),
                "hr_ci_upper": safe_float(hr_upper),
                "z": safe_float(summary.loc[covariate, "z"]),
                "p_value": safe_float(summary.loc[covariate, "p"]),
                "significant": bool(summary.loc[covariate, "p"] < alpha),
                "interpretation": (
                    f"Each unit increase in {covariate} is associated with a "
                    f"{abs(hr - 1) * 100:.1f}% {'increase' if hr > 1 else 'decrease'} in hazard"
                )
            }

        # Model fit statistics
        concordance = cph.concordance_index_
        log_likelihood = cph.log_likelihood_

        # Likelihood ratio test
        ll_null = cph._log_likelihood_null
        lr_stat = 2 * (log_likelihood - ll_null)
        lr_df = len(covariate_cols)
        lr_pvalue = 1 - scipy_stats.chi2.cdf(lr_stat, lr_df)

        model_fit = {
            "concordance_index": safe_float(concordance),
            "concordance_interpretation": (
                "Excellent" if concordance >= 0.8 else
                "Good" if concordance >= 0.7 else
                "Moderate" if concordance >= 0.6 else
                "Poor"
            ),
            "log_likelihood": safe_float(log_likelihood),
            "log_likelihood_null": safe_float(ll_null),
            "likelihood_ratio_test": {
                "statistic": safe_float(lr_stat),
                "df": lr_df,
                "p_value": safe_float(lr_pvalue),
                "significant": bool(lr_pvalue < alpha)
            },
            "aic": safe_float(-2 * log_likelihood + 2 * len(covariate_cols)),
            "n_observations": len(times),
            "n_events": int(np.sum(events))
        }

        # Proportional hazards test (Schoenfeld residuals)
        try:
            ph_test = cph.check_assumptions(df, show_plots=False, p_value_threshold=alpha)
            ph_results = {}
            # The test returns a tuple or dataframe with test results
        except Exception:
            ph_results = {"message": "Proportional hazards test not available"}

        # Baseline survival function
        baseline_sf = cph.baseline_survival_
        baseline_times = baseline_sf.index.tolist()
        baseline_survival = baseline_sf.values.flatten().tolist()

        # Forest plot data (for visualization)
        forest_plot_data = []
        for covariate in summary.index:
            hr = np.exp(summary.loc[covariate, "coef"])
            hr_lower = np.exp(summary.loc[covariate, f"coef lower {int(request.confidence_level*100)}%"])
            hr_upper = np.exp(summary.loc[covariate, f"coef upper {int(request.confidence_level*100)}%"])

            forest_plot_data.append({
                "covariate": str(covariate),
                "hr": safe_float(hr),
                "hr_lower": safe_float(hr_lower),
                "hr_upper": safe_float(hr_upper),
                "p_value": safe_float(summary.loc[covariate, "p"])
            })

        return make_json_safe({
            "hazard_ratios": hazard_ratios,
            "model_fit": model_fit,
            "baseline_survival": {
                "times": [safe_float(t) for t in baseline_times],
                "survival": [safe_float(s) for s in baseline_survival]
            },
            "forest_plot_data": forest_plot_data,
            "summary_table": summary.to_dict(),
            "covariates_used": covariate_cols,
            "confidence_level": request.confidence_level,
            "message": "Cox Proportional Hazards analysis completed successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cox PH analysis failed: {str(e)}")


# =============================================================================
# Accelerated Life Testing (ALT) Endpoint
# =============================================================================

@router.post("/alt")
async def accelerated_life_testing(request: ALTRequest):
    """
    Perform Accelerated Life Testing (ALT) analysis.

    Fits accelerated failure time (AFT) models to estimate reliability
    under normal use conditions from accelerated test data.
    """
    if not LIFELINES_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="lifelines library not available. Install with: pip install lifelines"
        )

    try:
        # Validate input
        times = np.array(request.times, dtype=float)
        events = np.array(request.events, dtype=int)
        stress = np.array(request.stress_values, dtype=float)

        if len(times) != len(events) or len(times) != len(stress):
            raise HTTPException(
                status_code=400,
                detail="times, events, and stress_values must have the same length"
            )

        # Build dataframe
        df = pd.DataFrame({
            "T": times,
            "E": events,
            request.stress_variable: stress
        })

        # Select AFT model based on distribution
        model_classes = {
            "weibull": WeibullAFTFitter,
            "lognormal": LogNormalAFTFitter,
            "loglogistic": LogLogisticAFTFitter
        }

        if request.model_type.lower() not in model_classes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type. Choose from: {list(model_classes.keys())}"
            )

        alpha = 1 - request.confidence_level

        # Fit AFT model
        aft_class = model_classes[request.model_type.lower()]
        aft = aft_class()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aft.fit(df, duration_col="T", event_col="E")

        # Extract coefficients
        summary = aft.summary
        coefficients = {}

        for idx in summary.index:
            param_name = str(idx[0]) if isinstance(idx, tuple) else str(idx)
            coef_name = str(idx[1]) if isinstance(idx, tuple) else str(idx)

            if param_name not in coefficients:
                coefficients[param_name] = {}

            coefficients[param_name][coef_name] = {
                "estimate": safe_float(summary.loc[idx, "coef"]),
                "se": safe_float(summary.loc[idx, "se(coef)"]),
                "z": safe_float(summary.loc[idx, "z"]),
                "p_value": safe_float(summary.loc[idx, "p"]),
                "ci_lower": safe_float(summary.loc[idx, f"coef lower {int(request.confidence_level*100)}%"]),
                "ci_upper": safe_float(summary.loc[idx, f"coef upper {int(request.confidence_level*100)}%"])
            }

        # Model fit statistics
        model_fit = {
            "log_likelihood": safe_float(aft.log_likelihood_),
            "aic": safe_float(aft.AIC_),
            "concordance_index": safe_float(aft.concordance_index_),
            "n_observations": len(times),
            "n_events": int(np.sum(events))
        }

        # Acceleration factor interpretation
        stress_coef = None
        for param_group in coefficients.values():
            if request.stress_variable in param_group:
                stress_coef = param_group[request.stress_variable]["estimate"]
                break

        acceleration_factor = None
        if stress_coef is not None and request.use_stress is not None:
            # Calculate acceleration factor relative to use stress
            unique_stresses = np.unique(stress)
            acceleration_factors = {}
            for test_stress in unique_stresses:
                # AF = exp(coef * (test_stress - use_stress))
                af = np.exp(stress_coef * (test_stress - request.use_stress))
                acceleration_factors[str(test_stress)] = safe_float(af)
            acceleration_factor = acceleration_factors

        # Predictions at use conditions
        use_predictions = None
        if request.use_stress is not None:
            # Create prediction dataframe at use stress level
            pred_df = pd.DataFrame({
                request.stress_variable: [request.use_stress]
            })

            try:
                # Predict survival function at use conditions
                sf = aft.predict_survival_function(pred_df)
                median_time = aft.predict_median(pred_df)

                use_predictions = {
                    "stress_level": request.use_stress,
                    "median_life": safe_float(median_time.values[0]),
                    "survival_curve": {
                        "times": [safe_float(t) for t in sf.index.tolist()],
                        "survival": [safe_float(s) for s in sf.values.flatten().tolist()]
                    }
                }
            except Exception:
                use_predictions = {
                    "stress_level": request.use_stress,
                    "message": "Could not generate predictions at use conditions"
                }

        # Stress-life relationship plot data
        stress_life_data = []
        unique_stresses = np.unique(stress)
        for s in unique_stresses:
            mask = stress == s
            stress_times = times[mask]
            stress_events = events[mask]

            # Fit distribution at each stress level
            try:
                wf = WeibullFitter()
                wf.fit(stress_times, stress_events)
                median = wf.median_survival_time_
            except Exception:
                median = np.median(stress_times[stress_events == 1]) if np.any(stress_events) else np.median(stress_times)

            stress_life_data.append({
                "stress": safe_float(s),
                "median_life": safe_float(median),
                "n_observations": int(np.sum(mask)),
                "n_events": int(np.sum(stress_events))
            })

        return make_json_safe({
            "model_type": request.model_type,
            "stress_variable": request.stress_variable,
            "coefficients": coefficients,
            "model_fit": model_fit,
            "acceleration_factors": acceleration_factor,
            "use_condition_predictions": use_predictions,
            "stress_life_data": stress_life_data,
            "summary_table": summary.to_dict(),
            "confidence_level": request.confidence_level,
            "interpretation": (
                f"The {request.stress_variable} coefficient indicates the effect of stress on lifetime. "
                f"Positive values mean higher stress reduces life; negative values mean higher stress increases life."
            ),
            "message": "Accelerated Life Testing analysis completed successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ALT analysis failed: {str(e)}")


# =============================================================================
# Reliability Test Planning Endpoint
# =============================================================================

@router.post("/test-planning")
async def reliability_test_planning(request: TestPlanningRequest):
    """
    Calculate sample size requirements for reliability testing.

    Supports demonstration tests (prove a reliability level),
    estimation tests (estimate reliability with precision), and
    comparison tests (detect difference between designs).
    """
    try:
        alpha = 1 - request.confidence_level
        R = request.target_reliability
        t = request.test_time
        c = request.allowable_failures

        results = {}

        if request.test_type == "demonstration":
            # Reliability Demonstration Test
            # Based on binomial distribution: find n such that P(X <= c | p=1-R) >= confidence

            if request.distribution == "exponential":
                # For exponential, use chi-square relationship
                # 2*t*n / MTTF ~ chi-square(2*(r+1)) where r = number of failures

                # MTTF = t / (-ln(R))
                mttf_factor = -np.log(R)

                # Using the formula: n >= chi2(alpha, 2*(c+1)) / (2 * mttf_factor)
                chi2_val = scipy_stats.chi2.ppf(request.confidence_level, 2 * (c + 1))
                n_required = np.ceil(chi2_val / (2 * mttf_factor))

                results["demonstration_test"] = {
                    "sample_size": int(n_required),
                    "test_time": t,
                    "allowable_failures": c,
                    "target_reliability": R,
                    "confidence_level": request.confidence_level,
                    "distribution": "exponential",
                    "interpretation": (
                        f"Test {int(n_required)} units for {t} time units. "
                        f"If {c} or fewer failures occur, conclude R >= {R} with "
                        f"{request.confidence_level*100}% confidence."
                    )
                }

            elif request.distribution == "weibull":
                # Weibull demonstration test
                beta = request.shape_parameter or 2.0

                # Adjust time based on shape parameter
                # t_adj = t^beta
                t_adj = t ** beta
                mttf_factor = -np.log(R)

                chi2_val = scipy_stats.chi2.ppf(request.confidence_level, 2 * (c + 1))
                n_required = np.ceil(chi2_val / (2 * mttf_factor))

                results["demonstration_test"] = {
                    "sample_size": int(n_required),
                    "test_time": t,
                    "allowable_failures": c,
                    "target_reliability": R,
                    "shape_parameter": beta,
                    "confidence_level": request.confidence_level,
                    "distribution": "weibull",
                    "interpretation": (
                        f"Test {int(n_required)} units for {t} time units (Weibull shape={beta}). "
                        f"If {c} or fewer failures occur, conclude R >= {R} with "
                        f"{request.confidence_level*100}% confidence."
                    )
                }

            else:
                # Binomial-based approach for general case
                # Find n such that P(X <= c | n, p=1-R) >= confidence
                for n in range(c + 1, 10000):
                    prob = scipy_stats.binom.cdf(c, n, 1 - R)
                    if prob >= request.confidence_level:
                        break

                results["demonstration_test"] = {
                    "sample_size": int(n),
                    "test_time": t,
                    "allowable_failures": c,
                    "target_reliability": R,
                    "confidence_level": request.confidence_level,
                    "distribution": "binomial",
                    "interpretation": (
                        f"Test {int(n)} units for {t} time units. "
                        f"If {c} or fewer failures occur, conclude R >= {R} with "
                        f"{request.confidence_level*100}% confidence."
                    )
                }

        elif request.test_type == "estimation":
            # Reliability Estimation Test
            # Find n to estimate reliability with desired precision

            # Using normal approximation for confidence interval width
            # SE = sqrt(R*(1-R)/n)
            # For 95% CI: width = 2 * 1.96 * SE

            z = scipy_stats.norm.ppf(1 - alpha / 2)

            # Target half-width of CI (use 10% of R as default precision)
            precision = 0.10 * R

            # n = (z^2 * R * (1-R)) / precision^2
            n_required = np.ceil((z ** 2 * R * (1 - R)) / (precision ** 2))

            results["estimation_test"] = {
                "sample_size": int(n_required),
                "target_reliability": R,
                "precision": precision,
                "confidence_level": request.confidence_level,
                "expected_ci_width": 2 * precision,
                "interpretation": (
                    f"Test {int(n_required)} units to estimate reliability with a "
                    f"{request.confidence_level*100}% CI width of approximately {2*precision:.3f}."
                )
            }

        elif request.test_type == "comparison":
            # Reliability Comparison Test
            # Find n to detect difference between two designs

            if request.comparison_reliability is None:
                raise HTTPException(
                    status_code=400,
                    detail="comparison_reliability is required for comparison tests"
                )

            R1 = R
            R2 = request.comparison_reliability

            z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
            z_beta = scipy_stats.norm.ppf(request.power)

            # Pooled estimate under null
            p_bar = (R1 + R2) / 2

            # Sample size per group (two-sample proportion test)
            numerator = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
                        z_beta * np.sqrt(R1 * (1 - R1) + R2 * (1 - R2))) ** 2
            denominator = (R1 - R2) ** 2

            n_per_group = np.ceil(numerator / denominator)

            results["comparison_test"] = {
                "sample_size_per_group": int(n_per_group),
                "total_sample_size": int(2 * n_per_group),
                "reliability_1": R1,
                "reliability_2": R2,
                "difference_to_detect": abs(R1 - R2),
                "power": request.power,
                "confidence_level": request.confidence_level,
                "interpretation": (
                    f"Test {int(n_per_group)} units per group ({int(2*n_per_group)} total) "
                    f"to detect a difference of {abs(R1-R2):.3f} in reliability "
                    f"with {request.power*100}% power at {request.confidence_level*100}% confidence."
                )
            }

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid test_type. Choose from: 'demonstration', 'estimation', 'comparison'"
            )

        # Operating characteristic curve data
        oc_curve = []
        if request.test_type == "demonstration":
            n = results.get("demonstration_test", {}).get("sample_size", 10)
            for true_R in np.linspace(0.5, 0.999, 50):
                # Probability of passing test if true reliability is true_R
                prob_pass = scipy_stats.binom.cdf(c, n, 1 - true_R)
                oc_curve.append({
                    "true_reliability": safe_float(true_R),
                    "probability_of_passing": safe_float(prob_pass)
                })
            results["operating_characteristic"] = oc_curve

        return make_json_safe({
            "test_type": request.test_type,
            "results": results,
            "parameters": {
                "target_reliability": R,
                "test_time": t,
                "confidence_level": request.confidence_level,
                "allowable_failures": c,
                "distribution": request.distribution
            },
            "message": "Reliability test planning completed successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test planning failed: {str(e)}")
