from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
from scipy import stats
from statsmodels.stats.power import ttest_power, FTestAnovaPower

router = APIRouter()

# Helper functions for effect sizes and diagnostics
def cohens_d(sample1, sample2=None, mu=0, paired=False):
    """Calculate Cohen's d effect size"""
    if sample2 is None:
        # One-sample
        mean_diff = np.mean(sample1) - mu
        sd = np.std(sample1, ddof=1)
        d = mean_diff / sd
    elif paired:
        # Paired samples
        diff = sample1 - sample2
        mean_diff = np.mean(diff)
        sd = np.std(diff, ddof=1)
        d = mean_diff / sd
    else:
        # Independent samples - pooled standard deviation
        n1, n2 = len(sample1), len(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        mean_diff = np.mean(sample1) - np.mean(sample2)
        d = mean_diff / pooled_std

    return float(d)

def hedges_g(sample1, sample2):
    """Calculate Hedges' g (bias-corrected Cohen's d)"""
    n1, n2 = len(sample1), len(sample2)
    d = cohens_d(sample1, sample2)
    # Bias correction factor
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    g = d * correction
    return float(g)

def interpret_effect_size(d):
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def shapiro_wilk_test(data, alpha=0.05):
    """Perform Shapiro-Wilk normality test"""
    if len(data) < 3:
        return None
    try:
        stat, p_value = stats.shapiro(data)
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "is_normal": bool(p_value > alpha),
            "test_name": "Shapiro-Wilk"
        }
    except:
        return None

def levene_test(sample1, sample2, alpha=0.05):
    """Perform Levene's test for equality of variances"""
    try:
        stat, p_value = stats.levene(sample1, sample2)
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "equal_variances": bool(p_value > alpha),
            "test_name": "Levene's Test"
        }
    except:
        return None

def qq_plot_data(data):
    """Generate Q-Q plot data for normality assessment"""
    try:
        # Use scipy's probplot for proper Q-Q plot
        # This returns (theoretical_quantiles, ordered_values), and (slope, intercept, r)
        (theoretical, observed), (slope, intercept, r) = stats.probplot(data, dist="norm", fit=True)
        return {
            "observed": [float(x) for x in observed],
            "theoretical": [float(x) for x in theoretical],
            "fit_line": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r**2)
            }
        }
    except:
        return None

def calculate_post_hoc_power(n1, n2, effect_size, alpha=0.05, alternative='two-sided'):
    """Calculate post-hoc statistical power"""
    try:
        if n2 is None:
            # One-sample test
            power = ttest_power(effect_size, n1, alpha, alternative=alternative)
        else:
            # Two-sample test
            ratio = n2 / n1 if n1 > 0 else 1
            power = ttest_power(effect_size, n1, alpha, ratio=ratio, alternative=alternative)
        return float(power)
    except:
        return None

def check_assumptions_and_recommend(normality_tests, variance_test=None):
    """Check assumptions and recommend appropriate test"""
    violations = []
    recommendations = []

    # Check normality
    normality_violated = False
    if normality_tests:
        for test_name, test_result in normality_tests.items():
            if test_result and not test_result.get("is_normal", True):
                normality_violated = True
                violations.append(f"{test_name} indicates non-normality (p={test_result['p_value']:.4f})")

    # Check variance equality
    variance_violated = False
    if variance_test and not variance_test.get("equal_variances", True):
        variance_violated = True
        violations.append(f"Levene's test indicates unequal variances (p={variance_test['p_value']:.4f})")

    # Make recommendations
    if normality_violated and variance_violated:
        recommendations.append("Consider using a non-parametric test (Mann-Whitney U or Wilcoxon)")
        recommendations.append("Or use Welch's t-test for unequal variances")
    elif normality_violated:
        recommendations.append("Consider using a non-parametric test due to non-normality")
    elif variance_violated:
        recommendations.append("Consider using Welch's t-test for unequal variances")
    else:
        recommendations.append("Assumptions met - t-test is appropriate")

    return {
        "violations": violations,
        "recommendations": recommendations,
        "all_assumptions_met": len(violations) == 0
    }

class TTestRequest(BaseModel):
    sample1: List[float] = Field(..., description="First sample data")
    sample2: Optional[List[float]] = Field(None, description="Second sample data (for two-sample test)")
    alternative: str = Field("two-sided", description="Alternative hypothesis: 'two-sided', 'less', or 'greater'")
    alpha: float = Field(0.05, description="Significance level")
    paired: bool = Field(False, description="Whether samples are paired")
    mu: float = Field(0.0, description="Hypothesized mean for one-sample test")

class FTestRequest(BaseModel):
    sample1: List[float] = Field(..., description="First sample data")
    sample2: List[float] = Field(..., description="Second sample data")
    alpha: float = Field(0.05, description="Significance level")

@router.post("/t-test")
async def t_test(request: TTestRequest):
    """
    Perform t-test for means comparison with comprehensive diagnostics
    - One-sample t-test if only sample1 provided
    - Two-sample t-test (independent or paired)
    Includes: effect sizes, assumptions tests, power analysis, and visualizations
    """
    try:
        sample1 = np.array(request.sample1)
        sample2 = np.array(request.sample2) if request.sample2 is not None else None

        # Perform appropriate t-test
        if sample2 is None:
            # One-sample t-test
            result = stats.ttest_1samp(sample1, request.mu, alternative=request.alternative)
            test_type = "One-sample t-test"
            df = len(sample1) - 1
        else:
            if request.paired:
                # Paired t-test
                result = stats.ttest_rel(sample1, sample2, alternative=request.alternative)
                test_type = "Paired t-test"
                df = len(sample1) - 1
            else:
                # Independent two-sample t-test
                result = stats.ttest_ind(sample1, sample2, alternative=request.alternative)
                test_type = "Independent two-sample t-test"
                df = len(sample1) + len(sample2) - 2

        t_statistic = float(result.statistic)
        p_value = float(result.pvalue)

        # Calculate confidence interval
        if sample2 is None:
            mean_diff = float(np.mean(sample1) - request.mu)
            se = stats.sem(sample1)
        elif request.paired:
            diff = sample1 - sample2
            mean_diff = float(np.mean(diff))
            se = stats.sem(diff)
        else:
            mean_diff = float(np.mean(sample1) - np.mean(sample2))
            n1, n2 = len(sample1), len(sample2)
            var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
            se = np.sqrt(var1/n1 + var2/n2)

        t_critical = stats.t.ppf(1 - request.alpha/2, df)
        ci_lower = mean_diff - t_critical * se
        ci_upper = mean_diff + t_critical * se

        # Calculate effect sizes
        if sample2 is None:
            effect_size = cohens_d(sample1, mu=request.mu)
            hedges_g_value = None
        elif request.paired:
            effect_size = cohens_d(sample1, sample2, paired=True)
            hedges_g_value = None
        else:
            effect_size = cohens_d(sample1, sample2)
            hedges_g_value = hedges_g(sample1, sample2)

        effect_size_interpretation = interpret_effect_size(effect_size)

        # Effect size confidence interval (approximate)
        effect_size_se = np.sqrt((len(sample1) + (len(sample2) if sample2 is not None else 0)) /
                                 (len(sample1) * (len(sample2) if sample2 is not None else len(sample1))) +
                                 (effect_size ** 2) / (2 * (len(sample1) + (len(sample2) if sample2 is not None else 0))))
        effect_ci_lower = effect_size - 1.96 * effect_size_se
        effect_ci_upper = effect_size + 1.96 * effect_size_se

        # Assumptions testing
        normality_tests = {}
        normality_tests["sample1"] = shapiro_wilk_test(sample1, request.alpha)
        if sample2 is not None and not request.paired:
            normality_tests["sample2"] = shapiro_wilk_test(sample2, request.alpha)

        variance_test = None
        if sample2 is not None and not request.paired:
            variance_test = levene_test(sample1, sample2, request.alpha)

        # Q-Q plot data
        qq_data = {}
        qq_data["sample1"] = qq_plot_data(sample1)
        if sample2 is not None:
            qq_data["sample2"] = qq_plot_data(sample2)

        # Check assumptions and get recommendations
        assumption_check = check_assumptions_and_recommend(normality_tests, variance_test)

        # Post-hoc power analysis
        n2_for_power = len(sample2) if sample2 is not None else None
        post_hoc_power = calculate_post_hoc_power(
            len(sample1),
            n2_for_power,
            abs(effect_size),
            request.alpha,
            request.alternative
        )

        # Distribution data for visualization
        x_range = np.linspace(-4, 4, 200)
        t_dist = stats.t.pdf(x_range, df)

        # Critical values for two-sided test
        if request.alternative == 'two-sided':
            critical_lower = -abs(t_critical)
            critical_upper = abs(t_critical)
        elif request.alternative == 'greater':
            critical_lower = None
            critical_upper = stats.t.ppf(1 - request.alpha, df)
        else:  # less
            critical_lower = stats.t.ppf(request.alpha, df)
            critical_upper = None

        distribution_plot_data = {
            "x": [float(x) for x in x_range],
            "y": [float(y) for y in t_dist],
            "test_statistic": t_statistic,
            "critical_lower": float(critical_lower) if critical_lower is not None else None,
            "critical_upper": float(critical_upper) if critical_upper is not None else None,
            "df": df
        }

        # Calculate box plot data
        def calculate_boxplot_data(data, label):
            q1 = float(np.percentile(data, 25))
            median = float(np.median(data))
            q3 = float(np.percentile(data, 75))
            iqr = q3 - q1
            lower_whisker = float(np.min(data[data >= q1 - 1.5 * iqr]))
            upper_whisker = float(np.max(data[data <= q3 + 1.5 * iqr]))
            outliers = [float(x) for x in data if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr]

            return {
                "label": label,
                "min": lower_whisker,
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": upper_whisker,
                "outliers": outliers
            }

        boxplot_data = [calculate_boxplot_data(sample1, "Sample 1")]
        if sample2 is not None:
            boxplot_data.append(calculate_boxplot_data(sample2, "Sample 2"))

        return {
            "test_type": test_type,
            "t_statistic": round(t_statistic, 4),
            "p_value": round(p_value, 6),
            "degrees_of_freedom": df,
            "alpha": request.alpha,
            "reject_null": bool(p_value < request.alpha),
            "mean_difference": round(mean_diff, 4),
            "confidence_interval": {
                "lower": round(ci_lower, 4),
                "upper": round(ci_upper, 4),
                "level": 1 - request.alpha
            },
            "sample_stats": {
                "sample1_mean": round(float(np.mean(sample1)), 4),
                "sample1_std": round(float(np.std(sample1, ddof=1)), 4),
                "sample1_n": len(sample1),
                "sample2_mean": round(float(np.mean(sample2)), 4) if sample2 is not None else None,
                "sample2_std": round(float(np.std(sample2, ddof=1)), 4) if sample2 is not None else None,
                "sample2_n": len(sample2) if sample2 is not None else None
            },
            "effect_size": {
                "cohens_d": round(effect_size, 4),
                "hedges_g": round(hedges_g_value, 4) if hedges_g_value is not None else None,
                "interpretation": effect_size_interpretation,
                "confidence_interval": {
                    "lower": round(effect_ci_lower, 4),
                    "upper": round(effect_ci_upper, 4)
                }
            },
            "assumptions": {
                "normality_tests": normality_tests,
                "variance_test": variance_test,
                "qq_plot_data": qq_data,
                "check": assumption_check
            },
            "power_analysis": {
                "post_hoc_power": round(post_hoc_power, 4) if post_hoc_power else None,
                "interpretation": "High" if post_hoc_power and post_hoc_power > 0.8 else
                                "Moderate" if post_hoc_power and post_hoc_power > 0.5 else "Low" if post_hoc_power else None
            },
            "distribution_plot_data": distribution_plot_data,
            "boxplot_data": boxplot_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/f-test")
async def f_test(request: FTestRequest):
    """
    Perform F-test for equality of variances
    """
    try:
        sample1 = np.array(request.sample1)
        sample2 = np.array(request.sample2)

        var1 = np.var(sample1, ddof=1)
        var2 = np.var(sample2, ddof=1)

        # F-statistic (larger variance in numerator)
        if var1 >= var2:
            f_stat = var1 / var2
            df1 = len(sample1) - 1
            df2 = len(sample2) - 1
        else:
            f_stat = var2 / var1
            df1 = len(sample2) - 1
            df2 = len(sample1) - 1

        # Two-tailed p-value
        p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

        # Critical values
        f_critical_lower = stats.f.ppf(request.alpha/2, df1, df2)
        f_critical_upper = stats.f.ppf(1 - request.alpha/2, df1, df2)

        # Calculate box plot data for F-test
        def calculate_boxplot_data(data, label):
            q1 = float(np.percentile(data, 25))
            median = float(np.median(data))
            q3 = float(np.percentile(data, 75))
            iqr = q3 - q1
            lower_whisker = float(np.min(data[data >= q1 - 1.5 * iqr]))
            upper_whisker = float(np.max(data[data <= q3 + 1.5 * iqr]))
            outliers = [float(x) for x in data if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr]

            return {
                "label": label,
                "min": lower_whisker,
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": upper_whisker,
                "outliers": outliers
            }

        boxplot_data = [
            calculate_boxplot_data(sample1, "Sample 1"),
            calculate_boxplot_data(sample2, "Sample 2")
        ]

        return {
            "test_type": "F-test for equality of variances",
            "f_statistic": round(float(f_stat), 4),
            "p_value": round(float(p_value), 6),
            "degrees_of_freedom": {"df1": df1, "df2": df2},
            "alpha": request.alpha,
            "reject_null": bool(p_value < request.alpha),
            "critical_values": {
                "lower": round(float(f_critical_lower), 4),
                "upper": round(float(f_critical_upper), 4)
            },
            "sample_stats": {
                "sample1_variance": round(float(var1), 4),
                "sample1_std": round(float(np.sqrt(var1)), 4),
                "sample1_n": len(sample1),
                "sample2_variance": round(float(var2), 4),
                "sample2_std": round(float(np.sqrt(var2)), 4),
                "sample2_n": len(sample2)
            },
            "boxplot_data": boxplot_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/z-test")
async def z_test(data: dict):
    """
    Perform Z-test for known population variance
    """
    try:
        sample = np.array(data['sample'])
        mu0 = data.get('mu0', 0)
        sigma = data['sigma']
        alternative = data.get('alternative', 'two-sided')
        alpha = data.get('alpha', 0.05)

        n = len(sample)
        sample_mean = np.mean(sample)

        # Z-statistic
        z_stat = (sample_mean - mu0) / (sigma / np.sqrt(n))

        # P-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(z_stat)
        else:  # less
            p_value = stats.norm.cdf(z_stat)

        z_critical = stats.norm.ppf(1 - alpha/2) if alternative == 'two-sided' else stats.norm.ppf(1 - alpha)

        # Calculate box plot data
        def calculate_boxplot_data(data, label):
            q1 = float(np.percentile(data, 25))
            median = float(np.median(data))
            q3 = float(np.percentile(data, 75))
            iqr = q3 - q1
            lower_whisker = float(np.min(data[data >= q1 - 1.5 * iqr]))
            upper_whisker = float(np.max(data[data <= q3 + 1.5 * iqr]))
            outliers = [float(x) for x in data if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr]

            return {
                "label": label,
                "min": lower_whisker,
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": upper_whisker,
                "outliers": outliers
            }

        boxplot_data = [calculate_boxplot_data(sample, "Sample")]

        return {
            "test_type": "Z-test",
            "z_statistic": round(float(z_stat), 4),
            "p_value": round(float(p_value), 6),
            "alpha": alpha,
            "reject_null": bool(p_value < alpha),
            "critical_value": round(float(z_critical), 4),
            "sample_mean": round(float(sample_mean), 4),
            "hypothesized_mean": mu0,
            "population_std": sigma,
            "sample_size": n,
            "boxplot_data": boxplot_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class NonParametricRequest(BaseModel):
    sample1: List[float] = Field(..., description="First sample data")
    sample2: List[float] = Field(..., description="Second sample data")
    alternative: str = Field("two-sided", description="Alternative hypothesis: 'two-sided', 'less', or 'greater'")
    alpha: float = Field(0.05, description="Significance level")

@router.post("/mann-whitney")
async def mann_whitney_test(request: NonParametricRequest):
    """
    Perform Mann-Whitney U test (non-parametric alternative to independent t-test)
    Tests whether two independent samples come from the same distribution
    """
    try:
        sample1 = np.array(request.sample1)
        sample2 = np.array(request.sample2)

        # Perform Mann-Whitney U test
        result = stats.mannwhitneyu(sample1, sample2, alternative=request.alternative)
        u_statistic = float(result.statistic)
        p_value = float(result.pvalue)

        # Calculate medians and ranks
        median1 = float(np.median(sample1))
        median2 = float(np.median(sample2))
        median_diff = median1 - median2

        # Rank-biserial correlation (effect size for Mann-Whitney)
        n1, n2 = len(sample1), len(sample2)
        rank_biserial = 1 - (2 * u_statistic) / (n1 * n2)

        # Interpret effect size
        abs_rb = abs(rank_biserial)
        if abs_rb < 0.1:
            effect_interpretation = "negligible"
        elif abs_rb < 0.3:
            effect_interpretation = "small"
        elif abs_rb < 0.5:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        # Calculate box plot data
        def calculate_boxplot_data(data, label):
            q1 = float(np.percentile(data, 25))
            median = float(np.median(data))
            q3 = float(np.percentile(data, 75))
            iqr = q3 - q1
            lower_whisker = float(np.min(data[data >= q1 - 1.5 * iqr]))
            upper_whisker = float(np.max(data[data <= q3 + 1.5 * iqr]))
            outliers = [float(x) for x in data if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr]

            return {
                "label": label,
                "min": lower_whisker,
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": upper_whisker,
                "outliers": outliers
            }

        boxplot_data = [
            calculate_boxplot_data(sample1, "Sample 1"),
            calculate_boxplot_data(sample2, "Sample 2")
        ]

        return {
            "test_type": "Mann-Whitney U Test",
            "u_statistic": round(u_statistic, 4),
            "p_value": round(p_value, 6),
            "alpha": request.alpha,
            "reject_null": bool(p_value < request.alpha),
            "median_difference": round(median_diff, 4),
            "sample_stats": {
                "sample1_median": round(median1, 4),
                "sample1_mean": round(float(np.mean(sample1)), 4),
                "sample1_n": n1,
                "sample2_median": round(median2, 4),
                "sample2_mean": round(float(np.mean(sample2)), 4),
                "sample2_n": n2
            },
            "effect_size": {
                "rank_biserial": round(rank_biserial, 4),
                "interpretation": effect_interpretation
            },
            "boxplot_data": boxplot_data,
            "interpretation": {
                "test_description": "Non-parametric test comparing distributions of two independent samples",
                "null_hypothesis": "The two samples come from the same distribution",
                "when_to_use": "When data violates normality assumptions for t-test"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/wilcoxon")
async def wilcoxon_test(request: NonParametricRequest):
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
    Tests whether two related samples come from the same distribution
    """
    try:
        sample1 = np.array(request.sample1)
        sample2 = np.array(request.sample2)

        if len(sample1) != len(sample2):
            raise HTTPException(status_code=400, detail="Samples must have equal length for paired test")

        # Perform Wilcoxon signed-rank test
        result = stats.wilcoxon(sample1, sample2, alternative=request.alternative)
        w_statistic = float(result.statistic)
        p_value = float(result.pvalue)

        # Calculate differences and median difference
        differences = sample1 - sample2
        median_diff = float(np.median(differences))
        mean_diff = float(np.mean(differences))

        # Rank-biserial correlation for paired samples
        n = len(sample1)
        rank_biserial = w_statistic / (n * (n + 1) / 4) - 1

        # Interpret effect size
        abs_rb = abs(rank_biserial)
        if abs_rb < 0.1:
            effect_interpretation = "negligible"
        elif abs_rb < 0.3:
            effect_interpretation = "small"
        elif abs_rb < 0.5:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        # Calculate box plot data
        def calculate_boxplot_data(data, label):
            q1 = float(np.percentile(data, 25))
            median = float(np.median(data))
            q3 = float(np.percentile(data, 75))
            iqr = q3 - q1
            lower_whisker = float(np.min(data[data >= q1 - 1.5 * iqr]))
            upper_whisker = float(np.max(data[data <= q3 + 1.5 * iqr]))
            outliers = [float(x) for x in data if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr]

            return {
                "label": label,
                "min": lower_whisker,
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": upper_whisker,
                "outliers": outliers
            }

        boxplot_data = [
            calculate_boxplot_data(sample1, "Sample 1"),
            calculate_boxplot_data(sample2, "Sample 2"),
            calculate_boxplot_data(differences, "Differences")
        ]

        return {
            "test_type": "Wilcoxon Signed-Rank Test",
            "w_statistic": round(w_statistic, 4),
            "p_value": round(p_value, 6),
            "alpha": request.alpha,
            "reject_null": bool(p_value < request.alpha),
            "median_difference": round(median_diff, 4),
            "mean_difference": round(mean_diff, 4),
            "sample_stats": {
                "sample1_median": round(float(np.median(sample1)), 4),
                "sample1_mean": round(float(np.mean(sample1)), 4),
                "sample2_median": round(float(np.median(sample2)), 4),
                "sample2_mean": round(float(np.mean(sample2)), 4),
                "n_pairs": n,
                "positive_diffs": int(np.sum(differences > 0)),
                "negative_diffs": int(np.sum(differences < 0)),
                "zero_diffs": int(np.sum(differences == 0))
            },
            "effect_size": {
                "rank_biserial": round(rank_biserial, 4),
                "interpretation": effect_interpretation
            },
            "boxplot_data": boxplot_data,
            "interpretation": {
                "test_description": "Non-parametric test comparing two related/paired samples",
                "null_hypothesis": "The median of differences between pairs is zero",
                "when_to_use": "When data violates normality assumptions for paired t-test"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
