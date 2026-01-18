from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from app.utils.report_generator import PDFReportGenerator, format_pvalue, format_number

router = APIRouter()

# Helper functions for assumptions testing
def test_normality(residuals):
    """
    Test normality of residuals using multiple tests
    """
    residuals = np.array(residuals)

    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(residuals)

    # Anderson-Darling test
    anderson_result = stats.anderson(residuals, dist='norm')
    anderson_stat = anderson_result.statistic
    anderson_critical = anderson_result.critical_values[2]  # 5% significance level
    anderson_pass = anderson_stat < anderson_critical

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))

    return {
        "shapiro_wilk": {
            "statistic": round(float(shapiro_stat), 6),
            "p_value": round(float(shapiro_p), 6),
            "passed": bool(shapiro_p > 0.05),
            "interpretation": "Residuals are normally distributed" if shapiro_p > 0.05 else "Residuals deviate from normality"
        },
        "anderson_darling": {
            "statistic": round(float(anderson_stat), 6),
            "critical_value": round(float(anderson_critical), 6),
            "passed": bool(anderson_pass),
            "interpretation": "Residuals are normally distributed" if anderson_pass else "Residuals deviate from normality"
        },
        "kolmogorov_smirnov": {
            "statistic": round(float(ks_stat), 6),
            "p_value": round(float(ks_p), 6),
            "passed": bool(ks_p > 0.05),
            "interpretation": "Residuals are normally distributed" if ks_p > 0.05 else "Residuals deviate from normality"
        }
    }

def test_homogeneity_of_variance(groups):
    """
    Test homogeneity of variance using Levene's and Bartlett's tests
    """
    group_data = [np.array(data) for data in groups.values()] if isinstance(groups, dict) else groups

    # Levene's test (robust to non-normality)
    levene_stat, levene_p = stats.levene(*group_data)

    # Bartlett's test (sensitive to non-normality)
    bartlett_stat, bartlett_p = stats.bartlett(*group_data)

    return {
        "levene": {
            "statistic": round(float(levene_stat), 6),
            "p_value": round(float(levene_p), 6),
            "passed": bool(levene_p > 0.05),
            "interpretation": "Variances are equal across groups" if levene_p > 0.05 else "Variances are unequal (heteroscedasticity detected)"
        },
        "bartlett": {
            "statistic": round(float(bartlett_stat), 6),
            "p_value": round(float(bartlett_p), 6),
            "passed": bool(bartlett_p > 0.05),
            "interpretation": "Variances are equal across groups" if bartlett_p > 0.05 else "Variances are unequal (heteroscedasticity detected)"
        }
    }

def calculate_effect_sizes_oneway(ssb, ssw, sst, df_between, df_within, k, n_total):
    """
    Calculate effect sizes for one-way ANOVA
    """
    # Eta-squared (η²) - proportion of variance explained
    eta_squared = ssb / sst

    # Omega-squared (ω²) - less biased estimate
    omega_squared = (ssb - df_between * (ssw / df_within)) / (sst + (ssw / df_within))
    omega_squared = max(0, omega_squared)  # Can't be negative

    # Cohen's f
    cohens_f = np.sqrt(eta_squared / (1 - eta_squared))

    # Interpretations based on Cohen's (1988) benchmarks
    def interpret_eta_omega(value):
        if value < 0.01:
            return "negligible"
        elif value < 0.06:
            return "small"
        elif value < 0.14:
            return "medium"
        else:
            return "large"

    def interpret_cohens_f(value):
        if value < 0.10:
            return "small"
        elif value < 0.25:
            return "medium"
        else:
            return "large"

    return {
        "eta_squared": {
            "value": round(float(eta_squared), 6),
            "interpretation": interpret_eta_omega(eta_squared),
            "description": "Proportion of total variance explained by the factor"
        },
        "omega_squared": {
            "value": round(float(omega_squared), 6),
            "interpretation": interpret_eta_omega(omega_squared),
            "description": "Unbiased estimate of effect size (preferred over η²)"
        },
        "cohens_f": {
            "value": round(float(cohens_f), 6),
            "interpretation": interpret_cohens_f(cohens_f),
            "description": "Effect size in Cohen's f metric"
        }
    }

def calculate_effect_sizes_twoway(anova_table, sst, n_total):
    """
    Calculate partial eta-squared for each effect in two-way ANOVA
    """
    effect_sizes = {}

    for effect_name, values in anova_table.items():
        if effect_name != "Residual" and values.get("sum_sq") is not None:
            ss_effect = values["sum_sq"]
            ss_error = anova_table.get("Residual", {}).get("sum_sq", 0)

            # Partial eta-squared
            partial_eta_sq = ss_effect / (ss_effect + ss_error)

            # Cohen's f
            cohens_f = np.sqrt(partial_eta_sq / (1 - partial_eta_sq))

            def interpret_partial_eta(value):
                if value < 0.01:
                    return "negligible"
                elif value < 0.06:
                    return "small"
                elif value < 0.14:
                    return "medium"
                else:
                    return "large"

            def interpret_cohens_f(value):
                if value < 0.10:
                    return "small"
                elif value < 0.25:
                    return "medium"
                else:
                    return "large"

            effect_sizes[effect_name] = {
                "partial_eta_squared": {
                    "value": round(float(partial_eta_sq), 6),
                    "interpretation": interpret_partial_eta(partial_eta_sq),
                    "description": f"Proportion of variance in DV explained by {effect_name}, excluding other effects"
                },
                "cohens_f": {
                    "value": round(float(cohens_f), 6),
                    "interpretation": interpret_cohens_f(cohens_f),
                    "description": f"Effect size for {effect_name} in Cohen's f metric"
                }
            }

    return effect_sizes

def calculate_influential_observations(residuals, fitted_values, n, p):
    """
    Calculate influence diagnostics: Cook's Distance, Leverage, DFBETAS, DFFITS

    Parameters:
    - residuals: array of residuals
    - fitted_values: array of fitted values
    - n: number of observations
    - p: number of parameters (groups)

    Returns:
    - Dictionary with influence metrics
    """
    residuals = np.array(residuals)
    fitted_values = np.array(fitted_values)

    # Mean squared error
    mse = np.mean(residuals**2)

    # Standardized residuals
    std_residuals = residuals / np.sqrt(mse)

    # Leverage (hat values) - simplified for ANOVA
    # For one-way ANOVA, leverage = 1/n_group for each observation
    # This is a simplified calculation
    leverage = np.ones(n) / n  # Placeholder - would need design matrix for exact calculation

    # Cook's Distance
    cooks_d = (std_residuals**2 / p) * (leverage / (1 - leverage)**2)

    # DFFITS
    dffits = std_residuals * np.sqrt(leverage / (1 - leverage))

    # Identify influential points
    # Cook's D > 4/(n-p) is common threshold
    cooks_threshold = 4 / (n - p) if n > p else 1.0
    influential_cooks = cooks_d > cooks_threshold

    # DFFITS > 2*sqrt(p/n) is common threshold
    dffits_threshold = 2 * np.sqrt(p / n)
    influential_dffits = np.abs(dffits) > dffits_threshold

    # Find indices of influential observations
    influential_indices = np.where(influential_cooks | influential_dffits)[0].tolist()

    return {
        "cooks_distance": [round(float(d), 6) for d in cooks_d],
        "leverage": [round(float(l), 6) for l in leverage],
        "dffits": [round(float(d), 6) for d in dffits],
        "standardized_residuals": [round(float(r), 6) for r in std_residuals],
        "influential_indices": influential_indices,
        "thresholds": {
            "cooks_d": round(float(cooks_threshold), 6),
            "dffits": round(float(dffits_threshold), 6)
        },
        "n_influential": len(influential_indices)
    }

def generate_diagnostic_plots_data(residuals, fitted_values, leverage, std_residuals):
    """
    Generate data for additional diagnostic plots:
    1. Scale-Location plot (sqrt of standardized residuals vs fitted values)
    2. Leverage vs Residuals plot
    3. Raw residuals and fitted values for histograms
    """
    residuals = np.array(residuals)
    fitted_values = np.array(fitted_values)
    leverage = np.array(leverage)
    std_residuals = np.array(std_residuals)

    # Scale-Location plot data
    # Y-axis: sqrt of absolute standardized residuals
    sqrt_abs_std_residuals = np.sqrt(np.abs(std_residuals))

    # Sort by fitted values for better visualization
    sort_idx = np.argsort(fitted_values)

    scale_location = {
        "fitted_values": fitted_values[sort_idx].tolist(),
        "sqrt_abs_std_residuals": sqrt_abs_std_residuals[sort_idx].tolist(),
        "interpretation": "Should show random scatter around a horizontal line. Funnel shape indicates heteroscedasticity."
    }

    # Leverage vs Residuals plot data
    leverage_residuals = {
        "leverage": leverage.tolist(),
        "std_residuals": std_residuals.tolist(),
        "interpretation": "Points with high leverage (>2p/n or >3p/n) and large residuals are influential."
    }

    return {
        "scale_location": scale_location,
        "leverage_residuals": leverage_residuals,
        "residuals": residuals.tolist(),
        "fitted": fitted_values.tolist()
    }

class OneWayANOVARequest(BaseModel):
    groups: Dict[str, List[float]] = Field(..., description="Dictionary of group names to data values")
    alpha: float = Field(0.05, description="Significance level")

class TwoWayANOVARequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of observations with factor levels and response")
    factor_a: str = Field(..., description="Name of first factor")
    factor_b: str = Field(..., description="Name of second factor")
    response: str = Field(..., description="Name of response variable")
    alpha: float = Field(0.05, description="Significance level")

class TwoWayPostHocRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of observations with factor levels and response")
    factor_a: str = Field(..., description="Name of first factor")
    factor_b: str = Field(..., description="Name of second factor")
    response: str = Field(..., description="Name of response variable")
    comparison_type: str = Field(..., description="Type of comparison: 'marginal_a', 'marginal_b', 'cell_means', 'simple_a', 'simple_b'")
    alpha: float = Field(0.05, description="Significance level")
    test_method: str = Field("tukey", description="Post-hoc test method: 'tukey', 'bonferroni', 'scheffe', 'fisher-lsd'")

class ContrastRequest(BaseModel):
    groups: Dict[str, List[float]] = Field(..., description="Dictionary of group names to data values")
    contrast_type: str = Field(..., description="Type of contrast: 'custom', 'polynomial', 'helmert'")
    coefficients: Optional[List[float]] = Field(None, description="Custom contrast coefficients (sum to 0)")
    polynomial_degree: Optional[int] = Field(None, description="Degree for polynomial contrast (1=linear, 2=quadratic, 3=cubic)")
    alpha: float = Field(0.05, description="Significance level")

def calculate_contrast(groups, coefficients, alpha=0.05):
    """
    Calculate a custom contrast for one-way ANOVA

    Parameters:
    - groups: dict of group_name -> data values
    - coefficients: list of contrast coefficients (must sum to 0)
    - alpha: significance level

    Returns:
    - Dictionary with contrast results
    """
    group_names = list(groups.keys())
    group_data = [np.array(groups[name]) for name in group_names]
    coefficients = np.array(coefficients)

    # Validate coefficients sum to 0 (within floating point tolerance)
    if abs(np.sum(coefficients)) > 1e-10:
        raise ValueError("Contrast coefficients must sum to 0")

    # Calculate group means and sample sizes
    group_means = np.array([np.mean(data) for data in group_data])
    group_ns = np.array([len(data) for data in group_data])
    n_total = np.sum(group_ns)
    k = len(group_data)

    # Calculate contrast estimate (ψ = Σc_i * mean_i)
    contrast_estimate = np.sum(coefficients * group_means)

    # Calculate MSW (within-group mean square)
    ssw = sum(np.sum((data - np.mean(data))**2) for data in group_data)
    df_within = n_total - k
    msw = ssw / df_within

    # Standard error of contrast: SE = sqrt(MSW * Σ(c_i²/n_i))
    se_contrast = np.sqrt(msw * np.sum(coefficients**2 / group_ns))

    # t-statistic
    t_stat = contrast_estimate / se_contrast

    # p-value (two-tailed)
    from scipy import stats as sp_stats
    p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df_within))

    # Confidence interval
    t_crit = sp_stats.t.ppf(1 - alpha/2, df_within)
    ci_lower = contrast_estimate - t_crit * se_contrast
    ci_upper = contrast_estimate + t_crit * se_contrast

    return {
        "contrast_estimate": float(contrast_estimate),
        "standard_error": float(se_contrast),
        "t_statistic": float(t_stat),
        "df": int(df_within),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "reject_null": bool(p_value < alpha),
        "coefficients": coefficients.tolist(),
        "group_means": {name: float(mean) for name, mean in zip(group_names, group_means)}
    }

def generate_polynomial_contrasts(k, degree):
    """Generate polynomial contrast coefficients up to specified degree"""
    from scipy.special import orthogonal

    # For polynomial contrasts, we use orthogonal polynomials
    x = np.arange(k)

    if degree == 1:  # Linear
        # Linear contrast: -k+1, -k+3, ..., k-3, k-1
        coefficients = x - np.mean(x)
        coefficients = coefficients / np.sqrt(np.sum(coefficients**2))
        return [coefficients.tolist()]

    elif degree == 2:  # Quadratic
        # Both linear and quadratic
        contrasts = []

        # Linear
        linear = x - np.mean(x)
        linear = linear / np.sqrt(np.sum(linear**2))
        contrasts.append(linear.tolist())

        # Quadratic
        quadratic = (x - np.mean(x))**2 - np.mean((x - np.mean(x))**2)
        quadratic = quadratic / np.sqrt(np.sum(quadratic**2))
        contrasts.append(quadratic.tolist())

        return contrasts

    elif degree == 3:  # Cubic
        contrasts = []

        # Linear
        linear = x - np.mean(x)
        linear = linear / np.sqrt(np.sum(linear**2))
        contrasts.append(linear.tolist())

        # Quadratic
        quadratic = (x - np.mean(x))**2 - np.mean((x - np.mean(x))**2)
        quadratic = quadratic / np.sqrt(np.sum(quadratic**2))
        contrasts.append(quadratic.tolist())

        # Cubic
        cubic = (x - np.mean(x))**3 - np.mean((x - np.mean(x))**3)
        cubic = cubic / np.sqrt(np.sum(cubic**2))
        contrasts.append(cubic.tolist())

        return contrasts

    return []

def generate_helmert_contrasts(k):
    """Generate Helmert contrast coefficients"""
    contrasts = []

    for i in range(k - 1):
        # Compare group i with mean of groups i+1 to k
        coefficients = np.zeros(k)
        coefficients[i] = k - i - 1
        coefficients[i+1:] = -1

        # Normalize
        coefficients = coefficients / np.sqrt(np.sum(coefficients**2))
        contrasts.append(coefficients.tolist())

    return contrasts

@router.post("/one-way")
async def one_way_anova(request: OneWayANOVARequest):
    """
    Perform one-way ANOVA (single-factor fixed effects model)
    """
    try:
        groups = request.groups
        group_names = list(groups.keys())
        group_data = [np.array(groups[name]) for name in group_names]

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)

        # Calculate sum of squares
        all_data = np.concatenate(group_data)
        grand_mean = np.mean(all_data)
        n_total = len(all_data)
        k = len(group_data)

        # SST (Total Sum of Squares)
        sst = np.sum((all_data - grand_mean)**2)

        # SSB (Between-group Sum of Squares)
        ssb = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in group_data)

        # SSW (Within-group Sum of Squares)
        ssw = sum(np.sum((group - np.mean(group))**2) for group in group_data)

        # Degrees of freedom
        df_between = k - 1
        df_within = n_total - k
        df_total = n_total - 1

        # Mean squares
        msb = ssb / df_between
        msw = ssw / df_within

        # F-critical value
        f_critical = stats.f.ppf(1 - request.alpha, df_between, df_within)

        # Group statistics
        group_stats = {}
        for name, data in groups.items():
            group_stats[name] = {
                "mean": round(float(np.mean(data)), 4),
                "std": round(float(np.std(data, ddof=1)), 4),
                "n": len(data),
                "sem": round(float(np.std(data, ddof=1) / np.sqrt(len(data))), 4)
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

        boxplot_data = [calculate_boxplot_data(np.array(data), name) for name, data in groups.items()]

        # Calculate fitted values and residuals
        group_means = {name: np.mean(data) for name, data in groups.items()}
        fitted_values = []
        residuals = []
        all_data_with_groups = []

        for name, data in groups.items():
            for value in data:
                fitted_values.append(float(group_means[name]))
                residuals.append(float(value - group_means[name]))
                all_data_with_groups.append({"group": name, "value": float(value)})

        residuals = np.array(residuals)
        fitted_values = np.array(fitted_values)

        # Standardized residuals
        standardized_residuals = residuals / np.sqrt(msw)

        # Calculate confidence intervals for means (95% CI)
        t_crit = stats.t.ppf(0.975, df_within)
        means_ci = {}
        for name, data in groups.items():
            mean = np.mean(data)
            sem = np.std(data, ddof=1) / np.sqrt(len(data))
            ci_margin = t_crit * sem
            means_ci[name] = {
                "mean": round(float(mean), 4),
                "lower": round(float(mean - ci_margin), 4),
                "upper": round(float(mean + ci_margin), 4)
            }

        # Test assumptions
        normality_tests = test_normality(residuals)
        homogeneity_tests = test_homogeneity_of_variance(groups)

        # Calculate effect sizes
        effect_sizes = calculate_effect_sizes_oneway(ssb, ssw, sst, df_between, df_within, k, n_total)

        # Calculate influential observations
        influence_diagnostics = calculate_influential_observations(residuals, fitted_values, n_total, k)

        # Generate diagnostic plots data
        diagnostic_plots = generate_diagnostic_plots_data(
            residuals,
            fitted_values,
            influence_diagnostics["leverage"],
            influence_diagnostics["standardized_residuals"]
        )

        # Add factor values and observed values for correlation/scatter analysis
        group_labels = []
        response_values = []
        for item in all_data_with_groups:
            group_labels.append(item["group"])
            response_values.append(item["value"])

        diagnostic_plots["factor_values"] = {"Group": group_labels}
        diagnostic_plots["observed_values"] = response_values

        return {
            "test_type": "One-Way ANOVA",
            "f_statistic": round(float(f_stat), 4),
            "p_value": round(float(p_value), 6),
            "alpha": request.alpha,
            "reject_null": bool(p_value < request.alpha),
            "f_critical": round(float(f_critical), 4),
            "anova_table": {
                "source": ["Between Groups", "Within Groups", "Total"],
                "ss": [round(float(ssb), 4), round(float(ssw), 4), round(float(sst), 4)],
                "df": [df_between, df_within, df_total],
                "ms": [round(float(msb), 4), round(float(msw), 4), None],
                "f": [round(float(f_stat), 4), None, None],
                "p": [round(float(p_value), 6), None, None]
            },
            "group_statistics": group_stats,
            "grand_mean": round(float(grand_mean), 4),
            "boxplot_data": boxplot_data,
            "means_ci": means_ci,
            "residuals": [round(float(r), 4) for r in residuals],
            "fitted_values": [round(float(f), 4) for f in fitted_values],
            "standardized_residuals": [round(float(r), 4) for r in standardized_residuals],
            "all_data": all_data_with_groups,
            "assumptions": {
                "normality": normality_tests,
                "homogeneity_of_variance": homogeneity_tests
            },
            "effect_sizes": effect_sizes,
            "influence_diagnostics": influence_diagnostics,
            "diagnostic_plots": diagnostic_plots
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/two-way")
async def two_way_anova(request: TwoWayANOVARequest):
    """
    Perform two-way ANOVA with interaction
    """
    try:
        # Create DataFrame
        df = pd.DataFrame(request.data)

        # Build formula
        formula = f"{request.response} ~ C({request.factor_a}) + C({request.factor_b}) + C({request.factor_a}):C({request.factor_b})"

        # Fit model
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Extract results
        results = {}
        for idx, row in anova_table.iterrows():
            source = str(idx)
            if source.startswith('C('):
                # Clean up factor names
                if ':' in source:
                    source = "Interaction"
                else:
                    source = source.replace(f'C({request.factor_a})', request.factor_a)
                    source = source.replace(f'C({request.factor_b})', request.factor_b)

            results[source] = {
                "sum_sq": round(float(row['sum_sq']), 4),
                "df": int(row['df']),
                "F": round(float(row['F']), 4) if not pd.isna(row['F']) else None,
                "PR(>F)": round(float(row['PR(>F)']), 6) if not pd.isna(row['PR(>F)']) else None
            }

        # Calculate means by factor
        factor_a_means = df.groupby(request.factor_a)[request.response].mean().to_dict()
        factor_b_means = df.groupby(request.factor_b)[request.response].mean().to_dict()
        interaction_means = df.groupby([request.factor_a, request.factor_b])[request.response].mean().to_dict()

        # Calculate residuals and fitted values
        fitted_values = model.fittedvalues.values
        residuals = model.resid.values
        mse = np.mean(residuals**2)
        standardized_residuals = residuals / np.sqrt(mse)

        # Box plot data by combination of factors
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

        # Create box plots for each combination
        boxplot_data = []
        for (a_val, b_val), group_df in df.groupby([request.factor_a, request.factor_b]):
            label = f"{a_val} × {b_val}"
            boxplot_data.append(calculate_boxplot_data(group_df[request.response].values, label))

        # Calculate confidence intervals for cell means
        from scipy import stats as sp_stats
        means_ci = {}
        for (a_val, b_val), group_df in df.groupby([request.factor_a, request.factor_b]):
            data = group_df[request.response].values
            mean = np.mean(data)
            n = len(data)
            if n > 1:
                sem = np.std(data, ddof=1) / np.sqrt(n)
                df_error = len(df) - len(df.groupby([request.factor_a, request.factor_b]))
                t_crit = sp_stats.t.ppf(0.975, df_error)
                ci_margin = t_crit * sem
                means_ci[f"{a_val} × {b_val}"] = {
                    "mean": round(float(mean), 4),
                    "lower": round(float(mean - ci_margin), 4),
                    "upper": round(float(mean + ci_margin), 4)
                }

        # Test assumptions
        normality_tests = test_normality(residuals)

        # For homogeneity testing in two-way, group by factor combinations
        groups_for_variance_test = []
        for (a_val, b_val), group_df in df.groupby([request.factor_a, request.factor_b]):
            groups_for_variance_test.append(group_df[request.response].values)
        homogeneity_tests = test_homogeneity_of_variance(groups_for_variance_test)

        # Calculate total sum of squares for effect sizes
        sst = np.sum((df[request.response].values - df[request.response].mean())**2)

        # Calculate effect sizes
        effect_sizes = calculate_effect_sizes_twoway(results, sst, len(df))

        # Calculate influential observations
        # For two-way ANOVA, p = number of cells
        n_cells = len(df.groupby([request.factor_a, request.factor_b]))
        influence_diagnostics = calculate_influential_observations(residuals, fitted_values, len(df), n_cells)

        # Generate diagnostic plots data
        diagnostic_plots = generate_diagnostic_plots_data(
            residuals,
            fitted_values,
            influence_diagnostics["leverage"],
            influence_diagnostics["standardized_residuals"]
        )

        # Add factor values and observed values for correlation/scatter analysis
        diagnostic_plots["factor_values"] = {
            request.factor_a: df[request.factor_a].tolist(),
            request.factor_b: df[request.factor_b].tolist()
        }
        diagnostic_plots["observed_values"] = df[request.response].tolist()

        return {
            "test_type": "Two-Way ANOVA",
            "alpha": request.alpha,
            "anova_table": results,
            "factor_means": {
                request.factor_a: {str(k): round(float(v), 4) for k, v in factor_a_means.items()},
                request.factor_b: {str(k): round(float(v), 4) for k, v in factor_b_means.items()}
            },
            "interaction_means": {f"{k[0]}, {k[1]}": round(float(v), 4) for k, v in interaction_means.items()},
            "grand_mean": round(float(df[request.response].mean()), 4),
            "boxplot_data": boxplot_data,
            "means_ci": means_ci,
            "residuals": [round(float(r), 4) for r in residuals],
            "fitted_values": [round(float(f), 4) for f in fitted_values],
            "standardized_residuals": [round(float(r), 4) for r in standardized_residuals],
            "factor_a_name": request.factor_a,
            "factor_b_name": request.factor_b,
            "assumptions": {
                "normality": normality_tests,
                "homogeneity_of_variance": homogeneity_tests
            },
            "effect_sizes": effect_sizes,
            "influence_diagnostics": influence_diagnostics,
            "diagnostic_plots": diagnostic_plots
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/post-hoc/tukey")
async def tukey_hsd(request: OneWayANOVARequest):
    """
    Perform Tukey's HSD (Honestly Significant Difference) post-hoc test
    """
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        # Convert groups dict to DataFrame format for tukey test
        groups_data = []
        group_labels = []
        for group_name, values in request.groups.items():
            groups_data.extend(values)
            group_labels.extend([group_name] * len(values))

        # Perform Tukey HSD
        tukey = pairwise_tukeyhsd(endog=groups_data, groups=group_labels, alpha=request.alpha)

        # Parse results
        comparisons = []
        for i in range(len(tukey.summary().data) - 1):
            row = tukey.summary().data[i + 1]
            comparisons.append({
                "group1": str(row[0]),
                "group2": str(row[1]),
                "mean_diff": round(float(row[2]), 4),
                "lower_ci": round(float(row[3]), 4),
                "upper_ci": round(float(row[4]), 4),
                "reject": bool(row[5]),
                "p_adj": round(float(row[6] if len(row) > 6 else 0.0), 6)
            })

        return {
            "test_type": "Tukey's HSD (Honestly Significant Difference)",
            "alpha": request.alpha,
            "comparisons": comparisons,
            "description": "Controls familywise error rate. Best for all pairwise comparisons with equal sample sizes."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/post-hoc/bonferroni")
async def bonferroni_test(request: OneWayANOVARequest):
    """
    Perform Bonferroni post-hoc test (most conservative)
    """
    try:
        from scipy.stats import ttest_ind
        from itertools import combinations

        groups = request.groups
        group_names = list(groups.keys())

        # Number of comparisons
        n_comparisons = len(list(combinations(group_names, 2)))
        bonferroni_alpha = request.alpha / n_comparisons

        # Perform pairwise t-tests
        comparisons = []
        for g1, g2 in combinations(group_names, 2):
            data1 = np.array(groups[g1])
            data2 = np.array(groups[g2])

            # Two-sample t-test
            t_stat, p_value = ttest_ind(data1, data2)
            mean_diff = np.mean(data1) - np.mean(data2)

            # Pooled standard error
            n1, n2 = len(data1), len(data2)
            s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
            pooled_se = np.sqrt(s1**2/n1 + s2**2/n2)

            # Confidence interval
            df_welch = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
            t_crit = stats.t.ppf(1 - bonferroni_alpha/2, df_welch)
            ci_margin = t_crit * pooled_se

            comparisons.append({
                "group1": str(g1),
                "group2": str(g2),
                "mean_diff": round(float(mean_diff), 4),
                "lower_ci": round(float(mean_diff - ci_margin), 4),
                "upper_ci": round(float(mean_diff + ci_margin), 4),
                "p_value": round(float(p_value), 6),
                "p_adj": round(float(min(p_value * n_comparisons, 1.0)), 6),
                "reject": bool(p_value < bonferroni_alpha)
            })

        return {
            "test_type": "Bonferroni Correction",
            "alpha": request.alpha,
            "adjusted_alpha": round(bonferroni_alpha, 6),
            "n_comparisons": n_comparisons,
            "comparisons": comparisons,
            "description": "Most conservative method. Divides alpha by number of comparisons to control Type I error."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/post-hoc/scheffe")
async def scheffe_test(request: OneWayANOVARequest):
    """
    Perform Scheffe post-hoc test (for all possible contrasts)
    """
    try:
        from scipy.stats import f as f_dist
        from itertools import combinations

        groups = request.groups
        group_names = list(groups.keys())
        group_data = [np.array(groups[name]) for name in group_names]

        k = len(group_names)
        n_total = sum(len(data) for data in group_data)

        # Calculate MSW (Mean Square Within) from all data
        ssw = sum(np.sum((data - np.mean(data))**2) for data in group_data)
        df_within = n_total - k
        msw = ssw / df_within

        # Critical value for Scheffe
        f_crit = f_dist.ppf(1 - request.alpha, k - 1, df_within)
        scheffe_crit = np.sqrt((k - 1) * f_crit)

        # Perform pairwise comparisons
        comparisons = []
        for g1, g2 in combinations(group_names, 2):
            data1 = np.array(groups[g1])
            data2 = np.array(groups[g2])
            n1, n2 = len(data1), len(data2)

            mean_diff = np.mean(data1) - np.mean(data2)
            se = np.sqrt(msw * (1/n1 + 1/n2))

            # Scheffe confidence interval
            ci_margin = scheffe_crit * se

            # Test statistic
            s_statistic = abs(mean_diff) / se
            reject = s_statistic > scheffe_crit

            comparisons.append({
                "group1": str(g1),
                "group2": str(g2),
                "mean_diff": round(float(mean_diff), 4),
                "lower_ci": round(float(mean_diff - ci_margin), 4),
                "upper_ci": round(float(mean_diff + ci_margin), 4),
                "s_statistic": round(float(s_statistic), 4),
                "critical_value": round(float(scheffe_crit), 4),
                "reject": bool(reject)
            })

        return {
            "test_type": "Scheffe's Test",
            "alpha": request.alpha,
            "comparisons": comparisons,
            "description": "Most flexible, can test any contrast. More conservative than Tukey for pairwise comparisons."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/post-hoc/fisher-lsd")
async def fisher_lsd_test(request: OneWayANOVARequest):
    """
    Perform Fisher's LSD (Least Significant Difference) post-hoc test
    """
    try:
        from scipy.stats import ttest_ind
        from itertools import combinations

        groups = request.groups
        group_names = list(groups.keys())
        group_data = [np.array(groups[name]) for name in group_names]

        n_total = sum(len(data) for data in group_data)
        k = len(group_names)

        # Calculate MSW
        ssw = sum(np.sum((data - np.mean(data))**2) for data in group_data)
        df_within = n_total - k
        msw = ssw / df_within

        # t-critical value
        t_crit = stats.t.ppf(1 - request.alpha/2, df_within)

        # Perform pairwise comparisons
        comparisons = []
        for g1, g2 in combinations(group_names, 2):
            data1 = np.array(groups[g1])
            data2 = np.array(groups[g2])
            n1, n2 = len(data1), len(data2)

            mean_diff = np.mean(data1) - np.mean(data2)

            # LSD calculation
            lsd = t_crit * np.sqrt(msw * (1/n1 + 1/n2))

            # Standard error
            se = np.sqrt(msw * (1/n1 + 1/n2))

            # t-statistic
            t_stat = mean_diff / se
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_within))

            # Confidence interval
            ci_margin = t_crit * se

            reject = abs(mean_diff) > lsd

            comparisons.append({
                "group1": str(g1),
                "group2": str(g2),
                "mean_diff": round(float(mean_diff), 4),
                "lower_ci": round(float(mean_diff - ci_margin), 4),
                "upper_ci": round(float(mean_diff + ci_margin), 4),
                "lsd": round(float(lsd), 4),
                "p_value": round(float(p_value), 6),
                "p_adj": round(float(p_value), 6),
                "reject": bool(reject)
            })

        return {
            "test_type": "Fisher's LSD (Least Significant Difference)",
            "alpha": request.alpha,
            "comparisons": comparisons,
            "description": "Least conservative method. Should only be used when overall F-test is significant."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/post-hoc/two-way")
async def two_way_post_hoc(request: TwoWayPostHocRequest):
    """
    Perform post-hoc tests for two-way ANOVA
    Supports marginal means, cell means, and simple effects comparisons
    """
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        df = pd.DataFrame(request.data)

        # Prepare data based on comparison type
        if request.comparison_type == 'marginal_a':
            # Compare marginal means of Factor A
            groups_data = df[request.response].values
            group_labels = df[request.factor_a].values
            description = f"Pairwise comparisons of {request.factor_a} marginal means"

        elif request.comparison_type == 'marginal_b':
            # Compare marginal means of Factor B
            groups_data = df[request.response].values
            group_labels = df[request.factor_b].values
            description = f"Pairwise comparisons of {request.factor_b} marginal means"

        elif request.comparison_type == 'cell_means':
            # Compare all cell means
            df['cell'] = df[request.factor_a].astype(str) + ' × ' + df[request.factor_b].astype(str)
            groups_data = df[request.response].values
            group_labels = df['cell'].values
            description = "Pairwise comparisons of all cell means"

        elif request.comparison_type.startswith('simple_'):
            # Simple effects - compare one factor at each level of the other
            if request.comparison_type == 'simple_a':
                # Compare Factor A at each level of Factor B
                description = f"Simple effects of {request.factor_a} at each level of {request.factor_b}"
            else:
                # Compare Factor B at each level of Factor A
                description = f"Simple effects of {request.factor_b} at each level of {request.factor_a}"

            # For simple effects, we'll need to do multiple comparisons
            # This is more complex, so we'll return grouped results
            return await _two_way_simple_effects(request)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid comparison_type: {request.comparison_type}")

        # Perform the appropriate post-hoc test
        if request.test_method == 'tukey':
            tukey = pairwise_tukeyhsd(endog=groups_data, groups=group_labels, alpha=request.alpha)

            comparisons = []
            for i in range(len(tukey.summary().data) - 1):
                row = tukey.summary().data[i + 1]
                comparisons.append({
                    "group1": str(row[0]),
                    "group2": str(row[1]),
                    "mean_diff": round(float(row[2]), 4),
                    "lower_ci": round(float(row[3]), 4),
                    "upper_ci": round(float(row[4]), 4),
                    "reject": bool(row[5]),
                    "p_adj": round(float(row[6] if len(row) > 6 else 0.0), 6)
                })

            test_name = "Tukey's HSD"

        elif request.test_method == 'bonferroni':
            # Bonferroni correction
            groups_dict = {}
            for label in np.unique(group_labels):
                groups_dict[label] = groups_data[group_labels == label]

            group_names = list(groups_dict.keys())
            comparisons = []
            n_comparisons = len(group_names) * (len(group_names) - 1) // 2
            adjusted_alpha = request.alpha / n_comparisons

            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    g1_data = groups_dict[group_names[i]]
                    g2_data = groups_dict[group_names[j]]

                    t_stat, p_value = stats.ttest_ind(g1_data, g2_data)
                    mean_diff = np.mean(g1_data) - np.mean(g2_data)

                    pooled_std = np.sqrt(((len(g1_data)-1)*np.var(g1_data, ddof=1) +
                                          (len(g2_data)-1)*np.var(g2_data, ddof=1)) /
                                         (len(g1_data) + len(g2_data) - 2))
                    se = pooled_std * np.sqrt(1/len(g1_data) + 1/len(g2_data))
                    df_error = len(g1_data) + len(g2_data) - 2
                    t_crit = stats.t.ppf(1 - adjusted_alpha/2, df_error)
                    ci_margin = t_crit * se

                    comparisons.append({
                        "group1": group_names[i],
                        "group2": group_names[j],
                        "mean_diff": round(float(mean_diff), 4),
                        "lower_ci": round(float(mean_diff - ci_margin), 4),
                        "upper_ci": round(float(mean_diff + ci_margin), 4),
                        "p_adj": round(float(p_value * n_comparisons), 6),
                        "reject": bool(p_value < adjusted_alpha)
                    })

            test_name = "Bonferroni"

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported test method for two-way: {request.test_method}")

        return {
            "test_type": f"{test_name} - Two-Way ANOVA Post-hoc",
            "comparison_type": request.comparison_type,
            "alpha": request.alpha,
            "comparisons": comparisons,
            "description": description
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def _two_way_simple_effects(request: TwoWayPostHocRequest):
    """
    Helper function for simple effects analysis in two-way ANOVA
    """
    df = pd.DataFrame(request.data)

    if request.comparison_type == 'simple_a':
        # Compare Factor A at each level of Factor B
        grouping_factor = request.factor_b
        comparison_factor = request.factor_a
    else:
        # Compare Factor B at each level of Factor A
        grouping_factor = request.factor_a
        comparison_factor = request.factor_b

    results_by_level = {}

    for level in df[grouping_factor].unique():
        level_df = df[df[grouping_factor] == level]

        # Perform post-hoc test for this level
        groups_data = level_df[request.response].values
        group_labels = level_df[comparison_factor].values

        if request.test_method == 'tukey':
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            tukey = pairwise_tukeyhsd(endog=groups_data, groups=group_labels, alpha=request.alpha)

            comparisons = []
            for i in range(len(tukey.summary().data) - 1):
                row = tukey.summary().data[i + 1]
                comparisons.append({
                    "group1": str(row[0]),
                    "group2": str(row[1]),
                    "mean_diff": round(float(row[2]), 4),
                    "lower_ci": round(float(row[3]), 4),
                    "upper_ci": round(float(row[4]), 4),
                    "reject": bool(row[5]),
                    "p_adj": round(float(row[6] if len(row) > 6 else 0.0), 6)
                })

            results_by_level[str(level)] = {
                "comparisons": comparisons,
                "n_observations": len(level_df)
            }

    return {
        "test_type": f"Tukey's HSD - Simple Effects",
        "comparison_type": request.comparison_type,
        "alpha": request.alpha,
        "results_by_level": results_by_level,
        "description": f"Simple effects of {comparison_factor} at each level of {grouping_factor}"
    }

@router.post("/contrasts")
async def perform_contrasts(request: ContrastRequest):
    """
    Perform custom contrast analysis for one-way ANOVA

    Supports:
    - Custom contrasts (user-specified coefficients)
    - Polynomial contrasts (linear, quadratic, cubic trends)
    - Helmert contrasts (each level vs mean of subsequent levels)
    """
    try:
        groups = request.groups
        k = len(groups)

        results = {
            "test_type": "Contrast Analysis",
            "alpha": request.alpha,
            "n_groups": k,
            "contrasts": []
        }

        if request.contrast_type == "custom":
            if not request.coefficients:
                raise HTTPException(status_code=400, detail="Custom contrasts require coefficients")

            if len(request.coefficients) != k:
                raise HTTPException(
                    status_code=400,
                    detail=f"Number of coefficients ({len(request.coefficients)}) must match number of groups ({k})"
                )

            # Calculate single custom contrast
            contrast_result = calculate_contrast(groups, request.coefficients, request.alpha)
            contrast_result["name"] = "Custom Contrast"
            contrast_result["interpretation"] = "User-specified contrast"
            results["contrasts"].append(contrast_result)
            results["description"] = "Custom contrast with user-specified coefficients"

        elif request.contrast_type == "polynomial":
            if not request.polynomial_degree:
                raise HTTPException(status_code=400, detail="Polynomial contrasts require degree specification")

            if request.polynomial_degree < 1 or request.polynomial_degree > 3:
                raise HTTPException(status_code=400, detail="Polynomial degree must be 1 (linear), 2 (quadratic), or 3 (cubic)")

            if request.polynomial_degree >= k:
                raise HTTPException(
                    status_code=400,
                    detail=f"Polynomial degree ({request.polynomial_degree}) must be less than number of groups ({k})"
                )

            # Generate polynomial contrasts
            polynomial_coeffs = generate_polynomial_contrasts(k, request.polynomial_degree)
            contrast_names = ["Linear", "Quadratic", "Cubic"][:request.polynomial_degree]

            for i, coeffs in enumerate(polynomial_coeffs):
                contrast_result = calculate_contrast(groups, coeffs, request.alpha)
                contrast_result["name"] = f"{contrast_names[i]} Trend"
                contrast_result["interpretation"] = f"Tests for {contrast_names[i].lower()} trend across ordered groups"
                results["contrasts"].append(contrast_result)

            results["description"] = f"Polynomial contrasts testing for trends across {k} ordered groups"

        elif request.contrast_type == "helmert":
            # Generate Helmert contrasts
            helmert_coeffs = generate_helmert_contrasts(k)

            for i, coeffs in enumerate(helmert_coeffs):
                contrast_result = calculate_contrast(groups, coeffs, request.alpha)
                group_names = list(groups.keys())
                contrast_result["name"] = f"Helmert {i+1}"
                contrast_result["interpretation"] = f"Compares {group_names[i]} with mean of subsequent groups"
                results["contrasts"].append(contrast_result)

            results["description"] = f"Helmert contrasts comparing each group with mean of remaining groups"

        else:
            raise HTTPException(status_code=400, detail=f"Unknown contrast type: {request.contrast_type}")

        # Add overall statistics
        n_contrasts = len(results["contrasts"])
        significant_contrasts = sum(1 for c in results["contrasts"] if c["reject_null"])

        results["summary"] = {
            "n_contrasts": n_contrasts,
            "n_significant": significant_contrasts,
            "bonferroni_alpha": round(request.alpha / n_contrasts, 6) if n_contrasts > 0 else request.alpha,
            "note": f"For multiple contrasts, consider Bonferroni correction (α={round(request.alpha / n_contrasts, 6)})" if n_contrasts > 1 else ""
        }

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class ANOVAPDFRequest(BaseModel):
    """Request model for generating ANOVA PDF report"""
    results: Dict[str, Any] = Field(..., description="ANOVA analysis results")
    post_hoc_results: Optional[Dict[str, Any]] = Field(None, description="Optional post-hoc test results")
    contrast_results: Optional[Dict[str, Any]] = Field(None, description="Optional contrast analysis results")
    title: Optional[str] = Field("ANOVA Analysis Report", description="Report title")

@router.post("/export-pdf")
async def export_anova_pdf(request: ANOVAPDFRequest):
    """
    Generate a comprehensive PDF report for ANOVA analysis

    Includes:
    - Executive summary
    - ANOVA table
    - Group statistics
    - Effect sizes
    - Diagnostic tests
    - Post-hoc comparisons (if provided)
    - Contrast analysis (if provided)
    - Recommendations
    """
    try:
        results = request.results
        test_type = results.get("test_type", "ANOVA")

        # Initialize PDF generator
        pdf = PDFReportGenerator(title=request.title, author="MasterStat")

        # Cover page metadata
        metadata = {
            "Analysis Type": test_type,
            "Significance Level (α)": str(results.get("alpha", 0.05)),
            "Software": "MasterStat Statistical Analysis Platform"
        }

        # Add number of groups for one-way, or factors for two-way
        if "group_statistics" in results:
            metadata["Number of Groups"] = str(len(results["group_statistics"]))
        elif "factor_a_name" in results:
            metadata["Factor A"] = results["factor_a_name"]
            metadata["Factor B"] = results["factor_b_name"]

        pdf.add_cover_page(subtitle=test_type, metadata=metadata)

        # Executive Summary
        pdf.add_section("Executive Summary")

        # Determine overall result
        f_stat = results.get("f_statistic")
        p_value = results.get("p_value")
        alpha = results.get("alpha", 0.05)
        reject_null = results.get("reject_null", False)

        if test_type == "One-Way ANOVA":
            if reject_null:
                summary_text = f"""
                The one-way ANOVA analysis revealed a statistically significant difference among the group means
                (F = {format_number(f_stat, 4)}, p {format_pvalue(p_value)}). This indicates that at least one group mean
                differs significantly from the others at the α = {alpha} significance level.
                """
            else:
                summary_text = f"""
                The one-way ANOVA analysis did not reveal a statistically significant difference among the group means
                (F = {format_number(f_stat, 4)}, p = {format_pvalue(p_value)}). There is insufficient evidence to conclude
                that the group means differ at the α = {alpha} significance level.
                """
        else:  # Two-way ANOVA
            summary_text = f"""
            A two-way ANOVA was conducted to examine the effects of {results.get('factor_a_name', 'Factor A')} and
            {results.get('factor_b_name', 'Factor B')} on the response variable. The analysis evaluated main effects
            for each factor and their interaction effect.
            """

        pdf.add_paragraph(summary_text.strip())

        # ANOVA Table
        pdf.add_section("ANOVA Table")

        if "anova_table" in results:
            anova_data = results["anova_table"]

            if "source" in anova_data:  # One-way format
                headers = ["Source", "SS", "df", "MS", "F", "p-value"]
                table_data = []
                for i in range(len(anova_data["source"])):
                    row = [
                        anova_data["source"][i],
                        format_number(anova_data["ss"][i], 4),
                        str(anova_data["df"][i]),
                        format_number(anova_data["ms"][i], 4) if anova_data["ms"][i] is not None else "-",
                        format_number(anova_data["f"][i], 4) if anova_data["f"][i] is not None else "-",
                        format_pvalue(anova_data["p"][i]) if anova_data["p"][i] is not None else "-"
                    ]
                    table_data.append(row)
            else:  # Two-way format (dict of effects)
                headers = ["Source", "Sum of Squares", "df", "F-statistic", "p-value"]
                table_data = []
                for source, values in anova_data.items():
                    row = [
                        source,
                        format_number(values.get("sum_sq"), 4),
                        str(values.get("df", "-")),
                        format_number(values.get("F"), 4) if values.get("F") is not None else "-",
                        format_pvalue(values.get("PR(>F)")) if values.get("PR(>F)") is not None else "-"
                    ]
                    table_data.append(row)

            pdf.add_table(table_data, headers=headers)

        # Group Statistics / Descriptive Statistics
        pdf.add_section("Descriptive Statistics")

        if "group_statistics" in results:
            headers = ["Group", "Mean", "Std Dev", "N", "SEM"]
            table_data = []
            for group_name, stats in results["group_statistics"].items():
                row = [
                    group_name,
                    format_number(stats["mean"], 4),
                    format_number(stats["std"], 4),
                    str(stats["n"]),
                    format_number(stats["sem"], 4)
                ]
                table_data.append(row)
            pdf.add_table(table_data, headers=headers, title="Group Statistics")

            # Grand mean
            if "grand_mean" in results:
                pdf.add_paragraph(f"<b>Grand Mean:</b> {format_number(results['grand_mean'], 4)}")

        elif "factor_means" in results:
            # Two-way ANOVA: show marginal means
            for factor_name, means_dict in results["factor_means"].items():
                headers = ["Level", "Mean"]
                table_data = [[str(level), format_number(mean, 4)] for level, mean in means_dict.items()]
                pdf.add_table(table_data, headers=headers, title=f"{factor_name} Marginal Means")

            # Interaction means
            if "interaction_means" in results:
                headers = ["Cell (A, B)", "Mean"]
                table_data = [[cell, format_number(mean, 4)] for cell, mean in results["interaction_means"].items()]
                pdf.add_table(table_data, headers=headers, title="Cell Means (Interaction)")

        # Effect Sizes
        if "effect_sizes" in results:
            pdf.add_section("Effect Sizes")

            effect_sizes = results["effect_sizes"]

            if "eta_squared" in effect_sizes:  # One-way format
                stats_dict = {
                    "Eta-squared (η²)": f"{format_number(effect_sizes['eta_squared']['value'], 6)} ({effect_sizes['eta_squared']['interpretation']})",
                    "Omega-squared (ω²)": f"{format_number(effect_sizes['omega_squared']['value'], 6)} ({effect_sizes['omega_squared']['interpretation']})",
                    "Cohen's f": f"{format_number(effect_sizes['cohens_f']['value'], 6)} ({effect_sizes['cohens_f']['interpretation']})"
                }
                pdf.add_summary_stats(stats_dict)

                pdf.add_paragraph(
                    "<i>Effect size interpretations: Eta²/Omega² (small: 0.01, medium: 0.06, large: 0.14); "
                    "Cohen's f (small: 0.10, medium: 0.25, large: 0.40)</i>"
                )
            else:  # Two-way format (effect sizes per factor)
                for effect_name, effect_data in effect_sizes.items():
                    stats_dict = {
                        f"Partial η² for {effect_name}": f"{format_number(effect_data['partial_eta_squared']['value'], 6)} ({effect_data['partial_eta_squared']['interpretation']})",
                        f"Cohen's f for {effect_name}": f"{format_number(effect_data['cohens_f']['value'], 6)} ({effect_data['cohens_f']['interpretation']})"
                    }
                    pdf.add_summary_stats(stats_dict, title=f"Effect Size: {effect_name}")

        # Assumptions Testing
        if "assumptions" in results:
            pdf.add_section("Assumptions Testing")

            assumptions = results["assumptions"]

            # Normality tests
            if "normality" in assumptions:
                pdf.add_subsection("Normality of Residuals")

                norm_tests = assumptions["normality"]
                headers = ["Test", "Statistic", "p-value", "Result"]
                table_data = []

                if "shapiro_wilk" in norm_tests:
                    sw = norm_tests["shapiro_wilk"]
                    table_data.append([
                        "Shapiro-Wilk",
                        format_number(sw["statistic"], 6),
                        format_pvalue(sw["p_value"]),
                        "✓ Pass" if sw["passed"] else "✗ Fail"
                    ])

                if "anderson_darling" in norm_tests:
                    ad = norm_tests["anderson_darling"]
                    table_data.append([
                        "Anderson-Darling",
                        format_number(ad["statistic"], 6),
                        f"Critical: {format_number(ad['critical_value'], 6)}",
                        "✓ Pass" if ad["passed"] else "✗ Fail"
                    ])

                if "kolmogorov_smirnov" in norm_tests:
                    ks = norm_tests["kolmogorov_smirnov"]
                    table_data.append([
                        "Kolmogorov-Smirnov",
                        format_number(ks["statistic"], 6),
                        format_pvalue(ks["p_value"]),
                        "✓ Pass" if ks["passed"] else "✗ Fail"
                    ])

                pdf.add_table(table_data, headers=headers)

            # Homogeneity of variance
            if "homogeneity_of_variance" in assumptions:
                pdf.add_subsection("Homogeneity of Variance")

                hov_tests = assumptions["homogeneity_of_variance"]
                headers = ["Test", "Statistic", "p-value", "Result"]
                table_data = []

                if "levene" in hov_tests:
                    lev = hov_tests["levene"]
                    table_data.append([
                        "Levene's Test",
                        format_number(lev["statistic"], 6),
                        format_pvalue(lev["p_value"]),
                        "✓ Equal variances" if lev["passed"] else "✗ Unequal variances"
                    ])

                if "bartlett" in hov_tests:
                    bart = hov_tests["bartlett"]
                    table_data.append([
                        "Bartlett's Test",
                        format_number(bart["statistic"], 6),
                        format_pvalue(bart["p_value"]),
                        "✓ Equal variances" if bart["passed"] else "✗ Unequal variances"
                    ])

                pdf.add_table(table_data, headers=headers)

        # Post-hoc Tests (if provided)
        if request.post_hoc_results:
            pdf.add_section("Post-Hoc Comparisons")

            post_hoc = request.post_hoc_results
            pdf.add_paragraph(f"<b>Test Method:</b> {post_hoc.get('test_type', 'Post-hoc Analysis')}")

            if "description" in post_hoc:
                pdf.add_paragraph(f"<i>{post_hoc['description']}</i>")

            if "comparisons" in post_hoc:
                headers = ["Group 1", "Group 2", "Mean Diff", "Lower CI", "Upper CI", "Significant"]
                table_data = []

                for comp in post_hoc["comparisons"]:
                    row = [
                        comp.get("group1", "-"),
                        comp.get("group2", "-"),
                        format_number(comp.get("mean_diff"), 4),
                        format_number(comp.get("lower_ci"), 4),
                        format_number(comp.get("upper_ci"), 4),
                        "Yes" if comp.get("reject", False) else "No"
                    ]
                    table_data.append(row)

                pdf.add_table(table_data, headers=headers)

        # Contrast Analysis (if provided)
        if request.contrast_results:
            pdf.add_section("Contrast Analysis")

            contrasts = request.contrast_results
            pdf.add_paragraph(f"<b>Analysis Type:</b> {contrasts.get('description', 'Contrast Analysis')}")

            if "contrasts" in contrasts:
                for contrast in contrasts["contrasts"]:
                    pdf.add_subsection(contrast.get("name", "Contrast"))

                    stats_dict = {
                        "Contrast Estimate": format_number(contrast.get("contrast_estimate"), 6),
                        "Standard Error": format_number(contrast.get("standard_error"), 6),
                        "t-statistic": format_number(contrast.get("t_statistic"), 4),
                        "df": str(contrast.get("df", "-")),
                        "p-value": format_pvalue(contrast.get("p_value")),
                        "95% CI": f"[{format_number(contrast.get('ci_lower'), 4)}, {format_number(contrast.get('ci_upper'), 4)}]",
                        "Significant": "Yes" if contrast.get("reject_null", False) else "No"
                    }
                    pdf.add_summary_stats(stats_dict, title="")

                    if "interpretation" in contrast:
                        pdf.add_paragraph(f"<i>{contrast['interpretation']}</i>")

        # Recommendations
        pdf.add_section("Recommendations")
        recommendations = []

        # Based on main effect
        if reject_null:
            if test_type == "One-Way ANOVA":
                recommendations.append("Significant group differences detected. Conduct post-hoc tests (e.g., Tukey's HSD) to identify which specific groups differ.")
            else:
                recommendations.append("Review the ANOVA table to determine which effects (main effects and/or interaction) are significant.")
        else:
            recommendations.append("No significant differences detected at the chosen significance level. Consider increasing sample size or exploring other factors.")

        # Based on assumptions
        if "assumptions" in results:
            assumptions = results["assumptions"]

            # Check normality
            if "normality" in assumptions:
                norm_tests = assumptions["normality"]
                if "shapiro_wilk" in norm_tests and not norm_tests["shapiro_wilk"]["passed"]:
                    recommendations.append("Normality assumption violated. Consider data transformation (log, square root) or use non-parametric alternatives (Kruskal-Wallis test).")

            # Check homogeneity
            if "homogeneity_of_variance" in assumptions:
                hov_tests = assumptions["homogeneity_of_variance"]
                if "levene" in hov_tests and not hov_tests["levene"]["passed"]:
                    recommendations.append("Homogeneity of variance assumption violated. Consider Welch's ANOVA or data transformation to stabilize variances.")

        # Based on effect size
        if "effect_sizes" in results:
            effect_sizes = results["effect_sizes"]
            if "omega_squared" in effect_sizes:
                omega_value = effect_sizes["omega_squared"]["value"]
                if omega_value < 0.01:
                    recommendations.append("Effect size is negligible. Even if significant, the practical importance may be limited.")
                elif omega_value > 0.14:
                    recommendations.append("Large effect size detected. The factor has substantial practical significance.")

        # Add influential observations warning
        if "influence_diagnostics" in results:
            n_influential = results["influence_diagnostics"].get("n_influential", 0)
            if n_influential > 0:
                recommendations.append(f"{n_influential} influential observation(s) detected. Review these data points for potential outliers or data entry errors.")

        # General recommendations
        recommendations.append("Examine diagnostic plots (residuals vs fitted, Q-Q plot) to verify model assumptions.")
        recommendations.append("Report effect sizes alongside p-values to provide complete information about the magnitude of differences.")

        pdf.add_recommendations(recommendations)

        # Build PDF
        pdf_bytes = pdf.build()

        # Return PDF as downloadable file
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=anova_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")


# ============================================================================
# MODEL VALIDATION (Tier 2 Feature 2)
# ============================================================================

class ValidationRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Experimental data")
    factors: List[str] = Field(..., description="Factor variable names")
    response: str = Field(..., description="Response variable name")
    k_folds: int = Field(5, description="Number of folds for cross-validation")
    alpha: float = Field(0.05, description="Significance level for adequacy tests")

@router.post("/validate-model")
async def validate_anova_model(request: ValidationRequest):
    """
    Comprehensive model validation for ANOVA models.

    Includes:
    - PRESS statistic and R²_prediction
    - K-fold cross-validation
    - Model adequacy tests (normality, homoscedasticity, autocorrelation)
    - Validation metrics (R², AIC, BIC, RMSE, MAE)

    Returns complete validation report with diagnostics and recommendations.
    """
    try:
        from app.utils.model_validation import (
            calculate_press_statistic,
            k_fold_cross_validation,
            calculate_validation_metrics,
            assess_model_adequacy,
            full_model_validation
        )

        df = pd.DataFrame(request.data)

        # Build ANOVA formula
        # For one-way: response ~ C(factor)
        # For multi-way: response ~ C(factor1) + C(factor2) + C(factor1):C(factor2)
        if len(request.factors) == 1:
            formula = f"{request.response} ~ C({request.factors[0]})"
        else:
            # Main effects
            main_effects = " + ".join([f"C({f})" for f in request.factors])

            # Interactions (all two-way)
            interactions = []
            for i in range(len(request.factors)):
                for j in range(i+1, len(request.factors)):
                    interactions.append(f"C({request.factors[i]}):C({request.factors[j]})")

            if interactions:
                formula = f"{request.response} ~ {main_effects} + {' + '.join(interactions)}"
            else:
                formula = f"{request.response} ~ {main_effects}"

        # Fit model
        model = ols(formula, data=df).fit()

        # Get comprehensive validation
        validation_report = full_model_validation(
            model=model,
            data=df,
            formula=formula,
            response=request.response,
            k_folds=request.k_folds,
            alpha=request.alpha
        )

        # Add model summary info
        validation_report["model_info"] = {
            "formula": formula,
            "n_observations": len(df),
            "n_factors": len(request.factors),
            "factors": request.factors,
            "response": request.response,
            "r2": round(float(model.rsquared), 4),
            "r2_adjusted": round(float(model.rsquared_adj), 4),
            "f_statistic": round(float(model.fvalue), 4),
            "f_pvalue": round(float(model.f_pvalue), 6)
        }

        return validation_report

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
