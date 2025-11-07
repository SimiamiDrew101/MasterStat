from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

router = APIRouter()

class OneWayANOVARequest(BaseModel):
    groups: Dict[str, List[float]] = Field(..., description="Dictionary of group names to data values")
    alpha: float = Field(0.05, description="Significance level")

class TwoWayANOVARequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of observations with factor levels and response")
    factor_a: str = Field(..., description="Name of first factor")
    factor_b: str = Field(..., description="Name of second factor")
    response: str = Field(..., description="Name of response variable")
    alpha: float = Field(0.05, description="Significance level")

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
            "all_data": all_data_with_groups
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
            "factor_b_name": request.factor_b
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
