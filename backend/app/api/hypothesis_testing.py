from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
from scipy import stats

router = APIRouter()

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
    Perform t-test for means comparison
    - One-sample t-test if only sample1 provided
    - Two-sample t-test (independent or paired)
    """
    try:
        sample1 = np.array(request.sample1)

        if request.sample2 is None:
            # One-sample t-test
            result = stats.ttest_1samp(sample1, request.mu, alternative=request.alternative)
            test_type = "One-sample t-test"
            df = len(sample1) - 1
        else:
            sample2 = np.array(request.sample2)
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
        if request.sample2 is None:
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
        if request.sample2 is not None:
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
                "sample2_mean": round(float(np.mean(sample2)), 4) if request.sample2 else None,
                "sample2_std": round(float(np.std(sample2, ddof=1)), 4) if request.sample2 else None,
                "sample2_n": len(sample2) if request.sample2 else None
            },
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
