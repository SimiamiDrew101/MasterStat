"""
Statistical Analysis API endpoints
Provides utility endpoints for correlation, descriptive statistics, and visualization support
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats

router = APIRouter()


class CorrelationRequest(BaseModel):
    """Request model for correlation matrix calculation"""
    data: Dict[str, List[float]] = Field(..., description="Dictionary of variable names to data arrays")
    method: Optional[str] = Field("pearson", description="Correlation method: pearson, spearman, or kendall")
    calculate_pvalues: Optional[bool] = Field(True, description="Whether to calculate p-values")


@router.post("/correlation")
async def calculate_correlation(request: CorrelationRequest):
    """
    Calculate correlation matrix with optional p-values

    Returns:
    - correlation_matrix: 2D array of correlation coefficients
    - p_values: 2D array of p-values (if calculate_pvalues=True)
    - variable_names: List of variable names
    - method: Correlation method used
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)

        if df.empty:
            raise ValueError("No data provided")

        # Check for sufficient data
        if len(df) < 3:
            raise ValueError("At least 3 observations required for correlation analysis")

        variable_names = list(df.columns)
        n_vars = len(variable_names)

        # Calculate correlation matrix
        if request.method == "pearson":
            corr_matrix = df.corr(method='pearson').values.tolist()
        elif request.method == "spearman":
            corr_matrix = df.corr(method='spearman').values.tolist()
        elif request.method == "kendall":
            corr_matrix = df.corr(method='kendall').values.tolist()
        else:
            raise ValueError(f"Invalid correlation method: {request.method}")

        # Calculate p-values if requested
        p_values = None
        if request.calculate_pvalues:
            p_values = [[1.0] * n_vars for _ in range(n_vars)]

            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    # Get data for the two variables
                    x = df.iloc[:, i].dropna()
                    y = df.iloc[:, j].dropna()

                    # Find common indices
                    common_idx = x.index.intersection(y.index)
                    if len(common_idx) < 3:
                        p_values[i][j] = p_values[j][i] = np.nan
                        continue

                    x_common = x.loc[common_idx]
                    y_common = y.loc[common_idx]

                    # Calculate correlation and p-value
                    if request.method == "pearson":
                        _, p_val = stats.pearsonr(x_common, y_common)
                    elif request.method == "spearman":
                        _, p_val = stats.spearmanr(x_common, y_common)
                    elif request.method == "kendall":
                        _, p_val = stats.kendalltau(x_common, y_common)

                    p_values[i][j] = p_values[j][i] = float(p_val)

        # Calculate summary statistics
        flat_corrs = [corr_matrix[i][j] for i in range(n_vars) for j in range(i + 1, n_vars)]

        if flat_corrs:
            mean_correlation = np.mean([abs(c) for c in flat_corrs])
            max_correlation = max([abs(c) for c in flat_corrs])
            strong_correlations = sum(1 for c in flat_corrs if abs(c) > 0.7)
        else:
            mean_correlation = 0
            max_correlation = 0
            strong_correlations = 0

        return {
            "correlation_matrix": corr_matrix,
            "p_values": p_values,
            "variable_names": variable_names,
            "method": request.method,
            "summary": {
                "n_variables": n_vars,
                "n_observations": len(df),
                "mean_abs_correlation": round(mean_correlation, 4),
                "max_abs_correlation": round(max_correlation, 4),
                "strong_correlations_count": strong_correlations,
                "strong_correlations_threshold": 0.7
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class DescriptiveStatsRequest(BaseModel):
    """Request model for descriptive statistics"""
    data: List[float] = Field(..., description="Array of numeric data")
    variable_name: Optional[str] = Field("Variable", description="Name of the variable")


@router.post("/descriptive")
async def calculate_descriptive_stats(request: DescriptiveStatsRequest):
    """
    Calculate comprehensive descriptive statistics

    Returns:
    - Basic stats: n, mean, std, min, max, range
    - Percentiles: 25th, 50th (median), 75th
    - Distribution: skewness, kurtosis
    - Normality: Shapiro-Wilk test
    """
    try:
        data = np.array(request.data)

        # Remove NaN values
        valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            raise ValueError("No valid data provided")

        # Basic statistics
        n = len(valid_data)
        mean = float(np.mean(valid_data))
        std = float(np.std(valid_data, ddof=1)) if n > 1 else 0.0
        min_val = float(np.min(valid_data))
        max_val = float(np.max(valid_data))
        range_val = max_val - min_val

        # Percentiles
        q25, median, q75 = np.percentile(valid_data, [25, 50, 75])
        iqr = q75 - q25

        # Distribution shape
        skewness = float(stats.skew(valid_data)) if n > 2 else np.nan
        kurtosis = float(stats.kurtosis(valid_data)) if n > 3 else np.nan

        # Normality test (only for n >= 3)
        shapiro_statistic = None
        shapiro_p_value = None
        if n >= 3:
            try:
                shapiro_statistic, shapiro_p_value = stats.shapiro(valid_data)
                shapiro_statistic = float(shapiro_statistic)
                shapiro_p_value = float(shapiro_p_value)
            except:
                pass

        # Outlier detection using IQR method
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
        n_outliers = len(outliers)

        return {
            "variable_name": request.variable_name,
            "n": n,
            "n_missing": len(data) - n,
            "mean": round(mean, 6),
            "std": round(std, 6),
            "min": round(min_val, 6),
            "max": round(max_val, 6),
            "range": round(range_val, 6),
            "percentiles": {
                "25th": round(float(q25), 6),
                "50th (median)": round(float(median), 6),
                "75th": round(float(q75), 6)
            },
            "iqr": round(float(iqr), 6),
            "distribution": {
                "skewness": round(skewness, 6) if not np.isnan(skewness) else None,
                "kurtosis": round(kurtosis, 6) if not np.isnan(kurtosis) else None
            },
            "normality_test": {
                "test": "Shapiro-Wilk",
                "statistic": round(shapiro_statistic, 6) if shapiro_statistic is not None else None,
                "p_value": round(shapiro_p_value, 6) if shapiro_p_value is not None else None,
                "is_normal": shapiro_p_value > 0.05 if shapiro_p_value is not None else None
            },
            "outliers": {
                "count": n_outliers,
                "lower_bound": round(float(lower_bound), 6),
                "upper_bound": round(float(upper_bound), 6),
                "values": [round(float(x), 6) for x in outliers.tolist()[:10]]  # Limit to 10
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
