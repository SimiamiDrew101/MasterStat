"""
Missing Data Imputation API endpoints
Provides various imputation methods for handling missing values
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal, Union
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

router = APIRouter()


class ImputationRequest(BaseModel):
    """Request model for imputation"""
    data: List[Union[float, None]] = Field(..., description="Data values with missing values (None for missing)")
    method: Literal['mean', 'median', 'knn', 'mice', 'linear', 'locf'] = Field(
        ..., description="Imputation method"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Method-specific parameters"
    )


class ImputationPreviewRequest(BaseModel):
    """Request model for imputation preview"""
    data: List[Union[float, None]] = Field(..., description="Data values with missing values")
    method: Literal['mean', 'median', 'knn', 'mice', 'linear', 'locf'] = Field(
        ..., description="Imputation method"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Method-specific parameters"
    )


class ComparisonRequest(BaseModel):
    """Request model for comparing multiple imputation methods"""
    data: List[Union[float, None]] = Field(..., description="Data values with missing values")
    methods: List[Literal['mean', 'median', 'knn', 'mice', 'linear', 'locf']] = Field(
        ..., description="List of methods to compare"
    )
    parameters: Optional[Dict[str, Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Parameters for each method"
    )


def convert_to_numeric_array(data: List[Union[float, None]]) -> np.ndarray:
    """Convert input data to numpy array, handling None as NaN"""
    return np.array([np.nan if x is None else float(x) for x in data])


def get_missing_mask(data: np.ndarray) -> np.ndarray:
    """Get boolean mask of missing values"""
    return np.isnan(data)


def calculate_imputation_statistics(original: np.ndarray, imputed: np.ndarray) -> Dict[str, Any]:
    """Calculate statistics for imputed data"""
    missing_mask = get_missing_mask(original)
    imputed_values = imputed[missing_mask]

    return {
        'n_total': len(original),
        'n_missing': int(np.sum(missing_mask)),
        'n_observed': int(np.sum(~missing_mask)),
        'percent_missing': float(np.sum(missing_mask) / len(original) * 100),
        'imputed_mean': float(np.mean(imputed)),
        'imputed_std': float(np.std(imputed, ddof=1)),
        'imputed_min': float(np.min(imputed)),
        'imputed_max': float(np.max(imputed)),
        'original_mean': float(np.nanmean(original)),
        'original_std': float(np.nanstd(original, ddof=1)),
        'mean_imputed_value': float(np.mean(imputed_values)) if len(imputed_values) > 0 else None
    }


def impute_mean(data: np.ndarray) -> np.ndarray:
    """Mean imputation"""
    imputer = SimpleImputer(strategy='mean')
    return imputer.fit_transform(data.reshape(-1, 1)).flatten()


def impute_median(data: np.ndarray) -> np.ndarray:
    """Median imputation"""
    imputer = SimpleImputer(strategy='median')
    return imputer.fit_transform(data.reshape(-1, 1)).flatten()


def impute_knn(data: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """KNN imputation"""
    # For 1D data, we need to create a 2D array
    # We'll use index as a second feature to preserve order
    data_2d = np.column_stack([data, np.arange(len(data))])

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_2d = imputer.fit_transform(data_2d)

    return imputed_2d[:, 0]


def impute_mice(data: np.ndarray, max_iter: int = 10, random_state: int = 42) -> np.ndarray:
    """MICE (Multiple Imputation by Chained Equations) imputation"""
    # For 1D data, we need to create a 2D array
    # We'll use index as a second feature
    data_2d = np.column_stack([data, np.arange(len(data))])

    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    imputed_2d = imputer.fit_transform(data_2d)

    return imputed_2d[:, 0]


def impute_linear(data: np.ndarray) -> np.ndarray:
    """Linear interpolation"""
    df = pd.Series(data)
    return df.interpolate(method='linear', limit_direction='both').values


def impute_locf(data: np.ndarray) -> np.ndarray:
    """Last Observation Carried Forward (forward fill)"""
    df = pd.Series(data)
    # Forward fill first, then backward fill for leading NaNs
    return df.fillna(method='ffill').fillna(method='bfill').values


@router.post("/impute")
async def impute_data(request: ImputationRequest):
    """
    Apply imputation to data with missing values

    Returns:
    - imputed_data: List of imputed values
    - statistics: Statistics about the imputation
    - method: Method used
    - parameters: Parameters used
    """
    try:
        data = convert_to_numeric_array(request.data)

        if len(data) == 0:
            raise ValueError("No data provided")

        missing_mask = get_missing_mask(data)
        n_missing = np.sum(missing_mask)

        if n_missing == 0:
            return {
                'imputed_data': request.data,
                'statistics': {
                    'n_total': len(data),
                    'n_missing': 0,
                    'n_observed': len(data),
                    'percent_missing': 0.0
                },
                'method': request.method,
                'parameters': request.parameters,
                'message': 'No missing values detected'
            }

        if n_missing == len(data):
            raise ValueError("All values are missing - cannot impute")

        # Apply imputation method
        imputed = None

        if request.method == 'mean':
            imputed = impute_mean(data)

        elif request.method == 'median':
            imputed = impute_median(data)

        elif request.method == 'knn':
            n_neighbors = request.parameters.get('knn_neighbors', 5)
            # Ensure n_neighbors doesn't exceed number of observed values
            n_observed = np.sum(~missing_mask)
            n_neighbors = min(n_neighbors, n_observed)
            imputed = impute_knn(data, n_neighbors=n_neighbors)

        elif request.method == 'mice':
            max_iter = request.parameters.get('mice_iterations', 10)
            random_state = request.parameters.get('mice_random_state', 42)
            imputed = impute_mice(data, max_iter=max_iter, random_state=random_state)

        elif request.method == 'linear':
            imputed = impute_linear(data)

        elif request.method == 'locf':
            imputed = impute_locf(data)

        else:
            raise ValueError(f"Unknown imputation method: {request.method}")

        # Calculate statistics
        statistics = calculate_imputation_statistics(data, imputed)

        # Get sample of imputed values
        imputed_indices = np.where(missing_mask)[0]
        sample_imputed_values = imputed[imputed_indices[:min(10, len(imputed_indices))]].tolist()

        return {
            'imputed_data': imputed.tolist(),
            'statistics': statistics,
            'method': request.method,
            'parameters': request.parameters,
            'sample_imputed_values': sample_imputed_values,
            'imputed_indices': imputed_indices.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/preview")
async def preview_imputation(request: ImputationPreviewRequest):
    """
    Preview imputation results without applying

    Returns:
    - sample_imputed_values: Sample of values that would be imputed
    - statistics: Statistics about the imputation
    - method: Method to be used
    """
    try:
        data = convert_to_numeric_array(request.data)

        if len(data) == 0:
            raise ValueError("No data provided")

        missing_mask = get_missing_mask(data)
        n_missing = np.sum(missing_mask)

        if n_missing == 0:
            return {
                'sample_imputed_values': [],
                'statistics': {
                    'n_total': len(data),
                    'n_missing': 0,
                    'percent_missing': 0.0
                },
                'method': request.method,
                'message': 'No missing values detected'
            }

        # Apply imputation to get preview
        imputed = None

        if request.method == 'mean':
            imputed = impute_mean(data)

        elif request.method == 'median':
            imputed = impute_median(data)

        elif request.method == 'knn':
            n_neighbors = request.parameters.get('knn_neighbors', 5)
            n_observed = np.sum(~missing_mask)
            n_neighbors = min(n_neighbors, n_observed)
            imputed = impute_knn(data, n_neighbors=n_neighbors)

        elif request.method == 'mice':
            max_iter = request.parameters.get('mice_iterations', 10)
            random_state = request.parameters.get('mice_random_state', 42)
            imputed = impute_mice(data, max_iter=max_iter, random_state=random_state)

        elif request.method == 'linear':
            imputed = impute_linear(data)

        elif request.method == 'locf':
            imputed = impute_locf(data)

        else:
            raise ValueError(f"Unknown imputation method: {request.method}")

        # Get sample of imputed values
        imputed_indices = np.where(missing_mask)[0]
        sample_imputed_values = imputed[imputed_indices].tolist()

        # Calculate statistics
        statistics = {
            'mean': float(np.mean(imputed)),
            'std': float(np.std(imputed, ddof=1)),
            'min': float(np.min(imputed)),
            'max': float(np.max(imputed)),
            'n_missing': int(n_missing)
        }

        return {
            'sample_imputed_values': sample_imputed_values,
            'statistics': statistics,
            'method': request.method
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def calculate_cv_rmse(data: np.ndarray, method: str, params: Dict[str, Any]) -> float:
    """
    Calculate cross-validation RMSE for imputation method
    Removes 20% of observed values, imputes them, and calculates RMSE
    """
    try:
        missing_mask = get_missing_mask(data)
        observed_mask = ~missing_mask
        observed_indices = np.where(observed_mask)[0]

        if len(observed_indices) < 5:
            return None  # Not enough observed values for CV

        # Remove 20% of observed values for testing
        np.random.seed(42)
        n_test = max(1, int(len(observed_indices) * 0.2))
        test_indices = np.random.choice(observed_indices, size=n_test, replace=False)

        # Create CV data with additional missing values
        cv_data = data.copy()
        true_values = cv_data[test_indices].copy()
        cv_data[test_indices] = np.nan

        # Impute the CV data
        imputed_cv = None

        if method == 'mean':
            imputed_cv = impute_mean(cv_data)
        elif method == 'median':
            imputed_cv = impute_median(cv_data)
        elif method == 'knn':
            n_neighbors = params.get('knn_neighbors', 5)
            n_observed = np.sum(~get_missing_mask(cv_data))
            n_neighbors = min(n_neighbors, max(1, n_observed - 1))
            imputed_cv = impute_knn(cv_data, n_neighbors=n_neighbors)
        elif method == 'mice':
            max_iter = params.get('mice_iterations', 10)
            random_state = params.get('mice_random_state', 42)
            imputed_cv = impute_mice(cv_data, max_iter=max_iter, random_state=random_state)
        elif method == 'linear':
            imputed_cv = impute_linear(cv_data)
        elif method == 'locf':
            imputed_cv = impute_locf(cv_data)

        # Calculate RMSE
        predicted_values = imputed_cv[test_indices]
        rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))

        return float(rmse)

    except Exception as e:
        # If CV fails, return None
        return None


@router.post("/compare")
async def compare_methods(request: ComparisonRequest):
    """
    Compare multiple imputation methods

    Returns:
    - comparison: Dictionary of method -> results
    - recommendations: Recommended method based on data characteristics
    """
    try:
        data = convert_to_numeric_array(request.data)

        if len(data) == 0:
            raise ValueError("No data provided")

        missing_mask = get_missing_mask(data)
        n_missing = np.sum(missing_mask)

        if n_missing == 0:
            return {
                'comparison': {},
                'recommendations': 'No missing values detected',
                'message': 'No imputation needed'
            }

        comparison = {}

        for method in request.methods:
            try:
                params = request.parameters.get(method, {})

                # Apply imputation
                imputed = None

                if method == 'mean':
                    imputed = impute_mean(data)
                elif method == 'median':
                    imputed = impute_median(data)
                elif method == 'knn':
                    n_neighbors = params.get('knn_neighbors', 5)
                    n_observed = np.sum(~missing_mask)
                    n_neighbors = min(n_neighbors, n_observed)
                    imputed = impute_knn(data, n_neighbors=n_neighbors)
                elif method == 'mice':
                    max_iter = params.get('mice_iterations', 10)
                    random_state = params.get('mice_random_state', 42)
                    imputed = impute_mice(data, max_iter=max_iter, random_state=random_state)
                elif method == 'linear':
                    imputed = impute_linear(data)
                elif method == 'locf':
                    imputed = impute_locf(data)

                # Calculate statistics
                statistics = calculate_imputation_statistics(data, imputed)

                # Get imputed values for comparison
                imputed_values = imputed[missing_mask]

                # Calculate cross-validation RMSE
                cv_rmse = calculate_cv_rmse(data, method, params)

                comparison[method] = {
                    'statistics': statistics,
                    'imputed_values_sample': imputed_values[:10].tolist(),
                    'imputed_mean': float(np.mean(imputed_values)),
                    'imputed_std': float(np.std(imputed_values, ddof=1)),
                    'distribution_preservation': calculate_distribution_preservation(data, imputed),
                    'cv_rmse': cv_rmse
                }

            except Exception as e:
                comparison[method] = {
                    'error': str(e),
                    'status': 'failed'
                }

        # Generate recommendations
        recommendations = generate_recommendations(data, comparison, n_missing)

        return {
            'comparison': comparison,
            'recommendations': recommendations,
            'n_missing': int(n_missing),
            'percent_missing': float(n_missing / len(data) * 100)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def calculate_distribution_preservation(original: np.ndarray, imputed: np.ndarray) -> Dict[str, float]:
    """
    Calculate how well the imputation preserves the original distribution
    """
    observed = original[~get_missing_mask(original)]

    # Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = stats.ks_2samp(observed, imputed)

    # Mean difference
    mean_diff = abs(np.mean(imputed) - np.mean(observed))

    # Variance ratio
    var_ratio = np.var(imputed, ddof=1) / np.var(observed, ddof=1)

    return {
        'ks_statistic': float(ks_statistic),
        'ks_pvalue': float(ks_pvalue),
        'mean_difference': float(mean_diff),
        'variance_ratio': float(var_ratio),
        'distribution_preserved': bool(ks_pvalue > 0.05)
    }


def generate_recommendations(data: np.ndarray, comparison: Dict, n_missing: int) -> str:
    """
    Generate recommendations based on comparison results
    """
    percent_missing = n_missing / len(data) * 100

    recommendations = []

    # Based on percentage missing
    if percent_missing < 5:
        recommendations.append("Low percentage of missing data (<5%). Simple methods like mean or median are appropriate.")
    elif percent_missing < 10:
        recommendations.append("Moderate percentage of missing data (5-10%). Consider KNN or MICE for better accuracy.")
    else:
        recommendations.append(f"High percentage of missing data ({percent_missing:.1f}%). Advanced methods (MICE, KNN) recommended. Interpret results with caution.")

    # Check for successful methods
    successful_methods = [m for m, r in comparison.items() if 'error' not in r]

    if successful_methods:
        # Find method with best distribution preservation
        best_method = None
        best_ks_pvalue = 0

        for method in successful_methods:
            if 'distribution_preservation' in comparison[method]:
                ks_pvalue = comparison[method]['distribution_preservation']['ks_pvalue']
                if ks_pvalue > best_ks_pvalue:
                    best_ks_pvalue = ks_pvalue
                    best_method = method

        if best_method:
            recommendations.append(f"Based on distribution preservation, '{best_method}' appears most suitable (KS p-value: {best_ks_pvalue:.4f}).")

    return " ".join(recommendations)


@router.get("/methods")
async def get_available_methods():
    """
    Get list of available imputation methods with descriptions
    """
    return {
        'methods': [
            {
                'value': 'mean',
                'label': 'Mean Imputation',
                'description': 'Replace missing values with the mean of observed values',
                'complexity': 'low',
                'recommended_for': 'Small amounts of missing data (<5%), normally distributed data'
            },
            {
                'value': 'median',
                'label': 'Median Imputation',
                'description': 'Replace missing values with the median of observed values',
                'complexity': 'low',
                'recommended_for': 'Small amounts of missing data, skewed distributions, presence of outliers'
            },
            {
                'value': 'knn',
                'label': 'KNN Imputation',
                'description': 'Use K-Nearest Neighbors to estimate missing values',
                'complexity': 'medium',
                'recommended_for': 'Moderate missing data (5-20%), when similar observations can inform estimates',
                'parameters': ['knn_neighbors (1-20, default: 5)']
            },
            {
                'value': 'mice',
                'label': 'MICE (Multiple Imputation)',
                'description': 'Multivariate Imputation by Chained Equations',
                'complexity': 'high',
                'recommended_for': 'Complex missing patterns, multiple variables, when accuracy is critical',
                'parameters': ['mice_iterations (1-50, default: 10)', 'mice_random_state (default: 42)']
            },
            {
                'value': 'linear',
                'label': 'Linear Interpolation',
                'description': 'Interpolate missing values linearly between observed values',
                'complexity': 'low',
                'recommended_for': 'Time series or ordered data with temporal/spatial continuity'
            },
            {
                'value': 'locf',
                'label': 'LOCF (Last Observation Carried Forward)',
                'description': 'Forward fill missing values with the last observed value',
                'complexity': 'low',
                'recommended_for': 'Time series data, repeated measures, when values are expected to be stable'
            }
        ]
    }
