"""
Data Preprocessing API endpoints
Provides transformation, centering, scaling, and outlier detection capabilities
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import boxcox1p
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

router = APIRouter()


class TransformRequest(BaseModel):
    """Request model for data transformation"""
    data: List[float] = Field(..., description="Data values to transform")
    transform_type: Literal['log', 'log10', 'sqrt', 'boxcox', 'zscore', 'minmax', 'rank', 'none'] = Field(
        ..., description="Type of transformation to apply"
    )
    centering: Optional[Literal['mean', 'median', 'custom', 'none']] = Field('none', description="Centering method")
    scaling: Optional[Literal['std', 'range', 'custom', 'none']] = Field('none', description="Scaling method")
    custom_center: Optional[float] = Field(None, description="Custom center value")
    custom_scale: Optional[float] = Field(None, description="Custom scale value")
    boxcox_lambda: Optional[float] = Field(None, description="Lambda parameter for Box-Cox (auto if None)")


class OutlierDetectionRequest(BaseModel):
    """Request model for outlier detection"""
    data: List[float] = Field(..., description="Data values for outlier detection")
    method: Literal['zscore', 'iqr', 'isolation_forest', 'elliptic_envelope', 'dbscan'] = Field(
        ..., description="Outlier detection method"
    )
    threshold: Optional[float] = Field(3.0, description="Threshold for z-score method")
    contamination: Optional[float] = Field(0.1, description="Expected proportion of outliers (0.0-0.5)")


@router.post("/transform")
async def transform_data(request: TransformRequest):
    """
    Apply data transformation with optional centering and scaling

    Returns:
    - transformed_data: List of transformed values
    - statistics: Before and after statistics
    - parameters: Transformation parameters used
    - warnings: Any warnings about the transformation
    """
    try:
        data = np.array(request.data)

        if len(data) == 0:
            raise ValueError("No data provided")

        if np.any(np.isnan(data)):
            raise ValueError("Data contains NaN values")

        warnings = []
        transformed = data.copy()
        parameters = {
            'transform_type': request.transform_type,
            'centering': request.centering,
            'scaling': request.scaling
        }

        # Calculate original statistics
        original_stats = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data, ddof=1)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'q1': float(np.percentile(data, 25)),
            'q3': float(np.percentile(data, 75)),
            'n': len(data)
        }

        # Apply main transformation
        if request.transform_type == 'log':
            if np.any(data <= 0):
                raise ValueError("Log transformation requires all values > 0")
            transformed = np.log(data)

        elif request.transform_type == 'log10':
            if np.any(data <= 0):
                raise ValueError("Log10 transformation requires all values > 0")
            transformed = np.log10(data)

        elif request.transform_type == 'sqrt':
            if np.any(data < 0):
                raise ValueError("Square root transformation requires all values >= 0")
            transformed = np.sqrt(data)

        elif request.transform_type == 'boxcox':
            if np.any(data <= 0):
                raise ValueError("Box-Cox transformation requires all values > 0")

            if request.boxcox_lambda is not None:
                # Use specified lambda
                lmbda = request.boxcox_lambda
                if lmbda == 0:
                    transformed = np.log(data)
                else:
                    transformed = (np.power(data, lmbda) - 1) / lmbda
            else:
                # Find optimal lambda
                transformed, lmbda = stats.boxcox(data)
                parameters['boxcox_lambda'] = float(lmbda)
                warnings.append(f"Optimal Box-Cox lambda: {lmbda:.4f}")

        elif request.transform_type == 'zscore':
            scaler = StandardScaler()
            transformed = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            parameters['zscore_mean'] = float(scaler.mean_[0])
            parameters['zscore_std'] = float(np.sqrt(scaler.var_[0]))

        elif request.transform_type == 'minmax':
            scaler = MinMaxScaler()
            transformed = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            parameters['minmax_min'] = float(scaler.data_min_[0])
            parameters['minmax_max'] = float(scaler.data_max_[0])

        elif request.transform_type == 'rank':
            transformed = stats.rankdata(data, method='average')

        elif request.transform_type == 'none':
            transformed = data.copy()

        # Apply centering
        center_value = 0
        if request.centering == 'mean':
            center_value = np.mean(transformed)
            transformed = transformed - center_value
            parameters['center_value'] = float(center_value)

        elif request.centering == 'median':
            center_value = np.median(transformed)
            transformed = transformed - center_value
            parameters['center_value'] = float(center_value)

        elif request.centering == 'custom' and request.custom_center is not None:
            center_value = request.custom_center
            transformed = transformed - center_value
            parameters['center_value'] = float(center_value)

        # Apply scaling
        scale_value = 1
        if request.scaling == 'std':
            scale_value = np.std(transformed, ddof=1)
            if scale_value > 0:
                transformed = transformed / scale_value
                parameters['scale_value'] = float(scale_value)
            else:
                warnings.append("Standard deviation is zero, scaling skipped")

        elif request.scaling == 'range':
            min_val = np.min(transformed)
            max_val = np.max(transformed)
            scale_value = max_val - min_val
            if scale_value > 0:
                transformed = (transformed - min_val) / scale_value
                parameters['scale_value'] = float(scale_value)
                parameters['scale_min'] = float(min_val)
            else:
                warnings.append("Range is zero, scaling skipped")

        elif request.scaling == 'custom' and request.custom_scale is not None:
            scale_value = request.custom_scale
            if scale_value != 0:
                transformed = transformed / scale_value
                parameters['scale_value'] = float(scale_value)
            else:
                warnings.append("Custom scale is zero, scaling skipped")

        # Calculate transformed statistics
        transformed_stats = {
            'mean': float(np.mean(transformed)),
            'std': float(np.std(transformed, ddof=1)),
            'min': float(np.min(transformed)),
            'max': float(np.max(transformed)),
            'median': float(np.median(transformed)),
            'q1': float(np.percentile(transformed, 25)),
            'q3': float(np.percentile(transformed, 75)),
            'n': len(transformed)
        }

        return {
            'success': True,
            'transformed_data': transformed.tolist(),
            'original_statistics': original_stats,
            'transformed_statistics': transformed_stats,
            'parameters': parameters,
            'warnings': warnings
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformation error: {str(e)}")


@router.post("/detect-outliers")
async def detect_outliers(request: OutlierDetectionRequest):
    """
    Detect outliers using various methods

    Returns:
    - outlier_indices: List of indices identified as outliers
    - outlier_scores: Score for each data point (higher = more likely outlier)
    - is_outlier: Boolean array indicating outliers
    - statistics: Statistics about the data and outliers
    - method_info: Information about the detection method used
    """
    try:
        data = np.array(request.data).reshape(-1, 1)

        if len(data) == 0:
            raise ValueError("No data provided")

        if np.any(np.isnan(data)):
            raise ValueError("Data contains NaN values")

        outlier_mask = np.zeros(len(data), dtype=bool)
        outlier_scores = np.zeros(len(data))
        method_info = {'method': request.method}

        # Apply outlier detection method
        if request.method == 'zscore':
            # Z-score method
            mean = np.mean(data)
            std = np.std(data, ddof=1)

            if std == 0:
                raise ValueError("Standard deviation is zero, cannot use z-score method")

            z_scores = np.abs((data.flatten() - mean) / std)
            outlier_mask = z_scores > request.threshold
            outlier_scores = z_scores

            method_info.update({
                'threshold': request.threshold,
                'mean': float(mean),
                'std': float(std)
            })

        elif request.method == 'iqr':
            # IQR method
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (data.flatten() < lower_bound) | (data.flatten() > upper_bound)

            # Calculate scores as distance from bounds
            scores_lower = np.maximum(0, lower_bound - data.flatten())
            scores_upper = np.maximum(0, data.flatten() - upper_bound)
            outlier_scores = scores_lower + scores_upper

            method_info.update({
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            })

        elif request.method == 'isolation_forest':
            # Isolation Forest
            if len(data) < 10:
                raise ValueError("Isolation Forest requires at least 10 data points")

            clf = IsolationForest(
                contamination=request.contamination,
                random_state=42
            )
            predictions = clf.fit_predict(data)
            outlier_mask = predictions == -1

            # Get anomaly scores (more negative = more anomalous)
            scores = clf.score_samples(data)
            outlier_scores = -scores  # Invert so higher = more outlier-like

            method_info.update({
                'contamination': request.contamination,
                'n_estimators': clf.n_estimators
            })

        elif request.method == 'elliptic_envelope':
            # Robust covariance (Minimum Covariance Determinant)
            if len(data) < 5:
                raise ValueError("Elliptic Envelope requires at least 5 data points")

            clf = EllipticEnvelope(
                contamination=request.contamination,
                random_state=42
            )
            predictions = clf.fit_predict(data)
            outlier_mask = predictions == -1

            # Get Mahalanobis distances
            mahal_dist = clf.mahalanobis(data - clf.location_)
            outlier_scores = mahal_dist

            method_info.update({
                'contamination': request.contamination,
                'location': float(clf.location_[0])
            })

        elif request.method == 'dbscan':
            # DBSCAN clustering
            if len(data) < 5:
                raise ValueError("DBSCAN requires at least 5 data points")

            # Auto-determine eps using heuristic
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=min(5, len(data) - 1))
            neighbors.fit(data)
            distances, _ = neighbors.kneighbors(data)
            eps = np.percentile(distances[:, -1], 90)

            clustering = DBSCAN(eps=eps, min_samples=3)
            labels = clustering.fit_predict(data)

            # Points with label -1 are outliers
            outlier_mask = labels == -1

            # Calculate scores as distance to nearest core point
            core_mask = np.isin(clustering.core_sample_indices_, range(len(data)))
            if np.any(core_mask):
                core_points = data[core_mask]
                distances_to_core = np.min(
                    np.abs(data - core_points.reshape(1, -1, 1)),
                    axis=1
                ).flatten()
                outlier_scores = distances_to_core
            else:
                outlier_scores = np.ones(len(data))

            method_info.update({
                'eps': float(eps),
                'min_samples': 3,
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
            })

        # Calculate statistics
        outlier_indices = np.where(outlier_mask)[0].tolist()
        n_outliers = int(np.sum(outlier_mask))
        outlier_percentage = float(n_outliers / len(data) * 100)

        # Statistics for non-outliers
        inlier_data = data[~outlier_mask]
        if len(inlier_data) > 0:
            inlier_stats = {
                'mean': float(np.mean(inlier_data)),
                'std': float(np.std(inlier_data, ddof=1)) if len(inlier_data) > 1 else 0.0,
                'min': float(np.min(inlier_data)),
                'max': float(np.max(inlier_data)),
                'median': float(np.median(inlier_data))
            }
        else:
            inlier_stats = None

        return {
            'success': True,
            'outlier_indices': outlier_indices,
            'outlier_scores': outlier_scores.tolist(),
            'is_outlier': outlier_mask.tolist(),
            'statistics': {
                'n_total': len(data),
                'n_outliers': n_outliers,
                'n_inliers': len(data) - n_outliers,
                'outlier_percentage': outlier_percentage,
                'data_mean': float(np.mean(data)),
                'data_std': float(np.std(data, ddof=1)),
                'inlier_statistics': inlier_stats
            },
            'method_info': method_info
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outlier detection error: {str(e)}")


@router.post("/recommend-transform")
async def recommend_transform(data: List[float]):
    """
    Recommend appropriate transformation based on data distribution

    Analyzes skewness, range, and normality to suggest transformations
    """
    try:
        values = np.array(data)

        if len(values) < 3:
            raise ValueError("Need at least 3 data points for recommendation")

        # Calculate distribution metrics
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        skewness = stats.skew(values)
        min_val = np.min(values)
        max_val = np.max(values)

        # Normality tests
        if len(values) >= 3:
            _, shapiro_p = stats.shapiro(values)
        else:
            shapiro_p = None

        # Recommendation logic
        recommendations = []

        # Check if data is already normal
        if shapiro_p and shapiro_p > 0.05:
            recommendations.append({
                'transform': 'none',
                'reason': 'Data appears normally distributed',
                'priority': 1
            })

        # Check for positive skew
        if skewness > 1:
            if min_val > 0:
                recommendations.append({
                    'transform': 'log',
                    'reason': 'Positive skew with positive values',
                    'priority': 2
                })
                recommendations.append({
                    'transform': 'boxcox',
                    'reason': 'Optimal transformation for positive skew',
                    'priority': 1
                })
            elif min_val >= 0:
                recommendations.append({
                    'transform': 'sqrt',
                    'reason': 'Positive skew with non-negative values',
                    'priority': 2
                })

        # Check for standardization needs
        if abs(mean) > 10 * std or std > abs(mean):
            recommendations.append({
                'transform': 'zscore',
                'reason': 'Large mean or variance, consider standardization',
                'priority': 3
            })

        # Check for scaling needs
        if max_val - min_val > 100:
            recommendations.append({
                'transform': 'minmax',
                'reason': 'Large range, consider min-max scaling',
                'priority': 3
            })

        # Default recommendation
        if not recommendations:
            recommendations.append({
                'transform': 'none',
                'reason': 'No transformation strongly indicated',
                'priority': 1
            })

        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])

        return {
            'success': True,
            'recommendations': recommendations,
            'data_characteristics': {
                'mean': float(mean),
                'std': float(std),
                'skewness': float(skewness),
                'min': float(min_val),
                'max': float(max_val),
                'range': float(max_val - min_val),
                'shapiro_p_value': float(shapiro_p) if shapiro_p else None,
                'is_normal': bool(shapiro_p > 0.05) if shapiro_p else None
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")
