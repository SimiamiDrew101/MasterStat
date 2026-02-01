"""
Time Series Analysis API Module

Provides endpoints for time series analysis including:
- ARIMA/SARIMA modeling
- Seasonal decomposition
- Autocorrelation analysis (ACF/PACF)
- Forecasting with confidence intervals
- Stationarity tests
- Trend analysis
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import acf as acf_plot, pacf as pacf_plot
import warnings

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/api/time-series", tags=["Time Series Analysis"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TimeSeriesData(BaseModel):
    values: List[float] = Field(..., description="Time series values")
    timestamps: Optional[List[str]] = Field(default=None, description="Optional timestamps")
    frequency: Optional[str] = Field(default=None, description="Data frequency: D, W, M, Q, Y")


class DecomposeRequest(BaseModel):
    data: TimeSeriesData
    model: str = Field(default="additive", description="Decomposition model: additive or multiplicative")
    period: Optional[int] = Field(default=None, description="Seasonal period (auto-detected if not provided)")


class ARIMARequest(BaseModel):
    data: TimeSeriesData
    order: Optional[Tuple[int, int, int]] = Field(default=None, description="ARIMA order (p, d, q)")
    seasonal_order: Optional[Tuple[int, int, int, int]] = Field(default=None, description="Seasonal order (P, D, Q, s)")
    auto_order: bool = Field(default=True, description="Automatically select best order")
    max_p: int = Field(default=5, description="Max AR order for auto selection")
    max_q: int = Field(default=5, description="Max MA order for auto selection")
    max_d: int = Field(default=2, description="Max differencing order")


class ForecastRequest(BaseModel):
    data: TimeSeriesData
    horizon: int = Field(..., ge=1, description="Number of periods to forecast")
    order: Optional[Tuple[int, int, int]] = Field(default=None, description="ARIMA order")
    seasonal_order: Optional[Tuple[int, int, int, int]] = Field(default=None, description="Seasonal order")
    confidence_level: float = Field(default=0.95, description="Confidence level for intervals")


class ACFRequest(BaseModel):
    data: TimeSeriesData
    nlags: int = Field(default=40, description="Number of lags to compute")
    alpha: float = Field(default=0.05, description="Significance level for confidence bands")


class StationarityRequest(BaseModel):
    data: TimeSeriesData
    test: str = Field(default="both", description="Test type: adf, kpss, or both")


class TrendRequest(BaseModel):
    data: TimeSeriesData
    method: str = Field(default="linear", description="Trend method: linear, polynomial, moving_average")
    window: int = Field(default=5, description="Window size for moving average")
    degree: int = Field(default=2, description="Polynomial degree")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_float(val):
    """Convert numpy types to Python floats"""
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if pd.isna(val):
        return None
    return val


def make_json_safe(obj):
    """Recursively convert numpy types to Python types"""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


def detect_seasonality(data: np.ndarray, max_period: int = 52) -> int:
    """Detect seasonal period using autocorrelation"""
    n = len(data)
    if n < 2 * max_period:
        max_period = n // 2

    if max_period < 2:
        return 1

    # Compute autocorrelation
    acf_values = acf(data, nlags=max_period, fft=True)

    # Find peaks in ACF (excluding lag 0)
    peaks = []
    for i in range(2, len(acf_values) - 1):
        if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
            if acf_values[i] > 0.1:  # Minimum correlation threshold
                peaks.append((i, acf_values[i]))

    if not peaks:
        return 1

    # Return lag with highest ACF peak
    best_period = max(peaks, key=lambda x: x[1])[0]
    return best_period


def auto_arima_order(data: np.ndarray, max_p: int = 5, max_q: int = 5, max_d: int = 2) -> Tuple[int, int, int]:
    """Automatically select ARIMA order using AIC"""
    best_aic = np.inf
    best_order = (0, 0, 0)

    # Determine differencing order
    d = 0
    temp_data = data.copy()
    for i in range(max_d + 1):
        try:
            result = adfuller(temp_data, autolag='AIC')
            if result[1] < 0.05:  # Stationary
                d = i
                break
            temp_data = np.diff(temp_data)
            d = i + 1
        except:
            break

    d = min(d, max_d)

    # Grid search for p and q
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(data, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except:
                continue

    return best_order


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/decompose")
async def decompose_time_series(request: DecomposeRequest):
    """
    Perform seasonal decomposition of time series.
    Separates data into trend, seasonal, and residual components.
    """
    try:
        data = np.array(request.data.values)
        n = len(data)

        if n < 4:
            raise HTTPException(status_code=400, detail="Need at least 4 data points")

        # Detect or use provided period
        period = request.period
        if period is None:
            period = detect_seasonality(data)
            if period < 2:
                period = min(12, n // 2)  # Default to 12 or half the data

        # Ensure period is valid
        period = max(2, min(period, n // 2))

        # Perform decomposition
        result = seasonal_decompose(
            data,
            model=request.model,
            period=period,
            extrapolate_trend='freq'
        )

        # Handle NaN values at edges
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid

        # Fill NaN values
        trend = pd.Series(trend).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
        residual = pd.Series(residual).fillna(0).values

        # Compute strength of components
        var_total = np.var(data)
        var_trend = np.var(trend) if var_total > 0 else 0
        var_seasonal = np.var(seasonal) if var_total > 0 else 0
        var_residual = np.var(residual) if var_total > 0 else 0

        trend_strength = 1 - (var_residual / (var_trend + var_residual + 1e-10))
        seasonal_strength = 1 - (var_residual / (var_seasonal + var_residual + 1e-10))

        return make_json_safe({
            "original": data.tolist(),
            "trend": trend.tolist(),
            "seasonal": seasonal.tolist(),
            "residual": residual.tolist(),
            "period": period,
            "model": request.model,
            "statistics": {
                "trend_strength": max(0, min(1, trend_strength)),
                "seasonal_strength": max(0, min(1, seasonal_strength)),
                "variance_original": float(var_total),
                "variance_trend": float(var_trend),
                "variance_seasonal": float(var_seasonal),
                "variance_residual": float(var_residual)
            },
            "interpretation": {
                "trend": "Strong trend detected" if trend_strength > 0.5 else "Weak or no trend",
                "seasonality": f"Strong seasonality with period {period}" if seasonal_strength > 0.5 else "Weak seasonality"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stationarity")
async def test_stationarity(request: StationarityRequest):
    """
    Test time series for stationarity using ADF and KPSS tests.
    """
    try:
        data = np.array(request.data.values)
        results = {}

        if request.test in ["adf", "both"]:
            # Augmented Dickey-Fuller test
            # H0: Unit root exists (non-stationary)
            adf_result = adfuller(data, autolag='AIC')
            results["adf"] = {
                "test_statistic": float(adf_result[0]),
                "p_value": float(adf_result[1]),
                "lags_used": int(adf_result[2]),
                "n_observations": int(adf_result[3]),
                "critical_values": {k: float(v) for k, v in adf_result[4].items()},
                "is_stationary": adf_result[1] < 0.05,
                "interpretation": "Stationary (reject unit root)" if adf_result[1] < 0.05 else "Non-stationary (cannot reject unit root)"
            }

        if request.test in ["kpss", "both"]:
            # KPSS test
            # H0: Series is stationary
            try:
                kpss_result = kpss(data, regression='c', nlags='auto')
                results["kpss"] = {
                    "test_statistic": float(kpss_result[0]),
                    "p_value": float(kpss_result[1]),
                    "lags_used": int(kpss_result[2]),
                    "critical_values": {k: float(v) for k, v in kpss_result[3].items()},
                    "is_stationary": kpss_result[1] > 0.05,
                    "interpretation": "Stationary (cannot reject stationarity)" if kpss_result[1] > 0.05 else "Non-stationary (reject stationarity)"
                }
            except Exception as e:
                results["kpss"] = {"error": str(e)}

        # Overall conclusion
        if request.test == "both" and "adf" in results and "kpss" in results:
            adf_stationary = results["adf"].get("is_stationary", False)
            kpss_stationary = results["kpss"].get("is_stationary", False)

            if adf_stationary and kpss_stationary:
                conclusion = "Series is stationary (both tests agree)"
                recommendation = "No differencing needed for ARIMA modeling"
            elif not adf_stationary and not kpss_stationary:
                conclusion = "Series is non-stationary (both tests agree)"
                recommendation = "Apply differencing (d=1 or d=2) before ARIMA modeling"
            elif adf_stationary and not kpss_stationary:
                conclusion = "Series may be trend-stationary"
                recommendation = "Consider detrending or differencing"
            else:
                conclusion = "Series may be difference-stationary"
                recommendation = "Apply first differencing"

            results["conclusion"] = conclusion
            results["recommendation"] = recommendation

        return make_json_safe(results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acf-pacf")
async def compute_acf_pacf(request: ACFRequest):
    """
    Compute Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).
    """
    try:
        data = np.array(request.data.values)
        n = len(data)
        nlags = min(request.nlags, n // 2 - 1)

        if nlags < 1:
            raise HTTPException(status_code=400, detail="Not enough data points for ACF/PACF")

        # Compute ACF
        acf_values = acf(data, nlags=nlags, fft=True)
        acf_confint = 1.96 / np.sqrt(n)  # Approximate confidence interval

        # Compute PACF
        try:
            pacf_values = pacf(data, nlags=nlags, method='ywm')
        except:
            pacf_values = pacf(data, nlags=min(nlags, n // 4), method='ols')

        pacf_confint = 1.96 / np.sqrt(n)

        # Identify significant lags
        significant_acf = [i for i, v in enumerate(acf_values) if abs(v) > acf_confint and i > 0]
        significant_pacf = [i for i, v in enumerate(pacf_values) if abs(v) > pacf_confint and i > 0]

        # Suggest ARIMA orders based on ACF/PACF patterns
        suggested_p = len([l for l in significant_pacf if l <= 5])
        suggested_q = len([l for l in significant_acf if l <= 5])

        return make_json_safe({
            "acf": {
                "values": acf_values.tolist(),
                "lags": list(range(len(acf_values))),
                "confidence_interval": float(acf_confint),
                "significant_lags": significant_acf[:10]  # First 10 significant
            },
            "pacf": {
                "values": pacf_values.tolist(),
                "lags": list(range(len(pacf_values))),
                "confidence_interval": float(pacf_confint),
                "significant_lags": significant_pacf[:10]
            },
            "suggestions": {
                "ar_order_p": min(suggested_p, 5),
                "ma_order_q": min(suggested_q, 5),
                "interpretation": {
                    "acf_pattern": "Gradual decay" if len(significant_acf) > 5 else "Quick cutoff",
                    "pacf_pattern": "Gradual decay" if len(significant_pacf) > 5 else "Quick cutoff"
                }
            },
            "n_observations": n,
            "nlags": nlags
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fit-arima")
async def fit_arima_model(request: ARIMARequest):
    """
    Fit ARIMA or SARIMA model to time series data.
    """
    try:
        data = np.array(request.data.values)
        n = len(data)

        if n < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 data points for ARIMA")

        # Determine order
        if request.auto_order or request.order is None:
            order = auto_arima_order(data, request.max_p, request.max_q, request.max_d)
        else:
            order = tuple(request.order)

        # Fit model
        if request.seasonal_order:
            model = SARIMAX(data, order=order, seasonal_order=tuple(request.seasonal_order))
        else:
            model = ARIMA(data, order=order)

        fitted = model.fit()

        # Get fitted values and residuals
        fitted_values = fitted.fittedvalues
        residuals = fitted.resid

        # Diagnostic statistics
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)

        # Ljung-Box test for residual autocorrelation
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pvalue = float(lb_test['lb_pvalue'].values[0])

        # Normality test on residuals
        _, normality_pvalue = stats.shapiro(residuals[:min(len(residuals), 500)])

        return make_json_safe({
            "order": order,
            "seasonal_order": request.seasonal_order,
            "fitted_values": fitted_values.tolist(),
            "residuals": residuals.tolist(),
            "original": data.tolist(),
            "model_summary": {
                "aic": float(fitted.aic),
                "bic": float(fitted.bic),
                "log_likelihood": float(fitted.llf),
                "n_observations": n
            },
            "parameters": {
                "ar_coefficients": fitted.arparams.tolist() if hasattr(fitted, 'arparams') and fitted.arparams is not None else [],
                "ma_coefficients": fitted.maparams.tolist() if hasattr(fitted, 'maparams') and fitted.maparams is not None else [],
            },
            "diagnostics": {
                "residual_mean": float(residual_mean),
                "residual_std": float(residual_std),
                "ljung_box_pvalue": lb_pvalue,
                "residuals_uncorrelated": lb_pvalue > 0.05,
                "normality_pvalue": float(normality_pvalue),
                "residuals_normal": normality_pvalue > 0.05
            },
            "interpretation": {
                "model_fit": "Good fit" if lb_pvalue > 0.05 else "Residuals show autocorrelation - consider different order",
                "order_meaning": f"AR({order[0]}), I({order[1]}), MA({order[2]})"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast")
async def forecast_time_series(request: ForecastRequest):
    """
    Generate forecasts with confidence intervals.
    """
    try:
        data = np.array(request.data.values)
        n = len(data)

        if n < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 data points for forecasting")

        # Determine order
        if request.order is None:
            order = auto_arima_order(data)
        else:
            order = tuple(request.order)

        # Fit model
        if request.seasonal_order:
            model = SARIMAX(data, order=order, seasonal_order=tuple(request.seasonal_order))
        else:
            model = ARIMA(data, order=order)

        fitted = model.fit()

        # Generate forecast
        forecast_result = fitted.get_forecast(steps=request.horizon)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int(alpha=1 - request.confidence_level)

        # Create forecast index
        forecast_index = list(range(n, n + request.horizon))

        return make_json_safe({
            "historical": {
                "values": data.tolist(),
                "index": list(range(n))
            },
            "forecast": {
                "values": forecast_mean.tolist(),
                "index": forecast_index,
                "lower_ci": forecast_ci.iloc[:, 0].tolist(),
                "upper_ci": forecast_ci.iloc[:, 1].tolist(),
                "confidence_level": request.confidence_level
            },
            "model": {
                "order": order,
                "seasonal_order": request.seasonal_order,
                "aic": float(fitted.aic)
            },
            "summary": {
                "horizon": request.horizon,
                "forecast_mean": float(np.mean(forecast_mean)),
                "forecast_std": float(np.std(forecast_mean)),
                "trend": "increasing" if forecast_mean[-1] > forecast_mean[0] else "decreasing" if forecast_mean[-1] < forecast_mean[0] else "stable"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trend")
async def analyze_trend(request: TrendRequest):
    """
    Analyze and extract trend from time series.
    """
    try:
        data = np.array(request.data.values)
        n = len(data)
        x = np.arange(n)

        if request.method == "linear":
            # Linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            trend = intercept + slope * x
            trend_info = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "std_error": float(std_err),
                "significant": p_value < 0.05
            }

        elif request.method == "polynomial":
            # Polynomial trend
            coeffs = np.polyfit(x, data, request.degree)
            trend = np.polyval(coeffs, x)
            residuals = data - trend
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            trend_info = {
                "coefficients": coeffs.tolist(),
                "degree": request.degree,
                "r_squared": float(r_squared)
            }

        elif request.method == "moving_average":
            # Moving average trend
            window = min(request.window, n // 2)
            trend = pd.Series(data).rolling(window=window, center=True).mean().values
            # Fill NaN at edges
            trend = pd.Series(trend).interpolate().fillna(method='bfill').fillna(method='ffill').values
            trend_info = {
                "window": window
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")

        # Detrended series
        detrended = data - trend

        # Trend direction
        if request.method in ["linear", "polynomial"]:
            if trend[-1] > trend[0]:
                direction = "upward"
            elif trend[-1] < trend[0]:
                direction = "downward"
            else:
                direction = "flat"
        else:
            direction = "upward" if trend[-1] > trend[0] else "downward" if trend[-1] < trend[0] else "flat"

        return make_json_safe({
            "original": data.tolist(),
            "trend": trend.tolist(),
            "detrended": detrended.tolist(),
            "method": request.method,
            "trend_info": trend_info,
            "direction": direction,
            "statistics": {
                "original_mean": float(np.mean(data)),
                "original_std": float(np.std(data)),
                "detrended_mean": float(np.mean(detrended)),
                "detrended_std": float(np.std(detrended))
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summary")
async def time_series_summary(request: TimeSeriesData):
    """
    Get comprehensive summary statistics for time series.
    """
    try:
        data = np.array(request.values)
        n = len(data)

        # Basic statistics
        basic_stats = {
            "n": n,
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "q1": float(np.percentile(data, 25)),
            "q3": float(np.percentile(data, 75)),
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data))
        }

        # Trend test (Mann-Kendall would be ideal, using linear regression as proxy)
        x = np.arange(n)
        slope, _, r_value, p_value, _ = stats.linregress(x, data)
        trend_test = {
            "slope": float(slope),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "has_trend": p_value < 0.05
        }

        # Stationarity (quick ADF test)
        try:
            adf_result = adfuller(data, autolag='AIC')
            stationarity = {
                "adf_statistic": float(adf_result[0]),
                "adf_pvalue": float(adf_result[1]),
                "is_stationary": adf_result[1] < 0.05
            }
        except:
            stationarity = {"error": "Could not compute"}

        # Seasonality detection
        period = detect_seasonality(data)
        seasonality = {
            "detected_period": period,
            "has_seasonality": period > 1
        }

        # First differences statistics
        diff_data = np.diff(data)
        diff_stats = {
            "mean_change": float(np.mean(diff_data)),
            "std_change": float(np.std(diff_data)),
            "positive_changes": int(np.sum(diff_data > 0)),
            "negative_changes": int(np.sum(diff_data < 0))
        }

        return make_json_safe({
            "basic_statistics": basic_stats,
            "trend_analysis": trend_test,
            "stationarity": stationarity,
            "seasonality": seasonality,
            "changes": diff_stats,
            "recommendations": {
                "differencing": "Recommended" if not stationarity.get("is_stationary", True) else "Not needed",
                "seasonal_model": f"Consider SARIMA with period={period}" if period > 1 else "ARIMA should suffice"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
