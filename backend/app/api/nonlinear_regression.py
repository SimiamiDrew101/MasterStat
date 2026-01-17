from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import warnings

router = APIRouter()

# ============================================================================
# Built-in Nonlinear Model Functions
# ============================================================================

def exponential(x, a, b):
    """
    Exponential growth/decay model
    y = a * exp(b*x)

    Parameters:
    - a: Initial value (y-intercept multiplier)
    - b: Growth rate (positive = growth, negative = decay)
    """
    return a * np.exp(b * x)

def logistic(x, L, k, x0):
    """
    Logistic (S-curve) model
    y = L / (1 + exp(-k*(x-x0)))

    Parameters:
    - L: Maximum value (carrying capacity)
    - k: Growth rate (steepness)
    - x0: Midpoint (inflection point)
    """
    return L / (1 + np.exp(-k * (x - x0)))

def michaelis_menten(x, Vmax, Km):
    """
    Michaelis-Menten enzyme kinetics model
    y = Vmax*x / (Km + x)

    Parameters:
    - Vmax: Maximum reaction rate
    - Km: Michaelis constant (substrate concentration at half Vmax)
    """
    return Vmax * x / (Km + x)

def power_law(x, a, b):
    """
    Power law model
    y = a * x^b

    Parameters:
    - a: Scaling constant
    - b: Exponent (power)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return a * np.power(np.abs(x), b)

def gompertz(x, a, b, c):
    """
    Gompertz growth model (asymmetric S-curve)
    y = a * exp(-b * exp(-c*x))

    Parameters:
    - a: Asymptote (maximum value)
    - b: Displacement along x-axis
    - c: Growth rate
    """
    return a * np.exp(-b * np.exp(-c * x))

def weibull(x, a, b, c):
    """
    Weibull model
    y = a * (1 - exp(-(x/b)^c))

    Parameters:
    - a: Asymptote
    - b: Scale parameter
    - c: Shape parameter
    """
    return a * (1 - np.exp(-np.power(x / b, c)))

def logarithmic(x, a, b):
    """
    Logarithmic model
    y = a + b*ln(x)

    Parameters:
    - a: Intercept
    - b: Slope
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return a + b * np.log(np.abs(x) + 1e-10)

# ============================================================================
# Model Library
# ============================================================================

BUILTIN_MODELS = {
    "exponential": {
        "function": exponential,
        "params": ["a", "b"],
        "equation": "y = a * exp(b*x)",
        "description": "Exponential growth or decay",
        "typical_use": "Population growth, radioactive decay, compound interest"
    },
    "logistic": {
        "function": logistic,
        "params": ["L", "k", "x0"],
        "equation": "y = L / (1 + exp(-k*(x-x0)))",
        "description": "S-shaped growth curve with saturation",
        "typical_use": "Population growth with carrying capacity, market adoption"
    },
    "michaelis_menten": {
        "function": michaelis_menten,
        "params": ["Vmax", "Km"],
        "equation": "y = Vmax*x / (Km + x)",
        "description": "Enzyme kinetics and saturation phenomena",
        "typical_use": "Enzyme reactions, substrate saturation, pharmacokinetics"
    },
    "power_law": {
        "function": power_law,
        "params": ["a", "b"],
        "equation": "y = a * x^b",
        "description": "Power relationship between variables",
        "typical_use": "Allometric scaling, physics relationships, dose-response"
    },
    "gompertz": {
        "function": gompertz,
        "params": ["a", "b", "c"],
        "equation": "y = a * exp(-b * exp(-c*x))",
        "description": "Asymmetric S-curve with slow start",
        "typical_use": "Tumor growth, mortality rates, technology adoption"
    },
    "weibull": {
        "function": weibull,
        "params": ["a", "b", "c"],
        "equation": "y = a * (1 - exp(-(x/b)^c))",
        "description": "Flexible S-curve for reliability analysis",
        "typical_use": "Failure analysis, reliability testing, survival analysis"
    },
    "logarithmic": {
        "function": logarithmic,
        "params": ["a", "b"],
        "equation": "y = a + b*ln(x)",
        "description": "Logarithmic relationship",
        "typical_use": "Sensory perception, learning curves, information theory"
    }
}

# ============================================================================
# Pydantic Models
# ============================================================================

class NonlinearFitRequest(BaseModel):
    x_data: List[float] = Field(..., description="Independent variable values")
    y_data: List[float] = Field(..., description="Dependent variable values")
    model_name: str = Field(..., description="Name of built-in model to fit")
    initial_params: Optional[Dict[str, float]] = Field(None, description="Initial parameter guesses")
    bounds: Optional[Dict[str, Tuple[float, float]]] = Field(None, description="Parameter bounds (min, max)")
    x_label: str = Field("X", description="Label for x-axis")
    y_label: str = Field("Y", description="Label for y-axis")

class SuggestInitialRequest(BaseModel):
    x_data: List[float]
    y_data: List[float]
    model_name: str

class PredictRequest(BaseModel):
    model_name: str
    parameters: Dict[str, float]
    x_values: List[float]

# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/models")
async def list_models():
    """
    List all available nonlinear models with descriptions
    """
    return {
        "models": [
            {
                "name": name,
                "equation": info["equation"],
                "description": info["description"],
                "typical_use": info["typical_use"],
                "parameters": info["params"]
            }
            for name, info in BUILTIN_MODELS.items()
        ]
    }

@router.post("/suggest-initial")
async def suggest_initial_parameters(request: SuggestInitialRequest):
    """
    Suggest initial parameter values based on data characteristics
    Uses heuristics to provide reasonable starting points for optimization
    """
    try:
        x = np.array(request.x_data)
        y = np.array(request.y_data)

        if len(x) != len(y):
            raise ValueError("x_data and y_data must have the same length")

        if len(x) < 3:
            raise ValueError("Need at least 3 data points")

        model_info = BUILTIN_MODELS.get(request.model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {request.model_name}")

        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        # Calculate basic statistics
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        x_range = x_max - x_min
        y_range = y_max - y_min

        suggestions = {}

        if request.model_name == "exponential":
            # y = a * exp(b*x)
            # Use first and last points to estimate
            if y[0] > 0 and y[-1] > 0:
                suggestions["a"] = float(y[0])
                suggestions["b"] = float((np.log(y[-1]) - np.log(y[0])) / (x[-1] - x[0]))
            else:
                suggestions["a"] = float(np.abs(y[0]) + 1)
                suggestions["b"] = 0.1

        elif request.model_name == "logistic":
            # y = L / (1 + exp(-k*(x-x0)))
            suggestions["L"] = float(y_max * 1.1)  # Slightly above max
            suggestions["k"] = 1.0  # Default steepness
            suggestions["x0"] = float(np.median(x))  # Midpoint

        elif request.model_name == "michaelis_menten":
            # y = Vmax*x / (Km + x)
            suggestions["Vmax"] = float(y_max * 1.2)
            # Km is x value where y ≈ Vmax/2
            half_max_idx = np.argmin(np.abs(y - y_max/2))
            suggestions["Km"] = float(x[half_max_idx] if half_max_idx < len(x) else x_range/2)

        elif request.model_name == "power_law":
            # y = a * x^b
            # Use log-log linear regression
            x_pos = x[x > 0]
            y_pos = y[y > 0]
            if len(x_pos) >= 2:
                log_x = np.log(x_pos)
                log_y = np.log(y_pos)
                # Linear fit in log-log space
                coeffs = np.polyfit(log_x, log_y, 1)
                suggestions["b"] = float(coeffs[0])  # Slope
                suggestions["a"] = float(np.exp(coeffs[1]))  # Intercept
            else:
                suggestions["a"] = 1.0
                suggestions["b"] = 1.0

        elif request.model_name == "gompertz":
            # y = a * exp(-b * exp(-c*x))
            suggestions["a"] = float(y_max * 1.1)
            suggestions["b"] = 2.0
            suggestions["c"] = 0.1

        elif request.model_name == "weibull":
            # y = a * (1 - exp(-(x/b)^c))
            suggestions["a"] = float(y_max * 1.1)
            suggestions["b"] = float(x_range / 2)
            suggestions["c"] = 1.5

        elif request.model_name == "logarithmic":
            # y = a + b*ln(x)
            x_pos = x[x > 0]
            y_pos = y[y > 0]
            if len(x_pos) >= 2:
                log_x = np.log(x_pos)
                coeffs = np.polyfit(log_x, y_pos, 1)
                suggestions["b"] = float(coeffs[0])
                suggestions["a"] = float(coeffs[1])
            else:
                suggestions["a"] = float(y_min)
                suggestions["b"] = 1.0

        return {
            "suggested_parameters": suggestions,
            "model_name": request.model_name,
            "data_summary": {
                "n_points": int(len(x)),
                "x_range": [float(x_min), float(x_max)],
                "y_range": [float(y_min), float(y_max)]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/fit")
async def fit_nonlinear_model(request: NonlinearFitRequest):
    """
    Fit nonlinear regression model using Levenberg-Marquardt algorithm
    Returns parameter estimates, confidence intervals, and fit statistics
    """
    try:
        # Validate input
        x = np.array(request.x_data)
        y = np.array(request.y_data)

        if len(x) != len(y):
            raise ValueError("x_data and y_data must have the same length")

        if len(x) < 3:
            raise ValueError("Need at least 3 data points")

        # Get model info
        model_info = BUILTIN_MODELS.get(request.model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {request.model_name}. Available models: {list(BUILTIN_MODELS.keys())}")

        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        # Get initial parameters
        if request.initial_params:
            p0 = [request.initial_params.get(p, 1.0) for p in model_info["params"]]
        else:
            # Auto-suggest initial parameters
            suggest_response = await suggest_initial_parameters(
                SuggestInitialRequest(
                    x_data=x.tolist(),
                    y_data=y.tolist(),
                    model_name=request.model_name
                )
            )
            p0 = [suggest_response["suggested_parameters"][p] for p in model_info["params"]]

        # Extract bounds if provided
        bounds_tuple = (-np.inf, np.inf)
        if request.bounds:
            lower = [request.bounds.get(p, [-np.inf])[0] if isinstance(request.bounds.get(p, [-np.inf]), list) else request.bounds.get(p, (-np.inf, np.inf))[0] for p in model_info["params"]]
            upper = [request.bounds.get(p, [np.inf])[1] if isinstance(request.bounds.get(p, [np.inf]), list) else request.bounds.get(p, (-np.inf, np.inf))[1] for p in model_info["params"]]
            bounds_tuple = (lower, upper)

        # Fit model using Levenberg-Marquardt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                model_info["function"],
                x, y,
                p0=p0,
                bounds=bounds_tuple,
                maxfev=10000
            )

        # Calculate fitted values and residuals
        y_fitted = model_info["function"](x, *popt)
        residuals = y - y_fitted

        # Calculate statistics
        n = len(y)
        p = len(popt)

        # Sum of squares
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # R-squared
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Adjusted R-squared
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared

        # RMSE
        rmse = np.sqrt(ss_res / (n - p)) if n > p else np.sqrt(ss_res / n)

        # Mean Absolute Error
        mae = np.mean(np.abs(residuals))

        # AIC and BIC
        if ss_res > 0:
            aic = n * np.log(ss_res / n) + 2 * p
            bic = n * np.log(ss_res / n) + p * np.log(n)
        else:
            aic = -np.inf
            bic = -np.inf

        # Parameter standard errors and confidence intervals
        perr = np.sqrt(np.diag(pcov))
        alpha = 0.05
        t_val = stats.t.ppf(1 - alpha/2, n - p) if n > p else 1.96

        param_results = {}
        for i, param in enumerate(model_info["params"]):
            param_results[param] = {
                "estimate": float(popt[i]),
                "std_error": float(perr[i]),
                "ci_lower": float(popt[i] - t_val * perr[i]),
                "ci_upper": float(popt[i] + t_val * perr[i]),
                "t_statistic": float(popt[i] / perr[i]) if perr[i] > 0 else np.inf,
                "p_value": float(2 * (1 - stats.t.cdf(abs(popt[i] / perr[i]), n - p))) if perr[i] > 0 else 0.0
            }

        # Generate prediction points for smooth curve
        x_sorted_idx = np.argsort(x)
        x_sorted = x[x_sorted_idx]
        x_pred = np.linspace(x_sorted[0], x_sorted[-1], 200)
        y_pred = model_info["function"](x_pred, *popt)

        return {
            "success": True,
            "model": request.model_name,
            "equation": model_info["equation"],
            "parameters": param_results,
            "statistics": {
                "r_squared": float(r_squared),
                "adj_r_squared": float(adj_r_squared),
                "rmse": float(rmse),
                "mae": float(mae),
                "aic": float(aic),
                "bic": float(bic),
                "n_observations": int(n),
                "n_parameters": int(p),
                "degrees_of_freedom": int(n - p)
            },
            "fitted_values": y_fitted.tolist(),
            "residuals": residuals.tolist(),
            "x_data": x.tolist(),
            "y_data": y.tolist(),
            "prediction_curve": {
                "x": x_pred.tolist(),
                "y": y_pred.tolist()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model fitting failed: {str(e)}")

@router.post("/predict")
async def predict_from_model(request: PredictRequest):
    """
    Predict y values for given x values using fitted model parameters
    """
    try:
        model_info = BUILTIN_MODELS.get(request.model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {request.model_name}")

        # Extract parameter values in correct order
        params = [request.parameters[p] for p in model_info["params"]]

        # Generate predictions
        x = np.array(request.x_values)
        y_pred = model_info["function"](x, *params)

        return {
            "x_values": x.tolist(),
            "y_predictions": y_pred.tolist(),
            "model": request.model_name,
            "parameters_used": request.parameters
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
