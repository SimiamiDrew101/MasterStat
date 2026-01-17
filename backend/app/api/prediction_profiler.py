from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

router = APIRouter()

class ModelImportRequest(BaseModel):
    """Import existing RSM/Factorial model"""
    model_type: str = Field(..., description="rsm_quadratic, rsm_cubic, factorial")
    coefficients: Dict[str, float]
    factors: List[str]
    factor_ranges: Dict[str, Tuple[float, float]]
    response_name: str

class PredictionRequest(BaseModel):
    """Request prediction at specific factor levels"""
    model: ModelImportRequest
    factor_levels: Dict[str, float]

class DesirabilityGoal(BaseModel):
    """Desirability goal specification"""
    response: str
    target_type: str  # 'maximize', 'minimize', 'target'
    low: Optional[float] = None
    high: Optional[float] = None
    target: Optional[float] = None
    tolerance: Optional[float] = None

class DesirabilityRequest(BaseModel):
    """Multi-response optimization with desirability"""
    models: List[ModelImportRequest]
    goals: List[DesirabilityGoal]
    factor_ranges: Dict[str, Tuple[float, float]]

class SurfaceRequest(BaseModel):
    """Request for response surface generation"""
    model: ModelImportRequest
    factor1: str
    factor2: str
    factor_ranges: Dict[str, Tuple[float, float]]
    fixed_factors: Dict[str, float]

@router.post("/predict")
async def predict_response(request: PredictionRequest):
    """
    Predict response at given factor levels with confidence intervals
    """
    try:
        # Build polynomial from coefficients
        y_pred = evaluate_model(
            request.model.coefficients,
            request.factor_levels
        )

        # Calculate prediction interval (simplified - assumes constant variance)
        # In production, this would use model variance from fit
        std_error = estimate_prediction_se(request.model, request.factor_levels)
        ci_lower = y_pred - 1.96 * std_error
        ci_upper = y_pred + 1.96 * std_error

        return {
            "prediction": float(y_pred),
            "confidence_interval": {
                "lower": float(ci_lower),
                "upper": float(ci_upper),
                "level": 0.95
            },
            "factor_levels": request.factor_levels
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/optimize-desirability")
async def optimize_desirability(request: DesirabilityRequest):
    """
    Find optimal factor settings using desirability functions
    """
    try:
        # Define desirability objective function
        def objective(x):
            # x is vector of factor levels
            factor_dict = dict(zip(request.models[0].factors, x))

            # Calculate desirability for each response
            desirabilities = []
            for model, goal in zip(request.models, request.goals):
                y_pred = evaluate_model(model.coefficients, factor_dict)
                d = calculate_desirability(y_pred, goal)
                desirabilities.append(d)

            # Overall desirability (geometric mean)
            overall_d = np.prod(desirabilities) ** (1/len(desirabilities))
            return -overall_d  # Minimize negative = maximize

        # Bounds for each factor
        bounds = [request.factor_ranges[f] for f in request.models[0].factors]

        # Initial guess (center point)
        x0 = [(b[0] + b[1]) / 2 for b in bounds]

        # Optimize
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

        optimal_levels = dict(zip(request.models[0].factors, result.x))

        # Predict all responses at optimal point
        predictions = []
        for model in request.models:
            y_pred = evaluate_model(model.coefficients, optimal_levels)
            predictions.append({
                "response": model.response_name,
                "predicted_value": float(y_pred)
            })

        return {
            "optimal_settings": optimal_levels,
            "predictions": predictions,
            "overall_desirability": float(-result.fun),
            "success": result.success
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/generate-surface")
async def generate_response_surface(request: SurfaceRequest):
    """
    Generate 2D surface data for contour plots
    """
    try:
        # Create grid for two factors
        x1_range = np.linspace(*request.factor_ranges[request.factor1], 50)
        x2_range = np.linspace(*request.factor_ranges[request.factor2], 50)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        # Evaluate model at each grid point
        Z = np.zeros_like(X1)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                factor_levels = {
                    request.factor1: X1[i, j],
                    request.factor2: X2[i, j],
                    **request.fixed_factors  # Hold other factors constant
                }
                Z[i, j] = evaluate_model(request.model.coefficients, factor_levels)

        return {
            "x": x1_range.tolist(),
            "y": x2_range.tolist(),
            "z": Z.tolist(),
            "factor1": request.factor1,
            "factor2": request.factor2
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Helper functions
def evaluate_model(coefficients: Dict, factor_levels: Dict) -> float:
    """Evaluate polynomial model at given factor levels"""
    # Handle intercept, linear, quadratic, interaction terms
    y = coefficients.get('Intercept', 0)

    # Linear terms
    for factor, value in factor_levels.items():
        y += coefficients.get(factor, 0) * value

    # Quadratic terms
    for factor, value in factor_levels.items():
        y += coefficients.get(f'{factor}^2', 0) * value**2

    # Interaction terms
    factors = list(factor_levels.keys())
    for i, f1 in enumerate(factors):
        for f2 in factors[i+1:]:
            interaction_key = f'{f1}*{f2}'
            y += coefficients.get(interaction_key, 0) * factor_levels[f1] * factor_levels[f2]

    return y

def estimate_prediction_se(model: ModelImportRequest, factor_levels: Dict) -> float:
    """
    Estimate standard error of prediction (simplified)
    In production, this would use the actual variance-covariance matrix from model fit
    """
    # Simplified estimate - assumes 5% relative error
    y_pred = evaluate_model(model.coefficients, factor_levels)
    return abs(y_pred * 0.05)

def calculate_desirability(value: float, goal: DesirabilityGoal) -> float:
    """Calculate desirability (0 to 1) based on goal"""
    target_type = goal.target_type

    if target_type == 'maximize':
        low, high = goal.low, goal.high
        if value <= low:
            return 0.0
        elif value >= high:
            return 1.0
        else:
            return ((value - low) / (high - low)) ** 2

    elif target_type == 'minimize':
        low, high = goal.low, goal.high
        if value >= high:
            return 0.0
        elif value <= low:
            return 1.0
        else:
            return ((high - value) / (high - low)) ** 2

    else:  # target
        target, tolerance = goal.target, goal.tolerance
        if abs(value - target) >= tolerance:
            return 0.0
        else:
            return 1.0 - (abs(value - target) / tolerance) ** 2
