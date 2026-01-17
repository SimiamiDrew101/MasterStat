from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional
import numpy as np
from app.algorithms.coordinate_exchange import CoordinateExchange, evaluate_design_efficiency

router = APIRouter()


class OptimalDesignRequest(BaseModel):
    """Request for optimal design generation"""
    n_runs: int = Field(..., description="Number of experimental runs", ge=3)
    factors: List[str] = Field(..., description="Factor names", min_items=1)
    factor_ranges: Dict[str, Tuple[float, float]] = Field(..., description="Factor ranges (min, max)")
    model_order: int = Field(2, description="Model order: 1=linear, 2=quadratic", ge=1, le=2)
    criterion: str = Field("d_optimal", description="Optimality criterion: d_optimal, i_optimal, a_optimal")
    max_iterations: int = Field(1000, description="Maximum iterations", ge=10, le=10000)
    n_candidates: int = Field(20, description="Number of candidate points per coordinate", ge=5, le=100)


class EvaluateDesignRequest(BaseModel):
    """Request for design evaluation"""
    design: List[Dict[str, float]] = Field(..., description="Design matrix as list of dicts")
    factors: List[str] = Field(..., description="Factor names")
    factor_ranges: Dict[str, Tuple[float, float]] = Field(..., description="Factor ranges")
    model_order: int = Field(2, description="Model order", ge=1, le=2)


@router.post("/generate")
async def generate_optimal_design(request: OptimalDesignRequest):
    """
    Generate optimal experimental design using coordinate exchange algorithm.

    Supports three optimality criteria:
    - D-optimal: Maximizes |X'X| (minimizes confidence ellipsoid volume)
    - I-optimal: Minimizes average prediction variance
    - A-optimal: Minimizes trace((X'X)^-1) (average parameter variance)
    """
    try:
        # Validate factor ranges
        for factor in request.factors:
            if factor not in request.factor_ranges:
                raise ValueError(f"Missing range for factor: {factor}")

            low, high = request.factor_ranges[factor]
            if low >= high:
                raise ValueError(f"Invalid range for {factor}: low must be < high")

        # Create coordinate exchange instance
        ce = CoordinateExchange(
            n_runs=request.n_runs,
            factors=request.factors,
            factor_ranges=request.factor_ranges,
            model_order=request.model_order,
            n_candidates=request.n_candidates
        )

        # Generate design based on criterion
        if request.criterion == "d_optimal":
            design = ce.generate_d_optimal(max_iterations=request.max_iterations)
        elif request.criterion == "i_optimal":
            design = ce.generate_i_optimal(max_iterations=request.max_iterations)
        elif request.criterion == "a_optimal":
            design = ce.generate_a_optimal(max_iterations=request.max_iterations)
        else:
            raise ValueError(f"Unknown criterion: {request.criterion}. Use 'd_optimal', 'i_optimal', or 'a_optimal'")

        # Evaluate design efficiency
        efficiency = evaluate_design_efficiency(
            design,
            request.factor_ranges,
            request.factors,
            request.model_order
        )

        # Convert to list of dicts for JSON response
        design_table = []
        for i, row in enumerate(design):
            run = {"Run": i + 1}
            for j, factor in enumerate(request.factors):
                run[factor] = round(float(row[j]), 6)
            design_table.append(run)

        return {
            "design": design_table,
            "efficiency": efficiency,
            "criterion": request.criterion,
            "n_runs": request.n_runs,
            "n_factors": len(request.factors),
            "model_order": request.model_order,
            "factors": request.factors,
            "factor_ranges": request.factor_ranges
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/evaluate")
async def evaluate_design(request: EvaluateDesignRequest):
    """
    Evaluate efficiency of a user-provided design.

    Calculates D-efficiency, A-criterion, G-efficiency, condition number,
    and variance inflation factors (VIFs).
    """
    try:
        # Convert design from list of dicts to numpy array
        design_array = np.array([
            [row[f] for f in request.factors]
            for row in request.design
        ])

        # Validate design dimensions
        if design_array.shape[1] != len(request.factors):
            raise ValueError(f"Design has {design_array.shape[1]} columns but {len(request.factors)} factors specified")

        # Evaluate efficiency
        efficiency = evaluate_design_efficiency(
            design_array,
            request.factor_ranges,
            request.factors,
            request.model_order
        )

        return {
            "efficiency": efficiency,
            "n_runs": design_array.shape[0],
            "n_factors": design_array.shape[1],
            "model_order": request.model_order
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/compare")
async def compare_designs(designs: List[EvaluateDesignRequest]):
    """
    Compare efficiency of multiple designs.

    Returns efficiency metrics for each design with ranking.
    """
    try:
        if len(designs) < 2:
            raise ValueError("At least 2 designs required for comparison")

        results = []
        for idx, design_req in enumerate(designs):
            # Convert to array
            design_array = np.array([
                [row[f] for f in design_req.factors]
                for row in design_req.design
            ])

            # Evaluate
            efficiency = evaluate_design_efficiency(
                design_array,
                design_req.factor_ranges,
                design_req.factors,
                design_req.model_order
            )

            results.append({
                "design_id": idx + 1,
                "efficiency": efficiency
            })

        # Rank by D-efficiency (higher is better)
        ranked = sorted(results, key=lambda x: x["efficiency"]["d_efficiency"], reverse=True)
        for i, result in enumerate(ranked):
            result["rank"] = i + 1

        return {
            "designs": ranked,
            "best_design_id": ranked[0]["design_id"],
            "n_designs_compared": len(designs)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/info")
async def get_optimal_design_info():
    """
    Get information about optimal design types and efficiency metrics.
    """
    return {
        "optimality_criteria": {
            "d_optimal": {
                "name": "D-Optimal",
                "description": "Maximizes the determinant of X'X (information matrix)",
                "minimizes": "Volume of the confidence ellipsoid for parameter estimates",
                "best_for": "Parameter estimation with maximum precision",
                "metric": "D-efficiency = |X'X|^(1/p) / n"
            },
            "i_optimal": {
                "name": "I-Optimal",
                "description": "Minimizes average prediction variance across design space",
                "minimizes": "Average variance of predictions",
                "best_for": "Prediction accuracy across the entire design region",
                "metric": "Average of x'(X'X)^-1 x over prediction points"
            },
            "a_optimal": {
                "name": "A-Optimal",
                "description": "Minimizes trace of (X'X)^-1",
                "minimizes": "Average variance of parameter estimates",
                "best_for": "Overall parameter estimation accuracy",
                "metric": "A-criterion = trace((X'X)^-1)"
            }
        },
        "efficiency_metrics": {
            "d_efficiency": "Relative efficiency compared to orthogonal design (0-1)",
            "d_criterion": "Determinant of information matrix |X'X|",
            "a_criterion": "Trace of inverse information matrix tr((X'X)^-1)",
            "g_efficiency": "Reciprocal of maximum prediction variance",
            "condition_number": "Ratio of largest to smallest eigenvalue (lower is better, <10 is good)",
            "vif": "Variance Inflation Factors for each parameter (<10 is good, <5 is excellent)"
        },
        "model_orders": {
            "1": "Linear model: y = β0 + Σβi*xi",
            "2": "Quadratic model: y = β0 + Σβi*xi + Σβii*xi² + ΣΣβij*xi*xj"
        }
    }
