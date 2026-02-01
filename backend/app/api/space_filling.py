"""
Space-Filling Designs API Module

Provides endpoints for generating space-filling experimental designs
used in computer experiments, simulation studies, and surrogate modeling.

Design Types:
- Latin Hypercube Sampling (LHS)
- Sobol Sequences (quasi-random)
- Halton Sequences (quasi-random)
- Maximum Projection Designs
- Uniform Random Sampling
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import pdist, cdist
import json

router = APIRouter(prefix="/api/space-filling", tags=["Space-Filling Designs"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class FactorSpec(BaseModel):
    name: str = Field(..., description="Factor name")
    low: float = Field(..., description="Lower bound")
    high: float = Field(..., description="Upper bound")
    type: str = Field(default="continuous", description="Factor type: continuous or discrete")


class SpaceFillingRequest(BaseModel):
    factors: List[FactorSpec] = Field(..., description="Factor specifications")
    n_points: int = Field(..., ge=2, description="Number of design points")
    method: str = Field(default="lhs", description="Method: lhs, sobol, halton, uniform, maximin")
    optimization: Optional[str] = Field(default=None, description="LHS optimization: None, maximin, correlation, centermaximin")
    scramble: bool = Field(default=True, description="Scramble quasi-random sequences")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class AugmentRequest(BaseModel):
    existing_design: List[Dict[str, float]] = Field(..., description="Existing design points")
    factors: List[FactorSpec] = Field(..., description="Factor specifications")
    n_new_points: int = Field(..., ge=1, description="Number of new points to add")
    method: str = Field(default="maximin", description="Method for augmentation")
    seed: Optional[int] = Field(default=None, description="Random seed")


class EvaluateRequest(BaseModel):
    design: List[Dict[str, float]] = Field(..., description="Design to evaluate")
    factors: List[FactorSpec] = Field(..., description="Factor specifications")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_float(val):
    """Convert numpy types to Python floats for JSON serialization"""
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
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
    return obj


def scale_to_bounds(design_unit: np.ndarray, factors: List[FactorSpec]) -> np.ndarray:
    """Scale unit hypercube design to actual factor bounds"""
    scaled = np.zeros_like(design_unit)
    for i, factor in enumerate(factors):
        scaled[:, i] = factor.low + design_unit[:, i] * (factor.high - factor.low)
    return scaled


def compute_design_metrics(design: np.ndarray) -> Dict[str, float]:
    """Compute space-filling quality metrics for a design"""
    n, d = design.shape

    # Minimum distance (maximin criterion)
    if n > 1:
        distances = pdist(design)
        min_dist = np.min(distances)
        mean_dist = np.mean(distances)
        max_dist = np.max(distances)
    else:
        min_dist = mean_dist = max_dist = 0.0

    # Coverage metric (average nearest neighbor distance)
    if n > 1:
        dist_matrix = cdist(design, design)
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_neighbor_dists = np.min(dist_matrix, axis=1)
        coverage = np.mean(nearest_neighbor_dists)
    else:
        coverage = 0.0

    # Discrepancy (measure of uniformity) - simplified centered L2 discrepancy
    # For unit cube
    design_unit = (design - design.min(axis=0)) / (design.max(axis=0) - design.min(axis=0) + 1e-10)
    discrepancy = qmc.discrepancy(design_unit) if n > 1 else 0.0

    # Projection uniformity (1D projections)
    projection_scores = []
    for col in range(d):
        sorted_vals = np.sort(design_unit[:, col])
        ideal_spacing = 1.0 / n
        actual_spacings = np.diff(sorted_vals)
        if len(actual_spacings) > 0:
            uniformity = 1.0 - np.std(actual_spacings) / (ideal_spacing + 1e-10)
            projection_scores.append(max(0, uniformity))

    projection_uniformity = np.mean(projection_scores) if projection_scores else 0.0

    return {
        "n_points": int(n),
        "n_factors": int(d),
        "min_distance": float(min_dist),
        "mean_distance": float(mean_dist),
        "max_distance": float(max_dist),
        "coverage": float(coverage),
        "discrepancy": float(discrepancy),
        "projection_uniformity": float(projection_uniformity),
        "space_filling_score": float((1 - discrepancy) * 50 + projection_uniformity * 50)
    }


def optimize_lhs(design: np.ndarray, criterion: str = "maximin", iterations: int = 1000) -> np.ndarray:
    """Optimize LHS design using specified criterion"""
    n, d = design.shape
    best_design = design.copy()

    if criterion == "maximin":
        best_score = np.min(pdist(best_design))

        for _ in range(iterations):
            # Random column and two rows to swap
            col = np.random.randint(d)
            i, j = np.random.choice(n, 2, replace=False)

            # Try swap
            new_design = best_design.copy()
            new_design[i, col], new_design[j, col] = new_design[j, col], new_design[i, col]

            new_score = np.min(pdist(new_design))
            if new_score > best_score:
                best_design = new_design
                best_score = new_score

    elif criterion == "correlation":
        # Minimize maximum absolute correlation between columns
        best_score = np.max(np.abs(np.corrcoef(best_design.T) - np.eye(d)))

        for _ in range(iterations):
            col = np.random.randint(d)
            i, j = np.random.choice(n, 2, replace=False)

            new_design = best_design.copy()
            new_design[i, col], new_design[j, col] = new_design[j, col], new_design[i, col]

            corr_matrix = np.corrcoef(new_design.T)
            new_score = np.max(np.abs(corr_matrix - np.eye(d)))
            if new_score < best_score:
                best_design = new_design
                best_score = new_score

    elif criterion == "centermaximin":
        # Maximin with centered points
        centered = best_design - 0.5
        best_score = np.min(pdist(centered))

        for _ in range(iterations):
            col = np.random.randint(d)
            i, j = np.random.choice(n, 2, replace=False)

            new_design = best_design.copy()
            new_design[i, col], new_design[j, col] = new_design[j, col], new_design[i, col]

            centered = new_design - 0.5
            new_score = np.min(pdist(centered))
            if new_score > best_score:
                best_design = new_design
                best_score = new_score

    return best_design


# ============================================================================
# MAIN ENDPOINTS
# ============================================================================

@router.post("/generate")
async def generate_space_filling_design(request: SpaceFillingRequest):
    """
    Generate a space-filling experimental design.

    Methods:
    - lhs: Latin Hypercube Sampling
    - sobol: Sobol quasi-random sequence
    - halton: Halton quasi-random sequence
    - uniform: Uniform random sampling
    - maximin: Maximin distance design (optimized LHS)
    """
    try:
        n_factors = len(request.factors)
        n_points = request.n_points

        # Set random seed if provided
        if request.seed is not None:
            np.random.seed(request.seed)
            rng = np.random.default_rng(request.seed)
        else:
            rng = np.random.default_rng()

        # Generate design in unit hypercube
        if request.method == "lhs":
            sampler = qmc.LatinHypercube(d=n_factors, seed=rng)
            design_unit = sampler.random(n=n_points)

            # Apply optimization if requested
            if request.optimization:
                design_unit = optimize_lhs(design_unit, request.optimization)

        elif request.method == "sobol":
            sampler = qmc.Sobol(d=n_factors, scramble=request.scramble, seed=rng)
            design_unit = sampler.random(n=n_points)

        elif request.method == "halton":
            sampler = qmc.Halton(d=n_factors, scramble=request.scramble, seed=rng)
            design_unit = sampler.random(n=n_points)

        elif request.method == "uniform":
            design_unit = rng.random((n_points, n_factors))

        elif request.method == "maximin":
            # Generate optimized maximin LHS
            sampler = qmc.LatinHypercube(d=n_factors, seed=rng)
            design_unit = sampler.random(n=n_points)
            design_unit = optimize_lhs(design_unit, "maximin", iterations=2000)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")

        # Scale to actual bounds
        design_scaled = scale_to_bounds(design_unit, request.factors)

        # Build design matrix with factor names
        factor_names = [f.name for f in request.factors]
        design_matrix = []
        for i, row in enumerate(design_scaled):
            point = {"Run": i + 1}
            for j, name in enumerate(factor_names):
                point[name] = float(row[j])
            design_matrix.append(point)

        # Compute quality metrics
        metrics = compute_design_metrics(design_scaled)

        # Method descriptions
        method_descriptions = {
            "lhs": "Latin Hypercube Sampling ensures each factor level appears exactly once in each row and column of the design grid.",
            "sobol": "Sobol sequences are quasi-random low-discrepancy sequences that provide excellent uniformity in high dimensions.",
            "halton": "Halton sequences use prime-based digit scrambling to generate well-distributed points.",
            "uniform": "Uniform random sampling provides independent random points without structure.",
            "maximin": "Maximin LHS maximizes the minimum distance between any two points for optimal space coverage."
        }

        return make_json_safe({
            "design_matrix": design_matrix,
            "design_unit": design_unit.tolist(),
            "factor_names": factor_names,
            "factors": [{"name": f.name, "low": f.low, "high": f.high, "type": f.type} for f in request.factors],
            "n_points": n_points,
            "n_factors": n_factors,
            "method": request.method,
            "optimization": request.optimization,
            "metrics": metrics,
            "method_description": method_descriptions.get(request.method, ""),
            "interpretation": {
                "discrepancy": "Lower is better (0 = perfect uniformity)",
                "min_distance": "Higher is better for space-filling",
                "projection_uniformity": "Higher is better (1 = perfect 1D projections)",
                "space_filling_score": f"Overall quality: {metrics['space_filling_score']:.1f}/100"
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/augment")
async def augment_design(request: AugmentRequest):
    """
    Add new points to an existing design while maintaining space-filling properties.
    Uses maximin criterion to place new points far from existing ones.
    """
    try:
        n_factors = len(request.factors)
        factor_names = [f.name for f in request.factors]

        # Convert existing design to array
        existing_array = np.array([
            [point[name] for name in factor_names]
            for point in request.existing_design
        ])

        # Normalize to unit cube for distance calculations
        bounds_low = np.array([f.low for f in request.factors])
        bounds_high = np.array([f.high for f in request.factors])
        bounds_range = bounds_high - bounds_low

        existing_unit = (existing_array - bounds_low) / bounds_range

        # Set random seed
        if request.seed is not None:
            rng = np.random.default_rng(request.seed)
        else:
            rng = np.random.default_rng()

        # Generate candidate points
        n_candidates = max(1000, request.n_new_points * 100)
        candidates = rng.random((n_candidates, n_factors))

        # Select new points using maximin criterion
        new_points_unit = []
        all_points = existing_unit.copy()

        for _ in range(request.n_new_points):
            # Compute minimum distance from each candidate to all existing/selected points
            if len(all_points) > 0:
                distances = cdist(candidates, all_points)
                min_distances = np.min(distances, axis=1)
            else:
                min_distances = np.ones(len(candidates))

            # Select candidate with maximum minimum distance
            best_idx = np.argmax(min_distances)
            new_point = candidates[best_idx]

            new_points_unit.append(new_point)
            all_points = np.vstack([all_points, new_point])

            # Remove selected candidate
            candidates = np.delete(candidates, best_idx, axis=0)

        new_points_unit = np.array(new_points_unit)

        # Scale back to original bounds
        new_points_scaled = new_points_unit * bounds_range + bounds_low

        # Build new design points
        new_design_points = []
        start_run = len(request.existing_design) + 1
        for i, row in enumerate(new_points_scaled):
            point = {"Run": start_run + i}
            for j, name in enumerate(factor_names):
                point[name] = float(row[j])
            new_design_points.append(point)

        # Combined design
        combined_design = request.existing_design + new_design_points

        # Compute metrics for combined design
        combined_array = np.vstack([existing_array, new_points_scaled])
        metrics = compute_design_metrics(combined_array)

        return make_json_safe({
            "new_points": new_design_points,
            "combined_design": combined_design,
            "n_existing": len(request.existing_design),
            "n_new": request.n_new_points,
            "n_total": len(combined_design),
            "metrics": metrics,
            "message": f"Added {request.n_new_points} new points using maximin criterion"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate_design(request: EvaluateRequest):
    """
    Evaluate the space-filling properties of an existing design.
    """
    try:
        factor_names = [f.name for f in request.factors]

        # Convert to array
        design_array = np.array([
            [point[name] for name in factor_names]
            for point in request.design
        ])

        # Compute comprehensive metrics
        metrics = compute_design_metrics(design_array)

        # Compute pairwise distances for visualization
        if len(design_array) > 1:
            distances = pdist(design_array)
            distance_histogram = np.histogram(distances, bins=20)
        else:
            distance_histogram = ([], [])

        # Compute 1D projections
        projections = {}
        for i, name in enumerate(factor_names):
            values = design_array[:, i]
            projections[name] = {
                "values": values.tolist(),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }

        # Compute 2D projections (for pairs of factors)
        projections_2d = []
        if len(factor_names) >= 2:
            for i in range(min(len(factor_names), 4)):
                for j in range(i + 1, min(len(factor_names), 4)):
                    projections_2d.append({
                        "x_factor": factor_names[i],
                        "y_factor": factor_names[j],
                        "x_values": design_array[:, i].tolist(),
                        "y_values": design_array[:, j].tolist()
                    })

        # Quality assessment
        score = metrics["space_filling_score"]
        if score >= 80:
            quality = "Excellent"
            recommendation = "This design has excellent space-filling properties."
        elif score >= 60:
            quality = "Good"
            recommendation = "This design has good coverage. Consider optimization for critical applications."
        elif score >= 40:
            quality = "Moderate"
            recommendation = "Consider using an optimized LHS or Sobol sequence for better coverage."
        else:
            quality = "Poor"
            recommendation = "This design has poor space-filling properties. Use a structured method like LHS or Sobol."

        return make_json_safe({
            "metrics": metrics,
            "quality_assessment": {
                "overall_quality": quality,
                "score": score,
                "recommendation": recommendation
            },
            "projections_1d": projections,
            "projections_2d": projections_2d,
            "distance_distribution": {
                "counts": distance_histogram[0].tolist() if len(distance_histogram[0]) > 0 else [],
                "bin_edges": distance_histogram[1].tolist() if len(distance_histogram[1]) > 0 else []
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_methods(request: SpaceFillingRequest):
    """
    Generate designs using multiple methods and compare their properties.
    Useful for selecting the best method for a given application.
    """
    try:
        methods = ["lhs", "sobol", "halton", "maximin"]
        results = {}

        for method in methods:
            # Set same seed for fair comparison
            if request.seed is not None:
                rng = np.random.default_rng(request.seed)
            else:
                rng = np.random.default_rng(42)  # Default seed for reproducibility

            n_factors = len(request.factors)
            n_points = request.n_points

            if method == "lhs":
                sampler = qmc.LatinHypercube(d=n_factors, seed=rng)
                design_unit = sampler.random(n=n_points)
            elif method == "sobol":
                sampler = qmc.Sobol(d=n_factors, scramble=True, seed=rng)
                design_unit = sampler.random(n=n_points)
            elif method == "halton":
                sampler = qmc.Halton(d=n_factors, scramble=True, seed=rng)
                design_unit = sampler.random(n=n_points)
            elif method == "maximin":
                sampler = qmc.LatinHypercube(d=n_factors, seed=rng)
                design_unit = sampler.random(n=n_points)
                design_unit = optimize_lhs(design_unit, "maximin", iterations=1000)

            # Scale to bounds
            design_scaled = scale_to_bounds(design_unit, request.factors)

            # Compute metrics
            metrics = compute_design_metrics(design_scaled)

            results[method] = {
                "metrics": metrics,
                "design_unit": design_unit.tolist()
            }

        # Rank methods
        rankings = {
            "discrepancy": sorted(methods, key=lambda m: results[m]["metrics"]["discrepancy"]),
            "min_distance": sorted(methods, key=lambda m: results[m]["metrics"]["min_distance"], reverse=True),
            "projection_uniformity": sorted(methods, key=lambda m: results[m]["metrics"]["projection_uniformity"], reverse=True),
            "overall": sorted(methods, key=lambda m: results[m]["metrics"]["space_filling_score"], reverse=True)
        }

        # Recommendations
        best_overall = rankings["overall"][0]
        recommendations = {
            "best_overall": best_overall,
            "best_for_uniformity": rankings["discrepancy"][0],
            "best_for_coverage": rankings["min_distance"][0],
            "recommendation": f"For {n_factors} factors and {n_points} points, {best_overall.upper()} provides the best balance of space-filling properties."
        }

        return make_json_safe({
            "comparison": results,
            "rankings": rankings,
            "recommendations": recommendations,
            "factors": [{"name": f.name, "low": f.low, "high": f.high} for f in request.factors],
            "n_points": n_points,
            "n_factors": n_factors
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods")
async def get_available_methods():
    """
    Get information about available space-filling design methods.
    """
    return {
        "methods": [
            {
                "id": "lhs",
                "name": "Latin Hypercube Sampling",
                "description": "Stratified sampling ensuring each factor level appears once per stratum. Good balance of uniformity and randomness.",
                "best_for": "General-purpose computer experiments",
                "optimization_options": ["None", "maximin", "correlation", "centermaximin"]
            },
            {
                "id": "sobol",
                "name": "Sobol Sequence",
                "description": "Quasi-random low-discrepancy sequence with excellent uniformity properties, especially in high dimensions.",
                "best_for": "High-dimensional problems, integration, sensitivity analysis",
                "optimization_options": []
            },
            {
                "id": "halton",
                "name": "Halton Sequence",
                "description": "Quasi-random sequence using prime bases. Good uniformity but may show correlations in very high dimensions.",
                "best_for": "Moderate-dimensional problems, sequential sampling",
                "optimization_options": []
            },
            {
                "id": "maximin",
                "name": "Maximin LHS",
                "description": "Latin Hypercube optimized to maximize the minimum distance between points. Best coverage guarantee.",
                "best_for": "Surrogate modeling, Gaussian process fitting",
                "optimization_options": []
            },
            {
                "id": "uniform",
                "name": "Uniform Random",
                "description": "Independent uniform random sampling. Simple but may have clustering.",
                "best_for": "Monte Carlo simulation, baseline comparison",
                "optimization_options": []
            }
        ],
        "metrics": [
            {
                "id": "discrepancy",
                "name": "Discrepancy",
                "description": "Measures deviation from perfect uniformity. Lower is better.",
                "range": "[0, 1]"
            },
            {
                "id": "min_distance",
                "name": "Minimum Distance",
                "description": "Smallest distance between any two points. Higher indicates better coverage.",
                "range": "[0, sqrt(d)]"
            },
            {
                "id": "projection_uniformity",
                "name": "Projection Uniformity",
                "description": "How uniform the 1D projections are. Higher is better.",
                "range": "[0, 1]"
            },
            {
                "id": "space_filling_score",
                "name": "Space-Filling Score",
                "description": "Overall quality score combining multiple metrics.",
                "range": "[0, 100]"
            }
        ]
    }
