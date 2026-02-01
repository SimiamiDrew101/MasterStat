"""
Mixture Design Module - Advanced Mixture Experiments

Provides:
- Extreme Vertices designs for constrained mixture regions
- Mixture + Process factor combined experiments
- Ternary contour data generation
- Component trace plots
- Scheffé polynomial models with constraints

Advanced Mixture Designs Feature
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import ConvexHull
from itertools import combinations, product
import math

router = APIRouter(prefix="/api/mixture", tags=["Mixture Designs"])


# ==================== Request/Response Models ====================

class ComponentConstraint(BaseModel):
    """Constraint on a mixture component"""
    name: str = Field(..., description="Component name")
    min_prop: float = Field(0.0, ge=0.0, le=1.0, description="Minimum proportion")
    max_prop: float = Field(1.0, ge=0.0, le=1.0, description="Maximum proportion")


class ExtremeVerticesRequest(BaseModel):
    """Request for extreme vertices design"""
    components: List[ComponentConstraint] = Field(..., description="Component constraints")
    include_centroids: bool = Field(True, description="Include centroid points")
    include_axial: bool = Field(False, description="Include axial check blends")
    n_center_points: int = Field(1, ge=0, le=5, description="Number of overall centroid replicates")


class MixtureProcessRequest(BaseModel):
    """Request for mixture + process factor design"""
    n_components: int = Field(..., ge=2, le=10, description="Number of mixture components")
    component_names: Optional[List[str]] = None
    mixture_design_type: str = Field("simplex-centroid", description="simplex-lattice or simplex-centroid")
    lattice_degree: int = Field(2, ge=2, le=4, description="Degree for simplex-lattice")
    process_factors: List[Dict[str, Any]] = Field(..., description="Process factors with levels")
    process_design_type: str = Field("full-factorial", description="full-factorial or fractional")


class TernaryContourRequest(BaseModel):
    """Request for ternary contour data"""
    component_names: List[str] = Field(..., min_length=3, max_length=3, description="Names of 3 components")
    model_coefficients: Dict[str, float] = Field(..., description="Scheffé model coefficients")
    model_type: str = Field("quadratic", description="linear, quadratic, or cubic")
    grid_resolution: int = Field(50, ge=20, le=100, description="Grid resolution for contour")
    constraints: Optional[List[ComponentConstraint]] = None


class TracePlotRequest(BaseModel):
    """Request for trace plot (Cox direction) data"""
    component_names: List[str] = Field(..., description="Names of components")
    reference_blend: List[float] = Field(..., description="Reference blend proportions")
    model_coefficients: Dict[str, float] = Field(..., description="Scheffé model coefficients")
    model_type: str = Field("quadratic", description="linear, quadratic, or cubic")
    n_points: int = Field(50, ge=20, le=100, description="Number of points per trace")


class AugmentDesignRequest(BaseModel):
    """Request to augment existing mixture design"""
    existing_design: List[List[float]] = Field(..., description="Existing design points")
    component_names: List[str] = Field(..., description="Component names")
    n_augment_points: int = Field(5, ge=1, le=20, description="Number of points to add")
    criterion: str = Field("d-optimal", description="d-optimal or i-optimal")
    constraints: Optional[List[ComponentConstraint]] = None


# ==================== Helper Functions ====================

def safe_float(value) -> float:
    """Convert numpy types to Python float safely"""
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return float(value)


def make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    elif isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def validate_mixture_proportions(proportions: List[float], tol: float = 1e-6) -> bool:
    """Validate that proportions sum to 1"""
    return abs(sum(proportions) - 1.0) < tol and all(0 <= p <= 1 for p in proportions)


def generate_simplex_grid(n_components: int, resolution: int = 50) -> np.ndarray:
    """Generate a grid of points on the simplex for contour plotting"""
    if n_components != 3:
        raise ValueError("Simplex grid currently supports only 3 components")

    points = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            x1 = i / resolution
            x2 = j / resolution
            x3 = k / resolution
            points.append([x1, x2, x3])

    return np.array(points)


def evaluate_scheffe_model(proportions: np.ndarray, coefficients: Dict[str, float],
                          component_names: List[str], model_type: str) -> float:
    """Evaluate Scheffé polynomial model at given proportions"""
    n = len(component_names)
    x = {name: proportions[i] for i, name in enumerate(component_names)}

    result = 0.0

    # Linear terms (b_i * x_i)
    for name in component_names:
        key = name
        if key in coefficients:
            result += coefficients[key] * x[name]

    if model_type in ["quadratic", "cubic"]:
        # Quadratic interaction terms (b_ij * x_i * x_j)
        for i, name_i in enumerate(component_names):
            for j, name_j in enumerate(component_names):
                if j > i:
                    key = f"{name_i}*{name_j}"
                    alt_key = f"{name_j}*{name_i}"
                    if key in coefficients:
                        result += coefficients[key] * x[name_i] * x[name_j]
                    elif alt_key in coefficients:
                        result += coefficients[alt_key] * x[name_i] * x[name_j]

    if model_type == "cubic":
        # Cubic terms (b_ijk * x_i * x_j * x_k) and (b_iij * x_i * x_i * x_j)
        for i, name_i in enumerate(component_names):
            for j, name_j in enumerate(component_names):
                if j > i:
                    # Delta terms: (x_i - x_j) * x_i * x_j
                    delta_key = f"delta_{name_i}_{name_j}"
                    if delta_key in coefficients:
                        result += coefficients[delta_key] * (x[name_i] - x[name_j]) * x[name_i] * x[name_j]

        # Triple interaction
        if n >= 3:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        key = f"{component_names[i]}*{component_names[j]}*{component_names[k]}"
                        if key in coefficients:
                            result += coefficients[key] * x[component_names[i]] * x[component_names[j]] * x[component_names[k]]

    return result


def find_extreme_vertices(constraints: List[ComponentConstraint]) -> List[List[float]]:
    """
    Find extreme vertices of constrained mixture region using vertex enumeration.
    Uses the fact that extreme vertices occur where constraints are active.
    """
    n = len(constraints)
    vertices = []

    # Get bounds
    mins = [c.min_prop for c in constraints]
    maxs = [c.max_prop for c in constraints]

    # Check if the region is feasible
    if sum(mins) > 1.0 + 1e-9 or sum(maxs) < 1.0 - 1e-9:
        return []

    # For each component, try setting it to min or max
    # A vertex occurs when n-1 components are at their bounds
    for active_set in combinations(range(n), n - 1):
        # Try all 2^(n-1) combinations of min/max for active components
        for bound_choice in product([0, 1], repeat=n-1):
            point = [0.0] * n
            total = 0.0
            valid = True

            # Set active components to their bounds
            for idx, comp_idx in enumerate(active_set):
                if bound_choice[idx] == 0:
                    point[comp_idx] = mins[comp_idx]
                else:
                    point[comp_idx] = maxs[comp_idx]
                total += point[comp_idx]

            # Find the free component (the one not in active_set)
            free_idx = [i for i in range(n) if i not in active_set][0]

            # Set free component to make sum = 1
            point[free_idx] = 1.0 - total

            # Check if this point satisfies all constraints
            if mins[free_idx] - 1e-9 <= point[free_idx] <= maxs[free_idx] + 1e-9:
                # Verify all constraints
                for i in range(n):
                    if not (mins[i] - 1e-9 <= point[i] <= maxs[i] + 1e-9):
                        valid = False
                        break

                if valid and abs(sum(point) - 1.0) < 1e-9:
                    # Round small negative values to 0
                    point = [max(0.0, min(1.0, p)) for p in point]
                    # Check for duplicates
                    is_duplicate = False
                    for existing in vertices:
                        if all(abs(point[i] - existing[i]) < 1e-6 for i in range(n)):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        vertices.append(point)

    return vertices


def compute_edge_centroids(vertices: List[List[float]]) -> List[List[float]]:
    """Compute centroids of edges (pairs of vertices)"""
    centroids = []
    n_vertices = len(vertices)

    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            centroid = [(vertices[i][k] + vertices[j][k]) / 2 for k in range(len(vertices[0]))]
            centroids.append(centroid)

    return centroids


def compute_face_centroids(vertices: List[List[float]], n_components: int) -> List[List[float]]:
    """Compute centroids of faces for higher-dimensional simplices"""
    if len(vertices) < 3:
        return []

    centroids = []
    n_vertices = len(vertices)

    # For faces (triplets of vertices)
    for combo in combinations(range(n_vertices), 3):
        centroid = [sum(vertices[combo[k]][i] for k in range(3)) / 3
                   for i in range(n_components)]
        if validate_mixture_proportions(centroid, tol=0.01):
            centroids.append(centroid)

    return centroids


def compute_overall_centroid(vertices: List[List[float]]) -> List[float]:
    """Compute the overall centroid of all vertices"""
    n = len(vertices[0])
    return [sum(v[i] for v in vertices) / len(vertices) for i in range(n)]


def compute_axial_points(vertices: List[List[float]], overall_centroid: List[float]) -> List[List[float]]:
    """Compute axial check blend points (midpoints between centroid and vertices)"""
    axial = []
    for vertex in vertices:
        midpoint = [(vertex[i] + overall_centroid[i]) / 2 for i in range(len(vertex))]
        axial.append(midpoint)
    return axial


def generate_simplex_lattice(n_components: int, degree: int = 2) -> np.ndarray:
    """Generate Simplex-Lattice design of degree m"""
    def generate_compositions(n, total, current=[]):
        if n == 1:
            yield current + [total]
        else:
            for i in range(total + 1):
                yield from generate_compositions(n - 1, total - i, current + [i])

    points = []
    for composition in generate_compositions(n_components, degree):
        point = [c / degree for c in composition]
        points.append(point)

    return np.array(points)


def generate_simplex_centroid(n_components: int) -> np.ndarray:
    """Generate Simplex-Centroid design"""
    points = []

    # Pure components
    for i in range(n_components):
        point = [0.0] * n_components
        point[i] = 1.0
        points.append(point)

    # Binary blends (50-50)
    for i in range(n_components):
        for j in range(i + 1, n_components):
            point = [0.0] * n_components
            point[i] = 0.5
            point[j] = 0.5
            points.append(point)

    # Ternary blends (1/3 each)
    if n_components >= 3:
        for combo in combinations(range(n_components), 3):
            point = [0.0] * n_components
            for idx in combo:
                point[idx] = 1/3
            points.append(point)

    # Higher-order blends up to full centroid
    for k in range(4, n_components + 1):
        for combo in combinations(range(n_components), k):
            point = [0.0] * n_components
            for idx in combo:
                point[idx] = 1/k
            points.append(point)

    return np.array(points)


def to_ternary_coordinates(proportions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert mixture proportions to ternary (x, y) coordinates for plotting"""
    # For a ternary plot with component A at top, B at bottom-left, C at bottom-right
    # x = 0.5 * (2*B + C) / (A + B + C) = 0.5 * (2*p2 + p3)
    # y = (sqrt(3)/2) * C / (A + B + C) = (sqrt(3)/2) * p1
    if proportions.ndim == 1:
        proportions = proportions.reshape(1, -1)

    p1, p2, p3 = proportions[:, 0], proportions[:, 1], proportions[:, 2]

    # Standard ternary coordinates
    x = 0.5 * (2 * p2 + p3)
    y = (np.sqrt(3) / 2) * p3

    return x, y


# ==================== API Endpoints ====================

@router.post("/extreme-vertices/generate")
async def generate_extreme_vertices_design(request: ExtremeVerticesRequest):
    """
    Generate Extreme Vertices design for constrained mixture region.

    Extreme vertices designs are used when mixture components have
    lower and upper bounds, creating a constrained design space.
    """
    try:
        n_components = len(request.components)
        component_names = [c.name for c in request.components]

        if n_components < 2:
            raise ValueError("Need at least 2 components for mixture design")

        # Validate constraints
        min_sum = sum(c.min_prop for c in request.components)
        max_sum = sum(c.max_prop for c in request.components)

        if min_sum > 1.0 + 1e-9:
            raise ValueError(f"Minimum proportions sum to {min_sum:.4f}, exceeding 1.0. Constraints are infeasible.")
        if max_sum < 1.0 - 1e-9:
            raise ValueError(f"Maximum proportions sum to {max_sum:.4f}, less than 1.0. Constraints are infeasible.")

        # Find extreme vertices
        vertices = find_extreme_vertices(request.components)

        if not vertices:
            raise ValueError("No feasible vertices found. Check constraint bounds.")

        design_points = list(vertices)
        point_types = ["vertex"] * len(vertices)

        # Add edge centroids
        if request.include_centroids and len(vertices) > 1:
            edge_centroids = compute_edge_centroids(vertices)
            # Filter valid centroids
            valid_centroids = []
            for c in edge_centroids:
                valid = True
                for i, comp in enumerate(request.components):
                    if not (comp.min_prop - 1e-6 <= c[i] <= comp.max_prop + 1e-6):
                        valid = False
                        break
                if valid:
                    valid_centroids.append(c)
            design_points.extend(valid_centroids)
            point_types.extend(["edge_centroid"] * len(valid_centroids))

            # Face centroids for 4+ components
            if n_components >= 4:
                face_centroids = compute_face_centroids(vertices, n_components)
                valid_face = []
                for c in face_centroids:
                    valid = True
                    for i, comp in enumerate(request.components):
                        if not (comp.min_prop - 1e-6 <= c[i] <= comp.max_prop + 1e-6):
                            valid = False
                            break
                    if valid:
                        valid_face.append(c)
                design_points.extend(valid_face)
                point_types.extend(["face_centroid"] * len(valid_face))

        # Add overall centroid(s)
        if request.n_center_points > 0:
            overall_centroid = compute_overall_centroid(vertices)
            for _ in range(request.n_center_points):
                design_points.append(overall_centroid)
                point_types.append("overall_centroid")

        # Add axial points
        if request.include_axial and len(vertices) > 0:
            overall_centroid = compute_overall_centroid(vertices)
            axial_points = compute_axial_points(vertices, overall_centroid)
            # Filter valid axial points
            valid_axial = []
            for a in axial_points:
                valid = True
                for i, comp in enumerate(request.components):
                    if not (comp.min_prop - 1e-6 <= a[i] <= comp.max_prop + 1e-6):
                        valid = False
                        break
                if valid:
                    valid_axial.append(a)
            design_points.extend(valid_axial)
            point_types.extend(["axial"] * len(valid_axial))

        # Format design matrix
        design_matrix = []
        for i, point in enumerate(design_points):
            row = {"run": i + 1, "point_type": point_types[i]}
            for j, name in enumerate(component_names):
                row[name] = round(point[j], 6)
            design_matrix.append(row)

        return make_json_safe({
            "success": True,
            "design": design_matrix,
            "n_runs": len(design_matrix),
            "n_vertices": len(vertices),
            "n_components": n_components,
            "component_names": component_names,
            "constraints": [
                {"name": c.name, "min": c.min_prop, "max": c.max_prop}
                for c in request.components
            ],
            "design_summary": {
                "vertices": sum(1 for t in point_types if t == "vertex"),
                "edge_centroids": sum(1 for t in point_types if t == "edge_centroid"),
                "face_centroids": sum(1 for t in point_types if t == "face_centroid"),
                "overall_centroids": sum(1 for t in point_types if t == "overall_centroid"),
                "axial_points": sum(1 for t in point_types if t == "axial")
            },
            "interpretation": f"Extreme Vertices design with {len(design_matrix)} runs for {n_components}-component mixture"
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mixture-process/generate")
async def generate_mixture_process_design(request: MixtureProcessRequest):
    """
    Generate combined Mixture + Process factor design.

    Combines a mixture design with process factors (temperature, time, etc.)
    using a crossed design structure.
    """
    try:
        n_comp = request.n_components

        if n_comp < 2:
            raise ValueError("Need at least 2 mixture components")

        component_names = request.component_names or [f"X{i+1}" for i in range(n_comp)]

        if len(component_names) != n_comp:
            raise ValueError(f"Expected {n_comp} component names, got {len(component_names)}")

        # Generate mixture design
        if request.mixture_design_type == "simplex-lattice":
            mixture_points = generate_simplex_lattice(n_comp, request.lattice_degree)
        else:
            mixture_points = generate_simplex_centroid(n_comp)

        # Generate process factor levels
        process_levels = []
        process_names = []
        for factor in request.process_factors:
            name = factor.get("name", f"P{len(process_names)+1}")
            levels = factor.get("levels", [-1, 1])
            process_names.append(name)
            process_levels.append(levels)

        # Generate process design points
        if request.process_design_type == "full-factorial":
            process_points = list(product(*process_levels))
        else:
            # Fractional factorial - for now just use a reduced set
            # This is simplified; real fractional would use generator relations
            process_points = list(product(*process_levels))
            if len(process_points) > 8:
                step = len(process_points) // 8
                process_points = process_points[::step]

        # Cross mixture and process designs
        design_matrix = []
        run = 1
        for mixture in mixture_points:
            for process in process_points:
                row = {"run": run}
                for i, name in enumerate(component_names):
                    row[name] = round(float(mixture[i]), 6)
                for i, name in enumerate(process_names):
                    row[name] = float(process[i])
                design_matrix.append(row)
                run += 1

        return make_json_safe({
            "success": True,
            "design": design_matrix,
            "n_runs": len(design_matrix),
            "n_mixture_points": len(mixture_points),
            "n_process_points": len(process_points),
            "component_names": component_names,
            "process_factor_names": process_names,
            "mixture_design_type": request.mixture_design_type,
            "process_design_type": request.process_design_type,
            "interpretation": (
                f"Combined Mixture + Process design with {len(design_matrix)} runs. "
                f"{len(mixture_points)} mixture points × {len(process_points)} process combinations."
            )
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ternary-contour")
async def generate_ternary_contour(request: TernaryContourRequest):
    """
    Generate data for ternary contour plot.

    Returns grid points in ternary coordinates with predicted response values
    for visualization of mixture response surface.
    """
    try:
        if len(request.component_names) != 3:
            raise ValueError("Ternary contour requires exactly 3 components")

        component_names = request.component_names
        resolution = request.grid_resolution

        # Generate simplex grid
        simplex_points = generate_simplex_grid(3, resolution)

        # Apply constraints if provided
        if request.constraints:
            valid_mask = np.ones(len(simplex_points), dtype=bool)
            for i, constraint in enumerate(request.constraints):
                idx = component_names.index(constraint.name)
                valid_mask &= (simplex_points[:, idx] >= constraint.min_prop - 1e-9)
                valid_mask &= (simplex_points[:, idx] <= constraint.max_prop + 1e-9)
            simplex_points = simplex_points[valid_mask]

        # Evaluate model at each point
        predictions = []
        for point in simplex_points:
            pred = evaluate_scheffe_model(
                point, request.model_coefficients,
                component_names, request.model_type
            )
            predictions.append(pred)

        predictions = np.array(predictions)

        # Convert to ternary coordinates
        x_coords, y_coords = to_ternary_coordinates(simplex_points)

        # Create contour data
        contour_data = {
            "x": x_coords.tolist(),
            "y": y_coords.tolist(),
            "z": predictions.tolist(),
            "proportions": simplex_points.tolist(),
            "component_names": component_names,
            "z_min": float(np.min(predictions)),
            "z_max": float(np.max(predictions)),
            "z_mean": float(np.mean(predictions))
        }

        # Find optimal blend
        opt_idx = np.argmax(predictions)  # For maximization
        optimal_blend = {
            component_names[i]: float(simplex_points[opt_idx, i])
            for i in range(3)
        }
        optimal_blend["predicted_response"] = float(predictions[opt_idx])

        return make_json_safe({
            "success": True,
            "contour_data": contour_data,
            "n_points": len(simplex_points),
            "optimal_blend": optimal_blend,
            "model_type": request.model_type,
            "interpretation": (
                f"Ternary contour for {request.model_type} Scheffé model. "
                f"Response range: {contour_data['z_min']:.4f} to {contour_data['z_max']:.4f}"
            )
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trace-plot")
async def generate_trace_plot(request: TracePlotRequest):
    """
    Generate trace plot (Cox direction) data.

    Shows how the response changes as each component varies from the reference blend
    while maintaining mixture constraint (sum = 1).
    """
    try:
        n_components = len(request.component_names)
        reference = np.array(request.reference_blend)

        if len(reference) != n_components:
            raise ValueError(f"Reference blend must have {n_components} proportions")

        if abs(sum(reference) - 1.0) > 1e-6:
            raise ValueError("Reference blend proportions must sum to 1.0")

        traces = {}

        for i, comp_name in enumerate(request.component_names):
            # Vary component i from 0 to 1
            x_values = np.linspace(0, 1, request.n_points)
            y_values = []
            valid_x = []

            for x_i in x_values:
                # Scale other components proportionally to maintain sum = 1
                other_sum = sum(reference[j] for j in range(n_components) if j != i)

                if other_sum < 1e-9:
                    # All weight was on component i
                    if x_i == 1.0:
                        new_point = [0.0] * n_components
                        new_point[i] = 1.0
                    else:
                        continue  # Can't compute this point
                else:
                    scale = (1.0 - x_i) / other_sum
                    new_point = [reference[j] * scale if j != i else x_i for j in range(n_components)]

                # Evaluate model
                pred = evaluate_scheffe_model(
                    np.array(new_point), request.model_coefficients,
                    request.component_names, request.model_type
                )

                valid_x.append(float(x_i))
                y_values.append(pred)

            traces[comp_name] = {
                "x": valid_x,
                "y": y_values,
                "reference_value": float(reference[i])
            }

        # Evaluate at reference point
        reference_response = evaluate_scheffe_model(
            reference, request.model_coefficients,
            request.component_names, request.model_type
        )

        return make_json_safe({
            "success": True,
            "traces": traces,
            "reference_blend": {name: float(reference[i]) for i, name in enumerate(request.component_names)},
            "reference_response": reference_response,
            "n_components": n_components,
            "model_type": request.model_type,
            "interpretation": (
                f"Trace plots showing effect of varying each component from reference blend. "
                f"Reference response: {reference_response:.4f}"
            )
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/augment-design")
async def augment_mixture_design(request: AugmentDesignRequest):
    """
    Augment an existing mixture design with additional optimal points.

    Uses exchange algorithm to find points that maximize D-optimality
    or minimize prediction variance (I-optimal).
    """
    try:
        existing = np.array(request.existing_design)
        n_components = len(request.component_names)

        if existing.shape[1] != n_components:
            raise ValueError(f"Existing design has {existing.shape[1]} columns, expected {n_components}")

        # Get constraints
        if request.constraints:
            mins = [c.min_prop for c in request.constraints]
            maxs = [c.max_prop for c in request.constraints]
        else:
            mins = [0.0] * n_components
            maxs = [1.0] * n_components

        # Generate candidate points using simplex-centroid or lattice
        candidates = generate_simplex_lattice(n_components, degree=4)

        # Filter by constraints
        valid_candidates = []
        for point in candidates:
            valid = True
            for i in range(n_components):
                if not (mins[i] - 1e-6 <= point[i] <= maxs[i] + 1e-6):
                    valid = False
                    break
            if valid:
                valid_candidates.append(point)

        candidates = np.array(valid_candidates)

        # Build model matrix for Scheffé quadratic
        def build_model_matrix(X):
            n = X.shape[0]
            p = X.shape[1]
            # Linear terms
            terms = [X[:, i] for i in range(p)]
            # Quadratic interaction terms
            for i in range(p):
                for j in range(i+1, p):
                    terms.append(X[:, i] * X[:, j])
            return np.column_stack(terms)

        current_design = existing.copy()
        augmented_points = []

        for _ in range(request.n_augment_points):
            best_point = None
            best_criterion = -np.inf if request.criterion == "d-optimal" else np.inf

            for candidate in candidates:
                # Try adding this candidate
                trial_design = np.vstack([current_design, candidate])
                M = build_model_matrix(trial_design)

                try:
                    if request.criterion == "d-optimal":
                        # D-criterion: determinant of information matrix
                        XtX = M.T @ M
                        det = np.linalg.det(XtX)
                        if det > best_criterion:
                            best_criterion = det
                            best_point = candidate
                    else:
                        # I-criterion: average prediction variance
                        XtX_inv = np.linalg.inv(M.T @ M)
                        avg_var = np.mean([M[i] @ XtX_inv @ M[i].T for i in range(len(M))])
                        if avg_var < best_criterion:
                            best_criterion = avg_var
                            best_point = candidate
                except np.linalg.LinAlgError:
                    continue

            if best_point is not None:
                current_design = np.vstack([current_design, best_point])
                augmented_points.append(best_point.tolist())

        # Format augmented design
        augmented_design = []
        for i, point in enumerate(current_design):
            row = {"run": i + 1, "is_new": i >= len(existing)}
            for j, name in enumerate(request.component_names):
                row[name] = round(float(point[j]), 6)
            augmented_design.append(row)

        return make_json_safe({
            "success": True,
            "augmented_design": augmented_design,
            "new_points": augmented_points,
            "n_original": len(existing),
            "n_augmented": len(augmented_points),
            "n_total": len(augmented_design),
            "criterion": request.criterion,
            "interpretation": (
                f"Added {len(augmented_points)} points to existing {len(existing)}-run design "
                f"using {request.criterion} criterion."
            )
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def mixture_info():
    """Get information about available mixture design features"""
    return {
        "module": "Mixture Designs",
        "version": "1.0.0",
        "description": "Advanced mixture experiment design and analysis",
        "features": [
            "Extreme Vertices designs for constrained mixtures",
            "Mixture + Process factor combined designs",
            "Ternary contour plots for 3-component mixtures",
            "Cox direction trace plots",
            "Design augmentation with D/I-optimal criteria"
        ],
        "endpoints": {
            "/extreme-vertices/generate": "Generate extreme vertices design",
            "/mixture-process/generate": "Generate mixture + process design",
            "/ternary-contour": "Get ternary contour plot data",
            "/trace-plot": "Get component trace plot data",
            "/augment-design": "Augment existing design optimally"
        },
        "note": "Basic simplex-lattice and simplex-centroid designs are available in /api/rsm/mixture-design/generate"
    }
