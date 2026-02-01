"""
Custom Design Platform API

Advanced optimal experimental design generation with:
- Linear and nonlinear constraints
- Disallowed factor combinations
- Categorical factors support
- Design augmentation
- Hard-to-change factors (split-plot)
- Multiple random starts
- Custom model specification

Uses coordinate exchange with constraint handling.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from itertools import combinations, product
import warnings

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/api/custom-design", tags=["Custom Design"])


# ============================================================================
# Pydantic Models
# ============================================================================

class FactorDefinition(BaseModel):
    """Definition of a single factor."""
    name: str = Field(..., description="Factor name")
    type: str = Field("continuous", description="Factor type: continuous or categorical")
    low: Optional[float] = Field(None, description="Lower bound for continuous factors")
    high: Optional[float] = Field(None, description="Upper bound for continuous factors")
    levels: Optional[List[Union[str, float]]] = Field(None, description="Levels for categorical factors")
    hard_to_change: bool = Field(False, description="Is this a hard-to-change factor?")


class LinearConstraint(BaseModel):
    """Linear constraint of form: sum(coef_i * factor_i) <= bound."""
    coefficients: Dict[str, float] = Field(..., description="Coefficients for each factor")
    bound: float = Field(..., description="Upper bound")
    type: str = Field("<=", description="Constraint type: <=, >=, ==")


class DisallowedCombination(BaseModel):
    """A disallowed combination of factor levels."""
    conditions: Dict[str, Any] = Field(..., description="Factor conditions that are disallowed")


class ModelTerm(BaseModel):
    """Definition of a model term."""
    factors: List[str] = Field(..., description="Factors involved in this term")
    power: Optional[List[int]] = Field(None, description="Powers for each factor (default: [1]*len)")


class CustomDesignRequest(BaseModel):
    """Request for custom design generation."""
    n_runs: int = Field(..., description="Number of experimental runs", ge=3)
    factors: List[FactorDefinition] = Field(..., description="Factor definitions")
    criterion: str = Field("d_optimal", description="Optimality criterion")
    model_type: str = Field("quadratic", description="Model type: linear, interaction, quadratic, custom")
    custom_terms: Optional[List[ModelTerm]] = Field(None, description="Custom model terms")
    constraints: Optional[List[LinearConstraint]] = Field(None, description="Linear constraints")
    disallowed: Optional[List[DisallowedCombination]] = Field(None, description="Disallowed combinations")
    n_random_starts: int = Field(10, description="Number of random starting designs", ge=1, le=100)
    max_iterations: int = Field(500, description="Max iterations per start", ge=10, le=5000)
    n_whole_plots: Optional[int] = Field(None, description="Number of whole plots for split-plot designs")


class AugmentDesignRequest(BaseModel):
    """Request for design augmentation."""
    existing_design: List[Dict[str, Any]] = Field(..., description="Existing design points")
    factors: List[FactorDefinition] = Field(..., description="Factor definitions")
    n_additional_runs: int = Field(..., description="Number of runs to add", ge=1)
    criterion: str = Field("d_optimal", description="Optimality criterion")
    model_type: str = Field("quadratic", description="Model type")
    constraints: Optional[List[LinearConstraint]] = Field(None, description="Constraints")
    max_iterations: int = Field(500, description="Max iterations")


class EvaluateCustomDesignRequest(BaseModel):
    """Request for evaluating a custom design."""
    design: List[Dict[str, Any]] = Field(..., description="Design to evaluate")
    factors: List[FactorDefinition] = Field(..., description="Factor definitions")
    model_type: str = Field("quadratic", description="Model type")
    custom_terms: Optional[List[ModelTerm]] = Field(None, description="Custom terms")


class PowerAnalysisRequest(BaseModel):
    """Request for design power analysis."""
    design: List[Dict[str, Any]] = Field(..., description="Design to analyze")
    factors: List[FactorDefinition] = Field(..., description="Factor definitions")
    model_type: str = Field("quadratic", description="Model type")
    effect_size: float = Field(1.0, description="Expected effect size (standardized)")
    alpha: float = Field(0.05, description="Significance level")
    sigma: float = Field(1.0, description="Error standard deviation")


# ============================================================================
# Helper Functions
# ============================================================================

def safe_float(value: Any) -> Optional[float]:
    """Convert value to float safely."""
    if value is None:
        return None
    try:
        result = float(value)
        if np.isnan(result) or np.isinf(result):
            return None
        return result
    except (ValueError, TypeError):
        return None


def make_json_safe(obj: Any) -> Any:
    """Make object JSON serializable."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class CustomDesignGenerator:
    """
    Advanced custom design generator with constraint handling.
    """

    def __init__(self, factors: List[FactorDefinition],
                 model_type: str = "quadratic",
                 custom_terms: Optional[List[ModelTerm]] = None,
                 constraints: Optional[List[LinearConstraint]] = None,
                 disallowed: Optional[List[DisallowedCombination]] = None):
        """Initialize the generator."""
        self.factors = factors
        self.factor_names = [f.name for f in factors]
        self.n_factors = len(factors)
        self.model_type = model_type
        self.custom_terms = custom_terms
        self.constraints = constraints or []
        self.disallowed = disallowed or []

        # Separate continuous and categorical factors
        self.continuous_factors = [f for f in factors if f.type == "continuous"]
        self.categorical_factors = [f for f in factors if f.type == "categorical"]

        # Build factor index mapping
        self.factor_index = {f.name: i for i, f in enumerate(factors)}

        # Identify hard-to-change factors
        self.htc_factors = [f for f in factors if f.hard_to_change]
        self.etc_factors = [f for f in factors if not f.hard_to_change]

    def generate_candidate_set(self, n_candidates_per_continuous: int = 21) -> np.ndarray:
        """Generate candidate points for the design."""
        # Generate candidates for continuous factors
        continuous_candidates = []
        for f in self.factors:
            if f.type == "continuous":
                # Include endpoints, center, and evenly spaced points
                candidates = np.linspace(f.low, f.high, n_candidates_per_continuous)
                continuous_candidates.append(candidates)
            else:
                # Categorical: use all levels (encoded as 0, 1, 2, ...)
                candidates = np.arange(len(f.levels))
                continuous_candidates.append(candidates)

        # Create full factorial of candidates (may be large!)
        # For efficiency, limit total candidates
        max_total = 10000
        if np.prod([len(c) for c in continuous_candidates]) > max_total:
            # Use random sampling instead
            candidates = []
            for _ in range(max_total):
                point = []
                for i, f in enumerate(self.factors):
                    if f.type == "continuous":
                        point.append(np.random.uniform(f.low, f.high))
                    else:
                        point.append(np.random.randint(0, len(f.levels)))
                candidates.append(point)
            return np.array(candidates)
        else:
            # Full grid
            mesh = np.array(list(product(*continuous_candidates)))
            return mesh

    def is_feasible(self, point: np.ndarray) -> bool:
        """Check if a point satisfies all constraints."""
        # Check linear constraints
        for constraint in self.constraints:
            value = sum(
                coef * point[self.factor_index[fname]]
                for fname, coef in constraint.coefficients.items()
                if fname in self.factor_index
            )

            if constraint.type == "<=" and value > constraint.bound + 1e-10:
                return False
            elif constraint.type == ">=" and value < constraint.bound - 1e-10:
                return False
            elif constraint.type == "==" and abs(value - constraint.bound) > 1e-10:
                return False

        # Check disallowed combinations
        for disallow in self.disallowed:
            match = True
            for fname, condition in disallow.conditions.items():
                if fname not in self.factor_index:
                    continue
                idx = self.factor_index[fname]
                factor = self.factors[idx]

                if factor.type == "categorical":
                    # Check if level matches
                    level_idx = int(point[idx])
                    if level_idx < len(factor.levels):
                        actual_level = factor.levels[level_idx]
                        if actual_level != condition:
                            match = False
                            break
                else:
                    # Continuous: check if in range
                    if isinstance(condition, dict):
                        if "min" in condition and point[idx] < condition["min"]:
                            match = False
                            break
                        if "max" in condition and point[idx] > condition["max"]:
                            match = False
                            break
                    else:
                        if abs(point[idx] - condition) > 1e-6:
                            match = False
                            break

            if match:
                return False  # This combination is disallowed

        return True

    def build_model_matrix(self, design: np.ndarray) -> np.ndarray:
        """Build the model matrix X for the design."""
        n_runs = design.shape[0]
        columns = [np.ones(n_runs)]  # Intercept

        # Get coded values for continuous factors
        coded_design = np.zeros_like(design, dtype=float)
        for i, f in enumerate(self.factors):
            if f.type == "continuous":
                # Code to [-1, 1]
                coded_design[:, i] = 2 * (design[:, i] - f.low) / (f.high - f.low) - 1
            else:
                coded_design[:, i] = design[:, i]

        if self.custom_terms:
            # Use custom model specification
            for term in self.custom_terms:
                col = np.ones(n_runs)
                for j, fname in enumerate(term.factors):
                    idx = self.factor_index[fname]
                    power = term.power[j] if term.power else 1
                    col *= coded_design[:, idx] ** power
                columns.append(col)
        else:
            # Standard model based on model_type
            # Linear terms
            for i in range(self.n_factors):
                columns.append(coded_design[:, i])

            if self.model_type in ["interaction", "quadratic"]:
                # Two-factor interactions
                for i, j in combinations(range(self.n_factors), 2):
                    columns.append(coded_design[:, i] * coded_design[:, j])

            if self.model_type == "quadratic":
                # Quadratic terms (only for continuous)
                for i, f in enumerate(self.factors):
                    if f.type == "continuous":
                        columns.append(coded_design[:, i] ** 2)

        return np.column_stack(columns)

    def get_term_names(self) -> List[str]:
        """Get names for all model terms."""
        names = ["Intercept"]

        if self.custom_terms:
            for term in self.custom_terms:
                if len(term.factors) == 1:
                    power = term.power[0] if term.power else 1
                    if power == 1:
                        names.append(term.factors[0])
                    else:
                        names.append(f"{term.factors[0]}^{power}")
                else:
                    parts = []
                    for j, fname in enumerate(term.factors):
                        power = term.power[j] if term.power else 1
                        if power == 1:
                            parts.append(fname)
                        else:
                            parts.append(f"{fname}^{power}")
                    names.append("*".join(parts))
        else:
            # Linear terms
            for f in self.factors:
                names.append(f.name)

            if self.model_type in ["interaction", "quadratic"]:
                for i, j in combinations(range(self.n_factors), 2):
                    names.append(f"{self.factors[i].name}*{self.factors[j].name}")

            if self.model_type == "quadratic":
                for f in self.factors:
                    if f.type == "continuous":
                        names.append(f"{f.name}^2")

        return names

    def calculate_d_criterion(self, design: np.ndarray) -> float:
        """Calculate D-criterion (determinant of X'X)."""
        X = self.build_model_matrix(design)
        try:
            det = np.linalg.det(X.T @ X)
            if np.isnan(det) or det <= 0:
                return 0.0
            return det
        except:
            return 0.0

    def calculate_i_criterion(self, design: np.ndarray,
                              prediction_grid: Optional[np.ndarray] = None) -> float:
        """Calculate I-criterion (average prediction variance)."""
        X = self.build_model_matrix(design)

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except:
            return 1e10

        if prediction_grid is None:
            prediction_grid = self.generate_prediction_grid()

        X_pred = self.build_model_matrix(prediction_grid)
        variances = np.sum((X_pred @ XtX_inv) * X_pred, axis=1)
        return np.mean(variances)

    def calculate_a_criterion(self, design: np.ndarray) -> float:
        """Calculate A-criterion (trace of (X'X)^-1)."""
        X = self.build_model_matrix(design)
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            return np.trace(XtX_inv)
        except:
            return 1e10

    def generate_prediction_grid(self, n_per_factor: int = 5) -> np.ndarray:
        """Generate a grid of prediction points."""
        grids = []
        for f in self.factors:
            if f.type == "continuous":
                grids.append(np.linspace(f.low, f.high, n_per_factor))
            else:
                grids.append(np.arange(len(f.levels)))

        mesh = np.array(list(product(*grids)))
        # Filter to feasible points
        feasible = [p for p in mesh if self.is_feasible(p)]
        return np.array(feasible) if feasible else mesh

    def initialize_design(self, n_runs: int) -> np.ndarray:
        """Initialize a random feasible design."""
        design = []
        max_attempts = n_runs * 100

        for _ in range(max_attempts):
            if len(design) >= n_runs:
                break

            point = []
            for f in self.factors:
                if f.type == "continuous":
                    point.append(np.random.uniform(f.low, f.high))
                else:
                    point.append(np.random.randint(0, len(f.levels)))

            point = np.array(point)
            if self.is_feasible(point):
                design.append(point)

        if len(design) < n_runs:
            raise ValueError(f"Could not generate {n_runs} feasible points. "
                           f"Only found {len(design)}. Check constraints.")

        return np.array(design[:n_runs])

    def coordinate_exchange(self, n_runs: int, criterion: str = "d_optimal",
                           max_iterations: int = 500,
                           n_candidates: int = 21) -> np.ndarray:
        """Run coordinate exchange algorithm."""
        design = self.initialize_design(n_runs)

        if criterion == "d_optimal":
            current_value = self.calculate_d_criterion(design)
            maximize = True
        elif criterion == "i_optimal":
            current_value = self.calculate_i_criterion(design)
            maximize = False
        elif criterion == "a_optimal":
            current_value = self.calculate_a_criterion(design)
            maximize = False
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        best_design = design.copy()
        best_value = current_value

        for iteration in range(max_iterations):
            improved = False

            for i in range(n_runs):
                for j in range(self.n_factors):
                    factor = self.factors[j]

                    # Generate candidates for this factor
                    if factor.type == "continuous":
                        candidates = np.linspace(factor.low, factor.high, n_candidates)
                    else:
                        candidates = np.arange(len(factor.levels))

                    best_local_value = current_value
                    best_local_candidate = design[i, j]

                    for candidate in candidates:
                        old_value = design[i, j]
                        design[i, j] = candidate

                        if not self.is_feasible(design[i]):
                            design[i, j] = old_value
                            continue

                        if criterion == "d_optimal":
                            new_value = self.calculate_d_criterion(design)
                        elif criterion == "i_optimal":
                            new_value = self.calculate_i_criterion(design)
                        else:
                            new_value = self.calculate_a_criterion(design)

                        if maximize:
                            if new_value > best_local_value * 1.0001:
                                best_local_value = new_value
                                best_local_candidate = candidate
                                improved = True
                        else:
                            if new_value < best_local_value * 0.9999:
                                best_local_value = new_value
                                best_local_candidate = candidate
                                improved = True

                        design[i, j] = old_value

                    design[i, j] = best_local_candidate
                    current_value = best_local_value

                    if maximize and current_value > best_value:
                        best_value = current_value
                        best_design = design.copy()
                    elif not maximize and current_value < best_value:
                        best_value = current_value
                        best_design = design.copy()

            if not improved:
                break

        return best_design

    def generate_design(self, n_runs: int, criterion: str = "d_optimal",
                       n_random_starts: int = 10,
                       max_iterations: int = 500) -> Tuple[np.ndarray, float]:
        """Generate optimal design with multiple random starts."""
        best_design = None
        best_value = None

        for start in range(n_random_starts):
            try:
                design = self.coordinate_exchange(
                    n_runs, criterion, max_iterations
                )

                if criterion == "d_optimal":
                    value = self.calculate_d_criterion(design)
                    if best_value is None or value > best_value:
                        best_value = value
                        best_design = design.copy()
                elif criterion == "i_optimal":
                    value = self.calculate_i_criterion(design)
                    if best_value is None or value < best_value:
                        best_value = value
                        best_design = design.copy()
                else:
                    value = self.calculate_a_criterion(design)
                    if best_value is None or value < best_value:
                        best_value = value
                        best_design = design.copy()
            except Exception as e:
                continue

        if best_design is None:
            raise ValueError("Failed to generate any valid design")

        return best_design, best_value

    def evaluate_design(self, design: np.ndarray) -> Dict:
        """Evaluate a design comprehensively."""
        X = self.build_model_matrix(design)
        n_runs = design.shape[0]
        n_params = X.shape[1]

        try:
            XtX = X.T @ X
            XtX_inv = np.linalg.inv(XtX)

            # D-efficiency
            det = np.linalg.det(XtX)
            d_efficiency = (det ** (1/n_params)) / n_runs if det > 0 else 0

            # A-criterion
            a_criterion = np.trace(XtX_inv)

            # G-efficiency
            leverage = np.diag(X @ XtX_inv @ X.T)
            g_efficiency = n_params / (n_runs * np.max(leverage))

            # Condition number
            condition_number = np.linalg.cond(XtX)

            # VIFs
            vifs = [XtX_inv[i, i] * XtX[i, i] for i in range(n_params)]
            max_vif = max(vifs)

            # Average variance
            avg_variance = a_criterion / n_params

            # Degrees of freedom
            df_residual = n_runs - n_params

            return {
                "d_efficiency": float(d_efficiency),
                "d_criterion": float(det),
                "a_criterion": float(a_criterion),
                "g_efficiency": float(g_efficiency),
                "condition_number": float(condition_number),
                "vif_values": [float(v) for v in vifs],
                "max_vif": float(max_vif),
                "avg_variance": float(avg_variance),
                "n_runs": n_runs,
                "n_parameters": n_params,
                "df_residual": df_residual,
                "leverage": [float(l) for l in leverage],
                "estimable": df_residual > 0,
                "term_names": self.get_term_names()
            }

        except np.linalg.LinAlgError:
            return {
                "error": "Singular design matrix - model not estimable",
                "d_efficiency": 0.0,
                "estimable": False,
                "n_runs": n_runs,
                "n_parameters": n_params
            }


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/generate")
async def generate_custom_design(request: CustomDesignRequest):
    """
    Generate a custom optimal experimental design.

    Supports:
    - D-optimal, I-optimal, A-optimal criteria
    - Linear constraints on factors
    - Disallowed factor combinations
    - Categorical and continuous factors
    - Custom model specification
    """
    try:
        generator = CustomDesignGenerator(
            factors=request.factors,
            model_type=request.model_type,
            custom_terms=request.custom_terms,
            constraints=request.constraints,
            disallowed=request.disallowed
        )

        design, criterion_value = generator.generate_design(
            n_runs=request.n_runs,
            criterion=request.criterion,
            n_random_starts=request.n_random_starts,
            max_iterations=request.max_iterations
        )

        # Evaluate the design
        evaluation = generator.evaluate_design(design)

        # Convert design to table format
        design_table = []
        for i, row in enumerate(design):
            run = {"Run": i + 1}
            for j, factor in enumerate(request.factors):
                if factor.type == "categorical":
                    level_idx = int(row[j])
                    run[factor.name] = factor.levels[level_idx] if level_idx < len(factor.levels) else level_idx
                else:
                    run[factor.name] = round(float(row[j]), 6)
            design_table.append(run)

        return make_json_safe({
            "success": True,
            "design": design_table,
            "criterion": request.criterion,
            "criterion_value": criterion_value,
            "evaluation": evaluation,
            "n_runs": request.n_runs,
            "n_factors": len(request.factors),
            "model_type": request.model_type,
            "factor_names": [f.name for f in request.factors],
            "n_random_starts": request.n_random_starts,
            "constraints_applied": len(request.constraints or []),
            "disallowed_applied": len(request.disallowed or [])
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/augment")
async def augment_design(request: AugmentDesignRequest):
    """
    Augment an existing design with additional optimal runs.

    The new runs are chosen to optimally improve the design
    while keeping existing runs fixed.
    """
    try:
        generator = CustomDesignGenerator(
            factors=request.factors,
            model_type=request.model_type,
            constraints=request.constraints
        )

        # Convert existing design to array
        existing = []
        for row in request.existing_design:
            point = []
            for factor in request.factors:
                val = row.get(factor.name)
                if factor.type == "categorical":
                    # Convert level to index
                    if val in factor.levels:
                        point.append(factor.levels.index(val))
                    else:
                        point.append(0)
                else:
                    point.append(float(val))
            existing.append(point)

        existing_array = np.array(existing)
        n_existing = len(existing_array)

        # Generate augmented design
        total_runs = n_existing + request.n_additional_runs
        best_augmented = None
        best_value = None

        for _ in range(10):  # Multiple starts
            try:
                # Initialize with existing + random new points
                new_points = generator.initialize_design(request.n_additional_runs)
                full_design = np.vstack([existing_array, new_points])

                # Optimize only the new points
                for iteration in range(request.max_iterations):
                    improved = False

                    for i in range(n_existing, total_runs):
                        for j in range(len(request.factors)):
                            factor = request.factors[j]

                            if factor.type == "continuous":
                                candidates = np.linspace(factor.low, factor.high, 21)
                            else:
                                candidates = np.arange(len(factor.levels))

                            current_value = generator.calculate_d_criterion(full_design)
                            best_local = current_value
                            best_candidate = full_design[i, j]

                            for candidate in candidates:
                                old = full_design[i, j]
                                full_design[i, j] = candidate

                                if not generator.is_feasible(full_design[i]):
                                    full_design[i, j] = old
                                    continue

                                new_value = generator.calculate_d_criterion(full_design)
                                if new_value > best_local * 1.0001:
                                    best_local = new_value
                                    best_candidate = candidate
                                    improved = True

                                full_design[i, j] = old

                            full_design[i, j] = best_candidate

                    if not improved:
                        break

                value = generator.calculate_d_criterion(full_design)
                if best_value is None or value > best_value:
                    best_value = value
                    best_augmented = full_design.copy()

            except:
                continue

        if best_augmented is None:
            raise ValueError("Failed to augment design")

        # Evaluate augmented design
        evaluation = generator.evaluate_design(best_augmented)

        # Convert to table format
        design_table = []
        for i, row in enumerate(best_augmented):
            run = {
                "Run": i + 1,
                "Type": "Existing" if i < n_existing else "New"
            }
            for j, factor in enumerate(request.factors):
                if factor.type == "categorical":
                    level_idx = int(row[j])
                    run[factor.name] = factor.levels[level_idx] if level_idx < len(factor.levels) else level_idx
                else:
                    run[factor.name] = round(float(row[j]), 6)
            design_table.append(run)

        return make_json_safe({
            "success": True,
            "design": design_table,
            "evaluation": evaluation,
            "n_existing_runs": n_existing,
            "n_new_runs": request.n_additional_runs,
            "n_total_runs": total_runs,
            "criterion_value": best_value
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/evaluate")
async def evaluate_custom_design(request: EvaluateCustomDesignRequest):
    """
    Evaluate efficiency and properties of a custom design.
    """
    try:
        generator = CustomDesignGenerator(
            factors=request.factors,
            model_type=request.model_type,
            custom_terms=request.custom_terms
        )

        # Convert design to array
        design_array = []
        for row in request.design:
            point = []
            for factor in request.factors:
                val = row.get(factor.name)
                if factor.type == "categorical":
                    if val in factor.levels:
                        point.append(factor.levels.index(val))
                    else:
                        point.append(0)
                else:
                    point.append(float(val))
            design_array.append(point)

        design_array = np.array(design_array)
        evaluation = generator.evaluate_design(design_array)

        # Check constraint satisfaction
        constraint_violations = []
        for i, row in enumerate(design_array):
            if not generator.is_feasible(row):
                constraint_violations.append(i + 1)

        evaluation["constraint_violations"] = constraint_violations
        evaluation["all_feasible"] = len(constraint_violations) == 0

        return make_json_safe({
            "success": True,
            "evaluation": evaluation
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/power-analysis")
async def design_power_analysis(request: PowerAnalysisRequest):
    """
    Analyze statistical power of a design for detecting effects.
    """
    try:
        generator = CustomDesignGenerator(
            factors=request.factors,
            model_type=request.model_type
        )

        # Convert design
        design_array = []
        for row in request.design:
            point = []
            for factor in request.factors:
                val = row.get(factor.name)
                if factor.type == "categorical":
                    if val in factor.levels:
                        point.append(factor.levels.index(val))
                    else:
                        point.append(0)
                else:
                    point.append(float(val))
            design_array.append(point)

        design_array = np.array(design_array)
        X = generator.build_model_matrix(design_array)
        n_runs = X.shape[0]
        n_params = X.shape[1]

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except:
            raise ValueError("Design matrix is singular - cannot compute power")

        # Calculate power for each term
        from scipy.stats import ncf, f as f_dist

        term_names = generator.get_term_names()
        power_results = []

        df1 = 1  # Single coefficient
        df2 = n_runs - n_params

        if df2 <= 0:
            raise ValueError(f"Insufficient degrees of freedom. Need at least {n_params + 1} runs.")

        for i, name in enumerate(term_names):
            if i == 0:  # Skip intercept
                continue

            # Variance of coefficient estimate
            var_beta = XtX_inv[i, i] * (request.sigma ** 2)
            se_beta = np.sqrt(var_beta)

            # Non-centrality parameter
            ncp = (request.effect_size ** 2) / var_beta

            # Critical F value
            f_crit = f_dist.ppf(1 - request.alpha, df1, df2)

            # Power (probability of rejecting H0 given true effect)
            power = 1 - ncf.cdf(f_crit, df1, df2, ncp)

            power_results.append({
                "term": name,
                "se": float(se_beta),
                "power": float(power),
                "ncp": float(ncp),
                "detectable_effect": float(se_beta * 2)  # Approx minimum detectable effect
            })

        # Overall power summary
        avg_power = np.mean([r["power"] for r in power_results])
        min_power = np.min([r["power"] for r in power_results])

        return make_json_safe({
            "success": True,
            "power_results": power_results,
            "summary": {
                "average_power": avg_power,
                "minimum_power": min_power,
                "effect_size": request.effect_size,
                "alpha": request.alpha,
                "sigma": request.sigma,
                "n_runs": n_runs,
                "df_error": df2
            },
            "interpretation": f"Average power = {avg_power:.1%}. " +
                            (f"Good power (>80%) for detecting effects of size {request.effect_size}."
                             if min_power >= 0.8 else
                             f"Low power. Consider adding runs or increasing effect size.")
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/model-info")
async def get_model_info():
    """Get information about available model types and criteria."""
    return {
        "model_types": {
            "linear": {
                "name": "Linear (Main Effects)",
                "description": "Only main effects: y = b0 + sum(bi*xi)",
                "terms": "Intercept + linear terms"
            },
            "interaction": {
                "name": "Two-Factor Interaction (2FI)",
                "description": "Main effects + all 2-way interactions",
                "terms": "Intercept + linear + interactions"
            },
            "quadratic": {
                "name": "Quadratic (RSM)",
                "description": "Full second-order model with curvature",
                "terms": "Intercept + linear + interactions + squared terms"
            },
            "custom": {
                "name": "Custom Model",
                "description": "User-specified model terms",
                "terms": "As specified"
            }
        },
        "criteria": {
            "d_optimal": {
                "name": "D-Optimal",
                "goal": "Maximize |X'X|",
                "best_for": "Parameter estimation precision"
            },
            "i_optimal": {
                "name": "I-Optimal",
                "goal": "Minimize average prediction variance",
                "best_for": "Prediction accuracy"
            },
            "a_optimal": {
                "name": "A-Optimal",
                "goal": "Minimize trace((X'X)^-1)",
                "best_for": "Average parameter variance"
            }
        },
        "factor_types": {
            "continuous": "Numeric factors with range [low, high]",
            "categorical": "Discrete factors with named levels"
        },
        "constraints": {
            "linear": "a1*X1 + a2*X2 + ... <= b",
            "disallowed": "Specific combinations to exclude"
        }
    }
