"""
Coordinate Exchange Algorithm for Optimal Experimental Design

Implements D-optimal, I-optimal, and A-optimal design generation using
the coordinate exchange algorithm. This is an iterative optimization
method that improves design efficiency by systematically trying different
values for each factor at each run.

References:
- Meyer & Nachtsheim (1995): "The Coordinate-Exchange Algorithm for
  Constructing Exact Optimal Experimental Designs"
- JMP Pro 16 documentation on Custom Design
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class CoordinateExchange:
    """
    Coordinate Exchange algorithm for optimal experimental design generation.

    This algorithm iteratively improves a design by changing one coordinate
    (factor level) at a time, keeping changes that improve the optimality
    criterion.
    """

    def __init__(self,
                 n_runs: int,
                 factors: List[str],
                 factor_ranges: Dict[str, Tuple[float, float]],
                 model_order: int = 2,
                 n_candidates: int = 20):
        """
        Initialize Coordinate Exchange algorithm.

        Args:
            n_runs: Number of experimental runs to generate
            factors: List of factor names
            factor_ranges: Dict mapping factor names to (min, max) tuples
            model_order: Model order (1=linear, 2=quadratic with interactions)
            n_candidates: Number of candidate values to try for each coordinate
        """
        self.n_runs = n_runs
        self.factors = factors
        self.factor_ranges = factor_ranges
        self.model_order = model_order
        self.n_factors = len(factors)
        self.n_candidates = n_candidates

    def generate_d_optimal(self, max_iterations: int = 1000,
                           tolerance: float = 1e-6) -> np.ndarray:
        """
        Generate D-optimal design by maximizing |X'X|^(1/p).

        D-optimality maximizes the determinant of the information matrix,
        which minimizes the volume of the confidence ellipsoid for the
        parameter estimates.

        Args:
            max_iterations: Maximum number of coordinate exchange iterations
            tolerance: Convergence tolerance for determinant improvement

        Returns:
            Design matrix (n_runs × n_factors)
        """
        # Initialize with random design
        design = self._initialize_design()

        # Build model matrix and calculate initial determinant
        X = self._build_model_matrix(design)
        try:
            current_det = np.linalg.det(X.T @ X)
        except np.linalg.LinAlgError:
            current_det = 0

        if current_det == 0:
            # Add small regularization if singular
            X_reg = X.T @ X + np.eye(X.shape[1]) * 1e-6
            current_det = np.linalg.det(X_reg)

        best_design = design.copy()
        best_det = current_det

        for iteration in range(max_iterations):
            improved = False

            # Try swapping each coordinate
            for i in range(self.n_runs):
                for j in range(self.n_factors):
                    best_value = design[i, j]
                    best_local_det = current_det

                    # Generate candidate values for this factor
                    candidates = self._generate_candidates(j)

                    for candidate in candidates:
                        # Temporarily change the value
                        old_value = design[i, j]
                        design[i, j] = candidate

                        # Calculate new determinant
                        X = self._build_model_matrix(design)
                        try:
                            XtX = X.T @ X
                            new_det = np.linalg.det(XtX)

                            if np.isnan(new_det) or new_det == 0:
                                new_det = 0
                        except (np.linalg.LinAlgError, FloatingPointError):
                            new_det = 0

                        # Keep if improvement
                        if new_det > best_local_det * (1 + tolerance):
                            best_value = candidate
                            best_local_det = new_det
                            improved = True

                        # Restore old value
                        design[i, j] = old_value

                    # Update to best value if improvement found
                    if best_value != design[i, j]:
                        design[i, j] = best_value
                        current_det = best_local_det

                        if current_det > best_det:
                            best_det = current_det
                            best_design = design.copy()

            # Check for convergence
            if not improved:
                break

        return best_design

    def generate_i_optimal(self,
                           prediction_points: Optional[np.ndarray] = None,
                           max_iterations: int = 1000,
                           tolerance: float = 1e-6) -> np.ndarray:
        """
        Generate I-optimal design by minimizing average prediction variance.

        I-optimality minimizes the average variance of predictions across
        the design space, making it ideal when the goal is accurate prediction
        rather than parameter estimation.

        Args:
            prediction_points: Points at which to evaluate prediction variance
                             (default: grid across design space)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Design matrix (n_runs × n_factors)
        """
        # Generate prediction grid if not provided
        if prediction_points is None:
            prediction_points = self._generate_prediction_grid(points_per_factor=5)

        # Initialize design
        design = self._initialize_design()

        # Calculate initial average prediction variance
        current_variance = self._average_prediction_variance(design, prediction_points)
        best_design = design.copy()
        best_variance = current_variance

        for iteration in range(max_iterations):
            improved = False

            for i in range(self.n_runs):
                for j in range(self.n_factors):
                    best_value = design[i, j]
                    best_local_variance = current_variance

                    candidates = self._generate_candidates(j)

                    for candidate in candidates:
                        old_value = design[i, j]
                        design[i, j] = candidate

                        # Calculate new average prediction variance
                        new_variance = self._average_prediction_variance(
                            design, prediction_points
                        )

                        # Keep if improvement (lower is better)
                        if new_variance < best_local_variance * (1 - tolerance):
                            best_value = candidate
                            best_local_variance = new_variance
                            improved = True

                        design[i, j] = old_value

                    if best_value != design[i, j]:
                        design[i, j] = best_value
                        current_variance = best_local_variance

                        if current_variance < best_variance:
                            best_variance = current_variance
                            best_design = design.copy()

            if not improved:
                break

        return best_design

    def generate_a_optimal(self, max_iterations: int = 1000,
                           tolerance: float = 1e-6) -> np.ndarray:
        """
        Generate A-optimal design by minimizing trace((X'X)^-1).

        A-optimality minimizes the average variance of the parameter estimates,
        which is equivalent to minimizing the trace of the inverse of the
        information matrix.

        Args:
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Design matrix (n_runs × n_factors)
        """
        # Initialize design
        design = self._initialize_design()

        # Calculate initial trace
        current_trace = self._trace_inv_XtX(design)
        best_design = design.copy()
        best_trace = current_trace

        for iteration in range(max_iterations):
            improved = False

            for i in range(self.n_runs):
                for j in range(self.n_factors):
                    best_value = design[i, j]
                    best_local_trace = current_trace

                    candidates = self._generate_candidates(j)

                    for candidate in candidates:
                        old_value = design[i, j]
                        design[i, j] = candidate

                        new_trace = self._trace_inv_XtX(design)

                        # Keep if improvement (lower is better)
                        if new_trace < best_local_trace * (1 - tolerance):
                            best_value = candidate
                            best_local_trace = new_trace
                            improved = True

                        design[i, j] = old_value

                    if best_value != design[i, j]:
                        design[i, j] = best_value
                        current_trace = best_local_trace

                        if current_trace < best_trace:
                            best_trace = current_trace
                            best_design = design.copy()

            if not improved:
                break

        return best_design

    def _build_model_matrix(self, design: np.ndarray) -> np.ndarray:
        """
        Build model matrix X from design points.

        For model_order=1: X = [1, x1, x2, ...]
        For model_order=2: X = [1, x1, x2, x1^2, x2^2, x1*x2, ...]
        """
        n_runs = design.shape[0]

        # Start with intercept
        X_columns = [np.ones(n_runs)]

        # Linear terms
        for j in range(self.n_factors):
            X_columns.append(design[:, j])

        if self.model_order >= 2:
            # Quadratic terms
            for j in range(self.n_factors):
                X_columns.append(design[:, j] ** 2)

            # Interaction terms
            for i, j in combinations(range(self.n_factors), 2):
                X_columns.append(design[:, i] * design[:, j])

        return np.column_stack(X_columns)

    def _initialize_design(self) -> np.ndarray:
        """Initialize design with random points in factor ranges."""
        design = np.zeros((self.n_runs, self.n_factors))

        for j, factor in enumerate(self.factors):
            low, high = self.factor_ranges[factor]
            design[:, j] = np.random.uniform(low, high, self.n_runs)

        return design

    def _generate_candidates(self, factor_index: int) -> np.ndarray:
        """Generate candidate values for a factor."""
        factor = self.factors[factor_index]
        low, high = self.factor_ranges[factor]
        return np.linspace(low, high, self.n_candidates)

    def _average_prediction_variance(self,
                                    design: np.ndarray,
                                    prediction_points: np.ndarray) -> float:
        """Calculate average prediction variance across prediction points."""
        X = self._build_model_matrix(design)

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            # Singular matrix - return large penalty
            return 1e10

        total_variance = 0
        for point in prediction_points:
            x = self._build_model_vector(point)
            variance = x.T @ XtX_inv @ x
            total_variance += variance

        return total_variance / len(prediction_points)

    def _trace_inv_XtX(self, design: np.ndarray) -> float:
        """Calculate trace of (X'X)^-1."""
        X = self._build_model_matrix(design)

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            return np.trace(XtX_inv)
        except np.linalg.LinAlgError:
            return 1e10  # Large penalty for singular matrix

    def _build_model_vector(self, point: np.ndarray) -> np.ndarray:
        """Build model vector for a single point."""
        # Intercept
        x = [1.0]

        # Linear terms
        for val in point:
            x.append(val)

        if self.model_order >= 2:
            # Quadratic terms
            for val in point:
                x.append(val ** 2)

            # Interaction terms
            for i in range(len(point)):
                for j in range(i + 1, len(point)):
                    x.append(point[i] * point[j])

        return np.array(x)

    def _generate_prediction_grid(self, points_per_factor: int = 5) -> np.ndarray:
        """Generate grid of prediction points across design space."""
        grids = []
        for factor in self.factors:
            low, high = self.factor_ranges[factor]
            grids.append(np.linspace(low, high, points_per_factor))

        # Create meshgrid
        mesh = np.meshgrid(*grids)
        points = np.column_stack([m.ravel() for m in mesh])
        return points


def evaluate_design_efficiency(design: np.ndarray,
                               factor_ranges: Dict[str, Tuple[float, float]],
                               factors: List[str],
                               model_order: int = 2) -> Dict:
    """
    Calculate efficiency metrics for a design.

    Args:
        design: Design matrix (n_runs × n_factors)
        factor_ranges: Factor ranges dict
        factors: Factor names list
        model_order: Model order (1 or 2)

    Returns:
        Dictionary with efficiency metrics
    """
    # Build model matrix
    ce = CoordinateExchange(
        n_runs=design.shape[0],
        factors=factors,
        factor_ranges=factor_ranges,
        model_order=model_order
    )

    X = ce._build_model_matrix(design)
    XtX = X.T @ X

    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return {
            "error": "Singular information matrix",
            "d_efficiency": 0.0,
            "condition_number": np.inf,
            "vif_values": [],
            "n_runs": design.shape[0],
            "n_parameters": X.shape[1]
        }

    # D-efficiency: |X'X|^(1/p) / n
    det = np.linalg.det(XtX)
    n_params = X.shape[1]
    d_efficiency = (det ** (1/n_params)) / design.shape[0]

    # Condition number (measure of collinearity)
    condition_number = np.linalg.cond(XtX)

    # VIF (Variance Inflation Factor) for each parameter
    vifs = []
    for i in range(X.shape[1]):
        vif = XtX_inv[i, i] * XtX[i, i] if XtX[i, i] != 0 else np.inf
        vifs.append(vif)

    # A-optimality: trace of (X'X)^-1
    a_criterion = np.trace(XtX_inv)

    # G-efficiency: maximum prediction variance
    g_efficiency = 1.0 / np.max(np.diag(X @ XtX_inv @ X.T))

    return {
        "d_efficiency": float(d_efficiency),
        "d_criterion": float(det),
        "a_criterion": float(a_criterion),
        "g_efficiency": float(g_efficiency),
        "condition_number": float(condition_number),
        "vif_values": [float(v) for v in vifs],
        "max_vif": float(np.max(vifs)) if vifs else 0,
        "n_runs": int(design.shape[0]),
        "n_parameters": int(X.shape[1])
    }
