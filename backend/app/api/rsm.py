from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pyDOE3 import ccdesign, bbdesign

router = APIRouter()

class RSMRequest(BaseModel):
    data: List[Dict[str, float]] = Field(..., description="Experimental data")
    factors: List[str] = Field(..., description="Factor variable names")
    response: str = Field(..., description="Response variable name")
    alpha: float = Field(0.05, description="Significance level")

class CCDRequest(BaseModel):
    n_factors: int = Field(..., description="Number of factors")
    design_type: str = Field('face-centered', description="CCD type: 'face-centered', 'rotatable', or 'inscribed'")
    n_center: int = Field(4, description="Number of center points")

class SteepestAscentRequest(BaseModel):
    current_point: Dict[str, float] = Field(..., description="Current factor levels")
    coefficients: Dict[str, float] = Field(..., description="First-order model coefficients")
    step_size: float = Field(1.0, description="Step size for path")
    n_steps: int = Field(5, description="Number of steps to calculate")

@router.post("/fit-model")
async def fit_rsm_model(request: RSMRequest):
    """
    Fit Response Surface Model (second-order polynomial)
    """
    try:
        df = pd.DataFrame(request.data)

        # Build second-order model formula
        # y ~ x1 + x2 + x1^2 + x2^2 + x1:x2
        linear_terms = " + ".join(request.factors)
        quadratic_terms = " + ".join([f"I({f}**2)" for f in request.factors])

        # Interaction terms (all pairs)
        interaction_terms = []
        for i in range(len(request.factors)):
            for j in range(i+1, len(request.factors)):
                interaction_terms.append(f"{request.factors[i]}:{request.factors[j]}")

        if interaction_terms:
            formula = f"{request.response} ~ {linear_terms} + {quadratic_terms} + {' + '.join(interaction_terms)}"
        else:
            formula = f"{request.response} ~ {linear_terms} + {quadratic_terms}"

        # Fit model
        model = ols(formula, data=df).fit()

        # Get coefficients (handle NaN values)
        coefficients = {}
        for param, coef in model.params.items():
            coefficients[param] = {
                "estimate": round(float(coef), 4) if not pd.isna(coef) else None,
                "std_error": round(float(model.bse[param]), 4) if not pd.isna(model.bse[param]) else None,
                "t_value": round(float(model.tvalues[param]), 4) if not pd.isna(model.tvalues[param]) else None,
                "p_value": round(float(model.pvalues[param]), 6) if not pd.isna(model.pvalues[param]) else None
            }

        # ANOVA for regression
        anova_table = sm.stats.anova_lm(model, typ=2)

        anova_results = {}
        for idx, row in anova_table.iterrows():
            source = str(idx)
            anova_results[source] = {
                "sum_sq": round(float(row['sum_sq']), 4),
                "df": int(row['df']),
                "F": round(float(row['F']), 4) if not pd.isna(row['F']) else None,
                "p_value": round(float(row['PR(>F)']), 6) if not pd.isna(row['PR(>F)']) else None
            }

        # Check for curvature (pure error vs lack of fit if center points exist)
        center_point_response = []
        factorial_response = []

        # Identify center points (all factors at 0 for coded designs)
        for idx, row in df.iterrows():
            is_center = all(abs(row[f]) < 0.1 for f in request.factors)
            if is_center:
                center_point_response.append(row[request.response])
            else:
                factorial_response.append(row[request.response])

        curvature_test = None
        if len(center_point_response) > 1:
            # Test for curvature
            yf_bar = np.mean(factorial_response)
            yc_bar = np.mean(center_point_response)
            nf = len(factorial_response)
            nc = len(center_point_response)

            # Curvature sum of squares
            ss_curvature = (nf * nc / (nf + nc)) * (yf_bar - yc_bar)**2

            # Pure error from center points
            ss_pure_error = sum((y - yc_bar)**2 for y in center_point_response)
            df_pure_error = nc - 1

            ms_pure_error = ss_pure_error / df_pure_error if df_pure_error > 0 else 0

            if ms_pure_error > 0:
                f_curvature = ss_curvature / ms_pure_error
                p_curvature = 1 - scipy_stats.f.cdf(f_curvature, 1, df_pure_error)

                curvature_test = {
                    "ss_curvature": round(float(ss_curvature), 4) if not pd.isna(ss_curvature) else None,
                    "f_statistic": round(float(f_curvature), 4) if not pd.isna(f_curvature) else None,
                    "p_value": round(float(p_curvature), 6) if not pd.isna(p_curvature) else None,
                    "significant_curvature": bool(p_curvature < request.alpha) if not pd.isna(p_curvature) else None
                }

        # Calculate diagnostic statistics for residual analysis
        residuals = model.resid
        fitted_values = model.fittedvalues
        standardized_residuals = model.resid_pearson

        # Studentized residuals
        influence = model.get_influence()
        studentized_residuals = influence.resid_studentized_internal

        # Leverage values (hat matrix diagonal)
        leverage = influence.hat_matrix_diag

        # Cook's distance
        cooks_d = influence.cooks_distance[0]

        # Durbin-Watson test for autocorrelation
        from statsmodels.stats.stattools import durbin_watson
        dw_statistic = durbin_watson(residuals)

        # Shapiro-Wilk test for normality
        shapiro_stat, shapiro_p = scipy_stats.shapiro(residuals)

        # Prepare diagnostic data
        diagnostics = {
            "residuals": [round(float(r), 4) for r in residuals],
            "fitted_values": [round(float(f), 4) for f in fitted_values],
            "standardized_residuals": [round(float(r), 4) for r in standardized_residuals],
            "studentized_residuals": [round(float(r), 4) for r in studentized_residuals],
            "leverage": [round(float(l), 4) for l in leverage],
            "cooks_distance": [round(float(c), 4) for c in cooks_d],
            "observed_values": [round(float(y), 4) for y in df[request.response]],
            "factor_values": df[request.factors].to_dict('list'),
            "tests": {
                "shapiro_wilk": {
                    "statistic": round(float(shapiro_stat), 4),
                    "p_value": round(float(shapiro_p), 6),
                    "interpretation": "Residuals are normally distributed" if shapiro_p > request.alpha else "Residuals are NOT normally distributed"
                },
                "durbin_watson": {
                    "statistic": round(float(dw_statistic), 4),
                    "interpretation": "No autocorrelation" if 1.5 < dw_statistic < 2.5 else "Possible autocorrelation detected"
                }
            }
        }

        # Enhanced ANOVA table with more details
        enhanced_anova = {
            "model": {
                "ss": round(float(model.ess), 4),  # Explained sum of squares
                "df": int(model.df_model),
                "ms": round(float(model.ess / model.df_model), 4),
                "f": round(float(model.fvalue), 4) if not pd.isna(model.fvalue) else None,
                "p_value": round(float(model.f_pvalue), 6) if not pd.isna(model.f_pvalue) else None
            },
            "residual": {
                "ss": round(float(model.ssr), 4),  # Residual sum of squares
                "df": int(model.df_resid),
                "ms": round(float(model.mse_resid), 4)
            },
            "total": {
                "ss": round(float(model.centered_tss), 4),  # Total sum of squares
                "df": int(model.df_model + model.df_resid)
            },
            "terms": anova_results  # Detailed breakdown by term
        }

        # Lack of fit test (if replicates exist)
        lof_test = None
        try:
            # Group by factor combinations to find replicates
            grouped = df.groupby(request.factors)[request.response]

            ss_pure_error = 0
            df_pure_error = 0

            for name, group in grouped:
                if len(group) > 1:
                    ss_pure_error += np.sum((group - group.mean())**2)
                    df_pure_error += len(group) - 1

            if df_pure_error > 0:
                ss_residual = model.ssr
                df_residual = model.df_resid

                ss_lof = ss_residual - ss_pure_error
                df_lof = df_residual - df_pure_error

                if df_lof > 0:
                    ms_lof = ss_lof / df_lof
                    ms_pure_error = ss_pure_error / df_pure_error

                    f_lof = ms_lof / ms_pure_error
                    p_lof = 1 - scipy_stats.f.cdf(f_lof, df_lof, df_pure_error)

                    lof_test = {
                        "lack_of_fit": {
                            "ss": round(float(ss_lof), 4),
                            "df": int(df_lof),
                            "ms": round(float(ms_lof), 4)
                        },
                        "pure_error": {
                            "ss": round(float(ss_pure_error), 4),
                            "df": int(df_pure_error),
                            "ms": round(float(ms_pure_error), 4)
                        },
                        "f_statistic": round(float(f_lof), 4),
                        "p_value": round(float(p_lof), 6),
                        "significant_lof": bool(p_lof < request.alpha)
                    }

                    # Add to enhanced ANOVA
                    enhanced_anova["lack_of_fit"] = lof_test["lack_of_fit"]
                    enhanced_anova["pure_error"] = lof_test["pure_error"]
        except:
            pass  # No replicates or error in calculation

        return {
            "model_type": "Response Surface Model (Second-Order)",
            "coefficients": coefficients,
            "anova_table": anova_results,  # Keep for backward compatibility
            "enhanced_anova": enhanced_anova,
            "r_squared": round(float(model.rsquared), 4) if not pd.isna(model.rsquared) else None,
            "adj_r_squared": round(float(model.rsquared_adj), 4) if not pd.isna(model.rsquared_adj) else None,
            "rmse": round(float(np.sqrt(model.mse_resid)), 4) if not pd.isna(model.mse_resid) else None,
            "curvature_test": curvature_test,
            "diagnostics": diagnostics,
            "lack_of_fit_test": lof_test
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/ccd/generate")
async def generate_ccd(request: CCDRequest):
    """
    Generate Central Composite Design (CCD)
    Supports face-centered, rotatable, and inscribed designs
    """
    try:
        from pyDOE3 import ccdesign

        k = request.n_factors

        # Determine face and alpha parameters based on design type
        if request.design_type == 'face-centered':
            face_param = 'faced'  # Face-centered design
            alpha_param = 'o'  # Use orthogonal
            alpha = 1.0  # Axial distance for face-centered
        elif request.design_type == 'rotatable':
            face_param = 'circumscribed'  # Original CCD form
            alpha_param = 'r'  # Rotatable
            alpha = (2**k) ** 0.25  # Fourth root of 2^k for rotatability
        elif request.design_type == 'inscribed':
            face_param = 'inscribed'  # Inscribed design
            alpha_param = 'o'  # Use orthogonal for inscribed
            alpha = 1.0  # Axial distance for inscribed
        else:
            raise ValueError(f"Unknown design_type: {request.design_type}")

        # Generate CCD
        design = ccdesign(k, center=(0, request.n_center), alpha=alpha_param, face=face_param)

        # Create factor names
        factor_names = [f"X{i+1}" for i in range(k)]

        # Convert to DataFrame
        design_df = pd.DataFrame(design, columns=factor_names)
        design_df['run'] = range(1, len(design_df) + 1)

        # Identify point types
        point_types = []
        for idx, row in design_df.iterrows():
            factor_values = [abs(row[f]) for f in factor_names]
            if all(v < 0.1 for v in factor_values):
                point_type = 'center'
            elif all(abs(v) in [0.0, 1.0] or abs(abs(v) - 1.0) < 0.1 for v in factor_values):
                point_type = 'factorial'
            else:
                point_type = 'axial'
            point_types.append(point_type)

        design_df['point_type'] = point_types

        # Calculate design properties
        n_factorial = 2**k
        n_axial = 2*k
        n_total = len(design_df)

        # Calculate rotatability property
        is_rotatable = request.design_type == 'rotatable'

        # Calculate orthogonality (simplified check)
        design_matrix = design[:-request.n_center] if request.n_center > 0 else design
        X = design_matrix
        XtX = np.dot(X.T, X)

        # Check if off-diagonal elements are close to zero (orthogonal)
        off_diag = XtX - np.diag(np.diag(XtX))
        is_orthogonal = np.allclose(off_diag, 0, atol=1e-10)

        return {
            "design_type": f"Central Composite Design ({request.design_type})",
            "n_factors": k,
            "alpha": round(float(alpha), 4),
            "n_runs": {
                "factorial": n_factorial,
                "axial": n_axial,
                "center": request.n_center,
                "total": n_total
            },
            "properties": {
                "rotatable": is_rotatable,
                "orthogonal": is_orthogonal,
                "alpha_description": f"Face-centered (α=1)" if request.design_type == 'face-centered'
                                   else f"Rotatable (α={(2**k)**0.25:.4f})" if request.design_type == 'rotatable'
                                   else "Inscribed (α=1)"
            },
            "factor_names": factor_names,
            "design_matrix": design_df.to_dict('records')
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/steepest-ascent")
async def steepest_ascent(request: SteepestAscentRequest):
    """
    Calculate steepest ascent/descent path from first-order model
    """
    try:
        # Get coefficients (excluding intercept)
        factors = list(request.coefficients.keys())
        coefs = np.array([request.coefficients[f] for f in factors])

        # Normalize by step size
        direction = coefs / np.linalg.norm(coefs) * request.step_size

        # Generate path
        current = np.array([request.current_point[f] for f in factors])
        path = []

        for step in range(request.n_steps + 1):
            point = current + step * direction
            point_dict = {factors[i]: round(float(point[i]), 4) for i in range(len(factors))}
            point_dict['step'] = step
            path.append(point_dict)

        # Calculate gradient magnitude
        gradient_magnitude = float(np.linalg.norm(coefs))

        return {
            "method": "Steepest Ascent",
            "factors": factors,
            "coefficients": {f: round(float(request.coefficients[f]), 4) for f in factors},
            "gradient_magnitude": round(gradient_magnitude, 4),
            "direction": {factors[i]: round(float(direction[i]), 4) for i in range(len(factors))},
            "path": path
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/lack-of-fit")
async def lack_of_fit_test(data: dict):
    """
    Perform lack-of-fit test when replications are available
    """
    try:
        df = pd.DataFrame(data['data'])
        factors = data['factors']
        response = data['response']
        alpha = data.get('alpha', 0.05)

        # Fit the model
        formula = f"{response} ~ " + " + ".join(factors)
        model = ols(formula, data=df).fit()

        # Calculate pure error from replicates
        grouped = df.groupby(factors)[response]

        ss_pure_error = 0
        df_pure_error = 0

        for name, group in grouped:
            if len(group) > 1:
                ss_pure_error += np.sum((group - group.mean())**2)
                df_pure_error += len(group) - 1

        # Lack of fit
        ss_residual = model.ssr
        df_residual = model.df_resid

        ss_lof = ss_residual - ss_pure_error
        df_lof = df_residual - df_pure_error

        if df_lof > 0 and df_pure_error > 0:
            ms_lof = ss_lof / df_lof
            ms_pure_error = ss_pure_error / df_pure_error

            f_lof = ms_lof / ms_pure_error
            p_value = 1 - scipy_stats.f.cdf(f_lof, df_lof, df_pure_error)

            return {
                "test_type": "Lack of Fit Test",
                "lack_of_fit": {
                    "ss": round(float(ss_lof), 4),
                    "df": int(df_lof),
                    "ms": round(float(ms_lof), 4)
                },
                "pure_error": {
                    "ss": round(float(ss_pure_error), 4),
                    "df": int(df_pure_error),
                    "ms": round(float(ms_pure_error), 4)
                },
                "f_statistic": round(float(f_lof), 4),
                "p_value": round(float(p_value), 6),
                "alpha": alpha,
                "significant_lof": p_value < alpha
            }
        else:
            return {
                "error": "Insufficient replication for lack-of-fit test",
                "pure_error_df": df_pure_error,
                "lof_df": df_lof
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class BoxBehnkenRequest(BaseModel):
    n_factors: int = Field(..., description="Number of factors (3-7)")
    n_center: int = Field(3, description="Number of center points")


class CanonicalAnalysisRequest(BaseModel):
    coefficients: Dict[str, Any] = Field(..., description="Model coefficients from second-order model")
    factors: List[str] = Field(..., description="Factor names")


class OptimizationRequest(BaseModel):
    coefficients: Dict[str, Any] = Field(..., description="Model coefficients")
    factors: List[str] = Field(..., description="Factor names")
    target: str = Field('maximize', description="'maximize' or 'minimize'")
    constraints: Optional[Dict[str, List[float]]] = Field(None, description="Factor constraints {factor: [min, max]}")
    variance_estimate: Optional[float] = Field(None, description="Residual variance (MSE) for interval calculation")


@router.post("/box-behnken/generate")
async def generate_box_behnken(request: BoxBehnkenRequest):
    """
    Generate Box-Behnken Design
    Efficient three-level design for 3-7 factors
    """
    try:
        from pyDOE3 import bbdesign

        k = request.n_factors

        if k < 3 or k > 7:
            raise ValueError("Box-Behnken designs require 3-7 factors")

        # Generate Box-Behnken design
        design = bbdesign(k, center=request.n_center)

        # Create factor names
        factor_names = [f"X{i+1}" for i in range(k)]

        # Convert to DataFrame
        design_df = pd.DataFrame(design, columns=factor_names)
        design_df['run'] = range(1, len(design_df) + 1)

        # Identify point types
        point_types = []
        for idx, row in design_df.iterrows():
            factor_values = [abs(row[f]) for f in factor_names]
            if all(v < 0.1 for v in factor_values):
                point_type = 'center'
            else:
                point_type = 'edge'  # Box-Behnken points are on edges of design space
            point_types.append(point_type)

        design_df['point_type'] = point_types

        # Calculate number of runs
        # Box-Behnken: 2k(k-1) + center points
        n_edge = 2 * k * (k - 1)
        n_total = len(design_df)

        return {
            "design_type": "Box-Behnken Design",
            "n_factors": k,
            "n_runs": {
                "edge_points": n_edge,
                "center": request.n_center,
                "total": n_total
            },
            "properties": {
                "levels": 3,
                "rotatable": False,
                "description": "Efficient three-level design with no corner points"
            },
            "factor_names": factor_names,
            "design_matrix": design_df.to_dict('records')
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/canonical-analysis")
async def canonical_analysis(request: CanonicalAnalysisRequest):
    """
    Perform canonical analysis to find stationary point and characterize surface
    """
    try:
        k = len(request.factors)

        # Extract linear coefficients (b)
        b = np.array([request.coefficients.get(f, 0.0) for f in request.factors])

        # Extract quadratic coefficients matrix (B)
        B = np.zeros((k, k))
        for i, f1 in enumerate(request.factors):
            # Diagonal (pure quadratic)
            quad_key = f"I({f1}**2)"
            if quad_key in request.coefficients:
                B[i, i] = request.coefficients[quad_key]

            # Off-diagonal (interactions)
            for j, f2 in enumerate(request.factors):
                if i < j:
                    int_key = f"{f1}:{f2}"
                    if int_key in request.coefficients:
                        # Divide by 2 because interaction appears once in model but twice in quadratic form
                        B[i, j] = B[j, i] = request.coefficients[int_key] / 2

        # Find stationary point: x_s = -0.5 * B^(-1) * b
        try:
            B_inv = np.linalg.inv(B)
            x_s = -0.5 * np.dot(B_inv, b)

            stationary_point = {request.factors[i]: round(float(x_s[i]), 4) for i in range(k)}

            # Eigenanalysis to characterize the surface
            eigenvalues, eigenvectors = np.linalg.eig(B)

            # Sort by eigenvalue magnitude
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Characterize stationary point
            if all(eigenvalues < 0):
                surface_type = "Maximum"
            elif all(eigenvalues > 0):
                surface_type = "Minimum"
            else:
                surface_type = "Saddle Point"

            # Canonical axes
            canonical_axes = []
            for i in range(k):
                axis = {
                    "eigenvalue": round(float(eigenvalues[i]), 4),
                    "direction": {request.factors[j]: round(float(eigenvectors[j, i]), 4) for j in range(k)}
                }
                canonical_axes.append(axis)

            return {
                "stationary_point": stationary_point,
                "surface_type": surface_type,
                "eigenvalues": [round(float(e), 4) for e in eigenvalues],
                "canonical_axes": canonical_axes,
                "B_matrix": B.tolist(),
                "interpretation": f"The response surface has a {surface_type.lower()} at the stationary point."
            }

        except np.linalg.LinAlgError:
            return {
                "error": "Singular matrix - cannot find stationary point",
                "B_matrix": B.tolist()
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/optimize")
async def optimize_response(request: OptimizationRequest):
    """
    Find optimal factor settings to maximize or minimize response
    Uses second-order model predictions
    """
    try:
        from scipy.optimize import differential_evolution

        k = len(request.factors)

        # Build prediction function
        def predict(x):
            # Intercept
            y = request.coefficients.get('Intercept', 0.0)

            # Linear terms
            for i, factor in enumerate(request.factors):
                y += request.coefficients.get(factor, 0.0) * x[i]

            # Quadratic terms
            for i, factor in enumerate(request.factors):
                quad_key = f"I({factor}**2)"
                y += request.coefficients.get(quad_key, 0.0) * x[i]**2

            # Interaction terms
            for i in range(k):
                for j in range(i+1, k):
                    int_key = f"{request.factors[i]}:{request.factors[j]}"
                    y += request.coefficients.get(int_key, 0.0) * x[i] * x[j]

            return y

        # Objective function
        if request.target == 'maximize':
            objective = lambda x: -predict(x)  # Negate for maximization
        else:
            objective = predict

        # Bounds
        if request.constraints:
            bounds = [request.constraints.get(f, [-2, 2]) for f in request.factors]
        else:
            bounds = [(-2, 2)] * k  # Default coded bounds

        # Optimize using differential evolution (global optimizer)
        result = differential_evolution(objective, bounds, seed=42, maxiter=1000)

        optimal_point = {request.factors[i]: round(float(result.x[i]), 4) for i in range(k)}
        optimal_response = round(float(predict(result.x)), 4)

        # Calculate confidence and prediction intervals if variance estimate is provided
        intervals = None
        if request.variance_estimate:
            # Use approximate intervals based on standard error
            # For coded designs, typical leverages range from 0.1 to 0.4
            # Use conservative estimate of h = 0.2 for optimal point
            se = np.sqrt(request.variance_estimate)
            t_value = 1.96  # Approximate 95% CI using normal approximation

            # Confidence interval (for mean response)
            ci_margin = t_value * se * np.sqrt(0.2)  # assuming moderate leverage

            # Prediction interval (for individual observation)
            pi_margin = t_value * se * np.sqrt(1 + 0.2)

            intervals = {
                "confidence_interval": {
                    "lower": round(float(optimal_response - ci_margin), 4),
                    "upper": round(float(optimal_response + ci_margin), 4),
                    "level": 0.95
                },
                "prediction_interval": {
                    "lower": round(float(optimal_response - pi_margin), 4),
                    "upper": round(float(optimal_response + pi_margin), 4),
                    "level": 0.95
                }
            }

        return {
            "target": request.target,
            "optimal_point": optimal_point,
            "predicted_response": optimal_response,
            "intervals": intervals,
            "success": result.success,
            "method": "Differential Evolution"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class ConfirmationRunsRequest(BaseModel):
    optimal_point: Dict[str, float] = Field(..., description="Optimal factor levels")
    coefficients: Dict[str, Any] = Field(..., description="Model coefficients")
    factors: List[str] = Field(..., description="Factor names")
    n_runs: int = Field(3, description="Number of confirmation runs to recommend")
    variance_estimate: Optional[float] = Field(None, description="Estimated variance from model")


class RidgeAnalysisRequest(BaseModel):
    coefficients: Dict[str, Any] = Field(..., description="Model coefficients from second-order model")
    factors: List[str] = Field(..., description="Factor names")
    target_response: float = Field(..., description="Target response value for ridge analysis")
    n_points: int = Field(50, description="Number of points for ridge contours")


class DesirabilitySpec(BaseModel):
    response_name: str = Field(..., description="Name of the response variable")
    coefficients: Dict[str, Any] = Field(..., description="Model coefficients for this response")
    goal: str = Field(..., description="'maximize', 'minimize', or 'target'")
    lower_bound: Optional[float] = Field(None, description="Lower acceptable value")
    upper_bound: Optional[float] = Field(None, description="Upper acceptable value")
    target: Optional[float] = Field(None, description="Target value (for 'target' goal)")
    weight: float = Field(1.0, description="Importance weight for this response")
    importance: float = Field(1.0, description="Shape parameter (s) for desirability function")


class DesirabilityRequest(BaseModel):
    responses: List[DesirabilitySpec] = Field(..., description="List of response specifications")
    factors: List[str] = Field(..., description="Factor names")
    constraints: Optional[Dict[str, List[float]]] = Field(None, description="Factor constraints {factor: [min, max]}")


class DesignAugmentationRequest(BaseModel):
    current_design: List[Dict[str, float]] = Field(..., description="Existing design points")
    factors: List[str] = Field(..., description="Factor names")
    n_points: int = Field(3, description="Number of points to add")
    strategy: str = Field('steep-ascent', description="'steep-ascent', 'model-based', or 'space-filling'")
    coefficients: Optional[Dict[str, Any]] = Field(None, description="Model coefficients (for model-based strategy)")
    target: Optional[str] = Field('maximize', description="'maximize' or 'minimize' (for steep-ascent)")


class ConstrainedOptimizationRequest(BaseModel):
    coefficients: Dict[str, Any] = Field(..., description="Model coefficients")
    factors: List[str] = Field(..., description="Factor names")
    target: str = Field('maximize', description="'maximize' or 'minimize'")
    linear_constraints: Optional[List[Dict[str, Any]]] = Field(None, description="Linear constraints: [{coefficients: {}, bound: value, type: 'ineq'/'eq'}]")
    bounds: Optional[Dict[str, List[float]]] = Field(None, description="Box constraints {factor: [min, max]}")
    variance_estimate: Optional[float] = Field(None, description="Residual variance (MSE) for interval calculation")


@router.post("/confirmation-runs")
async def calculate_confirmation_runs(request: ConfirmationRunsRequest):
    """
    Calculate recommended confirmation runs at optimal point
    Includes prediction interval and recommended replications
    """
    try:
        k = len(request.factors)

        # Build prediction function
        def predict(x):
            y = request.coefficients.get('Intercept', 0.0)
            for i, factor in enumerate(request.factors):
                y += request.coefficients.get(factor, 0.0) * x[i]
            for i, factor in enumerate(request.factors):
                quad_key = f"I({factor}**2)"
                y += request.coefficients.get(quad_key, 0.0) * x[i]**2
            for i in range(k):
                for j in range(i+1, k):
                    int_key = f"{request.factors[i]}:{request.factors[j]}"
                    y += request.coefficients.get(int_key, 0.0) * x[i] * x[j]
            return y

        # Predicted response at optimal point
        x_opt = np.array([request.optimal_point[f] for f in request.factors])
        y_pred = predict(x_opt)

        # Calculate prediction interval if variance provided
        if request.variance_estimate and request.variance_estimate > 0:
            # Standard error of prediction (simplified - assumes center point)
            # For more accurate, would need X'X matrix from original fit
            se_pred = np.sqrt(request.variance_estimate * (1 + 1/request.n_runs))

            # 95% prediction interval (t-distribution with large df ≈ z-distribution)
            from scipy.stats import t
            t_value = t.ppf(0.975, df=20)  # Assuming ~20 df, adjust as needed

            lower_bound = y_pred - t_value * se_pred
            upper_bound = y_pred + t_value * se_pred

            prediction_interval = {
                "lower": round(float(lower_bound), 4),
                "upper": round(float(upper_bound), 4),
                "confidence_level": 0.95
            }
        else:
            prediction_interval = None

        # Generate confirmation run design
        confirmation_runs = []
        for i in range(request.n_runs):
            run = {
                "run_number": i + 1,
                **{f: round(float(request.optimal_point[f]), 4) for f in request.factors},
                "predicted_response": round(float(y_pred), 4)
            }
            confirmation_runs.append(run)

        # Calculate statistical power (simplified)
        if request.variance_estimate:
            # Detectable difference with 80% power
            effect_size = 2.8 * np.sqrt(request.variance_estimate / request.n_runs)
        else:
            effect_size = None

        return {
            "optimal_point": {f: round(float(request.optimal_point[f]), 4) for f in request.factors},
            "predicted_response": round(float(y_pred), 4),
            "confirmation_runs": confirmation_runs,
            "n_runs": request.n_runs,
            "prediction_interval": prediction_interval,
            "minimum_detectable_effect": round(float(effect_size), 4) if effect_size else None,
            "recommendations": [
                f"Run {request.n_runs} confirmation experiments at the optimal point",
                "Monitor actual response vs. predicted response",
                "If actual response falls outside prediction interval, investigate model adequacy",
                "Consider additional center point replicates if high variability observed"
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ridge-analysis")
async def ridge_analysis(request: RidgeAnalysisRequest):
    """
    Perform ridge analysis to find factor combinations achieving target response
    Shows stable operating regions
    """
    try:
        k = len(request.factors)

        # Extract linear coefficients (b) and quadratic matrix (B)
        b = np.array([request.coefficients.get(f, 0.0) for f in request.factors])

        B = np.zeros((k, k))
        for i, f1 in enumerate(request.factors):
            quad_key = f"I({f1}**2)"
            if quad_key in request.coefficients:
                B[i, i] = request.coefficients[quad_key]
            for j, f2 in enumerate(request.factors):
                if i < j:
                    int_key = f"{f1}:{f2}"
                    if int_key in request.coefficients:
                        B[i, j] = B[j, i] = request.coefficients[int_key] / 2

        intercept = request.coefficients.get('Intercept', 0.0)

        # Find stationary point
        try:
            B_inv = np.linalg.inv(B)
            x_s = -0.5 * np.dot(B_inv, b)
            stationary_point = {request.factors[i]: round(float(x_s[i]), 4) for i in range(k)}

            # Predicted response at stationary point
            y_s = intercept + np.dot(b, x_s) + np.dot(x_s, np.dot(B, x_s))
        except np.linalg.LinAlgError:
            stationary_point = None
            y_s = None

        # Calculate ridge of target response
        # For 2D case, we can calculate the ridge contour
        if k == 2:
            # Generate ridge points
            ridge_points = []
            theta_values = np.linspace(0, 2*np.pi, request.n_points)

            for theta in theta_values:
                # Direction vector
                d = np.array([np.cos(theta), np.sin(theta)])

                # Solve for radius R where y(R*d) = target_response
                # This is a quadratic equation in R
                a_coef = np.dot(d, np.dot(B, d))
                b_coef = np.dot(b, d)
                c_coef = intercept - request.target_response

                # Solve quadratic: a*R^2 + b*R + c = 0
                discriminant = b_coef**2 - 4*a_coef*c_coef

                if discriminant >= 0 and abs(a_coef) > 1e-10:
                    r1 = (-b_coef + np.sqrt(discriminant)) / (2*a_coef)
                    r2 = (-b_coef - np.sqrt(discriminant)) / (2*a_coef)

                    # Use positive radius in reasonable range
                    for r in [r1, r2]:
                        if 0 < r < 5:  # Reasonable range for coded variables
                            point = r * d
                            ridge_points.append({
                                request.factors[0]: round(float(point[0]), 4),
                                request.factors[1]: round(float(point[1]), 4),
                                "radius": round(float(r), 4)
                            })
                            break

            # Calculate distance from stationary point to target
            distance_to_stationary = None
            if stationary_point and y_s is not None:
                distance_to_stationary = round(float(abs(y_s - request.target_response)), 4)
        else:
            ridge_points = []
            distance_to_stationary = None

        return {
            "target_response": request.target_response,
            "stationary_point": stationary_point,
            "stationary_response": round(float(y_s), 4) if y_s is not None else None,
            "distance_to_target": distance_to_stationary,
            "ridge_points": ridge_points,
            "n_points": len(ridge_points),
            "interpretation": (
                f"Ridge analysis shows factor combinations achieving response = {request.target_response}. "
                f"Points on the ridge represent stable operating conditions with equivalent response."
                if ridge_points else
                "Ridge analysis could not be computed. Target response may be outside feasible region."
            )
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/desirability-optimization")
async def desirability_optimization(request: DesirabilityRequest):
    """
    Multi-response optimization using desirability functions
    Combines multiple responses into a single composite desirability metric
    """
    try:
        from scipy.optimize import differential_evolution

        k = len(request.factors)

        # Build prediction functions for each response
        def predict_response(x, coefficients):
            y = coefficients.get('Intercept', 0.0)
            for i, factor in enumerate(request.factors):
                y += coefficients.get(factor, 0.0) * x[i]
            for i, factor in enumerate(request.factors):
                quad_key = f"I({factor}**2)"
                y += coefficients.get(quad_key, 0.0) * x[i]**2
            for i in range(k):
                for j in range(i+1, k):
                    int_key = f"{request.factors[i]}:{request.factors[j]}"
                    y += coefficients.get(int_key, 0.0) * x[i] * x[j]
            return y

        # Calculate individual desirability
        def calculate_desirability(y, spec: DesirabilitySpec):
            if spec.goal == 'maximize':
                L = spec.lower_bound if spec.lower_bound is not None else 0
                U = spec.upper_bound if spec.upper_bound is not None else y + 1
                if y <= L:
                    return 0.0
                elif y >= U:
                    return 1.0
                else:
                    return ((y - L) / (U - L)) ** spec.importance

            elif spec.goal == 'minimize':
                L = spec.lower_bound if spec.lower_bound is not None else y - 1
                U = spec.upper_bound if spec.upper_bound is not None else 0
                if y >= U:
                    return 0.0
                elif y <= L:
                    return 1.0
                else:
                    return ((U - y) / (U - L)) ** spec.importance

            elif spec.goal == 'target':
                T = spec.target if spec.target is not None else (spec.lower_bound + spec.upper_bound) / 2
                L = spec.lower_bound if spec.lower_bound is not None else T - 1
                U = spec.upper_bound if spec.upper_bound is not None else T + 1

                if y < L or y > U:
                    return 0.0
                elif y <= T:
                    return ((y - L) / (T - L)) ** spec.importance
                else:
                    return ((U - y) / (U - T)) ** spec.importance

            return 0.0

        # Composite desirability function (geometric mean with weights)
        def composite_desirability(x):
            desirabilities = []
            weights = []
            for spec in request.responses:
                y_pred = predict_response(x, spec.coefficients)
                d = calculate_desirability(y_pred, spec)
                desirabilities.append(d)
                weights.append(spec.weight)

            # Weighted geometric mean
            if any(d == 0 for d in desirabilities):
                return 0.0

            total_weight = sum(weights)
            D = 1.0
            for d, w in zip(desirabilities, weights):
                D *= d ** (w / total_weight)

            return D

        # Optimize (maximize composite desirability)
        objective = lambda x: -composite_desirability(x)

        # Bounds
        if request.constraints:
            bounds = [request.constraints.get(f, [-2, 2]) for f in request.factors]
        else:
            bounds = [(-2, 2)] * k

        result = differential_evolution(objective, bounds, seed=42, maxiter=1000)

        optimal_x = result.x
        optimal_point = {request.factors[i]: round(float(optimal_x[i]), 4) for i in range(k)}

        # Calculate individual responses and desirabilities at optimum
        individual_results = []
        for spec in request.responses:
            y_pred = predict_response(optimal_x, spec.coefficients)
            d = calculate_desirability(y_pred, spec)
            individual_results.append({
                "response_name": spec.response_name,
                "predicted_value": round(float(y_pred), 4),
                "desirability": round(float(d), 4),
                "goal": spec.goal,
                "weight": spec.weight
            })

        composite_d = composite_desirability(optimal_x)

        return {
            "optimal_point": optimal_point,
            "composite_desirability": round(float(composite_d), 4),
            "individual_results": individual_results,
            "success": result.success,
            "method": "Desirability Function Optimization",
            "interpretation": (
                f"Composite desirability = {round(float(composite_d), 4)} "
                f"(0 = unacceptable, 1 = ideal). "
                "This represents the best compromise among all {len(request.responses)} responses."
            )
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/design-augmentation")
async def design_augmentation(request: DesignAugmentationRequest):
    """
    Sequential design augmentation - add follow-up runs strategically
    """
    try:
        k = len(request.factors)
        current_points = np.array([[pt[f] for f in request.factors] for pt in request.current_design])

        new_points = []

        if request.strategy == 'space-filling':
            # Latin Hypercube Sampling for space-filling
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=k, seed=42)
            sample = sampler.random(n=request.n_points)
            # Scale to [-2, 2] range (coded units)
            scaled = qmc.scale(sample, [-2]*k, [2]*k)
            new_points = scaled.tolist()

        elif request.strategy == 'steep-ascent' and request.coefficients:
            # Add points along steepest ascent direction
            linear_coefs = np.array([request.coefficients.get(f, 0.0) for f in request.factors])
            direction = linear_coefs / np.linalg.norm(linear_coefs) if np.linalg.norm(linear_coefs) > 0 else np.zeros(k)

            # Start from center or last point
            start = current_points[-1] if len(current_points) > 0 else np.zeros(k)

            for i in range(1, request.n_points + 1):
                point = start + (i * 0.5 * direction)
                new_points.append(point.tolist())

        elif request.strategy == 'model-based' and request.coefficients:
            # Add points near predicted optimum
            def predict(x):
                y = request.coefficients.get('Intercept', 0.0)
                for i, factor in enumerate(request.factors):
                    y += request.coefficients.get(factor, 0.0) * x[i]
                    quad_key = f"I({factor}**2)"
                    y += request.coefficients.get(quad_key, 0.0) * x[i]**2
                for i in range(k):
                    for j in range(i+1, k):
                        int_key = f"{request.factors[i]}:{request.factors[j]}"
                        y += request.coefficients.get(int_key, 0.0) * x[i] * x[j]
                return y

            from scipy.optimize import differential_evolution
            objective = lambda x: -predict(x) if request.target == 'maximize' else predict
            result = differential_evolution(objective, [(-2, 2)] * k, seed=42, maxiter=500)

            # Add points around optimum
            optimum = result.x
            for i in range(request.n_points):
                # Add small random perturbations
                point = optimum + np.random.normal(0, 0.3, k)
                new_points.append(point.tolist())

        # Format output
        augmented_design = []
        for i, point in enumerate(new_points):
            design_point = {request.factors[j]: round(float(point[j]), 4) for j in range(k)}
            design_point['run_number'] = len(request.current_design) + i + 1
            augmented_design.append(design_point)

        return {
            "strategy": request.strategy,
            "n_new_points": len(augmented_design),
            "new_points": augmented_design,
            "total_runs": len(request.current_design) + len(augmented_design),
            "recommendation": f"Added {len(augmented_design)} runs using {request.strategy} strategy"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/constrained-optimization")
async def constrained_optimization(request: ConstrainedOptimizationRequest):
    """
    Optimization with linear and box constraints
    """
    try:
        from scipy.optimize import minimize, Bounds, LinearConstraint

        k = len(request.factors)

        def predict(x):
            y = request.coefficients.get('Intercept', 0.0)
            for i, factor in enumerate(request.factors):
                y += request.coefficients.get(factor, 0.0) * x[i]
                quad_key = f"I({factor}**2)"
                y += request.coefficients.get(quad_key, 0.0) * x[i]**2
            for i in range(k):
                for j in range(i+1, k):
                    int_key = f"{request.factors[i]}:{request.factors[j]}"
                    y += request.coefficients.get(int_key, 0.0) * x[i] * x[j]
            return y

        objective = lambda x: -predict(x) if request.target == 'maximize' else predict

        # Box constraints
        if request.bounds:
            lb = [request.bounds.get(f, [-2, 2])[0] for f in request.factors]
            ub = [request.bounds.get(f, [-2, 2])[1] for f in request.factors]
        else:
            lb, ub = [-2]*k, [2]*k

        bounds = Bounds(lb, ub)

        # Linear constraints
        constraints = []
        if request.linear_constraints:
            for lc in request.linear_constraints:
                A = [lc['coefficients'].get(f, 0.0) for f in request.factors]
                constraint = LinearConstraint(A, -np.inf if lc['type'] == 'ineq' else lc['bound'], lc['bound'])
                constraints.append(constraint)

        # Optimize
        x0 = np.zeros(k)  # Start at center
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints if constraints else ())

        optimal_point = {request.factors[i]: round(float(result.x[i]), 4) for i in range(k)}
        predicted_response = round(float(predict(result.x)), 4)

        # Calculate confidence and prediction intervals if variance estimate is provided
        intervals = None
        if request.variance_estimate:
            se = np.sqrt(request.variance_estimate)
            t_value = 1.96

            # Confidence interval (for mean response)
            ci_margin = t_value * se * np.sqrt(0.2)

            # Prediction interval (for individual observation)
            pi_margin = t_value * se * np.sqrt(1 + 0.2)

            intervals = {
                "confidence_interval": {
                    "lower": round(float(predicted_response - ci_margin), 4),
                    "upper": round(float(predicted_response + ci_margin), 4),
                    "level": 0.95
                },
                "prediction_interval": {
                    "lower": round(float(predicted_response - pi_margin), 4),
                    "upper": round(float(predicted_response + pi_margin), 4),
                    "level": 0.95
                }
            }

        return {
            "optimal_point": optimal_point,
            "predicted_response": predicted_response,
            "intervals": intervals,
            "success": result.success,
            "method": "Sequential Least Squares Programming (SLSQP)",
            "n_constraints": len(request.linear_constraints) if request.linear_constraints else 0,
            "interpretation": f"Optimal solution found respecting all constraints. Response = {predicted_response}"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Feature 7: Mixture Designs ====================

class MixtureDesignRequest(BaseModel):
    n_components: int = Field(..., description="Number of mixture components")
    design_type: str = Field(..., description="Design type: 'simplex-lattice' or 'simplex-centroid'")
    degree: Optional[int] = Field(2, description="Degree for simplex-lattice (2, 3, or 4)")
    component_names: Optional[List[str]] = Field(None, description="Names for components")

class MixtureModelRequest(BaseModel):
    data: List[Dict[str, float]] = Field(..., description="Experimental data")
    components: List[str] = Field(..., description="Component names")
    response: str = Field(..., description="Response variable name")
    model_type: str = Field("scheffe", description="Model type: 'scheffe' (linear, quadratic, cubic)")
    degree: int = Field(2, description="Polynomial degree (1, 2, or 3)")

class MixtureOptimizationRequest(BaseModel):
    coefficients: Dict[str, float] = Field(..., description="Model coefficients")
    components: List[str] = Field(..., description="Component names")
    target: str = Field(..., description="'maximize' or 'minimize'")
    lower_bounds: Optional[Dict[str, float]] = Field(None, description="Lower bounds for each component")
    upper_bounds: Optional[Dict[str, float]] = Field(None, description="Upper bounds for each component")


def generate_simplex_lattice(n_components: int, degree: int = 2):
    """
    Generate Simplex-Lattice design of degree m
    Each component takes (m+1) equally spaced levels from 0 to 1
    """
    import itertools

    # Generate lattice points
    levels = np.linspace(0, 1, degree + 1)

    # Generate all combinations that sum to 1
    design_points = []

    # Use itertools to generate all combinations with replacement
    for combo in itertools.combinations_with_replacement(range(degree + 1), n_components):
        point = np.array([levels[i] for i in combo])
        # Check if proportions sum to approximately 1
        if np.abs(np.sum(point) - 1.0) < 1e-10:
            # Generate all permutations of this combination
            for perm in itertools.permutations(point):
                perm_array = np.array(perm)
                # Check if this permutation is unique
                is_duplicate = any(np.allclose(perm_array, dp) for dp in design_points)
                if not is_duplicate:
                    design_points.append(perm_array)

    return np.array(design_points)


def generate_simplex_centroid(n_components: int):
    """
    Generate Simplex-Centroid design
    Includes all vertices, edges, faces, and overall centroid
    """
    import itertools

    design_points = []

    # Generate all non-empty subsets of components
    for r in range(1, n_components + 1):
        for subset in itertools.combinations(range(n_components), r):
            # Equal proportions for selected components, 0 for others
            point = np.zeros(n_components)
            point[list(subset)] = 1.0 / len(subset)
            design_points.append(point)

    return np.array(design_points)


@router.post("/mixture-design/generate")
async def generate_mixture_design(request: MixtureDesignRequest):
    """
    Generate mixture experimental design (Simplex-Lattice or Simplex-Centroid)
    """
    try:
        n = request.n_components

        if n < 2:
            raise ValueError("Need at least 2 components for mixture design")

        if request.design_type == "simplex-lattice":
            if request.degree not in [2, 3, 4]:
                raise ValueError("Degree must be 2, 3, or 4 for simplex-lattice design")
            design_matrix = generate_simplex_lattice(n, request.degree)
            design_name = f"Simplex-Lattice ({request.degree})"
        elif request.design_type == "simplex-centroid":
            design_matrix = generate_simplex_centroid(n)
            design_name = "Simplex-Centroid"
        else:
            raise ValueError(f"Unknown design type: {request.design_type}")

        # Create component names if not provided
        if request.component_names and len(request.component_names) == n:
            comp_names = request.component_names
        else:
            comp_names = [f"X{i+1}" for i in range(n)]

        # Convert to list of dicts
        design_data = []
        for i, point in enumerate(design_matrix):
            run = {"run": i + 1}
            for j, comp_name in enumerate(comp_names):
                run[comp_name] = round(point[j], 6)
            design_data.append(run)

        return {
            "design_type": design_name,
            "n_components": n,
            "n_runs": len(design_matrix),
            "components": comp_names,
            "design_matrix": design_data,
            "properties": {
                "constraint": "Components sum to 1.0",
                "description": f"{design_name} design for {n}-component mixture"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/mixture-design/fit-model")
async def fit_mixture_model(request: MixtureModelRequest):
    """
    Fit Scheffé mixture model
    """
    try:
        df = pd.DataFrame(request.data)
        components = request.components
        response = request.response
        degree = request.degree

        # Verify components sum to approximately 1
        component_sums = df[components].sum(axis=1)
        if not np.allclose(component_sums, 1.0, atol=0.01):
            raise ValueError("Component proportions must sum to 1.0 for each run")

        # Build Scheffé model formula based on degree
        terms = []

        # Linear terms (degree >= 1)
        if degree >= 1:
            # For Scheffé model, we exclude intercept and use pure mixture terms
            terms.extend(components)

        # Quadratic terms (degree >= 2): all pairwise interactions
        if degree >= 2:
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    terms.append(f"{components[i]}:{components[j]}")

        # Cubic terms (degree >= 3): three-way interactions
        if degree >= 3:
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    for k in range(j + 1, len(components)):
                        terms.append(f"{components[i]}:{components[j]}:{components[k]}")

        # Build formula without intercept (mixture constraint handles it)
        formula = f"{response} ~ {' + '.join(terms)} - 1"

        # Fit model
        model = ols(formula, data=df).fit()

        # Extract coefficients
        coefficients = {}
        for term, coef, se, t_val, p_val in zip(
            model.model.exog_names,
            model.params,
            model.bse,
            model.tvalues,
            model.pvalues
        ):
            coefficients[term] = {
                "estimate": round(float(coef), 6),
                "std_error": round(float(se), 6),
                "t_value": round(float(t_val), 4),
                "p_value": round(float(p_val), 6)
            }

        # Model statistics
        r_squared = float(model.rsquared)
        adj_r_squared = float(model.rsquared_adj)
        rmse = float(np.sqrt(model.mse_resid))

        # ANOVA table
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_dict = {}
        for idx, row in anova_table.iterrows():
            anova_dict[idx] = {
                "sum_sq": round(float(row.get('sum_sq', 0)), 6),
                "df": int(row.get('df', 0)),
                "F": round(float(row.get('F', 0)), 4) if pd.notna(row.get('F')) else None,
                "PR(>F)": round(float(row.get('PR(>F)', 0)), 6) if pd.notna(row.get('PR(>F)')) else None
            }

        return {
            "model_type": f"Scheffé {['Linear', 'Quadratic', 'Cubic'][degree-1]} Mixture Model",
            "formula": formula,
            "coefficients": coefficients,
            "r_squared": round(r_squared, 4),
            "adj_r_squared": round(adj_r_squared, 4),
            "rmse": round(rmse, 4),
            "anova": anova_dict,
            "n_obs": int(model.nobs),
            "df_resid": int(model.df_resid)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/mixture-design/optimize")
async def optimize_mixture(request: MixtureOptimizationRequest):
    """
    Optimize mixture response subject to component constraints
    """
    try:
        from scipy.optimize import minimize, LinearConstraint

        components = request.components
        n_comp = len(components)

        # Build prediction function from coefficients
        def predict(x_dict):
            """Predict response from component proportions"""
            y = 0.0

            # Process each term in the model
            for term, coef in request.coefficients.items():
                if ':' in term:
                    # Interaction term
                    parts = term.split(':')
                    value = coef
                    for part in parts:
                        value *= x_dict.get(part, 0)
                    y += value
                else:
                    # Linear term
                    y += coef * x_dict.get(term, 0)

            return y

        def objective(x):
            """Objective function for optimization"""
            x_dict = {comp: x[i] for i, comp in enumerate(components)}
            pred = predict(x_dict)
            return -pred if request.target == "maximize" else pred

        # Constraints
        constraints = []

        # Equality constraint: components must sum to 1
        A_eq = np.ones((1, n_comp))
        constraints.append(LinearConstraint(A_eq, 1.0, 1.0))

        # Box constraints (bounds for each component)
        lower_bounds = []
        upper_bounds = []
        for comp in components:
            lb = request.lower_bounds.get(comp, 0.0) if request.lower_bounds else 0.0
            ub = request.upper_bounds.get(comp, 1.0) if request.upper_bounds else 1.0
            lower_bounds.append(lb)
            upper_bounds.append(ub)

        bounds = list(zip(lower_bounds, upper_bounds))

        # Initial guess: equal proportions
        x0 = np.ones(n_comp) / n_comp

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Build optimal point dict
        optimal_point = {comp: round(float(result.x[i]), 6) for i, comp in enumerate(components)}

        # Calculate predicted response
        predicted_response = float(-result.fun if request.target == "maximize" else result.fun)

        return {
            "target": request.target,
            "optimal_point": optimal_point,
            "predicted_response": round(predicted_response, 4),
            "method": "SLSQP with mixture constraint",
            "success": True,
            "verification": {
                "components_sum": round(sum(optimal_point.values()), 6),
                "meets_constraint": abs(sum(optimal_point.values()) - 1.0) < 1e-6
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Feature 8: Robust Parameter Design ====================

class RobustDesignRequest(BaseModel):
    n_control_factors: int = Field(..., description="Number of control factors")
    n_noise_factors: int = Field(..., description="Number of noise factors")
    control_design_type: str = Field("orthogonal_array", description="Control array type")
    noise_design_type: str = Field("full_factorial", description="Noise array type")
    control_factor_names: Optional[List[str]] = Field(None, description="Names for control factors")
    noise_factor_names: Optional[List[str]] = Field(None, description="Names for noise factors")

class RobustAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Experimental data with control, noise, and response")
    control_factors: List[str] = Field(..., description="Control factor names")
    noise_factors: List[str] = Field(..., description="Noise factor names")
    response: str = Field(..., description="Response variable name")
    quality_characteristic: str = Field("smaller-is-better", description="'smaller-is-better', 'larger-is-better', or 'nominal-is-best'")
    target_value: Optional[float] = Field(None, description="Target value (for nominal-is-best)")


def generate_orthogonal_array_L8():
    """Generate L8 Orthogonal Array (7 factors, 2 levels)"""
    return np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1,  1,  1,  1,  1],
        [-1,  1,  1, -1, -1,  1,  1],
        [-1,  1,  1,  1,  1, -1, -1],
        [ 1, -1,  1, -1,  1, -1,  1],
        [ 1, -1,  1,  1, -1,  1, -1],
        [ 1,  1, -1, -1,  1,  1, -1],
        [ 1,  1, -1,  1, -1, -1,  1]
    ])


def generate_orthogonal_array_L4():
    """Generate L4 Orthogonal Array (3 factors, 2 levels)"""
    return np.array([
        [-1, -1, -1],
        [-1,  1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1]
    ])


@router.post("/robust-design/generate")
async def generate_robust_design(request: RobustDesignRequest):
    """
    Generate robust parameter design (inner-outer array)
    """
    try:
        n_control = request.n_control_factors
        n_noise = request.n_noise_factors

        # Generate control factor design (inner array)
        if request.control_design_type == "orthogonal_array":
            if n_control <= 3:
                control_array = generate_orthogonal_array_L4()[:, :n_control]
            elif n_control <= 7:
                control_array = generate_orthogonal_array_L8()[:, :n_control]
            else:
                raise ValueError("Currently support up to 7 control factors")
        else:
            # 2^k factorial for control factors
            from pyDOE3 import fullfact
            control_array = fullfact([2] * n_control)
            control_array = 2 * (control_array / (2-1)) - 1  # Convert to -1, +1

        # Generate noise factor design (outer array)
        if request.noise_design_type == "full_factorial":
            from pyDOE3 import fullfact
            noise_array = fullfact([2] * n_noise)
            noise_array = 2 * (noise_array / (2-1)) - 1  # Convert to -1, +1
        else:
            # Orthogonal array for noise
            if n_noise <= 3:
                noise_array = generate_orthogonal_array_L4()[:, :n_noise]
            elif n_noise <= 7:
                noise_array = generate_orthogonal_array_L8()[:, :n_noise]
            else:
                raise ValueError("Currently support up to 7 noise factors")

        # Create component names
        if request.control_factor_names and len(request.control_factor_names) == n_control:
            control_names = request.control_factor_names
        else:
            control_names = [f"C{i+1}" for i in range(n_control)]

        if request.noise_factor_names and len(request.noise_factor_names) == n_noise:
            noise_names = request.noise_factor_names
        else:
            noise_names = [f"N{i+1}" for i in range(n_noise)]

        # Build combined design (outer product of inner and outer arrays)
        combined_design = []
        run_number = 1

        for i, control_run in enumerate(control_array):
            for j, noise_run in enumerate(noise_array):
                run_data = {
                    "run": run_number,
                    "control_run": i + 1,
                    "noise_run": j + 1
                }

                # Add control factors
                for k, name in enumerate(control_names):
                    run_data[name] = round(float(control_run[k]), 2)

                # Add noise factors
                for k, name in enumerate(noise_names):
                    run_data[name] = round(float(noise_run[k]), 2)

                combined_design.append(run_data)
                run_number += 1

        return {
            "design_type": "Robust Parameter Design (Inner-Outer Array)",
            "n_control_factors": n_control,
            "n_noise_factors": n_noise,
            "control_array_size": len(control_array),
            "noise_array_size": len(noise_array),
            "total_runs": len(combined_design),
            "control_factors": control_names,
            "noise_factors": noise_names,
            "design_matrix": combined_design,
            "properties": {
                "control_design": request.control_design_type,
                "noise_design": request.noise_design_type,
                "description": f"Crossed design with {len(control_array)} control runs × {len(noise_array)} noise runs"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/robust-design/analyze")
async def analyze_robust_design(request: RobustAnalysisRequest):
    """
    Analyze robust parameter design using Signal-to-Noise ratios
    """
    try:
        df = pd.DataFrame(request.data)
        control_factors = request.control_factors
        noise_factors = request.noise_factors
        response = request.response
        quality_char = request.quality_characteristic

        # Group by control factor settings
        control_groups = df.groupby(control_factors)[response]

        # Calculate mean and variance for each control factor combination
        means = control_groups.mean()
        variances = control_groups.var()
        std_devs = control_groups.std()

        # Calculate Signal-to-Noise (SN) ratio based on quality characteristic
        sn_ratios = []

        for control_setting, group in df.groupby(control_factors):
            responses = group[response].values

            if quality_char == "smaller-is-better":
                # SN = -10 * log10(mean(y^2))
                sn = -10 * np.log10(np.mean(responses ** 2))
            elif quality_char == "larger-is-better":
                # SN = -10 * log10(mean(1/y^2))
                sn = -10 * np.log10(np.mean(1 / (responses ** 2 + 1e-10)))
            elif quality_char == "nominal-is-best":
                # SN = 10 * log10(mean^2 / variance)
                mean_val = np.mean(responses)
                var_val = np.var(responses)
                sn = 10 * np.log10((mean_val ** 2) / (var_val + 1e-10))
            else:
                raise ValueError(f"Unknown quality characteristic: {quality_char}")

            sn_ratios.append({
                "control_setting": control_setting if isinstance(control_setting, dict) else
                                 dict(zip(control_factors, control_setting if hasattr(control_setting, '__iter__') else [control_setting])),
                "mean_response": round(float(np.mean(responses)), 4),
                "std_dev": round(float(np.std(responses)), 4),
                "sn_ratio": round(float(sn), 4)
            })

        # Sort by SN ratio (descending - higher is better)
        sn_ratios.sort(key=lambda x: x['sn_ratio'], reverse=True)

        # Main effects for SN ratios
        main_effects = {}
        for factor in control_factors:
            # Group by individual factor level
            factor_groups = {}
            for entry in sn_ratios:
                level = entry['control_setting'][factor]
                if level not in factor_groups:
                    factor_groups[level] = []
                factor_groups[level].append(entry['sn_ratio'])

            # Calculate mean SN for each level
            level_means = {level: round(float(np.mean(sns)), 4)
                          for level, sns in factor_groups.items()}

            main_effects[factor] = {
                "level_means": level_means,
                "effect_size": round(float(max(level_means.values()) - min(level_means.values())), 4)
            }

        # Optimal settings (highest SN ratio)
        optimal = sn_ratios[0]

        return {
            "quality_characteristic": quality_char,
            "n_control_combinations": len(sn_ratios),
            "sn_ratios": sn_ratios,
            "main_effects": main_effects,
            "optimal_settings": {
                "control_factors": optimal['control_setting'],
                "predicted_mean": optimal['mean_response'],
                "predicted_std_dev": optimal['std_dev'],
                "sn_ratio": optimal['sn_ratio']
            },
            "interpretation": f"Optimal control factor settings maximize SN ratio ({optimal['sn_ratio']} dB), indicating robust performance across noise conditions."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# EXPORT TO INDUSTRY STANDARDS (Phase 1 - RSM Improvements)
# ============================================================================

class ExportRequest(BaseModel):
    format: str = Field(..., description="Export format: 'jmp', 'r', 'python', 'pdf'")
    model_data: Dict[str, Any] = Field(..., description="Complete model results")
    factors: List[str] = Field(..., description="Factor names")
    response: str = Field(..., description="Response variable name")
    data: List[Dict[str, float]] = Field(..., description="Experimental data")


@router.post("/export")
async def export_model(request: ExportRequest):
    """
    Export RSM model to industry standard formats:
    - JMP: JSL script for model reconstruction
    - R: R script with all analysis code
    - Python: Python script using statsmodels
    - PDF: Professional report (not yet implemented)
    """
    try:
        if request.format == 'jmp':
            # Generate JMP JSL script
            script = generate_jmp_script(request)
            return {
                "format": "jmp",
                "filename": f"rsm_analysis_{request.response}.jsl",
                "content": script,
                "mime_type": "text/plain"
            }

        elif request.format == 'r':
            # Generate R script
            script = generate_r_script(request)
            return {
                "format": "r",
                "filename": f"rsm_analysis_{request.response}.R",
                "content": script,
                "mime_type": "text/plain"
            }

        elif request.format == 'python':
            # Generate Python script
            script = generate_python_script(request)
            return {
                "format": "python",
                "filename": f"rsm_analysis_{request.response}.py",
                "content": script,
                "mime_type": "text/plain"
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def generate_jmp_script(request: ExportRequest) -> str:
    """Generate JMP JSL script for RSM analysis"""
    coeffs = request.model_data.get('coefficients', {})

    # Build data table creation
    data_lines = []
    for row in request.data:
        values = [str(row.get(f, 0)) for f in request.factors]
        values.append(str(row.get(request.response, 0)))
        data_lines.append(f"    {{{', '.join(values)}}}")

    # Build model formula
    terms = []
    for term, coef_data in coeffs.items():
        if term != 'Intercept':
            if '**2' in term:
                # Quadratic term
                factor = term.replace('I(', '').replace('**2)', '')
                terms.append(f"{factor}*{factor}")
            elif ':' in term:
                # Interaction
                terms.append('*'.join(term.split(':')))
            else:
                terms.append(term)

    formula = ' + '.join(terms) if terms else request.factors[0]

    script = f"""// JMP JSL Script for RSM Analysis
// Generated by MasterStat
// Response: {request.response}

// Create Data Table
dt = New Table( "RSM Data",
    Add Rows( {len(request.data)} ),
    New Column( "{request.factors[0]}", Numeric, "Continuous", Formula( None ) ),
"""

    for factor in request.factors[1:]:
        script += f'    New Column( "{factor}", Numeric, "Continuous", Formula( None ) ),\n'

    script += f'    New Column( "{request.response}", Numeric, "Continuous", Formula( None ) )\n);\n\n'
    script += f"// Fill Data\n"
    script += f"dt << BeginDataUpdate();\n"

    for i, row in enumerate(request.data, 1):
        for j, factor in enumerate(request.factors, 1):
            script += f'dt:Column({j})[{i}] = {row.get(factor, 0)};\n'
        script += f'dt:Column({len(request.factors) + 1})[{i}] = {row.get(request.response, 0)};\n'

    script += f"dt << EndDataUpdate();\n\n"

    # Add fit model platform
    script += f"""// Fit Response Surface Model
obj = dt << Fit Model(
    Y( :{request.response} ),
    Effects( """

    for factor in request.factors:
        script += f"{factor}, "

    script += f"""),
    Personality( "Response Surface" ),
    Emphasis( "Effect Screening" ),
    Run(
        Profiler( 1 ),
        :{request.response} << {{"Summary of Fit", "Analysis of Variance", "Parameter Estimates",
            "Effect Tests", "Prediction Profiler"}}
    )
);

// Save Prediction Formula
obj << Make Response Surface;
"""

    return script


def generate_r_script(request: ExportRequest) -> str:
    """Generate R script for RSM analysis"""
    coeffs = request.model_data.get('coefficients', {})

    # Build data frame creation
    data_lines = []
    for factor in request.factors:
        values = [str(row.get(factor, 0)) for row in request.data]
        data_lines.append(f"  {factor} = c({', '.join(values)})")

    response_values = [str(row.get(request.response, 0)) for row in request.data]
    data_lines.append(f"  {request.response} = c({', '.join(response_values)})")

    # Build model formula
    linear_terms = ' + '.join(request.factors)
    quad_terms = ' + '.join([f"I({f}^2)" for f in request.factors])

    int_terms = []
    for i in range(len(request.factors)):
        for j in range(i+1, len(request.factors)):
            int_terms.append(f"{request.factors[i]}:{request.factors[j]}")

    interaction_terms = ' + '.join(int_terms) if int_terms else ""

    script = f"""# R Script for RSM Analysis
# Generated by MasterStat
# Response: {request.response}

# Load required packages
library(rsm)

# Create data frame
data <- data.frame(
{',\\n'.join(data_lines)}
)

# Convert to coded data (if needed)
# Assuming data is already in coded units (-1, 0, +1)

# Fit Response Surface Model
model <- lm({request.response} ~ {linear_terms} + {quad_terms}"""

    if interaction_terms:
        script += f" + {interaction_terms}"

    script += f""", data = data)

# Model summary
summary(model)

# ANOVA table
anova(model)

# Canonical analysis
library(rsm)
canonical_data <- as.coded.data(data, formulas = list("""

    for i, factor in enumerate(request.factors):
        script += f'\n    {factor} ~ ({factor})'
        if i < len(request.factors) - 1:
            script += ','

    script += f"""
))

rsm_model <- rsm({request.response} ~ SO({', '.join(request.factors)}), data = canonical_data)
summary(rsm_model)

# Contour plots (for 2 factors)
if (length(c({', '.join([f"'{f}'" for f in request.factors])})) == 2) {{
    contour(rsm_model, ~ {request.factors[0]} + {request.factors[1]})
}}

# Prediction profiler
library(effects)
plot(allEffects(model))

# Model diagnostics
par(mfrow = c(2, 2))
plot(model)

# R-squared and adjusted R-squared
cat("\\nR-squared:", summary(model)$r.squared, "\\n")
cat("Adjusted R-squared:", summary(model)$adj.r.squared, "\\n")
"""

    return script


def generate_python_script(request: ExportRequest) -> str:
    """Generate Python script for RSM analysis"""

    # Build data lists
    data_dict = {factor: [] for factor in request.factors}
    data_dict[request.response] = []

    for row in request.data:
        for factor in request.factors:
            data_dict[factor].append(row.get(factor, 0))
        data_dict[request.response].append(row.get(request.response, 0))

    # Build model formula
    linear_terms = ' + '.join(request.factors)
    quad_terms = ' + '.join([f"I({f}**2)" for f in request.factors])

    int_terms = []
    for i in range(len(request.factors)):
        for j in range(i+1, len(request.factors)):
            int_terms.append(f"{request.factors[i]}:{request.factors[j]}")

    interaction_terms = ' + '.join(int_terms) if int_terms else ""

    script = f"""# Python Script for RSM Analysis
# Generated by MasterStat
# Response: {request.response}

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# Create DataFrame
data = pd.DataFrame({{
"""

    for factor in request.factors:
        script += f"    '{factor}': {data_dict[factor]},\n"

    script += f"    '{request.response}': {data_dict[request.response]}\n"
    script += f"}})\n\n"

    # Build formula
    script += f"""# Fit Response Surface Model
formula = '{request.response} ~ {linear_terms} + {quad_terms}"""

    if interaction_terms:
        script += f" + {interaction_terms}"

    script += f"""'

model = ols(formula, data=data).fit()

# Model summary
print(model.summary())

# ANOVA table
from statsmodels.stats.anova import anova_lm
anova_table = anova_lm(model, typ=2)
print("\\nANOVA Table:")
print(anova_table)

# Model diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
axes[0, 0].scatter(model.fittedvalues, model.resid)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Normal Q-Q
from scipy import stats
stats.probplot(model.resid, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot')

# Scale-Location
axes[1, 0].scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)))
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('Sqrt(|Residuals|)')
axes[1, 0].set_title('Scale-Location')

# Residuals vs Leverage
from statsmodels.stats.outliers_influence import OLSInfluence
influence = OLSInfluence(model)
leverage = influence.hat_matrix_diag
axes[1, 1].scatter(leverage, model.resid)
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residuals vs Leverage')

plt.tight_layout()
plt.show()

# Print model metrics
print(f"\\nR-squared: {{model.rsquared:.4f}}")
print(f"Adjusted R-squared: {{model.rsquared_adj:.4f}}")
print(f"RMSE: {{np.sqrt(model.mse_resid):.4f}}")

# Prediction at optimal point (example)
# Modify factor values as needed
optimal_point = {{{', '.join([f"'{f}': 0.0" for f in request.factors])}}}
prediction = model.predict(pd.DataFrame([optimal_point]))
print(f"\\nPrediction at center point: {{prediction[0]:.4f}}")
"""

    return script


# ============================================================================
# ADVANCED MODEL DIAGNOSTICS (Phase 1 - RSM Improvements)
# ============================================================================

class ModelDiagnosticsRequest(BaseModel):
    data: List[Dict[str, float]] = Field(..., description="Experimental data used in model fitting")
    factors: List[str] = Field(..., description="Factor variable names")
    response: str = Field(..., description="Response variable name")
    coefficients: Dict[str, Any] = Field(..., description="Model coefficients from fitted model")


@router.post("/advanced-diagnostics")
async def advanced_model_diagnostics(request: ModelDiagnosticsRequest):
    """
    Compute advanced model diagnostics for RSM models:
    - Leverage (Hat values): Identifies influential points based on factor space position
    - Cook's Distance: Measures overall influence of each observation
    - DFFITS: Measures change in prediction when observation is removed
    - VIF (Variance Inflation Factor): Detects multicollinearity among factors
    - PRESS: Prediction error sum of squares (leave-one-out CV)

    These diagnostics help identify:
    1. Outliers and influential observations
    2. Model reliability and prediction accuracy
    3. Multicollinearity issues
    """
    try:
        df = pd.DataFrame(request.data)
        n = len(df)

        # Build second-order model formula (same as fit-model)
        linear_terms = " + ".join(request.factors)
        quadratic_terms = " + ".join([f"I({f}**2)" for f in request.factors])

        interaction_terms = []
        for i in range(len(request.factors)):
            for j in range(i+1, len(request.factors)):
                interaction_terms.append(f"{request.factors[i]}:{request.factors[j]}")

        if interaction_terms:
            formula = f"{request.response} ~ {linear_terms} + {quadratic_terms} + {' + '.join(interaction_terms)}"
        else:
            formula = f"{request.response} ~ {linear_terms} + {quadratic_terms}"

        # Fit model to get influence measures
        model = ols(formula, data=df).fit()

        # Get influence measures from statsmodels
        influence = model.get_influence()

        # 1. LEVERAGE (Hat values)
        # High leverage points (h_ii > 2p/n or 3p/n are concerning)
        hat_values = influence.hat_matrix_diag
        p = len(model.params)  # number of parameters
        leverage_threshold = 2 * p / n
        high_leverage_threshold = 3 * p / n

        leverage_data = []
        for i, h in enumerate(hat_values):
            status = "normal"
            if h > high_leverage_threshold:
                status = "high"
            elif h > leverage_threshold:
                status = "moderate"

            leverage_data.append({
                "observation": i + 1,
                "leverage": round(float(h), 6),
                "status": status
            })

        # 2. COOK'S DISTANCE
        # Values > 1 are concerning, > 4/n warrant investigation
        cooks_d = influence.cooks_distance[0]
        cooks_threshold = 4 / n

        cooks_data = []
        for i, d in enumerate(cooks_d):
            status = "normal"
            if d > 1:
                status = "highly_influential"
            elif d > cooks_threshold:
                status = "influential"

            cooks_data.append({
                "observation": i + 1,
                "cooks_distance": round(float(d), 6),
                "status": status
            })

        # 3. DFFITS
        # Standardized difference in fitted values
        # Threshold: 2*sqrt(p/n) for large samples, 3*sqrt(p/n) for small samples
        dffits = influence.dffits[0]
        dffits_threshold = 2 * np.sqrt(p / n)

        dffits_data = []
        for i, dffit in enumerate(dffits):
            status = "normal"
            if abs(dffit) > dffits_threshold:
                status = "influential"

            dffits_data.append({
                "observation": i + 1,
                "dffits": round(float(dffit), 6),
                "status": status
            })

        # 4. VIF (Variance Inflation Factor)
        # VIF > 10 indicates serious multicollinearity
        # VIF > 5 indicates moderate multicollinearity
        # Create design matrix for VIF calculation
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # Build design matrix (X) without intercept for VIF
        X_design = df[request.factors].copy()

        # Add quadratic terms
        for factor in request.factors:
            X_design[f'{factor}_sq'] = X_design[factor] ** 2

        # Add interaction terms
        for i in range(len(request.factors)):
            for j in range(i+1, len(request.factors)):
                f1, f2 = request.factors[i], request.factors[j]
                X_design[f'{f1}_{f2}'] = X_design[f1] * X_design[f2]

        # Calculate VIF for each term
        vif_data = []
        X_matrix = X_design.values

        for i, col_name in enumerate(X_design.columns):
            try:
                vif = variance_inflation_factor(X_matrix, i)

                status = "excellent"
                if vif > 10:
                    status = "severe_multicollinearity"
                elif vif > 5:
                    status = "moderate_multicollinearity"
                elif vif > 2.5:
                    status = "low_multicollinearity"

                vif_data.append({
                    "term": col_name,
                    "vif": round(float(vif), 4) if not np.isinf(vif) else "Inf",
                    "status": status
                })
            except:
                vif_data.append({
                    "term": col_name,
                    "vif": "N/A",
                    "status": "calculation_failed"
                })

        # 5. PRESS Statistic (Prediction Error Sum of Squares)
        # Leave-one-out cross-validation prediction error
        # PRESS = sum of squared prediction errors
        # R²_prediction = 1 - PRESS/SST
        residuals = model.resid
        press_residuals = residuals / (1 - hat_values)
        press = float(np.sum(press_residuals ** 2))

        # Calculate R² prediction
        sst = float(np.sum((df[request.response] - df[request.response].mean()) ** 2))
        r2_prediction = 1 - (press / sst)

        # Summary statistics
        n_high_leverage = sum(1 for item in leverage_data if item['status'] in ['high', 'moderate'])
        n_influential_cooks = sum(1 for item in cooks_data if item['status'] != 'normal')
        n_influential_dffits = sum(1 for item in dffits_data if item['status'] != 'normal')
        n_multicollinearity = sum(1 for item in vif_data if 'multicollinearity' in item['status'])

        # Generate recommendations
        recommendations = []

        if n_high_leverage > 0:
            recommendations.append(f"Found {n_high_leverage} high-leverage observations. Review these points for data quality.")

        if n_influential_cooks > 0:
            recommendations.append(f"Found {n_influential_cooks} influential observations (Cook's D). Consider refitting model without these points.")

        if n_influential_dffits > 0:
            recommendations.append(f"Found {n_influential_dffits} observations with high DFFITS. These significantly affect predictions.")

        if n_multicollinearity > 0:
            recommendations.append(f"Detected multicollinearity in {n_multicollinearity} terms. Consider model reduction or ridge regression.")

        if r2_prediction < 0.5:
            recommendations.append(f"Low R²_prediction ({r2_prediction:.3f}). Model may not predict well on new data.")
        elif r2_prediction > 0.9:
            recommendations.append(f"Excellent R²_prediction ({r2_prediction:.3f}). Model predicts well on new data.")

        if not recommendations:
            recommendations.append("All diagnostic checks passed. Model appears reliable.")

        return {
            "diagnostics": {
                "leverage": leverage_data,
                "cooks_distance": cooks_data,
                "dffits": dffits_data,
                "vif": vif_data
            },
            "press": {
                "value": round(press, 4),
                "r2_prediction": round(r2_prediction, 4),
                "interpretation": "R² prediction indicates how well the model predicts new observations"
            },
            "summary": {
                "n_observations": n,
                "n_parameters": p,
                "leverage_threshold": round(leverage_threshold, 4),
                "high_leverage_threshold": round(high_leverage_threshold, 4),
                "cooks_threshold": round(cooks_threshold, 4),
                "dffits_threshold": round(dffits_threshold, 4),
                "n_high_leverage": n_high_leverage,
                "n_influential_cooks": n_influential_cooks,
                "n_influential_dffits": n_influential_dffits,
                "n_multicollinearity_issues": n_multicollinearity
            },
            "recommendations": recommendations,
            "interpretation": (
                "Model diagnostics assess observation influence and multicollinearity. "
                "High leverage points are extreme in the factor space. "
                "Influential points (Cook's D, DFFITS) significantly affect model fit. "
                "High VIF indicates correlated predictors. "
                "PRESS evaluates predictive performance via cross-validation."
            )
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
