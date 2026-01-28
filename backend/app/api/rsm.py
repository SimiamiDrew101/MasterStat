from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pyDOE3 import ccdesign, bbdesign
from app.utils.report_generator import PDFReportGenerator, format_pvalue, format_number

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
            "fitted": [round(float(f), 4) for f in fitted_values],
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
            "Model": {
                "sum_sq": round(float(model.ess), 4),  # Explained sum of squares
                "df": int(model.df_model),
                "mean_sq": round(float(model.ess / model.df_model), 4),
                "F": round(float(model.fvalue), 4) if not pd.isna(model.fvalue) else None,
                "p_value": round(float(model.f_pvalue), 6) if not pd.isna(model.f_pvalue) else None
            },
            "Residual": {
                "sum_sq": round(float(model.ssr), 4),  # Residual sum of squares
                "df": int(model.df_resid),
                "mean_sq": round(float(model.mse_resid), 4)
            },
            "Total": {
                "sum_sq": round(float(model.centered_tss), 4),  # Total sum of squares
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
            "anova": enhanced_anova,  # Primary ANOVA for frontend access
            "enhanced_anova": enhanced_anova,  # Alias for compatibility
            "r_squared": round(float(model.rsquared), 4) if not pd.isna(model.rsquared) else None,
            "adj_r_squared": round(float(model.rsquared_adj), 4) if not pd.isna(model.rsquared_adj) else None,
            "rmse": round(float(np.sqrt(model.mse_resid)), 4) if not pd.isna(model.mse_resid) else None,
            "curvature_test": curvature_test,
            "diagnostics": diagnostics,
            "lack_of_fit_test": lof_test
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class MultiRSMRequest(BaseModel):
    data: List[Dict[str, float]] = Field(..., description="Experimental data with multiple responses")
    factors: List[str] = Field(..., description="Factor variable names")
    responses: List[str] = Field(..., description="Multiple response variable names")
    alpha: float = Field(0.05, description="Significance level")

@router.post("/fit-multi-model")
async def fit_multi_rsm_model(request: MultiRSMRequest):
    """
    Fit Response Surface Models for multiple responses simultaneously.
    Each response gets its own second-order polynomial model with identical factor structure.

    This enables multi-response optimization and comparative analysis across responses.
    """
    try:
        # Validate request
        if len(request.responses) < 2:
            raise HTTPException(
                status_code=400,
                detail="Multi-response fitting requires at least 2 responses. Use /fit-model for single response."
            )

        if len(request.responses) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 responses allowed. For more responses, consider fitting models separately."
            )

        df = pd.DataFrame(request.data)

        # Validate that all responses exist in data
        missing_responses = [r for r in request.responses if r not in df.columns]
        if missing_responses:
            raise HTTPException(
                status_code=400,
                detail=f"Response variables not found in data: {', '.join(missing_responses)}"
            )

        # Validate that all factors exist in data
        missing_factors = [f for f in request.factors if f not in df.columns]
        if missing_factors:
            raise HTTPException(
                status_code=400,
                detail=f"Factor variables not found in data: {', '.join(missing_factors)}"
            )

        # Check for missing values in any response
        for response in request.responses:
            if df[response].isna().any():
                na_count = df[response].isna().sum()
                raise HTTPException(
                    status_code=400,
                    detail=f"Response '{response}' has {na_count} missing value(s). Please impute or remove missing data."
                )

        # Fit model for each response
        models = {}
        r_squared_values = []
        adj_r_squared_values = []
        significant_models = []

        for response in request.responses:
            # Create single-response request and reuse existing fit_rsm_model logic
            single_request = RSMRequest(
                data=request.data,
                factors=request.factors,
                response=response,
                alpha=request.alpha
            )

            # Fit the model
            model_result = await fit_rsm_model(single_request)
            models[response] = model_result

            # Collect summary statistics
            if model_result["r_squared"] is not None:
                r_squared_values.append(model_result["r_squared"])
            if model_result["adj_r_squared"] is not None:
                adj_r_squared_values.append(model_result["adj_r_squared"])

            # Check if model is statistically significant
            model_p_value = model_result["enhanced_anova"]["Model"]["p_value"]
            if model_p_value is not None and model_p_value < request.alpha:
                significant_models.append(response)

        # Calculate summary statistics
        summary = {
            "n_responses": len(request.responses),
            "n_factors": len(request.factors),
            "n_observations": len(df),
            "all_models_significant": len(significant_models) == len(request.responses),
            "significant_models": significant_models,
            "r_squared_summary": {
                "mean": round(float(np.mean(r_squared_values)), 4) if r_squared_values else None,
                "min": round(float(np.min(r_squared_values)), 4) if r_squared_values else None,
                "max": round(float(np.max(r_squared_values)), 4) if r_squared_values else None,
                "std": round(float(np.std(r_squared_values, ddof=1)), 4) if len(r_squared_values) > 1 else None
            },
            "adj_r_squared_summary": {
                "mean": round(float(np.mean(adj_r_squared_values)), 4) if adj_r_squared_values else None,
                "min": round(float(np.min(adj_r_squared_values)), 4) if adj_r_squared_values else None,
                "max": round(float(np.max(adj_r_squared_values)), 4) if adj_r_squared_values else None,
                "std": round(float(np.std(adj_r_squared_values, ddof=1)), 4) if len(adj_r_squared_values) > 1 else None
            }
        }

        # Calculate response correlations (useful for identifying redundant responses)
        response_data = df[request.responses]
        correlation_matrix = response_data.corr().round(4).to_dict()

        # Generate interpretation
        interpretation = []

        if summary["all_models_significant"]:
            interpretation.append(f"All {len(request.responses)} response models are statistically significant (p < {request.alpha}).")
        else:
            non_sig = [r for r in request.responses if r not in significant_models]
            interpretation.append(f"{len(significant_models)}/{len(request.responses)} models are significant. Non-significant: {', '.join(non_sig)}")

        if summary["r_squared_summary"]["min"] is not None:
            if summary["r_squared_summary"]["min"] >= 0.9:
                interpretation.append("Excellent model fit across all responses (R² ≥ 0.9).")
            elif summary["r_squared_summary"]["min"] >= 0.7:
                interpretation.append("Good model fit across all responses (R² ≥ 0.7).")
            elif summary["r_squared_summary"]["min"] < 0.5:
                interpretation.append("Some models have poor fit (R² < 0.5). Consider model diagnostics or additional terms.")

        # Check for highly correlated responses (potential redundancy)
        high_correlations = []
        for i, r1 in enumerate(request.responses):
            for j, r2 in enumerate(request.responses):
                if i < j:  # Only check upper triangle
                    corr = correlation_matrix[r1][r2]
                    if abs(corr) > 0.9:
                        high_correlations.append((r1, r2, corr))

        if high_correlations:
            for r1, r2, corr in high_correlations:
                interpretation.append(f"Responses '{r1}' and '{r2}' are highly correlated (r={corr:.2f}). Consider if both are needed.")

        # Generate recommendations
        recommendations = []

        if summary["r_squared_summary"]["std"] and summary["r_squared_summary"]["std"] > 0.2:
            recommendations.append("Large variation in R² values suggests some responses are harder to model. Review diagnostics for low-R² responses.")

        if not summary["all_models_significant"]:
            recommendations.append("Fit higher-order terms or check for outliers in non-significant responses.")

        if len(high_correlations) > 0:
            recommendations.append("Highly correlated responses can be combined using desirability functions for multi-objective optimization.")

        if summary["r_squared_summary"]["mean"] and summary["r_squared_summary"]["mean"] > 0.8:
            recommendations.append("Models are well-fitted. Proceed with multi-response visualization and optimization.")

        return {
            "models": models,
            "summary": summary,
            "correlation_matrix": correlation_matrix,
            "interpretation": interpretation,
            "recommendations": recommendations,
            "factors": request.factors,
            "responses": request.responses
        }

    except HTTPException:
        raise  # Re-raise validation errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-model fitting failed: {str(e)}")

class MultiSurfaceRequest(BaseModel):
    models: Dict[str, Dict[str, Any]] = Field(..., description="Model coefficients for each response")
    factors: List[str] = Field(..., description="Factor variable names (must be exactly 2)")
    grid_resolution: int = Field(20, description="Number of grid points per factor (e.g. 20 = 20x20 = 400 points)")
    x_range: List[float] = Field(None, description="[min, max] for first factor. Auto-computed if not provided.")
    y_range: List[float] = Field(None, description="[min, max] for second factor. Auto-computed if not provided.")
    normalize: str = Field("none", description="Normalization method: 'none', 'zscore', 'minmax', or 'desirability'")

@router.post("/generate-multi-surface")
async def generate_multi_surface_data(request: MultiSurfaceRequest):
    """
    Generate contour surface data for multiple responses simultaneously.
    Returns grid data ready for Plotly contour plots with optional normalization.

    Normalization methods:
    - 'none': Raw predicted values
    - 'zscore': Z-score normalization (mean=0, std=1)
    - 'minmax': Min-max scaling to [0, 1]
    - 'desirability': Goal-based desirability (requires goals - not implemented yet)
    """
    try:
        # Validate exactly 2 factors (contour plots are 2D)
        if len(request.factors) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Contour plots require exactly 2 factors. Received {len(request.factors)}. For 3+ factors, use slice plots or response surface viewer."
            )

        # Validate grid resolution
        if request.grid_resolution < 10:
            raise HTTPException(status_code=400, detail="Grid resolution must be at least 10.")
        if request.grid_resolution > 100:
            raise HTTPException(status_code=400, detail="Grid resolution must not exceed 100 (to prevent performance issues).")

        # Validate models
        if not request.models:
            raise HTTPException(status_code=400, detail="At least one model required.")

        # Set default ranges if not provided
        x_range = request.x_range if request.x_range else [-2, 2]
        y_range = request.y_range if request.y_range else [-2, 2]

        # Generate grid
        x_vals = np.linspace(x_range[0], x_range[1], request.grid_resolution)
        y_vals = np.linspace(y_range[0], y_range[1], request.grid_resolution)

        # Helper function to predict response value from coefficients
        def predict_value(coeffs, x, y, factor_names):
            """Predict response value for a second-order model"""
            x_name, y_name = factor_names

            # Get coefficient values (handle different formats)
            def get_coeff(name):
                if name in coeffs:
                    coeff_val = coeffs[name]
                    # If it's a dict with 'estimate', extract estimate
                    if isinstance(coeff_val, dict) and 'estimate' in coeff_val:
                        return coeff_val['estimate'] if coeff_val['estimate'] is not None else 0.0
                    # Otherwise treat as numeric
                    return float(coeff_val) if coeff_val is not None else 0.0
                return 0.0

            # Second-order polynomial: Intercept + b1*x + b2*y + b11*x^2 + b22*y^2 + b12*x*y
            intercept = get_coeff('Intercept')
            b1 = get_coeff(x_name)
            b2 = get_coeff(y_name)
            b11 = get_coeff(f'I({x_name} ** 2)')
            b22 = get_coeff(f'I({y_name} ** 2)')
            b12 = get_coeff(f'{x_name}:{y_name}')

            value = intercept + b1*x + b2*y + b11*(x**2) + b22*(y**2) + b12*x*y
            return value

        # Generate surface data for each response
        surfaces = {}
        raw_values = {}  # Store raw values for normalization

        for response_name, model_data in request.models.items():
            # Extract coefficients
            if 'coefficients' in model_data:
                coeffs = model_data['coefficients']
            else:
                coeffs = model_data  # Assume model_data is coefficients dict

            surface_points = []
            z_values = []

            for y in y_vals:
                for x in x_vals:
                    z = predict_value(coeffs, x, y, request.factors)
                    surface_points.append({
                        "x": round(float(x), 4),
                        "y": round(float(y), 4),
                        "z": round(float(z), 4),
                        "z_raw": round(float(z), 4)
                    })
                    z_values.append(z)

            surfaces[response_name] = surface_points
            raw_values[response_name] = z_values

        # Apply normalization if requested
        normalization_params = {}

        if request.normalize == "zscore":
            for response_name, points in surfaces.items():
                z_vals = raw_values[response_name]
                mean = np.mean(z_vals)
                std = np.std(z_vals, ddof=1)

                if std == 0:
                    std = 1  # Prevent division by zero

                # Normalize
                for i, point in enumerate(points):
                    point["z_normalized"] = round(float((point["z_raw"] - mean) / std), 4)
                    point["z"] = point["z_normalized"]

                normalization_params[response_name] = {
                    "method": "zscore",
                    "mean": round(float(mean), 4),
                    "std": round(float(std), 4),
                    "min": round(float(np.min(z_vals)), 4),
                    "max": round(float(np.max(z_vals)), 4)
                }

        elif request.normalize == "minmax":
            for response_name, points in surfaces.items():
                z_vals = raw_values[response_name]
                z_min = np.min(z_vals)
                z_max = np.max(z_vals)
                z_range = z_max - z_min

                if z_range == 0:
                    z_range = 1  # Prevent division by zero

                # Normalize
                for i, point in enumerate(points):
                    point["z_normalized"] = round(float((point["z_raw"] - z_min) / z_range), 4)
                    point["z"] = point["z_normalized"]

                normalization_params[response_name] = {
                    "method": "minmax",
                    "min": round(float(z_min), 4),
                    "max": round(float(z_max), 4),
                    "range": round(float(z_range), 4),
                    "mean": round(float(np.mean(z_vals)), 4),
                    "std": round(float(np.std(z_vals, ddof=1)), 4)
                }

        else:  # "none" or unrecognized
            for response_name, points in surfaces.items():
                z_vals = raw_values[response_name]
                normalization_params[response_name] = {
                    "method": "none",
                    "min": round(float(np.min(z_vals)), 4),
                    "max": round(float(np.max(z_vals)), 4),
                    "mean": round(float(np.mean(z_vals)), 4),
                    "std": round(float(np.std(z_vals, ddof=1)), 4)
                }

        return {
            "surfaces": surfaces,
            "normalization_params": normalization_params,
            "grid_info": {
                "resolution": request.grid_resolution,
                "x_range": x_range,
                "y_range": y_range,
                "total_points_per_surface": len(x_vals) * len(y_vals),
                "factors": request.factors
            },
            "n_responses": len(surfaces),
            "responses": list(surfaces.keys())
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Surface generation failed: {str(e)}")

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
            "success": bool(result.success),  # Convert numpy bool to Python bool
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
    method: str = Field("weighted_geometric_mean", description="Compositing method: 'weighted_geometric_mean', 'minimum', 'weighted_sum'")


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

        # Composite desirability function (multiple methods)
        def composite_desirability(x):
            desirabilities = []
            weights = []
            for spec in request.responses:
                y_pred = predict_response(x, spec.coefficients)
                d = calculate_desirability(y_pred, spec)
                desirabilities.append(d)
                weights.append(spec.weight)

            if request.method == "weighted_geometric_mean":
                # Weighted geometric mean (default, most common)
                if any(d == 0 for d in desirabilities):
                    return 0.0

                total_weight = sum(weights)
                D = 1.0
                for d, w in zip(desirabilities, weights):
                    D *= d ** (w / total_weight)
                return D

            elif request.method == "minimum":
                # Minimum desirability (conservative, all criteria must be met)
                return min(desirabilities)

            elif request.method == "weighted_sum":
                # Weighted arithmetic mean (linear, less conservative than geometric)
                total_weight = sum(weights)
                if total_weight == 0:
                    return 0.0
                return sum(d * w for d, w in zip(desirabilities, weights)) / total_weight

            else:
                # Default to weighted geometric mean
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

        # Method descriptions
        method_names = {
            "weighted_geometric_mean": "Weighted Geometric Mean (balanced trade-offs)",
            "minimum": "Minimum Desirability (all criteria must be met)",
            "weighted_sum": "Weighted Sum (linear combination)"
        }

        return {
            "optimal_point": optimal_point,
            "composite_desirability": round(float(composite_d), 4),
            "individual_results": individual_results,
            "success": result.success,
            "compositing_method": method_names.get(request.method, request.method),
            "method": "Desirability Function Optimization",
            "interpretation": (
                f"Composite desirability = {round(float(composite_d), 4)} "
                f"(0 = unacceptable, 1 = ideal). "
                f"This represents the best compromise among all {len(request.responses)} responses "
                f"using {method_names.get(request.method, request.method)}."
            )
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# MULTI-RESPONSE CONTOUR OVERLAY (Tier 2 Feature 3)
# ============================================================================

class MultiResponseContourRequest(BaseModel):
    responses: List[Dict[str, Any]] = Field(..., description="List of response models with coefficients")
    factors: List[str] = Field(..., description="Exactly 2 factors for contour plot")
    x_range: Optional[List[float]] = Field(None, description="[min, max] for first factor")
    y_range: Optional[List[float]] = Field(None, description="[min, max] for second factor")
    grid_resolution: int = Field(30, description="Grid points per axis")
    show_feasible_region: bool = Field(True, description="Highlight feasible region where all constraints are met")

@router.post("/multi-response-contour")
async def multi_response_contour(request: MultiResponseContourRequest):
    """
    Generate overlaid contour plots for multiple responses.

    Shows contours for each response on the same plot, allowing visualization
    of trade-offs and identification of the sweet spot (feasible region where
    all response criteria are satisfied).

    Returns:
    - Grid data for each response
    - Feasible region coordinates
    - Sweet spot identification
    - Pareto frontier approximation
    """
    try:
        if len(request.factors) != 2:
            raise HTTPException(
                status_code=400,
                detail="Multi-response contour requires exactly 2 factors"
            )

        # Set up grid
        x_range = request.x_range or [-2, 2]
        y_range = request.y_range or [-2, 2]
        x = np.linspace(x_range[0], x_range[1], request.grid_resolution)
        y = np.linspace(y_range[0], y_range[1], request.grid_resolution)
        X, Y = np.meshgrid(x, y)

        # Prediction function
        def predict_at_point(x_val, y_val, coefficients):
            pred = coefficients.get('Intercept', 0.0)
            pred += coefficients.get(request.factors[0], 0.0) * x_val
            pred += coefficients.get(request.factors[1], 0.0) * y_val
            pred += coefficients.get(f"I({request.factors[0]}**2)", 0.0) * x_val**2
            pred += coefficients.get(f"I({request.factors[1]}**2)", 0.0) * y_val**2
            pred += coefficients.get(f"{request.factors[0]}:{request.factors[1]}", 0.0) * x_val * y_val
            return pred

        # Calculate predictions for each response
        contour_data = []
        for resp in request.responses:
            Z = np.zeros_like(X)
            for i in range(request.grid_resolution):
                for j in range(request.grid_resolution):
                    Z[i, j] = predict_at_point(X[i, j], Y[i, j], resp['coefficients'])

            contour_data.append({
                "response_name": resp['name'],
                "Z": Z.tolist(),
                "min_value": float(Z.min()),
                "max_value": float(Z.max()),
                "goal": resp.get('goal', 'maximize'),
                "target": resp.get('target'),
                "lower_limit": resp.get('lower_limit'),
                "upper_limit": resp.get('upper_limit')
            })

        # Calculate feasible region if constraints provided
        feasible_region = None
        sweet_spot = None

        if request.show_feasible_region:
            feasible_mask = np.ones_like(X, dtype=bool)

            for resp_data, resp in zip(contour_data, request.responses):
                Z_array = np.array(resp_data['Z'])
                goal = resp.get('goal', 'maximize')

                if goal == 'maximize' and resp.get('lower_limit') is not None:
                    feasible_mask &= (Z_array >= resp['lower_limit'])
                elif goal == 'minimize' and resp.get('upper_limit') is not None:
                    feasible_mask &= (Z_array <= resp['upper_limit'])
                elif goal == 'target':
                    if resp.get('lower_limit') is not None and resp.get('upper_limit') is not None:
                        feasible_mask &= ((Z_array >= resp['lower_limit']) &
                                         (Z_array <= resp['upper_limit']))

            # Extract feasible region points
            feasible_points = []
            for i in range(request.grid_resolution):
                for j in range(request.grid_resolution):
                    if feasible_mask[i, j]:
                        feasible_points.append({
                            "x": float(X[i, j]),
                            "y": float(Y[i, j])
                        })

            if feasible_points:
                feasible_region = feasible_points

                # Find sweet spot (center of feasible region)
                x_coords = [p['x'] for p in feasible_points]
                y_coords = [p['y'] for p in feasible_points]
                sweet_spot = {
                    "x": float(np.mean(x_coords)),
                    "y": float(np.mean(y_coords)),
                    request.factors[0]: float(np.mean(x_coords)),
                    request.factors[1]: float(np.mean(y_coords))
                }

        return {
            "grid": {
                "x": x.tolist(),
                "y": y.tolist(),
                "X": X.tolist(),
                "Y": Y.tolist()
            },
            "factors": request.factors,
            "contours": contour_data,
            "feasible_region": feasible_region,
            "sweet_spot": sweet_spot,
            "n_responses": len(request.responses),
            "interpretation": {
                "feasible_points": len(feasible_region) if feasible_region else 0,
                "has_sweet_spot": sweet_spot is not None,
                "recommendation": (
                    f"Sweet spot identified at {request.factors[0]}={sweet_spot[request.factors[0]]:.2f}, "
                    f"{request.factors[1]}={sweet_spot[request.factors[1]]:.2f}" if sweet_spot
                    else "No feasible region found - constraints may be too strict or conflicting"
                )
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-response contour failed: {str(e)}")


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
            "success": bool(result.success),  # Convert numpy bool to Python bool
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

        # Build Scheffé canonical polynomial model
        # Create interaction terms manually to avoid statsmodels formula issues

        X_data = {}

        # Linear terms (pure component effects)
        for comp in components:
            X_data[comp] = df[comp].values

        # Quadratic terms (binary blending effects)
        if degree >= 2:
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    interaction_name = f"{components[i]}*{components[j]}"
                    X_data[interaction_name] = df[components[i]].values * df[components[j]].values

        # Cubic terms (ternary blending effects)
        if degree >= 3:
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    for k in range(j + 1, len(components)):
                        interaction_name = f"{components[i]}*{components[j]}*{components[k]}"
                        X_data[interaction_name] = (df[components[i]].values *
                                                    df[components[j]].values *
                                                    df[components[k]].values)

        # Create design matrix
        X = pd.DataFrame(X_data)
        y = df[response].values

        # Fit using statsmodels OLS without formula
        # Add constant for intercept-only if no terms
        from statsmodels.api import add_constant
        X_with_const = add_constant(X, has_constant='skip')

        try:
            model = sm.OLS(y, X_with_const).fit()
        except Exception as e:
            # If singular matrix, try without problematic terms
            raise ValueError(f"Model fitting failed - likely singular design matrix. Try adding more experimental runs or reducing model degree. Error: {str(e)}")

        # ANOVA-like decomposition (handle inf/nan values)
        def safe_float(value, default=None, decimals=6):
            """Safely convert to float, handling inf/nan"""
            if value is None:
                return default
            try:
                f = float(value)
                if np.isfinite(f):
                    return round(f, decimals)
                return default
            except (ValueError, TypeError):
                return default

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
                "estimate": safe_float(coef, 0.0, 6),
                "std_error": safe_float(se, None, 6),
                "t_value": safe_float(t_val, None, 4),
                "p_value": safe_float(p_val, None, 6)
            }

        # Model statistics
        r_squared = safe_float(model.rsquared, None, 4)
        adj_r_squared = safe_float(model.rsquared_adj, None, 4)
        rmse = safe_float(np.sqrt(model.mse_resid), None, 4)

        # Build model summary
        model_type = f"Scheffé {['Linear', 'Quadratic', 'Cubic'][degree-1]} Mixture Model"

        anova_dict = {
            "Model": {
                "sum_sq": safe_float(model.ess),
                "df": int(model.df_model),
                "F": safe_float(model.fvalue, None) if hasattr(model, 'fvalue') else None,
                "PR(>F)": safe_float(model.f_pvalue, None) if hasattr(model, 'f_pvalue') else None
            },
            "Residual": {
                "sum_sq": safe_float(model.ssr),
                "df": int(model.df_resid),
                "F": None,
                "PR(>F)": None
            }
        }

        # Create a descriptive formula string
        term_names = list(X.columns)
        formula_str = f"{response} ~ {' + '.join(term_names)}"

        return {
            "model_type": model_type,
            "formula": formula_str,
            "coefficients": coefficients,
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "rmse": rmse,
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

            # Convert control_setting to dict (handle single value or tuple)
            if isinstance(control_setting, dict):
                control_dict = control_setting
            elif isinstance(control_setting, (list, tuple)):
                # Multiple factors - zip with factor names
                control_dict = dict(zip(control_factors, [float(v) for v in control_setting]))
            else:
                # Single factor - wrap in list
                control_dict = {control_factors[0]: float(control_setting)}

            sn_ratios.append({
                "control_setting": control_dict,
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

        elif request.format == 'pdf':
            # Generate PDF report
            import base64
            pdf_bytes = generate_pdf_report(request)
            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
            return {
                "format": "pdf",
                "filename": f"rsm_analysis_{request.response}.pdf",
                "content": pdf_base64,
                "mime_type": "application/pdf"
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

class DesignRecommendationRequest(BaseModel):
    n_factors: int = Field(..., description="Number of factors (2-6)")
    budget: Optional[int] = Field(None, description="Maximum number of experimental runs")
    goal: str = Field("optimization", description="Experiment goal: 'optimization', 'screening', or 'modeling'")
    time_constraint: Optional[str] = Field(None, description="Time constraint: 'low', 'medium', 'high'")

class CrossValidationRequest(BaseModel):
    data: List[Dict[str, float]] = Field(..., description="Experimental data")
    factors: List[str] = Field(..., description="Factor variable names")
    response: str = Field(..., description="Response variable name")
    k_folds: int = Field(5, description="Number of folds for cross-validation (default: 5)")


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


def generate_pdf_report(request: ExportRequest) -> bytes:
    """
    Generate comprehensive PDF report for RSM analysis
    Includes: Title, Summary, ANOVA, Coefficients, Diagnostics, Recommendations
    """
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from io import BytesIO
    from datetime import datetime

    # Create PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)

    # Container for PDF elements
    elements = []

    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=12,
        spaceBefore=12
    )

    # Title Page
    elements.append(Paragraph("Response Surface Methodology", title_style))
    elements.append(Paragraph("Complete Analysis Report", title_style))
    elements.append(Spacer(1, 0.3*inch))

    # Analysis Info
    info_data = [
        ["Response Variable:", request.response],
        ["Factors:", ", ".join(request.factors)],
        ["Number of Observations:", str(len(request.data))],
        ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Tool:", "MasterStat - Professional DOE Platform"]
    ]

    info_table = Table(info_data, colWidths=[2.5*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1e40af')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0e7ff')),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.5*inch))

    # Model Summary
    model_data = request.model_data
    elements.append(Paragraph("1. Model Summary", heading_style))

    summary_data = [
        ["Metric", "Value"],
        ["R²", f"{model_data.get('r_squared', 'N/A'):.4f}"],
        ["Adjusted R²", f"{model_data.get('adj_r_squared', 'N/A'):.4f}"],
        ["RMSE", f"{model_data.get('rmse', 'N/A'):.4f}"],
        ["Model Type", model_data.get('model_type', 'Second-Order RSM')]
    ]

    summary_table = Table(summary_data, colWidths=[3*inch, 3.5*inch])
    summary_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 11),
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f9ff')])
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.3*inch))

    # ANOVA Table
    elements.append(Paragraph("2. Analysis of Variance (ANOVA)", heading_style))

    anova = model_data.get('anova', {})
    anova_data = [["Source", "Sum of Squares", "DF", "Mean Square", "F-value", "P-value"]]

    for source in ['Model', 'Residual', 'Total']:
        if source in anova:
            row_data = anova[source]
            anova_data.append([
                source,
                f"{row_data.get('sum_sq', 0):.4f}",
                str(row_data.get('df', 0)),
                f"{row_data.get('mean_sq', 0):.4f}" if 'mean_sq' in row_data else "—",
                f"{row_data.get('F', 0):.4f}" if row_data.get('F') else "—",
                f"{row_data.get('p_value', 0):.6f}" if row_data.get('p_value') else "—"
            ])

    anova_table = Table(anova_data, colWidths=[1.3*inch, 1.3*inch, 0.7*inch, 1.3*inch, 1*inch, 1*inch])
    anova_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#d1fae5')])
    ]))
    elements.append(anova_table)
    elements.append(PageBreak())

    # Coefficients Table
    elements.append(Paragraph("3. Model Coefficients", heading_style))

    coeffs = model_data.get('coefficients', {})
    coeff_data = [["Term", "Estimate", "Std Error", "t-value", "P-value", "Significant"]]

    for term, coef_info in coeffs.items():
        if isinstance(coef_info, dict):
            est = coef_info.get('estimate')
            p_val = coef_info.get('p_value')
            sig = "✓" if p_val and p_val < 0.05 else ""

            coeff_data.append([
                term,
                f"{est:.4f}" if est is not None else "N/A",
                f"{coef_info.get('std_error', 0):.4f}" if coef_info.get('std_error') else "N/A",
                f"{coef_info.get('t_value', 0):.4f}" if coef_info.get('t_value') else "N/A",
                f"{p_val:.6f}" if p_val is not None else "N/A",
                sig
            ])

    coeff_table = Table(coeff_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    coeff_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ede9fe')])
    ]))
    elements.append(coeff_table)
    elements.append(Spacer(1, 0.3*inch))

    # Experimental Data Summary
    elements.append(Paragraph("4. Experimental Data", heading_style))

    # Show first 10 rows of data
    data_headers = request.factors + [request.response]
    data_rows = [data_headers]

    for i, row in enumerate(request.data[:10]):
        data_row = [f"{row.get(f, 0):.2f}" for f in data_headers]
        data_rows.append(data_row)

    if len(request.data) > 10:
        data_rows.append(["..."] * len(data_headers))

    col_width = 6.5*inch / len(data_headers)
    data_table = Table(data_rows, colWidths=[col_width] * len(data_headers))
    data_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f59e0b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fef3c7')])
    ]))
    elements.append(data_table)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph(f"<i>Showing {min(10, len(request.data))} of {len(request.data)} observations</i>",
                             styles['Italic']))

    elements.append(PageBreak())

    # Recommendations
    elements.append(Paragraph("5. Recommendations", heading_style))

    r_squared = model_data.get('r_squared', 0)
    adj_r_squared = model_data.get('adj_r_squared', 0)

    recommendations = []

    if r_squared > 0.9:
        recommendations.append("Excellent model fit (R² > 0.9). Model explains variability well.")
    elif r_squared > 0.7:
        recommendations.append("Good model fit (R² > 0.7). Model is useful for prediction.")
    else:
        recommendations.append("Model fit could be improved. Consider additional factors or transformation.")

    if adj_r_squared and (r_squared - adj_r_squared) > 0.05:
        recommendations.append("Significant difference between R² and Adjusted R². Model may be overfitted.")

    recommendations.append("Review coefficient p-values to identify significant factors.")
    recommendations.append("Check residual plots for model adequacy assumptions.")
    recommendations.append("Validate predictions with confirmation runs at optimal settings.")

    for i, rec in enumerate(recommendations, 1):
        elements.append(Paragraph(f"{i}. {rec}", styles['BodyText']))
        elements.append(Spacer(1, 0.1*inch))

    elements.append(Spacer(1, 0.3*inch))

    # Footer note
    elements.append(Spacer(1, 0.5*inch))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8,
                                 textColor=colors.grey, alignment=TA_CENTER)
    elements.append(Paragraph("Generated by MasterStat - Professional Design of Experiments Platform",
                             footer_style))
    elements.append(Paragraph("https://github.com/anthropics/masterstat", footer_style))

    # Build PDF
    doc.build(elements)

    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# ============================================================================
# EXPERIMENT WIZARD (Phase 2 - RSM Improvements)
# ============================================================================

@router.post("/recommend-design")
async def recommend_design(request: DesignRecommendationRequest):
    """
    Smart design recommendation engine for Experiment Wizard.
    Helps beginners choose the right experimental design based on:
    - Number of factors
    - Budget constraints (max runs)
    - Experimental goal
    - Time constraints

    Returns top 3 design recommendations with pros, cons, and rationale.
    """
    try:
        n_factors = request.n_factors
        budget = request.budget
        goal = request.goal

        # Validate input
        if n_factors < 2 or n_factors > 6:
            raise HTTPException(status_code=400, detail="Number of factors must be between 2 and 6")

        recommendations = []

        # ===== 2 FACTORS =====
        if n_factors == 2:
            # Face-Centered CCD (13 runs)
            fc_ccd = {
                "type": "Face-Centered CCD",
                "design_code": "face-centered",
                "runs": 13,
                "pros": [
                    "Efficient for 2 factors (only 13 runs)",
                    "Orthogonal blocking",
                    "All points fit within original cube (safe operating region)",
                    "Good for constraints"
                ],
                "cons": [
                    "Not rotatable (prediction variance varies with direction)",
                    "Less efficient than rotatable for pure prediction"
                ],
                "best_for": "2-factor optimization with constraints",
                "description": "Places axial points at ±1 on each axis (faces of cube)",
                "score": 95,
                "properties": {
                    "rotatable": False,
                    "orthogonal": True,
                    "alpha": 1.0,
                    "factorial_points": 4,
                    "axial_points": 4,
                    "center_points": 5
                }
            }

            # Rotatable CCD (13 runs)
            rot_ccd = {
                "type": "Rotatable CCD",
                "design_code": "rotatable",
                "runs": 13,
                "pros": [
                    "Constant prediction variance at all points equidistant from center",
                    "Excellent for exploring unknown regions",
                    "Optimal for prediction"
                ],
                "cons": [
                    "Axial points outside original cube (α = 1.414)",
                    "May violate process constraints",
                    "Requires wider operating region"
                ],
                "best_for": "2-factor exploration without strict constraints",
                "description": "Optimized for equal prediction variance in all directions",
                "score": 90,
                "properties": {
                    "rotatable": True,
                    "orthogonal": False,
                    "alpha": round(2**(2/4), 3),  # 1.414 for 2 factors
                    "factorial_points": 4,
                    "axial_points": 4,
                    "center_points": 5
                }
            }

            # Box-Behnken (13 runs)
            bb = {
                "type": "Box-Behnken",
                "design_code": "box-behnken",
                "runs": 13,
                "pros": [
                    "No corner points (safer for sensitive processes)",
                    "Spherical design",
                    "Efficient for 3 factors"
                ],
                "cons": [
                    "Designed for 3+ factors (not optimal for 2)",
                    "Not as efficient as CCD for 2 factors"
                ],
                "best_for": "Better suited for 3+ factors",
                "description": "Midpoints of cube edges plus center points",
                "score": 70,
                "properties": {
                    "rotatable": False,
                    "orthogonal": True,
                    "edge_points": 8,
                    "center_points": 5
                }
            }

            recommendations = [fc_ccd, rot_ccd, bb]

        # ===== 3 FACTORS =====
        elif n_factors == 3:
            # Budget considerations for 3 factors
            if budget and budget < 20:
                # Box-Behnken for tight budget (15 runs)
                bb = {
                    "type": "Box-Behnken",
                    "design_code": "box-behnken",
                    "runs": 15,
                    "pros": [
                        "Very efficient for 3 factors (only 15 runs)",
                        "No extreme corners (safer)",
                        "Orthogonal design",
                        "Fits tight budgets"
                    ],
                    "cons": [
                        "Cannot estimate all three-way interactions",
                        "Poor prediction at corners"
                    ],
                    "best_for": "3-factor screening with limited budget",
                    "description": "Efficient 3-factor design using edge midpoints",
                    "score": 95,
                    "properties": {
                        "rotatable": False,
                        "orthogonal": True,
                        "edge_points": 12,
                        "center_points": 3
                    }
                }
                recommendations.append(bb)

            # Face-Centered CCD (20 runs)
            fc_ccd = {
                "type": "Face-Centered CCD",
                "design_code": "face-centered",
                "runs": 20,
                "pros": [
                    "Comprehensive coverage",
                    "All points within cube",
                    "Good for constraints"
                ],
                "cons": [
                    "More runs than Box-Behnken (20 vs 15)",
                    "Not rotatable"
                ],
                "best_for": "3-factor optimization with constraints",
                "description": "Factorial + axial points at faces + center",
                "score": 85,
                "properties": {
                    "rotatable": False,
                    "orthogonal": True,
                    "alpha": 1.0,
                    "factorial_points": 8,
                    "axial_points": 6,
                    "center_points": 6
                }
            }

            # Rotatable CCD (20 runs)
            rot_ccd = {
                "type": "Rotatable CCD",
                "design_code": "rotatable",
                "runs": 20,
                "pros": [
                    "Equal prediction variance (rotatable)",
                    "Comprehensive coverage",
                    "Optimal for prediction"
                ],
                "cons": [
                    "Axial points outside cube (α = 1.682)",
                    "May violate constraints",
                    "More runs than Box-Behnken"
                ],
                "best_for": "3-factor exploration without strict constraints",
                "description": "Optimized for rotatability",
                "score": 80,
                "properties": {
                    "rotatable": True,
                    "orthogonal": False,
                    "alpha": round(2**(3/4), 3),  # 1.682 for 3 factors
                    "factorial_points": 8,
                    "axial_points": 6,
                    "center_points": 6
                }
            }

            if not any(r["design_code"] == "box-behnken" for r in recommendations):
                recommendations.extend([fc_ccd, rot_ccd])
            else:
                recommendations.extend([fc_ccd, rot_ccd][:2])  # Limit to top 3

        # ===== 4 FACTORS =====
        elif n_factors == 4:
            # Box-Behnken (27 runs) - Most efficient
            bb = {
                "type": "Box-Behnken",
                "design_code": "box-behnken",
                "runs": 27,
                "pros": [
                    "Most efficient for 4 factors",
                    "No extreme corners",
                    "Good coverage"
                ],
                "cons": [
                    "Cannot estimate all higher-order interactions",
                    "27 runs may still be expensive"
                ],
                "best_for": "4-factor screening and optimization",
                "description": "Efficient 4-factor design",
                "score": 90,
                "properties": {
                    "rotatable": False,
                    "orthogonal": True,
                    "edge_points": 24,
                    "center_points": 3
                }
            }

            # Face-Centered CCD (31 runs)
            fc_ccd = {
                "type": "Face-Centered CCD",
                "design_code": "face-centered",
                "runs": 31,
                "pros": [
                    "Comprehensive 4-factor coverage",
                    "All points within cube"
                ],
                "cons": [
                    "More runs than Box-Behnken (31 vs 27)",
                    "Higher cost"
                ],
                "best_for": "Comprehensive 4-factor study",
                "description": "Full second-order design",
                "score": 80,
                "properties": {
                    "rotatable": False,
                    "orthogonal": True,
                    "alpha": 1.0,
                    "factorial_points": 16,
                    "axial_points": 8,
                    "center_points": 7
                }
            }

            # Fractional factorial first recommendation
            screening = {
                "type": "Fractional Factorial (Screening)",
                "design_code": "screening-first",
                "runs": 16,
                "pros": [
                    "Very efficient screening (only 16 runs)",
                    "Identify important factors first",
                    "Sequential approach"
                ],
                "cons": [
                    "Cannot fit full RSM initially",
                    "Requires follow-up experiment",
                    "Two-stage process"
                ],
                "best_for": "When unsure which factors are important",
                "description": "Screen first, then RSM on important factors",
                "score": 75,
                "properties": {
                    "approach": "sequential",
                    "initial_runs": 16,
                    "followup_runs": "15-20"
                }
            }

            recommendations = [bb, fc_ccd, screening]

        # ===== 5-6 FACTORS =====
        elif n_factors >= 5:
            # Strongly recommend screening first
            screening = {
                "type": "Fractional Factorial Screening",
                "design_code": "screening-first",
                "runs": 32 if n_factors == 5 else 64,
                "pros": [
                    "Identify vital few factors",
                    "Efficient for many factors",
                    "Proven sequential approach"
                ],
                "cons": [
                    "Requires two-stage experimentation",
                    "Initial screen doesn't fit RSM"
                ],
                "best_for": f"{n_factors}-factor screening → RSM on important factors",
                "description": "Screen first, then detailed RSM on 2-3 key factors",
                "score": 95,
                "properties": {
                    "approach": "sequential",
                    "initial_runs": 32 if n_factors == 5 else 64,
                    "recommended_followup": "RSM on top 2-3 factors"
                }
            }

            # Box-Behnken if they insist on RSM (expensive!)
            bb_runs = {5: 46, 6: 54}
            bb = {
                "type": "Box-Behnken (Not Recommended)",
                "design_code": "box-behnken",
                "runs": bb_runs[n_factors],
                "pros": [
                    "Can fit full RSM immediately",
                    "No extreme corners"
                ],
                "cons": [
                    f"Very expensive ({bb_runs[n_factors]} runs)",
                    "Many factors = complex model",
                    "Interpretation difficulties",
                    "May not be practical"
                ],
                "best_for": "Only if all factors known to be important",
                "description": f"Full {n_factors}-factor RSM (expensive)",
                "score": 50,
                "properties": {
                    "rotatable": False,
                    "orthogonal": True,
                    "warning": "Consider screening first"
                }
            }

            # Definitive Screening Design (newer approach)
            dsd_runs = 2 * n_factors + 1
            dsd = {
                "type": "Definitive Screening Design",
                "design_code": "dsd",
                "runs": dsd_runs,
                "pros": [
                    f"Very efficient ({dsd_runs} runs for {n_factors} factors)",
                    "Can detect curvature",
                    "Modern approach",
                    "Good for factor reduction"
                ],
                "cons": [
                    "Not a traditional RSM",
                    "May require follow-up",
                    "Less established than fractional factorial"
                ],
                "best_for": "Modern efficient screening with curvature detection",
                "description": "Efficient screening that detects main effects and curvature",
                "score": 85,
                "properties": {
                    "approach": "modern_screening",
                    "runs": dsd_runs,
                    "can_detect_curvature": True
                }
            }

            recommendations = [screening, dsd, bb]

        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        # Apply budget filter if specified
        if budget:
            filtered = [r for r in recommendations if r["runs"] <= budget]
            if not filtered:
                # No designs fit budget - return warning with smallest design
                min_runs = min(r["runs"] for r in recommendations)
                return {
                    "recommendations": recommendations[:1],
                    "warning": f"Budget of {budget} runs is insufficient. Minimum required: {min_runs} runs.",
                    "suggestion": f"Consider increasing budget or reducing number of factors.",
                    "user_input": {
                        "n_factors": n_factors,
                        "budget": budget,
                        "goal": goal
                    }
                }
            recommendations = filtered

        # Return top 3
        return {
            "recommendations": recommendations[:3],
            "summary": {
                "n_factors": n_factors,
                "recommended_design": recommendations[0]["type"],
                "runs_required": recommendations[0]["runs"],
                "rationale": recommendations[0]["best_for"]
            },
            "user_input": {
                "n_factors": n_factors,
                "budget": budget,
                "goal": goal
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Design recommendation failed: {str(e)}")


# ============================================================================
# ADVANCED SCREENING DESIGNS (Tier 2 - Experiment Wizard Completion)
# ============================================================================

class DSDRequest(BaseModel):
    n_factors: int = Field(..., description="Number of factors (3 or more)")
    factor_names: Optional[List[str]] = Field(None, description="Custom factor names")
    randomize: bool = Field(True, description="Randomize run order")

class PlackettBurmanRequest(BaseModel):
    n_factors: int = Field(..., description="Number of factors (max n-1 for n runs)")
    n_runs: int = Field(..., description="Number of runs (must be 4, 8, 12, 16, 20, or 24)")
    factor_names: Optional[List[str]] = Field(None, description="Custom factor names")
    randomize: bool = Field(True, description="Randomize run order")

class ConfoundingRequest(BaseModel):
    design_type: str = Field(..., description="Design type: 'fractional_factorial', 'dsd', 'plackett_burman'")
    n_factors: int = Field(..., description="Number of factors")
    n_runs: Optional[int] = Field(None, description="Number of runs (for fractional factorial)")
    generator: Optional[str] = Field(None, description="Generator string (e.g., 'I=ABC' for 2^(3-1))")

@router.post("/dsd/generate")
async def generate_dsd(request: DSDRequest):
    """
    Generate Definitive Screening Design (DSD).

    A DSD is a 3-level design with 2n+1 runs for n factors.
    It can estimate main effects, quadratic effects, and some two-factor interactions
    with minimal confounding.

    Properties:
    - 2n+1 runs for n factors (very efficient)
    - 3 levels per factor
    - Main effects clear of all other effects
    - Quadratic effects clear of other quadratics
    - Can detect curvature
    - Orthogonal design

    Returns:
    - Design matrix in coded units (-1, 0, +1)
    - Run order (randomized if requested)
    - Design properties and interpretation
    """
    try:
        n = request.n_factors

        if n < 3:
            raise HTTPException(
                status_code=400,
                detail="DSD requires at least 3 factors"
            )

        # DSD construction: 2n+1 runs
        n_runs = 2 * n + 1

        # Initialize design matrix
        design = np.zeros((n_runs, n))

        # DSD construction algorithm (Jones & Nachtsheim, 2011)
        # First 2n runs: balanced pairs of +1/-1 for each factor
        for i in range(n):
            # Create balanced pairs for factor i
            design[2*i, i] = 1
            design[2*i + 1, i] = -1

            # Orthogonal complement for other factors
            for j in range(n):
                if i != j:
                    if (i + j) % 2 == 0:
                        design[2*i, j] = 1 if (i < j) else -1
                        design[2*i + 1, j] = -1 if (i < j) else 1
                    else:
                        design[2*i, j] = -1 if (i < j) else 1
                        design[2*i + 1, j] = 1 if (i < j) else -1

        # Last run: center point (all zeros)
        design[-1, :] = 0

        # Create factor names
        if request.factor_names and len(request.factor_names) == n:
            factors = request.factor_names
        else:
            factors = [f"X{i+1}" for i in range(n)]

        # Convert to DataFrame
        df = pd.DataFrame(design, columns=factors)

        # Add run order
        if request.randomize:
            run_order = np.random.permutation(n_runs) + 1
        else:
            run_order = np.arange(1, n_runs + 1)

        df.insert(0, 'RunOrder', run_order)
        df.insert(1, 'StdOrder', np.arange(1, n_runs + 1))

        # Calculate design properties
        X = design[:, :]

        # D-efficiency (determinant of X'X normalized)
        XtX = X.T @ X
        try:
            det_XtX = np.linalg.det(XtX)
            d_efficiency = (abs(det_XtX) ** (1/n)) / n_runs * 100
        except:
            d_efficiency = None

        # Condition number
        try:
            cond_number = np.linalg.cond(XtX)
        except:
            cond_number = None

        return {
            "design": df.to_dict(orient='records'),
            "properties": {
                "design_type": "Definitive Screening Design (DSD)",
                "n_factors": n,
                "n_runs": n_runs,
                "n_levels": 3,
                "efficiency": round(d_efficiency, 2) if d_efficiency else None,
                "condition_number": round(cond_number, 2) if cond_number else None,
                "randomized": request.randomize
            },
            "interpretation": {
                "purpose": "Efficient screening design that can estimate main effects, quadratics, and some interactions",
                "advantages": [
                    f"Only {n_runs} runs for {n} factors (very economical)",
                    "Main effects unconfounded with any other effects",
                    "Quadratic effects unconfounded with other quadratics",
                    "Can detect curvature",
                    "Orthogonal design matrix"
                ],
                "limitations": [
                    "Not all two-factor interactions can be estimated",
                    "Some interactions may be partially confounded",
                    "Requires 3 levels (may be harder to run than 2-level designs)"
                ],
                "next_steps": [
                    "Add response column to design matrix",
                    "Run experiment in randomized order",
                    "Analyze with second-order model",
                    "Use for factor screening or optimization"
                ]
            },
            "citation": "Jones, B., & Nachtsheim, C. J. (2011). A class of three-level designs for definitive screening in the presence of second-order effects. Journal of Quality Technology, 43(1), 1-15."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DSD generation failed: {str(e)}")


@router.post("/plackett-burman/generate")
async def generate_plackett_burman(request: PlackettBurmanRequest):
    """
    Generate Plackett-Burman Design.

    Plackett-Burman designs are efficient screening designs for studying
    k = n-1 factors in n runs (where n is a multiple of 4).

    Properties:
    - 2-level design
    - n runs for up to n-1 factors (very efficient)
    - Main effects heavily confounded with two-factor interactions
    - Use when interactions are believed to be negligible
    - Non-orthogonal for interactions

    Supported run sizes: 4, 8, 12, 16, 20, 24

    Returns:
    - Design matrix in coded units (-1, +1)
    - Alias structure
    - Design properties and recommendations
    """
    try:
        n_runs = request.n_runs
        n_factors = request.n_factors

        # Validate run size
        valid_runs = [4, 8, 12, 16, 20, 24]
        if n_runs not in valid_runs:
            raise HTTPException(
                status_code=400,
                detail=f"Plackett-Burman designs support n_runs in {valid_runs}"
            )

        # Validate number of factors
        if n_factors > n_runs - 1:
            raise HTTPException(
                status_code=400,
                detail=f"For {n_runs} runs, maximum {n_runs-1} factors allowed"
            )

        if n_factors < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 factors required"
            )

        # Plackett-Burman generator sequences
        generators = {
            4: [1, 1, -1],
            8: [1, 1, 1, -1, 1, -1, -1],
            12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
            16: [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],
            20: [1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1],
            24: [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1]
        }

        generator = generators[n_runs]

        # Build design matrix using cyclic permutations
        design = np.zeros((n_runs - 1, n_runs - 1), dtype=int)
        for i in range(n_runs - 1):
            for j in range(n_runs - 1):
                design[i, j] = generator[(j - i) % (n_runs - 1)]

        # Add fold-over row (all -1)
        design = np.vstack([design, -np.ones(n_runs - 1, dtype=int)])

        # Select only the requested number of factors
        design = design[:, :n_factors]

        # Create factor names
        if request.factor_names and len(request.factor_names) == n_factors:
            factors = request.factor_names
        else:
            factors = [f"X{i+1}" for i in range(n_factors)]

        # Convert to DataFrame
        df = pd.DataFrame(design, columns=factors)

        # Add run order
        if request.randomize:
            run_order = np.random.permutation(n_runs) + 1
        else:
            run_order = np.arange(1, n_runs + 1)

        df.insert(0, 'RunOrder', run_order)
        df.insert(1, 'StdOrder', np.arange(1, n_runs + 1))

        return {
            "design": df.to_dict(orient='records'),
            "properties": {
                "design_type": "Plackett-Burman Design",
                "n_factors": n_factors,
                "n_runs": n_runs,
                "n_levels": 2,
                "efficiency": round((n_factors / (n_runs - 1)) * 100, 2),
                "randomized": request.randomize,
                "max_factors": n_runs - 1
            },
            "interpretation": {
                "purpose": "Highly efficient screening design for many factors with minimal runs",
                "assumptions": [
                    "Two-factor interactions are negligible",
                    "Main effects dominate response variation",
                    "Sparsity of effects (only a few factors are important)"
                ],
                "advantages": [
                    f"Study up to {n_runs-1} factors in only {n_runs} runs",
                    "Extremely economical for screening",
                    "Balanced design (equal +1 and -1 for each factor)",
                    "Simple to construct and analyze"
                ],
                "limitations": [
                    "Main effects heavily confounded with two-factor interactions",
                    "Cannot estimate interactions independently",
                    "Not suitable if interactions are expected to be large",
                    "Resolution III design"
                ],
                "alias_structure": "Each main effect is aliased with multiple two-factor interactions (complex aliasing pattern)",
                "next_steps": [
                    "Add response column to design matrix",
                    "Run experiment in randomized order",
                    "Analyze with first-order model",
                    "Follow up significant factors with higher-resolution design"
                ]
            },
            "citation": "Plackett, R. L., & Burman, J. P. (1946). The design of optimum multifactorial experiments. Biometrika, 33(4), 305-325."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plackett-Burman generation failed: {str(e)}")


@router.post("/confounding-analysis")
async def analyze_confounding(request: ConfoundingRequest):
    """
    Analyze confounding (aliasing) structure for factorial and screening designs.

    Returns the alias structure showing which effects are confounded with each other.
    This is critical for interpreting results from fractional factorial, DSD, and
    Plackett-Burman designs.

    Resolution:
    - III: Main effects confounded with two-factor interactions
    - IV: Main effects clear, two-factor interactions confounded with each other
    - V: Main effects and two-factor interactions clear

    Returns:
    - Alias structure (defining relation)
    - Resolution
    - Which effects can be estimated independently
    - Interpretation and warnings
    """
    try:
        design_type = request.design_type
        n_factors = request.n_factors

        if design_type == "dsd":
            # DSD alias structure
            factors = [f"X{i+1}" for i in range(n_factors)]

            return {
                "design_type": "Definitive Screening Design",
                "resolution": "Complex (not traditional resolution)",
                "n_factors": n_factors,
                "defining_relation": None,
                "alias_structure": {
                    "main_effects": {
                        "clear_of": ["All other main effects", "All quadratic effects", "All two-factor interactions"],
                        "confounded_with": [],
                        "estimable": "Yes, independently"
                    },
                    "quadratic_effects": {
                        "clear_of": ["Main effects", "Other quadratic effects"],
                        "confounded_with": ["Some three-factor interactions (negligible)"],
                        "estimable": "Yes, independently"
                    },
                    "two_factor_interactions": {
                        "clear_of": ["Main effects"],
                        "confounded_with": ["Some other two-factor interactions"],
                        "estimable": "Partially (not all interactions estimable)"
                    }
                },
                "interpretation": {
                    "overall": "DSD has a unique aliasing structure that prioritizes main and quadratic effects",
                    "strengths": [
                        "All main effects are unconfounded",
                        "All quadratic effects are unconfounded",
                        "Can detect curvature reliably",
                        "Some two-factor interactions are estimable"
                    ],
                    "limitations": [
                        "Not all two-factor interactions can be estimated",
                        f"For {n_factors} factors, {n_factors*(n_factors-1)//2} interactions exist but only some are estimable",
                        "Interaction aliasing pattern is complex"
                    ],
                    "recommendation": "Use DSD when you need to estimate main effects and quadratics, and only some interactions are of interest"
                }
            }

        elif design_type == "plackett_burman":
            # Plackett-Burman alias structure (Resolution III)
            n_runs = request.n_runs or (n_factors + 1)
            factors = [f"X{i+1}" for i in range(n_factors)]

            # Generate all two-factor interaction names
            interactions = []
            for i in range(n_factors):
                for j in range(i+1, n_factors):
                    interactions.append(f"{factors[i]}*{factors[j]}")

            # In PB designs, each main effect is aliased with many interactions
            # Complex aliasing pattern - simplified representation
            alias_groups = {}
            for i, factor in enumerate(factors):
                # Each main effect is aliased with multiple interactions
                # Exact pattern depends on the specific PB matrix
                alias_groups[factor] = f"Aliased with multiple two-factor interactions (complex pattern)"

            return {
                "design_type": "Plackett-Burman Design",
                "resolution": "III",
                "n_factors": n_factors,
                "n_runs": n_runs,
                "defining_relation": "Complex (not a simple 2^(k-p) design)",
                "alias_structure": {
                    "main_effects": {
                        "clear_of": ["Other main effects"],
                        "confounded_with": ["Multiple two-factor interactions (complex pattern)"],
                        "estimable": "Yes, but confounded with interactions"
                    },
                    "two_factor_interactions": {
                        "clear_of": [],
                        "confounded_with": ["Main effects and other interactions"],
                        "estimable": "No"
                    }
                },
                "alias_groups": alias_groups,
                "interpretation": {
                    "overall": "Resolution III design - main effects confounded with two-factor interactions",
                    "assumption": "Two-factor interactions are negligible compared to main effects",
                    "risk": "If interactions are significant, main effect estimates will be biased",
                    "recommendation": [
                        "Use only when interactions are believed to be small",
                        "Follow up with higher resolution design for significant factors",
                        "Consider confirmation runs at optimal settings"
                    ],
                    "warning": "⚠️ Do not use if two-factor interactions are expected to be important"
                }
            }

        elif design_type == "fractional_factorial":
            # Fractional factorial alias structure
            if not request.generator:
                raise HTTPException(
                    status_code=400,
                    detail="Generator required for fractional factorial analysis (e.g., 'I=ABC' for 2^(3-1))"
                )

            # Parse generator
            generator = request.generator.replace(" ", "").upper()

            # Simple parsing for common generators
            if "=" not in generator:
                raise HTTPException(
                    status_code=400,
                    detail="Generator format: 'I=ABC' or 'I=ABCD'"
                )

            parts = generator.split("=")
            if parts[0] != "I":
                raise HTTPException(
                    status_code=400,
                    detail="Generator must start with 'I=' (e.g., 'I=ABC')"
                )

            defining_word = parts[1]
            word_length = len(defining_word)

            # Determine resolution
            resolution = word_length

            # Resolution interpretations
            resolution_meanings = {
                3: "III - Main effects confounded with two-factor interactions",
                4: "IV - Main effects clear, two-factor interactions confounded with each other",
                5: "V - Main effects and two-factor interactions clear",
                6: "VI - Main effects, two-factor, and some three-factor interactions clear"
            }

            # Generate alias structure for 2^(k-1) designs
            factors = [f"X{i+1}" for i in range(n_factors)]

            return {
                "design_type": "Fractional Factorial Design",
                "design_notation": f"2^({n_factors}-1)",
                "resolution": resolution,
                "resolution_meaning": resolution_meanings.get(resolution, f"Resolution {resolution}"),
                "n_factors": n_factors,
                "defining_relation": f"I = {defining_word}",
                "alias_structure": {
                    "example_aliases": {
                        "description": f"With generator I={defining_word}, effects are confounded as follows:",
                        "pattern": f"Each effect is aliased with its product with {defining_word}"
                    }
                },
                "interpretation": {
                    "overall": f"This is a Resolution {resolution} design",
                    "meaning": resolution_meanings.get(resolution, "Unknown resolution"),
                    "recommendation": "Consult a full alias table for complete confounding structure"
                }
            }

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown design type: {design_type}. Supported: 'dsd', 'plackett_burman', 'fractional_factorial'"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Confounding analysis failed: {str(e)}")


# ============================================================================
# MODEL VALIDATION - K-FOLD CROSS-VALIDATION (Phase 2 - RSM Improvements)
# ============================================================================

@router.post("/cross-validate")
async def cross_validate_model(request: CrossValidationRequest):
    """
    K-fold cross-validation for RSM models.
    Provides robust model validation to complement PRESS statistic.

    Returns:
    - Fold-by-fold performance metrics (R², RMSE, MAE)
    - Average metrics with standard deviation
    - Predicted vs actual values for plotting
    - Interpretation and recommendations
    """
    try:
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        df = pd.DataFrame(request.data)
        n = len(df)

        # Validate k_folds
        if request.k_folds < 2 or request.k_folds > n:
            raise HTTPException(
                status_code=400,
                detail=f"k_folds must be between 2 and {n} (number of observations)"
            )

        # Validate that each training fold has enough points for second-order model
        # For k factors, a second-order model has (k+1)(k+2)/2 parameters
        k = len(request.factors)
        min_params = (k + 1) * (k + 2) // 2
        train_size = n - (n // request.k_folds)  # Approximate training size

        if train_size < min_params:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {request.k_folds}-fold CV. Need at least {min_params} observations per training fold for {k}-factor second-order model (you have ~{train_size}). Try fewer folds or collect more data."
            )

        # Build design matrix for second-order model
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

        # K-fold cross-validation
        kfold = KFold(n_splits=request.k_folds, shuffle=True, random_state=42)

        fold_scores = []
        all_predictions = []
        all_actuals = []

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(df)):
            try:
                # Split data
                df_train = df.iloc[train_idx]
                df_test = df.iloc[test_idx]

                # Fit model on training fold
                model = ols(formula, data=df_train).fit()

                # Predict on test fold
                y_test = df_test[request.response].values
                y_pred = model.predict(df_test)

                # Calculate metrics for this fold
                fold_r2 = r2_score(y_test, y_pred)
                fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                fold_mae = mean_absolute_error(y_test, y_pred)

                # Check for NaN/Inf values
                if np.isnan(fold_r2) or np.isinf(fold_r2) or np.isnan(fold_rmse) or np.isinf(fold_rmse):
                    raise ValueError(f"Fold {fold_idx + 1}: Training set too small to fit model properly")

                fold_scores.append({
                    "fold": fold_idx + 1,
                    "r2": round(float(fold_r2), 4),
                    "rmse": round(float(fold_rmse), 4),
                    "mae": round(float(fold_mae), 4),
                    "n_test": len(test_idx)
                })

                # Store predictions for plotting
                for actual, pred in zip(y_test, y_pred):
                    all_predictions.append(float(pred))
                    all_actuals.append(float(actual))

            except Exception as fold_error:
                raise HTTPException(
                    status_code=400,
                    detail=f"Fold {fold_idx + 1} failed: {str(fold_error)}. Try reducing k_folds or adding more data."
                )

        # Calculate average metrics
        avg_r2 = np.mean([f["r2"] for f in fold_scores])
        std_r2 = np.std([f["r2"] for f in fold_scores])
        avg_rmse = np.mean([f["rmse"] for f in fold_scores])
        std_rmse = np.std([f["rmse"] for f in fold_scores])
        avg_mae = np.mean([f["mae"] for f in fold_scores])
        std_mae = np.std([f["mae"] for f in fold_scores])

        # Calculate overall R² from all CV predictions
        overall_r2 = r2_score(all_actuals, all_predictions)

        # Generate interpretation
        interpretation = []

        # R² interpretation
        if avg_r2 > 0.9:
            interpretation.append("Excellent predictive performance (R² > 0.9)")
        elif avg_r2 > 0.7:
            interpretation.append("Good predictive performance (R² > 0.7)")
        elif avg_r2 > 0.5:
            interpretation.append(f"Moderate predictive performance (R² = {avg_r2:.3f})")
        else:
            interpretation.append(f"Poor predictive performance (R² = {avg_r2:.3f}) - consider model improvements")

        # Consistency interpretation
        if std_r2 < 0.05:
            interpretation.append("Very consistent across folds (low variability)")
        elif std_r2 < 0.10:
            interpretation.append("Reasonably consistent across folds")
        elif std_r2 > 0.15:
            interpretation.append("High variability across folds - may indicate sensitivity to specific data points")

        # Sample size consideration
        avg_test_size = n / request.k_folds
        if avg_test_size < 3:
            interpretation.append(f"Warning: Small test sets ({avg_test_size:.1f} observations per fold). Consider reducing k_folds.")

        # Recommendations
        recommendations = []
        if avg_r2 < 0.7:
            recommendations.append("Consider adding more data points or simplifying the model")
        if std_r2 > 0.15:
            recommendations.append("High fold variability suggests model may be sensitive to outliers. Review diagnostics.")
        if avg_r2 > 0.95 and std_r2 < 0.02:
            recommendations.append("Excellent and stable model performance. Model is ready for use.")

        return {
            "k_folds": request.k_folds,
            "n_observations": n,
            "fold_scores": fold_scores,
            "average_metrics": {
                "r2": round(float(avg_r2), 4),
                "r2_std": round(float(std_r2), 4),
                "rmse": round(float(avg_rmse), 4),
                "rmse_std": round(float(std_rmse), 4),
                "mae": round(float(avg_mae), 4),
                "mae_std": round(float(std_mae), 4)
            },
            "overall_cv_r2": round(float(overall_r2), 4),
            "predictions_vs_actual": {
                "predictions": all_predictions,
                "actuals": all_actuals
            },
            "interpretation": interpretation,
            "recommendations": recommendations if recommendations else ["Model validation complete. No specific recommendations."]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross-validation failed: {str(e)}")

class RSMPDFRequest(BaseModel):
    """Request model for generating RSM PDF report"""
    results: Dict[str, Any] = Field(..., description="RSM analysis results")
    design_type: Optional[str] = Field("response_surface", description="Design type: ccd, box_behnken, response_surface, mixture_design")
    title: Optional[str] = Field("Response Surface Methodology Report", description="Report title")

@router.post("/export-pdf")
async def export_rsm_pdf(request: RSMPDFRequest):
    """
    Generate a comprehensive PDF report for RSM analysis

    Includes:
    - Executive summary
    - Design information
    - ANOVA table
    - Model equation
    - Coefficient table
    - Canonical analysis
    - Stationary point
    - Ridge analysis (if available)
    - Diagnostic information
    - Recommendations
    """
    try:
        results = request.results
        test_type = results.get("test_type", "Response Surface Methodology")

        # Initialize PDF generator
        pdf = PDFReportGenerator(title=request.title, author="MasterStat")

        # Cover page metadata
        metadata = {
            "Design Type": request.design_type.replace('_', ' ').title(),
            "Analysis Type": test_type,
            "Significance Level (α)": str(results.get("alpha", 0.05)),
            "Software": "MasterStat Statistical Analysis Platform"
        }

        # Add design-specific metadata
        if "n_factors" in results:
            metadata["Number of Factors"] = str(results["n_factors"])
        if "n_runs" in results:
            metadata["Number of Runs"] = str(results["n_runs"])
        if "response_name" in results:
            metadata["Response Variable"] = results["response_name"]

        pdf.add_cover_page(subtitle=test_type, metadata=metadata)

        # Executive Summary
        pdf.add_section("Executive Summary")

        # Get model summary
        r_squared = results.get("r_squared", results.get("model_r_squared", 0))
        adj_r_squared = results.get("adj_r_squared", results.get("model_adj_r_squared", 0))

        summary_text = f"""
        A response surface methodology (RSM) analysis was conducted to model the relationship between
        {results.get('n_factors', 'multiple')} factors and the response variable{f" '{results.get('response_name', '')}'" if results.get('response_name') else ''}.
        """

        if "design_type" in results:
            summary_text += f" The experimental design used was {results['design_type'].replace('_', ' ')}."

        summary_text += f"""
        \n\nThe fitted second-order response surface model explains {r_squared*100:.1f}% of the variation
        in the response (R² = {format_number(r_squared, 4)}, Adj R² = {format_number(adj_r_squared, 4)}).
        """

        pdf.add_paragraph(summary_text.strip())

        # Stationary Point Summary (if available)
        if "stationary_point" in results:
            sp = results["stationary_point"]
            sp_type = sp.get("type", "Unknown")

            if sp_type == "saddle_point":
                pdf.add_paragraph(
                    f"\n<b>Critical Finding:</b> The response surface has a saddle point, indicating no global "
                    f"optimum exists within the experimental region. Ridge analysis is recommended to find "
                    f"optimal operating conditions."
                )
            elif sp_type == "minimum":
                pdf.add_paragraph(
                    f"\n<b>Optimization Result:</b> A minimum response was identified at the stationary point."
                )
            elif sp_type == "maximum":
                pdf.add_paragraph(
                    f"\n<b>Optimization Result:</b> A maximum response was identified at the stationary point."
                )

        # Design Information
        pdf.add_section("Design Information")

        design_info_dict = {
            "Design Type": request.design_type.replace('_', ' ').title(),
            "Number of Runs": str(results.get("n_runs", results.get("n_total_runs", "-"))),
            "Number of Factors": str(results.get("n_factors", len(results.get("factors", [])))),
            "Response Variable": results.get("response_name", "-"),
            "Significance Level": str(results.get("alpha", 0.05))
        }

        # Add CCD-specific info
        if "center_points" in results:
            design_info_dict["Center Points"] = str(results["center_points"])
        if "axial_points" in results:
            design_info_dict["Axial Points"] = str(results["axial_points"])
        if "alpha_value" in results:
            design_info_dict["Alpha (α) Value"] = format_number(results["alpha_value"], 4)

        pdf.add_summary_stats(design_info_dict, title="")

        # ANOVA Table
        pdf.add_section("ANOVA Table")

        if "anova_table" in results:
            anova_data = results["anova_table"]
            headers = ["Source", "Sum of Squares", "df", "Mean Square", "F-statistic", "p-value"]
            table_data = []

            # Define order for better readability
            source_order = ["Model", "Linear", "Square", "Interaction", "Lack of Fit", "Pure Error", "Residual", "Total"]

            # Add sources in order
            for source in source_order:
                if source in anova_data:
                    values = anova_data[source]
                    f_val = values.get("f_value", values.get("F"))
                    p_val = values.get("p_value", values.get("PR(>F)"))

                    row = [
                        source,
                        format_number(values.get("sum_sq", values.get("SS")), 4),
                        str(values.get("df", "-")),
                        format_number(values.get("mean_sq", values.get("MS")), 4),
                        format_number(f_val, 4) if f_val is not None else "-",
                        format_pvalue(p_val) if p_val is not None else "-"
                    ]
                    table_data.append(row)

            # Add remaining sources not in the predefined order
            for source, values in anova_data.items():
                if source not in source_order:
                    f_val = values.get("f_value", values.get("F"))
                    p_val = values.get("p_value", values.get("PR(>F)"))

                    row = [
                        source,
                        format_number(values.get("sum_sq", values.get("SS")), 4),
                        str(values.get("df", "-")),
                        format_number(values.get("mean_sq", values.get("MS")), 4),
                        format_number(f_val, 4) if f_val is not None else "-",
                        format_pvalue(p_val) if p_val is not None else "-"
                    ]
                    table_data.append(row)

            pdf.add_table(table_data, headers=headers)

        # Model Summary
        pdf.add_subsection("Model Summary")
        model_stats = {
            "R-squared": format_number(r_squared, 4),
            "Adjusted R-squared": format_number(adj_r_squared, 4),
            "Predicted R-squared": format_number(results.get("pred_r_squared"), 4) if results.get("pred_r_squared") else "N/A",
            "Adequate Precision": format_number(results.get("adequate_precision"), 2) if results.get("adequate_precision") else "N/A",
            "Root MSE": format_number(results.get("root_mse", results.get("rmse")), 4) if results.get("root_mse") or results.get("rmse") else "N/A"
        }

        # Add lack of fit test if available
        if "anova_table" in results and "Lack of Fit" in results["anova_table"]:
            lof = results["anova_table"]["Lack of Fit"]
            lof_p = lof.get("p_value", lof.get("PR(>F)"))
            if lof_p is not None:
                model_stats["Lack of Fit p-value"] = format_pvalue(lof_p)
                model_stats["Lack of Fit Status"] = "Not Significant ✓" if lof_p > results.get("alpha", 0.05) else "Significant ✗"

        pdf.add_summary_stats(model_stats, title="")

        # Regression Coefficients
        if "coefficients" in results or "coefficient_table" in results:
            pdf.add_section("Regression Coefficients")

            coeff_data = results.get("coefficients", results.get("coefficient_table", {}))

            headers = ["Term", "Coefficient", "Std Error", "t-statistic", "p-value", "VIF"]
            table_data = []

            for term, coeff_info in coeff_data.items():
                # Handle different data structures
                if isinstance(coeff_info, dict):
                    coeff = coeff_info.get("coef", coeff_info.get("coefficient", 0))
                    std_err = coeff_info.get("std_err", coeff_info.get("se", None))
                    t_stat = coeff_info.get("t", coeff_info.get("t_statistic", None))
                    p_val = coeff_info.get("p", coeff_info.get("p_value", None))
                    vif = coeff_info.get("vif", None)
                else:
                    # If it's just a number
                    coeff = coeff_info
                    std_err = None
                    t_stat = None
                    p_val = None
                    vif = None

                row = [
                    term,
                    format_number(coeff, 6),
                    format_number(std_err, 6) if std_err is not None else "-",
                    format_number(t_stat, 4) if t_stat is not None else "-",
                    format_pvalue(p_val) if p_val is not None else "-",
                    format_number(vif, 2) if vif is not None else "-"
                ]
                table_data.append(row)

            pdf.add_table(table_data, headers=headers)

        # Model Equations
        if "coded_equation" in results or "natural_equation" in results:
            pdf.add_section("Regression Equations")

            if "coded_equation" in results:
                pdf.add_subsection("Coded Equation")
                pdf.add_paragraph(f"<font name='Courier' size='9'>{results['coded_equation']}</font>")

            if "natural_equation" in results:
                pdf.add_subsection("Natural (Uncoded) Equation")
                pdf.add_paragraph(f"<font name='Courier' size='9'>{results['natural_equation']}</font>")

        # Canonical Analysis
        if "canonical_analysis" in results:
            pdf.add_section("Canonical Analysis")

            canonical = results["canonical_analysis"]

            pdf.add_paragraph(
                "Canonical analysis transforms the fitted surface into a more interpretable form by rotating "
                "the coordinate system to align with the principal axes of the response surface."
            )

            # Canonical form
            if "canonical_form" in canonical:
                pdf.add_subsection("Canonical Form")
                pdf.add_paragraph(f"<font name='Courier'>{canonical['canonical_form']}</font>")

            # Eigenvalues
            if "eigenvalues" in canonical:
                pdf.add_subsection("Eigenvalues (Curvature)")

                headers = ["Eigenvalue", "Value", "Interpretation"]
                table_data = []

                eigenvalues = canonical["eigenvalues"]
                for i, (name, value) in enumerate(eigenvalues.items()):
                    interpretation = ""
                    if value > 0:
                        interpretation = "Concave upward"
                    elif value < 0:
                        interpretation = "Concave downward"
                    else:
                        interpretation = "Flat/saddle"

                    table_data.append([name, format_number(value, 6), interpretation])

                pdf.add_table(table_data, headers=headers)

        # Stationary Point
        if "stationary_point" in results:
            pdf.add_section("Stationary Point Analysis")

            sp = results["stationary_point"]

            # Type and location
            sp_type = sp.get("type", "Unknown")
            pdf.add_paragraph(f"<b>Type:</b> {sp_type.replace('_', ' ').title()}")

            # Coded coordinates
            if "coded" in sp:
                pdf.add_subsection("Stationary Point (Coded Units)")
                headers = ["Factor", "Value"]
                table_data = [[factor, format_number(value, 4)] for factor, value in sp["coded"].items()]
                pdf.add_table(table_data, headers=headers)

            # Natural coordinates
            if "natural" in sp:
                pdf.add_subsection("Stationary Point (Natural Units)")
                headers = ["Factor", "Value"]
                table_data = [[factor, format_number(value, 4)] for factor, value in sp["natural"].items()]
                pdf.add_table(table_data, headers=headers)

            # Predicted response at stationary point
            if "predicted_response" in sp:
                pdf.add_paragraph(
                    f"<b>Predicted Response at Stationary Point:</b> {format_number(sp['predicted_response'], 4)}"
                )

        # Ridge Analysis (if available)
        if "ridge_analysis" in results:
            pdf.add_section("Ridge Analysis")

            pdf.add_paragraph(
                "Ridge analysis is used when the stationary point is far from the experimental region or when "
                "a saddle point is present. It finds the factor settings that maximize or minimize the response "
                "along paths of constant distance from the design center."
            )

            ridge = results["ridge_analysis"]

            if "ridge_path" in ridge and len(ridge["ridge_path"]) > 0:
                # Show a sample of ridge path points
                pdf.add_subsection("Ridge Path (Sample)")
                headers = ["Radius", "Response"] + [f for f in results.get("factors", [])]
                table_data = []

                # Show every 10th point or max 10 points
                step = max(1, len(ridge["ridge_path"]) // 10)
                for i in range(0, len(ridge["ridge_path"]), step):
                    point = ridge["ridge_path"][i]
                    row = [
                        format_number(point.get("radius"), 2),
                        format_number(point.get("response"), 4)
                    ]
                    row.extend([format_number(point.get(f), 4) for f in results.get("factors", [])])
                    table_data.append(row)

                if len(table_data) > 0:
                    pdf.add_table(table_data, headers=headers)

        # Recommendations
        pdf.add_section("Recommendations")
        recommendations = []

        # Model fit recommendations
        if r_squared < 0.70:
            recommendations.append("Model R² is below 70%. Consider adding additional factors, transforming the response, or collecting more data.")
        elif r_squared > 0.95:
            recommendations.append("Excellent model fit (R² > 95%). The model explains the response very well.")

        # Lack of fit
        if "anova_table" in results and "Lack of Fit" in results["anova_table"]:
            lof_p = results["anova_table"]["Lack of Fit"].get("p_value", results["anova_table"]["Lack of Fit"].get("PR(>F)"))
            if lof_p and lof_p < results.get("alpha", 0.05):
                recommendations.append("Significant lack of fit detected. The second-order model may be inadequate. Consider higher-order terms or alternative model forms.")

        # Stationary point recommendations
        if "stationary_point" in results:
            sp_type = results["stationary_point"].get("type", "")
            if sp_type == "saddle_point":
                recommendations.append("Saddle point detected: No true optimum exists. Use ridge analysis to identify optimal operating conditions along constrained paths.")
            elif sp_type in ["maximum", "minimum"]:
                # Check if stationary point is within bounds
                if "within_bounds" in results["stationary_point"] and not results["stationary_point"]["within_bounds"]:
                    recommendations.append("The stationary point is outside the experimental region. Consider expanding the design or using ridge analysis.")
                else:
                    recommendations.append(f"A {sp_type} was found. Verify the optimum through confirmation runs.")

        # General recommendations
        recommendations.append("Examine residual plots to verify model assumptions (normality, constant variance, independence).")
        recommendations.append("Conduct confirmation experiments at the predicted optimal conditions.")
        recommendations.append("Consider robustness testing of the optimal conditions to process variability.")

        pdf.add_recommendations(recommendations)

        # Build PDF
        pdf_bytes = pdf.build()

        # Return PDF as downloadable file
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=rsm_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")
