"""
Generalized Linear Models (GLM) API

Provides endpoints for fitting GLM models with various distributions:
- Poisson (count data)
- Binomial/Logistic (binary outcomes)
- Negative Binomial (overdispersed counts)
- Gamma (positive continuous)
- Gaussian (normal - for comparison)

Uses statsmodels for model fitting and diagnostics.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import (
    Poisson, Binomial, NegativeBinomial, Gamma, Gaussian, Tweedie, InverseGaussian
)
from statsmodels.genmod.families.links import (
    log, logit, identity, inverse_power, probit, cloglog
)
import warnings

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/api/glm", tags=["Generalized Linear Models"])


# ============================================================================
# Pydantic Models
# ============================================================================

class GLMFitRequest(BaseModel):
    """Request model for fitting a GLM."""
    y_data: List[float] = Field(..., description="Response variable values")
    x_data: List[List[float]] = Field(..., description="Predictor variables (each inner list is a row)")
    predictor_names: Optional[List[str]] = Field(None, description="Names for predictor variables")
    response_name: Optional[str] = Field("Y", description="Name for response variable")
    family: str = Field(..., description="Distribution family: poisson, binomial, negativebinomial, gamma, gaussian, tweedie, inverse_gaussian")
    link: Optional[str] = Field(None, description="Link function: log, logit, identity, inverse, probit, cloglog (default depends on family)")
    n_trials: Optional[List[int]] = Field(None, description="Number of trials for binomial (if not provided, assumes binary 0/1)")
    alpha: Optional[float] = Field(0.05, description="Significance level for confidence intervals")
    add_intercept: bool = Field(True, description="Whether to add intercept term")


class GLMPredictRequest(BaseModel):
    """Request model for GLM predictions."""
    coefficients: Dict[str, float] = Field(..., description="Model coefficients")
    x_new: List[List[float]] = Field(..., description="New predictor values for prediction")
    predictor_names: Optional[List[str]] = Field(None, description="Predictor names")
    family: str = Field(..., description="Distribution family")
    link: Optional[str] = Field(None, description="Link function")
    scale: Optional[float] = Field(1.0, description="Scale parameter (dispersion)")
    alpha: Optional[float] = Field(0.05, description="Significance level for prediction intervals")
    add_intercept: bool = Field(True, description="Whether model includes intercept")


class GLMCompareRequest(BaseModel):
    """Request model for comparing multiple GLM models."""
    y_data: List[float] = Field(..., description="Response variable values")
    x_data: List[List[float]] = Field(..., description="Predictor variables")
    predictor_names: Optional[List[str]] = Field(None, description="Names for predictor variables")
    families: List[str] = Field(..., description="List of families to compare")
    n_trials: Optional[List[int]] = Field(None, description="Number of trials for binomial")
    add_intercept: bool = Field(True, description="Whether to add intercept term")


class GLMDiagnosticsRequest(BaseModel):
    """Request model for GLM diagnostics."""
    y_data: List[float] = Field(..., description="Response variable values")
    x_data: List[List[float]] = Field(..., description="Predictor variables")
    family: str = Field(..., description="Distribution family")
    link: Optional[str] = Field(None, description="Link function")
    n_trials: Optional[List[int]] = Field(None, description="Number of trials for binomial")
    add_intercept: bool = Field(True, description="Whether to add intercept term")


# ============================================================================
# Helper Functions
# ============================================================================

def safe_float(value: Any) -> Optional[float]:
    """Convert value to float, returning None if not possible."""
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
    """Recursively convert numpy types to Python native types for JSON serialization."""
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
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    else:
        return obj


def get_family(family_name: str, link_name: Optional[str] = None):
    """Get statsmodels family object from name."""
    family_name = family_name.lower()

    # Default links for each family
    default_links = {
        'poisson': 'log',
        'binomial': 'logit',
        'negativebinomial': 'log',
        'gamma': 'inverse',
        'gaussian': 'identity',
        'tweedie': 'log',
        'inverse_gaussian': 'inverse_squared'
    }

    # Get link function
    if link_name is None:
        link_name = default_links.get(family_name, 'identity')

    link_name = link_name.lower()

    # Map link names to objects
    link_map = {
        'log': log(),
        'logit': logit(),
        'identity': identity(),
        'inverse': inverse_power(),
        'inverse_squared': inverse_power(power=-2),
        'probit': probit(),
        'cloglog': cloglog(),
        'sqrt': inverse_power(power=0.5)
    }

    link_obj = link_map.get(link_name, identity())

    # Map family names to objects
    if family_name == 'poisson':
        return Poisson(link=link_obj)
    elif family_name == 'binomial':
        return Binomial(link=link_obj)
    elif family_name == 'negativebinomial':
        return NegativeBinomial(link=link_obj, alpha=1.0)
    elif family_name == 'gamma':
        return Gamma(link=link_obj)
    elif family_name == 'gaussian':
        return Gaussian(link=link_obj)
    elif family_name == 'tweedie':
        return Tweedie(link=link_obj, var_power=1.5)
    elif family_name == 'inverse_gaussian':
        return InverseGaussian(link=link_obj)
    else:
        raise ValueError(f"Unknown family: {family_name}")


def get_family_info(family_name: str) -> Dict:
    """Get information about a distribution family."""
    info = {
        'poisson': {
            'name': 'Poisson',
            'description': 'Count data (non-negative integers)',
            'typical_use': 'Defect counts, event counts, rare events',
            'default_link': 'log',
            'available_links': ['log', 'identity', 'sqrt'],
            'response_type': 'Count (0, 1, 2, ...)',
            'variance_function': 'Var(Y) = μ'
        },
        'binomial': {
            'name': 'Binomial',
            'description': 'Binary outcomes or proportions',
            'typical_use': 'Pass/fail, yes/no, success rates',
            'default_link': 'logit',
            'available_links': ['logit', 'probit', 'cloglog', 'log', 'identity'],
            'response_type': 'Binary (0/1) or proportion (0-1)',
            'variance_function': 'Var(Y) = μ(1-μ)'
        },
        'negativebinomial': {
            'name': 'Negative Binomial',
            'description': 'Overdispersed count data',
            'typical_use': 'Counts with variance > mean',
            'default_link': 'log',
            'available_links': ['log', 'identity', 'sqrt'],
            'response_type': 'Count (0, 1, 2, ...)',
            'variance_function': 'Var(Y) = μ + αμ²'
        },
        'gamma': {
            'name': 'Gamma',
            'description': 'Positive continuous data, often skewed',
            'typical_use': 'Wait times, costs, concentrations',
            'default_link': 'inverse',
            'available_links': ['inverse', 'log', 'identity'],
            'response_type': 'Positive continuous (> 0)',
            'variance_function': 'Var(Y) = μ²/ν'
        },
        'gaussian': {
            'name': 'Gaussian (Normal)',
            'description': 'Continuous data, symmetric distribution',
            'typical_use': 'General continuous responses',
            'default_link': 'identity',
            'available_links': ['identity', 'log', 'inverse'],
            'response_type': 'Continuous (-∞ to +∞)',
            'variance_function': 'Var(Y) = σ²'
        },
        'tweedie': {
            'name': 'Tweedie',
            'description': 'Mixed continuous-discrete (zeros + positives)',
            'typical_use': 'Insurance claims, rainfall amounts',
            'default_link': 'log',
            'available_links': ['log', 'identity'],
            'response_type': 'Zero or positive continuous',
            'variance_function': 'Var(Y) = μ^p'
        },
        'inverse_gaussian': {
            'name': 'Inverse Gaussian',
            'description': 'Positive continuous, highly skewed',
            'typical_use': 'Failure times, highly skewed positive data',
            'default_link': 'inverse_squared',
            'available_links': ['inverse_squared', 'inverse', 'log', 'identity'],
            'response_type': 'Positive continuous (> 0)',
            'variance_function': 'Var(Y) = μ³/λ'
        }
    }
    return info.get(family_name.lower(), {})


def interpret_coefficient(coef: float, family: str, link: str, predictor_name: str) -> str:
    """Generate interpretation text for a coefficient."""
    link = link.lower() if link else 'identity'

    if link == 'log':
        # Log link: exp(coef) is multiplicative effect
        multiplier = np.exp(coef)
        if multiplier > 1:
            pct_change = (multiplier - 1) * 100
            return f"A 1-unit increase in {predictor_name} multiplies the expected response by {multiplier:.3f} ({pct_change:.1f}% increase)"
        else:
            pct_change = (1 - multiplier) * 100
            return f"A 1-unit increase in {predictor_name} multiplies the expected response by {multiplier:.3f} ({pct_change:.1f}% decrease)"

    elif link == 'logit':
        # Logit link: exp(coef) is odds ratio
        odds_ratio = np.exp(coef)
        if odds_ratio > 1:
            return f"A 1-unit increase in {predictor_name} multiplies the odds by {odds_ratio:.3f} (OR = {odds_ratio:.3f})"
        else:
            return f"A 1-unit increase in {predictor_name} multiplies the odds by {odds_ratio:.3f} (OR = {odds_ratio:.3f})"

    elif link == 'identity':
        if coef > 0:
            return f"A 1-unit increase in {predictor_name} increases the expected response by {coef:.4f}"
        else:
            return f"A 1-unit increase in {predictor_name} decreases the expected response by {abs(coef):.4f}"

    elif link == 'inverse':
        return f"Effect of {predictor_name}: coefficient = {coef:.4f} (inverse link scale)"

    else:
        return f"Coefficient for {predictor_name}: {coef:.4f}"


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/families")
async def get_available_families():
    """Get information about available distribution families."""
    families = ['poisson', 'binomial', 'negativebinomial', 'gamma', 'gaussian', 'tweedie', 'inverse_gaussian']
    return {
        "families": [
            {
                "id": f,
                **get_family_info(f)
            }
            for f in families
        ]
    }


@router.post("/fit")
async def fit_glm(request: GLMFitRequest):
    """
    Fit a Generalized Linear Model.

    Returns coefficient estimates, standard errors, confidence intervals,
    p-values, and model fit statistics.
    """
    try:
        # Prepare data
        y = np.array(request.y_data)
        X = np.array(request.x_data)

        if len(y) != len(X):
            raise HTTPException(status_code=400, detail="Length of y_data must match number of rows in x_data")

        if len(y) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 observations")

        n_obs = len(y)
        n_predictors = X.shape[1] if len(X.shape) > 1 else 1

        # Generate predictor names if not provided
        if request.predictor_names:
            predictor_names = request.predictor_names
        else:
            predictor_names = [f"X{i+1}" for i in range(n_predictors)]

        # Add intercept if requested
        if request.add_intercept:
            X = sm.add_constant(X)
            all_names = ["Intercept"] + predictor_names
        else:
            all_names = predictor_names

        # Get family
        family = get_family(request.family, request.link)
        link_name = request.link or get_family_info(request.family).get('default_link', 'identity')

        # Handle binomial with trials
        if request.family.lower() == 'binomial' and request.n_trials:
            # Convert to proportion format for binomial with trials
            n_trials = np.array(request.n_trials)
            y_prop = y / n_trials
            model = sm.GLM(y_prop, X, family=family, freq_weights=n_trials)
        else:
            model = sm.GLM(y, X, family=family)

        # Fit model
        try:
            result = model.fit()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Model fitting failed: {str(e)}")

        # Extract results
        coefficients = {}
        for i, name in enumerate(all_names):
            coef = result.params[i]
            se = result.bse[i]
            z_val = result.tvalues[i]
            p_val = result.pvalues[i]

            # Confidence intervals
            ci = result.conf_int(alpha=request.alpha)
            ci_lower = ci[i, 0]
            ci_upper = ci[i, 1]

            # For log/logit links, also compute exp(coef)
            exp_coef = None
            exp_ci_lower = None
            exp_ci_upper = None
            if link_name.lower() in ['log', 'logit']:
                exp_coef = float(np.exp(coef))
                exp_ci_lower = float(np.exp(ci_lower))
                exp_ci_upper = float(np.exp(ci_upper))

            interpretation = interpret_coefficient(coef, request.family, link_name, name)

            coefficients[name] = {
                "estimate": safe_float(coef),
                "std_error": safe_float(se),
                "z_value": safe_float(z_val),
                "p_value": safe_float(p_val),
                "ci_lower": safe_float(ci_lower),
                "ci_upper": safe_float(ci_upper),
                "significant": bool(p_val < request.alpha) if p_val is not None else None,
                "exp_estimate": exp_coef,
                "exp_ci_lower": exp_ci_lower,
                "exp_ci_upper": exp_ci_upper,
                "interpretation": interpretation
            }

        # Model statistics
        deviance = safe_float(result.deviance)
        null_deviance = safe_float(result.null_deviance)
        pearson_chi2 = safe_float(result.pearson_chi2)

        # Pseudo R-squared (McFadden's)
        pseudo_r2 = None
        if null_deviance and null_deviance > 0:
            pseudo_r2 = safe_float(1 - (deviance / null_deviance))

        # AIC and BIC
        aic = safe_float(result.aic)
        bic = safe_float(result.bic)

        # Log-likelihood
        llf = safe_float(result.llf)

        # Degrees of freedom
        df_model = int(result.df_model)
        df_resid = int(result.df_resid)

        # Dispersion/scale parameter
        scale = safe_float(result.scale)

        # Fitted values and residuals
        fitted_values = result.fittedvalues.tolist()

        # Different types of residuals
        deviance_resid = result.resid_deviance.tolist()
        pearson_resid = result.resid_pearson.tolist()
        response_resid = result.resid_response.tolist()
        working_resid = result.resid_working.tolist()

        # Determine family-specific interpretations
        family_info = get_family_info(request.family)

        # Overall model interpretation
        if pseudo_r2 is not None:
            if pseudo_r2 >= 0.4:
                model_quality = "Good fit"
            elif pseudo_r2 >= 0.2:
                model_quality = "Moderate fit"
            else:
                model_quality = "Weak fit"
        else:
            model_quality = "Unable to assess"

        return make_json_safe({
            "success": True,
            "family": request.family,
            "link": link_name,
            "family_info": family_info,
            "response_name": request.response_name,
            "predictor_names": predictor_names,
            "coefficients": coefficients,
            "statistics": {
                "n_observations": n_obs,
                "n_predictors": n_predictors,
                "df_model": df_model,
                "df_residual": df_resid,
                "deviance": deviance,
                "null_deviance": null_deviance,
                "pearson_chi2": pearson_chi2,
                "pseudo_r_squared": pseudo_r2,
                "aic": aic,
                "bic": bic,
                "log_likelihood": llf,
                "scale": scale,
                "model_quality": model_quality
            },
            "fitted_values": fitted_values,
            "residuals": {
                "deviance": deviance_resid,
                "pearson": pearson_resid,
                "response": response_resid,
                "working": working_resid
            },
            "y_data": request.y_data,
            "interpretation": f"GLM with {family_info.get('name', request.family)} distribution and {link_name} link function. Pseudo R² = {pseudo_r2:.4f if pseudo_r2 else 'N/A'} ({model_quality}). AIC = {aic:.2f if aic else 'N/A'}."
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fitting GLM: {str(e)}")


@router.post("/predict")
async def predict_glm(request: GLMPredictRequest):
    """
    Make predictions using fitted GLM coefficients.

    Returns predicted values on both link and response scales,
    with confidence/prediction intervals.
    """
    try:
        X_new = np.array(request.x_new)
        n_pred = len(X_new)

        # Add intercept if needed
        if request.add_intercept:
            X_new = sm.add_constant(X_new)
            coef_names = ["Intercept"] + (request.predictor_names or [f"X{i+1}" for i in range(X_new.shape[1]-1)])
        else:
            coef_names = request.predictor_names or [f"X{i+1}" for i in range(X_new.shape[1])]

        # Extract coefficients in order
        coefs = np.array([request.coefficients.get(name, 0) for name in coef_names])

        # Linear predictor (eta)
        eta = X_new @ coefs

        # Get link function for inverse transformation
        link_name = request.link or get_family_info(request.family).get('default_link', 'identity')
        link_name = link_name.lower()

        # Apply inverse link to get predictions on response scale
        if link_name == 'log':
            mu = np.exp(eta)
        elif link_name == 'logit':
            mu = 1 / (1 + np.exp(-eta))
        elif link_name == 'probit':
            from scipy.stats import norm
            mu = norm.cdf(eta)
        elif link_name == 'cloglog':
            mu = 1 - np.exp(-np.exp(eta))
        elif link_name == 'inverse':
            mu = 1 / eta
        elif link_name == 'inverse_squared':
            mu = 1 / np.sqrt(eta)
        elif link_name == 'sqrt':
            mu = eta ** 2
        else:  # identity
            mu = eta

        # Standard error of prediction (approximate)
        # For now, provide a simple estimate based on scale
        se_eta = np.sqrt(request.scale) * np.ones(n_pred)  # Simplified

        # Confidence intervals on link scale
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - request.alpha / 2)
        eta_lower = eta - z_alpha * se_eta
        eta_upper = eta + z_alpha * se_eta

        # Transform CI to response scale
        if link_name == 'log':
            mu_lower = np.exp(eta_lower)
            mu_upper = np.exp(eta_upper)
        elif link_name == 'logit':
            mu_lower = 1 / (1 + np.exp(-eta_lower))
            mu_upper = 1 / (1 + np.exp(-eta_upper))
        elif link_name == 'identity':
            mu_lower = eta_lower
            mu_upper = eta_upper
        else:
            mu_lower = mu - z_alpha * np.sqrt(request.scale)
            mu_upper = mu + z_alpha * np.sqrt(request.scale)

        predictions = []
        for i in range(n_pred):
            predictions.append({
                "x_values": X_new[i].tolist(),
                "linear_predictor": safe_float(eta[i]),
                "predicted_mean": safe_float(mu[i]),
                "ci_lower": safe_float(mu_lower[i]),
                "ci_upper": safe_float(mu_upper[i]),
                "se_link": safe_float(se_eta[i])
            })

        return make_json_safe({
            "success": True,
            "family": request.family,
            "link": link_name,
            "n_predictions": n_pred,
            "predictions": predictions,
            "confidence_level": 1 - request.alpha
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")


@router.post("/compare")
async def compare_glm_models(request: GLMCompareRequest):
    """
    Compare multiple GLM models with different families.

    Returns AIC, BIC, deviance, and pseudo R² for each family
    to help select the best distribution.
    """
    try:
        y = np.array(request.y_data)
        X = np.array(request.x_data)

        if len(y) != len(X):
            raise HTTPException(status_code=400, detail="Length of y_data must match number of rows in x_data")

        if request.add_intercept:
            X = sm.add_constant(X)

        results = []

        for family_name in request.families:
            try:
                family = get_family(family_name, None)
                family_info = get_family_info(family_name)

                # Handle binomial
                if family_name.lower() == 'binomial' and request.n_trials:
                    n_trials = np.array(request.n_trials)
                    y_prop = y / n_trials
                    model = sm.GLM(y_prop, X, family=family, freq_weights=n_trials)
                else:
                    model = sm.GLM(y, X, family=family)

                result = model.fit()

                # Calculate pseudo R²
                pseudo_r2 = None
                if result.null_deviance and result.null_deviance > 0:
                    pseudo_r2 = 1 - (result.deviance / result.null_deviance)

                results.append({
                    "family": family_name,
                    "family_name": family_info.get('name', family_name),
                    "link": family_info.get('default_link', 'identity'),
                    "aic": safe_float(result.aic),
                    "bic": safe_float(result.bic),
                    "deviance": safe_float(result.deviance),
                    "null_deviance": safe_float(result.null_deviance),
                    "pseudo_r_squared": safe_float(pseudo_r2),
                    "log_likelihood": safe_float(result.llf),
                    "converged": True,
                    "error": None
                })

            except Exception as e:
                results.append({
                    "family": family_name,
                    "family_name": get_family_info(family_name).get('name', family_name),
                    "link": get_family_info(family_name).get('default_link', 'identity'),
                    "aic": None,
                    "bic": None,
                    "deviance": None,
                    "null_deviance": None,
                    "pseudo_r_squared": None,
                    "log_likelihood": None,
                    "converged": False,
                    "error": str(e)
                })

        # Sort by AIC (best first)
        valid_results = [r for r in results if r['aic'] is not None]
        if valid_results:
            valid_results.sort(key=lambda x: x['aic'])
            best_family = valid_results[0]['family']
            best_aic = valid_results[0]['aic']

            # Calculate delta AIC
            for r in results:
                if r['aic'] is not None:
                    r['delta_aic'] = r['aic'] - best_aic
                else:
                    r['delta_aic'] = None
        else:
            best_family = None
            best_aic = None

        return make_json_safe({
            "success": True,
            "n_observations": len(y),
            "families_compared": len(request.families),
            "results": results,
            "best_family": best_family,
            "recommendation": f"Based on AIC, the {get_family_info(best_family).get('name', best_family)} distribution provides the best fit." if best_family else "Unable to determine best model."
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing models: {str(e)}")


@router.post("/diagnostics")
async def glm_diagnostics(request: GLMDiagnosticsRequest):
    """
    Compute comprehensive diagnostics for a fitted GLM.

    Returns residual analysis, influence measures, and
    goodness-of-fit tests.
    """
    try:
        y = np.array(request.y_data)
        X = np.array(request.x_data)

        if request.add_intercept:
            X = sm.add_constant(X)

        family = get_family(request.family, request.link)
        link_name = request.link or get_family_info(request.family).get('default_link', 'identity')

        # Fit model
        if request.family.lower() == 'binomial' and request.n_trials:
            n_trials = np.array(request.n_trials)
            y_prop = y / n_trials
            model = sm.GLM(y_prop, X, family=family, freq_weights=n_trials)
        else:
            model = sm.GLM(y, X, family=family)

        result = model.fit()

        # Get influence measures
        influence = result.get_influence()

        # Hat values (leverage)
        hat_values = influence.hat_matrix_diag.tolist()

        # Cook's distance
        cooks_d = influence.cooks_distance[0].tolist()

        # DFBETAS
        dfbetas = influence.dfbetas.tolist() if hasattr(influence, 'dfbetas') else None

        # Residuals
        deviance_resid = result.resid_deviance.tolist()
        pearson_resid = result.resid_pearson.tolist()

        # Standardized residuals
        std_resid = (result.resid_pearson / np.sqrt(result.scale)).tolist()

        # Fitted values
        fitted = result.fittedvalues.tolist()

        # Identify potential outliers/influential points
        outliers = []
        high_leverage = []
        influential = []

        n = len(y)
        p = X.shape[1]
        leverage_threshold = 2 * p / n
        cooks_threshold = 4 / n

        for i in range(n):
            if abs(std_resid[i]) > 2:
                outliers.append({
                    "index": i,
                    "value": safe_float(y[i]),
                    "std_residual": safe_float(std_resid[i]),
                    "type": "potential outlier"
                })

            if hat_values[i] > leverage_threshold:
                high_leverage.append({
                    "index": i,
                    "value": safe_float(y[i]),
                    "leverage": safe_float(hat_values[i]),
                    "threshold": safe_float(leverage_threshold)
                })

            if cooks_d[i] > cooks_threshold:
                influential.append({
                    "index": i,
                    "value": safe_float(y[i]),
                    "cooks_d": safe_float(cooks_d[i]),
                    "threshold": safe_float(cooks_threshold)
                })

        # Dispersion test (for Poisson: check if variance > mean suggesting negative binomial)
        dispersion_test = None
        if request.family.lower() == 'poisson':
            # Cameron-Trivedi test for overdispersion
            pearson_chi2 = result.pearson_chi2
            df_resid = result.df_resid
            dispersion_ratio = pearson_chi2 / df_resid

            if dispersion_ratio > 1.5:
                dispersion_interpretation = "Evidence of overdispersion. Consider Negative Binomial."
            elif dispersion_ratio < 0.5:
                dispersion_interpretation = "Evidence of underdispersion."
            else:
                dispersion_interpretation = "Dispersion appears appropriate for Poisson."

            dispersion_test = {
                "pearson_chi2": safe_float(pearson_chi2),
                "df": int(df_resid),
                "dispersion_ratio": safe_float(dispersion_ratio),
                "interpretation": dispersion_interpretation
            }

        # Link test (simple version)
        # Check if linear predictor squared is significant
        eta = result.predict(X, linear=True)
        eta_sq = eta ** 2
        X_link = np.column_stack([X, eta_sq])
        try:
            link_test_model = sm.GLM(y, X_link, family=family).fit()
            link_test_pval = link_test_model.pvalues[-1]
            if link_test_pval < 0.05:
                link_interpretation = f"Link test suggests potential misspecification (p={link_test_pval:.4f}). Consider alternative link function."
            else:
                link_interpretation = f"Link function appears appropriate (p={link_test_pval:.4f})."

            link_test = {
                "p_value": safe_float(link_test_pval),
                "interpretation": link_interpretation
            }
        except:
            link_test = None

        return make_json_safe({
            "success": True,
            "family": request.family,
            "link": link_name,
            "n_observations": len(y),
            "residuals": {
                "deviance": deviance_resid,
                "pearson": pearson_resid,
                "standardized": std_resid
            },
            "fitted_values": fitted,
            "influence": {
                "hat_values": hat_values,
                "cooks_distance": cooks_d,
                "leverage_threshold": safe_float(leverage_threshold),
                "cooks_threshold": safe_float(cooks_threshold)
            },
            "diagnostics": {
                "outliers": outliers,
                "high_leverage_points": high_leverage,
                "influential_points": influential,
                "dispersion_test": dispersion_test,
                "link_test": link_test
            },
            "summary": {
                "n_outliers": len(outliers),
                "n_high_leverage": len(high_leverage),
                "n_influential": len(influential)
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing diagnostics: {str(e)}")


@router.post("/deviance-analysis")
async def deviance_analysis(request: GLMFitRequest):
    """
    Perform analysis of deviance (analogous to ANOVA for GLM).

    Tests significance of each predictor using likelihood ratio tests.
    """
    try:
        y = np.array(request.y_data)
        X = np.array(request.x_data)

        n_obs = len(y)
        n_predictors = X.shape[1] if len(X.shape) > 1 else 1

        # Generate predictor names
        if request.predictor_names:
            predictor_names = request.predictor_names
        else:
            predictor_names = [f"X{i+1}" for i in range(n_predictors)]

        family = get_family(request.family, request.link)

        # Fit full model
        if request.add_intercept:
            X_full = sm.add_constant(X)
        else:
            X_full = X

        if request.family.lower() == 'binomial' and request.n_trials:
            n_trials = np.array(request.n_trials)
            y_fit = y / n_trials
            full_model = sm.GLM(y_fit, X_full, family=family, freq_weights=n_trials).fit()
        else:
            full_model = sm.GLM(y, X_full, family=family).fit()

        full_deviance = full_model.deviance
        full_df = full_model.df_resid

        # Fit null model (intercept only)
        if request.add_intercept:
            X_null = np.ones((n_obs, 1))
            if request.family.lower() == 'binomial' and request.n_trials:
                null_model = sm.GLM(y_fit, X_null, family=family, freq_weights=n_trials).fit()
            else:
                null_model = sm.GLM(y, X_null, family=family).fit()
            null_deviance = null_model.deviance
            null_df = null_model.df_resid
        else:
            null_deviance = full_model.null_deviance
            null_df = n_obs - 1

        # Test each predictor by dropping it
        from scipy.stats import chi2

        deviance_table = []

        # Model deviance (all predictors)
        model_dev_change = null_deviance - full_deviance
        model_df_change = null_df - full_df
        model_p_value = 1 - chi2.cdf(model_dev_change, model_df_change) if model_df_change > 0 else 1.0

        deviance_table.append({
            "source": "Model",
            "df": int(model_df_change),
            "deviance": safe_float(model_dev_change),
            "mean_deviance": safe_float(model_dev_change / model_df_change) if model_df_change > 0 else None,
            "chi_square": safe_float(model_dev_change),
            "p_value": safe_float(model_p_value),
            "significant": bool(model_p_value < request.alpha)
        })

        # Test each predictor individually (Type III - like)
        for i, name in enumerate(predictor_names):
            # Create X matrix without this predictor
            col_idx = i + 1 if request.add_intercept else i
            X_reduced = np.delete(X_full, col_idx, axis=1)

            try:
                if request.family.lower() == 'binomial' and request.n_trials:
                    reduced_model = sm.GLM(y_fit, X_reduced, family=family, freq_weights=n_trials).fit()
                else:
                    reduced_model = sm.GLM(y, X_reduced, family=family).fit()

                dev_change = reduced_model.deviance - full_deviance
                df_change = 1
                p_value = 1 - chi2.cdf(dev_change, df_change)

                deviance_table.append({
                    "source": name,
                    "df": 1,
                    "deviance": safe_float(dev_change),
                    "mean_deviance": safe_float(dev_change),
                    "chi_square": safe_float(dev_change),
                    "p_value": safe_float(p_value),
                    "significant": bool(p_value < request.alpha)
                })
            except Exception as e:
                deviance_table.append({
                    "source": name,
                    "df": 1,
                    "deviance": None,
                    "mean_deviance": None,
                    "chi_square": None,
                    "p_value": None,
                    "significant": None,
                    "error": str(e)
                })

        # Residual deviance
        deviance_table.append({
            "source": "Residual",
            "df": int(full_df),
            "deviance": safe_float(full_deviance),
            "mean_deviance": safe_float(full_deviance / full_df) if full_df > 0 else None,
            "chi_square": None,
            "p_value": None,
            "significant": None
        })

        # Total deviance
        deviance_table.append({
            "source": "Total",
            "df": int(null_df),
            "deviance": safe_float(null_deviance),
            "mean_deviance": None,
            "chi_square": None,
            "p_value": None,
            "significant": None
        })

        return make_json_safe({
            "success": True,
            "family": request.family,
            "link": request.link or get_family_info(request.family).get('default_link', 'identity'),
            "n_observations": n_obs,
            "deviance_table": deviance_table,
            "interpretation": "Analysis of Deviance table. Significant p-values (< {:.2f}) indicate the predictor contributes significantly to the model.".format(request.alpha)
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in deviance analysis: {str(e)}")
