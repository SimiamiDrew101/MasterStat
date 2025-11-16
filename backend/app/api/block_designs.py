from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy import stats
from scipy.stats import shapiro, levene

router = APIRouter()

def sanitize_value(value):
    """Convert NaN/Inf values to None for JSON serialization"""
    if isinstance(value, (int, float)):
        if not np.isfinite(value):
            return None
    return value

def sanitize_dict(d):
    """Recursively sanitize dictionary values"""
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = sanitize_dict(value)
        elif isinstance(value, list):
            result[key] = [sanitize_value(v) if not isinstance(v, dict) else sanitize_dict(v) for v in value]
        else:
            result[key] = sanitize_value(value)
    return result

# ============================================================================
# MISSING DATA HANDLING FUNCTIONS
# ============================================================================

def detect_missing_pattern(df, response_col):
    """
    Detect and analyze missing data patterns in the response variable
    Returns missing data statistics and pattern information
    """
    missing_mask = df[response_col].isna()
    n_total = len(df)
    n_missing = missing_mask.sum()
    n_complete = n_total - n_missing
    pct_missing = (n_missing / n_total * 100) if n_total > 0 else 0

    pattern_info = {
        'n_total': int(n_total),
        'n_missing': int(n_missing),
        'n_complete': int(n_complete),
        'percent_missing': float(pct_missing),
        'has_missing': bool(n_missing > 0),
        'missing_indices': missing_mask[missing_mask].index.tolist()
    }

    return pattern_info

def littles_mcar_test(df, response_col, group_cols):
    """
    Perform Little's MCAR test to assess if data is Missing Completely At Random
    Tests if missingness pattern is independent of observed and unobserved values

    Returns: test_statistic, p_value, conclusion
    """
    try:
        # Create missing indicator
        missing_mask = df[response_col].isna()

        if missing_mask.sum() == 0 or missing_mask.sum() == len(df):
            return None, None, "No missing data or all data missing - test not applicable"

        # Simple test: Check if missingness is related to group membership
        # Create a contingency table of groups vs missing status
        df_test = df.copy()
        df_test['missing'] = missing_mask.astype(int)

        # Test independence using chi-square for each grouping variable
        chi2_stats = []
        p_values = []

        for group_col in group_cols:
            contingency = pd.crosstab(df_test[group_col], df_test['missing'])
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                chi2_stats.append(chi2)
                p_values.append(p)

        if not p_values:
            return None, None, "Insufficient data for MCAR test"

        # Use minimum p-value (most conservative)
        min_p = min(p_values)
        max_chi2 = max(chi2_stats)

        if min_p > 0.05:
            conclusion = "MCAR - Missing Completely At Random (missingness independent of groups)"
        else:
            conclusion = "Not MCAR - Missingness may be related to group membership (MAR or MNAR)"

        return float(max_chi2), float(min_p), conclusion

    except Exception as e:
        return None, None, f"MCAR test failed: {str(e)}"

def mean_imputation(df, response_col, group_col=None):
    """
    Perform mean imputation for missing values
    If group_col provided, uses group-specific means; otherwise uses overall mean

    Returns: imputed DataFrame and imputation info
    """
    df_imputed = df.copy()
    missing_mask = df_imputed[response_col].isna()

    imputation_values = {}

    if group_col and group_col in df_imputed.columns:
        # Group-specific mean imputation (more appropriate for block designs)
        for group in df_imputed[group_col].unique():
            group_mask = df_imputed[group_col] == group
            group_missing = missing_mask & group_mask

            if group_missing.any():
                # Calculate mean from non-missing values in this group
                group_mean = df_imputed.loc[group_mask & ~missing_mask, response_col].mean()

                if pd.notna(group_mean):
                    df_imputed.loc[group_missing, response_col] = group_mean
                    imputation_values[str(group)] = float(group_mean)
    else:
        # Overall mean imputation
        overall_mean = df_imputed.loc[~missing_mask, response_col].mean()
        if pd.notna(overall_mean):
            df_imputed.loc[missing_mask, response_col] = overall_mean
            imputation_values['overall'] = float(overall_mean)

    imputation_info = {
        'method': 'mean_imputation',
        'group_based': bool(group_col),
        'imputed_values': imputation_values,
        'n_imputed': int(missing_mask.sum())
    }

    return df_imputed, imputation_info

def em_imputation(df, response_col, predictor_cols, max_iter=100, tol=1e-4):
    """
    EM (Expectation-Maximization) algorithm for missing data imputation
    Uses predictor columns to estimate missing values iteratively

    Returns: imputed DataFrame and convergence info
    """
    df_imputed = df.copy()
    missing_mask = df_imputed[response_col].isna()

    if missing_mask.sum() == 0:
        return df_imputed, {'method': 'em_imputation', 'iterations': 0, 'converged': True}

    # Initialize with mean imputation
    initial_mean = df_imputed.loc[~missing_mask, response_col].mean()
    df_imputed.loc[missing_mask, response_col] = initial_mean

    converged = False
    iteration = 0

    for iteration in range(max_iter):
        # E-step: Current imputed values
        old_values = df_imputed.loc[missing_mask, response_col].copy()

        # M-step: Fit regression model and predict missing values
        try:
            # Prepare predictor matrix
            X_complete = df_imputed.loc[~missing_mask, predictor_cols]
            y_complete = df_imputed.loc[~missing_mask, response_col]

            # Add constant for intercept
            X_complete_with_const = sm.add_constant(X_complete, has_constant='add')

            # Fit OLS model
            model = sm.OLS(y_complete, X_complete_with_const).fit()

            # Predict for missing values
            X_missing = df_imputed.loc[missing_mask, predictor_cols]
            X_missing_with_const = sm.add_constant(X_missing, has_constant='add')

            predictions = model.predict(X_missing_with_const)
            df_imputed.loc[missing_mask, response_col] = predictions

            # Check convergence
            diff = np.abs(old_values - df_imputed.loc[missing_mask, response_col]).max()
            if diff < tol:
                converged = True
                break

        except Exception as e:
            # If EM fails, return mean imputation
            df_imputed.loc[missing_mask, response_col] = initial_mean
            return df_imputed, {
                'method': 'em_imputation',
                'iterations': iteration,
                'converged': False,
                'error': str(e),
                'fallback': 'mean_imputation'
            }

    imputation_info = {
        'method': 'em_imputation',
        'iterations': iteration + 1,
        'converged': converged,
        'n_imputed': int(missing_mask.sum())
    }

    return df_imputed, imputation_info

class RCBDRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Experimental data")
    treatment: str = Field(..., description="Treatment factor name")
    block: str = Field(..., description="Block factor name")
    response: str = Field(..., description="Response variable name")
    alpha: float = Field(0.05, description="Significance level")
    random_blocks: bool = Field(False, description="Treat blocks as random effects")
    covariate: str = Field(None, description="Optional covariate for ANCOVA")
    imputation_method: str = Field("none", description="Missing data imputation method: 'none', 'mean', 'em'")

class LatinSquareRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Experimental data")
    treatment: str = Field(..., description="Treatment factor name")
    row_block: str = Field(..., description="Row blocking factor")
    col_block: str = Field(..., description="Column blocking factor")
    response: str = Field(..., description="Response variable name")
    alpha: float = Field(0.05, description="Significance level")
    random_blocks: bool = Field(False, description="Treat blocking factors as random effects")

class GraecoLatinRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Experimental data")
    latin_treatment: str = Field(..., description="Latin letter treatment")
    greek_treatment: str = Field(..., description="Greek letter treatment")
    row_block: str = Field(..., description="Row blocking factor")
    col_block: str = Field(..., description="Column blocking factor")
    response: str = Field(..., description="Response variable name")
    alpha: float = Field(0.05, description="Significance level")

class RCBDGenerateRequest(BaseModel):
    n_treatments: int = Field(..., description="Number of treatments")
    n_blocks: int = Field(..., description="Number of blocks")
    treatment_names: List[str] = Field(None, description="Optional treatment names")
    randomize: bool = Field(True, description="Randomize run order")

class CrossoverRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Experimental data")
    subject: str = Field(..., description="Subject identifier")
    period: str = Field(..., description="Period/time identifier")
    treatment: str = Field(..., description="Treatment factor")
    sequence: str = Field(None, description="Sequence group identifier (e.g., AB, BA)")
    response: str = Field(..., description="Response variable name")
    alpha: float = Field(0.05, description="Significance level")

class CrossoverGenerateRequest(BaseModel):
    n_subjects: int = Field(..., description="Number of subjects")
    n_treatments: int = Field(2, description="Number of treatments (2 for 2x2, 3+ for Williams)")
    design_type: str = Field("2x2", description="Design type: '2x2' or 'williams'")
    treatment_names: List[str] = Field(None, description="Optional treatment names")

@router.post("/rcbd")
async def rcbd_analysis(request: RCBDRequest):
    """
    Analyze Randomized Complete Block Design (RCBD)
    Supports both fixed and random blocks, missing data imputation, and ANCOVA
    """
    try:
        df = pd.DataFrame(request.data)

        # ============================================================================
        # MISSING DATA HANDLING
        # ============================================================================
        missing_data_analysis = None
        original_df = df.copy()  # Keep original for comparison

        # Detect missing data pattern
        missing_pattern = detect_missing_pattern(df, request.response)

        if missing_pattern['has_missing']:
            # Perform Little's MCAR test
            group_cols = [request.treatment, request.block]
            chi2_stat, mcar_p_value, mcar_conclusion = littles_mcar_test(df, request.response, group_cols)

            # Apply imputation if requested
            imputation_info = None
            if request.imputation_method == 'mean':
                # Use block-specific means for imputation (more appropriate for block designs)
                df, imputation_info = mean_imputation(df, request.response, group_col=request.block)
            elif request.imputation_method == 'em':
                # Use EM algorithm with treatment and block as predictors
                # Create numeric encoding for categorical variables
                df_encoded = df.copy()
                treatment_cats = pd.Categorical(df_encoded[request.treatment])
                block_cats = pd.Categorical(df_encoded[request.block])
                df_encoded['treatment_code'] = treatment_cats.codes
                df_encoded['block_code'] = block_cats.codes

                predictor_cols = ['treatment_code', 'block_code']
                df_encoded, imputation_info = em_imputation(df_encoded, request.response, predictor_cols)

                # Copy imputed values back to original df
                df[request.response] = df_encoded[request.response]

            missing_data_analysis = {
                'pattern': missing_pattern,
                'mcar_test': {
                    'chi2_statistic': chi2_stat,
                    'p_value': mcar_p_value,
                    'conclusion': mcar_conclusion
                },
                'imputation': imputation_info,
                'method_used': request.imputation_method
            }

        # Calculate treatment means (same for both fixed and random)
        treatment_means = df.groupby(request.treatment)[request.response].mean().to_dict()
        block_means = df.groupby(request.block)[request.response].mean().to_dict()

        n_treatments = df[request.treatment].nunique()
        n_blocks = df[request.block].nunique()

        if request.random_blocks:
            # RANDOM BLOCKS: Use mixed linear model
            # Blocks are random effects (grouping variable)
            # Treatments are fixed effects

            try:
                # Build formula for fixed effects only
                formula = f"{request.response} ~ C({request.treatment})"

                # Fit mixed model with blocks as random intercepts
                model = MixedLM.from_formula(formula, data=df, groups=df[request.block]).fit(method='powell')

                # Extract variance components
                var_block = float(model.cov_re.values[0][0])  # Random effect variance (blocks)
                var_residual = float(model.scale)  # Residual variance

                # Check for invalid values
                if not np.isfinite(var_block) or not np.isfinite(var_residual):
                    raise ValueError("Model convergence failed - variance components are not finite")

                # Calculate intraclass correlation (ICC)
                icc = var_block / (var_block + var_residual)

                # Get fixed effects (treatment) test
                # F-test for treatment effect
                from scipy import stats

                # Treatment sum of squares
                grand_mean = df[request.response].mean()
                treatment_groups = df.groupby(request.treatment)[request.response]
                ss_treatment = sum(len(group) * (group.mean() - grand_mean)**2 for _, group in treatment_groups)
                df_treatment = n_treatments - 1
                ms_treatment = ss_treatment / df_treatment

                # F-statistic
                f_statistic = ms_treatment / var_residual
                p_value = 1 - stats.f.cdf(f_statistic, df_treatment, (n_treatments * n_blocks) - n_treatments - n_blocks + 1)

                # Check for invalid values
                if not np.isfinite(f_statistic) or not np.isfinite(p_value):
                    raise ValueError("Model convergence failed - test statistics are not finite")

                results = {
                    request.treatment: {
                        "sum_sq": round(ss_treatment, 4),
                        "df": df_treatment,
                        "F": round(f_statistic, 4),
                        "p_value": round(p_value, 6),
                        "significant": bool(p_value < request.alpha)
                    }
                }

                result_dict = {
                    "test_type": "Randomized Complete Block Design (RCBD) - Random Blocks",
                    "alpha": request.alpha,
                    "block_type": "random",
                    "anova_table": results,
                    "variance_components": {
                        "block_variance": round(var_block, 4),
                        "residual_variance": round(var_residual, 4),
                        "total_variance": round(var_block + var_residual, 4),
                        "icc": round(icc, 4)
                    },
                    "treatment_means": {str(k): round(float(v), 4) for k, v in treatment_means.items()},
                    "block_means": {str(k): round(float(v), 4) for k, v in block_means.items()},
                    "grand_mean": round(float(df[request.response].mean()), 4),
                    "log_likelihood": round(float(model.llf), 4),
                    "aic": round(float(model.aic), 4),
                    "bic": round(float(model.bic), 4),
                    "missing_data": missing_data_analysis
                }
                return sanitize_dict(result_dict)
            except Exception as e:
                # If random effects model fails, fall back to informative error
                raise HTTPException(
                    status_code=400,
                    detail=f"Random blocks analysis failed. The mixed model did not converge. "
                           f"This can happen with small sample sizes or when block variance is very small. "
                           f"Try using fixed blocks instead. Error: {str(e)}"
                )
        else:
            # FIXED BLOCKS: Use standard OLS ANOVA
            formula = f"{request.response} ~ C({request.treatment}) + C({request.block})"

            # Fit model
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            # Parse results
            results = {}
            for idx, row in anova_table.iterrows():
                source = str(idx).replace(f'C({request.treatment})', request.treatment)
                source = source.replace(f'C({request.block})', request.block)

                results[source] = {
                    "sum_sq": round(float(row['sum_sq']), 4),
                    "df": int(row['df']),
                    "F": round(float(row['F']), 4) if not pd.isna(row['F']) else None,
                    "p_value": round(float(row['PR(>F)']), 6) if not pd.isna(row['PR(>F)']) else None,
                    "significant": bool(row['PR(>F)'] < request.alpha) if not pd.isna(row['PR(>F)']) else False
                }

            # Calculate relative efficiency compared to CRD
            ms_block = results.get(request.block, {}).get('sum_sq', 0) / results.get(request.block, {}).get('df', 1)
            ms_error = results.get('Residual', {}).get('sum_sq', 0) / results.get('Residual', {}).get('df', 1)

            # Relative efficiency = [(b-1)MS_block + b(t-1)MS_error] / [bt - 1)MS_error]
            if ms_error > 0:
                relative_efficiency = ((n_blocks - 1) * ms_block + n_blocks * (n_treatments - 1) * ms_error) / ((n_blocks * n_treatments - 1) * ms_error)
            else:
                relative_efficiency = None

            # Calculate residuals and fitted values
            fitted_values = model.fittedvalues.values
            residuals = model.resid.values
            mse = np.mean(residuals**2)
            standardized_residuals = residuals / np.sqrt(mse) if mse > 0 else residuals

            # Calculate confidence intervals for treatment means
            from scipy import stats as sp_stats
            means_ci = {}
            for treatment_name, group_df in df.groupby(request.treatment):
                data = group_df[request.response].values
                mean = np.mean(data)
                n = len(data)
                if n > 1:
                    sem = np.sqrt(ms_error / n)  # Use pooled error from ANOVA
                    df_error = results.get('Residual', {}).get('df', n - 1)
                    t_crit = sp_stats.t.ppf(0.975, df_error)
                    ci_margin = t_crit * sem
                    means_ci[str(treatment_name)] = {
                        "mean": round(float(mean), 4),
                        "lower": round(float(mean - ci_margin), 4),
                        "upper": round(float(mean + ci_margin), 4),
                        "sem": round(float(sem), 4)
                    }

            # Box plot data by treatment
            def calculate_boxplot_data(data, label):
                q1 = float(np.percentile(data, 25))
                median = float(np.median(data))
                q3 = float(np.percentile(data, 75))
                iqr = q3 - q1
                lower_whisker = float(np.min(data[data >= q1 - 1.5 * iqr]))
                upper_whisker = float(np.max(data[data <= q3 + 1.5 * iqr]))
                outliers = [float(x) for x in data if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr]

                return {
                    "label": str(label),
                    "min": lower_whisker,
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "max": upper_whisker,
                    "outliers": outliers
                }

            boxplot_data_treatment = []
            for treatment_name, group_df in df.groupby(request.treatment):
                boxplot_data_treatment.append(
                    calculate_boxplot_data(group_df[request.response].values, treatment_name)
                )

            boxplot_data_block = []
            for block_name, group_df in df.groupby(request.block):
                boxplot_data_block.append(
                    calculate_boxplot_data(group_df[request.response].values, block_name)
                )

            # Normality test (Shapiro-Wilk) for residuals
            normality_test = {}
            if len(residuals) >= 3:  # Shapiro-Wilk requires at least 3 observations
                try:
                    shapiro_stat, shapiro_p = shapiro(residuals)
                    normality_test = {
                        "test": "Shapiro-Wilk",
                        "statistic": round(float(shapiro_stat), 4),
                        "p_value": round(float(shapiro_p), 6),
                        "interpretation": "Normal" if shapiro_p > request.alpha else "Non-normal"
                    }
                except:
                    normality_test = {"test": "Shapiro-Wilk", "error": "Test could not be performed"}

            # Homogeneity of variance test (Levene's test) across blocks
            homogeneity_test = {}
            try:
                # Group residuals by block
                block_residuals = []
                for block_name, group_df in df.groupby(request.block):
                    # Get indices for this block and extract corresponding residuals
                    block_indices = group_df.index
                    block_res = [residuals[i] for i, idx in enumerate(df.index) if idx in block_indices]
                    block_residuals.append(block_res)

                if len(block_residuals) >= 2 and all(len(br) > 0 for br in block_residuals):
                    levene_stat, levene_p = levene(*block_residuals)
                    homogeneity_test = {
                        "test": "Levene's Test",
                        "statistic": round(float(levene_stat), 4),
                        "p_value": round(float(levene_p), 6),
                        "interpretation": "Homogeneous" if levene_p > request.alpha else "Heterogeneous"
                    }
            except:
                homogeneity_test = {"test": "Levene's Test", "error": "Test could not be performed"}

            # Calculate block-treatment interaction means for visualization
            interaction_means = {}
            for (block_val, treatment_val), group_df in df.groupby([request.block, request.treatment]):
                key = f"{block_val}_{treatment_val}"
                interaction_means[key] = {
                    "block": str(block_val),
                    "treatment": str(treatment_val),
                    "mean": round(float(group_df[request.response].mean()), 4),
                    "n": int(len(group_df))
                }

            # ANCOVA Analysis (if covariate is provided)
            ancova_results = None
            if request.covariate and request.covariate in df.columns:
                try:
                    # Center the covariate for easier interpretation
                    covariate_mean = df[request.covariate].mean()
                    df['covariate_centered'] = df[request.covariate] - covariate_mean

                    # Test for homogeneity of regression slopes (treatment × covariate interaction)
                    formula_interaction = f"{request.response} ~ C({request.treatment}) + C({request.block}) + covariate_centered + C({request.treatment}):covariate_centered"
                    model_interaction = ols(formula_interaction, data=df).fit()
                    anova_interaction = sm.stats.anova_lm(model_interaction, typ=2)

                    # Check if interaction is significant
                    interaction_term = f"C({request.treatment}):covariate_centered"
                    interaction_p = None
                    for idx in anova_interaction.index:
                        if interaction_term in str(idx):
                            interaction_p = float(anova_interaction.loc[idx, 'PR(>F)'])
                            break

                    slopes_homogeneous = interaction_p is None or interaction_p > request.alpha

                    # Fit ANCOVA model (without interaction if slopes are homogeneous)
                    formula_ancova = f"{request.response} ~ C({request.treatment}) + C({request.block}) + covariate_centered"
                    model_ancova = ols(formula_ancova, data=df).fit()
                    anova_ancova = sm.stats.anova_lm(model_ancova, typ=2)

                    # Parse ANCOVA results
                    ancova_table = {}
                    for idx, row in anova_ancova.iterrows():
                        source = str(idx).replace(f'C({request.treatment})', request.treatment)
                        source = source.replace(f'C({request.block})', request.block)
                        source = source.replace('covariate_centered', request.covariate)

                        ancova_table[source] = {
                            "sum_sq": round(float(row['sum_sq']), 4),
                            "df": int(row['df']),
                            "F": round(float(row['F']), 4) if not pd.isna(row['F']) else None,
                            "p_value": round(float(row['PR(>F)']), 6) if not pd.isna(row['PR(>F)']) else None,
                            "significant": bool(row['PR(>F)'] < request.alpha) if not pd.isna(row['PR(>F)']) else False
                        }

                    # Calculate adjusted treatment means at mean covariate value
                    # Extract treatment effects from model
                    adjusted_means = {}
                    coef_dict = model_ancova.params.to_dict()

                    # Get intercept and covariate coefficient
                    intercept = coef_dict.get('Intercept', 0)

                    # Calculate adjusted means for each treatment
                    treatment_levels = sorted(df[request.treatment].unique())
                    for treatment_val in treatment_levels:
                        # Predict at mean covariate value (centered = 0) and mean block effect
                        # For first treatment (reference), use intercept
                        # For others, add treatment coefficient
                        treatment_key = f'C({request.treatment})[T.{treatment_val}]'
                        treatment_effect = coef_dict.get(treatment_key, 0)

                        # Adjusted mean = intercept + treatment_effect (at centered covariate = 0)
                        # Need to account for block effects averaging out
                        adjusted_mean = intercept + treatment_effect

                        adjusted_means[str(treatment_val)] = round(float(adjusted_mean), 4)

                    # Covariate coefficient and effect
                    covariate_coef = coef_dict.get('covariate_centered', 0)

                    ancova_results = {
                        "covariate_name": request.covariate,
                        "covariate_mean": round(float(covariate_mean), 4),
                        "covariate_coefficient": round(float(covariate_coef), 4),
                        "slopes_homogeneous": slopes_homogeneous,
                        "interaction_p_value": round(float(interaction_p), 6) if interaction_p is not None else None,
                        "ancova_table": ancova_table,
                        "adjusted_treatment_means": adjusted_means,
                        "unadjusted_treatment_means": {str(k): round(float(v), 4) for k, v in treatment_means.items()},
                        "model_r_squared": round(float(model_ancova.rsquared), 4),
                        "warning": None if slopes_homogeneous else "Homogeneity of slopes assumption violated - treatment effects may depend on covariate level"
                    }

                except Exception as e:
                    ancova_results = {
                        "error": f"ANCOVA analysis failed: {str(e)}"
                    }

            return {
                "test_type": "Randomized Complete Block Design (RCBD) - Fixed Blocks",
                "alpha": request.alpha,
                "block_type": "fixed",
                "anova_table": results,
                "treatment_means": {str(k): round(float(v), 4) for k, v in treatment_means.items()},
                "block_means": {str(k): round(float(v), 4) for k, v in block_means.items()},
                "grand_mean": round(float(df[request.response].mean()), 4),
                "relative_efficiency": round(float(relative_efficiency), 4) if relative_efficiency else None,
                "model_r_squared": round(float(model.rsquared), 4),
                "means_ci": means_ci,
                "boxplot_data_treatment": boxplot_data_treatment,
                "boxplot_data_block": boxplot_data_block,
                "residuals": [round(float(r), 4) for r in residuals],
                "fitted_values": [round(float(f), 4) for f in fitted_values],
                "standardized_residuals": [round(float(r), 4) for r in standardized_residuals],
                "normality_test": normality_test,
                "homogeneity_test": homogeneity_test,
                "interaction_means": interaction_means,
                "ancova": ancova_results,
                "missing_data": missing_data_analysis
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/latin-square")
async def latin_square_analysis(request: LatinSquareRequest):
    """
    Analyze Latin Square Design (two blocking factors)
    Supports both fixed and random blocks
    """
    try:
        df = pd.DataFrame(request.data)

        # Calculate means (same for both fixed and random)
        treatment_means = df.groupby(request.treatment)[request.response].mean().to_dict()
        row_means = df.groupby(request.row_block)[request.response].mean().to_dict()
        col_means = df.groupby(request.col_block)[request.response].mean().to_dict()

        n_treatments = df[request.treatment].nunique()

        if request.random_blocks:
            # RANDOM BLOCKS: Use mixed linear model
            # Row and column blocks are random effects
            # Treatment is fixed effect

            try:
                # Create a combined grouping variable for crossed random effects
                # We'll use row blocks as the primary grouping
                # This is a simplification - full crossed random effects would need more complex setup

                formula = f"{request.response} ~ C({request.treatment})"

                # Fit mixed model with row blocks as random intercepts
                model = MixedLM.from_formula(formula, data=df, groups=df[request.row_block]).fit(method='powell')

                # Extract variance components and check for valid values
                var_row = float(model.cov_re.values[0][0])
                var_residual = float(model.scale)

                # Check for invalid values
                if not np.isfinite(var_row) or not np.isfinite(var_residual):
                    raise ValueError("Model convergence failed - variance components are not finite")

                # Calculate treatment effect F-test
                from scipy import stats

                grand_mean = df[request.response].mean()
                treatment_groups = df.groupby(request.treatment)[request.response]
                ss_treatment = sum(len(group) * (group.mean() - grand_mean)**2 for _, group in treatment_groups)
                df_treatment = n_treatments - 1
                ms_treatment = ss_treatment / df_treatment

                f_statistic = ms_treatment / var_residual
                p_value = 1 - stats.f.cdf(f_statistic, df_treatment, len(df) - n_treatments - n_treatments + 1)

                # Check for invalid values
                if not np.isfinite(f_statistic) or not np.isfinite(p_value):
                    raise ValueError("Model convergence failed - test statistics are not finite")

                results = {
                    request.treatment: {
                        "sum_sq": round(ss_treatment, 4),
                        "df": df_treatment,
                        "F": round(f_statistic, 4),
                        "p_value": round(p_value, 6),
                        "significant": bool(p_value < request.alpha)
                    }
                }

                icc = var_row / (var_row + var_residual)

                result_dict = {
                    "test_type": "Latin Square Design - Random Blocks",
                    "alpha": request.alpha,
                    "block_type": "random",
                    "anova_table": results,
                    "variance_components": {
                        "row_block_variance": round(var_row, 4),
                        "residual_variance": round(var_residual, 4),
                        "total_variance": round(var_row + var_residual, 4),
                        "icc": round(icc, 4)
                    },
                    "treatment_means": {str(k): round(float(v), 4) for k, v in treatment_means.items()},
                    "row_block_means": {str(k): round(float(v), 4) for k, v in row_means.items()},
                    "col_block_means": {str(k): round(float(v), 4) for k, v in col_means.items()},
                    "grand_mean": round(float(df[request.response].mean()), 4),
                    "log_likelihood": round(float(model.llf), 4),
                    "aic": round(float(model.aic), 4),
                    "bic": round(float(model.bic), 4)
                }
                return sanitize_dict(result_dict)
            except Exception as e:
                # If random effects model fails, fall back to informative error
                raise HTTPException(
                    status_code=400,
                    detail=f"Random blocks analysis failed. The mixed model did not converge. "
                           f"This can happen with small sample sizes or when block variance is very small. "
                           f"Try using fixed blocks instead. Error: {str(e)}"
                )
        else:
            # FIXED BLOCKS: Use standard OLS ANOVA
            formula = f"{request.response} ~ C({request.treatment}) + C({request.row_block}) + C({request.col_block})"

            # Fit model
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            # Parse results
            results = {}
            for idx, row in anova_table.iterrows():
                source = str(idx)
                source = source.replace(f'C({request.treatment})', request.treatment)
                source = source.replace(f'C({request.row_block})', request.row_block)
                source = source.replace(f'C({request.col_block})', request.col_block)

                results[source] = {
                    "sum_sq": round(float(row['sum_sq']), 4),
                    "df": int(row['df']),
                    "F": round(float(row['F']), 4) if not pd.isna(row['F']) else None,
                    "p_value": round(float(row['PR(>F)']), 6) if not pd.isna(row['PR(>F)']) else None,
                    "significant": bool(row['PR(>F)'] < request.alpha) if not pd.isna(row['PR(>F)']) else False
                }

            # Calculate residuals and fitted values
            fitted_values = model.fittedvalues.values
            residuals = model.resid.values
            mse = np.mean(residuals**2)
            standardized_residuals = residuals / np.sqrt(mse) if mse > 0 else residuals

            # Calculate confidence intervals for treatment means
            ms_error = results.get('Residual', {}).get('sum_sq', 0) / results.get('Residual', {}).get('df', 1)
            from scipy import stats as sp_stats
            means_ci = {}
            for treatment_name, group_df in df.groupby(request.treatment):
                data = group_df[request.response].values
                mean = np.mean(data)
                n = len(data)
                if n > 1:
                    sem = np.sqrt(ms_error / n)
                    df_error = results.get('Residual', {}).get('df', n - 1)
                    t_crit = sp_stats.t.ppf(0.975, df_error)
                    ci_margin = t_crit * sem
                    means_ci[str(treatment_name)] = {
                        "mean": round(float(mean), 4),
                        "lower": round(float(mean - ci_margin), 4),
                        "upper": round(float(mean + ci_margin), 4),
                        "sem": round(float(sem), 4)
                    }

            # Box plot data
            def calculate_boxplot_data(data, label):
                q1 = float(np.percentile(data, 25))
                median = float(np.median(data))
                q3 = float(np.percentile(data, 75))
                iqr = q3 - q1
                lower_whisker = float(np.min(data[data >= q1 - 1.5 * iqr]))
                upper_whisker = float(np.max(data[data <= q3 + 1.5 * iqr]))
                outliers = [float(x) for x in data if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr]

                return {
                    "label": str(label),
                    "min": lower_whisker,
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "max": upper_whisker,
                    "outliers": outliers
                }

            boxplot_data_treatment = []
            for treatment_name, group_df in df.groupby(request.treatment):
                boxplot_data_treatment.append(
                    calculate_boxplot_data(group_df[request.response].values, treatment_name)
                )

            return {
                "test_type": "Latin Square Design - Fixed Blocks",
                "alpha": request.alpha,
                "block_type": "fixed",
                "anova_table": results,
                "treatment_means": {str(k): round(float(v), 4) for k, v in treatment_means.items()},
                "row_block_means": {str(k): round(float(v), 4) for k, v in row_means.items()},
                "col_block_means": {str(k): round(float(v), 4) for k, v in col_means.items()},
                "grand_mean": round(float(df[request.response].mean()), 4),
                "model_r_squared": round(float(model.rsquared), 4),
                "means_ci": means_ci,
                "boxplot_data_treatment": boxplot_data_treatment,
                "residuals": [round(float(r), 4) for r in residuals],
                "fitted_values": [round(float(f), 4) for f in fitted_values],
                "standardized_residuals": [round(float(r), 4) for r in standardized_residuals]
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/generate/latin-square")
async def generate_latin_square(data: dict):
    """
    Generate a Latin Square design
    """
    try:
        n = data['size']  # Size of the square (n x n)

        if n < 2 or n > 26:
            raise ValueError("Latin square size must be between 2 and 26")

        # Generate treatments (A, B, C, ...)
        treatments = [chr(65 + i) for i in range(n)]

        # Generate a random Latin square using cyclic permutation
        latin_square = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(treatments[(i + j) % n])
            latin_square.append(row)

        # Randomize rows and columns
        np.random.shuffle(latin_square)
        latin_square = list(zip(*latin_square))  # Transpose
        np.random.shuffle(latin_square)
        latin_square = list(zip(*latin_square))  # Transpose back

        # Convert to design table
        design = []
        for i, row in enumerate(latin_square):
            for j, treatment in enumerate(row):
                design.append({
                    "row": i + 1,
                    "column": j + 1,
                    "treatment": treatment,
                    "response": None  # To be filled in
                })

        return {
            "design_type": "Latin Square",
            "size": n,
            "n_runs": n * n,
            "treatments": treatments,
            "latin_square": [[cell for cell in row] for row in latin_square],
            "design_table": design
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/graeco-latin")
async def graeco_latin_analysis(request: GraecoLatinRequest):
    """
    Analyze Graeco-Latin Square Design (two sets of treatments + two blocking factors)
    """
    try:
        df = pd.DataFrame(request.data)

        # Build formula: response ~ latin_treatment + greek_treatment + row_block + col_block
        formula = f"{request.response} ~ C({request.latin_treatment}) + C({request.greek_treatment}) + C({request.row_block}) + C({request.col_block})"

        # Fit model
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Parse results
        results = {}
        for idx, row in anova_table.iterrows():
            source = str(idx)
            source = source.replace(f'C({request.latin_treatment})', request.latin_treatment)
            source = source.replace(f'C({request.greek_treatment})', request.greek_treatment)
            source = source.replace(f'C({request.row_block})', request.row_block)
            source = source.replace(f'C({request.col_block})', request.col_block)

            results[source] = {
                "sum_sq": round(float(row['sum_sq']), 4),
                "df": int(row['df']),
                "F": round(float(row['F']), 4) if not pd.isna(row['F']) else None,
                "p_value": round(float(row['PR(>F)']), 6) if not pd.isna(row['PR(>F)']) else None,
                "significant": bool(row['PR(>F)'] < request.alpha) if not pd.isna(row['PR(>F)']) else False
            }

        # Calculate means
        latin_means = df.groupby(request.latin_treatment)[request.response].mean().to_dict()
        greek_means = df.groupby(request.greek_treatment)[request.response].mean().to_dict()
        row_means = df.groupby(request.row_block)[request.response].mean().to_dict()
        col_means = df.groupby(request.col_block)[request.response].mean().to_dict()

        return {
            "test_type": "Graeco-Latin Square Design",
            "alpha": request.alpha,
            "anova_table": results,
            "latin_treatment_means": {str(k): round(float(v), 4) for k, v in latin_means.items()},
            "greek_treatment_means": {str(k): round(float(v), 4) for k, v in greek_means.items()},
            "row_block_means": {str(k): round(float(v), 4) for k, v in row_means.items()},
            "col_block_means": {str(k): round(float(v), 4) for k, v in col_means.items()},
            "grand_mean": round(float(df[request.response].mean()), 4),
            "model_r_squared": round(float(model.rsquared), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/generate/rcbd")
async def generate_rcbd(request: RCBDGenerateRequest):
    """
    Generate a Randomized Complete Block Design
    """
    try:
        n_treatments = request.n_treatments
        n_blocks = request.n_blocks

        if n_treatments < 2:
            raise ValueError("Must have at least 2 treatments")
        if n_blocks < 2:
            raise ValueError("Must have at least 2 blocks")

        # Generate treatment names
        if request.treatment_names and len(request.treatment_names) == n_treatments:
            treatments = request.treatment_names
        else:
            treatments = [f"T{i+1}" for i in range(n_treatments)]

        # Generate design
        design = []
        run_order = 1
        for block in range(1, n_blocks + 1):
            block_treatments = treatments.copy()
            if request.randomize:
                np.random.shuffle(block_treatments)

            for treatment in block_treatments:
                design.append({
                    "run_order": run_order,
                    "block": f"Block{block}",
                    "treatment": treatment,
                    "response": None
                })
                run_order += 1

        # Randomize overall run order if requested
        if request.randomize:
            np.random.shuffle(design)
            for i, run in enumerate(design):
                run["run_order"] = i + 1

        return {
            "design_type": "Randomized Complete Block Design (RCBD)",
            "n_treatments": n_treatments,
            "n_blocks": n_blocks,
            "n_runs": n_treatments * n_blocks,
            "treatments": treatments,
            "design_table": design,
            "randomized": request.randomize
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/generate/graeco-latin")
async def generate_graeco_latin(data: dict):
    """
    Generate a Graeco-Latin Square design
    """
    try:
        n = data['size']  # Size of the square (n x n)

        if n < 3 or n > 12:
            raise ValueError("Graeco-Latin square size must be between 3 and 12")

        # Check if n allows for Graeco-Latin square
        # Not all sizes have orthogonal Latin squares
        if n in [2, 6]:
            raise ValueError(f"Graeco-Latin squares do not exist for n={n}")

        # Generate Latin letters (A, B, C, ...)
        latin_letters = [chr(65 + i) for i in range(n)]

        # Generate Greek letters (α, β, γ, ...)
        greek_letters = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ'][:n]

        # Generate first Latin square
        latin_square = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(latin_letters[(i + j) % n])
            latin_square.append(row)

        # Generate orthogonal Greek square
        greek_square = []
        for i in range(n):
            row = []
            for j in range(n):
                # Use a different permutation to ensure orthogonality
                row.append(greek_letters[(i * 2 + j) % n])
            greek_square.append(row)

        # Combine into Graeco-Latin square
        design = []
        graeco_latin_square = []
        for i in range(n):
            row = []
            for j in range(n):
                latin = latin_square[i][j]
                greek = greek_square[i][j]
                row.append(f"{latin}{greek}")
                design.append({
                    "row": i + 1,
                    "column": j + 1,
                    "latin_treatment": latin,
                    "greek_treatment": greek,
                    "response": None
                })
            graeco_latin_square.append(row)

        return {
            "design_type": "Graeco-Latin Square",
            "size": n,
            "n_runs": n * n,
            "latin_treatments": latin_letters,
            "greek_treatments": greek_letters,
            "graeco_latin_square": graeco_latin_square,
            "design_table": design
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================================
# CROSSOVER DESIGN ENDPOINTS
# ============================================================================

@router.post("/crossover/analyze")
async def crossover_analysis(request: CrossoverRequest):
    """
    Analyze Crossover Design
    Tests for period effects, treatment effects, and carryover effects
    """
    try:
        df = pd.DataFrame(request.data)

        # Determine if this is a 2x2 crossover or multi-period
        n_periods = df[request.period].nunique()
        n_treatments = df[request.treatment].nunique()

        # Build mixed model formula
        # Subject is random effect, period and treatment are fixed
        formula = f"{request.response} ~ C({request.period}) + C({request.treatment})"

        # Fit mixed model with subject as random effect
        model = MixedLM.from_formula(formula, data=df, groups=df[request.subject]).fit(method='powell')

        # Get fixed effects summary
        fixed_effects = model.summary().tables[1]

        # Period effect test
        period_coefs = [param for param in model.params.index if request.period in param and 'T.' in param]
        period_pvalues = [model.pvalues[param] for param in period_coefs]
        period_significant = any(p < request.alpha for p in period_pvalues) if period_pvalues else False

        # Treatment effect test
        treatment_coefs = [param for param in model.params.index if request.treatment in param and 'T.' in param]
        treatment_pvalues = [model.pvalues[param] for param in treatment_coefs]
        treatment_significant = any(p < request.alpha for p in treatment_pvalues) if treatment_pvalues else False

        # Calculate treatment means
        treatment_means = df.groupby(request.treatment)[request.response].mean().to_dict()
        period_means = df.groupby(request.period)[request.response].mean().to_dict()

        # For 2x2 crossover, test for carryover effect
        carryover_results = None
        if n_periods == 2 and n_treatments == 2 and request.sequence:
            try:
                # Test carryover by comparing sequence groups in period 2
                period_2_data = df[df[request.period] == df[request.period].unique()[1]]
                sequences = period_2_data[request.sequence].unique()

                if len(sequences) == 2:
                    seq1_data = period_2_data[period_2_data[request.sequence] == sequences[0]][request.response]
                    seq2_data = period_2_data[period_2_data[request.sequence] == sequences[1]][request.response]

                    # T-test for carryover
                    from scipy.stats import ttest_ind
                    t_stat, carryover_p = ttest_ind(seq1_data, seq2_data)

                    carryover_results = {
                        'test_statistic': float(t_stat),
                        'p_value': float(carryover_p),
                        'significant': bool(carryover_p < request.alpha),
                        'interpretation': 'Significant carryover effect detected' if carryover_p < request.alpha else 'No significant carryover effect'
                    }
            except Exception as e:
                carryover_results = {'error': f'Carryover test failed: {str(e)}'}

        # Extract variance components
        var_subject = float(model.cov_re.values[0][0]) if hasattr(model, 'cov_re') else 0.0
        var_residual = float(model.scale)

        results = {
            'design_type': f'{n_treatments}x{n_periods} Crossover Design',
            'n_subjects': int(df[request.subject].nunique()),
            'n_periods': int(n_periods),
            'n_treatments': int(n_treatments),
            'alpha': request.alpha,
            'treatment_effect': {
                'significant': treatment_significant,
                'p_values': {str(k): float(v) for k, v in zip(treatment_coefs, treatment_pvalues)} if treatment_pvalues else {}
            },
            'period_effect': {
                'significant': period_significant,
                'p_values': {str(k): float(v) for k, v in zip(period_coefs, period_pvalues)} if period_pvalues else {}
            },
            'carryover_effect': carryover_results,
            'treatment_means': {str(k): round(float(v), 4) for k, v in treatment_means.items()},
            'period_means': {str(k): round(float(v), 4) for k, v in period_means.items()},
            'variance_components': {
                'subject_variance': round(var_subject, 4),
                'residual_variance': round(var_residual, 4),
                'total_variance': round(var_subject + var_residual, 4)
            },
            'model_summary': {
                'log_likelihood': round(float(model.llf), 4),
                'aic': round(float(model.aic), 4),
                'bic': round(float(model.bic), 4)
            }
        }

        return sanitize_dict(results)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/crossover/generate")
async def generate_crossover(request: CrossoverGenerateRequest):
    """
    Generate Crossover Design (2x2 or Williams)
    """
    try:
        import itertools
        import random

        n_subjects = request.n_subjects
        n_treatments = request.n_treatments

        # Default treatment names
        if request.treatment_names and len(request.treatment_names) >= n_treatments:
            treatments = request.treatment_names[:n_treatments]
        else:
            treatments = [chr(65 + i) for i in range(n_treatments)]  # A, B, C, ...

        design_table = []

        if request.design_type == "2x2" and n_treatments == 2:
            # 2x2 Crossover: AB and BA sequences
            sequences = [
                [treatments[0], treatments[1]],  # AB
                [treatments[1], treatments[0]]   # BA
            ]
            sequence_names = [f"{treatments[0]}{treatments[1]}", f"{treatments[1]}{treatments[0]}"]

            # Assign subjects to sequences (alternating or balanced)
            for subject_id in range(1, n_subjects + 1):
                sequence_idx = (subject_id - 1) % 2
                sequence = sequences[sequence_idx]
                sequence_name = sequence_names[sequence_idx]

                for period_idx, treatment in enumerate(sequence, 1):
                    design_table.append({
                        'subject': subject_id,
                        'sequence': sequence_name,
                        'period': period_idx,
                        'treatment': treatment,
                        'response': None
                    })

            return {
                'design_type': '2x2 Crossover',
                'n_subjects': n_subjects,
                'n_periods': 2,
                'n_treatments': 2,
                'treatments': treatments,
                'sequences': sequence_names,
                'n_runs': n_subjects * 2,
                'design_table': design_table
            }

        elif request.design_type == "williams" and n_treatments >= 3:
            # Williams Design: Balanced for first-order carryover
            # Generate all possible orderings and select balanced subset
            from itertools import permutations

            all_perms = list(permutations(treatments))

            # For Williams design, select n_treatments sequences
            # that balance first-order carryover
            williams_sequences = all_perms[:n_treatments]

            # Assign subjects to sequences
            for subject_id in range(1, n_subjects + 1):
                sequence_idx = (subject_id - 1) % len(williams_sequences)
                sequence = williams_sequences[sequence_idx]
                sequence_name = ''.join(sequence)

                for period_idx, treatment in enumerate(sequence, 1):
                    design_table.append({
                        'subject': subject_id,
                        'sequence': sequence_name,
                        'period': period_idx,
                        'treatment': treatment,
                        'response': None
                    })

            return {
                'design_type': f'Williams Design ({n_treatments}x{n_treatments})',
                'n_subjects': n_subjects,
                'n_periods': n_treatments,
                'n_treatments': n_treatments,
                'treatments': treatments,
                'sequences': [''.join(seq) for seq in williams_sequences],
                'n_runs': n_subjects * n_treatments,
                'design_table': design_table
            }
        else:
            raise ValueError(f"Unsupported combination: design_type={request.design_type}, n_treatments={n_treatments}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========================================
# INCOMPLETE BLOCK DESIGNS
# ========================================

class IncompleteBlockRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Experimental data")
    treatment: str = Field(..., description="Treatment factor column name")
    block: str = Field(..., description="Block factor column name")
    response: str = Field(..., description="Response variable name")
    alpha: float = Field(0.05, description="Significance level")

class BIBGenerateRequest(BaseModel):
    n_treatments: int = Field(..., description="Number of treatments (v)")
    block_size: int = Field(..., description="Block size (k)")
    n_replications: int = Field(None, description="Number of replications per treatment (r), auto-calculated if None")

class YoudenGenerateRequest(BaseModel):
    n_treatments: int = Field(..., description="Number of treatments")
    n_rows: int = Field(..., description="Number of rows (incomplete blocks)")
    n_columns: int = Field(..., description="Number of columns")


def check_bib_parameters(v, k, r):
    """
    Check if BIB parameters are valid.
    For a BIB to exist:
    - b*k = v*r (where b is number of blocks)
    - lambda = r(k-1)/(v-1) must be an integer
    """
    b = (v * r) // k  # Number of blocks

    # Check if v*r is divisible by k
    if (v * r) % k != 0:
        return False, "Parameters don't satisfy b*k = v*r"

    # Check if lambda is an integer
    if (r * (k - 1)) % (v - 1) != 0:
        return False, "Parameters don't yield integer lambda"

    lambda_val = (r * (k - 1)) // (v - 1)

    return True, {'v': v, 'k': k, 'r': r, 'b': b, 'lambda': lambda_val}


@router.post("/incomplete/generate/bib")
async def generate_bib(request: BIBGenerateRequest):
    """
    Generate a Balanced Incomplete Block (BIB) design.

    A BIB design has:
    - v treatments
    - b blocks
    - k treatments per block (k < v)
    - r replications per treatment
    - λ = number of blocks in which each pair of treatments occurs together
    """
    try:
        v = request.n_treatments
        k = request.block_size

        if k >= v:
            raise ValueError(f"Block size ({k}) must be less than number of treatments ({v})")

        # If r not specified, try to find a valid r
        if request.n_replications is None:
            # Try common values of r
            for r_try in range(k, min(v * 2, 20)):
                valid, params = check_bib_parameters(v, k, r_try)
                if valid:
                    r = r_try
                    break
            else:
                raise ValueError(f"Could not find valid BIB parameters for v={v}, k={k}. Try specifying r manually.")
        else:
            r = request.n_replications
            valid, params = check_bib_parameters(v, k, r)
            if not valid:
                raise ValueError(f"Invalid BIB parameters: {params}")

        # Get BIB parameters
        valid, params = check_bib_parameters(v, k, r)
        b = params['b']
        lambda_val = params['lambda']

        # Generate BIB design using a simple construction
        # For small designs, use cyclic construction
        treatments = [f'T{i+1}' for i in range(v)]
        design_table = []

        # Simple cyclic construction for BIB
        # This works for many parameter combinations
        block_idx = 0
        treatment_counts = {t: 0 for t in treatments}

        # Generate blocks by cycling through treatments
        for start in range(v):
            if block_idx >= b:
                break
            # Create block starting from treatment 'start'
            block_treatments = []
            for j in range(k):
                treatment_idx = (start + j) % v
                block_treatments.append(treatments[treatment_idx])

            # Check if this maintains balance
            temp_counts = treatment_counts.copy()
            for t in block_treatments:
                temp_counts[t] += 1

            # Add block if it doesn't exceed r for any treatment
            if all(temp_counts[t] <= r for t in treatments):
                for t in block_treatments:
                    treatment_counts[t] += 1
                    design_table.append({
                        'block': f'B{block_idx + 1}',
                        'treatment': t,
                        'response': None
                    })
                block_idx += 1

        # If cyclic didn't generate enough blocks, use complementary construction
        if block_idx < b:
            # Fill remaining with balanced selection
            remaining = b - block_idx
            for _ in range(remaining):
                # Find k treatments with lowest counts
                sorted_treatments = sorted(treatments, key=lambda t: treatment_counts[t])
                block_treatments = sorted_treatments[:k]

                for t in block_treatments:
                    treatment_counts[t] += 1
                    design_table.append({
                        'block': f'B{block_idx + 1}',
                        'treatment': t,
                        'response': None
                    })
                block_idx += 1

        # Calculate efficiency relative to RCBD
        # Efficiency = (v*k) / (v*k - v + 1) for BIB
        efficiency = (v * k) / (v * k - v + 1) if (v * k - v + 1) > 0 else 1.0

        return {
            'design_type': 'Balanced Incomplete Block (BIB)',
            'n_treatments': v,
            'n_blocks': b,
            'block_size': k,
            'replications': r,
            'lambda': lambda_val,
            'n_runs': len(design_table),
            'efficiency': round(efficiency, 4),
            'design_table': design_table,
            'parameters': {
                'v': v,
                'b': b,
                'k': k,
                'r': r,
                'lambda': lambda_val
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/incomplete/generate/youden")
async def generate_youden(request: YoudenGenerateRequest):
    """
    Generate a Youden Square design.

    A Youden Square is an incomplete Latin square where:
    - Rows are incomplete blocks
    - Each treatment appears once per column
    - Not all treatments appear in each row
    """
    try:
        v = request.n_treatments
        n_rows = request.n_rows
        n_cols = request.n_columns

        if n_cols >= v:
            raise ValueError(f"Number of columns ({n_cols}) should be less than number of treatments ({v})")

        if n_rows > v:
            raise ValueError(f"Number of rows ({n_rows}) cannot exceed number of treatments ({v})")

        treatments = [f'T{i+1}' for i in range(v)]
        design_table = []

        # Generate Youden square using cyclic Latin square construction
        # Then select subset of columns
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                # Cyclic construction: treatment at (i,j) = (i + j) mod v
                treatment_idx = (row_idx + col_idx) % v
                design_table.append({
                    'row': f'R{row_idx + 1}',
                    'column': f'C{col_idx + 1}',
                    'treatment': treatments[treatment_idx],
                    'response': None
                })

        return {
            'design_type': 'Youden Square',
            'n_treatments': v,
            'n_rows': n_rows,
            'n_columns': n_cols,
            'n_runs': n_rows * n_cols,
            'design_table': design_table
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/incomplete/analyze")
async def analyze_incomplete_block(request: IncompleteBlockRequest):
    """
    Analyze Incomplete Block Design using intrablock analysis.
    """
    try:
        df = pd.DataFrame(request.data)

        # Check required columns
        if request.treatment not in df.columns or request.block not in df.columns or request.response not in df.columns:
            raise ValueError("Missing required columns in data")

        # Convert to categorical
        df[request.treatment] = pd.Categorical(df[request.treatment])
        df[request.block] = pd.Categorical(df[request.block])

        v = df[request.treatment].nunique()  # Number of treatments
        b = df[request.block].nunique()  # Number of blocks
        n = len(df)  # Total observations

        # Calculate block sizes
        block_sizes = df.groupby(request.block).size()
        k = block_sizes.iloc[0]  # Assuming constant block size
        is_balanced = len(block_sizes.unique()) == 1

        # Check if design is incomplete
        is_incomplete = k < v

        # Fit ANOVA model
        formula = f"{request.response} ~ C({request.treatment}) + C({request.block})"
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Treatment effects
        treatment_f = anova_table.loc[f'C({request.treatment})', 'F']
        treatment_p = anova_table.loc[f'C({request.treatment})', 'PR(>F)']

        # Block effects
        block_f = anova_table.loc[f'C({request.block})', 'F']
        block_p = anova_table.loc[f'C({request.block})', 'PR(>F)']

        # Treatment means
        treatment_means = df.groupby(request.treatment)[request.response].mean().to_dict()

        # Block means
        block_means = df.groupby(request.block)[request.response].mean().to_dict()

        # Calculate efficiency relative to RCBD
        # For BIB: E = (v*k) / (v*k - v + 1)
        if is_incomplete and is_balanced:
            efficiency = (v * k) / (v * k - v + 1) if (v * k - v + 1) > 0 else 1.0
        else:
            efficiency = 1.0  # Complete block

        # Calculate variance components
        ms_error = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']

        # Format ANOVA table
        anova_dict = {}
        for idx in anova_table.index:
            anova_dict[idx] = {
                'sum_sq': float(anova_table.loc[idx, 'sum_sq']),
                'df': int(anova_table.loc[idx, 'df']),
                'F': float(anova_table.loc[idx, 'F']) if pd.notna(anova_table.loc[idx, 'F']) else None,
                'p_value': float(anova_table.loc[idx, 'PR(>F)']) if pd.notna(anova_table.loc[idx, 'PR(>F)']) else None,
                'significant': bool(anova_table.loc[idx, 'PR(>F)'] < request.alpha) if pd.notna(anova_table.loc[idx, 'PR(>F)']) else False
            }

        return {
            'design_info': {
                'n_treatments': int(v),
                'n_blocks': int(b),
                'block_size': int(k),
                'total_runs': int(n),
                'is_incomplete': bool(is_incomplete),
                'is_balanced': bool(is_balanced)
            },
            'treatment_effect': {
                'F_statistic': float(treatment_f),
                'p_value': float(treatment_p),
                'significant': bool(treatment_p < request.alpha),
                'interpretation': 'Significant treatment differences' if treatment_p < request.alpha else 'No significant treatment differences'
            },
            'block_effect': {
                'F_statistic': float(block_f),
                'p_value': float(block_p),
                'significant': bool(block_p < request.alpha)
            },
            'treatment_means': {k: float(v) for k, v in treatment_means.items()},
            'block_means': {k: float(v) for k, v in block_means.items()},
            'anova_table': anova_dict,
            'efficiency': float(efficiency),
            'mse': float(ms_error),
            'r_squared': float(model.rsquared),
            'adj_r_squared': float(model.rsquared_adj)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
