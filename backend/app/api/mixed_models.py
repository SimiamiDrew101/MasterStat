from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm, ols
from scipy import stats
import itertools

router = APIRouter()

# ============================================================================
# MIXED MODEL ANOVA WITH EXPECTED MEAN SQUARES
# ============================================================================

class MixedModelANOVARequest(BaseModel):
    """Request for Mixed Model ANOVA analysis"""
    data: List[Dict] = Field(..., description="Experimental data")
    fixed_factors: List[str] = Field(..., description="Fixed effect factor names")
    random_factors: List[str] = Field(..., description="Random effect factor names")
    response: str = Field(..., description="Response variable name")
    alpha: float = Field(0.05, description="Significance level")
    include_interactions: bool = Field(True, description="Include factor interactions")


@router.post("/mixed-model-anova")
async def mixed_model_anova(request: MixedModelANOVARequest):
    """
    Comprehensive Mixed Model ANOVA with:
    - Expected Mean Squares (EMS)
    - Variance component estimation
    - Proper F-tests with correct error terms
    - Confidence intervals for variance components
    """
    try:
        df = pd.DataFrame(request.data)

        # Validate data
        all_factors = request.fixed_factors + request.random_factors
        required_cols = all_factors + [request.response]

        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in data")

        # Convert factors to categorical
        for factor in all_factors:
            df[factor] = df[factor].astype('category')

        # Build ANOVA model
        if request.include_interactions and len(all_factors) == 2:
            # Two-factor model with interaction
            formula = f"{request.response} ~ C({all_factors[0]}) * C({all_factors[1]})"
        elif len(all_factors) == 1:
            # One-factor model
            formula = f"{request.response} ~ C({all_factors[0]})"
        else:
            # Multiple factors without interaction (additive model)
            formula = f"{request.response} ~ " + " + ".join([f"C({f})" for f in all_factors])

        # Fit model using OLS
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=1)  # Type I SS

        # Build ANOVA results table
        anova_results = {}
        ms_dict = {}  # Store MS values for variance component estimation
        df_dict = {}  # Store df values

        for idx, row in anova_table.iterrows():
            source = str(idx)

            # Clean up source names
            for factor in all_factors:
                source = source.replace(f'C({factor})', factor)

            # Calculate Mean Square
            ms = row['sum_sq'] / row['df'] if row['df'] > 0 else 0
            ms_dict[source] = ms
            df_dict[source] = int(row['df'])

            anova_results[source] = {
                "sum_sq": round(float(row['sum_sq']), 4),
                "df": int(row['df']),
                "mean_sq": round(float(ms), 4),
                "F": round(float(row['F']), 4) if not pd.isna(row['F']) else None,
                "p_value": round(float(row['PR(>F)']), 6) if not pd.isna(row['PR(>F)']) else None
            }

        # Calculate Expected Mean Squares (EMS)
        ems_dict = calculate_ems(
            request.fixed_factors,
            request.random_factors,
            df,
            request.include_interactions and len(all_factors) == 2
        )

        # Add EMS to ANOVA results
        for source in anova_results:
            if source in ems_dict:
                anova_results[source]["ems"] = ems_dict[source]

        # Recalculate F-tests with correct error terms for mixed models
        f_tests = recalculate_f_tests(
            ms_dict,
            df_dict,
            request.fixed_factors,
            request.random_factors,
            request.include_interactions and len(all_factors) == 2
        )

        # Update F and p-values
        for source, f_info in f_tests.items():
            if source in anova_results:
                anova_results[source]["F_corrected"] = round(f_info["F"], 4)
                anova_results[source]["p_value_corrected"] = round(f_info["p_value"], 6)
                anova_results[source]["error_term"] = f_info["error_term"]

        # Estimate variance components
        variance_components = estimate_variance_components(
            ms_dict,
            df,
            request.fixed_factors,
            request.random_factors,
            request.include_interactions and len(all_factors) == 2
        )

        # Calculate percentage of variance
        total_var = sum([v for v in variance_components.values() if v is not None and v > 0])
        variance_percentages = {}
        if total_var > 0:
            for component, var in variance_components.items():
                if var is not None and var > 0:
                    variance_percentages[component] = round((var / total_var) * 100, 2)

        # Calculate plot data
        plot_data = calculate_plot_data_mixed_anova(
            df,
            model,
            all_factors,
            request.response,
            request.include_interactions and len(all_factors) == 2
        )

        # Calculate ICC for random factors
        icc_results = {}
        for random_factor in request.random_factors:
            icc_results[random_factor] = calculate_icc(
                df,
                random_factor,
                request.response,
                icc_type="icc2"
            )

        # Enhanced model fit metrics
        n_params = len(all_factors) + (1 if request.include_interactions and len(all_factors) == 2 else 0)
        model_fit = calculate_model_fit_metrics(model, df, n_params)

        return {
            "model_type": "Mixed Model ANOVA",
            "fixed_factors": request.fixed_factors,
            "random_factors": request.random_factors,
            "include_interactions": request.include_interactions,
            "anova_table": anova_results,
            "variance_components": {
                k: round(v, 6) if v is not None else None
                for k, v in variance_components.items()
            },
            "variance_percentages": variance_percentages,
            "icc": icc_results,
            "model_summary": {
                "r_squared": round(float(model.rsquared), 4),
                "adj_r_squared": round(float(model.rsquared_adj), 4),
                "f_statistic": round(float(model.fvalue), 4),
                "aic": round(float(model.aic), 4),
                "bic": round(float(model.bic), 4)
            },
            "model_fit": model_fit,
            "blups": extract_blups(model, df, request.random_factors, request.response),
            "plot_data": plot_data,
            "interpretation": generate_interpretation(
                anova_results,
                variance_components,
                request.fixed_factors,
                request.random_factors,
                request.alpha
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in analysis: {str(e)}")


def calculate_plot_data_mixed_anova(df: pd.DataFrame, model, all_factors: List[str],
                                     response: str, has_interaction: bool) -> Dict:
    """
    Calculate all data needed for visualizations

    Returns:
    - cell_means: Mean response for each factor combination
    - marginal_means: Mean response for each factor level
    - fitted_values: Model predictions
    - residuals: Model residuals
    - factor_levels: Unique levels for each factor
    - box_plot_data: Raw data grouped by factors
    """
    def safe_float(value):
        """Convert value to float, handling NaN/inf by returning None"""
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)

    plot_data = {}

    # Extract fitted values and residuals from model
    plot_data["fitted_values"] = [safe_float(v) for v in model.fittedvalues.tolist()]
    plot_data["residuals"] = [safe_float(v) for v in model.resid.tolist()]

    # Get factor levels
    plot_data["factor_levels"] = {}
    for factor in all_factors:
        plot_data["factor_levels"][factor] = sorted(df[factor].unique().tolist())

    # Calculate marginal means for main effects plots
    plot_data["marginal_means"] = {}
    for factor in all_factors:
        marginal_data = []
        for level in sorted(df[factor].unique()):
            level_data = df[df[factor] == level][response]
            marginal_data.append({
                "level": str(level),
                "mean": safe_float(level_data.mean()),
                "std": safe_float(level_data.std()),
                "n": int(len(level_data))
            })
        plot_data["marginal_means"][factor] = marginal_data

    # Calculate cell means for interaction plots (if two factors)
    if len(all_factors) == 2 and has_interaction:
        factor1, factor2 = all_factors
        cell_means = []

        for level1 in sorted(df[factor1].unique()):
            for level2 in sorted(df[factor2].unique()):
                cell_data = df[(df[factor1] == level1) & (df[factor2] == level2)][response]
                if len(cell_data) > 0:
                    cell_means.append({
                        factor1: str(level1),
                        factor2: str(level2),
                        "mean": safe_float(cell_data.mean()),
                        "std": safe_float(cell_data.std()),
                        "n": int(len(cell_data))
                    })

        plot_data["cell_means"] = cell_means

    # Prepare box plot data
    plot_data["box_plot_data"] = {}
    for factor in all_factors:
        box_data = []
        for level in sorted(df[factor].unique()):
            level_values = df[df[factor] == level][response].tolist()
            box_data.append({
                "level": str(level),
                "values": [safe_float(v) for v in level_values]
            })
        plot_data["box_plot_data"][factor] = box_data

    return plot_data


def calculate_ems(fixed_factors: List[str], random_factors: List[str],
                  df: pd.DataFrame, has_interaction: bool) -> Dict[str, str]:
    """
    Calculate Expected Mean Squares (EMS) for mixed model

    EMS describes what each mean square estimates:
    - For fixed effects: σ² + nσ²(effect)
    - For random effects: σ² + nσ²(effect)
    - Interactions follow specific rules
    """
    ems = {}

    if len(fixed_factors) == 1 and len(random_factors) == 0:
        # One-way fixed ANOVA
        fixed_factor = fixed_factors[0]
        n = len(df) // df[fixed_factor].nunique()
        ems[fixed_factor] = f"σ² + {n}σ²({fixed_factor})"
        ems["Residual"] = "σ²"

    elif len(fixed_factors) == 0 and len(random_factors) == 1:
        # One-way random ANOVA
        random_factor = random_factors[0]
        n = len(df) // df[random_factor].nunique()
        ems[random_factor] = f"σ² + {n}σ²({random_factor})"
        ems["Residual"] = "σ²"

    elif len(fixed_factors) == 1 and len(random_factors) == 1:
        # Two-factor mixed model (most common case)
        fixed = fixed_factors[0]
        random = random_factors[0]

        # Calculate sample sizes
        n_fixed = df[fixed].nunique()
        n_random = df[random].nunique()
        n_per_cell = len(df) // (n_fixed * n_random)

        if has_interaction:
            # With interaction
            ems[fixed] = f"σ² + {n_per_cell}σ²({fixed}×{random}) + {n_random * n_per_cell}σ²({fixed})"
            ems[random] = f"σ² + {n_per_cell}σ²({fixed}×{random}) + {n_fixed * n_per_cell}σ²({random})"
            ems[f"{fixed}:{random}"] = f"σ² + {n_per_cell}σ²({fixed}×{random})"
            ems["Residual"] = "σ²"
        else:
            # Additive model (no interaction)
            ems[fixed] = f"σ² + {n_random * n_per_cell}σ²({fixed})"
            ems[random] = f"σ² + {n_fixed * n_per_cell}σ²({random})"
            ems["Residual"] = "σ²"

    return ems


def recalculate_f_tests(ms_dict: Dict, df_dict: Dict, fixed_factors: List[str],
                        random_factors: List[str], has_interaction: bool) -> Dict:
    """
    Recalculate F-tests using appropriate error terms for mixed models

    In mixed models, the denominator MS depends on the model structure:
    - Fixed effects may use interaction MS as error term
    - Random effects use residual MS
    """
    f_tests = {}

    if len(fixed_factors) == 1 and len(random_factors) == 1 and has_interaction:
        # Mixed model with interaction
        fixed = fixed_factors[0]
        random = random_factors[0]
        interaction_key = f"{fixed}:{random}"

        # Fixed effect: test against interaction
        if fixed in ms_dict and interaction_key in ms_dict:
            if ms_dict[interaction_key] > 0:
                F = ms_dict[fixed] / ms_dict[interaction_key]
                df1 = df_dict[fixed]
                df2 = df_dict[interaction_key]
                p_value = 1 - stats.f.cdf(F, df1, df2)
                f_tests[fixed] = {
                    "F": F,
                    "p_value": p_value,
                    "error_term": interaction_key,
                    "df1": df1,
                    "df2": df2
                }

        # Random effect: test against interaction
        if random in ms_dict and interaction_key in ms_dict:
            if ms_dict[interaction_key] > 0:
                F = ms_dict[random] / ms_dict[interaction_key]
                df1 = df_dict[random]
                df2 = df_dict[interaction_key]
                p_value = 1 - stats.f.cdf(F, df1, df2)
                f_tests[random] = {
                    "F": F,
                    "p_value": p_value,
                    "error_term": interaction_key,
                    "df1": df1,
                    "df2": df2
                }

        # Interaction: test against residual (already correct from OLS)
        if interaction_key in ms_dict and "Residual" in ms_dict:
            if ms_dict["Residual"] > 0:
                F = ms_dict[interaction_key] / ms_dict["Residual"]
                df1 = df_dict[interaction_key]
                df2 = df_dict["Residual"]
                p_value = 1 - stats.f.cdf(F, df1, df2)
                f_tests[interaction_key] = {
                    "F": F,
                    "p_value": p_value,
                    "error_term": "Residual",
                    "df1": df1,
                    "df2": df2
                }

    return f_tests


def estimate_variance_components(ms_dict: Dict, df: pd.DataFrame,
                                 fixed_factors: List[str], random_factors: List[str],
                                 has_interaction: bool) -> Dict[str, float]:
    """
    Estimate variance components using ANOVA method

    Variance components represent the variability contributed by each source:
    - σ²: residual variance (within-cell)
    - σ²(random): between-level variance for random factors
    - σ²(interaction): interaction variance
    """
    variance_components = {}

    # Get residual MS (always σ²)
    ms_error = ms_dict.get("Residual", 0)
    variance_components["σ²_error"] = max(ms_error, 0)

    if len(fixed_factors) == 1 and len(random_factors) == 1:
        fixed = fixed_factors[0]
        random = random_factors[0]

        # Calculate sample sizes
        n_fixed = df[fixed].nunique()
        n_random = df[random].nunique()
        n_per_cell = len(df) // (n_fixed * n_random)

        if has_interaction:
            interaction_key = f"{fixed}:{random}"

            # Interaction variance component
            ms_interaction = ms_dict.get(interaction_key, 0)
            sigma_sq_interaction = max((ms_interaction - ms_error) / n_per_cell, 0)
            variance_components[f"σ²_{fixed}×{random}"] = sigma_sq_interaction

            # Random factor variance component
            ms_random = ms_dict.get(random, 0)
            sigma_sq_random = max((ms_random - ms_interaction) / (n_fixed * n_per_cell), 0)
            variance_components[f"σ²_{random}"] = sigma_sq_random

        else:
            # Additive model
            ms_random = ms_dict.get(random, 0)
            sigma_sq_random = max((ms_random - ms_error) / (n_fixed * n_per_cell), 0)
            variance_components[f"σ²_{random}"] = sigma_sq_random

    elif len(random_factors) == 1 and len(fixed_factors) == 0:
        # One-way random
        random = random_factors[0]
        n = len(df) // df[random].nunique()
        ms_random = ms_dict.get(random, 0)
        sigma_sq_random = max((ms_random - ms_error) / n, 0)
        variance_components[f"σ²_{random}"] = sigma_sq_random

    return variance_components


def generate_interpretation(anova_results: Dict, variance_components: Dict,
                           fixed_factors: List[str], random_factors: List[str],
                           alpha: float) -> List[str]:
    """Generate interpretation of results"""
    interpretations = []

    # Interpret significance of effects
    for source, results in anova_results.items():
        if source != "Residual":
            p_value = results.get("p_value_corrected") or results.get("p_value")
            if p_value is not None:
                if p_value < alpha:
                    interpretations.append(
                        f"{source}: Statistically significant (p = {p_value:.4f})"
                    )
                else:
                    interpretations.append(
                        f"{source}: Not statistically significant (p = {p_value:.4f})"
                    )

    # Interpret variance components
    total_var = sum([v for v in variance_components.values() if v > 0])
    if total_var > 0:
        interpretations.append("\nVariance Component Breakdown:")
        for component, var in variance_components.items():
            if var > 0:
                pct = (var / total_var) * 100
                interpretations.append(
                    f"{component}: {var:.6f} ({pct:.1f}% of total variance)"
                )

    return interpretations


# ============================================================================
# SPLIT-PLOT DESIGN
# ============================================================================

class SplitPlotRequest(BaseModel):
    data: List[Dict] = Field(..., description="Experimental data")
    whole_plot_factor: str = Field(..., description="Whole-plot factor")
    subplot_factor: str = Field(..., description="Sub-plot factor")
    block: Optional[str] = Field(None, description="Block/replicate factor")
    response: str = Field(..., description="Response variable")
    alpha: float = Field(0.05, description="Significance level")


@router.post("/split-plot")
async def split_plot_analysis(request: SplitPlotRequest):
    """
    Analyze split-plot design with proper error terms

    Split-plot designs have TWO error terms:
    - Whole-plot error (Error A): for testing whole-plot factor
    - Sub-plot error (Error B): for testing sub-plot factor and interaction

    Structure:
    - Whole plots are grouped (e.g., by blocks/replicates)
    - Whole-plot factor is applied to entire plots
    - Sub-plot factor is applied within whole plots
    """
    try:
        df = pd.DataFrame(request.data)

        # Validate columns
        required_cols = [request.whole_plot_factor, request.subplot_factor, request.response]
        if request.block:
            required_cols.append(request.block)

        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in data")

        # Convert factors to categorical
        df[request.whole_plot_factor] = df[request.whole_plot_factor].astype('category')
        df[request.subplot_factor] = df[request.subplot_factor].astype('category')
        if request.block:
            df[request.block] = df[request.block].astype('category')

        # Get factor levels
        wp_levels = df[request.whole_plot_factor].nunique()
        sp_levels = df[request.subplot_factor].nunique()
        n_blocks = df[request.block].nunique() if request.block else 1
        n_per_cell = len(df) // (wp_levels * sp_levels * n_blocks)

        # Build formula
        if request.block:
            # Randomized Complete Block Design (RCBD) at whole-plot level
            formula = f"{request.response} ~ C({request.block}) + C({request.whole_plot_factor}) + "
            formula += f"C({request.whole_plot_factor}):C({request.block}) + "  # Whole-plot error
            formula += f"C({request.subplot_factor}) + C({request.whole_plot_factor}):C({request.subplot_factor})"
        else:
            # Completely Randomized Design (CRD) at whole-plot level
            formula = f"{request.response} ~ C({request.whole_plot_factor}) + C({request.subplot_factor}) + "
            formula += f"C({request.whole_plot_factor}):C({request.subplot_factor})"

        # Fit model
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=1)

        # Parse ANOVA table
        anova_results = {}
        ms_dict = {}
        df_dict = {}

        for idx, row in anova_table.iterrows():
            source = str(idx)

            # Clean up source names
            source = source.replace(f'C({request.whole_plot_factor})', request.whole_plot_factor)
            source = source.replace(f'C({request.subplot_factor})', request.subplot_factor)
            if request.block:
                source = source.replace(f'C({request.block})', request.block)

            ms = row['sum_sq'] / row['df'] if row['df'] > 0 else 0
            ms_dict[source] = ms
            df_dict[source] = int(row['df'])

            anova_results[source] = {
                "sum_sq": round(float(row['sum_sq']), 4),
                "df": int(row['df']),
                "mean_sq": round(float(ms), 4),
                "F": round(float(row['F']), 4) if not pd.isna(row['F']) else None,
                "p_value": round(float(row['PR(>F)']), 6) if not pd.isna(row['PR(>F)']) else None
            }

        # Identify error terms
        wp_error_key = f"{request.whole_plot_factor}:{request.block}" if request.block else None
        sp_error_key = "Residual"

        # Calculate Expected Mean Squares for split-plot
        ems_dict = calculate_split_plot_ems(
            request.whole_plot_factor,
            request.subplot_factor,
            request.block,
            wp_levels,
            sp_levels,
            n_blocks,
            n_per_cell
        )

        # Add EMS to results
        for source in anova_results:
            if source in ems_dict:
                anova_results[source]["ems"] = ems_dict[source]

        # Recalculate F-tests with proper error terms
        f_tests_corrected = {}

        # Whole-plot factor: test against whole-plot error
        if request.whole_plot_factor in ms_dict:
            if wp_error_key and wp_error_key in ms_dict and ms_dict[wp_error_key] > 0:
                # With blocks: use WP×Block as error
                F = ms_dict[request.whole_plot_factor] / ms_dict[wp_error_key]
                df1 = df_dict[request.whole_plot_factor]
                df2 = df_dict[wp_error_key]
                p_value = 1 - stats.f.cdf(F, df1, df2)
                f_tests_corrected[request.whole_plot_factor] = {
                    "F": F,
                    "p_value": p_value,
                    "error_term": wp_error_key,
                    "df1": df1,
                    "df2": df2
                }
            elif sp_error_key in ms_dict and ms_dict[sp_error_key] > 0:
                # Without blocks: use residual
                F = ms_dict[request.whole_plot_factor] / ms_dict[sp_error_key]
                df1 = df_dict[request.whole_plot_factor]
                df2 = df_dict[sp_error_key]
                p_value = 1 - stats.f.cdf(F, df1, df2)
                f_tests_corrected[request.whole_plot_factor] = {
                    "F": F,
                    "p_value": p_value,
                    "error_term": sp_error_key,
                    "df1": df1,
                    "df2": df2
                }

        # Sub-plot factor: test against sub-plot error (residual)
        if request.subplot_factor in ms_dict and sp_error_key in ms_dict:
            if ms_dict[sp_error_key] > 0:
                F = ms_dict[request.subplot_factor] / ms_dict[sp_error_key]
                df1 = df_dict[request.subplot_factor]
                df2 = df_dict[sp_error_key]
                p_value = 1 - stats.f.cdf(F, df1, df2)
                f_tests_corrected[request.subplot_factor] = {
                    "F": F,
                    "p_value": p_value,
                    "error_term": sp_error_key,
                    "df1": df1,
                    "df2": df2
                }

        # Interaction: test against sub-plot error (residual)
        interaction_key = f"{request.whole_plot_factor}:{request.subplot_factor}"
        if interaction_key in ms_dict and sp_error_key in ms_dict:
            if ms_dict[sp_error_key] > 0:
                F = ms_dict[interaction_key] / ms_dict[sp_error_key]
                df1 = df_dict[interaction_key]
                df2 = df_dict[sp_error_key]
                p_value = 1 - stats.f.cdf(F, df1, df2)
                f_tests_corrected[interaction_key] = {
                    "F": F,
                    "p_value": p_value,
                    "error_term": sp_error_key,
                    "df1": df1,
                    "df2": df2
                }

        # Block effect (if present): test against whole-plot error
        if request.block and request.block in ms_dict:
            if wp_error_key and wp_error_key in ms_dict and ms_dict[wp_error_key] > 0:
                F = ms_dict[request.block] / ms_dict[wp_error_key]
                df1 = df_dict[request.block]
                df2 = df_dict[wp_error_key]
                p_value = 1 - stats.f.cdf(F, df1, df2)
                f_tests_corrected[request.block] = {
                    "F": F,
                    "p_value": p_value,
                    "error_term": wp_error_key,
                    "df1": df1,
                    "df2": df2
                }

        # Update with corrected F-tests
        for source, f_info in f_tests_corrected.items():
            if source in anova_results:
                anova_results[source]["F_corrected"] = round(f_info["F"], 4)
                anova_results[source]["p_value_corrected"] = round(f_info["p_value"], 6)
                anova_results[source]["error_term"] = f_info["error_term"]

        # Label error terms explicitly
        if wp_error_key and wp_error_key in anova_results:
            anova_results[wp_error_key]["label"] = "Error (Whole-plot)"
        if sp_error_key in anova_results:
            anova_results[sp_error_key]["label"] = "Error (Sub-plot)"

        # Estimate variance components for split-plot
        variance_components = estimate_split_plot_variance_components(
            ms_dict,
            request.whole_plot_factor,
            request.subplot_factor,
            request.block,
            wp_levels,
            sp_levels,
            n_blocks,
            n_per_cell
        )

        # Calculate variance percentages
        total_var = sum([v for v in variance_components.values() if v is not None and v > 0])
        variance_percentages = {}
        if total_var > 0:
            for component, var in variance_components.items():
                if var is not None and var > 0:
                    variance_percentages[component] = round((var / total_var) * 100, 2)

        # Calculate plot data
        plot_data = calculate_plot_data_split_plot(
            df,
            model,
            request.whole_plot_factor,
            request.subplot_factor,
            request.response,
            request.block
        )

        # Calculate ICC for blocks (random effect)
        icc_results = {}
        if request.block:
            icc_results[request.block] = calculate_icc(
                df,
                request.block,
                request.response,
                icc_type="icc2"
            )

        # Enhanced model fit metrics
        n_params = 3 + (1 if request.block else 0)  # whole-plot + subplot + interaction + block
        model_fit = calculate_model_fit_metrics(model, df, n_params)

        # Handle NaN/inf values in model summary
        def safe_float(value):
            """Convert float to rounded value, handling NaN/inf"""
            if pd.isna(value) or np.isinf(value):
                return None
            return round(float(value), 4)

        return {
            "model_type": "Split-Plot Design",
            "whole_plot_factor": request.whole_plot_factor,
            "subplot_factor": request.subplot_factor,
            "block": request.block,
            "anova_table": anova_results,
            "variance_components": {
                k: round(v, 6) if v is not None and not pd.isna(v) else None
                for k, v in variance_components.items()
            },
            "variance_percentages": variance_percentages,
            "icc": icc_results,
            "model_summary": {
                "r_squared": safe_float(model.rsquared),
                "adj_r_squared": safe_float(model.rsquared_adj),
                "f_statistic": safe_float(model.fvalue),
                "aic": safe_float(model.aic),
                "bic": safe_float(model.bic)
            },
            "model_fit": model_fit,
            "blups": extract_blups(model, df, [request.whole_plot_factor], request.response),
            "plot_data": plot_data,
            "interpretation": generate_split_plot_interpretation(
                anova_results,
                variance_components,
                request.whole_plot_factor,
                request.subplot_factor,
                request.alpha
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in split-plot analysis: {str(e)}")


def calculate_plot_data_split_plot(df: pd.DataFrame, model, whole_plot_factor: str,
                                    subplot_factor: str, response: str,
                                    block: Optional[str] = None) -> Dict:
    """
    Calculate all data needed for split-plot visualizations

    Returns:
    - cell_means: Mean response for whole-plot × sub-plot combinations
    - marginal_means: Mean response for each factor level
    - fitted_values: Model predictions
    - residuals: Model residuals
    - factor_levels: Unique levels for each factor
    - box_plot_data: Raw data grouped by factors
    """
    def safe_float(value):
        """Convert value to float, handling NaN/inf by returning None"""
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)

    plot_data = {}

    # Extract fitted values and residuals from model
    plot_data["fitted_values"] = [safe_float(v) for v in model.fittedvalues.tolist()]
    plot_data["residuals"] = [safe_float(v) for v in model.resid.tolist()]

    # Factor list
    all_factors = [whole_plot_factor, subplot_factor]
    if block:
        all_factors.append(block)

    # Get factor levels
    plot_data["factor_levels"] = {}
    for factor in all_factors:
        plot_data["factor_levels"][factor] = sorted(df[factor].unique().tolist())

    # Calculate marginal means for main effects plots
    plot_data["marginal_means"] = {}
    for factor in [whole_plot_factor, subplot_factor]:
        marginal_data = []
        for level in sorted(df[factor].unique()):
            level_data = df[df[factor] == level][response]
            marginal_data.append({
                "level": str(level),
                "mean": safe_float(level_data.mean()),
                "std": safe_float(level_data.std()),
                "n": int(len(level_data))
            })
        plot_data["marginal_means"][factor] = marginal_data

    # Calculate cell means for interaction plot (WP × SP)
    cell_means = []
    for wp_level in sorted(df[whole_plot_factor].unique()):
        for sp_level in sorted(df[subplot_factor].unique()):
            cell_data = df[(df[whole_plot_factor] == wp_level) & (df[subplot_factor] == sp_level)][response]
            if len(cell_data) > 0:
                cell_means.append({
                    whole_plot_factor: str(wp_level),
                    subplot_factor: str(sp_level),
                    "mean": safe_float(cell_data.mean()),
                    "std": safe_float(cell_data.std()),
                    "n": int(len(cell_data))
                })
    plot_data["cell_means"] = cell_means

    # Prepare box plot data
    plot_data["box_plot_data"] = {}
    for factor in [whole_plot_factor, subplot_factor]:
        box_data = []
        for level in sorted(df[factor].unique()):
            level_values = df[df[factor] == level][response].tolist()
            box_data.append({
                "level": str(level),
                "values": [safe_float(v) for v in level_values]
            })
        plot_data["box_plot_data"][factor] = box_data

    return plot_data


def calculate_split_plot_ems(whole_plot_factor: str, subplot_factor: str, block: Optional[str],
                             wp_levels: int, sp_levels: int, n_blocks: int, n_per_cell: int) -> Dict[str, str]:
    """
    Calculate Expected Mean Squares for split-plot design

    Split-plot has hierarchical structure:
    - Blocks (if present)
    - Whole-plot factor (applied to large units)
    - Whole-plot error (WP × Block interaction)
    - Sub-plot factor (applied within whole plots)
    - WP × SP interaction
    - Sub-plot error (residual)
    """
    ems = {}

    if block:
        # RCBD at whole-plot level
        ems[block] = f"σ² + {sp_levels * n_per_cell}σ²({whole_plot_factor}×{block}) + {wp_levels * sp_levels * n_per_cell}σ²({block})"
        ems[whole_plot_factor] = f"σ² + {sp_levels * n_per_cell}σ²({whole_plot_factor}×{block}) + {n_blocks * sp_levels * n_per_cell}σ²({whole_plot_factor})"
        ems[f"{whole_plot_factor}:{block}"] = f"σ² + {sp_levels * n_per_cell}σ²({whole_plot_factor}×{block})"
        ems[subplot_factor] = f"σ² + {wp_levels * n_blocks * n_per_cell}σ²({subplot_factor})"
        ems[f"{whole_plot_factor}:{subplot_factor}"] = f"σ² + {n_blocks * n_per_cell}σ²({whole_plot_factor}×{subplot_factor})"
        ems["Residual"] = "σ²"
    else:
        # CRD at whole-plot level (simpler structure)
        ems[whole_plot_factor] = f"σ² + {sp_levels * n_per_cell}σ²(whole-plot) + {sp_levels * n_per_cell}σ²({whole_plot_factor})"
        ems[subplot_factor] = f"σ² + {wp_levels * n_per_cell}σ²({subplot_factor})"
        ems[f"{whole_plot_factor}:{subplot_factor}"] = f"σ² + {n_per_cell}σ²({whole_plot_factor}×{subplot_factor})"
        ems["Residual"] = "σ²"

    return ems


def estimate_split_plot_variance_components(ms_dict: Dict, whole_plot_factor: str,
                                             subplot_factor: str, block: Optional[str],
                                             wp_levels: int, sp_levels: int,
                                             n_blocks: int, n_per_cell: int) -> Dict[str, float]:
    """
    Estimate variance components for split-plot design

    Components:
    - σ²_subplot: Sub-plot error (within whole-plots)
    - σ²_wholeplot: Whole-plot error (between whole-plots)
    - σ²_interaction: WP × SP interaction
    - σ²_block: Block variance (if blocks present)
    """
    variance_components = {}

    # Sub-plot error (residual)
    ms_subplot_error = ms_dict.get("Residual", 0)
    variance_components["σ²_subplot"] = max(ms_subplot_error, 0)

    if block:
        # With blocks: RCBD structure
        wp_error_key = f"{whole_plot_factor}:{block}"
        ms_wp_error = ms_dict.get(wp_error_key, 0)

        # Whole-plot error variance component
        sigma_sq_wp_error = max((ms_wp_error - ms_subplot_error) / (sp_levels * n_per_cell), 0)
        variance_components[f"σ²_wholeplot"] = sigma_sq_wp_error

        # Interaction variance
        interaction_key = f"{whole_plot_factor}:{subplot_factor}"
        ms_interaction = ms_dict.get(interaction_key, 0)
        sigma_sq_interaction = max((ms_interaction - ms_subplot_error) / n_per_cell, 0)
        variance_components[f"σ²_{whole_plot_factor}×{subplot_factor}"] = sigma_sq_interaction

        # Block variance
        ms_block = ms_dict.get(block, 0)
        sigma_sq_block = max((ms_block - ms_wp_error) / (wp_levels * sp_levels * n_per_cell), 0)
        variance_components[f"σ²_{block}"] = sigma_sq_block

        # Whole-plot factor variance (fixed effect, for reference)
        ms_wp_factor = ms_dict.get(whole_plot_factor, 0)
        sigma_sq_wp_factor = max((ms_wp_factor - ms_wp_error) / (n_blocks * sp_levels * n_per_cell), 0)
        variance_components[f"σ²_{whole_plot_factor}"] = sigma_sq_wp_factor

    else:
        # Without blocks: CRD structure
        # Estimate whole-plot error from WP factor MS
        ms_wp_factor = ms_dict.get(whole_plot_factor, 0)
        sigma_sq_wp_error = max((ms_wp_factor - ms_subplot_error) / (sp_levels * n_per_cell), 0)
        variance_components["σ²_wholeplot"] = sigma_sq_wp_error

        # Interaction variance
        interaction_key = f"{whole_plot_factor}:{subplot_factor}"
        ms_interaction = ms_dict.get(interaction_key, 0)
        sigma_sq_interaction = max((ms_interaction - ms_subplot_error) / n_per_cell, 0)
        variance_components[f"σ²_{whole_plot_factor}×{subplot_factor}"] = sigma_sq_interaction

    return variance_components


def generate_split_plot_interpretation(anova_results: Dict, variance_components: Dict,
                                        whole_plot_factor: str, subplot_factor: str,
                                        alpha: float) -> List[str]:
    """Generate interpretation for split-plot design results"""
    interpretations = []

    interpretations.append("Split-Plot Design Analysis:")
    interpretations.append(f"• Whole-plot factor: {whole_plot_factor}")
    interpretations.append(f"• Sub-plot factor: {subplot_factor}")
    interpretations.append("")

    # Interpret significance
    for source, results in anova_results.items():
        if source not in ["Residual"] and "label" not in results:
            p_value = results.get("p_value_corrected") or results.get("p_value")
            error_term = results.get("error_term", "")

            if p_value is not None:
                if p_value < alpha:
                    interpretations.append(
                        f"{source}: Significant (p = {p_value:.4f}, tested against {error_term})"
                    )
                else:
                    interpretations.append(
                        f"{source}: Not significant (p = {p_value:.4f}, tested against {error_term})"
                    )

    # Variance component interpretation
    total_var = sum([v for v in variance_components.values() if v > 0])
    if total_var > 0:
        interpretations.append("")
        interpretations.append("Variance Components:")
        for component, var in variance_components.items():
            if var > 0:
                pct = (var / total_var) * 100
                interpretations.append(f"  {component}: {var:.6f} ({pct:.1f}%)")

    return interpretations


# ============================================================================
# NESTED DESIGN
# ============================================================================

class NestedDesignRequest(BaseModel):
    data: List[Dict] = Field(..., description="Experimental data")
    factor_a: str = Field(..., description="Higher-level factor")
    factor_b_nested: str = Field(..., description="Factor nested within A")
    response: str = Field(..., description="Response variable")
    alpha: float = Field(0.05, description="Significance level")


@router.post("/nested-design")
async def nested_design_analysis(request: NestedDesignRequest):
    """
    Analyze nested (hierarchical) design

    In nested designs, levels of B are unique to each level of A.
    For example: Teachers nested within Schools, Samples nested within Batches.

    Provides:
    - Expected Mean Squares (EMS) showing nesting structure
    - Variance components for each hierarchical level
    - Proper F-tests with correct error terms
    - Intraclass correlation coefficient (ICC)
    """
    def safe_float(value):
        """Convert value to float, handling NaN/inf by returning None"""
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)

    try:
        df = pd.DataFrame(request.data)

        # Validate data
        required_cols = [request.factor_a, request.factor_b_nested, request.response]
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in data")

        # Convert factors to categorical
        df[request.factor_a] = df[request.factor_a].astype('category')
        df[request.factor_b_nested] = df[request.factor_b_nested].astype('category')

        # Get factor levels
        a_levels = sorted(df[request.factor_a].unique())
        n_a = len(a_levels)

        # Count B levels nested within each A level
        b_per_a = df.groupby(request.factor_a)[request.factor_b_nested].nunique().values
        n_b_per_a = b_per_a[0]  # Assumes balanced (same number of B in each A)

        # Count observations per B
        n_per_b = df.groupby([request.factor_a, request.factor_b_nested]).size().values[0]

        # Build ANOVA model for nested design
        # Formula: Response ~ A + B(A)
        # In statsmodels, we need to create unique labels for B within each A
        df['B_in_A'] = df[request.factor_a].astype(str) + ':' + df[request.factor_b_nested].astype(str)

        formula = f"{request.response} ~ C({request.factor_a}) + C(B_in_A)"
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=1)

        # Calculate degrees of freedom
        df_a = n_a - 1
        df_b_in_a = n_a * (n_b_per_a - 1)
        df_error = n_a * n_b_per_a * (n_per_b - 1)

        # Extract sum of squares and mean squares
        ss_a = anova_table.loc[f'C({request.factor_a})', 'sum_sq']
        ss_b_in_a = anova_table.loc['C(B_in_A)', 'sum_sq']
        ss_error = anova_table.loc['Residual', 'sum_sq']

        ms_a = ss_a / df_a
        ms_b_in_a = ss_b_in_a / df_b_in_a
        ms_error = ss_error / df_error

        # Expected Mean Squares for nested design:
        # MS(A) = σ² + n*σ²(B(A)) + b*n*σ²(A)
        # MS(B(A)) = σ² + n*σ²(B(A))
        # MS(Error) = σ²

        # Estimate variance components using ANOVA method
        sigma2_error = ms_error
        sigma2_b_in_a = (ms_b_in_a - ms_error) / n_per_b
        sigma2_a = (ms_a - ms_b_in_a) / (n_b_per_a * n_per_b)

        # Ensure non-negative variance components
        sigma2_b_in_a = max(0, sigma2_b_in_a)
        sigma2_a = max(0, sigma2_a)

        # Calculate F-statistics with proper error terms
        # For nested design:
        # F(A) = MS(A) / MS(B(A))  [test A against B nested in A]
        # F(B(A)) = MS(B(A)) / MS(Error)  [test B(A) against residual]

        f_a = ms_a / ms_b_in_a if ms_b_in_a > 0 else None
        p_a = 1 - stats.f.cdf(f_a, df_a, df_b_in_a) if f_a is not None else None

        f_b_in_a = ms_b_in_a / ms_error if ms_error > 0 else None
        p_b_in_a = 1 - stats.f.cdf(f_b_in_a, df_b_in_a, df_error) if f_b_in_a is not None else None

        # Build ANOVA results table
        anova_results = {
            request.factor_a: {
                "sum_sq": safe_float(ss_a),
                "df": int(df_a),
                "mean_sq": safe_float(ms_a),
                "ems": f"σ² + {n_per_b}σ²({request.factor_b_nested}({request.factor_a})) + {n_b_per_a * n_per_b}σ²({request.factor_a})",
                "F": safe_float(f_a) if f_a is not None else None,
                "p_value": safe_float(p_a) if p_a is not None else None,
                "error_term": f"{request.factor_b_nested}({request.factor_a})"
            },
            f"{request.factor_b_nested}({request.factor_a})": {
                "sum_sq": safe_float(ss_b_in_a),
                "df": int(df_b_in_a),
                "mean_sq": safe_float(ms_b_in_a),
                "ems": f"σ² + {n_per_b}σ²({request.factor_b_nested}({request.factor_a}))",
                "F": safe_float(f_b_in_a) if f_b_in_a is not None else None,
                "p_value": safe_float(p_b_in_a) if p_b_in_a is not None else None,
                "error_term": "Error"
            },
            "Error": {
                "sum_sq": safe_float(ss_error),
                "df": int(df_error),
                "mean_sq": safe_float(ms_error),
                "ems": "σ²",
                "F": None,
                "p_value": None
            }
        }

        # Variance components
        variance_components = {
            "σ²_error": safe_float(sigma2_error),
            f"σ²_{request.factor_b_nested}({request.factor_a})": safe_float(sigma2_b_in_a),
            f"σ²_{request.factor_a}": safe_float(sigma2_a)
        }

        # Calculate variance percentages
        total_var = sigma2_error + sigma2_b_in_a + sigma2_a
        variance_percentages = {}
        if total_var > 0:
            err_pct = safe_float((sigma2_error / total_var) * 100)
            b_pct = safe_float((sigma2_b_in_a / total_var) * 100)
            a_pct = safe_float((sigma2_a / total_var) * 100)
            variance_percentages = {
                "σ²_error": round(err_pct, 2) if err_pct is not None else 0,
                f"σ²_{request.factor_b_nested}({request.factor_a})": round(b_pct, 2) if b_pct is not None else 0,
                f"σ²_{request.factor_a}": round(a_pct, 2) if a_pct is not None else 0
            }

        # Intraclass Correlation Coefficient (ICC)
        # ICC measures the proportion of variance due to grouping structure
        # ICC(A) = σ²(A) / (σ²(A) + σ²(B(A)) + σ²(error))
        icc_a_val = sigma2_a / total_var if total_var > 0 else 0
        icc_b_in_a_val = (sigma2_a + sigma2_b_in_a) / total_var if total_var > 0 else 0
        icc_a = safe_float(icc_a_val) if icc_a_val is not None else 0
        icc_b_in_a = safe_float(icc_b_in_a_val) if icc_b_in_a_val is not None else 0

        # Model fit statistics
        model_summary = {
            "r_squared": safe_float(model.rsquared),
            "adj_r_squared": safe_float(model.rsquared_adj),
            "f_statistic": safe_float(model.fvalue),
            "aic": safe_float(model.aic),
            "bic": safe_float(model.bic)
        }

        # Calculate plot data
        plot_data = calculate_plot_data_nested(
            df,
            model,
            request.factor_a,
            request.factor_b_nested,
            request.response
        )

        # Enhanced model fit metrics
        n_params = 2  # factor_a + factor_b_nested
        model_fit = calculate_model_fit_metrics(model, df, n_params)

        # Generate interpretation
        interpretation = generate_nested_interpretation(
            request.factor_a,
            request.factor_b_nested,
            anova_results,
            variance_components,
            variance_percentages,
            icc_a,
            icc_b_in_a,
            request.alpha
        )

        return {
            "model_type": "Nested Design",
            "factor_a": request.factor_a,
            "factor_b_nested": request.factor_b_nested,
            "anova_table": anova_results,
            "variance_components": variance_components,
            "variance_percentages": variance_percentages,
            "icc": {
                f"ICC({request.factor_a})": round(icc_a, 4) if icc_a is not None else 0,
                f"ICC(Total)": round(icc_b_in_a, 4) if icc_b_in_a is not None else 0
            },
            "model_summary": model_summary,
            "model_fit": model_fit,
            "blups": extract_blups(model, df, [request.factor_a, request.factor_b_nested], request.response),
            "plot_data": plot_data,
            "interpretation": interpretation
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def calculate_plot_data_nested(df: pd.DataFrame, model, factor_a: str,
                                 factor_b_nested: str, response: str) -> Dict:
    """
    Calculate all data needed for nested design visualizations
    """
    def safe_float(value):
        """Convert value to float, handling NaN/inf by returning None"""
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)

    plot_data = {}

    # Extract fitted values and residuals from model
    plot_data["fitted_values"] = [safe_float(v) for v in model.fittedvalues.tolist()]
    plot_data["residuals"] = [safe_float(v) for v in model.resid.tolist()]

    # Get factor levels
    plot_data["factor_levels"] = {
        factor_a: sorted(df[factor_a].unique().tolist()),
        factor_b_nested: sorted(df[factor_b_nested].unique().tolist())
    }

    # Calculate means for factor A (higher level)
    marginal_means_a = []
    for level in sorted(df[factor_a].unique()):
        level_data = df[df[factor_a] == level][response]
        marginal_means_a.append({
            "level": str(level),
            "mean": safe_float(level_data.mean()),
            "std": safe_float(level_data.std()),
            "n": int(len(level_data))
        })
    plot_data["marginal_means_a"] = marginal_means_a

    # Calculate means for B nested within each A
    nested_means = []
    for a_level in sorted(df[factor_a].unique()):
        a_data = df[df[factor_a] == a_level]
        for b_level in sorted(a_data[factor_b_nested].unique()):
            cell_data = a_data[a_data[factor_b_nested] == b_level][response]
            if len(cell_data) > 0:
                nested_means.append({
                    factor_a: str(a_level),
                    factor_b_nested: str(b_level),
                    "mean": safe_float(cell_data.mean()),
                    "std": safe_float(cell_data.std()),
                    "n": int(len(cell_data))
                })
    plot_data["nested_means"] = nested_means

    # Prepare box plot data for factor A
    box_data_a = []
    for level in sorted(df[factor_a].unique()):
        level_values = df[df[factor_a] == level][response].tolist()
        box_data_a.append({
            "level": str(level),
            "values": [safe_float(v) for v in level_values]
        })
    plot_data["box_plot_data_a"] = box_data_a

    # Prepare box plot data for B within each A
    box_data_nested = {}
    for a_level in sorted(df[factor_a].unique()):
        a_data = df[df[factor_a] == a_level]
        box_data = []
        for b_level in sorted(a_data[factor_b_nested].unique()):
            b_values = a_data[a_data[factor_b_nested] == b_level][response].tolist()
            box_data.append({
                "level": str(b_level),
                "values": [safe_float(v) for v in b_values]
            })
        box_data_nested[str(a_level)] = box_data
    plot_data["box_plot_data_nested"] = box_data_nested

    return plot_data


def generate_nested_interpretation(factor_a: str, factor_b_nested: str,
                                    anova_results: Dict, variance_components: Dict,
                                    variance_percentages: Dict, icc_a: float,
                                    icc_b_in_a: float, alpha: float) -> List[str]:
    """
    Generate human-readable interpretation of nested design results
    """
    interpretations = ["Nested Design Analysis:", f"• Higher-level factor: {factor_a}",
                      f"• Nested factor: {factor_b_nested} nested within {factor_a}", ""]

    # Test results
    p_a = anova_results[factor_a]["p_value"]
    p_b_in_a = anova_results[f"{factor_b_nested}({factor_a})"]["p_value"]

    if p_a is not None:
        sig_a = "Significant" if p_a < alpha else "Not significant"
        interpretations.append(f"{factor_a}: {sig_a} (p = {p_a:.4f}, tested against {factor_b_nested}({factor_a}))")

    if p_b_in_a is not None:
        sig_b = "Significant" if p_b_in_a < alpha else "Not significant"
        interpretations.append(f"{factor_b_nested}({factor_a}): {sig_b} (p = {p_b_in_a:.4f}, tested against Error)")

    interpretations.append("")

    # Variance components
    interpretations.append("Variance Components:")
    for component, value in variance_components.items():
        if component in variance_percentages:
            pct = variance_percentages[component]
            interpretations.append(f"  {component}: {value:.6f} ({pct:.1f}%)")

    interpretations.append("")

    # Intraclass correlation
    interpretations.append("Intraclass Correlation Coefficients (ICC):")
    interpretations.append(f"  ICC({factor_a}): {icc_a:.4f}")
    interpretations.append(f"    - Proportion of variance due to {factor_a} differences")
    interpretations.append(f"  ICC(Total): {icc_b_in_a:.4f}")
    interpretations.append(f"    - Proportion of variance due to {factor_a} and {factor_b_nested}({factor_a}) combined")

    interpretations.append("")
    interpretations.append("Interpretation:")
    if icc_a > 0.1:
        interpretations.append(f"  - Substantial variability exists between different {factor_a} levels")
    if icc_b_in_a - icc_a > 0.1:
        interpretations.append(f"  - Additional variability exists between {factor_b_nested} levels within {factor_a}")
    if icc_b_in_a < 0.3:
        interpretations.append(f"  - Most variability is at the individual observation level (within {factor_b_nested})")

    return interpretations


# ============================================================================
# REPEATED MEASURES ANOVA
# ============================================================================

class RepeatedMeasuresRequest(BaseModel):
    """Request for Repeated Measures ANOVA"""
    data: List[Dict] = Field(..., description="Experimental data with repeated measurements")
    subject: str = Field(..., description="Subject/participant identifier")
    within_factor: str = Field(..., description="Within-subjects factor (e.g., Time, Condition)")
    response: str = Field(..., description="Response variable name")
    alpha: float = Field(0.05, description="Significance level")


@router.post("/repeated-measures")
async def repeated_measures_anova(request: RepeatedMeasuresRequest):
    """
    Repeated Measures ANOVA for within-subjects designs

    Features:
    - Mauchly's test for sphericity
    - Greenhouse-Geisser and Huynh-Feldt corrections
    - Effect sizes (partial eta squared)
    - Profile plots and within-subject variability
    """
    def safe_float(value):
        """Convert value to float, handling NaN/inf by returning None"""
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)

    try:
        df = pd.DataFrame(request.data)

        # Validate data
        required_cols = [request.subject, request.within_factor, request.response]
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in data")

        # Convert to wide format for repeated measures
        # Each row = one subject, columns = different time points/conditions
        df_wide = df.pivot(index=request.subject, columns=request.within_factor, values=request.response)

        # Get number of subjects and conditions
        n_subjects = len(df_wide)
        conditions = df_wide.columns.tolist()
        n_conditions = len(conditions)

        if n_conditions < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 conditions for repeated measures")

        # Calculate means for each condition
        condition_means = df_wide.mean()
        grand_mean = df_wide.values.flatten().mean()

        # Calculate Sum of Squares
        # Between-subjects SS
        subject_means = df_wide.mean(axis=1)
        ss_subjects = n_conditions * np.sum((subject_means - grand_mean) ** 2)

        # Within-subjects SS (condition effect)
        ss_within = n_subjects * np.sum((condition_means - grand_mean) ** 2)

        # Error SS (residual)
        ss_error = 0
        for subj_idx in range(n_subjects):
            for cond_idx in range(n_conditions):
                observed = df_wide.iloc[subj_idx, cond_idx]
                expected = subject_means.iloc[subj_idx] + condition_means.iloc[cond_idx] - grand_mean
                ss_error += (observed - expected) ** 2

        # Total SS
        ss_total = np.sum((df_wide.values - grand_mean) ** 2)

        # Degrees of freedom
        df_within = n_conditions - 1
        df_subjects = n_subjects - 1
        df_error = (n_subjects - 1) * (n_conditions - 1)
        df_total = n_subjects * n_conditions - 1

        # Mean Squares
        ms_within = ss_within / df_within
        ms_error = ss_error / df_error
        ms_subjects = ss_subjects / df_subjects

        # F-statistic and p-value
        f_statistic = ms_within / ms_error if ms_error > 0 else None
        p_value = 1 - stats.f.cdf(f_statistic, df_within, df_error) if f_statistic is not None else None

        # Effect size (partial eta squared)
        partial_eta_sq = ss_within / (ss_within + ss_error) if (ss_within + ss_error) > 0 else 0

        # Mauchly's Test for Sphericity (only if k >= 3)
        sphericity_test = None
        sphericity_p = None
        epsilon_gg = 1.0  # Greenhouse-Geisser epsilon
        epsilon_hf = 1.0  # Huynh-Feldt epsilon

        if n_conditions >= 3:
            # Calculate difference scores for all pairs
            diff_matrix = np.zeros((n_subjects, n_conditions - 1))
            for i in range(n_conditions - 1):
                diff_matrix[:, i] = df_wide.iloc[:, i] - df_wide.iloc[:, i + 1]

            # Covariance matrix of differences
            cov_matrix = np.cov(diff_matrix.T)

            # Mauchly's W statistic
            det_cov = np.linalg.det(cov_matrix)
            trace_cov = np.trace(cov_matrix)
            k = n_conditions - 1

            if trace_cov > 0 and k > 0:
                w_statistic = det_cov / ((trace_cov / k) ** k)

                # Chi-square approximation
                chi_sq = -(n_subjects - 1 - (2 * k + 5) / 6) * np.log(w_statistic)
                df_chi = k * (k + 1) / 2 - 1
                sphericity_p = 1 - stats.chi2.cdf(chi_sq, df_chi)
                sphericity_test = safe_float(w_statistic)

                # Greenhouse-Geisser epsilon
                eigenvalues = np.linalg.eigvals(cov_matrix)
                lambda_sum = np.sum(eigenvalues)
                lambda_sq_sum = np.sum(eigenvalues ** 2)
                epsilon_gg = (lambda_sum ** 2) / (k * lambda_sq_sum) if lambda_sq_sum > 0 else 1.0
                epsilon_gg = max(1 / k, min(1.0, epsilon_gg))  # Bound between 1/k and 1

                # Huynh-Feldt epsilon
                n = n_subjects
                epsilon_hf = (n * k * epsilon_gg - 2) / (k * (n - 1 - k * epsilon_gg))
                epsilon_hf = max(1 / k, min(1.0, epsilon_hf))

        # Corrected tests
        f_gg = f_statistic
        df_within_gg = df_within * epsilon_gg
        df_error_gg = df_error * epsilon_gg
        p_value_gg = 1 - stats.f.cdf(f_gg, df_within_gg, df_error_gg) if f_gg is not None else None

        f_hf = f_statistic
        df_within_hf = df_within * epsilon_hf
        df_error_hf = df_error * epsilon_hf
        p_value_hf = 1 - stats.f.cdf(f_hf, df_within_hf, df_error_hf) if f_hf is not None else None

        # Build ANOVA table
        anova_table = {
            request.within_factor: {
                "sum_sq": safe_float(ss_within),
                "df": int(df_within),
                "mean_sq": safe_float(ms_within),
                "F": safe_float(f_statistic),
                "p_value": safe_float(p_value),
                "partial_eta_sq": safe_float(partial_eta_sq)
            },
            "Error": {
                "sum_sq": safe_float(ss_error),
                "df": int(df_error),
                "mean_sq": safe_float(ms_error),
                "F": None,
                "p_value": None
            },
            "Subjects": {
                "sum_sq": safe_float(ss_subjects),
                "df": int(df_subjects),
                "mean_sq": safe_float(ms_subjects),
                "F": None,
                "p_value": None
            }
        }

        # Sphericity results
        spher_assumed = bool(sphericity_p is None or sphericity_p >= request.alpha)
        sphericity_results = {
            "w_statistic": sphericity_test,
            "p_value": safe_float(sphericity_p) if sphericity_p is not None else None,
            "epsilon_gg": safe_float(epsilon_gg),
            "epsilon_hf": safe_float(epsilon_hf),
            "sphericity_assumed": spher_assumed,
            "recommendation": "Use uncorrected test" if spher_assumed else "Use Greenhouse-Geisser or Huynh-Feldt correction"
        }

        # Corrected tests
        corrected_tests = {
            "greenhouse_geisser": {
                "F": safe_float(f_gg),
                "df_numerator": safe_float(df_within_gg),
                "df_denominator": safe_float(df_error_gg),
                "p_value": safe_float(p_value_gg)
            },
            "huynh_feldt": {
                "F": safe_float(f_hf),
                "df_numerator": safe_float(df_within_hf),
                "df_denominator": safe_float(df_error_hf),
                "p_value": safe_float(p_value_hf)
            }
        }

        # Calculate plot data
        plot_data = calculate_plot_data_repeated_measures(df, request.subject, request.within_factor, request.response, condition_means, conditions)

        # Generate interpretation
        interpretation = generate_repeated_measures_interpretation(
            request.within_factor,
            anova_table,
            sphericity_results,
            corrected_tests,
            request.alpha
        )

        # Calculate ICC for subjects (random effect in repeated measures)
        icc_results = {}
        if request.subject:
            icc_results[request.subject] = calculate_icc(
                df, request.subject, request.response, icc_type="icc2"
            )

        # Enhanced model fit metrics
        # For repeated measures: within_factor + subjects (+ error)
        n_params = n_conditions + n_subjects
        # Create a simple model object wrapper for compatibility
        class ModelWrapper:
            def __init__(self, anova_table, n_obs):
                # Approximate log-likelihood from SS and df
                ss_error = anova_table["Error"]["sum_sq"]
                df_error = anova_table["Error"]["df"]
                mse = ss_error / df_error if df_error > 0 else 1

                # Ensure mse is positive and finite
                if mse <= 0 or np.isinf(mse) or np.isnan(mse):
                    mse = 1

                self.llf = -0.5 * n_obs * (np.log(2 * np.pi * mse) + 1)

                # Ensure llf is finite
                if np.isinf(self.llf) or np.isnan(self.llf):
                    self.llf = -n_obs  # Reasonable default

                self.aic = -2 * self.llf + 2 * n_params
                self.bic = -2 * self.llf + np.log(n_obs) * n_params

        model_wrapper = ModelWrapper(anova_table, len(df))
        model_fit = calculate_model_fit_metrics(model_wrapper, df, n_params)

        return {
            "model_type": "Repeated Measures ANOVA",
            "within_factor": request.within_factor,
            "n_subjects": n_subjects,
            "n_conditions": n_conditions,
            "anova_table": anova_table,
            "sphericity": sphericity_results,
            "corrected_tests": corrected_tests,
            "plot_data": plot_data,
            "interpretation": interpretation,
            "icc": icc_results,
            "model_fit": model_fit
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def calculate_plot_data_repeated_measures(df: pd.DataFrame, subject: str,
                                           within_factor: str, response: str,
                                           condition_means: pd.Series, conditions: list) -> Dict:
    """Calculate plot data for repeated measures visualizations"""
    def safe_float(value):
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)

    plot_data = {}

    # Profile plot data (mean and error bars)
    profile_data = []
    for condition in conditions:
        cond_data = df[df[within_factor] == condition][response]
        profile_data.append({
            "condition": str(condition),
            "mean": safe_float(cond_data.mean()),
            "std": safe_float(cond_data.std()),
            "sem": safe_float(cond_data.sem()),
            "n": int(len(cond_data))
        })
    plot_data["profile_data"] = profile_data

    # Individual trajectories (for spaghetti plot)
    trajectories = []
    for subj in df[subject].unique():
        subj_data = df[df[subject] == subj].sort_values(within_factor)
        trajectory = {
            "subject": str(subj),
            "values": [safe_float(v) for v in subj_data[response].values]
        }
        trajectories.append(trajectory)
    plot_data["trajectories"] = trajectories

    # Within-subject variability
    df_wide = df.pivot(index=subject, columns=within_factor, values=response)
    within_subj_std = df_wide.std(axis=1)
    variability_data = {
        "mean_within_subj_std": safe_float(within_subj_std.mean()),
        "subjects": [
            {
                "subject": str(subj),
                "std": safe_float(std_val)
            }
            for subj, std_val in within_subj_std.items()
        ]
    }
    plot_data["within_subject_variability"] = variability_data

    return plot_data


def generate_repeated_measures_interpretation(within_factor: str, anova_table: Dict,
                                               sphericity: Dict, corrected_tests: Dict,
                                               alpha: float) -> List[str]:
    """Generate interpretation for repeated measures ANOVA"""
    interpretations = [
        "Repeated Measures ANOVA Results:",
        f"• Within-subjects factor: {within_factor}",
        ""
    ]

    # Main effect
    p_val = anova_table[within_factor]["p_value"]
    eta_sq = anova_table[within_factor]["partial_eta_sq"]

    if p_val is not None:
        sig_status = "Significant" if p_val < alpha else "Not significant"
        interpretations.append(f"Main Effect of {within_factor}: {sig_status}")
        interpretations.append(f"  F = {anova_table[within_factor]['F']:.4f}, p = {p_val:.4f}")
        interpretations.append(f"  Partial η² = {eta_sq:.4f}")

        # Effect size interpretation
        if eta_sq >= 0.14:
            interpretations.append("  (Large effect size)")
        elif eta_sq >= 0.06:
            interpretations.append("  (Medium effect size)")
        elif eta_sq >= 0.01:
            interpretations.append("  (Small effect size)")

    interpretations.append("")

    # Sphericity
    if sphericity["p_value"] is not None:
        interpretations.append("Sphericity Test (Mauchly's):")
        interpretations.append(f"  W = {sphericity['w_statistic']:.4f}, p = {sphericity['p_value']:.4f}")

        if sphericity["sphericity_assumed"]:
            interpretations.append("  ✓ Sphericity assumption met")
        else:
            interpretations.append("  ✗ Sphericity assumption violated")
            interpretations.append(f"  Recommendation: {sphericity['recommendation']}")
            interpretations.append("")
            interpretations.append("Corrected Tests:")
            interpretations.append(f"  Greenhouse-Geisser: F = {corrected_tests['greenhouse_geisser']['F']:.4f}, p = {corrected_tests['greenhouse_geisser']['p_value']:.4f}")
            interpretations.append(f"  Huynh-Feldt: F = {corrected_tests['huynh_feldt']['F']:.4f}, p = {corrected_tests['huynh_feldt']['p_value']:.4f}")
    else:
        interpretations.append("Sphericity test not applicable (< 3 conditions)")

    return interpretations


# ============================================================================
# ICC (INTRACLASS CORRELATION COEFFICIENT) CALCULATIONS
# ============================================================================

def calculate_icc(df: pd.DataFrame, groups: str, response: str, icc_type: str = "icc2") -> Dict:
    """
    Calculate Intraclass Correlation Coefficient (ICC)
    
    Parameters:
    - df: DataFrame with data
    - groups: Column name for grouping variable (e.g., subjects, clusters)
    - response: Column name for response variable
    - icc_type: Type of ICC to calculate
      - "icc1": ICC(1) - Each target is rated by a different set of k raters randomly selected from a larger population
      - "icc2": ICC(2) - Each target is rated by the same set of k raters (consistency)
      - "icc3": ICC(3) - Each target is rated by the same set of k raters (absolute agreement)
    
    Returns:
    Dictionary with ICC value, confidence interval, F-statistic, and interpretation
    """
    try:
        # Group data
        grouped = df.groupby(groups)[response]
        
        # Calculate group statistics
        k = grouped.count().mean()  # Average number of observations per group
        n = grouped.ngroups  # Number of groups
        
        # Calculate between-group and within-group variance
        grand_mean = df[response].mean()
        
        # Between-group sum of squares (BMS)
        group_means = grouped.mean()
        group_sizes = grouped.count()
        ss_between = sum(group_sizes * (group_means - grand_mean) ** 2)
        ms_between = ss_between / (n - 1)
        
        # Within-group sum of squares (WMS)
        ss_within = sum([
            sum((df[df[groups] == group][response] - group_means[group]) ** 2)
            for group in group_means.index
        ])
        df_within = sum(group_sizes - 1)
        ms_within = ss_within / df_within if df_within > 0 else 0
        
        # Calculate ICC based on type
        if icc_type == "icc1":
            # ICC(1): Single rater, absolute agreement
            icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        elif icc_type == "icc2":
            # ICC(2): Average of k raters, consistency
            icc = (ms_between - ms_within) / ms_between
        elif icc_type == "icc3":
            # ICC(3): Average of k raters, absolute agreement  
            icc = (ms_between - ms_within) / (ms_between + (ms_within / k))
        else:
            icc = (ms_between - ms_within) / ms_between  # Default to ICC(2)
        
        # Bound ICC between 0 and 1
        icc = max(0, min(1, icc))
        
        # F-statistic for significance test
        f_statistic = ms_between / ms_within if ms_within > 0 else np.inf
        df1 = n - 1
        df2 = df_within
        p_value = 1 - stats.f.cdf(f_statistic, df1, df2) if ms_within > 0 else 0
        
        # Confidence interval (95%)
        alpha = 0.05
        f_lower = stats.f.ppf(alpha / 2, df1, df2)
        f_upper = stats.f.ppf(1 - alpha / 2, df1, df2)
        
        ci_lower = (f_statistic / f_upper - 1) / (f_statistic / f_upper + k - 1)
        ci_upper = (f_statistic / f_lower - 1) / (f_statistic / f_lower + k - 1)
        
        ci_lower = max(0, min(1, ci_lower))
        ci_upper = max(0, min(1, ci_upper))
        
        # Interpretation
        if icc < 0.5:
            interpretation = "Poor reliability"
            quality = "poor"
        elif icc < 0.75:
            interpretation = "Moderate reliability"
            quality = "moderate"
        elif icc < 0.9:
            interpretation = "Good reliability"
            quality = "good"
        else:
            interpretation = "Excellent reliability"
            quality = "excellent"
        
        return {
            "icc": round(float(icc), 4),
            "icc_type": icc_type.upper(),
            "ci_lower": round(float(ci_lower), 4),
            "ci_upper": round(float(ci_upper), 4),
            "f_statistic": round(float(f_statistic), 4),
            "p_value": round(float(p_value), 6),
            "df1": int(df1),
            "df2": int(df2),
            "n_groups": int(n),
            "avg_group_size": round(float(k), 2),
            "interpretation": interpretation,
            "quality": quality,
            "ms_between": round(float(ms_between), 4),
            "ms_within": round(float(ms_within), 4)
        }
        
    except Exception as e:
        return {
            "error": f"Could not calculate ICC: {str(e)}",
            "icc": None
        }


def calculate_model_fit_metrics(model, df: pd.DataFrame, n_params: int) -> Dict:
    """
    Calculate comprehensive model fit metrics
    
    Parameters:
    - model: Fitted statsmodels model
    - df: Original dataframe
    - n_params: Number of parameters in model
    
    Returns:
    Dictionary with AIC, BIC, log-likelihood, and other fit metrics
    """
    try:
        n_obs = len(df)
        
        # Extract from model if available
        aic = float(model.aic) if hasattr(model, 'aic') else None
        bic = float(model.bic) if hasattr(model, 'bic') else None
        log_likelihood = float(model.llf) if hasattr(model, 'llf') else None
        
        # Calculate additional metrics
        if aic is not None and bic is not None:
            # CAIC (Consistent AIC)
            caic = aic + 2 * n_params * (n_params + 1) / (n_obs - n_params - 1)
            
            # Adjusted BIC
            adj_bic = bic - n_params * np.log(2 * np.pi)
            
            # AIC weight (for model comparison)
            # This is relative and needs multiple models, so we'll include delta_aic placeholder
            
            return {
                "aic": round(aic, 2),
                "bic": round(bic, 2),
                "caic": round(caic, 2),
                "adj_bic": round(adj_bic, 2),
                "log_likelihood": round(log_likelihood, 4) if log_likelihood else None,
                "n_parameters": n_params,
                "n_observations": n_obs,
                "aic_per_obs": round(aic / n_obs, 4),
                "bic_per_obs": round(bic / n_obs, 4)
            }
        
        return {
            "aic": aic,
            "bic": bic,
            "log_likelihood": log_likelihood,
            "n_parameters": n_params,
            "n_observations": n_obs
        }
        
    except Exception as e:
        return {
            "error": f"Could not calculate fit metrics: {str(e)}"
        }

def extract_blups(model, df: pd.DataFrame, random_factors: List[str], response: str) -> Dict:
    """
    Extract Best Linear Unbiased Predictions (BLUPs) for random effects

    BLUPs are predictions of the random effects in a mixed model. They represent
    the deviation of each group from the overall population mean, accounting for
    shrinkage toward zero based on the amount of information available.

    Parameters:
    - model: Fitted statsmodels MixedLM model
    - df: Original dataframe
    - random_factors: List of random effect factor names
    - response: Response variable name

    Returns:
    Dictionary with BLUPs, standard errors, and shrinkage information for each random factor
    """
    def safe_float(value, default=0.0):
        """Convert value to float, handling NaN/inf by returning default"""
        try:
            if pd.isna(value) or np.isinf(value):
                return default
            return float(value)
        except:
            return default

    try:
        blups_data = {}
        
        for factor in random_factors:
            factor_blups = []
            
            # Get unique levels of this factor
            levels = df[factor].unique()
            
            # Calculate observed means for each level (before shrinkage)
            observed_means = []
            grand_mean = df[response].mean()
            
            for level in levels:
                level_data = df[df[factor] == level][response]
                obs_mean = level_data.mean()
                obs_deviation = obs_mean - grand_mean
                n_obs = len(level_data)
                
                observed_means.append({
                    "level": str(level),
                    "observed_deviation": safe_float(obs_deviation),
                    "n_observations": int(n_obs)
                })
            
            # Try to extract BLUPs from model if available
            if hasattr(model, 'random_effects'):
                # statsmodels MixedLM stores random effects
                random_effects = model.random_effects
                
                for level_info in observed_means:
                    level = level_info["level"]
                    obs_dev = level_info["observed_deviation"]
                    
                    # Try to find this level in random effects
                    blup_value = None
                    if level in random_effects:
                        # Random effects dict contains BLUP values
                        re = random_effects[level]
                        if hasattr(re, 'values'):
                            blup_value = safe_float(re.values[0]) if len(re.values) > 0 else None
                        elif isinstance(re, (int, float)):
                            blup_value = safe_float(re)
                    
                    # If we couldn't extract BLUP, estimate it using empirical Bayes
                    if blup_value is None:
                        # Shrinkage factor approximation
                        # blup ≈ shrinkage * observed_deviation
                        # Where shrinkage = τ² / (τ² + σ²/n)
                        # For now, use observed deviation (can be improved with actual variance components)
                        blup_value = obs_dev * 0.7  # Rough shrinkage approximation
                    
                    # Calculate shrinkage: how much was the observed mean "pulled" toward zero
                    if obs_dev != 0:
                        shrinkage_factor = 1 - abs(blup_value / obs_dev)
                    else:
                        shrinkage_factor = 0
                    
                    # Estimate standard error (rough approximation)
                    n_obs = level_info["n_observations"]
                    residual_var = df[response].var()
                    se = np.sqrt(residual_var / n_obs)
                    
                    # Confidence interval (95%)
                    ci_lower = blup_value - 1.96 * se
                    ci_upper = blup_value + 1.96 * se
                    
                    factor_blups.append({
                        "level": level,
                        "blup": round(safe_float(blup_value), 4),
                        "observed_deviation": round(safe_float(obs_dev), 4),
                        "shrinkage_factor": round(safe_float(shrinkage_factor), 4),
                        "se": round(safe_float(se, 0.01), 4),
                        "ci_lower": round(safe_float(ci_lower), 4),
                        "ci_upper": round(safe_float(ci_upper), 4),
                        "n_observations": int(n_obs)
                    })
            else:
                # Model doesn't have random_effects attribute, use empirical approach
                for level_info in observed_means:
                    level = level_info["level"]
                    obs_dev = level_info["observed_deviation"]
                    n_obs = level_info["n_observations"]
                    
                    # Empirical Bayes shrinkage estimate
                    # Shrinkage = τ² / (τ² + σ²/n)
                    # Use rough estimates from data
                    between_var = df.groupby(factor)[response].mean().var()
                    within_var = df[response].var() - between_var
                    
                    if between_var + within_var / n_obs > 0:
                        empirical_shrinkage = between_var / (between_var + within_var / n_obs)
                    else:
                        empirical_shrinkage = 0.5
                    
                    blup_value = obs_dev * empirical_shrinkage
                    shrinkage_factor = 1 - empirical_shrinkage
                    
                    # Standard error
                    se = np.sqrt(within_var / n_obs)
                    ci_lower = blup_value - 1.96 * se
                    ci_upper = blup_value + 1.96 * se
                    
                    factor_blups.append({
                        "level": level,
                        "blup": round(safe_float(blup_value), 4),
                        "observed_deviation": round(safe_float(obs_dev), 4),
                        "shrinkage_factor": round(safe_float(shrinkage_factor), 4),
                        "se": round(safe_float(se, 0.01), 4),
                        "ci_lower": round(safe_float(ci_lower), 4),
                        "ci_upper": round(safe_float(ci_upper), 4),
                        "n_observations": int(n_obs)
                    })
            
            # Sort by BLUP value
            factor_blups.sort(key=lambda x: x["blup"])
            
            # Calculate summary statistics
            blup_values = [b["blup"] for b in factor_blups]
            summary = {
                "mean_blup": round(safe_float(np.mean(blup_values)), 4),
                "std_blup": round(safe_float(np.std(blup_values)), 4),
                "min_blup": round(safe_float(np.min(blup_values)), 4),
                "max_blup": round(safe_float(np.max(blup_values)), 4),
                "mean_shrinkage": round(safe_float(np.mean([b["shrinkage_factor"] for b in factor_blups])), 4)
            }
            
            blups_data[factor] = {
                "blups": factor_blups,
                "summary": summary,
                "n_levels": len(factor_blups)
            }
        
        return blups_data
        
    except Exception as e:
        return {
            "error": f"Could not extract BLUPs: {str(e)}"
        }
