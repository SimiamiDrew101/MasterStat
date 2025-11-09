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
            "model_summary": {
                "r_squared": round(float(model.rsquared), 4),
                "adj_r_squared": round(float(model.rsquared_adj), 4),
                "f_statistic": round(float(model.fvalue), 4),
                "aic": round(float(model.aic), 4),
                "bic": round(float(model.bic), 4)
            },
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
            "model_summary": {
                "r_squared": safe_float(model.rsquared),
                "adj_r_squared": safe_float(model.rsquared_adj),
                "f_statistic": safe_float(model.fvalue),
                "aic": safe_float(model.aic),
                "bic": safe_float(model.bic)
            },
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

    In nested designs, levels of B are unique to each level of A
    Example: Students nested within Schools
    """
    try:
        df = pd.DataFrame(request.data)

        # Implementation will be enhanced next
        return {
            "message": "Nested design analysis - full implementation coming soon",
            "model_type": "Nested Design"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
