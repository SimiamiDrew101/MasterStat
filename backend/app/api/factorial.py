from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import numpy as np
import pandas as pd
from itertools import product
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

router = APIRouter()

def generate_factorial_interpretation(results: dict, factors: list, alpha: float,
                                      r_squared: float, adj_r_squared: float,
                                      has_replicates: bool) -> dict:
    """
    Generate interpretation and recommendations from factorial analysis results
    """
    # Identify significant factors and interactions
    significant_factors = []
    significant_interactions = []
    insignificant_factors = []

    for source, values in results.items():
        if source in ['Residual', 'Lack of Fit', 'Pure Error']:
            continue

        p_value = values.get('p_value')
        if p_value is not None:
            is_significant = p_value < alpha

            # Check if it's an interaction (contains ':' or '×')
            if ':' in source or '×' in source:
                if is_significant:
                    significant_interactions.append({
                        'name': source,
                        'p_value': p_value
                    })
            else:
                # It's a main effect
                if is_significant:
                    significant_factors.append({
                        'name': source,
                        'p_value': p_value
                    })
                else:
                    insignificant_factors.append({
                        'name': source,
                        'p_value': p_value
                    })

    # Assess model fit
    model_fit_quality = "excellent" if r_squared >= 0.90 else \
                       "good" if r_squared >= 0.75 else \
                       "moderate" if r_squared >= 0.50 else "poor"

    # Check lack of fit (if available)
    lacks_fit = False
    lof_p_value = None
    if 'Lack of Fit' in results:
        lof_p_value = results['Lack of Fit'].get('p_value')
        if lof_p_value is not None:
            lacks_fit = lof_p_value < alpha

    # Generate summary
    summary_parts = []

    # Model fit statement
    if has_replicates and lof_p_value is not None:
        if lacks_fit:
            summary_parts.append(f"⚠️ The model shows significant lack of fit (p = {lof_p_value:.4f}), indicating the current model may not adequately describe the response.")
        else:
            summary_parts.append(f"✓ The model fits the data well with no significant lack of fit (p = {lof_p_value:.4f}).")

    summary_parts.append(f"The model explains {r_squared*100:.1f}% of the variation in the response (R² = {r_squared:.4f}, Adj R² = {adj_r_squared:.4f}), indicating {model_fit_quality} fit.")

    # Significant factors
    if significant_factors:
        factor_names = ', '.join([f['name'] for f in significant_factors])
        summary_parts.append(f"Significant main effects: {factor_names}.")
    else:
        summary_parts.append("No main effects are statistically significant at the chosen α level.")

    # Significant interactions
    if significant_interactions:
        interaction_names = ', '.join([f['name'] for f in significant_interactions])
        summary_parts.append(f"Significant interactions: {interaction_names}.")

    # Recommendations
    recommendations = []

    if significant_factors:
        recommendations.append(f"✓ Proceed with factors: {', '.join([f['name'] for f in significant_factors])}. These have significant effects on the response.")
    else:
        recommendations.append("Consider re-evaluating the factor levels or exploring additional factors, as none of the current factors show significant effects.")

    if insignificant_factors:
        recommendations.append(f"Consider removing: {', '.join([f['name'] for f in insignificant_factors])}. These factors do not significantly affect the response at α = {alpha}.")

    if significant_interactions:
        recommendations.append(f"Important: Interactions detected between factors. Analyze {', '.join([f['name'] for f in significant_interactions])} carefully, as the effect of one factor depends on the level of another.")

    if lacks_fit and has_replicates:
        recommendations.append("⚠️ Consider adding quadratic terms or additional factors to improve model fit, as the current linear model is inadequate.")
    elif not has_replicates and len(factors) > 2:
        recommendations.append("💡 Consider running replicates to enable pure error estimation and lack-of-fit testing for more robust conclusions.")

    return {
        "summary": " ".join(summary_parts),
        "recommendations": recommendations,
        "model_fit_quality": model_fit_quality,
        "r_squared": round(r_squared, 4),
        "significant_factors": [f['name'] for f in significant_factors],
        "significant_interactions": [f['name'] for f in significant_interactions],
        "insignificant_factors": [f['name'] for f in insignificant_factors],
        "lacks_fit": lacks_fit
    }

class FactorialDesignRequest(BaseModel):
    data: List[Dict[str, Union[str, float]]] = Field(..., description="Experimental data")
    factors: List[str] = Field(..., description="List of factor names")
    response: str = Field(..., description="Response variable name")
    alpha: float = Field(0.05, description="Significance level")

class FractionalFactorialRequest(BaseModel):
    factors: int = Field(..., description="Number of factors (k)")
    fraction: str = Field(..., description="Fraction specification (e.g., '1/2', '1/4')")
    generator: Optional[str] = Field(None, description="Generator for fractional design (e.g., 'D=ABC')")

@router.post("/full-factorial")
async def full_factorial_analysis(request: FactorialDesignRequest):
    """
    Analyze full factorial design with main effects and interactions
    """
    try:
        df = pd.DataFrame(request.data)

        # Rename columns to avoid conflicts with Python built-ins
        # Add 'factor_' prefix to factors and 'response_' to response
        column_mapping = {}
        for factor in request.factors:
            column_mapping[factor] = f"factor_{factor}"
        column_mapping[request.response] = f"response_{request.response}"

        df_renamed = df.rename(columns=column_mapping)

        # Build formula with renamed columns
        renamed_factors = [f"factor_{f}" for f in request.factors]
        renamed_response = f"response_{request.response}"
        factor_terms = [f"C({f})" for f in renamed_factors]

        # Create formula with main effects and interactions
        # For unreplicated designs, exclude highest-order interaction to have error degrees of freedom
        if len(request.factors) == 2:
            formula = f"{renamed_response} ~ {factor_terms[0]} * {factor_terms[1]}"
        elif len(request.factors) == 3:
            # For 2^3 designs: include main effects and 2-way interactions, but NOT 3-way
            # This pools the 3-way interaction as error
            main_effects = " + ".join(factor_terms)
            two_way = []
            for i in range(len(factor_terms)):
                for j in range(i+1, len(factor_terms)):
                    two_way.append(f"{factor_terms[i]}:{factor_terms[j]}")
            formula = f"{renamed_response} ~ {main_effects} + {' + '.join(two_way)}"
        else:
            # For more factors, use additive model with 2-way interactions
            main_effects = " + ".join(factor_terms)
            interactions = []
            for i in range(len(factor_terms)):
                for j in range(i+1, len(factor_terms)):
                    interactions.append(f"{factor_terms[i]}:{factor_terms[j]}")
            formula = f"{renamed_response} ~ {main_effects} + {' + '.join(interactions)}"

        # Fit model
        model = ols(formula, data=df_renamed).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Check for replicates to calculate pure error and lack of fit
        # Group by all factors to find replicates
        factor_cols = request.factors
        grouped = df.groupby(factor_cols)[request.response]

        # Check if we have replicates (any group size > 1)
        has_replicates = any(grouped.count() > 1)
        pure_error_ss = 0.0
        pure_error_df = 0

        if has_replicates:
            # Calculate pure error from within-group variation
            for name, group in grouped:
                if len(group) > 1:
                    group_var = group.var(ddof=1)
                    if pd.notna(group_var):
                        pure_error_ss += group_var * (len(group) - 1)
                        pure_error_df += len(group) - 1

        # Parse ANOVA results
        results = {}
        for idx, row in anova_table.iterrows():
            source = str(idx)

            # Clean up source names - remove C() wrappers and factor_ prefix
            for factor in request.factors:
                source = source.replace(f"C(factor_{factor})", factor)
                source = source.replace(f"factor_{factor}", factor)

            results[source] = {
                "sum_sq": round(float(row['sum_sq']), 4),
                "df": int(row['df']),
                "F": round(float(row['F']), 4) if not pd.isna(row['F']) else None,
                "p_value": round(float(row['PR(>F)']), 6) if not pd.isna(row['PR(>F)']) else None,
                "significant": bool(row['PR(>F)'] < request.alpha) if not pd.isna(row['PR(>F)']) else False
            }

        # Add pure error and lack of fit if we have replicates
        if has_replicates and pure_error_df > 0:
            # Get residual row from results
            residual_ss = results.get('Residual', {}).get('sum_sq', 0)
            residual_df = results.get('Residual', {}).get('df', 0)

            # Lack of fit = Residual - Pure Error
            lof_ss = residual_ss - pure_error_ss
            lof_df = residual_df - pure_error_df

            if lof_df > 0 and pure_error_df > 0:
                lof_ms = lof_ss / lof_df
                pure_error_ms = pure_error_ss / pure_error_df

                # F-test for lack of fit
                if pure_error_ms > 0:
                    f_lof = lof_ms / pure_error_ms
                    from scipy import stats
                    p_lof = 1 - stats.f.cdf(f_lof, lof_df, pure_error_df)

                    # Remove old Residual, add Lack of Fit and Pure Error
                    if 'Residual' in results:
                        del results['Residual']

                    results['Lack of Fit'] = {
                        "sum_sq": round(float(lof_ss), 4),
                        "df": int(lof_df),
                        "F": round(float(f_lof), 4),
                        "p_value": round(float(p_lof), 6),
                        "significant": bool(p_lof < request.alpha)
                    }

                    results['Pure Error'] = {
                        "sum_sq": round(float(pure_error_ss), 4),
                        "df": int(pure_error_df),
                        "F": None,
                        "p_value": None,
                        "significant": False
                    }

        # Calculate effect estimates (for 2-level designs)
        # Use original df for data access since we only renamed for formula
        effects = {}
        for factor in request.factors:
            if df[factor].nunique() == 2:
                levels = sorted(df[factor].unique())
                high = df[df[factor] == levels[1]][request.response].mean()
                low = df[df[factor] == levels[0]][request.response].mean()

                # Check for NaN
                if pd.notna(high) and pd.notna(low):
                    effects[factor] = round(float(high - low), 4)
                else:
                    effects[factor] = 0.0

        # Calculate interaction effects for 2-factor case
        interaction_effects = {}
        if len(request.factors) == 2:
            f1, f2 = request.factors
            for level1 in sorted(df[f1].unique()):
                for level2 in sorted(df[f2].unique()):
                    subset = df[(df[f1] == level1) & (df[f2] == level2)]
                    if len(subset) > 0:
                        key = f"{f1}={level1}, {f2}={level2}"
                        interaction_effects[key] = round(float(subset[request.response].mean()), 4)

        # Calculate residuals and fitted values
        fitted_values = model.fittedvalues.values
        residuals = model.resid.values
        mse = np.mean(residuals**2)

        # Handle potential NaN in standardized residuals
        if mse > 0:
            standardized_residuals = residuals / np.sqrt(mse)
        else:
            standardized_residuals = residuals

        # Replace any NaN or inf with 0
        fitted_values = np.nan_to_num(fitted_values, nan=0.0, posinf=0.0, neginf=0.0)
        residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)
        standardized_residuals = np.nan_to_num(standardized_residuals, nan=0.0, posinf=0.0, neginf=0.0)

        # Prepare data for Pareto chart (absolute effects)
        effect_magnitudes = []
        for name, value in effects.items():
            effect_magnitudes.append({
                "name": name,
                "effect": value,
                "abs_effect": abs(value)
            })

        # Add interaction effects for 2-level designs
        if len(request.factors) >= 2:
            for i in range(len(request.factors)):
                for j in range(i+1, len(request.factors)):
                    if df[request.factors[i]].nunique() == 2 and df[request.factors[j]].nunique() == 2:
                        f1, f2 = request.factors[i], request.factors[j]
                        levels_f1 = sorted(df[f1].unique())
                        levels_f2 = sorted(df[f2].unique())

                        # Calculate interaction effect
                        high_high = df[(df[f1] == levels_f1[1]) & (df[f2] == levels_f2[1])][request.response].mean()
                        high_low = df[(df[f1] == levels_f1[1]) & (df[f2] == levels_f2[0])][request.response].mean()
                        low_high = df[(df[f1] == levels_f1[0]) & (df[f2] == levels_f2[1])][request.response].mean()
                        low_low = df[(df[f1] == levels_f1[0]) & (df[f2] == levels_f2[0])][request.response].mean()

                        # Check for NaN values
                        if pd.notna(high_high) and pd.notna(high_low) and pd.notna(low_high) and pd.notna(low_low):
                            interaction_effect = ((high_high + low_low) - (high_low + low_high)) / 2

                            effect_magnitudes.append({
                                "name": f"{f1} × {f2}",
                                "effect": float(interaction_effect),
                                "abs_effect": abs(float(interaction_effect))
                            })

        # Sort by absolute magnitude for Pareto chart
        effect_magnitudes.sort(key=lambda x: x['abs_effect'], reverse=True)

        # Main effects data for plotting
        main_effects_plot_data = {}
        for factor in request.factors:
            levels = sorted(df[factor].unique())
            means = [df[df[factor] == level][request.response].mean() for level in levels]

            # Filter out NaN values
            valid_means = [round(float(m), 4) if pd.notna(m) else 0.0 for m in means]

            main_effects_plot_data[factor] = {
                "levels": [str(l) for l in levels],
                "means": valid_means
            }

        # Generate interpretation and recommendations
        interpretation = generate_factorial_interpretation(
            results=results,
            factors=request.factors,
            alpha=request.alpha,
            r_squared=model.rsquared,
            adj_r_squared=model.rsquared_adj,
            has_replicates=has_replicates
        )

        # Cube plot data for 2^3 designs
        cube_data = None
        if len(request.factors) == 3 and all(df[f].nunique() == 2 for f in request.factors):
            cube_data = []
            for combo in product([0, 1], repeat=3):
                f1, f2, f3 = request.factors
                levels_f1 = sorted(df[f1].unique())
                levels_f2 = sorted(df[f2].unique())
                levels_f3 = sorted(df[f3].unique())

                subset = df[
                    (df[f1] == levels_f1[combo[0]]) &
                    (df[f2] == levels_f2[combo[1]]) &
                    (df[f3] == levels_f3[combo[2]])
                ]

                if len(subset) > 0:
                    mean_val = subset[request.response].mean()
                    if pd.notna(mean_val):
                        cube_data.append({
                            "x": combo[0],
                            "y": combo[1],
                            "z": combo[2],
                            "response": round(float(mean_val), 4),
                            "label": f"({levels_f1[combo[0]]}, {levels_f2[combo[1]]}, {levels_f3[combo[2]]})"
                        })

        return {
            "test_type": "Full Factorial Design Analysis",
            "n_factors": len(request.factors),
            "factors": request.factors,
            "alpha": request.alpha,
            "anova_table": results,
            "main_effects": effects,
            "interaction_means": interaction_effects if interaction_effects else None,
            "model_r_squared": round(float(model.rsquared), 4),
            "model_adj_r_squared": round(float(model.rsquared_adj), 4),
            "residuals": [round(float(r), 4) for r in residuals],
            "fitted_values": [round(float(f), 4) for f in fitted_values],
            "standardized_residuals": [round(float(r), 4) for r in standardized_residuals],
            "effect_magnitudes": effect_magnitudes,
            "main_effects_plot_data": main_effects_plot_data,
            "cube_data": cube_data,
            "response_name": request.response,
            "interpretation": interpretation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/three-level-factorial")
async def three_level_factorial_analysis(request: FactorialDesignRequest):
    """
    Analyze 3^k factorial design with quadratic (second-order) models
    Fits model including main effects, interactions, and quadratic terms
    """
    try:
        df = pd.DataFrame(request.data)

        # Rename columns to avoid conflicts
        column_mapping = {}
        for factor in request.factors:
            column_mapping[factor] = f"factor_{factor}"
        column_mapping[request.response] = f"response_{request.response}"

        df_renamed = df.rename(columns=column_mapping)

        # Build formula with quadratic terms
        renamed_factors = [f"factor_{f}" for f in request.factors]
        renamed_response = f"response_{request.response}"

        # For 3-level designs, include linear, quadratic, and interaction terms
        linear_terms = [f"C({f})" for f in renamed_factors]

        # For numeric coding, we need to convert categorical to numeric
        # Map Low/Medium/High to -1/0/1 for quadratic modeling
        for factor in request.factors:
            renamed_col = f"factor_{factor}"
            unique_vals = sorted(df_renamed[renamed_col].unique())

            # Create numeric mapping
            if len(unique_vals) == 3:
                mapping = {unique_vals[0]: -1, unique_vals[1]: 0, unique_vals[2]: 1}
                df_renamed[f"{renamed_col}_numeric"] = df_renamed[renamed_col].map(mapping)

        # Build quadratic formula
        numeric_factors = [f"factor_{f}_numeric" for f in request.factors]

        # Main effects (linear)
        main_effects = " + ".join(numeric_factors)

        # Quadratic terms
        quadratic_terms = " + ".join([f"I({f}**2)" for f in numeric_factors])

        # Two-way interactions
        interactions = []
        for i in range(len(numeric_factors)):
            for j in range(i+1, len(numeric_factors)):
                interactions.append(f"{numeric_factors[i]}:{numeric_factors[j]}")

        interaction_str = " + ".join(interactions) if interactions else ""

        if interaction_str:
            formula = f"{renamed_response} ~ {main_effects} + {quadratic_terms} + {interaction_str}"
        else:
            formula = f"{renamed_response} ~ {main_effects} + {quadratic_terms}"

        # Fit model
        model = ols(formula, data=df_renamed).fit()

        # Get ANOVA table
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Parse ANOVA results
        results = {}
        for idx, row in anova_table.iterrows():
            source = str(idx)

            # Clean up source names
            for factor in request.factors:
                source = source.replace(f"factor_{factor}_numeric", factor)
                source = source.replace(f"I(factor_{factor}_numeric ** 2)", f"{factor}²")

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

        if mse > 0:
            standardized_residuals = residuals / np.sqrt(mse)
        else:
            standardized_residuals = residuals

        # Replace any NaN or inf with 0
        fitted_values = np.nan_to_num(fitted_values, nan=0.0, posinf=0.0, neginf=0.0)
        residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)
        standardized_residuals = np.nan_to_num(standardized_residuals, nan=0.0, posinf=0.0, neginf=0.0)

        # Main effects plot data
        main_effects_plot_data = {}
        for factor in request.factors:
            levels = sorted(df[factor].unique())
            means = [df[df[factor] == level][request.response].mean() for level in levels]
            valid_means = [round(float(m), 4) if pd.notna(m) else 0.0 for m in means]

            main_effects_plot_data[factor] = {
                "levels": [str(l) for l in levels],
                "means": valid_means
            }

        # Generate interpretation and recommendations
        interpretation = generate_factorial_interpretation(
            results=results,
            factors=request.factors,
            alpha=request.alpha,
            r_squared=model.rsquared,
            adj_r_squared=model.rsquared_adj,
            has_replicates=False  # 3k designs typically don't have replicates in this implementation
        )

        return {
            "test_type": "3^k Full Factorial Design Analysis",
            "n_factors": len(request.factors),
            "factors": request.factors,
            "alpha": request.alpha,
            "anova_table": results,
            "model_r_squared": round(float(model.rsquared), 4),
            "model_adj_r_squared": round(float(model.rsquared_adj), 4),
            "residuals": [round(float(r), 4) for r in residuals],
            "fitted_values": [round(float(f), 4) for f in fitted_values],
            "standardized_residuals": [round(float(r), 4) for r in standardized_residuals],
            "main_effects_plot_data": main_effects_plot_data,
            "response_name": request.response,
            "model_summary": str(model.summary()),
            "interpretation": interpretation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/fractional-factorial/design")
async def generate_fractional_factorial(request: FractionalFactorialRequest):
    """
    Generate fractional factorial design with alias structure
    """
    try:
        from pyDOE2 import fracfact

        # Parse fraction
        k = request.factors

        # Generate design based on fraction
        if request.fraction == "1/2":
            p = 1
        elif request.fraction == "1/4":
            p = 2
        elif request.fraction == "1/8":
            p = 3
        else:
            raise ValueError(f"Unsupported fraction: {request.fraction}")

        # Create generator string
        if request.generator:
            gen = request.generator
        else:
            # Default generators for common designs
            if k == 4 and p == 1:
                gen = "a b c abc"  # 2^(4-1) design
            elif k == 5 and p == 1:
                gen = "a b c d abcd"  # 2^(5-1) design
            elif k == 5 and p == 2:
                gen = "a b c abc bcd"  # 2^(5-2) design
            else:
                raise ValueError("Please specify a generator for this design")

        # Generate design
        design = fracfact(gen)

        # Create factor names
        factor_names = [chr(65 + i) for i in range(k)]

        # Convert to DataFrame
        design_df = pd.DataFrame(design, columns=factor_names[:design.shape[1]])

        # Determine resolution
        resolution = "Unknown"
        if k == 4 and p == 1:
            resolution = "IV"
        elif k == 5 and p == 1:
            resolution = "V"
        elif k == 5 and p == 2:
            resolution = "III"

        return {
            "design_type": f"2^({k}-{p}) Fractional Factorial",
            "resolution": resolution,
            "n_runs": len(design),
            "n_factors": k,
            "fraction": request.fraction,
            "generator": gen,
            "design_matrix": design_df.to_dict('records'),
            "factor_names": factor_names[:design.shape[1]]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/effects/calculate")
async def calculate_effects(data: dict):
    """
    Calculate main effects and interaction effects for 2-level factorial designs
    """
    try:
        df = pd.DataFrame(data['data'])
        factors = data['factors']
        response = data['response']

        results = {
            "main_effects": {},
            "interaction_effects": {}
        }

        # Calculate main effects
        for factor in factors:
            high = df[df[factor] == 1][response].mean()
            low = df[df[factor] == -1][response].mean()
            effect = high - low
            results["main_effects"][factor] = round(float(effect), 4)

        # Calculate 2-way interactions
        for i in range(len(factors)):
            for j in range(i+1, len(factors)):
                f1, f2 = factors[i], factors[j]
                interaction_col = df[f1] * df[f2]
                high = df[interaction_col == 1][response].mean()
                low = df[interaction_col == -1][response].mean()
                effect = high - low
                results["interaction_effects"][f"{f1}*{f2}"] = round(float(effect), 4)

        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
