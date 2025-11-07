from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm, ols

router = APIRouter()

class MixedModelRequest(BaseModel):
    data: List[Dict] = Field(..., description="Experimental data")
    fixed_factors: List[str] = Field(..., description="Fixed effect factor names")
    random_factors: List[str] = Field(..., description="Random effect factor names")
    response: str = Field(..., description="Response variable name")
    alpha: float = Field(0.05, description="Significance level")

class SplitPlotRequest(BaseModel):
    data: List[Dict] = Field(..., description="Experimental data")
    whole_plot_factor: str = Field(..., description="Whole-plot factor")
    subplot_factor: str = Field(..., description="Sub-plot factor")
    block: Optional[str] = Field(None, description="Block/replicate factor")
    response: str = Field(..., description="Response variable")
    alpha: float = Field(0.05, description="Significance level")

class NestedDesignRequest(BaseModel):
    data: List[Dict] = Field(..., description="Experimental data")
    factor_a: str = Field(..., description="Higher-level factor")
    factor_b_nested: str = Field(..., description="Factor nested within A")
    response: str = Field(..., description="Response variable")
    alpha: float = Field(0.05, description="Significance level")

@router.post("/mixed-model")
async def mixed_model_analysis(request: MixedModelRequest):
    """
    Analyze mixed model with fixed and random effects
    Calculate Expected Mean Squares (EMS) and variance components
    """
    try:
        df = pd.DataFrame(request.data)

        # For simple mixed model with one random factor
        if len(request.random_factors) == 1:
            random_factor = request.random_factors[0]

            # Build formula for fixed effects
            fixed_formula = " + ".join([f"C({f})" for f in request.fixed_factors])
            formula = f"{request.response} ~ {fixed_formula}"

            # Fit mixed model
            model = mixedlm(formula, df, groups=df[random_factor]).fit()

            # Calculate variance components
            variance_components = {
                "random_effect": round(float(model.cov_re.iloc[0, 0]), 4),
                "residual": round(float(model.scale), 4)
            }

            # Get fixed effects tests
            fixed_effects = {}
            for param in model.params.index:
                if param != 'Group Var':
                    fixed_effects[param] = {
                        "coefficient": round(float(model.params[param]), 4),
                        "std_error": round(float(model.bse[param]), 4),
                        "z_value": round(float(model.tvalues[param]), 4),
                        "p_value": round(float(model.pvalues[param]), 6)
                    }

            return {
                "model_type": "Mixed Model",
                "fixed_factors": request.fixed_factors,
                "random_factors": request.random_factors,
                "variance_components": variance_components,
                "fixed_effects": fixed_effects,
                "log_likelihood": round(float(model.llf), 4),
                "aic": round(float(model.aic), 4),
                "bic": round(float(model.bic), 4)
            }
        else:
            # For more complex models, use OLS with type 1 SS as approximation
            all_factors = request.fixed_factors + request.random_factors
            formula = f"{request.response} ~ " + " + ".join([f"C({f})" for f in all_factors])

            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=1)

            results = {}
            for idx, row in anova_table.iterrows():
                source = str(idx)
                for factor in all_factors:
                    source = source.replace(f'C({factor})', factor)

                results[source] = {
                    "sum_sq": round(float(row['sum_sq']), 4),
                    "df": int(row['df']),
                    "F": round(float(row['F']), 4) if not pd.isna(row['F']) else None,
                    "p_value": round(float(row['PR(>F)']), 6) if not pd.isna(row['PR(>F)']) else None
                }

            return {
                "model_type": "Mixed Model (OLS approximation)",
                "fixed_factors": request.fixed_factors,
                "random_factors": request.random_factors,
                "anova_table": results,
                "note": "For complex mixed models, consider using specialized software for exact EMS calculations"
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/split-plot")
async def split_plot_analysis(request: SplitPlotRequest):
    """
    Analyze split-plot design with whole-plot and sub-plot factors
    """
    try:
        df = pd.DataFrame(request.data)

        # Create a whole-plot identifier
        if request.block:
            df['whole_plot'] = df[request.block].astype(str) + "_" + df[request.whole_plot_factor].astype(str)
        else:
            df['whole_plot'] = df[request.whole_plot_factor].astype(str)

        # Build formula with whole-plot and sub-plot factors
        formula = f"{request.response} ~ C({request.whole_plot_factor}) * C({request.subplot_factor})"

        # Fit model treating whole_plot as random
        try:
            model = mixedlm(formula, df, groups=df['whole_plot']).fit()

            return {
                "model_type": "Split-Plot Design",
                "whole_plot_factor": request.whole_plot_factor,
                "subplot_factor": request.subplot_factor,
                "variance_components": {
                    "whole_plot_error": round(float(model.cov_re.iloc[0, 0]), 4),
                    "subplot_error": round(float(model.scale), 4)
                },
                "fixed_effects": {
                    param: {
                        "coefficient": round(float(model.params[param]), 4),
                        "p_value": round(float(model.pvalues[param]), 6)
                    }
                    for param in model.params.index if param != 'Group Var'
                },
                "aic": round(float(model.aic), 4)
            }
        except:
            # Fallback to OLS
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            results = {}
            for idx, row in anova_table.iterrows():
                source = str(idx)
                source = source.replace(f'C({request.whole_plot_factor})', request.whole_plot_factor)
                source = source.replace(f'C({request.subplot_factor})', request.subplot_factor)

                results[source] = {
                    "sum_sq": round(float(row['sum_sq']), 4),
                    "df": int(row['df']),
                    "F": round(float(row['F']), 4) if not pd.isna(row['F']) else None,
                    "p_value": round(float(row['PR(>F)']), 6) if not pd.isna(row['PR(>F)']) else None
                }

            return {
                "model_type": "Split-Plot Design (OLS approximation)",
                "whole_plot_factor": request.whole_plot_factor,
                "subplot_factor": request.subplot_factor,
                "anova_table": results
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/nested-design")
async def nested_design_analysis(request: NestedDesignRequest):
    """
    Analyze nested (hierarchical) design
    """
    try:
        df = pd.DataFrame(request.data)

        # For nested design: B is nested within A
        # Create interaction term that represents nesting
        df['nested_term'] = df[request.factor_a].astype(str) + "_" + df[request.factor_b_nested].astype(str)

        # Build formula
        formula = f"{request.response} ~ C({request.factor_a}) + C(nested_term)"

        # Fit model
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=1)

        # Calculate variance components
        results = {}
        ms_values = {}

        for idx, row in anova_table.iterrows():
            source = str(idx)
            source = source.replace(f'C({request.factor_a})', request.factor_a)
            source = source.replace('C(nested_term)', f'{request.factor_b_nested}({request.factor_a})')

            ms = row['sum_sq'] / row['df'] if row['df'] > 0 else 0
            ms_values[source] = ms

            results[source] = {
                "sum_sq": round(float(row['sum_sq']), 4),
                "df": int(row['df']),
                "ms": round(float(ms), 4),
                "F": round(float(row['F']), 4) if not pd.isna(row['F']) else None,
                "p_value": round(float(row['PR(>F)']), 6) if not pd.isna(row['PR(>F)']) else None
            }

        # Estimate variance components
        # For balanced nested design
        ms_error = ms_values.get('Residual', 0)
        ms_b_in_a = ms_values.get(f'{request.factor_b_nested}({request.factor_a})', 0)

        # Number of observations per B level
        n_per_b = df.groupby(['nested_term']).size().values[0] if len(df.groupby(['nested_term'])) > 0 else 1

        variance_components = {
            "sigma_squared_error": round(float(ms_error), 4),
            "sigma_squared_B_in_A": round(float((ms_b_in_a - ms_error) / n_per_b), 4) if n_per_b > 0 else None
        }

        return {
            "model_type": "Nested Design",
            "factor_a": request.factor_a,
            "factor_b_nested_in_a": request.factor_b_nested,
            "anova_table": results,
            "variance_components": variance_components,
            "model_r_squared": round(float(model.rsquared), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
