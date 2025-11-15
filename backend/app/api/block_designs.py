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

class RCBDRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Experimental data")
    treatment: str = Field(..., description="Treatment factor name")
    block: str = Field(..., description="Block factor name")
    response: str = Field(..., description="Response variable name")
    alpha: float = Field(0.05, description="Significance level")
    random_blocks: bool = Field(False, description="Treat blocks as random effects")

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

@router.post("/rcbd")
async def rcbd_analysis(request: RCBDRequest):
    """
    Analyze Randomized Complete Block Design (RCBD)
    Supports both fixed and random blocks
    """
    try:
        df = pd.DataFrame(request.data)

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
                "interaction_means": interaction_means
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
