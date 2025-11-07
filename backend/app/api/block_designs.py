from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

router = APIRouter()

class RCBDRequest(BaseModel):
    data: List[Dict[str, float]] = Field(..., description="Experimental data")
    treatment: str = Field(..., description="Treatment factor name")
    block: str = Field(..., description="Block factor name")
    response: str = Field(..., description="Response variable name")
    alpha: float = Field(0.05, description="Significance level")

class LatinSquareRequest(BaseModel):
    data: List[Dict[str, float]] = Field(..., description="Experimental data")
    treatment: str = Field(..., description="Treatment factor name")
    row_block: str = Field(..., description="Row blocking factor")
    col_block: str = Field(..., description="Column blocking factor")
    response: str = Field(..., description="Response variable name")
    alpha: float = Field(0.05, description="Significance level")

@router.post("/rcbd")
async def rcbd_analysis(request: RCBDRequest):
    """
    Analyze Randomized Complete Block Design (RCBD)
    """
    try:
        df = pd.DataFrame(request.data)

        # Build formula: response ~ treatment + block
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

        # Calculate treatment means
        treatment_means = df.groupby(request.treatment)[request.response].mean().to_dict()
        block_means = df.groupby(request.block)[request.response].mean().to_dict()

        # Calculate relative efficiency compared to CRD
        n_treatments = df[request.treatment].nunique()
        n_blocks = df[request.block].nunique()

        ms_block = results.get(request.block, {}).get('sum_sq', 0) / results.get(request.block, {}).get('df', 1)
        ms_error = results.get('Residual', {}).get('sum_sq', 0) / results.get('Residual', {}).get('df', 1)

        # Relative efficiency = [(b-1)MS_block + b(t-1)MS_error] / [bt - 1)MS_error]
        # Simplified version
        if ms_error > 0:
            relative_efficiency = ((n_blocks - 1) * ms_block + n_blocks * (n_treatments - 1) * ms_error) / ((n_blocks * n_treatments - 1) * ms_error)
        else:
            relative_efficiency = None

        return {
            "test_type": "Randomized Complete Block Design (RCBD)",
            "alpha": request.alpha,
            "anova_table": results,
            "treatment_means": {str(k): round(float(v), 4) for k, v in treatment_means.items()},
            "block_means": {str(k): round(float(v), 4) for k, v in block_means.items()},
            "grand_mean": round(float(df[request.response].mean()), 4),
            "relative_efficiency": round(float(relative_efficiency), 4) if relative_efficiency else None,
            "model_r_squared": round(float(model.rsquared), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/latin-square")
async def latin_square_analysis(request: LatinSquareRequest):
    """
    Analyze Latin Square Design (two blocking factors)
    """
    try:
        df = pd.DataFrame(request.data)

        # Build formula: response ~ treatment + row_block + col_block
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

        # Calculate means
        treatment_means = df.groupby(request.treatment)[request.response].mean().to_dict()
        row_means = df.groupby(request.row_block)[request.response].mean().to_dict()
        col_means = df.groupby(request.col_block)[request.response].mean().to_dict()

        return {
            "test_type": "Latin Square Design",
            "alpha": request.alpha,
            "anova_table": results,
            "treatment_means": {str(k): round(float(v), 4) for k, v in treatment_means.items()},
            "row_block_means": {str(k): round(float(v), 4) for k, v in row_means.items()},
            "col_block_means": {str(k): round(float(v), 4) for k, v in col_means.items()},
            "grand_mean": round(float(df[request.response].mean()), 4),
            "model_r_squared": round(float(model.rsquared), 4)
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
