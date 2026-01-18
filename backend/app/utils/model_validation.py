"""
Model Validation Utilities for MasterStat

Provides comprehensive model validation methods including:
- PRESS statistic (Prediction Error Sum of Squares)
- K-fold cross-validation
- Model adequacy tests
- Validation metrics (R²_pred, AIC, BIC, RMSE, MAE)
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
from typing import Dict, List, Any, Optional


def calculate_press_statistic(model, data: pd.DataFrame, response: str) -> Dict[str, float]:
    """
    Calculate PRESS statistic (Prediction Error Sum of Squares).

    PRESS uses leave-one-out cross-validation to assess prediction error.
    PRESS = sum of squared prediction errors
    R²_prediction = 1 - PRESS/SST

    Args:
        model: Fitted statsmodels OLS model
        data: DataFrame containing the data
        response: Name of response variable

    Returns:
        Dictionary with PRESS, R²_prediction, and interpretation
    """
    try:
        # Get hat matrix diagonal (leverage values)
        influence = model.get_influence()
        hat_values = influence.hat_matrix_diag

        # Calculate PRESS residuals
        # PRESS residual_i = residual_i / (1 - h_i)
        residuals = model.resid
        press_residuals = residuals / (1 - hat_values)
        press = float(np.sum(press_residuals ** 2))

        # Calculate R² prediction
        sst = float(np.sum((data[response] - data[response].mean()) ** 2))
        r2_prediction = 1 - (press / sst)

        # Interpretation
        if r2_prediction < 0.5:
            interpretation = "Poor predictive ability"
            recommendation = "Model may not predict well on new data. Consider adding terms or collecting more data."
        elif r2_prediction < 0.7:
            interpretation = "Moderate predictive ability"
            recommendation = "Model has acceptable prediction performance."
        elif r2_prediction < 0.9:
            interpretation = "Good predictive ability"
            recommendation = "Model predicts well on new data."
        else:
            interpretation = "Excellent predictive ability"
            recommendation = "Model has very strong predictive performance."

        return {
            "press": round(press, 4),
            "r2_prediction": round(r2_prediction, 4),
            "sst": round(sst, 4),
            "interpretation": interpretation,
            "recommendation": recommendation
        }

    except Exception as e:
        return {
            "press": None,
            "r2_prediction": None,
            "sst": None,
            "interpretation": "Calculation failed",
            "recommendation": f"Error: {str(e)}"
        }


def k_fold_cross_validation(
    data: pd.DataFrame,
    formula: str,
    response: str,
    k_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform K-fold cross-validation for model validation.

    Args:
        data: DataFrame with experimental data
        formula: statsmodels formula string (e.g., "y ~ x1 + x2 + x1:x2")
        response: Name of response variable
        k_folds: Number of folds (default 5)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with fold-by-fold results and summary statistics
    """
    try:
        n = len(data)

        if k_folds < 2 or k_folds > n:
            return {
                "error": f"k_folds must be between 2 and {n}",
                "success": False
            }

        # K-fold cross-validation
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

        fold_scores = []
        all_predictions = []
        all_actuals = []

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(data)):
            try:
                # Split data
                df_train = data.iloc[train_idx]
                df_test = data.iloc[test_idx]

                # Fit model on training fold
                model = ols(formula, data=df_train).fit()

                # Predict on test fold
                y_test = df_test[response].values
                y_pred = model.predict(df_test)

                # Calculate metrics for this fold
                fold_r2 = r2_score(y_test, y_pred)
                fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                fold_mae = mean_absolute_error(y_test, y_pred)

                # Check for invalid values
                if np.isnan(fold_r2) or np.isinf(fold_r2):
                    raise ValueError("Invalid R² value - training set may be too small")

                fold_scores.append({
                    "fold": fold_idx + 1,
                    "r2": round(float(fold_r2), 4),
                    "rmse": round(float(fold_rmse), 4),
                    "mae": round(float(fold_mae), 4),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx)
                })

                # Store predictions for overall metrics
                for actual, pred in zip(y_test, y_pred):
                    all_predictions.append(float(pred))
                    all_actuals.append(float(actual))

            except Exception as fold_error:
                return {
                    "error": f"Fold {fold_idx + 1} failed: {str(fold_error)}",
                    "success": False
                }

        # Calculate summary statistics
        avg_r2 = float(np.mean([f["r2"] for f in fold_scores]))
        std_r2 = float(np.std([f["r2"] for f in fold_scores]))
        avg_rmse = float(np.mean([f["rmse"] for f in fold_scores]))
        std_rmse = float(np.std([f["rmse"] for f in fold_scores]))
        avg_mae = float(np.mean([f["mae"] for f in fold_scores]))
        std_mae = float(np.std([f["mae"] for f in fold_scores]))

        # Overall R² from all predictions
        overall_r2 = float(r2_score(all_actuals, all_predictions))

        # Interpretation
        if avg_r2 < 0.5:
            interpretation = "Poor cross-validation performance"
        elif avg_r2 < 0.7:
            interpretation = "Moderate cross-validation performance"
        elif avg_r2 < 0.9:
            interpretation = "Good cross-validation performance"
        else:
            interpretation = "Excellent cross-validation performance"

        return {
            "success": True,
            "k_folds": k_folds,
            "fold_results": fold_scores,
            "summary": {
                "avg_r2": round(avg_r2, 4),
                "std_r2": round(std_r2, 4),
                "avg_rmse": round(avg_rmse, 4),
                "std_rmse": round(std_rmse, 4),
                "avg_mae": round(avg_mae, 4),
                "std_mae": round(std_mae, 4),
                "overall_r2": round(overall_r2, 4)
            },
            "predictions": {
                "actual": all_actuals,
                "predicted": all_predictions
            },
            "interpretation": interpretation
        }

    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


def calculate_validation_metrics(model, data: pd.DataFrame, response: str) -> Dict[str, Any]:
    """
    Calculate comprehensive validation metrics for a fitted model.

    Metrics include:
    - R², R²_adjusted
    - R²_prediction (via PRESS)
    - AIC, BIC
    - RMSE, MAE
    - Standard error of regression

    Args:
        model: Fitted statsmodels OLS model
        data: DataFrame with data
        response: Response variable name

    Returns:
        Dictionary of validation metrics
    """
    try:
        n = len(data)
        p = len(model.params)  # Number of parameters

        # R-squared metrics
        r2 = float(model.rsquared)
        r2_adj = float(model.rsquared_adj)

        # PRESS and R²_prediction
        press_results = calculate_press_statistic(model, data, response)
        r2_pred = press_results.get("r2_prediction", None)
        press = press_results.get("press", None)

        # Information criteria
        aic = float(model.aic)
        bic = float(model.bic)

        # Prediction errors
        y_true = data[response].values
        y_pred = model.fittedvalues.values
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        # Standard error of regression
        ssr = float(np.sum(model.resid ** 2))
        s = np.sqrt(ssr / (n - p))

        # Mean squared error
        mse = float(model.mse_resid)

        return {
            "n_observations": n,
            "n_parameters": p,
            "r2": round(r2, 4),
            "r2_adjusted": round(r2_adj, 4),
            "r2_prediction": round(r2_pred, 4) if r2_pred is not None else None,
            "press": round(press, 4) if press is not None else None,
            "aic": round(aic, 2),
            "bic": round(bic, 2),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "std_error": round(s, 4),
            "interpretation": {
                "r2_quality": "Excellent" if r2 > 0.9 else "Good" if r2 > 0.7 else "Moderate" if r2 > 0.5 else "Poor",
                "prediction_quality": press_results.get("interpretation", "Unknown"),
                "note": "Lower AIC/BIC indicates better model fit. Compare models using these criteria."
            }
        }

    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


def assess_model_adequacy(model, data: pd.DataFrame, response: str, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Assess model adequacy using diagnostic tests.

    Tests include:
    - Normality of residuals (Shapiro-Wilk test)
    - Homoscedasticity (Breusch-Pagan test)
    - Autocorrelation (Durbin-Watson statistic)
    - Outliers (standardized residuals)

    Args:
        model: Fitted statsmodels OLS model
        data: DataFrame with data
        response: Response variable name
        alpha: Significance level for tests

    Returns:
        Dictionary with test results and diagnostics
    """
    try:
        residuals = model.resid
        n = len(residuals)

        # 1. Normality test (Shapiro-Wilk)
        shapiro_stat, shapiro_p = scipy_stats.shapiro(residuals)
        normality_pass = shapiro_p > alpha

        # 2. Homoscedasticity test (Breusch-Pagan)
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
            homoscedasticity_pass = bp_p > alpha
        except:
            bp_stat, bp_p = None, None
            homoscedasticity_pass = None

        # 3. Autocorrelation (Durbin-Watson)
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(residuals)
        # DW ~= 2 indicates no autocorrelation
        # DW < 1.5 or > 2.5 suggests autocorrelation
        autocorrelation_pass = 1.5 <= dw_stat <= 2.5

        # 4. Outlier detection (standardized residuals)
        std_residuals = residuals / np.std(residuals)
        outliers = np.abs(std_residuals) > 3
        n_outliers = int(np.sum(outliers))

        # 5. Leverage and influence
        influence = model.get_influence()
        hat_values = influence.hat_matrix_diag
        cooks_d = influence.cooks_distance[0]

        # High leverage threshold
        leverage_threshold = 2 * len(model.params) / n
        high_leverage = hat_values > leverage_threshold
        n_high_leverage = int(np.sum(high_leverage))

        # Influential points (Cook's D > 4/n)
        cooks_threshold = 4 / n
        influential = cooks_d > cooks_threshold
        n_influential = int(np.sum(influential))

        # Overall adequacy score (0-100)
        score = 0
        if normality_pass:
            score += 30
        elif shapiro_p > 0.01:  # Partial credit
            score += 15

        if homoscedasticity_pass:
            score += 30
        elif bp_p and bp_p > 0.01:  # Partial credit
            score += 15

        if autocorrelation_pass:
            score += 20
        elif 1.3 <= dw_stat <= 2.7:  # Partial credit
            score += 10

        if n_outliers == 0:
            score += 10
        elif n_outliers <= 2:
            score += 5

        if n_influential <= n * 0.05:  # < 5% influential
            score += 10
        elif n_influential <= n * 0.1:  # < 10% influential
            score += 5

        # Interpretation
        if score >= 80:
            overall = "Model assumptions satisfied - reliable for inference and prediction"
        elif score >= 60:
            overall = "Model adequate with minor violations - use with caution"
        elif score >= 40:
            overall = "Model has notable assumption violations - consider transformation or alternative model"
        else:
            overall = "Model inadequate - major assumption violations detected"

        # Recommendations
        recommendations = []
        if not normality_pass:
            recommendations.append("Non-normal residuals detected. Consider Box-Cox transformation or robust regression.")
        if not homoscedasticity_pass:
            recommendations.append("Heteroscedasticity detected. Consider weighted least squares or variance stabilizing transformation.")
        if not autocorrelation_pass:
            recommendations.append(f"Autocorrelation detected (DW={dw_stat:.2f}). Check for time trends or spatial correlation.")
        if n_outliers > 0:
            recommendations.append(f"Found {n_outliers} outliers. Investigate these observations.")
        if n_influential > n * 0.1:
            recommendations.append(f"Found {n_influential} influential observations (>10% of data). Consider robustness checks.")

        if not recommendations:
            recommendations.append("All diagnostic checks passed. Model assumptions satisfied.")

        return {
            "adequacy_score": score,
            "overall_assessment": overall,
            "tests": {
                "normality": {
                    "test": "Shapiro-Wilk",
                    "statistic": round(float(shapiro_stat), 4),
                    "p_value": round(float(shapiro_p), 4),
                    "pass": normality_pass,
                    "interpretation": "Normal" if normality_pass else "Non-normal"
                },
                "homoscedasticity": {
                    "test": "Breusch-Pagan",
                    "statistic": round(float(bp_stat), 4) if bp_stat is not None else None,
                    "p_value": round(float(bp_p), 4) if bp_p is not None else None,
                    "pass": homoscedasticity_pass,
                    "interpretation": "Homoscedastic" if homoscedasticity_pass else "Heteroscedastic"
                } if bp_stat is not None else None,
                "autocorrelation": {
                    "test": "Durbin-Watson",
                    "statistic": round(float(dw_stat), 4),
                    "pass": autocorrelation_pass,
                    "interpretation": "No autocorrelation" if autocorrelation_pass else "Autocorrelation present"
                }
            },
            "diagnostics": {
                "n_outliers": n_outliers,
                "n_high_leverage": n_high_leverage,
                "n_influential": n_influential,
                "outlier_threshold": 3.0,
                "leverage_threshold": round(leverage_threshold, 4),
                "influence_threshold": round(cooks_threshold, 4)
            },
            "recommendations": recommendations
        }

    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


def full_model_validation(
    model,
    data: pd.DataFrame,
    formula: str,
    response: str,
    k_folds: int = 5,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform comprehensive model validation combining all validation methods.

    Args:
        model: Fitted statsmodels OLS model
        data: DataFrame with data
        formula: Model formula string
        response: Response variable name
        k_folds: Number of folds for CV
        alpha: Significance level for tests

    Returns:
        Complete validation report
    """
    validation_report = {
        "metrics": calculate_validation_metrics(model, data, response),
        "press": calculate_press_statistic(model, data, response),
        "cross_validation": k_fold_cross_validation(data, formula, response, k_folds),
        "adequacy": assess_model_adequacy(model, data, response, alpha)
    }

    return validation_report
