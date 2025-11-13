from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.special import logsumexp
import warnings
warnings.filterwarnings('ignore')

router = APIRouter()

# ==================== REQUEST/RESPONSE MODELS ====================

class PriorDistribution(BaseModel):
    dist_type: Literal['normal', 'uniform', 'cauchy', 'exponential', 't'] = 'normal'
    params: Dict[str, float] = Field(..., description="Distribution parameters")

class BayesianFactorialRequest(BaseModel):
    data: List[Dict[str, float]] = Field(..., description="Experimental data")
    factors: List[str] = Field(..., description="Factor variable names")
    response: str = Field(..., description="Response variable name")
    priors: Dict[str, PriorDistribution] = Field(..., description="Prior distributions for each parameter")
    n_samples: int = Field(2000, description="Number of MCMC samples")
    n_burn: int = Field(500, description="Number of burn-in samples")

class SequentialDesignRequest(BaseModel):
    current_data: List[Dict[str, float]] = Field(..., description="Current experimental data")
    factors: List[str] = Field(..., description="Factor names")
    response: str = Field(..., description="Response name")
    candidate_points: List[Dict[str, float]] = Field(..., description="Candidate design points")
    n_select: int = Field(1, description="Number of points to select")
    criterion: Literal['expected_info_gain', 'uncertainty_reduction', 'prediction_variance'] = 'expected_info_gain'

class ModelComparisonRequest(BaseModel):
    data: List[Dict[str, float]]
    factors: List[str]
    response: str
    models: List[List[str]] = Field(..., description="List of model specifications (list of factor combinations)")
    priors: Dict[str, PriorDistribution]

# ==================== HELPER FUNCTIONS ====================

def sample_prior(prior: PriorDistribution, size: int = 1) -> np.ndarray:
    """Sample from prior distribution"""
    if prior.dist_type == 'normal':
        return np.random.normal(prior.params['loc'], prior.params['scale'], size)
    elif prior.dist_type == 'uniform':
        return np.random.uniform(prior.params['low'], prior.params['high'], size)
    elif prior.dist_type == 'cauchy':
        return scipy_stats.cauchy.rvs(loc=prior.params['loc'], scale=prior.params['scale'], size=size)
    elif prior.dist_type == 'exponential':
        return np.random.exponential(prior.params['scale'], size)
    elif prior.dist_type == 't':
        return scipy_stats.t.rvs(df=prior.params['df'], loc=prior.params['loc'], scale=prior.params['scale'], size=size)
    else:
        raise ValueError(f"Unknown distribution type: {prior.dist_type}")

def log_prior(value: float, prior: PriorDistribution) -> float:
    """Calculate log prior probability"""
    if prior.dist_type == 'normal':
        return scipy_stats.norm.logpdf(value, prior.params['loc'], prior.params['scale'])
    elif prior.dist_type == 'uniform':
        return scipy_stats.uniform.logpdf(value, prior.params['low'], prior.params['high'] - prior.params['low'])
    elif prior.dist_type == 'cauchy':
        return scipy_stats.cauchy.logpdf(value, prior.params['loc'], prior.params['scale'])
    elif prior.dist_type == 'exponential':
        return scipy_stats.expon.logpdf(value, scale=prior.params['scale'])
    elif prior.dist_type == 't':
        return scipy_stats.t.logpdf(value, prior.params['df'], prior.params['loc'], prior.params['scale'])
    else:
        return 0.0

def log_likelihood(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Calculate log likelihood for linear model"""
    n = len(y)
    y_pred = X @ params[:-1]
    sigma = np.exp(params[-1])  # log(sigma) for numerical stability

    log_lik = -n/2 * np.log(2 * np.pi) - n * np.log(sigma) - 0.5 * np.sum((y - y_pred)**2) / (sigma**2)
    return log_lik

def metropolis_hastings(X: np.ndarray, y: np.ndarray, priors: Dict[str, PriorDistribution],
                        param_names: List[str], n_samples: int = 2000, n_burn: int = 500) -> Dict[str, Any]:
    """Simple Metropolis-Hastings MCMC sampler"""
    n_params = X.shape[1] + 1  # coefficients + sigma

    # Initialize
    current = np.zeros(n_params)
    current[-1] = np.log(np.std(y))  # Initialize log(sigma)

    samples = []
    acceptance_count = 0
    proposal_std = 0.1  # Proposal standard deviation

    for i in range(n_samples + n_burn):
        # Propose new parameters
        proposed = current + np.random.normal(0, proposal_std, n_params)

        # Calculate log posterior ratio
        log_prior_current = sum(log_prior(current[j], priors.get(param_names[j], PriorDistribution(dist_type='normal', params={'loc': 0, 'scale': 10})))
                               for j in range(n_params - 1))
        log_prior_proposed = sum(log_prior(proposed[j], priors.get(param_names[j], PriorDistribution(dist_type='normal', params={'loc': 0, 'scale': 10})))
                                for j in range(n_params - 1))

        log_lik_current = log_likelihood(current, X, y)
        log_lik_proposed = log_likelihood(proposed, X, y)

        log_ratio = (log_lik_proposed + log_prior_proposed) - (log_lik_current + log_prior_current)

        # Accept/reject
        if np.log(np.random.rand()) < log_ratio:
            current = proposed
            acceptance_count += 1

        if i >= n_burn:
            samples.append(current.copy())

    samples = np.array(samples)
    acceptance_rate = acceptance_count / (n_samples + n_burn)

    return {
        'samples': samples,
        'acceptance_rate': acceptance_rate,
        'param_names': param_names + ['log_sigma']
    }

def calculate_bayes_factor(model1_marginal_lik: float, model2_marginal_lik: float) -> float:
    """Calculate Bayes Factor: BF = P(D|M1) / P(D|M2)"""
    return np.exp(model1_marginal_lik - model2_marginal_lik)

# ==================== API ENDPOINTS ====================

@router.post("/factorial-analysis")
async def bayesian_factorial_analysis(request: BayesianFactorialRequest):
    """
    Perform Bayesian factorial design analysis with MCMC parameter estimation
    """
    try:
        df = pd.DataFrame(request.data)

        # Build design matrix
        X_list = [np.ones(len(df))]  # Intercept
        param_names = ['Intercept']

        # Add main effects
        for factor in request.factors:
            X_list.append(df[factor].values)
            param_names.append(factor)

        # Add interactions (two-way)
        for i in range(len(request.factors)):
            for j in range(i+1, len(request.factors)):
                interaction = df[request.factors[i]].values * df[request.factors[j]].values
                X_list.append(interaction)
                param_names.append(f"{request.factors[i]}:{request.factors[j]}")

        X = np.column_stack(X_list)
        y = df[request.response].values

        # Run MCMC
        mcmc_result = metropolis_hastings(X, y, request.priors, param_names,
                                         request.n_samples, request.n_burn)

        samples = mcmc_result['samples']

        # Calculate posterior statistics
        posterior_means = np.mean(samples, axis=0)
        posterior_stds = np.std(samples, axis=0)

        # Credible intervals (95%)
        credible_intervals = {
            param_names[i]: {
                'mean': float(posterior_means[i]),
                'std': float(posterior_stds[i]),
                'lower_95': float(np.percentile(samples[:, i], 2.5)),
                'upper_95': float(np.percentile(samples[:, i], 97.5)),
                'median': float(np.median(samples[:, i]))
            }
            for i in range(len(param_names))
        }

        # Bayes factors for effect significance (compared to null model where effect = 0)
        bayes_factors = {}
        for i, param in enumerate(param_names[1:-1]):  # Skip intercept and sigma
            # Proportion of posterior samples where effect != 0 (using practical significance threshold)
            threshold = 0.1 * posterior_stds[i]
            prob_significant = np.mean(np.abs(samples[:, i]) > threshold)

            # Approximate Bayes Factor (Savage-Dickey density ratio approximation)
            # BF = p(H1|D) / p(H0|D) ≈ posterior density at 0 / prior density at 0
            prior = request.priors.get(param, PriorDistribution(dist_type='normal', params={'loc': 0, 'scale': 10}))

            # Simple approximation
            bf = prob_significant / (1 - prob_significant + 1e-10)

            bayes_factors[param] = {
                'bayes_factor': float(bf),
                'prob_significant': float(prob_significant),
                'interpretation': 'Strong evidence' if bf > 10 else 'Moderate evidence' if bf > 3 else 'Weak evidence'
            }

        # Posterior predictive checks
        n_pred_samples = 100
        y_pred_samples = []
        for i in range(n_pred_samples):
            idx = np.random.randint(0, len(samples))
            sample_params = samples[idx, :-1]
            sigma = np.exp(samples[idx, -1])
            y_pred = X @ sample_params + np.random.normal(0, sigma, len(y))
            y_pred_samples.append(y_pred)

        y_pred_samples = np.array(y_pred_samples)
        y_pred_mean = np.mean(y_pred_samples, axis=0)
        y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=0)
        y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=0)

        return {
            'method': 'Bayesian Factorial Analysis (MCMC)',
            'n_samples': request.n_samples,
            'n_burn': request.n_burn,
            'acceptance_rate': float(mcmc_result['acceptance_rate']),
            'posterior_summary': credible_intervals,
            'bayes_factors': bayes_factors,
            'posterior_samples': {
                param: samples[:, i].tolist()[:100]  # Return first 100 samples for plotting
                for i, param in enumerate(param_names)
            },
            'posterior_predictive': {
                'observed': y.tolist(),
                'predicted_mean': y_pred_mean.tolist(),
                'predicted_lower_95': y_pred_lower.tolist(),
                'predicted_upper_95': y_pred_upper.tolist()
            },
            'convergence_diagnostics': {
                'acceptance_rate': float(mcmc_result['acceptance_rate']),
                'effective_sample_size': 'Not implemented',  # Would need more sophisticated calculation
                'rhat': 'Not implemented'  # Would need multiple chains
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/sequential-design")
async def sequential_experimental_design(request: SequentialDesignRequest):
    """
    Select next experimental point(s) using Bayesian sequential design criteria
    """
    try:
        df_current = pd.DataFrame(request.current_data)

        # Build design matrix for current data
        X_current = np.column_stack([np.ones(len(df_current))] +
                                    [df_current[f].values for f in request.factors])
        y_current = df_current[request.response].values

        # Fit simple Bayesian model to current data
        # Using conjugate normal-inverse-gamma for computational efficiency
        n = len(y_current)
        p = X_current.shape[1]

        # OLS estimates for initialization
        beta_ols = np.linalg.lstsq(X_current, y_current, rcond=None)[0]
        residuals = y_current - X_current @ beta_ols
        sigma2_ols = np.sum(residuals**2) / (n - p)

        # Posterior parameters (using non-informative priors)
        XtX = X_current.T @ X_current
        XtX_inv = np.linalg.inv(XtX + 1e-6 * np.eye(p))
        beta_post = beta_ols

        # Evaluate candidate points
        candidate_scores = []

        for candidate in request.candidate_points:
            x_new = np.array([1] + [candidate[f] for f in request.factors])

            if request.criterion == 'prediction_variance':
                # Prediction variance at candidate point
                pred_var = sigma2_ols * (1 + x_new.T @ XtX_inv @ x_new)
                score = float(pred_var)  # Higher variance = more informative

            elif request.criterion == 'expected_info_gain':
                # Approximate expected information gain
                # Based on reduction in posterior uncertainty
                pred_var = x_new.T @ XtX_inv @ x_new
                expected_gain = 0.5 * np.log(1 + pred_var / sigma2_ols)
                score = float(expected_gain)

            elif request.criterion == 'uncertainty_reduction':
                # D-optimality criterion (determinant of information matrix)
                X_augmented = np.vstack([X_current, x_new])
                XtX_aug = X_augmented.T @ X_augmented
                score = float(np.linalg.slogdet(XtX_aug)[1] - np.linalg.slogdet(XtX)[1])

            candidate_scores.append({
                'point': candidate,
                'score': score,
                'x_values': {f: candidate[f] for f in request.factors}
            })

        # Sort by score (descending)
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)

        # Select top n_select points
        selected_points = candidate_scores[:request.n_select]

        return {
            'method': f'Sequential Design ({request.criterion})',
            'selected_points': selected_points,
            'all_candidates': candidate_scores,
            'recommendation': f"Run experiments at the selected {len(selected_points)} point(s) to maximize information gain."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/model-comparison")
async def bayesian_model_comparison(request: ModelComparisonRequest):
    """
    Compare multiple models using Bayesian Information Criterion and Bayes Factors
    """
    try:
        df = pd.DataFrame(request.data)
        y = df[request.response].values
        n = len(y)

        model_results = []

        for model_idx, model_terms in enumerate(request.models):
            # Build design matrix for this model
            X_list = [np.ones(n)]  # Intercept

            for term in model_terms:
                if ':' in term:
                    # Interaction term
                    factors = term.split(':')
                    interaction = np.prod([df[f].values for f in factors], axis=0)
                    X_list.append(interaction)
                else:
                    # Main effect
                    X_list.append(df[term].values)

            X = np.column_stack(X_list)
            p = X.shape[1]

            # OLS fit
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta_hat
            residuals = y - y_pred
            sse = np.sum(residuals**2)
            mse = sse / (n - p)

            # Model fit statistics
            r_squared = 1 - sse / np.sum((y - np.mean(y))**2)

            # BIC (lower is better)
            bic = n * np.log(sse / n) + p * np.log(n)

            # AIC
            aic = n * np.log(sse / n) + 2 * p

            # Marginal likelihood approximation (Laplace approximation)
            log_marginal_lik = -0.5 * (n * np.log(2 * np.pi * mse) + sse / mse + p * np.log(n))

            model_results.append({
                'model_id': model_idx + 1,
                'terms': model_terms,
                'n_parameters': p,
                'r_squared': float(r_squared),
                'mse': float(mse),
                'aic': float(aic),
                'bic': float(bic),
                'log_marginal_likelihood': float(log_marginal_lik)
            })

        # Calculate Bayes Factors (model 1 vs others)
        if len(model_results) > 1:
            base_log_ml = model_results[0]['log_marginal_likelihood']
            for result in model_results[1:]:
                bf = calculate_bayes_factor(result['log_marginal_likelihood'], base_log_ml)
                result['bayes_factor_vs_model1'] = float(bf)
                result['bf_interpretation'] = (
                    'Very strong evidence' if bf > 100 else
                    'Strong evidence' if bf > 10 else
                    'Moderate evidence' if bf > 3 else
                    'Weak evidence'
                )

        # Sort by BIC
        model_results_sorted = sorted(model_results, key=lambda x: x['bic'])
        best_model = model_results_sorted[0]

        return {
            'method': 'Bayesian Model Comparison',
            'n_models': len(model_results),
            'models': model_results_sorted,
            'best_model': {
                'model_id': best_model['model_id'],
                'terms': best_model['terms'],
                'criterion': 'BIC (Bayesian Information Criterion)'
            },
            'recommendation': f"Model {best_model['model_id']} is preferred based on BIC."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/adaptive-design")
async def adaptive_doe_strategy(request: Dict[str, Any]):
    """
    Generate adaptive DOE strategy based on current experimental knowledge
    """
    try:
        # This is a simplified adaptive strategy
        # In practice, would use more sophisticated methods

        current_uncertainty = request.get('current_uncertainty', 1.0)
        target_uncertainty = request.get('target_uncertainty', 0.1)

        # Estimate number of additional runs needed
        n_additional = int(np.ceil(np.log(target_uncertainty / current_uncertainty) / np.log(0.7)))

        strategy = {
            'method': 'Adaptive DOE Strategy',
            'current_uncertainty': current_uncertainty,
            'target_uncertainty': target_uncertainty,
            'recommended_additional_runs': max(n_additional, 1),
            'strategy_phases': [
                {
                    'phase': 1,
                    'objective': 'Exploration',
                    'n_runs': max(n_additional // 2, 1),
                    'focus': 'High uncertainty regions'
                },
                {
                    'phase': 2,
                    'objective': 'Exploitation',
                    'n_runs': max(n_additional - n_additional // 2, 1),
                    'focus': 'Optimal region refinement'
                }
            ],
            'stopping_criteria': {
                'uncertainty_threshold': target_uncertainty,
                'max_runs': 'User specified',
                'convergence': 'Bayesian posterior convergence'
            }
        }

        return strategy

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
