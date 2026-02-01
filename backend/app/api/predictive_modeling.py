"""
Predictive Modeling Suite
Decision Trees, Random Forest, Gradient Boosting, Regularized Regression
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Literal
import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/api/ml", tags=["Predictive Modeling"])

# ============================================================================
# Pydantic Models
# ============================================================================

class PredictiveModelRequest(BaseModel):
    X: List[List[float]] = Field(..., description="Feature matrix (n_samples x n_features)")
    y: List[Union[float, str, int]] = Field(..., description="Target variable")
    feature_names: Optional[List[str]] = Field(None, description="Names of features")
    problem_type: Literal["regression", "classification"] = Field("regression", description="Type of problem")
    test_size: float = Field(0.2, description="Fraction of data for testing (0-0.5)")
    random_state: int = Field(42, description="Random seed for reproducibility")

class DecisionTreeRequest(PredictiveModelRequest):
    max_depth: Optional[int] = Field(None, description="Maximum depth of tree (None for unlimited)")
    min_samples_split: int = Field(2, description="Minimum samples required to split")
    min_samples_leaf: int = Field(1, description="Minimum samples required in leaf")
    criterion: Optional[str] = Field(None, description="Split criterion (gini/entropy for classification, squared_error/absolute_error for regression)")

class RandomForestRequest(PredictiveModelRequest):
    n_estimators: int = Field(100, description="Number of trees")
    max_depth: Optional[int] = Field(None, description="Maximum depth of trees")
    min_samples_split: int = Field(2, description="Minimum samples to split")
    min_samples_leaf: int = Field(1, description="Minimum samples in leaf")
    max_features: Optional[Union[str, float]] = Field("sqrt", description="Features to consider at each split")
    bootstrap: bool = Field(True, description="Use bootstrap samples")

class GradientBoostingRequest(PredictiveModelRequest):
    n_estimators: int = Field(100, description="Number of boosting stages")
    learning_rate: float = Field(0.1, description="Learning rate (shrinkage)")
    max_depth: int = Field(3, description="Maximum depth of trees")
    min_samples_split: int = Field(2, description="Minimum samples to split")
    min_samples_leaf: int = Field(1, description="Minimum samples in leaf")
    subsample: float = Field(1.0, description="Fraction of samples for fitting trees")

class RegularizedRegressionRequest(PredictiveModelRequest):
    method: Literal["ridge", "lasso", "elasticnet"] = Field("ridge", description="Regularization method")
    alpha: float = Field(1.0, description="Regularization strength")
    l1_ratio: float = Field(0.5, description="L1 ratio for ElasticNet (0=Ridge, 1=Lasso)")
    standardize: bool = Field(True, description="Standardize features before fitting")

class ModelScreeningRequest(PredictiveModelRequest):
    methods: List[str] = Field(
        ["decision_tree", "random_forest", "gradient_boosting", "ridge", "lasso"],
        description="Methods to compare"
    )
    cv_folds: int = Field(5, description="Number of cross-validation folds")

# ============================================================================
# Helper Functions
# ============================================================================

def safe_float(val):
    """Convert value to JSON-safe float"""
    if val is None:
        return None
    if isinstance(val, (np.floating, np.integer)):
        val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val

def make_json_safe(obj):
    """Recursively make object JSON-safe"""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    elif isinstance(obj, (np.floating, np.integer)):
        return safe_float(float(obj))
    elif isinstance(obj, float):
        return safe_float(obj)
    return obj

def get_regression_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Adjusted R2
    n = len(y_true)
    p = 1  # Will be updated by caller if needed
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    return {
        "mse": safe_float(mse),
        "rmse": safe_float(rmse),
        "mae": safe_float(mae),
        "r2": safe_float(r2),
        "adjusted_r2": safe_float(adj_r2)
    }

def get_classification_metrics(y_true, y_pred, y_prob=None, classes=None):
    """Calculate classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)

    # Handle binary vs multiclass
    n_classes = len(np.unique(y_true))
    average = 'binary' if n_classes == 2 else 'weighted'

    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": safe_float(accuracy),
        "precision": safe_float(precision),
        "recall": safe_float(recall),
        "f1_score": safe_float(f1),
        "confusion_matrix": cm.tolist()
    }

    # AUC-ROC for binary classification with probabilities
    if y_prob is not None and n_classes == 2:
        try:
            if y_prob.ndim == 2:
                y_prob_positive = y_prob[:, 1]
            else:
                y_prob_positive = y_prob
            auc = roc_auc_score(y_true, y_prob_positive)
            fpr, tpr, thresholds = roc_curve(y_true, y_prob_positive)
            metrics["auc_roc"] = safe_float(auc)
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            }
        except Exception:
            pass

    if classes is not None:
        metrics["classes"] = classes.tolist() if hasattr(classes, 'tolist') else list(classes)

    return metrics

def extract_tree_structure(tree, feature_names=None, max_depth=5):
    """Extract tree structure for visualization"""
    tree_ = tree.tree_

    if feature_names is None:
        feature_names = [f"X{i}" for i in range(tree_.n_features)]

    def recurse(node, depth=0):
        if depth > max_depth:
            return {"type": "truncated", "depth": depth}

        if tree_.feature[node] != -2:  # Not a leaf
            feature_idx = tree_.feature[node]
            threshold = tree_.threshold[node]

            return {
                "type": "split",
                "feature": feature_names[feature_idx] if feature_idx < len(feature_names) else f"X{feature_idx}",
                "feature_index": int(feature_idx),
                "threshold": safe_float(threshold),
                "samples": int(tree_.n_node_samples[node]),
                "impurity": safe_float(tree_.impurity[node]),
                "left": recurse(tree_.children_left[node], depth + 1),
                "right": recurse(tree_.children_right[node], depth + 1)
            }
        else:  # Leaf node
            value = tree_.value[node]
            if value.shape[1] == 1:  # Regression
                prediction = value[0, 0]
            else:  # Classification
                prediction = int(np.argmax(value[0]))

            return {
                "type": "leaf",
                "samples": int(tree_.n_node_samples[node]),
                "value": value.tolist(),
                "prediction": safe_float(prediction) if isinstance(prediction, float) else prediction
            }

    return recurse(0)

def get_feature_importance(model, feature_names):
    """Extract feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        return None

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    return {
        "features": [feature_names[i] for i in indices],
        "importances": [safe_float(importances[i]) for i in indices],
        "indices": indices.tolist()
    }

# ============================================================================
# Model Training Functions
# ============================================================================

def train_decision_tree(X_train, X_test, y_train, y_test, params, problem_type, feature_names):
    """Train decision tree model"""
    if problem_type == "classification":
        criterion = params.get('criterion', 'gini')
        model = DecisionTreeClassifier(
            max_depth=params.get('max_depth'),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            criterion=criterion,
            random_state=params.get('random_state', 42)
        )
    else:
        criterion = params.get('criterion', 'squared_error')
        model = DecisionTreeRegressor(
            max_depth=params.get('max_depth'),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            criterion=criterion,
            random_state=params.get('random_state', 42)
        )

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    result = {
        "model_type": "decision_tree",
        "problem_type": problem_type,
        "parameters": {
            "max_depth": model.get_depth(),
            "n_leaves": model.get_n_leaves(),
            "min_samples_split": params.get('min_samples_split', 2),
            "min_samples_leaf": params.get('min_samples_leaf', 1),
            "criterion": criterion
        }
    }

    if problem_type == "classification":
        y_prob = model.predict_proba(X_test)
        result["train_metrics"] = get_classification_metrics(y_train, y_pred_train)
        result["test_metrics"] = get_classification_metrics(y_test, y_pred_test, y_prob, model.classes_)
    else:
        result["train_metrics"] = get_regression_metrics(y_train, y_pred_train)
        result["test_metrics"] = get_regression_metrics(y_test, y_pred_test)

    result["feature_importance"] = get_feature_importance(model, feature_names)
    result["tree_structure"] = extract_tree_structure(model, feature_names)

    return result

def train_random_forest(X_train, X_test, y_train, y_test, params, problem_type, feature_names):
    """Train random forest model"""
    if problem_type == "classification":
        model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth'),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            max_features=params.get('max_features', 'sqrt'),
            bootstrap=params.get('bootstrap', True),
            random_state=params.get('random_state', 42),
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth'),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            max_features=params.get('max_features', 'sqrt'),
            bootstrap=params.get('bootstrap', True),
            random_state=params.get('random_state', 42),
            n_jobs=-1
        )

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    result = {
        "model_type": "random_forest",
        "problem_type": problem_type,
        "parameters": {
            "n_estimators": params.get('n_estimators', 100),
            "max_depth": params.get('max_depth'),
            "min_samples_split": params.get('min_samples_split', 2),
            "min_samples_leaf": params.get('min_samples_leaf', 1),
            "max_features": params.get('max_features', 'sqrt'),
            "bootstrap": params.get('bootstrap', True)
        }
    }

    if problem_type == "classification":
        y_prob = model.predict_proba(X_test)
        result["train_metrics"] = get_classification_metrics(y_train, y_pred_train)
        result["test_metrics"] = get_classification_metrics(y_test, y_pred_test, y_prob, model.classes_)
    else:
        result["train_metrics"] = get_regression_metrics(y_train, y_pred_train)
        result["test_metrics"] = get_regression_metrics(y_test, y_pred_test)

    result["feature_importance"] = get_feature_importance(model, feature_names)

    # OOB score if bootstrap
    if params.get('bootstrap', True) and hasattr(model, 'oob_score_'):
        model.oob_score = True
        result["oob_score"] = safe_float(model.oob_score_)

    return result

def train_gradient_boosting(X_train, X_test, y_train, y_test, params, problem_type, feature_names):
    """Train gradient boosting model"""
    if problem_type == "classification":
        model = GradientBoostingClassifier(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 3),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            subsample=params.get('subsample', 1.0),
            random_state=params.get('random_state', 42)
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 3),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            subsample=params.get('subsample', 1.0),
            random_state=params.get('random_state', 42)
        )

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    result = {
        "model_type": "gradient_boosting",
        "problem_type": problem_type,
        "parameters": {
            "n_estimators": params.get('n_estimators', 100),
            "learning_rate": params.get('learning_rate', 0.1),
            "max_depth": params.get('max_depth', 3),
            "min_samples_split": params.get('min_samples_split', 2),
            "min_samples_leaf": params.get('min_samples_leaf', 1),
            "subsample": params.get('subsample', 1.0)
        }
    }

    if problem_type == "classification":
        y_prob = model.predict_proba(X_test)
        result["train_metrics"] = get_classification_metrics(y_train, y_pred_train)
        result["test_metrics"] = get_classification_metrics(y_test, y_pred_test, y_prob, model.classes_)
    else:
        result["train_metrics"] = get_regression_metrics(y_train, y_pred_train)
        result["test_metrics"] = get_regression_metrics(y_test, y_pred_test)

    result["feature_importance"] = get_feature_importance(model, feature_names)

    # Training deviance curve
    if hasattr(model, 'train_score_'):
        result["training_curve"] = {
            "train_score": [safe_float(s) for s in model.train_score_]
        }

    return result

def train_regularized_regression(X_train, X_test, y_train, y_test, params, problem_type, feature_names):
    """Train regularized regression model"""
    standardize = params.get('standardize', True)

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    method = params.get('method', 'ridge')
    alpha = params.get('alpha', 1.0)

    if problem_type == "classification":
        # Use logistic regression with regularization
        if method == 'ridge':
            penalty = 'l2'
        elif method == 'lasso':
            penalty = 'l1'
        else:  # elasticnet
            penalty = 'elasticnet'

        solver = 'saga' if penalty in ['l1', 'elasticnet'] else 'lbfgs'

        model = LogisticRegression(
            penalty=penalty,
            C=1.0 / alpha,  # sklearn uses inverse regularization
            l1_ratio=params.get('l1_ratio', 0.5) if penalty == 'elasticnet' else None,
            solver=solver,
            max_iter=1000,
            random_state=params.get('random_state', 42)
        )
    else:
        if method == 'ridge':
            model = Ridge(alpha=alpha, random_state=params.get('random_state', 42))
        elif method == 'lasso':
            model = Lasso(alpha=alpha, random_state=params.get('random_state', 42), max_iter=10000)
        else:  # elasticnet
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=params.get('l1_ratio', 0.5),
                random_state=params.get('random_state', 42),
                max_iter=10000
            )

    model.fit(X_train_scaled, y_train)
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    result = {
        "model_type": f"regularized_{method}",
        "problem_type": problem_type,
        "parameters": {
            "method": method,
            "alpha": alpha,
            "standardize": standardize
        }
    }

    if method == 'elasticnet':
        result["parameters"]["l1_ratio"] = params.get('l1_ratio', 0.5)

    if problem_type == "classification":
        y_prob = model.predict_proba(X_test_scaled)
        result["train_metrics"] = get_classification_metrics(y_train, y_pred_train)
        result["test_metrics"] = get_classification_metrics(y_test, y_pred_test, y_prob, model.classes_)
    else:
        result["train_metrics"] = get_regression_metrics(y_train, y_pred_train)
        result["test_metrics"] = get_regression_metrics(y_test, y_pred_test)

    # Coefficients
    coef = model.coef_.flatten() if hasattr(model, 'coef_') else None
    if coef is not None:
        # Sort by absolute magnitude
        sorted_indices = np.argsort(np.abs(coef))[::-1]
        result["coefficients"] = {
            "features": [feature_names[i] for i in sorted_indices],
            "values": [safe_float(coef[i]) for i in sorted_indices],
            "abs_values": [safe_float(abs(coef[i])) for i in sorted_indices]
        }

        # Number of non-zero coefficients (for Lasso/ElasticNet)
        result["n_nonzero_coef"] = int(np.sum(np.abs(coef) > 1e-10))

    if hasattr(model, 'intercept_'):
        result["intercept"] = safe_float(model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_)

    return result

# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/decision-tree")
async def fit_decision_tree(request: DecisionTreeRequest):
    """
    Fit a Decision Tree model (CART algorithm).
    Provides interpretable tree structure and feature importance.
    """
    try:
        X = np.array(request.X)
        y = np.array(request.y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if len(X) < 10:
            raise ValueError("Need at least 10 samples")

        n_features = X.shape[1]
        feature_names = request.feature_names or [f"X{i+1}" for i in range(n_features)]

        if len(feature_names) != n_features:
            feature_names = [f"X{i+1}" for i in range(n_features)]

        # Handle classification with string labels
        label_encoder = None
        if request.problem_type == "classification" and y.dtype == object:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=request.random_state
        )

        params = {
            'max_depth': request.max_depth,
            'min_samples_split': request.min_samples_split,
            'min_samples_leaf': request.min_samples_leaf,
            'criterion': request.criterion,
            'random_state': request.random_state
        }

        result = train_decision_tree(X_train, X_test, y_train, y_test, params, request.problem_type, feature_names)

        result["data_info"] = {
            "n_samples": len(X),
            "n_features": n_features,
            "n_train": len(X_train),
            "n_test": len(X_test)
        }

        if label_encoder is not None:
            result["class_labels"] = label_encoder.classes_.tolist()

        return make_json_safe(result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/random-forest")
async def fit_random_forest(request: RandomForestRequest):
    """
    Fit a Random Forest model.
    Ensemble of decision trees with bootstrap sampling and feature randomization.
    """
    try:
        X = np.array(request.X)
        y = np.array(request.y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if len(X) < 10:
            raise ValueError("Need at least 10 samples")

        n_features = X.shape[1]
        feature_names = request.feature_names or [f"X{i+1}" for i in range(n_features)]

        if len(feature_names) != n_features:
            feature_names = [f"X{i+1}" for i in range(n_features)]

        # Handle classification with string labels
        label_encoder = None
        if request.problem_type == "classification" and y.dtype == object:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=request.random_state
        )

        params = {
            'n_estimators': request.n_estimators,
            'max_depth': request.max_depth,
            'min_samples_split': request.min_samples_split,
            'min_samples_leaf': request.min_samples_leaf,
            'max_features': request.max_features,
            'bootstrap': request.bootstrap,
            'random_state': request.random_state
        }

        result = train_random_forest(X_train, X_test, y_train, y_test, params, request.problem_type, feature_names)

        result["data_info"] = {
            "n_samples": len(X),
            "n_features": n_features,
            "n_train": len(X_train),
            "n_test": len(X_test)
        }

        if label_encoder is not None:
            result["class_labels"] = label_encoder.classes_.tolist()

        return make_json_safe(result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/gradient-boosting")
async def fit_gradient_boosting(request: GradientBoostingRequest):
    """
    Fit a Gradient Boosting model.
    Sequential ensemble that corrects errors of previous trees.
    """
    try:
        X = np.array(request.X)
        y = np.array(request.y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if len(X) < 10:
            raise ValueError("Need at least 10 samples")

        n_features = X.shape[1]
        feature_names = request.feature_names or [f"X{i+1}" for i in range(n_features)]

        if len(feature_names) != n_features:
            feature_names = [f"X{i+1}" for i in range(n_features)]

        # Handle classification with string labels
        label_encoder = None
        if request.problem_type == "classification" and y.dtype == object:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=request.random_state
        )

        params = {
            'n_estimators': request.n_estimators,
            'learning_rate': request.learning_rate,
            'max_depth': request.max_depth,
            'min_samples_split': request.min_samples_split,
            'min_samples_leaf': request.min_samples_leaf,
            'subsample': request.subsample,
            'random_state': request.random_state
        }

        result = train_gradient_boosting(X_train, X_test, y_train, y_test, params, request.problem_type, feature_names)

        result["data_info"] = {
            "n_samples": len(X),
            "n_features": n_features,
            "n_train": len(X_train),
            "n_test": len(X_test)
        }

        if label_encoder is not None:
            result["class_labels"] = label_encoder.classes_.tolist()

        return make_json_safe(result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/regularized-regression")
async def fit_regularized_regression(request: RegularizedRegressionRequest):
    """
    Fit a regularized regression model (Ridge, Lasso, or ElasticNet).
    Useful for feature selection and preventing overfitting.
    """
    try:
        X = np.array(request.X)
        y = np.array(request.y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if len(X) < 10:
            raise ValueError("Need at least 10 samples")

        n_features = X.shape[1]
        feature_names = request.feature_names or [f"X{i+1}" for i in range(n_features)]

        if len(feature_names) != n_features:
            feature_names = [f"X{i+1}" for i in range(n_features)]

        # Handle classification with string labels
        label_encoder = None
        if request.problem_type == "classification" and y.dtype == object:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=request.random_state
        )

        params = {
            'method': request.method,
            'alpha': request.alpha,
            'l1_ratio': request.l1_ratio,
            'standardize': request.standardize,
            'random_state': request.random_state
        }

        result = train_regularized_regression(X_train, X_test, y_train, y_test, params, request.problem_type, feature_names)

        result["data_info"] = {
            "n_samples": len(X),
            "n_features": n_features,
            "n_train": len(X_train),
            "n_test": len(X_test)
        }

        if label_encoder is not None:
            result["class_labels"] = label_encoder.classes_.tolist()

        return make_json_safe(result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/model-screening")
async def model_screening(request: ModelScreeningRequest):
    """
    Compare multiple predictive models automatically.
    Returns ranked results with cross-validation scores.
    """
    try:
        X = np.array(request.X)
        y = np.array(request.y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if len(X) < 10:
            raise ValueError("Need at least 10 samples")

        n_features = X.shape[1]
        feature_names = request.feature_names or [f"X{i+1}" for i in range(n_features)]

        if len(feature_names) != n_features:
            feature_names = [f"X{i+1}" for i in range(n_features)]

        # Handle classification with string labels
        label_encoder = None
        original_classes = None
        if request.problem_type == "classification" and y.dtype == object:
            label_encoder = LabelEncoder()
            original_classes = list(np.unique(y))
            y = label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=request.random_state
        )

        results = []
        scoring = 'accuracy' if request.problem_type == 'classification' else 'r2'

        for method in request.methods:
            try:
                if method == 'decision_tree':
                    if request.problem_type == 'classification':
                        model = DecisionTreeClassifier(random_state=request.random_state)
                    else:
                        model = DecisionTreeRegressor(random_state=request.random_state)

                elif method == 'random_forest':
                    if request.problem_type == 'classification':
                        model = RandomForestClassifier(n_estimators=100, random_state=request.random_state, n_jobs=-1)
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=request.random_state, n_jobs=-1)

                elif method == 'gradient_boosting':
                    if request.problem_type == 'classification':
                        model = GradientBoostingClassifier(n_estimators=100, random_state=request.random_state)
                    else:
                        model = GradientBoostingRegressor(n_estimators=100, random_state=request.random_state)

                elif method == 'ridge':
                    if request.problem_type == 'classification':
                        model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=request.random_state)
                    else:
                        model = Ridge(alpha=1.0, random_state=request.random_state)

                elif method == 'lasso':
                    if request.problem_type == 'classification':
                        model = LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=1000, random_state=request.random_state)
                    else:
                        model = Lasso(alpha=1.0, random_state=request.random_state, max_iter=10000)

                elif method == 'elasticnet':
                    if request.problem_type == 'classification':
                        model = LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5, solver='saga', max_iter=1000, random_state=request.random_state)
                    else:
                        model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=request.random_state, max_iter=10000)
                else:
                    continue

                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=request.cv_folds, scoring=scoring)

                # Fit on full training set
                model.fit(X_train, y_train)

                # Test set performance
                y_pred_test = model.predict(X_test)

                if request.problem_type == 'classification':
                    test_score = accuracy_score(y_test, y_pred_test)
                else:
                    test_score = r2_score(y_test, y_pred_test)

                # Feature importance
                importance = get_feature_importance(model, feature_names)

                results.append({
                    "method": method,
                    "cv_mean": safe_float(np.mean(cv_scores)),
                    "cv_std": safe_float(np.std(cv_scores)),
                    "cv_scores": [safe_float(s) for s in cv_scores],
                    "test_score": safe_float(test_score),
                    "feature_importance": importance
                })

            except Exception as method_error:
                results.append({
                    "method": method,
                    "error": str(method_error)
                })

        # Sort by CV mean score
        valid_results = [r for r in results if 'cv_mean' in r]
        valid_results.sort(key=lambda x: x['cv_mean'], reverse=True)

        # Combine with errored results
        error_results = [r for r in results if 'error' in r]
        sorted_results = valid_results + error_results

        return make_json_safe({
            "problem_type": request.problem_type,
            "scoring_metric": scoring,
            "cv_folds": request.cv_folds,
            "results": sorted_results,
            "best_method": valid_results[0]["method"] if valid_results else None,
            "data_info": {
                "n_samples": len(X),
                "n_features": n_features,
                "n_train": len(X_train),
                "n_test": len(X_test)
            },
            "class_labels": original_classes
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/info")
async def get_info():
    """
    Get information about available predictive modeling methods
    """
    return {
        "methods": {
            "decision_tree": {
                "name": "Decision Tree (CART)",
                "description": "Recursive binary splitting for interpretable models",
                "pros": ["Highly interpretable", "Handles non-linear relationships", "No feature scaling needed"],
                "cons": ["Prone to overfitting", "Unstable (small data changes cause different trees)", "Can create biased trees with imbalanced data"],
                "hyperparameters": ["max_depth", "min_samples_split", "min_samples_leaf", "criterion"]
            },
            "random_forest": {
                "name": "Random Forest",
                "description": "Ensemble of decision trees with bootstrap aggregating",
                "pros": ["Reduces overfitting vs single tree", "Handles high dimensions well", "Provides feature importance"],
                "cons": ["Less interpretable", "Slower prediction", "Memory intensive"],
                "hyperparameters": ["n_estimators", "max_depth", "max_features", "bootstrap"]
            },
            "gradient_boosting": {
                "name": "Gradient Boosting",
                "description": "Sequential ensemble that minimizes loss function",
                "pros": ["Often best predictive accuracy", "Handles mixed feature types", "Feature importance"],
                "cons": ["Slower to train", "Prone to overfitting if not tuned", "Many hyperparameters"],
                "hyperparameters": ["n_estimators", "learning_rate", "max_depth", "subsample"]
            },
            "ridge": {
                "name": "Ridge Regression (L2)",
                "description": "Linear regression with L2 penalty to prevent overfitting",
                "pros": ["Fast training", "Handles multicollinearity", "Coefficients interpretable"],
                "cons": ["Assumes linear relationships", "Doesn't perform feature selection"],
                "hyperparameters": ["alpha"]
            },
            "lasso": {
                "name": "Lasso Regression (L1)",
                "description": "Linear regression with L1 penalty for sparse solutions",
                "pros": ["Automatic feature selection", "Interpretable", "Handles high dimensions"],
                "cons": ["Assumes linearity", "Selects arbitrarily among correlated features"],
                "hyperparameters": ["alpha"]
            },
            "elasticnet": {
                "name": "Elastic Net",
                "description": "Combines L1 and L2 penalties",
                "pros": ["Balances Ridge and Lasso", "Better with correlated features than Lasso"],
                "cons": ["Two hyperparameters to tune", "Assumes linearity"],
                "hyperparameters": ["alpha", "l1_ratio"]
            }
        },
        "metrics": {
            "regression": {
                "r2": "Coefficient of determination (variance explained)",
                "rmse": "Root Mean Squared Error",
                "mae": "Mean Absolute Error",
                "mse": "Mean Squared Error"
            },
            "classification": {
                "accuracy": "Fraction of correct predictions",
                "precision": "True positives / Predicted positives",
                "recall": "True positives / Actual positives",
                "f1_score": "Harmonic mean of precision and recall",
                "auc_roc": "Area Under ROC Curve (binary only)"
            }
        }
    }
