"""Hyperparameter optimization utilities."""

from typing import Dict, Any, Callable, Optional
import numpy as np
import pandas as pd

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

try:
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class HyperparameterOptimizer:
    """Hyperparameter optimization using Hyperopt."""
    
    def __init__(self, model_class, config_template: Dict[str, Any], max_evals: int = 50):
        """Initialize optimizer.
        
        Args:
            model_class: Model class to optimize.
            config_template: Template configuration dictionary.
            max_evals: Maximum number of evaluations.
        """
        if not HYPEROPT_AVAILABLE:
            raise ImportError("Hyperopt is required. Install with: pip install hyperopt")
        
        self.model_class = model_class
        self.config_template = config_template
        self.max_evals = max_evals
        self.trials = Trials()
        self.best_config = None
        self.best_score = None
    
    def optimize(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 space: Dict[str, Any],
                 objective_func: Optional[Callable] = None,
                 metric: str = "neg_mean_absolute_error",
                 cv: int = 3) -> Dict[str, Any]:
        """Optimize hyperparameters.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            space: Hyperparameter search space (using Hyperopt syntax).
            objective_func: Custom objective function. If None, uses cross-validation.
            metric: Metric to optimize (for sklearn cross-validation).
            cv: Number of cross-validation folds.
        
        Returns:
            Best hyperparameter configuration.
        """
        if objective_func is None:
            def objective_func(params):
                return self._default_objective(X, y, params, metric, cv)
        
        def objective(params):
            try:
                score = objective_func(params)
                return {'loss': -score, 'status': STATUS_OK}
            except Exception as e:
                return {'loss': float('inf'), 'status': STATUS_FAIL, 'error': str(e)}
        
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=self.trials
        )
        
        self.best_config = best
        self.best_score = -min([t['result']['loss'] for t in self.trials.trials if t['result']['status'] == STATUS_OK])
        
        return best
    
    def _default_objective(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], metric: str, cv: int) -> float:
        """Default objective function using cross-validation.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            params: Hyperparameters to evaluate.
            metric: Metric to optimize.
            cv: Number of CV folds.
        
        Returns:
            Average CV score.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for default objective")
        
        # Update config with hyperparameters
        config = self.config_template.copy()
        if "model_params" not in config:
            config["model_params"] = {}
        
        # Convert Hyperopt params to model params
        for key, value in params.items():
            if key.startswith("model_"):
                param_key = key.replace("model_", "")
                config["model_params"][param_key] = value
            else:
                config[key] = value
        
        # Create and train model
        model = self.model_class(config)
        model.fit(X, y)
        
        # Cross-validation
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model_temp = self.model_class(config)
            model_temp.fit(X_train, y_train)
            y_pred = model_temp.predict(X_val)
            
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            if metric == "neg_mean_absolute_error":
                score = -mean_absolute_error(y_val, y_pred)
            elif metric == "neg_mean_squared_error":
                score = -mean_squared_error(y_val, y_pred)
            elif metric == "r2":
                score = r2_score(y_val, y_pred)
            else:
                score = -mean_absolute_error(y_val, y_pred)
            
            scores.append(score)
        
        return np.mean(scores)
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get best configuration found during optimization.
        
        Returns:
            Best configuration dictionary.
        """
        if self.best_config is None:
            raise ValueError("Optimization has not been run yet")
        
        config = self.config_template.copy()
        if "model_params" not in config:
            config["model_params"] = {}
        
        for key, value in self.best_config.items():
            if key.startswith("model_"):
                param_key = key.replace("model_", "")
                config["model_params"][param_key] = value
            else:
                config[key] = value
        
        return config
    
    def get_trials_summary(self) -> pd.DataFrame:
        """Get summary of all trials.
        
        Returns:
            DataFrame with trial results.
        """
        results = []
        for trial in self.trials.trials:
            if trial['result']['status'] == STATUS_OK:
                results.append({
                    'loss': trial['result']['loss'],
                    'params': trial['misc']['vals']
                })
        
        return pd.DataFrame(results)

