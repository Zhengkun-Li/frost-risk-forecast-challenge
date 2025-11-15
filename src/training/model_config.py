"""Model configuration module for frost forecasting training.

This module handles:
- Model parameter configuration for different model types
- Model class selection
- Resource-aware configuration adjustment
"""

import os
from typing import Dict, Tuple, Optional, Type
from pathlib import Path

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_resource_aware_config() -> Tuple[int, int]:
    """Get resource-aware configuration based on available memory.
    
    Returns:
        Tuple of (hidden_size, batch_size) for LSTM models.
    """
    if not PSUTIL_AVAILABLE:
        # Default configuration if psutil is not available
        return 64, 16
    
    mem_gb = psutil.virtual_memory().total / (1024**3)
    if mem_gb >= 32:
        return 128, 128  # Increased batch size for better GPU utilization
    elif mem_gb >= 16:
        return 128, 64
    else:
        return 64, 32  # Increased minimum batch size


def get_model_params(
    model_type: str,
    task_type: str = "classification",
    max_workers: Optional[int] = None,
    for_loso: bool = False
) -> Dict:
    """Get model parameters for a specific model type and task.
    
    Args:
        model_type: Model type (lightgbm, xgboost, catboost, etc.).
        task_type: Task type (classification or regression).
        max_workers: Maximum number of workers (auto-determined if None).
        for_loso: Whether this is for LOSO evaluation (smaller config).
    
    Returns:
        Dictionary of model parameters.
    """
    # Validate model type
    supported_models = [
        "lightgbm", "xgboost", "catboost", "random_forest", 
        "ensemble", "lstm", "lstm_multitask", "prophet"
    ]
    if model_type not in supported_models:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {', '.join(supported_models)}"
        )
    
    if max_workers is None:
        max_workers = min(8, max(1, os.cpu_count() // 4))
    
    # Adjust parameters for LOSO (smaller config to save memory)
    if for_loso:
        n_estimators = 50
        max_depth = 6
        num_leaves = 31
        hidden_size, batch_size = get_resource_aware_config()
        if hidden_size > 64:
            hidden_size = 64
        if batch_size > 32:
            batch_size = 32
        epochs = 50
        patience = 8
    else:
        n_estimators = 200
        max_depth = 8
        num_leaves = 63
        hidden_size, batch_size = get_resource_aware_config()
        epochs = 100
        patience = 10
    
    if model_type == "lightgbm":
        if task_type == "classification":
            return {
                "n_estimators": n_estimators,
                "learning_rate": 0.05,
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": max_workers,
                "force_col_wise": True,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            }
        else:  # regression
            return {
                "n_estimators": n_estimators,
                "learning_rate": 0.05,
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": max_workers,
                "force_col_wise": True,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            }
    
    elif model_type == "xgboost":
        objective = "binary:logistic" if task_type == "classification" else "reg:squarederror"
        return {
            "n_estimators": n_estimators,
            "learning_rate": 0.05,
            "max_depth": max_depth,
            "random_state": 42,
            "n_jobs": max_workers,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "tree_method": "hist",
            "objective": objective,
        }
    
    elif model_type == "catboost":
        return {
            "iterations": n_estimators,
            "learning_rate": 0.05,
            "depth": max_depth,
            "random_state": 42,
            "thread_count": max_workers,
            "subsample": 0.8,
            "colsample_bylevel": 0.8,
            "l2_leaf_reg": 0.1,
            "verbose": False,
        }
    
    elif model_type == "random_forest":
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": max_workers,
        }
    
    elif model_type == "ensemble":
        base_n_estimators = 150 if not for_loso else 30
        return {
            "lightgbm": {
                "n_estimators": base_n_estimators,
                "learning_rate": 0.05,
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": max_workers,
                "force_col_wise": True,
            },
            "xgboost": {
                "n_estimators": base_n_estimators,
                "learning_rate": 0.05,
                "max_depth": max_depth,
                "random_state": 42,
                "n_jobs": max_workers,
                "tree_method": "hist",
                "objective": "binary:logistic" if task_type == "classification" else "reg:squarederror",
            },
            "catboost": {
                "iterations": base_n_estimators,
                "learning_rate": 0.05,
                "depth": max_depth,
                "random_state": 42,
                "thread_count": max_workers,
                "verbose": False,
            }
        }
    
    elif model_type == "lstm":
        return {
            "sequence_length": 24,
            "hidden_size": hidden_size,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.0001,  # Reduced from 0.001 to prevent NaN
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": True,
            "patience": patience,
            "min_delta": 1e-6,
            "lr_scheduler": True,
            "lr_scheduler_patience": 5,
            "lr_scheduler_factor": 0.5,
            "gradient_clip": 1.0,  # Gradient clipping to prevent explosion
            "save_best_model": True,
            "use_amp": True,  # Enable mixed precision training for 1.5-2x speedup
            "val_frequency": 5,  # Validate every 5 epochs instead of every epoch (faster training)
            "checkpoint_frequency": 10,  # Save checkpoint every 10 epochs (0 = disabled)
        }
    
    elif model_type == "lstm_multitask":
        return {
            "sequence_length": 24,
            "hidden_size": hidden_size,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.0001,  # Reduced from 0.001 to prevent NaN
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": True,
            "patience": patience,
            "min_delta": 1e-6,
            "lr_scheduler": True,
            "lr_scheduler_patience": 5,
            "lr_scheduler_factor": 0.5,
            "gradient_clip": 1.0,  # Gradient clipping to prevent explosion
            "save_best_model": True,
            "loss_weight_temp": 1.0,
            "loss_weight_frost": 1.0,
            "use_amp": True,  # Enable mixed precision training for 1.5-2x speedup
            "val_frequency": 5,  # Validate every 5 epochs instead of every epoch (faster training)
        }
    
    elif model_type == "prophet":
        return {
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": True,
            "seasonality_mode": "multiplicative",
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_model_class(model_type: str):
    """Get model class for a specific model type.
    
    Args:
        model_type: Model type (lightgbm, xgboost, catboost, etc.).
    
    Returns:
        Model class.
    """
    if model_type == "lightgbm":
        from src.models.ml.lightgbm_model import LightGBMModel
        return LightGBMModel
    elif model_type == "xgboost":
        from src.models.ml.xgboost_model import XGBoostModel
        return XGBoostModel
    elif model_type == "catboost":
        from src.models.ml.catboost_model import CatBoostModel
        return CatBoostModel
    elif model_type == "random_forest":
        from src.models.ml.random_forest_model import RandomForestModel
        return RandomForestModel
    elif model_type == "ensemble":
        from src.models.ml.ensemble_model import EnsembleModel
        return EnsembleModel
    elif model_type == "lstm":
        from src.models.deep.lstm_model import LSTMForecastModel
        return LSTMForecastModel
    elif model_type == "lstm_multitask":
        from src.models.deep.lstm_multitask_model import LSTMMultiTaskForecastModel
        return LSTMMultiTaskForecastModel
    elif model_type == "prophet":
        from src.models.traditional.prophet_model import ProphetModel
        return ProphetModel
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_model_config(
    model_type: str,
    horizon: int,
    task_type: str = "classification",
    max_workers: Optional[int] = None,
    for_loso: bool = False,
    station_id: Optional[int] = None
) -> Dict:
    """Get complete model configuration.
    
    Args:
        model_type: Model type (lightgbm, xgboost, catboost, etc.).
        horizon: Forecast horizon in hours.
        task_type: Task type (classification or regression).
        max_workers: Maximum number of workers (auto-determined if None).
        for_loso: Whether this is for LOSO evaluation.
        station_id: Optional station ID for LOSO (used in model name).
    
    Returns:
        Dictionary with complete model configuration.
    """
    model_params = get_model_params(model_type, task_type, max_workers, for_loso)
    
    # Create model name
    if for_loso and station_id is not None:
        model_name = f"{task_type}_{horizon}h_station_{station_id}"
    else:
        model_name = f"{task_type}_{horizon}h"
    
    config = {
        "model_name": model_name,
        "model_type": model_type,
        "task_type": task_type,
        "model_params": model_params
    }
    
    # Add model-specific config
    if model_type == "ensemble":
        config["base_models"] = ["lightgbm", "xgboost", "catboost"]
        config["ensemble_method"] = "mean"
    elif model_type in ["lstm", "lstm_multitask"]:
        config["date_column"] = "Date"
        if model_type == "lstm_multitask":
            config["task_type"] = "multitask"
    elif model_type == "prophet":
        config["date_column"] = "Date"
        config["target_column"] = f"{task_type.split('_')[0]}_{horizon}h"
        config["regressor_columns"] = []
    
    return config

