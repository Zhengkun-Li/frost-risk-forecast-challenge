"""Training modules for frost forecasting.

This package contains:
- data_preparation: Data loading, cleaning, and feature engineering
- model_config: Model parameter configuration
- model_trainer: Model training and evaluation
- loso_evaluator: Leave-One-Station-Out evaluation
"""

from .data_preparation import load_and_prepare_data, create_frost_labels, prepare_features_and_targets
from .model_config import get_model_params, get_model_class, get_model_config, get_resource_aware_config
from .model_trainer import train_models_for_horizon
from .loso_evaluator import perform_loso_evaluation, calculate_loso_summary

__all__ = [
    "load_and_prepare_data",
    "create_frost_labels",
    "prepare_features_and_targets",
    "get_model_params",
    "get_model_class",
    "get_model_config",
    "get_resource_aware_config",
    "train_models_for_horizon",
    "perform_loso_evaluation",
    "calculate_loso_summary",
]

