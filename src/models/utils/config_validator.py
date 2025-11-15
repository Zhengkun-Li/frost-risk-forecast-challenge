"""Configuration validation utility for model training."""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


class ConfigValidator:
    """Validate model and training configurations.
    
    This class provides validation for model configurations to catch
    errors early and provide helpful error messages.
    """
    
    @staticmethod
    def validate_model_config(
        model_type: str,
        config: Dict[str, Any],
        task_type: str = "classification"
    ) -> Tuple[bool, Optional[str]]:
        """Validate model configuration.
        
        Args:
            model_type: Type of model (lightgbm, xgboost, lstm, etc.).
            config: Configuration dictionary.
            task_type: Task type (classification or regression).
        
        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        if not isinstance(config, dict):
            return False, "Configuration must be a dictionary"
        
        model_params = config.get("model_params", {})
        if not isinstance(model_params, dict):
            return False, "model_params must be a dictionary"
        
        # Model-specific validation
        if model_type == "lstm" or model_type == "lstm_multitask":
            return ConfigValidator._validate_lstm_config(model_params)
        elif model_type in ["lightgbm", "xgboost", "catboost"]:
            return ConfigValidator._validate_tree_config(model_type, model_params)
        elif model_type == "random_forest":
            return ConfigValidator._validate_rf_config(model_params)
        elif model_type == "prophet":
            return ConfigValidator._validate_prophet_config(model_params)
        
        return True, None
    
    @staticmethod
    def _validate_lstm_config(model_params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate LSTM model configuration."""
        # Check required parameters
        required_params = ["sequence_length", "hidden_size", "batch_size", "epochs", "learning_rate"]
        for param in required_params:
            if param not in model_params:
                return False, f"Missing required parameter: {param}"
        
        # Validate parameter types and ranges
        if not isinstance(model_params["sequence_length"], int) or model_params["sequence_length"] <= 0:
            return False, "sequence_length must be a positive integer"
        
        if not isinstance(model_params["hidden_size"], int) or model_params["hidden_size"] <= 0:
            return False, "hidden_size must be a positive integer"
        
        if not isinstance(model_params["batch_size"], int) or model_params["batch_size"] <= 0:
            return False, "batch_size must be a positive integer"
        
        if not isinstance(model_params["epochs"], int) or model_params["epochs"] <= 0:
            return False, "epochs must be a positive integer"
        
        if not isinstance(model_params["learning_rate"], (int, float)) or model_params["learning_rate"] <= 0:
            return False, "learning_rate must be a positive number"
        
        # Validate optional parameters
        if "dropout" in model_params:
            dropout = model_params["dropout"]
            if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
                return False, "dropout must be a number between 0 and 1"
        
        if "checkpoint_frequency" in model_params:
            freq = model_params["checkpoint_frequency"]
            if not isinstance(freq, int) or freq < 0:
                return False, "checkpoint_frequency must be a non-negative integer"
        
        return True, None
    
    @staticmethod
    def _validate_tree_config(
        model_type: str,
        model_params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate tree model (LightGBM, XGBoost, CatBoost) configuration."""
        # Check common parameters
        if "n_estimators" in model_params:
            n_est = model_params["n_estimators"]
            if not isinstance(n_est, int) or n_est <= 0:
                return False, "n_estimators must be a positive integer"
        
        if "learning_rate" in model_params:
            lr = model_params["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                return False, "learning_rate must be a positive number"
        
        if "max_depth" in model_params:
            max_d = model_params["max_depth"]
            if not isinstance(max_d, int) or max_d <= 0:
                return False, "max_depth must be a positive integer"
        
        return True, None
    
    @staticmethod
    def _validate_rf_config(model_params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate Random Forest configuration."""
        if "n_estimators" in model_params:
            n_est = model_params["n_estimators"]
            if not isinstance(n_est, int) or n_est <= 0:
                return False, "n_estimators must be a positive integer"
        
        if "max_depth" in model_params:
            max_d = model_params["max_depth"]
            if max_d is not None and (not isinstance(max_d, int) or max_d <= 0):
                return False, "max_depth must be None or a positive integer"
        
        return True, None
    
    @staticmethod
    def _validate_prophet_config(model_params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate Prophet configuration."""
        # Prophet has minimal required parameters
        # Most parameters are optional with good defaults
        return True, None
    
    @staticmethod
    def validate_training_args(
        model_type: str,
        checkpoint_dir: Optional[Path] = None,
        log_file: Optional[Path] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Validate training arguments.
        
        Args:
            model_type: Type of model.
            checkpoint_dir: Optional checkpoint directory.
            log_file: Optional log file path.
            **kwargs: Additional training arguments.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        # Validate checkpoint directory
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            # Check if parent directory exists and is writable
            try:
                checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                return False, f"Cannot create checkpoint directory: {e}"
        
        # Validate log file
        if log_file is not None:
            log_file = Path(log_file)
            # Check if parent directory exists and is writable
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                return False, f"Cannot create log file directory: {e}"
        
        return True, None
    
    @staticmethod
    def suggest_fixes(
        model_type: str,
        config: Dict[str, Any],
        error_message: str
    ) -> List[str]:
        """Suggest fixes for configuration errors.
        
        Args:
            model_type: Type of model.
            config: Configuration dictionary.
            error_message: Error message from validation.
        
        Returns:
            List of suggested fixes.
        """
        suggestions = []
        
        if "Missing required parameter" in error_message:
            param = error_message.split(":")[-1].strip()
            if model_type in ["lstm", "lstm_multitask"]:
                if param == "sequence_length":
                    suggestions.append(f"Add '{param}': 24 to model_params")
                elif param == "hidden_size":
                    suggestions.append(f"Add '{param}': 64 or 128 to model_params")
                elif param == "batch_size":
                    suggestions.append(f"Add '{param}': 32 or 64 to model_params")
                elif param == "epochs":
                    suggestions.append(f"Add '{param}': 50 or 100 to model_params")
                elif param == "learning_rate":
                    suggestions.append(f"Add '{param}': 0.001 or 0.0001 to model_params")
        
        if "must be a positive" in error_message:
            suggestions.append("Ensure the parameter value is a positive number")
        
        if "must be between 0 and 1" in error_message:
            suggestions.append("Ensure the parameter value is between 0 and 1")
        
        return suggestions

