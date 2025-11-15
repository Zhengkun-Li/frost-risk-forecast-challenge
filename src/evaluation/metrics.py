"""Evaluation metrics for frost forecasting models."""

from typing import Dict, Optional
import numpy as np

try:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        brier_score_loss,
        roc_auc_score,
        average_precision_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Define fallback implementations
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


class MetricsCalculator:
    """Calculate all evaluation metrics for regression and classification tasks."""
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        additional_metrics: Optional[list] = None
    ) -> Dict[str, float]:
        """Calculate regression metrics.
        
        Args:
            y_true: True target values.
            y_pred: Predicted values.
            additional_metrics: List of additional metric names to calculate.
        
        Returns:
            Dictionary of metric names and values:
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
            - r2: R-squared
            - mape: Mean Absolute Percentage Error (if applicable)
        """
        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
        }
        
        # Mean Absolute Percentage Error (avoid division by zero)
        mask = y_true != 0
        if mask.sum() > 0:
            metrics["mape"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Additional metrics
        if additional_metrics:
            for metric_name in additional_metrics:
                if metric_name == "mape" and "mape" not in metrics:
                    mask = y_true != 0
                    if mask.sum() > 0:
                        metrics["mape"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        return metrics
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics.
        
        Args:
            y_true: True binary labels (0 or 1).
            y_pred: Predicted binary labels.
            y_proba: Predicted probabilities (optional, for probability-based metrics).
        
        Returns:
            Dictionary of metric names and values:
            - accuracy: Accuracy
            - precision: Precision
            - recall: Recall
            - f1: F1 score
            - roc_auc: ROC AUC (if y_proba provided)
            - pr_auc: Precision-Recall AUC (if y_proba provided)
            - brier_score: Brier Score (if y_proba provided)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for classification metrics. Install with: pip install scikit-learn")
        
        metrics = {
            "accuracy": np.mean(y_true == y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Probability-based metrics
        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics["roc_auc"] = np.nan
            
            try:
                metrics["pr_auc"] = average_precision_score(y_true, y_proba)
            except ValueError:
                metrics["pr_auc"] = np.nan
            
            try:
                metrics["brier_score"] = brier_score_loss(y_true, y_proba)
            except ValueError:
                metrics["brier_score"] = np.nan
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:  # Binary classification
            metrics["tn"] = int(cm[0, 0])  # True negatives
            metrics["fp"] = int(cm[0, 1])  # False positives
            metrics["fn"] = int(cm[1, 0])  # False negatives
            metrics["tp"] = int(cm[1, 1])  # True positives
        
        return metrics
    
    @staticmethod
    def calculate_probability_metrics(
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate probability-based metrics.
        
        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities.
        
        Returns:
            Dictionary of probability metrics.
        """
        metrics = {}
        
        try:
            metrics["brier_score"] = brier_score_loss(y_true, y_proba)
        except ValueError:
            metrics["brier_score"] = np.nan
        
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = np.nan
        
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics["pr_auc"] = np.nan
        
        # Expected Calibration Error (ECE)
        try:
            metrics["ece"] = MetricsCalculator.calculate_ece(y_true, y_proba)
        except Exception:
            metrics["ece"] = np.nan
        
        return metrics
    
    @staticmethod
    def calculate_ece(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error (ECE).
        
        ECE measures the difference between predicted probability and actual frequency.
        
        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities.
            n_bins: Number of bins for probability calibration.
        
        Returns:
            ECE value (lower is better, 0 is perfect calibration).
        """
        # Ensure 1D arrays
        y_true = np.asarray(y_true).flatten()
        y_proba = np.asarray(y_proba).flatten()
        
        if len(y_true) != len(y_proba):
            raise ValueError(f"y_true and y_proba must have the same length. Got {len(y_true)} and {len(y_proba)}")
        
        # Bin edges
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = y_true[in_bin].mean()
                # Average predicted probability in this bin
                avg_confidence_in_bin = y_proba[in_bin].mean()
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    @staticmethod
    def calculate_reliability_data(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """Calculate data for reliability diagram.
        
        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities.
            n_bins: Number of bins.
        
        Returns:
            Dictionary with bin centers, predicted probabilities, and actual frequencies.
        """
        # Ensure 1D arrays
        y_true = np.asarray(y_true).flatten()
        y_proba = np.asarray(y_proba).flatten()
        
        if len(y_true) != len(y_proba):
            raise ValueError(f"y_true and y_proba must have the same length. Got {len(y_true)} and {len(y_proba)}")
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        bin_centers = (bin_lowers + bin_uppers) / 2
        
        predicted_probs = []
        actual_freqs = []
        counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            count = in_bin.sum()
            counts.append(count)
            
            if count > 0:
                predicted_probs.append(y_proba[in_bin].mean())
                actual_freqs.append(y_true[in_bin].mean())
            else:
                predicted_probs.append(np.nan)
                actual_freqs.append(np.nan)
        
        return {
            "bin_centers": bin_centers,
            "predicted_probs": np.array(predicted_probs),
            "actual_freqs": np.array(actual_freqs),
            "counts": np.array(counts)
        }
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        task_type: str = "regression"
    ) -> Dict[str, float]:
        """Calculate all relevant metrics based on task type.
        
        Args:
            y_true: True target values or labels.
            y_pred: Predicted values or labels.
            y_proba: Predicted probabilities (optional).
            task_type: "regression" or "classification".
        
        Returns:
            Dictionary of all metrics.
        """
        if task_type == "regression":
            metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
            if y_proba is not None:
                # For regression with probability output, also calculate probability metrics
                # Convert y_true to binary if needed (e.g., frost event threshold)
                # This is a placeholder - should be configured based on use case
                pass
        else:
            metrics = MetricsCalculator.calculate_classification_metrics(y_true, y_pred, y_proba)
        
        return metrics
    
    @staticmethod
    def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
        """Format metrics dictionary as a readable string.
        
        Args:
            metrics: Dictionary of metrics.
            precision: Number of decimal places.
        
        Returns:
            Formatted string.
        """
        lines = []
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                lines.append(f"{key}: {value:.{precision}f}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

