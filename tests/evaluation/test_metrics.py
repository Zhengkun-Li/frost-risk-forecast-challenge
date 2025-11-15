"""Tests for metrics calculation."""

import pytest
import numpy as np

from src.evaluation.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Test cases for MetricsCalculator."""
    
    def test_calculate_regression_metrics(self):
        """Test regression metrics calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
    
    def test_calculate_classification_metrics(self):
        """Test classification metrics calculation."""
        try:
            from sklearn.metrics import precision_score
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6, 0.3])
        
        metrics = MetricsCalculator.calculate_classification_metrics(y_true, y_pred, y_proba)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "brier_score" in metrics
    
    def test_calculate_probability_metrics(self):
        """Test probability metrics calculation."""
        try:
            from sklearn.metrics import brier_score_loss
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
        
        metrics = MetricsCalculator.calculate_probability_metrics(y_true, y_proba)
        
        assert "brier_score" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert 0 <= metrics["brier_score"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
    
    def test_calculate_all_metrics_regression(self):
        """Test calculate_all_metrics for regression."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        
        metrics = MetricsCalculator.calculate_all_metrics(
            y_true, y_pred, task_type="regression"
        )
        
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
    
    def test_calculate_all_metrics_classification(self):
        """Test calculate_all_metrics for classification."""
        try:
            from sklearn.metrics import precision_score
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.8, 0.2])
        
        metrics = MetricsCalculator.calculate_all_metrics(
            y_true, y_pred, y_proba, task_type="classification"
        )
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "f1" in metrics
    
    def test_format_metrics(self):
        """Test metrics formatting."""
        metrics = {"mae": 1.2345, "rmse": 2.3456, "r2": 0.9876}
        formatted = MetricsCalculator.format_metrics(metrics, precision=2)
        
        assert "mae: 1.23" in formatted
        assert "rmse: 2.35" in formatted
        assert "r2: 0.99" in formatted

