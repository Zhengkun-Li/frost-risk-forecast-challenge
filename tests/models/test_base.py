"""Tests for base model interface."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.models.base import BaseModel


class DummyModel(BaseModel):
    """Dummy model for testing BaseModel interface."""
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "DummyModel":
        """Dummy fit implementation."""
        self.model = {"fitted": True, "n_samples": len(X)}
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Dummy predict implementation."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return np.random.randn(len(X))
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Dummy predict_proba implementation."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return np.random.rand(len(X))


class TestBaseModel:
    """Test cases for BaseModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = {"model_params": {"param1": 1}}
        model = DummyModel(config)
        
        assert model.config == config
        assert model.model is None
        assert not model.is_fitted
    
    def test_fit(self, sample_dataframe):
        """Test model fitting."""
        model = DummyModel({"model_params": {}})
        X = sample_dataframe[["Air Temp (C)", "Dew Point (C)"]]
        y = sample_dataframe["Air Temp (C)"]
        
        model.fit(X, y)
        assert model.is_fitted
        assert model.model is not None
    
    def test_predict_before_fit(self, sample_dataframe):
        """Test that predict raises error if model not fitted."""
        model = DummyModel({"model_params": {}})
        X = sample_dataframe[["Air Temp (C)"]]
        
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)
    
    def test_predict_after_fit(self, sample_dataframe):
        """Test prediction after fitting."""
        model = DummyModel({"model_params": {}})
        X = sample_dataframe[["Air Temp (C)", "Dew Point (C)"]]
        y = sample_dataframe["Air Temp (C)"]
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_proba(self, sample_dataframe):
        """Test probability prediction."""
        model = DummyModel({"model_params": {}})
        X = sample_dataframe[["Air Temp (C)", "Dew Point (C)"]]
        y = sample_dataframe["Air Temp (C)"]
        
        model.fit(X, y)
        probabilities = model.predict_proba(X)
        
        assert len(probabilities) == len(X)
        assert isinstance(probabilities, np.ndarray)
    
    def test_save_and_load(self, sample_dataframe, temp_dir):
        """Test saving and loading model."""
        model = DummyModel({"model_params": {"test": 123}})
        X = sample_dataframe[["Air Temp (C)", "Dew Point (C)"]]
        y = sample_dataframe["Air Temp (C)"]
        
        model.fit(X, y)
        
        # Save
        save_path = temp_dir / "test_model"
        model.save(save_path)
        
        assert (save_path / "model.pkl").exists()
        assert (save_path / "config.json").exists()
        
        # Load
        loaded_model = DummyModel.load(save_path)
        
        assert loaded_model.is_fitted
        assert loaded_model.config["model_params"]["test"] == 123
    
    def test_get_feature_importance_default(self):
        """Test default feature importance (returns None)."""
        model = DummyModel({"model_params": {}})
        assert model.get_feature_importance() is None
    
    def test_get_params(self):
        """Test getting model parameters."""
        config = {"model_params": {"param1": 1, "param2": 2}}
        model = DummyModel(config)
        
        params = model.get_params()
        assert params == {"param1": 1, "param2": 2}
    
    def test_set_params(self):
        """Test setting model parameters."""
        model = DummyModel({"model_params": {}})
        model.set_params(param1=1, param2=2)
        
        assert model.config["model_params"]["param1"] == 1
        assert model.config["model_params"]["param2"] == 2
    
    def test_repr(self):
        """Test string representation."""
        model = DummyModel({"model_params": {}})
        assert "DummyModel" in repr(model)
        assert "not fitted" in repr(model)
        
        # Fit and check again
        X = pd.DataFrame({"col": [1, 2, 3]})
        y = pd.Series([1, 2, 3])
        model.fit(X, y)
        
        assert "fitted" in repr(model)

