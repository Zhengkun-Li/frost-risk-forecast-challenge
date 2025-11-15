"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range("2020-01-01", periods=100, freq="h")
    return pd.DataFrame({
        "Date": dates,
        "Stn Id": [2] * 100,
        "Stn Name": ["TestStation"] * 100,
        "Air Temp (C)": np.random.normal(15, 5, 100),
        "Dew Point (C)": np.random.normal(10, 4, 100),
        "Rel Hum (%)": np.random.uniform(40, 80, 100),
        "Wind Speed (m/s)": np.random.uniform(0.5, 5, 100),
        "qc": [""] * 100,
    })


@pytest.fixture
def sample_dataframe_with_qc():
    """Create a sample DataFrame with QC flags."""
    dates = pd.date_range("2020-01-01", periods=10, freq="h")
    return pd.DataFrame({
        "Date": dates,
        "Stn Id": [2] * 10,
        "Air Temp (C)": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        "qc": ["", "Y", "R", "M", "", "Y", "Q", "P", "S", ""],
    })


@pytest.fixture
def sample_dataframe_with_sentinels():
    """Create a sample DataFrame with sentinel values."""
    return pd.DataFrame({
        "Sol Rad (W/sq.m)": [100.0, -6999, 200.0, -9999, 150.0],
        "Soil Temp (C)": [15.0, 16.0, -6999, 17.0, 18.0],
    })


@pytest.fixture
def sample_dataframe_with_missing():
    """Create a sample DataFrame with missing values."""
    dates = pd.date_range("2020-01-01", periods=10, freq="h")
    return pd.DataFrame({
        "Date": dates,
        "Stn Id": [2] * 10,
        "Air Temp (C)": [10.0, np.nan, np.nan, 13.0, 14.0, np.nan, 16.0, 17.0, 18.0, 19.0],
    })


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_feature_config():
    """Sample feature engineering configuration."""
    return {
        "time_features": True,
        "lag_features": {
            "enabled": True,
            "columns": ["Air Temp (C)", "Dew Point (C)"],
            "lags": [1, 3, 6]
        },
        "rolling_features": {
            "enabled": True,
            "columns": ["Air Temp (C)"],
            "windows": [6, 12],
            "functions": ["mean", "min", "max"]
        },
        "derived_features": True
    }

