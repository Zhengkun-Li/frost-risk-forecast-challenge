"""Tests for data cleaning module."""

import pytest
import pandas as pd
import numpy as np

from src.data.cleaners import DataCleaner


class TestDataCleaner:
    """Test cases for DataCleaner."""

    def test_apply_qc_filter(self):
        """Test QC flag filtering."""
        df = pd.DataFrame({
            "Air Temp (C)": [10.0, 11.0, 12.0, 13.0],
            "qc": ["", "Y", "R", "M"]
        })
        
        cleaner = DataCleaner()
        df_cleaned = cleaner.apply_qc_filter(df)
        
        # R and M should be marked as NaN
        assert pd.isna(df_cleaned.loc[2, "Air Temp (C)"])
        assert pd.isna(df_cleaned.loc[3, "Air Temp (C)"])
        # Blank and Y should be kept
        assert not pd.isna(df_cleaned.loc[0, "Air Temp (C)"])
        assert not pd.isna(df_cleaned.loc[1, "Air Temp (C)"])

    def test_handle_sentinels(self):
        """Test sentinel value replacement."""
        df = pd.DataFrame({
            "Sol Rad (W/sq.m)": [100.0, -6999, 200.0, -9999]
        })
        
        cleaner = DataCleaner()
        df_cleaned = cleaner.handle_sentinels(df)
        
        assert pd.isna(df_cleaned.loc[1, "Sol Rad (W/sq.m)"])
        assert pd.isna(df_cleaned.loc[3, "Sol Rad (W/sq.m)"])
        assert not pd.isna(df_cleaned.loc[0, "Sol Rad (W/sq.m)"])
        assert not pd.isna(df_cleaned.loc[2, "Sol Rad (W/sq.m)"])

    def test_impute_missing_forward_fill(self):
        """Test forward fill imputation."""
        df = pd.DataFrame({
            "Stn Id": [2, 2, 2, 2],
            "Date": pd.date_range("2020-01-01", periods=4, freq="h"),  # Use 'h' instead of 'H'
            "Air Temp (C)": [10.0, np.nan, np.nan, 13.0]
        })
        
        cleaner = DataCleaner()
        df_imputed = cleaner.impute_missing(df, strategy="forward_fill")
        
        # First NaN should be filled with 10.0, second with 10.0
        assert df_imputed.loc[1, "Air Temp (C)"] == 10.0
        assert df_imputed.loc[2, "Air Temp (C)"] == 10.0

