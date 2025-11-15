"""Tests for data cleaning module."""

import pytest
import pandas as pd
import numpy as np

from src.data.cleaners import DataCleaner


class TestDataCleaner:
    """Test cases for DataCleaner."""

    def test_apply_qc_filter(self, sample_dataframe_with_qc):
        """Test QC flag filtering."""
        cleaner = DataCleaner()
        df_cleaned = cleaner.apply_qc_filter(sample_dataframe_with_qc)
        
        # R, M, S should be marked as NaN
        assert pd.isna(df_cleaned.loc[2, "Air Temp (C)"])  # R
        assert pd.isna(df_cleaned.loc[3, "Air Temp (C)"])  # M
        assert pd.isna(df_cleaned.loc[8, "Air Temp (C)"])  # S
        
        # Blank and Y should be kept
        assert not pd.isna(df_cleaned.loc[0, "Air Temp (C)"])  # Blank
        assert not pd.isna(df_cleaned.loc[1, "Air Temp (C)"])  # Y

    def test_apply_qc_filter_custom_config(self):
        """Test QC filtering with custom configuration."""
        df = pd.DataFrame({
            "Air Temp (C)": [10.0, 11.0, 12.0],
            "qc": ["", "Q", "P"]
        })
        
        # Custom config: keep Q and P
        custom_config = {
            "": True,
            "Y": True,
            "M": False,
            "R": False,
            "S": False,
            "Q": True,  # Keep Q
            "P": True,  # Keep P
        }
        
        cleaner = DataCleaner(qc_config=custom_config)
        df_cleaned = cleaner.apply_qc_filter(df)
        
        # Q and P should be kept
        assert not pd.isna(df_cleaned.loc[1, "Air Temp (C)"])
        assert not pd.isna(df_cleaned.loc[2, "Air Temp (C)"])

    def test_handle_sentinels(self, sample_dataframe_with_sentinels):
        """Test sentinel value replacement."""
        cleaner = DataCleaner()
        df_cleaned = cleaner.handle_sentinels(sample_dataframe_with_sentinels)
        
        assert pd.isna(df_cleaned.loc[1, "Sol Rad (W/sq.m)"])
        assert pd.isna(df_cleaned.loc[3, "Sol Rad (W/sq.m)"])
        assert pd.isna(df_cleaned.loc[2, "Soil Temp (C)"])
        
        assert not pd.isna(df_cleaned.loc[0, "Sol Rad (W/sq.m)"])
        assert not pd.isna(df_cleaned.loc[4, "Soil Temp (C)"])

    def test_handle_sentinels_custom_values(self):
        """Test sentinel handling with custom values."""
        df = pd.DataFrame({"value": [1.0, -100, 2.0, -200]})
        cleaner = DataCleaner()
        df_cleaned = cleaner.handle_sentinels(df, sentinel_values=[-100, -200])
        
        assert pd.isna(df_cleaned.loc[1, "value"])
        assert pd.isna(df_cleaned.loc[3, "value"])

    def test_impute_missing_forward_fill(self, sample_dataframe_with_missing):
        """Test forward fill imputation."""
        cleaner = DataCleaner()
        df_imputed = cleaner.impute_missing(
            sample_dataframe_with_missing,
            strategy="forward_fill"
        )
        
        # First NaN should be filled with 10.0, second with 10.0
        assert df_imputed.loc[1, "Air Temp (C)"] == 10.0
        assert df_imputed.loc[2, "Air Temp (C)"] == 10.0
        # Third NaN should be filled with 14.0
        assert df_imputed.loc[5, "Air Temp (C)"] == 14.0

    def test_impute_missing_mean(self):
        """Test mean imputation."""
        df = pd.DataFrame({
            "value": [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        cleaner = DataCleaner()
        df_imputed = cleaner.impute_missing(df, strategy="mean")
        
        mean_val = (1.0 + 3.0 + 5.0) / 3
        assert df_imputed.loc[1, "value"] == mean_val
        assert df_imputed.loc[3, "value"] == mean_val

    def test_impute_missing_median(self):
        """Test median imputation."""
        df = pd.DataFrame({
            "value": [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        cleaner = DataCleaner()
        df_imputed = cleaner.impute_missing(df, strategy="median")
        
        median_val = 3.0
        assert df_imputed.loc[1, "value"] == median_val
        assert df_imputed.loc[3, "value"] == median_val

    def test_handle_outliers_iqr(self):
        """Test IQR-based outlier removal."""
        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        cleaner = DataCleaner()
        df_cleaned = cleaner.handle_outliers(df, columns=["value"], method="iqr")
        
        assert pd.isna(df_cleaned.loc[5, "value"])

    def test_clean_pipeline(self, sample_dataframe_with_qc):
        """Test complete cleaning pipeline."""
        cleaner = DataCleaner()
        df_cleaned = cleaner.clean_pipeline(
            sample_dataframe_with_qc,
            apply_qc=True,
            handle_sentinels=False,
            handle_outliers=False,
            impute_missing=True,
            imputation_strategy="forward_fill"
        )
        
        # Should have applied QC filtering and imputation
        assert len(df_cleaned) == len(sample_dataframe_with_qc)
        # R, M, S should be NaN initially, then filled
        assert not pd.isna(df_cleaned.loc[2, "Air Temp (C)"])

