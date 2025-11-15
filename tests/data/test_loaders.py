"""Tests for data loading module."""

import pytest
from pathlib import Path
import pandas as pd
import tempfile
import os

from src.data.loaders import DataLoader


class TestDataLoader:
    """Test cases for DataLoader."""

    def test_load_raw_data_csv(self, temp_dir):
        """Test loading CSV file."""
        # Create temporary CSV
        csv_path = temp_dir / "test.csv"
        df_test = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=3, freq="h"),
            "Stn Id": [2, 2, 2],
            "Air Temp (C)": [10.5, 11.0, 11.5]
        })
        df_test.to_csv(csv_path, index=False)
        
        df = DataLoader.load_raw_data(csv_path)
        assert len(df) == 3
        assert "Date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])

    def test_load_raw_data_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            DataLoader.load_raw_data(Path("nonexistent_file.csv"))

    def test_load_raw_data_parse_dates(self, temp_dir):
        """Test date parsing."""
        csv_path = temp_dir / "test.csv"
        df_test = pd.DataFrame({
            "Date": ["2020-01-01", "2020-01-02"],
            "Value": [1, 2]
        })
        df_test.to_csv(csv_path, index=False)
        
        df = DataLoader.load_raw_data(csv_path)
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])

    def test_save_data_parquet(self, temp_dir):
        """Test saving DataFrame as Parquet."""
        try:
            import pyarrow
        except ImportError:
            pytest.skip("pyarrow not available, skipping parquet test")
        
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        output_path = temp_dir / "test.parquet"
        
        DataLoader.save_data(df, output_path, format="parquet")
        assert output_path.exists()
        
        # Verify can load back
        df_loaded = pd.read_parquet(output_path)
        assert len(df_loaded) == 3
        assert list(df_loaded.columns) == ["col1", "col2"]

    def test_save_data_csv(self, temp_dir):
        """Test saving DataFrame as CSV."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        output_path = temp_dir / "test.csv"
        
        DataLoader.save_data(df, output_path, format="csv")
        assert output_path.exists()
        
        # Verify can load back
        df_loaded = pd.read_csv(output_path)
        assert len(df_loaded) == 3

    def test_save_data_creates_directory(self, temp_dir):
        """Test that save_data creates parent directories."""
        df = pd.DataFrame({"col": [1, 2]})
        output_path = temp_dir / "subdir" / "test.csv"
        
        DataLoader.save_data(df, output_path, format="csv")
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_load_station_metadata(self, temp_dir):
        """Test loading station metadata."""
        metadata_path = temp_dir / "metadata.csv"
        df_meta = pd.DataFrame({
            "Stn Id": [2, 7],
            "Latitude": [36.5, 36.7],
            "Longitude": [-119.5, -119.7]
        })
        df_meta.to_csv(metadata_path, index=False)
        
        df = DataLoader.load_station_metadata(metadata_path)
        assert len(df) == 2
        assert "Stn Id" in df.columns

