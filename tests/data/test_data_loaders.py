"""Tests for data loading module."""

import pytest
from pathlib import Path
import pandas as pd
import tempfile
import os

from src.data.loaders import DataLoader


class TestDataLoader:
    """Test cases for DataLoader."""

    def test_load_raw_data_csv(self):
        """Test loading CSV file."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Stn Id,Air Temp (C)\n")
            f.write("2020-01-01,2,10.5\n")
            f.write("2020-01-02,2,11.0\n")
            temp_path = Path(f.name)
        
        try:
            df = DataLoader.load_raw_data(temp_path)
            assert len(df) == 2
            assert "Date" in df.columns
            assert pd.api.types.is_datetime64_any_dtype(df["Date"])
        finally:
            os.unlink(temp_path)

    def test_load_raw_data_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            DataLoader.load_raw_data(Path("nonexistent_file.csv"))

    def test_save_data_parquet(self):
        """Test saving DataFrame as Parquet."""
        try:
            import pyarrow
        except ImportError:
            pytest.skip("pyarrow not available, skipping parquet test")
        
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.parquet"
            DataLoader.save_data(df, output_path, format="parquet")
            assert output_path.exists()
            
            # Verify can load back
            df_loaded = pd.read_parquet(output_path)
            assert len(df_loaded) == 3

    def test_save_data_csv(self):
        """Test saving DataFrame as CSV."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            DataLoader.save_data(df, output_path, format="csv")
            assert output_path.exists()

