"""Data loading utilities for CIMIS and external datasets."""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


class DataLoader:
    """Unified data loading interface for all data sources."""

    @staticmethod
    def load_raw_data(path: Path, **kwargs) -> pd.DataFrame:
        """Load raw CIMIS data from CSV file or stations directory.
        
        Args:
            path: Path to data file (CSV/CSV.gz) or stations directory.
            **kwargs: Additional arguments passed to pd.read_csv.
        
        Returns:
            DataFrame with parsed date column.
        
        Example:
            >>> loader = DataLoader()
            >>> # Load from single file
            >>> df = DataLoader.load_raw_data(Path("data/raw/cimis_all_stations.csv.gz"))
            >>> # Load from stations directory
            >>> df = DataLoader.load_raw_data(Path("data/raw/frost-risk-forecast-challenge/stations"))
        """
        path = Path(path)
        
        # If path is a directory, load all station files
        if path.is_dir():
            return DataLoader.load_from_stations_dir(path, **kwargs)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        # Parse dates by default
        parse_dates = kwargs.pop("parse_dates", ["Date"])
        
        df = pd.read_csv(path, parse_dates=parse_dates, **kwargs)
        
        # Ensure Date column is datetime
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        
        return df
    
    @staticmethod
    def load_from_stations_dir(stations_dir: Path, **kwargs) -> pd.DataFrame:
        """Load and combine all station CSV files from a directory.
        
        Args:
            stations_dir: Directory containing station CSV files.
            **kwargs: Additional arguments for pd.read_csv.
        
        Returns:
            Combined DataFrame with all stations.
        
        Example:
            >>> df = DataLoader.load_from_stations_dir(Path("data/raw/stations"))
        """
        stations_dir = Path(stations_dir)
        
        if not stations_dir.exists() or not stations_dir.is_dir():
            raise FileNotFoundError(f"Stations directory not found: {stations_dir}")
        
        # Find all CSV files
        csv_files = sorted(stations_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {stations_dir}")
        
        print(f"Loading {len(csv_files)} station files from {stations_dir}...")
        
        # Parse dates by default
        parse_dates = kwargs.pop("parse_dates", ["Date"])
        
        # Load and combine all files
        dfs = []
        for i, csv_file in enumerate(csv_files, 1):
            try:
                df = pd.read_csv(csv_file, parse_dates=parse_dates, **kwargs)
                dfs.append(df)
                if (i % 5 == 0) or (i == len(csv_files)):
                    print(f"  Loaded {i}/{len(csv_files)} files...", end='\r')
            except Exception as e:
                print(f"\nWarning: Failed to load {csv_file.name}: {e}")
                continue
        
        print(f"\nCombining {len(dfs)} station DataFrames...")
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Ensure Date column is datetime
        if "Date" in combined_df.columns:
            combined_df["Date"] = pd.to_datetime(combined_df["Date"])
        
        # Sort by station and date
        if "Stn Id" in combined_df.columns and "Date" in combined_df.columns:
            combined_df = combined_df.sort_values(["Stn Id", "Date"]).reset_index(drop=True)
        
        print(f"Combined data: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        print(f"Stations: {combined_df['Stn Id'].nunique() if 'Stn Id' in combined_df.columns else 'N/A'}")
        
        return combined_df

    @staticmethod
    def load_processed_data(path: Path, **kwargs) -> pd.DataFrame:
        """Load cleaned/processed data (supports CSV, Parquet, etc.).
        
        Args:
            path: Path to processed data file.
            **kwargs: Additional arguments for pd.read_csv or pd.read_parquet.
        
        Returns:
            Processed DataFrame.
        """
        if not path.exists():
            raise FileNotFoundError(f"Processed data file not found: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix == ".parquet":
            df = pd.read_parquet(path, **kwargs)
        elif suffix in [".csv", ".csv.gz"]:
            parse_dates = kwargs.pop("parse_dates", ["Date"])
            df = pd.read_csv(path, parse_dates=parse_dates, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        
        return df

    @staticmethod
    def load_station_metadata(path: Path) -> pd.DataFrame:
        """Load station metadata (coordinates, elevation, etc.).
        
        Args:
            path: Path to station metadata CSV.
        
        Returns:
            DataFrame with station information.
        """
        if not path.exists():
            raise FileNotFoundError(f"Station metadata not found: {path}")
        
        return pd.read_csv(path)

    @staticmethod
    def load_external_data(data_type: str, **kwargs) -> pd.DataFrame:
        """Load external data (ERA5, HRRR, etc.).
        
        Args:
            data_type: Type of external data ("era5", "hrrr", etc.).
            **kwargs: Data-specific parameters (dates, coordinates, etc.).
        
        Returns:
            DataFrame with external data.
        
        Note:
            This is a placeholder for future implementation.
            Actual implementation will depend on data source API.
        """
        if data_type.lower() == "era5":
            # TODO: Implement ERA5 loading
            raise NotImplementedError("ERA5 loading not yet implemented")
        elif data_type.lower() == "hrrr":
            # TODO: Implement HRRR loading
            raise NotImplementedError("HRRR loading not yet implemented")
        else:
            raise ValueError(f"Unknown external data type: {data_type}")

    @staticmethod
    def save_data(df: pd.DataFrame, path: Path, format: str = "parquet", **kwargs) -> None:
        """Save DataFrame to disk.
        
        Args:
            df: DataFrame to save.
            path: Output path.
            format: File format ("parquet", "csv", "csv.gz").
            **kwargs: Additional arguments for save function.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "parquet":
            df.to_parquet(path, index=False, **kwargs)
        elif format in ["csv", "csv.gz"]:
            df.to_csv(path, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Data saved to {path}")

