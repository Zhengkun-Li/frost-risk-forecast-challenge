"""Cross-validation strategies for time series and spatial data."""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

try:
    from sklearn.model_selection import GroupKFold, TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class CrossValidator:
    """Handle different cross-validation strategies for time series and spatial data."""
    
    @staticmethod
    def time_split(
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        date_col: str = "Date"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Time-based split (no data leakage).
        
        Args:
            df: Input DataFrame (must have date column).
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            date_col: Name of date column.
        
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        n = len(df_sorted)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
        
        return train_df, val_df, test_df
    
    @staticmethod
    def leave_one_station_out(
        df: pd.DataFrame,
        station_col: str = "Stn Id"
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Leave-One-Station-Out (LOSO) cross-validation.
        
        Args:
            df: Input DataFrame.
            station_col: Name of station ID column.
        
        Returns:
            List of (train_df, test_df) tuples, one for each station.
        """
        if station_col not in df.columns:
            raise ValueError(f"Station column '{station_col}' not found in DataFrame")
        
        stations = df[station_col].unique()
        splits = []
        
        for test_station in stations:
            train_df = df[df[station_col] != test_station].copy()
            test_df = df[df[station_col] == test_station].copy()
            splits.append((train_df, test_df))
        
        return splits
    
    @staticmethod
    def group_kfold(
        df: pd.DataFrame,
        n_splits: int = 5,
        group_col: str = "Stn Id"
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Group K-Fold cross-validation.
        
        Args:
            df: Input DataFrame.
            n_splits: Number of folds.
            group_col: Column to group by (e.g., station ID).
        
        Returns:
            List of (train_df, test_df) tuples.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for group_kfold. Install with: pip install scikit-learn")
        
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in DataFrame")
        
        groups = df[group_col].values
        gkf = GroupKFold(n_splits=n_splits)
        
        splits = []
        for train_idx, test_idx in gkf.split(df, groups=groups):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            splits.append((train_df, test_df))
        
        return splits
    
    @staticmethod
    def time_series_split(
        df: pd.DataFrame,
        n_splits: int = 5,
        date_col: str = "Date"
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Time Series Split cross-validation.
        
        Args:
            df: Input DataFrame (must be sorted by date).
            n_splits: Number of splits.
            date_col: Name of date column.
        
        Returns:
            List of (train_df, test_df) tuples.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for time_series_split. Install with: pip install scikit-learn")
        
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        splits = []
        for train_idx, test_idx in tscv.split(df_sorted):
            train_df = df_sorted.iloc[train_idx].copy()
            test_df = df_sorted.iloc[test_idx].copy()
            splits.append((train_df, test_df))
        
        return splits

