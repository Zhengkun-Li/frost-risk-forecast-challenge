"""Data cleaning utilities: QC flags, sentinel values, missing data handling."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class DataCleaner:
    """Handle QC flags, sentinel values, and missing data imputation.
    
    QC Flag Rules (CIMIS):
    - Blank/Y: Keep (high quality)
    - M: Missing data -> mark as NaN
    - R: Rejected (extreme outlier) -> mark as NaN
    - S: Severe outlier -> mark as NaN
    - Q: Questionable -> configurable (default: mark as NaN)
    - P: Provisional -> configurable (default: mark as NaN)
    """

    # Default QC mapping: True means keep, False means mark as missing
    DEFAULT_QC_CONFIG = {
        "": True,  # Blank: keep
        "Y": True,  # Moderate outlier but accepted: keep
        "M": False,  # Missing: mark as NaN
        "R": False,  # Rejected: mark as NaN
        "S": False,  # Severe outlier: mark as NaN
        "Q": False,  # Questionable: mark as NaN (configurable)
        "P": False,  # Provisional: mark as NaN (configurable)
    }

    def __init__(self, qc_config: Optional[Dict[str, bool]] = None):
        """Initialize cleaner with QC configuration.
        
        Args:
            qc_config: Custom QC flag mapping. If None, uses DEFAULT_QC_CONFIG.
        """
        self.qc_config = qc_config or self.DEFAULT_QC_CONFIG.copy()

    def apply_qc_filter(self, df: pd.DataFrame, qc_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Filter data based on QC flags.
        
        For each variable with a corresponding QC column, mark values as NaN
        if the QC flag indicates low quality.
        
        Args:
            df: Input DataFrame.
            qc_columns: List of QC column names. If None, auto-detect (columns starting with 'qc').
        
        Returns:
            DataFrame with low-quality values marked as NaN.
        """
        df = df.copy()
        
        if qc_columns is None:
            # Auto-detect QC columns
            qc_columns = [col for col in df.columns if col.lower().startswith("qc")]
        
        # Map each variable to its QC column
        # Pattern: variable column followed by its QC column
        # Example: "Air Temp (C)" -> "qc" (or "qc.4" if numbered)
        variable_qc_map = {}
        
        for i, col in enumerate(df.columns):
            if col.lower().startswith("qc"):
                continue
            
            # Find corresponding QC column
            # Check if next column is a QC column
            if i + 1 < len(df.columns):
                next_col = df.columns[i + 1]
                if next_col.lower().startswith("qc"):
                    variable_qc_map[col] = next_col
        
        # Apply QC filtering
        for var_col, qc_col in variable_qc_map.items():
            if qc_col not in df.columns:
                continue
            
            # Create mask: True = keep, False = mark as NaN
            keep_mask = df[qc_col].apply(
                lambda x: self.qc_config.get(str(x).strip() if pd.notna(x) else "", True)
            )
            
            # Mark low-quality values as NaN
            df.loc[~keep_mask, var_col] = np.nan
        
        return df

    def handle_sentinels(self, df: pd.DataFrame, sentinel_values: Optional[List[float]] = None) -> pd.DataFrame:
        """Replace sentinel values with NaN.
        
        Args:
            df: Input DataFrame.
            sentinel_values: List of sentinel values to replace. Default: [-6999, -9999].
        
        Returns:
            DataFrame with sentinels replaced by NaN.
        """
        df = df.copy()
        
        if sentinel_values is None:
            sentinel_values = [-6999, -9999]
        
        # Replace sentinel values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].replace(sentinel_values, np.nan)
        
        return df

    def handle_outliers(self, df: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = "iqr",
                       factor: float = 3.0) -> pd.DataFrame:
        """Remove or cap outliers using IQR or Z-score method.
        
        Args:
            df: Input DataFrame.
            columns: Columns to process. If None, process all numeric columns.
            method: Method to use ("iqr" or "zscore").
            factor: Factor for outlier detection (3.0 for z-score, 1.5 for IQR multiplier).
        
        Returns:
            DataFrame with outliers handled.
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan
                
            elif method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df.loc[z_scores > factor, col] = np.nan
        
        return df

    def impute_missing(self, df: pd.DataFrame, 
                      strategy: str = "forward_fill",
                      columns: Optional[List[str]] = None,
                      **kwargs) -> pd.DataFrame:
        """Impute missing values using specified strategy.
        
        Args:
            df: Input DataFrame.
            strategy: Imputation strategy:
                - "forward_fill": Forward fill (default for time series)
                - "backward_fill": Backward fill
                - "mean": Fill with column mean
                - "median": Fill with column median
                - "interpolate": Linear interpolation
            columns: Columns to impute. If None, impute all numeric columns.
            **kwargs: Additional arguments for imputation method.
        
        Returns:
            DataFrame with imputed values.
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ensure data is sorted by date for time-based imputation
        if "Date" in df.columns:
            df = df.sort_values(["Stn Id", "Date"]).reset_index(drop=True)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if strategy == "forward_fill":
                # Forward fill within each station
                if "Stn Id" in df.columns:
                    df[col] = df.groupby("Stn Id")[col].ffill(**kwargs)
                else:
                    df[col] = df[col].ffill(**kwargs)
                    
            elif strategy == "backward_fill":
                if "Stn Id" in df.columns:
                    df[col] = df.groupby("Stn Id")[col].bfill(**kwargs)
                else:
                    df[col] = df[col].bfill(**kwargs)
                    
            elif strategy == "mean":
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                
            elif strategy == "median":
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                
            elif strategy == "interpolate":
                if "Stn Id" in df.columns:
                    df[col] = df.groupby("Stn Id")[col].apply(
                        lambda x: x.interpolate(method="linear", **kwargs)
                    )
                else:
                    df[col] = df[col].interpolate(method="linear", **kwargs)
        
        return df

    def clean_pipeline(self, df: pd.DataFrame, 
                      apply_qc: bool = True,
                      handle_sentinels: bool = True,
                      handle_outliers: bool = False,
                      impute_missing: bool = True,
                      imputation_strategy: str = "forward_fill") -> pd.DataFrame:
        """Complete cleaning pipeline.
        
        Args:
            df: Input DataFrame.
            apply_qc: Whether to apply QC filtering.
            handle_sentinels: Whether to replace sentinel values.
            handle_outliers: Whether to remove outliers.
            impute_missing: Whether to impute missing values.
            imputation_strategy: Strategy for imputation.
        
        Returns:
            Cleaned DataFrame.
        """
        df_cleaned = df.copy()
        
        if apply_qc:
            df_cleaned = self.apply_qc_filter(df_cleaned)
        
        if handle_sentinels:
            df_cleaned = self.handle_sentinels(df_cleaned)
        
        if handle_outliers:
            df_cleaned = self.handle_outliers(df_cleaned)
        
        if impute_missing:
            df_cleaned = self.impute_missing(df_cleaned, strategy=imputation_strategy)
        
        return df_cleaned

