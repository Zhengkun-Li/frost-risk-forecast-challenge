"""Generate frost event labels for multi-horizon forecasting."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path


class FrostLabelGenerator:
    """Generate frost event labels for different forecast horizons."""
    
    def __init__(self, frost_threshold: float = 0.0):
        """Initialize frost label generator.
        
        Args:
            frost_threshold: Temperature threshold for frost (default: 0.0°C).
        """
        self.frost_threshold = frost_threshold
        self.horizons = [3, 6, 12, 24]  # hours
    
    def create_frost_labels(
        self,
        df: pd.DataFrame,
        temp_col: str = "Air Temp (C)",
        date_col: str = "Date",
        station_col: str = "Stn Id",
        horizons: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Create frost event labels for multiple forecast horizons.
        
        Args:
            df: Input DataFrame with temperature and date columns.
            temp_col: Name of temperature column.
            date_col: Name of date column.
            station_col: Name of station ID column.
            horizons: List of forecast horizons in hours (default: [3, 6, 12, 24]).
        
        Returns:
            DataFrame with added frost label columns for each horizon.
        """
        if horizons is None:
            horizons = self.horizons
        
        df = df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by station and date
        df = df.sort_values([station_col, date_col]).reset_index(drop=True)
        
        # Create datetime index for each station
        for station_id in df[station_col].unique():
            station_mask = df[station_col] == station_id
            station_df = df[station_mask].copy()
            
            # Create datetime index
            station_df = station_df.set_index(date_col)
            
            # For each horizon, check if temperature will be below threshold
            for horizon_h in horizons:
                label_col = f"frost_{horizon_h}h"
                temp_future_col = f"temp_{horizon_h}h"
                
                # Shift temperature forward by horizon hours
                # Negative shift means looking into the future
                future_temp = station_df[temp_col].shift(-horizon_h)
                
                # Create frost label (1 if temp < threshold, 0 otherwise)
                frost_label = (future_temp < self.frost_threshold).astype(int)
                
                # Update original dataframe
                station_indices = df[station_mask].index
                df.loc[station_indices, label_col] = frost_label.values
                df.loc[station_indices, temp_future_col] = future_temp.values
        
        return df
    
    def get_label_columns(self, horizons: Optional[List[int]] = None) -> List[str]:
        """Get column names for frost labels.
        
        Args:
            horizons: List of forecast horizons (default: [3, 6, 12, 24]).
        
        Returns:
            List of label column names.
        """
        if horizons is None:
            horizons = self.horizons
        return [f"frost_{h}h" for h in horizons]
    
    def get_temp_columns(self, horizons: Optional[List[int]] = None) -> List[str]:
        """Get column names for future temperatures.
        
        Args:
            horizons: List of forecast horizons (default: [3, 6, 12, 24]).
        
        Returns:
            List of temperature column names.
        """
        if horizons is None:
            horizons = self.horizons
        return [f"temp_{h}h" for h in horizons]


def create_multi_horizon_targets(
    df: pd.DataFrame,
    horizons: List[int] = [3, 6, 12, 24],
    frost_threshold: float = 0.0
) -> pd.DataFrame:
    """Convenience function to create multi-horizon frost labels.
    
    Args:
        df: Input DataFrame with temperature data.
        horizons: List of forecast horizons in hours.
        frost_threshold: Temperature threshold for frost.
    
    Returns:
        DataFrame with added frost label and temperature columns.
    """
    generator = FrostLabelGenerator(frost_threshold=frost_threshold)
    return generator.create_frost_labels(df, horizons=horizons)

