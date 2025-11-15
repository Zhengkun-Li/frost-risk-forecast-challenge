"""Feature engineering: time features, lags, rolling statistics, derived features."""

from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Create time-based and derived features for frost forecasting."""

    def __init__(self):
        """Initialize feature engineer."""
        pass

    def create_time_features(self, df: pd.DataFrame, date_col: str = "Date", inplace: bool = False) -> pd.DataFrame:
        """Create time-based features.
        
        Args:
            df: Input DataFrame with datetime column.
            date_col: Name of datetime column.
            inplace: If True, modify df in place (saves memory).
        
        Returns:
            DataFrame with added time features:
            - hour: Hour of day (0-23)
            - day_of_year: Day of year (1-366)
            - month: Month (1-12)
            - season: Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)
            - is_night: Boolean for night hours (18:00-06:00)
        """
        if not inplace:
            df = df.copy()
        
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        
        dt = pd.to_datetime(df[date_col])
        
        df["hour"] = dt.dt.hour
        df["day_of_year"] = dt.dt.dayofyear
        df["month"] = dt.dt.month
        df["day_of_week"] = dt.dt.dayofweek
        
        # Season: 1=Spring (Mar-May), 2=Summer (Jun-Aug), 3=Fall (Sep-Nov), 4=Winter (Dec-Feb)
        df["season"] = dt.dt.month.map({
            12: 4, 1: 4, 2: 4,  # Winter
            3: 1, 4: 1, 5: 1,    # Spring
            6: 2, 7: 2, 8: 2,    # Summer
            9: 3, 10: 3, 11: 3   # Fall
        })
        
        # Night hours: 18:00-06:00
        df["is_night"] = ((df["hour"] >= 18) | (df["hour"] < 6)).astype(int)
        
        # Cyclical encoding for hour and month (sine/cosine)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        return df

    def create_lag_features(self, df: pd.DataFrame, 
                           columns: List[str],
                           lags: List[int],
                           groupby_col: Optional[str] = "Stn Id") -> pd.DataFrame:
        """Create lagged features for specified columns.
        
        Args:
            df: Input DataFrame (must be sorted by date).
            columns: Columns to create lags for.
            lags: List of lag hours (e.g., [1, 3, 6, 12, 24]).
            groupby_col: Column to group by (usually "Stn Id" to avoid cross-station leakage).
        
        Returns:
            DataFrame with lagged features (e.g., "Air Temp (C)_lag_1").
        """
        df = df.copy()
        
        # Ensure sorted by date
        if "Date" in df.columns:
            if groupby_col:
                df = df.sort_values([groupby_col, "Date"]).reset_index(drop=True)
            else:
                df = df.sort_values("Date").reset_index(drop=True)
        
        # Collect all new features in a dictionary to avoid DataFrame fragmentation
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                lag_col_name = f"{col}_lag_{lag}"
                
                if groupby_col and groupby_col in df.columns:
                    # Create lag within each group
                    new_features[lag_col_name] = df.groupby(groupby_col)[col].shift(lag)
                else:
                    new_features[lag_col_name] = df[col].shift(lag)
        
        # Add all new features at once using pd.concat to avoid fragmentation
        if new_features:
            new_features_df = pd.DataFrame(new_features, index=df.index)
            df = pd.concat([df, new_features_df], axis=1)
        
        return df

    def create_rolling_features(self, df: pd.DataFrame,
                               columns: List[str],
                               windows: List[int],
                               functions: List[str] = ["mean", "min", "max", "std"],
                               groupby_col: Optional[str] = "Stn Id") -> pd.DataFrame:
        """Create rolling window statistics.
        
        Args:
            df: Input DataFrame (must be sorted by date).
            columns: Columns to create rolling features for.
            windows: List of window sizes in hours (e.g., [3, 6, 12, 24]).
            functions: List of functions to apply ("mean", "min", "max", "std").
            groupby_col: Column to group by.
        
        Returns:
            DataFrame with rolling features (e.g., "Air Temp (C)_rolling_6h_mean").
        """
        df = df.copy()
        
        # Ensure sorted by date
        if "Date" in df.columns:
            if groupby_col:
                df = df.sort_values([groupby_col, "Date"]).reset_index(drop=True)
            else:
                df = df.sort_values("Date").reset_index(drop=True)
        
        # Collect all new features in a dictionary to avoid DataFrame fragmentation
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                for func_name in functions:
                    feature_name = f"{col}_rolling_{window}h_{func_name}"
                    
                    if groupby_col and groupby_col in df.columns:
                        # Rolling within each group
                        if func_name == "mean":
                            new_features[feature_name] = df.groupby(groupby_col)[col].transform(
                                lambda x: x.rolling(window=window, min_periods=1).mean()
                            )
                        elif func_name == "min":
                            new_features[feature_name] = df.groupby(groupby_col)[col].transform(
                                lambda x: x.rolling(window=window, min_periods=1).min()
                            )
                        elif func_name == "max":
                            new_features[feature_name] = df.groupby(groupby_col)[col].transform(
                                lambda x: x.rolling(window=window, min_periods=1).max()
                            )
                        elif func_name == "std":
                            new_features[feature_name] = df.groupby(groupby_col)[col].transform(
                                lambda x: x.rolling(window=window, min_periods=1).std()
                            )
                        elif func_name == "sum":
                            new_features[feature_name] = df.groupby(groupby_col)[col].transform(
                                lambda x: x.rolling(window=window, min_periods=1).sum()
                            )
                    else:
                        # Rolling across all data
                        if func_name == "mean":
                            new_features[feature_name] = df[col].rolling(window=window, min_periods=1).mean()
                        elif func_name == "min":
                            new_features[feature_name] = df[col].rolling(window=window, min_periods=1).min()
                        elif func_name == "max":
                            new_features[feature_name] = df[col].rolling(window=window, min_periods=1).max()
                        elif func_name == "std":
                            new_features[feature_name] = df[col].rolling(window=window, min_periods=1).std()
                        elif func_name == "sum":
                            new_features[feature_name] = df[col].rolling(window=window, min_periods=1).sum()
        
        # Add all new features at once using pd.concat to avoid fragmentation
        if new_features:
            new_features_df = pd.DataFrame(new_features, index=df.index)
            df = pd.concat([df, new_features_df], axis=1)
        
        return df

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create physically derived features.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            DataFrame with derived features:
            - temp_dew_diff: Temperature - Dew Point (indicates humidity)
            - temp_humidity_index: Combined temperature-humidity index
            - wind_chill: Wind chill effect
            - heat_index: Heat index (for high temps)
        """
        df = df.copy()
        
        # Temperature - Dew Point difference (indicates relative humidity)
        if "Air Temp (C)" in df.columns and "Dew Point (C)" in df.columns:
            df["temp_dew_diff"] = df["Air Temp (C)"] - df["Dew Point (C)"]
        
        # Wind chill (simplified formula for low temperatures)
        if "Air Temp (C)" in df.columns and "Wind Speed (m/s)" in df.columns:
            # Convert wind speed from m/s to km/h
            wind_kmh = df["Wind Speed (m/s)"] * 3.6
            temp = df["Air Temp (C)"]
            # Wind chill formula (approximate)
            df["wind_chill"] = np.where(
                temp < 10,
                13.12 + 0.6215 * temp - 11.37 * (wind_kmh ** 0.16) + 0.3965 * temp * (wind_kmh ** 0.16),
                temp
            )
        
        # Heat index (for high temperatures and humidity)
        if "Air Temp (C)" in df.columns and "Rel Hum (%)" in df.columns:
            temp_f = df["Air Temp (C)"] * 9/5 + 32  # Convert to Fahrenheit
            humidity = df["Rel Hum (%)"]
            # Simplified heat index formula
            df["heat_index"] = np.where(
                (temp_f > 80) & (humidity > 40),
                -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity
                - 0.22475541 * temp_f * humidity - 6.83783e-3 * temp_f**2
                - 5.481717e-2 * humidity**2,
                temp_f
            )
        
        # Temperature change rate (if lag features exist)
        if "Air Temp (C)_lag_1" in df.columns:
            df["temp_change_rate"] = df["Air Temp (C)"] - df["Air Temp (C)_lag_1"]
        
        # Soil-air temperature difference
        if "Air Temp (C)" in df.columns and "Soil Temp (C)" in df.columns:
            df["soil_air_temp_diff"] = df["Soil Temp (C)"] - df["Air Temp (C)"]
        
        return df
    
    def create_radiation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create radiation-related features for frost prediction.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            DataFrame with radiation features:
            - daily_solar_radiation: Cumulative solar radiation from sunrise
            - sol_rad_change_rate: Rate of change in solar radiation
            - nighttime_cooling_rate: Temperature decline rate during night
            - radiation_temp_interaction: Interaction between radiation and temperature
        """
        df = df.copy()
        
        if "Sol Rad (W/sq.m)" not in df.columns:
            return df
        
        # Ensure sorted by date and station
        if "Date" in df.columns and "Stn Id" in df.columns:
            df = df.sort_values(["Stn Id", "Date"]).reset_index(drop=True)
        
        # Solar radiation change rate
        if "Sol Rad (W/sq.m)_lag_1" in df.columns:
            df["sol_rad_change_rate"] = np.where(
                df["Sol Rad (W/sq.m)_lag_1"] > 0,
                (df["Sol Rad (W/sq.m)"] - df["Sol Rad (W/sq.m)_lag_1"]) / df["Sol Rad (W/sq.m)_lag_1"],
                0
            )
        else:
            # Create lag_1 if not exists
            if "Stn Id" in df.columns:
                df["Sol Rad (W/sq.m)_lag_1"] = df.groupby("Stn Id")["Sol Rad (W/sq.m)"].shift(1)
            else:
                df["Sol Rad (W/sq.m)_lag_1"] = df["Sol Rad (W/sq.m)"].shift(1)
            df["sol_rad_change_rate"] = np.where(
                df["Sol Rad (W/sq.m)_lag_1"] > 0,
                (df["Sol Rad (W/sq.m)"] - df["Sol Rad (W/sq.m)_lag_1"]) / df["Sol Rad (W/sq.m)_lag_1"],
                0
            )
        
        # Daily cumulative solar radiation (from sunrise to current hour)
        # Simple approximation: sum from hour 6 (sunrise) to current hour
        if "hour" in df.columns:
            df["daily_solar_radiation"] = 0.0
            if "Stn Id" in df.columns:
                for station_id in df["Stn Id"].unique():
                    station_mask = df["Stn Id"] == station_id
                    station_df = df[station_mask].copy()
                    station_df = station_df.sort_values("Date").reset_index(drop=True)
                    
                    # For each day, calculate cumulative radiation from hour 6
                    for date in station_df["Date"].unique():
                        day_mask = (station_df["Date"] == date) & (station_df["hour"] >= 6)
                        if day_mask.any():
                            day_indices = station_df[day_mask].index
                            station_df.loc[day_indices, "daily_solar_radiation"] = (
                                station_df.loc[day_indices, "Sol Rad (W/sq.m)"].cumsum()
                            )
                    
                    df.loc[station_mask, "daily_solar_radiation"] = station_df["daily_solar_radiation"].values
            else:
                for date in df["Date"].unique():
                    day_mask = (df["Date"] == date) & (df["hour"] >= 6)
                    if day_mask.any():
                        df.loc[day_mask, "daily_solar_radiation"] = (
                            df.loc[day_mask, "Sol Rad (W/sq.m)"].cumsum()
                        )
        
        # Nighttime cooling rate (temperature decline during night hours)
        if "is_night" in df.columns and "Air Temp (C)" in df.columns:
            if "Air Temp (C)_lag_1" in df.columns:
                df["nighttime_cooling_rate"] = np.where(
                    df["is_night"] == 1,
                    df["Air Temp (C)"] - df["Air Temp (C)_lag_1"],
                    0
                )
            else:
                if "Stn Id" in df.columns:
                    df["Air Temp (C)_lag_1"] = df.groupby("Stn Id")["Air Temp (C)"].shift(1)
                else:
                    df["Air Temp (C)_lag_1"] = df["Air Temp (C)"].shift(1)
                df["nighttime_cooling_rate"] = np.where(
                    df["is_night"] == 1,
                    df["Air Temp (C)"] - df["Air Temp (C)_lag_1"],
                    0
                )
        
        # Radiation-temperature interaction
        if "Air Temp (C)" in df.columns:
            df["radiation_temp_interaction"] = df["Sol Rad (W/sq.m)"] * df["Air Temp (C)"]
        
        return df
    
    def create_wind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create wind-related features.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            DataFrame with wind features:
            - wind_dir_sin, wind_dir_cos: Cyclical encoding of wind direction
            - wind_dir_category: Categorical wind direction (N/E/S/W)
            - wind_speed_change_rate: Rate of change in wind speed
            - calm_wind_duration: Hours with calm wind in past N hours
            - wind_dir_temp_interaction: Interaction between wind direction and temperature
        """
        df = df.copy()
        
        # Wind direction cyclical encoding
        if "Wind Dir (0-360)" in df.columns:
            df["wind_dir_sin"] = np.sin(2 * np.pi * df["Wind Dir (0-360)"] / 360)
            df["wind_dir_cos"] = np.cos(2 * np.pi * df["Wind Dir (0-360)"] / 360)
            
            # Wind direction category: N (0-45, 315-360), E (45-135), S (135-225), W (225-315)
            # Handle the wrap-around case for North (0-45 and 315-360)
            wind_dir = df["Wind Dir (0-360)"].copy()
            df["wind_dir_category"] = np.where(
                (wind_dir >= 0) & (wind_dir < 45), 0,  # N
                np.where((wind_dir >= 45) & (wind_dir < 135), 1,  # E
                np.where((wind_dir >= 135) & (wind_dir < 225), 2,  # S
                np.where((wind_dir >= 225) & (wind_dir < 315), 3, 0))))  # W or N (315-360)
        
        # Wind speed change rate
        if "Wind Speed (m/s)" in df.columns:
            if "Wind Speed (m/s)_lag_1" in df.columns:
                df["wind_speed_change_rate"] = (
                    df["Wind Speed (m/s)"] - df["Wind Speed (m/s)_lag_1"]
                )
            else:
                if "Stn Id" in df.columns:
                    df["Wind Speed (m/s)_lag_1"] = df.groupby("Stn Id")["Wind Speed (m/s)"].shift(1)
                else:
                    df["Wind Speed (m/s)_lag_1"] = df["Wind Speed (m/s)"].shift(1)
                df["wind_speed_change_rate"] = (
                    df["Wind Speed (m/s)"] - df["Wind Speed (m/s)_lag_1"]
                )
        
        # Calm wind duration (hours with wind speed < 1.0 m/s in past 6 hours)
        if "Wind Speed (m/s)" in df.columns and "Stn Id" in df.columns:
            df["calm_wind_duration"] = 0.0
            for station_id in df["Stn Id"].unique():
                station_mask = df["Stn Id"] == station_id
                station_df = df[station_mask].copy()
                station_df = station_df.sort_values("Date").reset_index(drop=True)
                
                # Rolling window: count calm hours in past 6 hours
                calm_mask = station_df["Wind Speed (m/s)"] < 1.0
                station_df["calm_wind_duration"] = (
                    calm_mask.rolling(window=6, min_periods=1).sum()
                )
                
                df.loc[station_mask, "calm_wind_duration"] = station_df["calm_wind_duration"].values
        
        # Wind direction-temperature interaction
        if "wind_dir_sin" in df.columns and "Air Temp (C)" in df.columns:
            df["wind_dir_temp_interaction"] = df["wind_dir_sin"] * df["Air Temp (C)"]
        
        return df
    
    def create_humidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create humidity-related features.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            DataFrame with humidity features:
            - saturation_vapor_pressure: Saturation vapor pressure based on temperature
            - dew_point_proximity: Normalized (Air Temp - Dew Point) / Air Temp
            - humidity_change_rate: Rate of change in relative humidity
            - temp_humidity_interaction: Interaction between temperature and humidity
        """
        df = df.copy()
        
        # Saturation vapor pressure (based on temperature, using Clausius-Clapeyron)
        if "Air Temp (C)" in df.columns:
            # Simplified formula: e_s = 0.611 * exp(17.27 * T / (T + 237.3))
            T = df["Air Temp (C)"]
            df["saturation_vapor_pressure"] = 0.611 * np.exp(17.27 * T / (T + 237.3))
        
        # Dew point proximity (normalized)
        if "Air Temp (C)" in df.columns and "Dew Point (C)" in df.columns:
            df["dew_point_proximity"] = np.where(
                df["Air Temp (C)"] != 0,
                (df["Air Temp (C)"] - df["Dew Point (C)"]) / df["Air Temp (C)"],
                0
            )
        
        # Humidity change rate
        if "Rel Hum (%)" in df.columns:
            if "Rel Hum (%)_lag_1" in df.columns:
                df["humidity_change_rate"] = df["Rel Hum (%)"] - df["Rel Hum (%)_lag_1"]
            else:
                if "Stn Id" in df.columns:
                    df["Rel Hum (%)_lag_1"] = df.groupby("Stn Id")["Rel Hum (%)"].shift(1)
                else:
                    df["Rel Hum (%)_lag_1"] = df["Rel Hum (%)"].shift(1)
                df["humidity_change_rate"] = df["Rel Hum (%)"] - df["Rel Hum (%)_lag_1"]
        
        # Temperature-humidity interaction
        if "Air Temp (C)" in df.columns and "Rel Hum (%)" in df.columns:
            df["temp_humidity_interaction"] = df["Air Temp (C)"] * df["Rel Hum (%)"] / 100
        
        return df
    
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend and acceleration features.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            DataFrame with trend features:
            - temp_decline_rate: Average temperature decline rate over past 6 hours
            - cooling_acceleration: Second derivative of temperature (acceleration of cooling)
            - temp_trend: Temperature trend (rising/falling/stable)
        """
        df = df.copy()
        
        # Temperature decline rate (past 6 hours)
        if "Air Temp (C)" in df.columns:
            if "Air Temp (C)_lag_6" in df.columns:
                df["temp_decline_rate"] = (df["Air Temp (C)"] - df["Air Temp (C)_lag_6"]) / 6
            else:
                if "Stn Id" in df.columns:
                    df["Air Temp (C)_lag_6"] = df.groupby("Stn Id")["Air Temp (C)"].shift(6)
                else:
                    df["Air Temp (C)_lag_6"] = df["Air Temp (C)"].shift(6)
                df["temp_decline_rate"] = (df["Air Temp (C)"] - df["Air Temp (C)_lag_6"]) / 6
        
        # Cooling acceleration (second derivative)
        if "temp_decline_rate" in df.columns:
            if "Stn Id" in df.columns:
                df["temp_decline_rate_lag_1"] = df.groupby("Stn Id")["temp_decline_rate"].shift(1)
            else:
                df["temp_decline_rate_lag_1"] = df["temp_decline_rate"].shift(1)
            df["cooling_acceleration"] = df["temp_decline_rate"] - df["temp_decline_rate_lag_1"]
        
        # Temperature trend (categorical: -1=falling, 0=stable, 1=rising)
        if "temp_decline_rate" in df.columns:
            df["temp_trend"] = np.where(
                df["temp_decline_rate"] < -0.5, -1,  # Falling
                np.where(df["temp_decline_rate"] > 0.5, 1, 0)  # Rising or stable
            )
        
        return df
    
    def create_station_features(self, df: pd.DataFrame, 
                               metadata_path: Optional[Path] = None) -> pd.DataFrame:
        """Create station-related features.
        
        Args:
            df: Input DataFrame.
            metadata_path: Path to station metadata CSV file.
        
        Returns:
            DataFrame with station features:
            - station_id_encoded: Categorical encoding of station ID
            - region_encoded: Categorical encoding of CIMIS region
            - county_encoded: Categorical encoding of county
            - city_encoded: Categorical encoding of city
            - groundcover_encoded: Categorical encoding of ground cover
            - is_eto_station: Binary indicator for ETo station (0/1)
            - elevation_ft, elevation_m: Station elevation
            - latitude, longitude: GPS coordinates
            - latitude_cos, latitude_sin: Cyclical encoding of latitude
            - longitude_cos, longitude_sin: Cyclical encoding of longitude
            - distance_to_coast_approx: Approximate distance to coast
            - distance_to_nearest_station: Distance to nearest station
            - station_density: Number of stations within 50km radius
        """
        df = df.copy()
        
        # Station ID encoding (simple integer encoding, can be improved with one-hot or embedding)
        if "Stn Id" in df.columns:
            # Map station IDs to sequential integers starting from 0
            unique_stations = df["Stn Id"].unique()
            station_map = {sid: idx for idx, sid in enumerate(sorted(unique_stations))}
            df["station_id_encoded"] = df["Stn Id"].map(station_map)
        
        # Region encoding
        if "CIMIS Region" in df.columns:
            unique_regions = df["CIMIS Region"].unique()
            region_map = {region: idx for idx, region in enumerate(sorted(unique_regions))}
            df["region_encoded"] = df["CIMIS Region"].map(region_map)
        
        # Load station metadata if available
        if metadata_path is None:
            # Try default path
            default_path = Path("data/external/cimis_station_metadata.csv")
            if default_path.exists():
                metadata_path = default_path
        
        if metadata_path and Path(metadata_path).exists():
            try:
                metadata_df = pd.read_csv(metadata_path)
                
                # Merge elevation
                if "Elevation (ft)" in metadata_df.columns:
                    elevation_map = dict(zip(metadata_df["Stn Id"], metadata_df["Elevation (ft)"]))
                    # Convert to numeric, handling string values
                    elevation_map = {k: float(str(v).replace(',', '')) if pd.notna(v) else np.nan 
                                    for k, v in elevation_map.items()}
                    df["elevation_ft"] = df["Stn Id"].map(elevation_map)
                    # Convert to meters
                    df["elevation_m"] = df["elevation_ft"] * 0.3048
                
                # Merge GPS coordinates
                if "Latitude" in metadata_df.columns and "Longitude" in metadata_df.columns:
                    lat_map = dict(zip(metadata_df["Stn Id"], metadata_df["Latitude"]))
                    lon_map = dict(zip(metadata_df["Stn Id"], metadata_df["Longitude"]))
                    df["latitude"] = df["Stn Id"].map(lat_map)
                    df["longitude"] = df["Stn Id"].map(lon_map)
                    
                    # Cyclical encoding of latitude and longitude (for spatial patterns)
                    df["latitude_cos"] = np.cos(np.radians(df["latitude"]))
                    df["latitude_sin"] = np.sin(np.radians(df["latitude"]))
                    df["longitude_cos"] = np.cos(np.radians(df["longitude"]))
                    df["longitude_sin"] = np.sin(np.radians(df["longitude"]))
                    
                    # Approximate distance to coast (California coast is roughly at longitude -120)
                    # This is a simplified calculation
                    df["distance_to_coast_approx"] = np.abs(df["longitude"] - (-120.0)) * 111.0  # km
                    
                    # Distance to nearest station (spatial clustering)
                    # Calculate Haversine distance manually (scipy.cdist doesn't support haversine)
                    def haversine_distance(lat1, lon1, lat2, lon2):
                        """Calculate Haversine distance between two points in km."""
                        # Convert to radians
                        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                        
                        # Haversine formula
                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                        c = 2 * np.arcsin(np.sqrt(a))
                        r = 6371  # Earth radius in km
                        
                        return r * c
                    
                    station_coords = metadata_df[["Latitude", "Longitude"]].values
                    station_ids = metadata_df["Stn Id"].values
                    
                    nearest_distances = {}
                    station_density = {}
                    
                    # Calculate pairwise distances
                    for i, station_id in enumerate(station_ids):
                        lat1, lon1 = station_coords[i]
                        distances_list = []
                        
                        for j, other_station_id in enumerate(station_ids):
                            if i != j:
                                lat2, lon2 = station_coords[j]
                                # Calculate Haversine distance
                                dist = haversine_distance(lat1, lon1, lat2, lon2)
                                distances_list.append(dist)
                        
                        if distances_list:
                            nearest_distances[station_id] = np.min(distances_list)
                            station_density[station_id] = np.sum(np.array(distances_list) < 50)
                        else:
                            nearest_distances[station_id] = np.nan
                            station_density[station_id] = 0
                    
                    df["distance_to_nearest_station"] = df["Stn Id"].map(nearest_distances)
                    df["station_density"] = df["Stn Id"].map(station_density)
                
                # County encoding
                if "County" in metadata_df.columns:
                    county_map = dict(zip(metadata_df["Stn Id"], metadata_df["County"]))
                    unique_counties = sorted(metadata_df["County"].dropna().unique())
                    county_encoding = {county: idx for idx, county in enumerate(unique_counties)}
                    df["county_encoded"] = df["Stn Id"].map(county_map).map(county_encoding)
                
                # City encoding
                if "City" in metadata_df.columns:
                    city_map = dict(zip(metadata_df["Stn Id"], metadata_df["City"]))
                    unique_cities = sorted(metadata_df["City"].dropna().unique())
                    city_encoding = {city: idx for idx, city in enumerate(unique_cities)}
                    df["city_encoded"] = df["Stn Id"].map(city_map).map(city_encoding)
                
                # Ground cover encoding
                if "GroundCover" in metadata_df.columns:
                    groundcover_map = dict(zip(metadata_df["Stn Id"], metadata_df["GroundCover"]))
                    unique_groundcovers = sorted(metadata_df["GroundCover"].dropna().unique())
                    groundcover_encoding = {gc: idx for idx, gc in enumerate(unique_groundcovers)}
                    df["groundcover_encoded"] = df["Stn Id"].map(groundcover_map).map(groundcover_encoding)
                
                # IsEtoStation encoding (binary)
                if "IsEtoStation" in metadata_df.columns:
                    is_eto_map = dict(zip(metadata_df["Stn Id"], metadata_df["IsEtoStation"]))
                    # Convert to binary (True/False -> 1/0)
                    is_eto_binary = {k: 1 if str(v).lower() == 'true' else 0 
                                    for k, v in is_eto_map.items()}
                    df["is_eto_station"] = df["Stn Id"].map(is_eto_binary)
                
            except Exception as e:
                print(f"Warning: Could not load station metadata: {e}")
                import traceback
                traceback.print_exc()
        
        # Create interaction features between station features and weather variables
        df = self._create_station_weather_interactions(df)
        
        return df
    
    def _create_station_weather_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between station features and weather variables.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            DataFrame with interaction features:
            - elevation_temp_interaction: Elevation × Temperature
            - latitude_temp_interaction: Latitude × Temperature
            - distance_coast_temp_interaction: Distance to coast × Temperature
            - elevation_humidity_interaction: Elevation × Humidity
            - elevation_dewpoint_interaction: Elevation × Dew Point
        """
        df = df.copy()
        
        # Elevation-Temperature interaction (higher elevation = colder)
        if "elevation_m" in df.columns and "Air Temp (C)" in df.columns:
            df["elevation_temp_interaction"] = df["elevation_m"] * df["Air Temp (C)"]
        
        # Latitude-Temperature interaction (higher latitude = colder in winter)
        if "latitude" in df.columns and "Air Temp (C)" in df.columns:
            df["latitude_temp_interaction"] = df["latitude"] * df["Air Temp (C)"]
        
        # Distance to coast-Temperature interaction (coastal = warmer in winter)
        if "distance_to_coast_approx" in df.columns and "Air Temp (C)" in df.columns:
            df["distance_coast_temp_interaction"] = df["distance_to_coast_approx"] * df["Air Temp (C)"]
        
        # Elevation-Humidity interaction (higher elevation = lower humidity)
        if "elevation_m" in df.columns and "Rel Hum (%)" in df.columns:
            df["elevation_humidity_interaction"] = df["elevation_m"] * df["Rel Hum (%)"]
        
        # Elevation-Dew Point interaction (higher elevation = lower dew point)
        if "elevation_m" in df.columns and "Dew Point (C)" in df.columns:
            df["elevation_dewpoint_interaction"] = df["elevation_m"] * df["Dew Point (C)"]
        
        # Latitude-Humidity interaction
        if "latitude" in df.columns and "Rel Hum (%)" in df.columns:
            df["latitude_humidity_interaction"] = df["latitude"] * df["Rel Hum (%)"]
        
        # Distance to coast-Humidity interaction (coastal = higher humidity)
        if "distance_to_coast_approx" in df.columns and "Rel Hum (%)" in df.columns:
            df["distance_coast_humidity_interaction"] = df["distance_to_coast_approx"] * df["Rel Hum (%)"]
        
        return df

    def build_feature_set(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Main entry point: build all features based on configuration.
        
        Args:
            df: Input DataFrame.
            config: Configuration dictionary with feature engineering options:
                - time_features: bool
                - lag_features: dict with "columns" and "lags" lists
                - rolling_features: dict with "columns", "windows", "functions" lists
                - derived_features: bool
        
        Returns:
            DataFrame with all requested features.
        
        Example:
            >>> config = {
            ...     "time_features": True,
            ...     "lag_features": {
            ...         "columns": ["Air Temp (C)", "Dew Point (C)"],
            ...         "lags": [1, 3, 6, 12, 24]
            ...     },
            ...     "rolling_features": {
            ...         "columns": ["Air Temp (C)"],
            ...         "windows": [6, 12, 24],
            ...         "functions": ["mean", "min", "max"]
            ...     },
            ...     "derived_features": True
            ... }
            >>> engineer = FeatureEngineer()
            >>> df_features = engineer.build_feature_set(df, config)
        """
        # Use inplace operations where possible to save memory
        df_features = df.copy()
        
        # Time features
        if config.get("time_features", True):
            df_features = self.create_time_features(df_features, inplace=False)
        
        # Lag features
        if config.get("lag_features", {}).get("enabled", False):
            lag_config = config["lag_features"]
            df_features = self.create_lag_features(
                df_features,
                columns=lag_config.get("columns", []),
                lags=lag_config.get("lags", [1, 3, 6, 12, 24])
            )
        
        # Rolling features
        if config.get("rolling_features", {}).get("enabled", False):
            rolling_config = config["rolling_features"]
            df_features = self.create_rolling_features(
                df_features,
                columns=rolling_config.get("columns", []),
                windows=rolling_config.get("windows", [6, 12, 24]),
                functions=rolling_config.get("functions", ["mean", "min", "max", "std"])
            )
        
        # Derived features
        if config.get("derived_features", True):
            df_features = self.create_derived_features(df_features)
        
        # Radiation features (Priority 1)
        if config.get("radiation_features", False):
            df_features = self.create_radiation_features(df_features)
        
        # Wind features (Priority 1)
        if config.get("wind_features", False):
            df_features = self.create_wind_features(df_features)
        
        # Humidity features (Priority 2)
        if config.get("humidity_features", False):
            df_features = self.create_humidity_features(df_features)
        
        # Trend features (Priority 2)
        if config.get("trend_features", False):
            df_features = self.create_trend_features(df_features)
        
        # Station features (Priority 3)
        if config.get("station_features", False):
            metadata_path = config.get("station_metadata_path", None)
            if metadata_path:
                metadata_path = Path(metadata_path)
            df_features = self.create_station_features(df_features, metadata_path=metadata_path)
        
        # Optimize data types to reduce memory
        # Convert float64 to float32 where possible
        for col in df_features.select_dtypes(include=['float64']).columns:
            # Check if values fit in float32 range
            try:
                df_features[col] = df_features[col].astype('float32')
            except (ValueError, OverflowError):
                pass  # Keep as float64 if conversion fails
        
        # Convert int64 to smaller int types where possible
        for col in df_features.select_dtypes(include=['int64']).columns:
            if col not in ['Stn Id']:  # Keep grouping columns as int
                try:
                    df_features[col] = pd.to_numeric(df_features[col], downcast='integer')
                except (ValueError, OverflowError):
                    pass
        
        return df_features
    
    def engineer_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Alias for build_feature_set for backward compatibility.
        
        This method provides a simpler interface for feature engineering
        using the same configuration format as build_feature_set.
        
        Args:
            df: Input DataFrame.
            config: Configuration dictionary with feature engineering options.
                Supported keys:
                - create_time_features: bool (default: True)
                - create_lag_features: bool (default: True)
                - create_rolling_features: bool (default: True)
                - create_interaction_features: bool (default: False)
                - lag_periods: List[int] (default: [1, 3, 6, 12, 24])
                - rolling_windows: List[int] (default: [3, 6, 12, 24])
        
        Returns:
            DataFrame with all requested features.
        """
        # Convert simple config format to build_feature_set format
        feature_config = {
            "time_features": config.get("create_time_features", True),
            "lag_features": {
                "enabled": config.get("create_lag_features", True),
                "columns": config.get("lag_columns", None),  # If None, will use all numeric columns
                "lags": config.get("lag_periods", [1, 3, 6, 12, 24])
            },
            "rolling_features": {
                "enabled": config.get("create_rolling_features", True),
                "columns": config.get("rolling_columns", None),  # If None, will use all numeric columns
                "windows": config.get("rolling_windows", [3, 6, 12, 24]),
                "functions": config.get("rolling_functions", ["mean", "min", "max", "std"])
            },
            "derived_features": config.get("create_derived_features", True),
            "radiation_features": config.get("create_radiation_features", False),
            "wind_features": config.get("create_wind_features", False),
            "humidity_features": config.get("create_humidity_features", False),
            "trend_features": config.get("create_trend_features", False),
            "station_features": config.get("create_station_features", False),
            "station_metadata_path": config.get("station_metadata_path", None)
        }
        
        # If lag_columns or rolling_columns are not specified, use all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove date columns and ID columns
        numeric_cols = [col for col in numeric_cols if col not in ["Date", "Stn Id"]]
        
        if feature_config["lag_features"]["enabled"] and feature_config["lag_features"]["columns"] is None:
            feature_config["lag_features"]["columns"] = numeric_cols
        
        if feature_config["rolling_features"]["enabled"] and feature_config["rolling_features"]["columns"] is None:
            feature_config["rolling_features"]["columns"] = numeric_cols
        
        return self.build_feature_set(df, feature_config)

