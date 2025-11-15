# Feature Reference Guide

## Complete Feature List and Implementation Guide

This document provides a comprehensive reference for all 298 features used in the frost forecasting model, including how to obtain them, their implementation methods, and their functional purposes.

**Last Updated**: 2025-11-12

**Total Features**: 298 (281 original + 17 new station features)

---

## Table of Contents

1. [Time Features (11)](#1-time-features-11)
2. [Lag Features (51)](#2-lag-features-51)
3. [Rolling Features (180)](#3-rolling-features-180)
4. [Derived Features (11)](#4-derived-features-11)
5. [Radiation Features (3)](#5-radiation-features-3)
6. [Wind Features (7)](#6-wind-features-7)
7. [Humidity Features (6)](#7-humidity-features-6)
8. [Trend Features (3)](#8-trend-features-3)
9. [Temperature Features (2)](#9-temperature-features-2)
10. [Station Features (24)](#10-station-features-24)
11. [Other Features (3)](#11-other-features-3)

---

## 1. Time Features (11)

### Overview
Time features capture temporal patterns and seasonal cycles in the data. These features help the model understand when frost events are more likely to occur based on time of day, day of year, and seasonal patterns.

### Implementation
**Method**: `create_time_features()` in `src/data/feature_engineering.py`

**How to Obtain**:
```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_time_features(df, date_col="Date")
```

### Feature List

| Feature Name | Type | Range/Values | Implementation | Function |
|--------------|------|--------------|----------------|----------|
| `Jul` | int | 1-366 | `df['Date'].dt.dayofyear` | Day of year (1-366) - captures annual cycle |
| `day_of_week` | int | 0-6 | `df['Date'].dt.dayofweek` | Day of week (0=Monday, 6=Sunday) - captures weekly patterns |
| `day_of_year` | int | 1-366 | `df['Date'].dt.dayofyear` | Same as Jul (alias) - annual cycle |
| `hour` | int | 0-23 | `df['Date'].dt.hour` | Hour of day (0-23) - captures diurnal cycle |
| `hour_cos` | float | -1 to 1 | `np.cos(2 * np.pi * hour / 24)` | Cyclical encoding of hour - smooth transition at midnight |
| `hour_sin` | float | -1 to 1 | `np.sin(2 * np.pi * hour / 24)` | Cyclical encoding of hour - captures hour pattern |
| `is_night` | int | 0 or 1 | `(hour >= 18) \| (hour < 6)` | Night indicator (18:00-06:00) - frost more likely at night |
| `month` | int | 1-12 | `df['Date'].dt.month` | Month (1-12) - captures seasonal patterns |
| `month_cos` | float | -1 to 1 | `np.cos(2 * np.pi * month / 12)` | Cyclical encoding of month - smooth transition at year boundary |
| `month_sin` | float | -1 to 1 | `np.sin(2 * np.pi * month / 12)` | Cyclical encoding of month - captures seasonal cycle |
| `season` | int | 1-4 | `(month // 3) % 4 + 1` | Season (1=Spring, 2=Summer, 3=Fall, 4=Winter) - frost more likely in winter |

### Functional Purpose
- **Diurnal Cycle**: `hour`, `hour_cos`, `hour_sin`, `is_night` capture daily temperature patterns (coldest at night)
- **Annual Cycle**: `month`, `month_cos`, `month_sin`, `season`, `day_of_year` capture seasonal patterns (coldest in winter)
- **Cyclical Encoding**: Prevents discontinuity at boundaries (e.g., hour 23 → 0, month 12 → 1)

---

## 2. Lag Features (51)

### Overview
Lag features capture historical values at specific time points in the past. They help the model understand recent trends and patterns that influence current conditions.

### Implementation
**Method**: `create_lag_features()` in `src/data/feature_engineering.py`

**How to Obtain**:
```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_lag_features(
    df,
    columns=["Air Temp (C)", "Dew Point (C)", ...],
    lags=[1, 3, 6, 12, 24],
    groupby_col="Stn Id"
)
```

**Implementation Details**:
```python
# For each column and lag value
for col in columns:
    for lag in lags:
        # Shift values by lag hours (within each station)
        df[f"{col}_lag_{lag}"] = df.groupby("Stn Id")[col].shift(lag)
```

### Feature Categories

#### Air Temp (C) - 5 features
| Feature Name | Lag Hours | Implementation | Function |
|--------------|-----------|----------------|----------|
| `Air Temp (C)_lag_1` | 1 hour ago | `df.groupby("Stn Id")["Air Temp (C)"].shift(1)` | Recent temperature trend |
| `Air Temp (C)_lag_3` | 3 hours ago | `df.groupby("Stn Id")["Air Temp (C)"].shift(3)` | Short-term temperature pattern |
| `Air Temp (C)_lag_6` | 6 hours ago | `df.groupby("Stn Id")["Air Temp (C)"].shift(6)` | Medium-term temperature trend |
| `Air Temp (C)_lag_12` | 12 hours ago | `df.groupby("Stn Id")["Air Temp (C)"].shift(12)` | Half-day temperature cycle |
| `Air Temp (C)_lag_24` | 24 hours ago | `df.groupby("Stn Id")["Air Temp (C)"].shift(24)` | Daily temperature cycle |

#### Dew Point (C) - 5 features
| Feature Name | Lag Hours | Function |
|--------------|-----------|----------|
| `Dew Point (C)_lag_1` | 1 hour ago | Recent dew point (affects frost formation) |
| `Dew Point (C)_lag_3` | 3 hours ago | Short-term dew point trend |
| `Dew Point (C)_lag_6` | 6 hours ago | Medium-term dew point pattern |
| `Dew Point (C)_lag_12` | 12 hours ago | Half-day dew point cycle |
| `Dew Point (C)_lag_24` | 24 hours ago | Daily dew point cycle |

#### ETo (mm) - 5 features
| Feature Name | Lag Hours | Function |
|--------------|-----------|----------|
| `ETo (mm)_lag_1` | 1 hour ago | Recent evapotranspiration (affects humidity) |
| `ETo (mm)_lag_3` | 3 hours ago | Short-term ETo trend |
| `ETo (mm)_lag_6` | 6 hours ago | Medium-term ETo pattern |
| `ETo (mm)_lag_12` | 12 hours ago | Half-day ETo cycle |
| `ETo (mm)_lag_24` | 24 hours ago | Daily ETo cycle |

#### Precip (mm) - 5 features
| Feature Name | Lag Hours | Function |
|--------------|-----------|----------|
| `Precip (mm)_lag_1` | 1 hour ago | Recent precipitation (affects temperature) |
| `Precip (mm)_lag_3` | 3 hours ago | Short-term precipitation trend |
| `Precip (mm)_lag_6` | 6 hours ago | Medium-term precipitation pattern |
| `Precip (mm)_lag_12` | 12 hours ago | Half-day precipitation cycle |
| `Precip (mm)_lag_24` | 24 hours ago | Daily precipitation cycle |

#### Rel Hum (%) - 5 features
| Feature Name | Lag Hours | Function |
|--------------|-----------|----------|
| `Rel Hum (%)_lag_1` | 1 hour ago | Recent humidity (affects frost formation) |
| `Rel Hum (%)_lag_3` | 3 hours ago | Short-term humidity trend |
| `Rel Hum (%)_lag_6` | 6 hours ago | Medium-term humidity pattern |
| `Rel Hum (%)_lag_12` | 12 hours ago | Half-day humidity cycle |
| `Rel Hum (%)_lag_24` | 24 hours ago | Daily humidity cycle |

#### Soil Temp (C) - 5 features
| Feature Name | Lag Hours | Function |
|--------------|-----------|----------|
| `Soil Temp (C)_lag_1` | 1 hour ago | Recent soil temperature (affects air temperature) |
| `Soil Temp (C)_lag_3` | 3 hours ago | Short-term soil temperature trend |
| `Soil Temp (C)_lag_6` | 6 hours ago | Medium-term soil temperature pattern |
| `Soil Temp (C)_lag_12` | 12 hours ago | Half-day soil temperature cycle |
| `Soil Temp (C)_lag_24` | 24 hours ago | Daily soil temperature cycle |

#### Sol Rad (W/sq.m) - 5 features
| Feature Name | Lag Hours | Function |
|--------------|-----------|----------|
| `Sol Rad (W/sq.m)_lag_1` | 1 hour ago | Recent solar radiation (affects temperature) |
| `Sol Rad (W/sq.m)_lag_3` | 3 hours ago | Short-term solar radiation trend |
| `Sol Rad (W/sq.m)_lag_6` | 6 hours ago | Medium-term solar radiation pattern |
| `Sol Rad (W/sq.m)_lag_12` | 12 hours ago | Half-day solar radiation cycle |
| `Sol Rad (W/sq.m)_lag_24` | 24 hours ago | Daily solar radiation cycle |

#### Vap Pres (kPa) - 5 features
| Feature Name | Lag Hours | Function |
|--------------|-----------|----------|
| `Vap Pres (kPa)_lag_1` | 1 hour ago | Recent vapor pressure (affects humidity) |
| `Vap Pres (kPa)_lag_3` | 3 hours ago | Short-term vapor pressure trend |
| `Vap Pres (kPa)_lag_6` | 6 hours ago | Medium-term vapor pressure pattern |
| `Vap Pres (kPa)_lag_12` | 12 hours ago | Half-day vapor pressure cycle |
| `Vap Pres (kPa)_lag_24` | 24 hours ago | Daily vapor pressure cycle |

#### Wind Dir (0-360) - 5 features
| Feature Name | Lag Hours | Function |
|--------------|-----------|----------|
| `Wind Dir (0-360)_lag_1` | 1 hour ago | Recent wind direction (affects temperature advection) |
| `Wind Dir (0-360)_lag_3` | 3 hours ago | Short-term wind direction trend |
| `Wind Dir (0-360)_lag_6` | 6 hours ago | Medium-term wind direction pattern |
| `Wind Dir (0-360)_lag_12` | 12 hours ago | Half-day wind direction cycle |
| `Wind Dir (0-360)_lag_24` | 24 hours ago | Daily wind direction cycle |

#### Wind Speed (m/s) - 5 features
| Feature Name | Lag Hours | Function |
|--------------|-----------|----------|
| `Wind Speed (m/s)_lag_1` | 1 hour ago | Recent wind speed (affects temperature mixing) |
| `Wind Speed (m/s)_lag_3` | 3 hours ago | Short-term wind speed trend |
| `Wind Speed (m/s)_lag_6` | 6 hours ago | Medium-term wind speed pattern |
| `Wind Speed (m/s)_lag_12` | 12 hours ago | Half-day wind speed cycle |
| `Wind Speed (m/s)_lag_24` | 24 hours ago | Daily wind speed cycle |

#### Derived Feature Lag - 1 feature
| Feature Name | Lag Hours | Function |
|--------------|-----------|----------|
| `temp_decline_rate_lag_1` | 1 hour ago | Recent temperature decline rate (trend indicator) |

### Functional Purpose
- **Historical Context**: Captures what happened in the past at specific time points
- **Trend Detection**: Helps identify rising or falling trends (e.g., temperature dropping)
- **Cyclical Patterns**: Captures daily cycles (lag_24) and shorter-term patterns
- **Temporal Dependencies**: Models how past conditions influence current and future conditions

### Why These Specific Lags?
- **1 hour**: Immediate past (recent trend)
- **3 hours**: Short-term pattern (quarter of a day)
- **6 hours**: Medium-term trend (half of half-day)
- **12 hours**: Half-day cycle (opposite time of day)
- **24 hours**: Daily cycle (same time yesterday)

---

## 3. Rolling Features (180)

### Overview
Rolling features calculate statistics over a sliding window of time. They capture recent trends, variability, and patterns in the data.

### Implementation
**Method**: `create_rolling_features()` in `src/data/feature_engineering.py`

**How to Obtain**:
```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_rolling_features(
    df,
    columns=["Air Temp (C)", "Dew Point (C)", ...],
    windows=[3, 6, 12, 24],
    functions=["mean", "min", "max", "std", "sum"],
    groupby_col="Stn Id"
)
```

**Implementation Details**:
```python
# For each column, window, and function
for col in columns:
    for window in windows:
        for func_name in functions:
            # Calculate rolling statistic within each station
            if func_name == "mean":
                df[f"{col}_rolling_{window}h_mean"] = (
                    df.groupby("Stn Id")[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                )
            # Similar for min, max, std, sum
```

### Feature Structure

For each of 9 variables × 4 windows × 5 functions = **180 features**

**Variables**: Air Temp (C), Dew Point (C), ETo (mm), Precip (mm), Rel Hum (%), Soil Temp (C), Sol Rad (W/sq.m), Vap Pres (kPa), Wind Speed (m/s)

**Windows**: 3h, 6h, 12h, 24h

**Functions**: mean, min, max, std, sum

### Example: Air Temp (C) - 20 features

| Feature Name | Window | Function | Implementation | Function |
|--------------|--------|----------|----------------|----------|
| `Air Temp (C)_rolling_3h_mean` | 3 hours | mean | `rolling(window=3).mean()` | Average temperature over last 3 hours |
| `Air Temp (C)_rolling_3h_min` | 3 hours | min | `rolling(window=3).min()` | Minimum temperature over last 3 hours |
| `Air Temp (C)_rolling_3h_max` | 3 hours | max | `rolling(window=3).max()` | Maximum temperature over last 3 hours |
| `Air Temp (C)_rolling_3h_std` | 3 hours | std | `rolling(window=3).std()` | Temperature variability over last 3 hours |
| `Air Temp (C)_rolling_3h_sum` | 3 hours | sum | `rolling(window=3).sum()` | Cumulative temperature over last 3 hours |
| `Air Temp (C)_rolling_6h_mean` | 6 hours | mean | `rolling(window=6).mean()` | Average temperature over last 6 hours |
| `Air Temp (C)_rolling_6h_min` | 6 hours | min | `rolling(window=6).min()` | Minimum temperature over last 6 hours |
| `Air Temp (C)_rolling_6h_max` | 6 hours | max | `rolling(window=6).max()` | Maximum temperature over last 6 hours |
| `Air Temp (C)_rolling_6h_std` | 6 hours | std | `rolling(window=6).std()` | Temperature variability over last 6 hours |
| `Air Temp (C)_rolling_6h_sum` | 6 hours | sum | `rolling(window=6).sum()` | Cumulative temperature over last 6 hours |
| `Air Temp (C)_rolling_12h_mean` | 12 hours | mean | `rolling(window=12).mean()` | Average temperature over last 12 hours |
| `Air Temp (C)_rolling_12h_min` | 12 hours | min | `rolling(window=12).min()` | Minimum temperature over last 12 hours |
| `Air Temp (C)_rolling_12h_max` | 12 hours | max | `rolling(window=12).max()` | Maximum temperature over last 12 hours |
| `Air Temp (C)_rolling_12h_std` | 12 hours | std | `rolling(window=12).std()` | Temperature variability over last 12 hours |
| `Air Temp (C)_rolling_12h_sum` | 12 hours | sum | `rolling(window=12).sum()` | Cumulative temperature over last 12 hours |
| `Air Temp (C)_rolling_24h_mean` | 24 hours | mean | `rolling(window=24).mean()` | Average temperature over last 24 hours |
| `Air Temp (C)_rolling_24h_min` | 24 hours | min | `rolling(window=24).min()` | Minimum temperature over last 24 hours |
| `Air Temp (C)_rolling_24h_max` | 24 hours | max | `rolling(window=24).max()` | Maximum temperature over last 24 hours |
| `Air Temp (C)_rolling_24h_std` | 24 hours | std | `rolling(window=24).std()` | Temperature variability over last 24 hours |
| `Air Temp (C)_rolling_24h_sum` | 24 hours | sum | `rolling(window=24).sum()` | Cumulative temperature over last 24 hours |

### Complete Feature List by Variable

**Same pattern for all 9 variables**:
- Dew Point (C): 20 features
- ETo (mm): 20 features
- Precip (mm): 20 features
- Rel Hum (%): 20 features
- Soil Temp (C): 20 features
- Sol Rad (W/sq.m): 20 features
- Vap Pres (kPa): 20 features
- Wind Speed (m/s): 20 features

**Total**: 9 variables × 20 features = **180 features**

### Functional Purpose
- **Mean**: Captures average conditions over the window (smoothing)
- **Min**: Captures minimum values (important for frost - lowest temperature)
- **Max**: Captures maximum values (daily high temperature)
- **Std**: Captures variability (temperature stability)
- **Sum**: Captures cumulative effects (e.g., cumulative precipitation)

### Why These Windows?
- **3 hours**: Short-term trend (recent conditions)
- **6 hours**: Medium-term trend (half of half-day)
- **12 hours**: Half-day cycle (opposite time of day)
- **24 hours**: Daily cycle (full day average)

---

## 4. Derived Features (11)

### Overview
Derived features are calculated from combinations of existing variables. They capture physical relationships and non-linear interactions between variables.

### Implementation
**Method**: `create_derived_features()` in `src/data/feature_engineering.py`

**How to Obtain**:
```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_derived_features(df)
```

### Feature List

| Feature Name | Formula | Implementation | Function |
|--------------|---------|----------------|----------|
| `temp_dew_diff` | `Air Temp (C) - Dew Point (C)` | `df["Air Temp (C)"] - df["Dew Point (C)"]` | Temperature-dew point difference (indicates humidity - smaller = higher humidity, more likely frost) |
| `wind_chill` | Simplified wind chill formula | `35.74 + 0.6215*T - 35.75*V^0.16 + 0.4275*T*V^0.16` | Wind chill effect (feels colder with wind) |
| `heat_index` | Simplified heat index formula | `0.5*(T + 61.0 + ((T-68.0)*1.2) + (RH*0.094))` | Heat index (for high temperatures, less relevant for frost) |
| `temp_humidity_index` | `Air Temp (C) * Rel Hum (%) / 100` | `df["Air Temp (C)"] * df["Rel Hum (%)"] / 100` | Combined temperature-humidity index |
| `dew_point_spread` | `Air Temp (C) - Dew Point (C)` | Same as temp_dew_diff | Alternative name for temperature-dew point difference |
| `soil_air_temp_diff` | `Soil Temp (C) - Air Temp (C)` | `df["Soil Temp (C)"] - df["Air Temp (C)"]` | Soil-air temperature difference (affects heat transfer) |

### Detailed Formulas

#### 1. temp_dew_diff
```python
df["temp_dew_diff"] = df["Air Temp (C)"] - df["Dew Point (C)"]
```
**Purpose**: 
- Small difference (< 2°C) indicates high humidity and potential frost
- Large difference (> 5°C) indicates dry air and less likely frost
- Critical for frost formation: frost occurs when temperature = dew point

#### 2. wind_chill
```python
# Convert wind speed from m/s to km/h
wind_kmh = df["Wind Speed (m/s)"] * 3.6
temp = df["Air Temp (C)"]
# Simplified wind chill formula (for low temperatures)
df["wind_chill"] = np.where(
    temp < 10,
    35.74 + 0.6215*temp - 35.75*(wind_kmh**0.16) + 0.4275*temp*(wind_kmh**0.16),
    temp  # No wind chill effect for warm temperatures
)
```
**Purpose**: 
- Accounts for wind effect on perceived temperature
- Higher wind speed = lower perceived temperature
- Important for frost: wind can accelerate cooling

#### 3. heat_index
```python
temp = df["Air Temp (C)"]
rh = df["Rel Hum (%)"]
# Simplified heat index (for high temperatures)
df["heat_index"] = np.where(
    temp > 27,
    0.5 * (temp + 61.0 + ((temp - 68.0) * 1.2) + (rh * 0.094)),
    temp  # No heat index for low temperatures
)
```
**Purpose**: 
- Less relevant for frost prediction (only applies to high temperatures)
- Kept for completeness

#### 4. temp_humidity_index
```python
df["temp_humidity_index"] = df["Air Temp (C)"] * df["Rel Hum (%)"] / 100
```
**Purpose**: 
- Combined temperature-humidity measure
- Higher values indicate more humid conditions
- Related to frost formation

#### 5. soil_air_temp_diff
```python
df["soil_air_temp_diff"] = df["Soil Temp (C)"] - df["Air Temp (C)"]
```
**Purpose**: 
- Positive: Soil warmer than air (heat transfer from soil to air)
- Negative: Air warmer than soil (heat transfer from air to soil)
- Affects frost formation: warm soil can prevent frost

### Functional Purpose
- **Physical Relationships**: Captures known physical relationships (e.g., temperature-dew point difference)
- **Non-linear Interactions**: Captures interactions between variables (e.g., wind chill)
- **Frost Indicators**: Direct indicators of frost conditions (e.g., small temp-dew point difference)

---

## 5. Radiation Features (3)

### Overview
Radiation features capture solar radiation patterns and their effects on temperature. Solar radiation is a key driver of temperature changes.

### Implementation
**Method**: `create_radiation_features()` in `src/data/feature_engineering.py`

**How to Obtain**:
```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_radiation_features(df)
```

### Feature List

| Feature Name | Formula | Implementation | Function |
|--------------|---------|----------------|----------|
| `sol_rad_change_rate` | `(Sol Rad(t) - Sol Rad(t-1)) / Sol Rad(t-1)` | `(df["Sol Rad"] - df["Sol Rad_lag_1"]) / df["Sol Rad_lag_1"]` | Rate of change in solar radiation (positive = increasing, negative = decreasing) |
| `daily_solar_radiation` | Cumulative sum from sunrise (hour 6) | `df.groupby(["Stn Id", "Date"])["Sol Rad"].cumsum()` | Daily cumulative solar radiation (affects temperature buildup) |
| `nighttime_cooling_rate` | `Air Temp(t) - Air Temp(t-1)` (night only) | `df["Air Temp"] - df["Air Temp_lag_1"]` (when is_night=1) | Temperature decline rate during night (critical for frost) |

### Detailed Formulas

#### 1. sol_rad_change_rate
```python
if "Sol Rad (W/sq.m)_lag_1" in df.columns:
    df["sol_rad_change_rate"] = np.where(
        df["Sol Rad (W/sq.m)_lag_1"] > 0,
        (df["Sol Rad (W/sq.m)"] - df["Sol Rad (W/sq.m)_lag_1"]) / df["Sol Rad (W/sq.m)_lag_1"],
        0
    )
```
**Purpose**: 
- Positive: Solar radiation increasing (sunrise, warming)
- Negative: Solar radiation decreasing (sunset, cooling)
- Zero: No change or no previous value
- Critical for frost: Decreasing radiation = cooling = frost risk

#### 2. daily_solar_radiation
```python
# For each day, calculate cumulative radiation from hour 6 (sunrise)
for date in df["Date"].unique():
    day_mask = (df["Date"] == date) & (df["hour"] >= 6)
    if day_mask.any():
        df.loc[day_mask, "daily_solar_radiation"] = (
            df.loc[day_mask, "Sol Rad (W/sq.m)"].cumsum()
        )
```
**Purpose**: 
- Cumulative solar energy received during the day
- Higher values = more energy absorbed = warmer temperatures
- Affects nighttime cooling: more daytime energy = slower cooling

#### 3. nighttime_cooling_rate
```python
df["nighttime_cooling_rate"] = np.where(
    df["is_night"] == 1,
    df["Air Temp (C)"] - df["Air Temp (C)_lag_1"],
    0
)
```
**Purpose**: 
- Temperature decline rate during night hours
- Negative values = cooling (normal at night)
- Large negative values = rapid cooling = frost risk
- Critical for frost prediction

### Functional Purpose
- **Radiation-Temperature Relationship**: Captures how solar radiation affects temperature
- **Cooling Patterns**: Identifies rapid cooling conditions (frost risk)
- **Daily Energy Balance**: Tracks energy accumulation during day

---

## 6. Wind Features (7)

### Overview
Wind features capture wind direction and speed patterns. Wind affects temperature through advection (movement of air masses) and mixing.

### Implementation
**Method**: `create_wind_features()` in `src/data/feature_engineering.py`

**How to Obtain**:
```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_wind_features(df)
```

### Feature List

| Feature Name | Formula | Implementation | Function |
|--------------|---------|----------------|----------|
| `wind_dir_sin` | `sin(Wind Dir * π / 180)` | `np.sin(np.radians(df["Wind Dir (0-360)"]))` | Cyclical encoding of wind direction (smooth transition at 0/360°) |
| `wind_dir_cos` | `cos(Wind Dir * π / 180)` | `np.cos(np.radians(df["Wind Dir (0-360)"]))` | Cyclical encoding of wind direction (captures direction pattern) |
| `wind_dir_category` | Categorical (N=0, E=1, S=2, W=3) | `np.where()` logic | Wind direction category (simplified: N, E, S, W) |
| `wind_speed_change_rate` | `Wind Speed(t) - Wind Speed(t-1)` | `df["Wind Speed"] - df["Wind Speed_lag_1"]` | Rate of change in wind speed (positive = increasing, negative = decreasing) |
| `calm_wind_duration` | Count of hours with wind < 1.0 m/s in past 6h | `rolling(window=6).apply(lambda x: sum(x < 1.0))` | Duration of calm wind conditions (calm = frost risk) |
| `wind_dir_temp_interaction` | `wind_dir_sin * Air Temp (C)` | `df["wind_dir_sin"] * df["Air Temp (C)"]` | Interaction between wind direction and temperature |
| `wind_speed_temp_interaction` | `Wind Speed * Air Temp (C)` | `df["Wind Speed"] * df["Air Temp (C)"]` | Interaction between wind speed and temperature |

### Detailed Formulas

#### 1. wind_dir_sin / wind_dir_cos
```python
df["wind_dir_sin"] = np.sin(np.radians(df["Wind Dir (0-360)"]))
df["wind_dir_cos"] = np.cos(np.radians(df["Wind Dir (0-360)"]))
```
**Purpose**: 
- Cyclical encoding prevents discontinuity at 0°/360°
- sin: Positive for E/SE, negative for W/NW
- cos: Positive for N/NE, negative for S/SW
- Captures directional patterns (e.g., cold winds from N)

#### 2. wind_dir_category
```python
# North: 0-45° and 315-360°
# East: 45-135°
# South: 135-225°
# West: 225-315°
df["wind_dir_category"] = np.where(
    (df["Wind Dir (0-360)"] >= 0) & (df["Wind Dir (0-360)"] < 45) | 
    (df["Wind Dir (0-360)"] >= 315) & (df["Wind Dir (0-360)"] <= 360),
    0,  # North
    np.where(
        (df["Wind Dir (0-360)"] >= 45) & (df["Wind Dir (0-360)"] < 135),
        1,  # East
        np.where(
            (df["Wind Dir (0-360)"] >= 135) & (df["Wind Dir (0-360)"] < 225),
            2,  # South
            3   # West
        )
    )
)
```
**Purpose**: 
- Simplified wind direction (4 categories)
- Useful for categorical encoding
- North winds often bring cold air (frost risk)

#### 3. calm_wind_duration
```python
# Count hours with wind speed < 1.0 m/s in past 6 hours
df["calm_wind_duration"] = df.groupby("Stn Id")["Wind Speed (m/s)"].transform(
    lambda x: x.rolling(window=6, min_periods=1).apply(
        lambda y: (y < 1.0).sum()
    )
)
```
**Purpose**: 
- Calm wind conditions favor frost formation
- Wind mixes air and prevents temperature stratification
- Longer calm periods = higher frost risk

### Functional Purpose
- **Wind Direction**: Captures where wind comes from (cold air from north)
- **Wind Speed**: Affects temperature mixing and heat loss
- **Calm Conditions**: Identifies conditions favorable for frost
- **Wind-Temperature Interaction**: Captures how wind affects temperature

---

## 7. Humidity Features (6)

### Overview
Humidity features capture moisture content in the air. Humidity is critical for frost formation (frost occurs when temperature reaches dew point).

### Implementation
**Method**: `create_humidity_features()` in `src/data/feature_engineering.py`

**How to Obtain**:
```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_humidity_features(df)
```

### Feature List

| Feature Name | Formula | Implementation | Function |
|--------------|---------|----------------|----------|
| `saturation_vapor_pressure` | `0.611 * exp(17.27 * T / (T + 237.3))` | Clausius-Clapeyron equation | Maximum vapor pressure at given temperature |
| `dew_point_proximity` | `(Air Temp - Dew Point) / Air Temp` | `(df["Air Temp"] - df["Dew Point"]) / df["Air Temp"]` | Normalized temperature-dew point difference (smaller = higher humidity) |
| `humidity_change_rate` | `Rel Hum(t) - Rel Hum(t-1)` | `df["Rel Hum"] - df["Rel Hum_lag_1"]` | Rate of change in relative humidity |
| `temp_humidity_interaction` | `Air Temp * Rel Hum / 100` | `df["Air Temp"] * df["Rel Hum"] / 100` | Interaction between temperature and humidity |
| `vap_pres_deficit` | `Saturation Vap Pres - Vap Pres` | `df["saturation_vapor_pressure"] - df["Vap Pres (kPa)"]` | Vapor pressure deficit (affects evaporation) |
| `relative_humidity_trend` | Rolling mean of Rel Hum | `rolling(window=6).mean()` | Recent humidity trend |

### Detailed Formulas

#### 1. saturation_vapor_pressure
```python
T = df["Air Temp (C)"]
df["saturation_vapor_pressure"] = 0.611 * np.exp(17.27 * T / (T + 237.3))
```
**Purpose**: 
- Maximum water vapor pressure at given temperature
- Based on Clausius-Clapeyron equation
- Used to calculate relative humidity and dew point

#### 2. dew_point_proximity
```python
df["dew_point_proximity"] = np.where(
    df["Air Temp (C)"] != 0,
    (df["Air Temp (C)"] - df["Dew Point (C)"]) / df["Air Temp (C)"],
    0
)
```
**Purpose**: 
- Normalized temperature-dew point difference
- Smaller values = higher humidity = closer to frost
- Critical indicator: When this approaches 0, frost is likely

#### 3. humidity_change_rate
```python
df["humidity_change_rate"] = df["Rel Hum (%)"] - df["Rel Hum (%)_lag_1"]
```
**Purpose**: 
- Rate of change in relative humidity
- Positive: Humidity increasing (more moisture)
- Negative: Humidity decreasing (drying)
- Affects frost formation: Increasing humidity = frost risk

### Functional Purpose
- **Humidity Measurement**: Captures moisture content in air
- **Frost Indicators**: Small temperature-dew point difference indicates frost risk
- **Humidity Trends**: Tracks changes in humidity over time
- **Physical Relationships**: Captures relationships between temperature, humidity, and dew point

---

## 8. Trend Features (3)

### Overview
Trend features capture rate of change and acceleration in temperature. They help identify rapid cooling conditions that lead to frost.

### Implementation
**Method**: `create_trend_features()` in `src/data/feature_engineering.py`

**How to Obtain**:
```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_trend_features(df)
```

### Feature List

| Feature Name | Formula | Implementation | Function |
|--------------|---------|----------------|----------|
| `temp_decline_rate` | `(Air Temp(t) - Air Temp(t-6)) / 6` | `(df["Air Temp"] - df["Air Temp_lag_6"]) / 6` | Average temperature decline rate over past 6 hours (°C/hour) |
| `cooling_acceleration` | `temp_decline_rate(t) - temp_decline_rate(t-1)` | `df["temp_decline_rate"] - df["temp_decline_rate_lag_1"]` | Second derivative of temperature (acceleration of cooling) |
| `temp_trend` | Categorical (-1=falling, 0=stable, 1=rising) | `np.where()` logic | Temperature trend category |

### Detailed Formulas

#### 1. temp_decline_rate
```python
df["temp_decline_rate"] = (df["Air Temp (C)"] - df["Air Temp (C)_lag_6"]) / 6
```
**Purpose**: 
- Average temperature decline rate over past 6 hours
- Negative values = cooling (normal at night)
- Large negative values = rapid cooling = frost risk
- Critical for frost prediction

#### 2. cooling_acceleration
```python
df["cooling_acceleration"] = df["temp_decline_rate"] - df["temp_decline_rate_lag_1"]
```
**Purpose**: 
- Second derivative of temperature
- Negative: Cooling accelerating (frost risk increasing)
- Positive: Cooling decelerating (frost risk decreasing)
- Zero: Constant cooling rate

#### 3. temp_trend
```python
df["temp_trend"] = np.where(
    df["temp_decline_rate"] < -0.5, -1,  # Falling
    np.where(df["temp_decline_rate"] > 0.5, 1, 0)  # Rising or stable
)
```
**Purpose**: 
- Categorical temperature trend
- -1: Falling (frost risk)
- 0: Stable
- 1: Rising (no frost risk)

### Functional Purpose
- **Cooling Detection**: Identifies rapid cooling conditions
- **Trend Analysis**: Tracks temperature trends over time
- **Frost Risk Indicators**: Rapid cooling = frost risk
- **Acceleration Analysis**: Identifies accelerating cooling (high frost risk)

---

## 9. Temperature Features (2)

### Overview
Temperature features are the raw temperature measurements. These are the primary predictors for frost events.

### Implementation
**Source**: Raw data columns

**How to Obtain**:
```python
# Direct from raw data
df["Air Temp (C)"]  # Air temperature
df["Soil Temp (C)"]  # Soil temperature
```

### Feature List

| Feature Name | Source | Range | Function |
|--------------|--------|-------|----------|
| `Air Temp (C)` | Raw data | Typically -5°C to 40°C | Primary predictor: Frost occurs when T < 0°C |
| `Soil Temp (C)` | Raw data | Typically -5°C to 40°C | Secondary predictor: Soil temperature affects air temperature |

### Functional Purpose
- **Primary Predictor**: Air temperature is the main predictor for frost (T < 0°C)
- **Heat Transfer**: Soil temperature affects air temperature through heat transfer
- **Baseline Feature**: All other features are derived to improve temperature prediction

---

## 10. Station Features (24)

### Overview
Station features capture geographical and spatial characteristics of each weather station. They help the model understand how location affects frost risk.

### Implementation
**Method**: `create_station_features()` in `src/data/feature_engineering.py`

**How to Obtain**:
```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_station_features(
    df,
    metadata_path="data/external/cimis_station_metadata.csv"
)
```

### Base Geographic Features (7)

| Feature Name | Source | Implementation | Function |
|--------------|--------|----------------|----------|
| `distance_to_coast_approx` | Calculated from longitude | `abs(longitude - (-120.0)) * 111.0` | Approximate distance to coast (km) - coastal areas warmer |
| `elevation_ft` | Station metadata | `metadata["Elevation (ft)"]` | Station elevation in feet - higher elevation = colder |
| `elevation_m` | Calculated from elevation_ft | `elevation_ft * 0.3048` | Station elevation in meters - higher elevation = colder |
| `latitude` | Station metadata | `metadata["Latitude"]` | Station latitude - higher latitude = colder |
| `longitude` | Station metadata | `metadata["Longitude"]` | Station longitude - affects distance to coast |
| `region_encoded` | CIMIS Region | Categorical encoding | CIMIS region (San Joaquin Valley, etc.) |
| `station_id_encoded` | Stn Id | Categorical encoding | Station ID (0-17) |

### New Geographic Features (10)

| Feature Name | Source | Implementation | Function |
|--------------|--------|----------------|----------|
| `county_encoded` | Station metadata | Categorical encoding | County (9 counties) - local climate patterns |
| `city_encoded` | Station metadata | Categorical encoding | City (17 cities) - local climate patterns |
| `groundcover_encoded` | Station metadata | Categorical encoding | Ground cover type (usually Grass) |
| `is_eto_station` | Station metadata | Binary (0/1) | Whether station is ETo station (1) or not (0) |
| `distance_to_nearest_station` | Calculated from coordinates | Haversine formula | Distance to nearest station (km) - spatial correlation |
| `station_density` | Calculated from coordinates | Count within 50km | Number of stations within 50km radius - spatial density |
| `latitude_cos` | Calculated from latitude | `cos(latitude * π / 180)` | Cyclical encoding of latitude |
| `latitude_sin` | Calculated from latitude | `sin(latitude * π / 180)` | Cyclical encoding of latitude |
| `longitude_cos` | Calculated from longitude | `cos(longitude * π / 180)` | Cyclical encoding of longitude |
| `longitude_sin` | Calculated from longitude | `sin(longitude * π / 180)` | Cyclical encoding of longitude |

### Station-Weather Interaction Features (7)

| Feature Name | Formula | Implementation | Function |
|--------------|---------|----------------|----------|
| `elevation_temp_interaction` | `elevation_m * Air Temp (C)` | `df["elevation_m"] * df["Air Temp (C)"]` | Elevation-temperature interaction (higher elevation = colder) |
| `latitude_temp_interaction` | `latitude * Air Temp (C)` | `df["latitude"] * df["Air Temp (C)"]` | Latitude-temperature interaction (higher latitude = colder) |
| `distance_coast_temp_interaction` | `distance_to_coast * Air Temp (C)` | `df["distance_to_coast_approx"] * df["Air Temp (C)"]` | Distance-coast-temperature interaction (coastal = warmer) |
| `elevation_humidity_interaction` | `elevation_m * Rel Hum (%)` | `df["elevation_m"] * df["Rel Hum (%)"]` | Elevation-humidity interaction (higher elevation = lower humidity) |
| `elevation_dewpoint_interaction` | `elevation_m * Dew Point (C)` | `df["elevation_m"] * df["Dew Point (C)"]` | Elevation-dew point interaction (higher elevation = lower dew point) |
| `latitude_humidity_interaction` | `latitude * Rel Hum (%)` | `df["latitude"] * df["Rel Hum (%)"]` | Latitude-humidity interaction |
| `distance_coast_humidity_interaction` | `distance_to_coast * Rel Hum (%)` | `df["distance_to_coast_approx"] * df["Rel Hum (%)"]` | Distance-coast-humidity interaction (coastal = higher humidity) |

### Detailed Implementation

#### 1. Distance to Coast
```python
# California coast is roughly at longitude -120
df["distance_to_coast_approx"] = np.abs(df["longitude"] - (-120.0)) * 111.0  # km
```
**Purpose**: 
- Coastal areas are warmer in winter (ocean moderates temperature)
- Inland areas are colder (continental climate)
- Distance to coast affects frost risk

#### 2. Distance to Nearest Station
```python
from scipy.spatial.distance import cdist

# Calculate pairwise distances between stations using Haversine formula
distances = cdist(station_coords, station_coords, metric='haversine') * 6371  # km
np.fill_diagonal(distances, np.inf)  # Exclude self

# Find nearest station for each station
nearest_distances = {station_id: np.min(distances[i]) for i, station_id in enumerate(station_ids)}
df["distance_to_nearest_station"] = df["Stn Id"].map(nearest_distances)
```
**Purpose**: 
- Spatial correlation between stations
- Nearby stations have similar conditions
- Can be used for spatial interpolation

#### 3. Station Density
```python
# Count stations within 50km radius
station_density = {
    station_id: np.sum(distances[i] < 50) 
    for i, station_id in enumerate(station_ids)
}
df["station_density"] = df["Stn Id"].map(station_density)
```
**Purpose**: 
- Spatial density of stations
- Higher density = more data available
- Reflects local microclimate characteristics

#### 4. Cyclical Encoding
```python
df["latitude_cos"] = np.cos(np.radians(df["latitude"]))
df["latitude_sin"] = np.sin(np.radians(df["latitude"]))
df["longitude_cos"] = np.cos(np.radians(df["longitude"]))
df["longitude_sin"] = np.sin(np.radians(df["longitude"]))
```
**Purpose**: 
- Cyclical encoding for spatial patterns
- Captures spatial periodicity
- Useful for capturing climate zones

### Functional Purpose
- **Geographic Effects**: Captures how location affects frost risk
- **Spatial Patterns**: Identifies spatial correlations between stations
- **Interaction Effects**: Captures how geography interacts with weather
- **Local Climate**: Captures local microclimate characteristics

---

## 11. Other Features (3)

### Overview
Other features are raw meteorological variables that don't fit into other categories but are important for frost prediction.

### Implementation
**Source**: Raw data columns

**How to Obtain**:
```python
# Direct from raw data
df["ETo (mm)"]  # Reference evapotranspiration
df["Precip (mm)"]  # Precipitation
df["Vap Pres (kPa)"]  # Vapor pressure
```

### Feature List

| Feature Name | Source | Range | Function |
|--------------|--------|-------|----------|
| `ETo (mm)` | Raw data | Typically 0-10 mm | Reference evapotranspiration - affects humidity |
| `Precip (mm)` | Raw data | Typically 0-50 mm | Precipitation - affects temperature and humidity |
| `Vap Pres (kPa)` | Raw data | Typically 0-5 kPa | Vapor pressure - affects humidity and dew point |

### Functional Purpose
- **Evapotranspiration**: Affects humidity levels
- **Precipitation**: Affects temperature (cooling) and humidity (increasing)
- **Vapor Pressure**: Direct measure of atmospheric moisture

---

## Feature Creation Workflow

### Complete Feature Engineering Pipeline

```python
from src.data.feature_engineering import FeatureEngineer
from pathlib import Path

# Initialize feature engineer
engineer = FeatureEngineer()

# Configure feature creation
config = {
    "time_features": True,
    "lag_features": {
        "enabled": True,
        "columns": [
            "Air Temp (C)", "Dew Point (C)", "ETo (mm)", "Precip (mm)",
            "Rel Hum (%)", "Soil Temp (C)", "Sol Rad (W/sq.m)",
            "Vap Pres (kPa)", "Wind Dir (0-360)", "Wind Speed (m/s)"
        ],
        "lags": [1, 3, 6, 12, 24]
    },
    "rolling_features": {
        "enabled": True,
        "columns": [
            "Air Temp (C)", "Dew Point (C)", "ETo (mm)", "Precip (mm)",
            "Rel Hum (%)", "Soil Temp (C)", "Sol Rad (W/sq.m)",
            "Vap Pres (kPa)", "Wind Speed (m/s)"
        ],
        "windows": [3, 6, 12, 24],
        "functions": ["mean", "min", "max", "std", "sum"]
    },
    "derived_features": True,
    "radiation_features": True,
    "wind_features": True,
    "humidity_features": True,
    "trend_features": True,
    "station_features": True,
    "station_metadata_path": "data/external/cimis_station_metadata.csv"
}

# Create all features
df_features = engineer.build_feature_set(df, config)

# Result: DataFrame with 298 features
print(f"Total features: {len(df_features.columns)}")
print(f"Features: {list(df_features.columns)}")
```

### Feature Creation Order

1. **Time Features** (11): Extract from date column
2. **Station Features** (24): Merge from metadata, calculate spatial features
3. **Lag Features** (51): Shift values by lag hours
4. **Rolling Features** (180): Calculate rolling statistics
5. **Derived Features** (11): Calculate from combinations
6. **Radiation Features** (3): Calculate from solar radiation
7. **Wind Features** (7): Calculate from wind data
8. **Humidity Features** (6): Calculate from humidity data
9. **Trend Features** (3): Calculate from temperature trends
10. **Temperature Features** (2): Use raw temperature data
11. **Other Features** (3): Use raw other data

### Dependencies

**Important**: Some features depend on others:
- **Lag features** must be created before rolling features (rolling uses lag_1)
- **Station features** must be created before interaction features
- **Derived features** may use lag features (e.g., temp_decline_rate uses lag_6)
- **Radiation features** use lag_1 (sol_rad_change_rate)
- **Wind features** use lag_1 (wind_speed_change_rate)
- **Humidity features** use lag_1 (humidity_change_rate)
- **Trend features** use lag_6 (temp_decline_rate)

---

## Feature Usage in Model

### Feature Selection

Not all features are equally important. Feature importance analysis can help identify the most predictive features.

### Feature Scaling

LightGBM doesn't require feature scaling, but some features may benefit from normalization:
- **Cyclical features**: Already normalized (-1 to 1)
- **Interaction features**: May benefit from normalization
- **Distance features**: May benefit from normalization

### Feature Importance

Common important features for frost prediction:
1. **Air Temp (C)**: Primary predictor
2. **Dew Point (C)**: Critical for frost formation
3. **temp_dew_diff**: Direct frost indicator
4. **Rolling min/max**: Recent temperature extremes
5. **nighttime_cooling_rate**: Rapid cooling indicator
6. **elevation_m**: Geographic effect
7. **distance_to_coast_approx**: Coastal moderation effect
8. **calm_wind_duration**: Calm conditions favor frost

---

## Summary

### Feature Count by Category

| Category | Count | Percentage |
|----------|-------|------------|
| Rolling Features | 180 | 60.4% |
| Lag Features | 51 | 17.1% |
| Station Features | 24 | 8.1% |
| Time Features | 11 | 3.7% |
| Derived Features | 11 | 3.7% |
| Wind Features | 7 | 2.3% |
| Humidity Features | 6 | 2.0% |
| Trend Features | 3 | 1.0% |
| Radiation Features | 3 | 1.0% |
| Other Features | 3 | 1.0% |
| Temperature Features | 2 | 0.7% |
| **Total** | **298** | **100%** |

### Key Takeaways

1. **Rolling features dominate** (60.4%): Capture recent trends and patterns
2. **Lag features important** (17.1%): Capture historical context
3. **Station features significant** (8.1%): Capture geographic effects
4. **Time features essential** (3.7%): Capture temporal patterns
5. **Derived features valuable** (3.7%): Capture physical relationships

### Implementation Notes

- **Memory efficient**: Features created in-place where possible
- **Station-grouped**: Lag and rolling features calculated within each station
- **Missing value handling**: Uses `min_periods=1` for rolling features
- **Cyclical encoding**: Prevents boundary discontinuities
- **Interaction features**: Capture non-linear relationships

---

## References

- **Feature Engineering Module**: `src/data/feature_engineering.py`
- **Main Documentation**: `docs/FEATURE_ENGINEERING.md`
- **Training Script**: `scripts/train/train_frost_forecast.py`
- **Station Metadata**: `data/external/cimis_station_metadata.csv`

---

**Last Updated**: 2025-11-12
**Total Features**: 298
**Document Version**: 1.0

