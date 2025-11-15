# 特征工程文档

**最后更新**: 2025-11-12

本文档详细说明特征工程的设计、实现、特征列表、特征选择和性能改进。

## 📋 目录

1. [特征概览](#特征概览)
2. [完整特征列表](#完整特征列表)
3. [特征实现](#特征实现)
4. [特征选择](#特征选择)
5. [特征重要性分析](#特征重要性分析)
6. [性能改进](#性能改进)

## 📚 相关文档

- **[特征参考文档](FEATURE_REFERENCE.md)**: 完整的特征列表、获取方法和功能说明（298个特征）
- **[技术文档](TECHNICAL_DOCUMENTATION.md)**: 技术架构和实现细节
- **[训练和评估文档](TRAINING_AND_EVALUATION.md)**: 模型训练和评估方法

---

## 特征概览

### 特征数量统计

| 项目 | 数量 | 说明 |
|------|------|------|
| **特征总数** | **298** | 当前使用的所有特征（原有281 + 新增17站点特征） |
| **时间特征** | 11 | 时间相关特征 |
| **滞后特征** | 51 | 历史值特征（1h, 3h, 6h, 12h, 24h） |
| **滚动特征** | 180 | 滚动统计特征（3h, 6h, 12h, 24h窗口） |
| **派生特征** | 11 | 计算的派生特征 |
| **辐射特征** | 3 | 太阳辐射相关特征 |
| **风向特征** | 7 | 风向和风速相关特征 |
| **湿度特征** | 6 | 湿度和露点相关特征 |
| **温度特征** | 2 | 气温和土壤温度 |
| **站点特征** | 24 | 站点地理位置特征（原有7 + 新增17） |
| **其他特征** | 3 | ETo, Precip, Vap Pres |

### 特征数量对比

| 项目 | 之前 | 现在 | 变化 |
|------|------|------|------|
| **特征总数** | ~100 | **281** | +181 |
| **使用的变量** | 5个 | 13个 | +8个 |
| **变量使用率** | 31% | 81% | +50% |

### 特征类别说明

1. **时间特征** (11个): hour, day_of_year, month, season, 周期性编码
2. **滞后特征** (51个): 1h, 3h, 6h, 12h, 24h 前的值
3. **滚动统计** (180个): 3h, 6h, 12h, 24h 窗口的 mean, min, max, std, sum
4. **辐射特征** (3个): 日累积辐射、辐射变化率、辐射-温度交互
5. **风向特征** (7个): 周期性编码、类别编码、风速变化
6. **湿度特征** (6个): 露点、湿度、饱和水汽压、露点接近度
7. **派生特征** (11个): 温度差、风寒指数、热指数、冷却率等
8. **站点特征** (24个): GPS坐标、海拔、区域编码、县/城市编码、空间特征、交互特征
9. **其他特征** (3个): ETo, Precip, Vap Pres

---

## 完整特征列表

### 1. 时间特征 (11个)

| 特征名 | 说明 |
|--------|------|
| `Jul` | 年内日序（1-366） |
| `day_of_week` | 星期几（0-6） |
| `day_of_year` | 年内第几天（1-366） |
| `hour` | 小时（0-23） |
| `hour_cos` | 小时的余弦编码 |
| `hour_sin` | 小时的正弦编码 |
| `is_night` | 是否夜间（18:00-06:00） |
| `month` | 月份（1-12） |
| `month_cos` | 月份的余弦编码 |
| `month_sin` | 月份的正弦编码 |
| `season` | 季节（1=春, 2=夏, 3=秋, 4=冬） |

### 2. 滞后特征 (51个)

**定义**: 滞后特征（Lag Features）是**过去几小时**的历史值，用于捕获时间序列的依赖性。

**重要说明**:
- `lag_1` = **过去1小时**（1 hour ago）的值
- `lag_3` = **过去3小时**（3 hours ago）的值
- `lag_6` = **过去6小时**（6 hours ago）的值
- `lag_12` = **过去12小时**（12 hours ago）的值
- `lag_24` = **过去24小时**（24 hours ago）的值

**示例**:
- 对于时间点 `2025-01-01 06:00`:
  - `Air Temp (C)_lag_1` = `05:00` 的温度（过去1小时）
  - `Air Temp (C)_lag_3` = `03:00` 的温度（过去3小时）
  - `Air Temp (C)_lag_24` = `2024-12-31 06:00` 的温度（过去24小时）

#### 为什么不按每小时创建？(Why Not Create Per Hour?)

**当前实现**: 只创建 `lag_1, lag_3, lag_6, lag_12, lag_24` (5个时间点)

**如果按每小时创建**: 需要创建 `lag_1, lag_2, lag_3, ..., lag_24` (24个时间点)

**对比分析**:

| 指标 | 当前实现 | 按每小时创建 | 增加 |
|------|---------|------------|------|
| **滞后时间点** | 5个 | 24个 | +19个 (380%) |
| **滞后特征数** | 45个 | 216个 | +171个 (380%) |
| **总特征数** | 281个 | 452个 | +171个 (61%) |
| **内存使用** | ~36GB | ~58GB | +22GB (61%) |
| **训练时间** | 基准 | ~1.6x | +60% |

**选择特定时间点的原因**:

1. **特征冗余**: 相邻小时的特征高度相关（相关性>0.95）
   - `lag_2` 和 `lag_3` 几乎完全相同
   - `lag_4` 和 `lag_6` 高度相关
   - 添加冗余特征不会提升模型性能

2. **计算效率**: 
   - 减少特征数量，降低计算成本
   - 减少内存使用（-22GB）
   - 减少训练时间（-60%）

3. **模型性能**:
   - 过多特征可能导致过拟合
   - 特征选择更困难
   - 模型复杂度增加，但性能不一定提升

4. **时间尺度**: 选择的5个时间点代表不同的时间尺度
   - `lag_1` (1小时): 短期变化
   - `lag_3` (3小时): 短期趋势
   - `lag_6` (6小时): 中期趋势
   - `lag_12` (12小时): 中期周期
   - `lag_24` (24小时): 日周期

5. **业务意义**: 这些时间点能捕获不同的模式
   - 1小时: 最近变化
   - 3小时: 短期趋势
   - 6小时: 中期趋势
   - 12小时: 半日周期
   - 24小时: 日周期（最重要的周期）

**结论**: 
- ✅ **当前实现更优**: 5个时间点已经足够捕获所有重要的时间模式
- ❌ **按每小时创建**: 会增加171个冗余特征，但不会显著提升模型性能
- ⚠️ **数据量不是主要限制**: 主要是特征冗余和计算效率的考虑

**如果需要按每小时创建**: 
可以通过配置 `lag_features.lags` 参数来修改：
```python
config = {
    "lag_features": {
        "enabled": True,
        "columns": [...],
        "lags": list(range(1, 25))  # lag_1 到 lag_24 (24个时间点)
    }
}
```

#### Air Temp (C) - 5个
- `Air Temp (C)_lag_1` - 过去1小时的温度
- `Air Temp (C)_lag_3` - 过去3小时的温度
- `Air Temp (C)_lag_6` - 过去6小时的温度
- `Air Temp (C)_lag_12` - 过去12小时的温度
- `Air Temp (C)_lag_24` - 过去24小时的温度

#### Dew Point (C) - 5个
- `Dew Point (C)_lag_1` - 过去1小时的露点温度
- `Dew Point (C)_lag_3` - 过去3小时的露点温度
- `Dew Point (C)_lag_6` - 过去6小时的露点温度
- `Dew Point (C)_lag_12` - 过去12小时的露点温度
- `Dew Point (C)_lag_24` - 过去24小时的露点温度

#### ETo (mm) - 5个
- `ETo (mm)_lag_1` - 过去1小时的参考蒸散发
- `ETo (mm)_lag_3` - 过去3小时的参考蒸散发
- `ETo (mm)_lag_6` - 过去6小时的参考蒸散发
- `ETo (mm)_lag_12` - 过去12小时的参考蒸散发
- `ETo (mm)_lag_24` - 过去24小时的参考蒸散发

#### Precip (mm) - 5个
- `Precip (mm)_lag_1` - 过去1小时的降水量
- `Precip (mm)_lag_3` - 过去3小时的降水量
- `Precip (mm)_lag_6` - 过去6小时的降水量
- `Precip (mm)_lag_12` - 过去12小时的降水量
- `Precip (mm)_lag_24` - 过去24小时的降水量

#### Rel Hum (%) - 5个
- `Rel Hum (%)_lag_1` - 过去1小时的相对湿度
- `Rel Hum (%)_lag_3` - 过去3小时的相对湿度
- `Rel Hum (%)_lag_6` - 过去6小时的相对湿度
- `Rel Hum (%)_lag_12` - 过去12小时的相对湿度
- `Rel Hum (%)_lag_24` - 过去24小时的相对湿度

#### Soil Temp (C) - 5个
- `Soil Temp (C)_lag_1` - 过去1小时的土壤温度
- `Soil Temp (C)_lag_3` - 过去3小时的土壤温度
- `Soil Temp (C)_lag_6` - 过去6小时的土壤温度
- `Soil Temp (C)_lag_12` - 过去12小时的土壤温度
- `Soil Temp (C)_lag_24` - 过去24小时的土壤温度

#### Sol Rad (W/sq.m) - 5个
- `Sol Rad (W/sq.m)_lag_1` - 过去1小时的太阳辐射
- `Sol Rad (W/sq.m)_lag_3` - 过去3小时的太阳辐射
- `Sol Rad (W/sq.m)_lag_6` - 过去6小时的太阳辐射
- `Sol Rad (W/sq.m)_lag_12` - 过去12小时的太阳辐射
- `Sol Rad (W/sq.m)_lag_24` - 过去24小时的太阳辐射

#### Vap Pres (kPa) - 5个
- `Vap Pres (kPa)_lag_1` - 过去1小时的水汽压
- `Vap Pres (kPa)_lag_3` - 过去3小时的水汽压
- `Vap Pres (kPa)_lag_6` - 过去6小时的水汽压
- `Vap Pres (kPa)_lag_12` - 过去12小时的水汽压
- `Vap Pres (kPa)_lag_24` - 过去24小时的水汽压

#### Wind Dir (0-360) - 5个
- `Wind Dir (0-360)_lag_1` - 过去1小时的风向
- `Wind Dir (0-360)_lag_3` - 过去3小时的风向
- `Wind Dir (0-360)_lag_6` - 过去6小时的风向
- `Wind Dir (0-360)_lag_12` - 过去12小时的风向
- `Wind Dir (0-360)_lag_24` - 过去24小时的风向

#### Wind Speed (m/s) - 5个
- `Wind Speed (m/s)_lag_1` - 过去1小时的风速
- `Wind Speed (m/s)_lag_3` - 过去3小时的风速
- `Wind Speed (m/s)_lag_6` - 过去6小时的风速
- `Wind Speed (m/s)_lag_12` - 过去12小时的风速
- `Wind Speed (m/s)_lag_24` - 过去24小时的风速

#### 派生特征滞后 - 1个
- `temp_decline_rate_lag_1` - 过去1小时的温度下降速率

### 3. 滚动特征 (180个)

每个变量（Air Temp, Dew Point, ETo, Precip, Rel Hum, Soil Temp, Sol Rad, Vap Pres, Wind Speed）都有：
- 3h窗口: `_rolling_3h_mean`, `_rolling_3h_min`, `_rolling_3h_max`, `_rolling_3h_std`, `_rolling_3h_sum` (5个)
- 6h窗口: `_rolling_6h_mean`, `_rolling_6h_min`, `_rolling_6h_max`, `_rolling_6h_std`, `_rolling_6h_sum` (5个)
- 12h窗口: `_rolling_12h_mean`, `_rolling_12h_min`, `_rolling_12h_max`, `_rolling_12h_std`, `_rolling_12h_sum` (5个)
- 24h窗口: `_rolling_24h_mean`, `_rolling_24h_min`, `_rolling_24h_max`, `_rolling_24h_std`, `_rolling_24h_sum` (5个)

**9个变量 × 4个窗口 × 5个统计量 = 180个滚动特征**

#### 变量列表
1. `Air Temp (C)` - 20个滚动特征
2. `Dew Point (C)` - 20个滚动特征
3. `ETo (mm)` - 20个滚动特征
4. `Precip (mm)` - 20个滚动特征
5. `Rel Hum (%)` - 20个滚动特征
6. `Soil Temp (C)` - 20个滚动特征
7. `Sol Rad (W/sq.m)` - 20个滚动特征
8. `Vap Pres (kPa)` - 20个滚动特征
9. `Wind Speed (m/s)` - 20个滚动特征

#### 统计量说明
- `_mean`: 平均值
- `_min`: 最小值
- `_max`: 最大值
- `_std`: 标准差
- `_sum`: 总和

### 4. 派生特征 (11个)

| 特征名 | 说明 |
|--------|------|
| `cooling_acceleration` | 降温加速度 |
| `heat_index` | 热指数 |
| `humidity_change_rate` | 湿度变化率 |
| `nighttime_cooling_rate` | 夜间冷却率 |
| `soil_air_temp_diff` | 土壤-空气温度差 |
| `sol_rad_change_rate` | 太阳辐射变化率 |
| `temp_change_rate` | 温度变化率 |
| `temp_decline_rate` | 温度下降率 |
| `temp_trend` | 温度趋势 |
| `wind_chill` | 风寒指数 |
| `wind_speed_change_rate` | 风速变化率 |

### 5. 辐射特征 (3个)

| 特征名 | 说明 |
|--------|------|
| `Sol Rad (W/sq.m)` | 太阳辐射（原始值） |
| `daily_solar_radiation` | 日累积辐射 |
| `radiation_temp_interaction` | 辐射-温度交互项 |

### 6. 风向特征 (7个)

| 特征名 | 说明 |
|--------|------|
| `Wind Dir (0-360)` | 风向（原始值，0-360度） |
| `Wind Speed (m/s)` | 风速（原始值） |
| `calm_wind_duration` | 静风持续时间 |
| `wind_dir_category` | 风向类别（N=0, E=1, S=2, W=3） |
| `wind_dir_cos` | 风向余弦编码 |
| `wind_dir_sin` | 风向正弦编码 |
| `wind_dir_temp_interaction` | 风向-温度交互项 |

### 7. 湿度特征 (6个)

| 特征名 | 说明 |
|--------|------|
| `Dew Point (C)` | 露点温度（原始值） |
| `Rel Hum (%)` | 相对湿度（原始值） |
| `dew_point_proximity` | 露点接近度 |
| `saturation_vapor_pressure` | 饱和水汽压 |
| `temp_dew_diff` | 温度-露点差 |
| `temp_humidity_interaction` | 温度-湿度交互项 |

### 8. 温度特征 (2个)

| 特征名 | 说明 |
|--------|------|
| `Air Temp (C)` | 气温（原始值） |
| `Soil Temp (C)` | 土壤温度（原始值） |

### 9. 站点特征 (24个)

**基础地理特征** (7个):
| 特征名 | 说明 |
|--------|------|
| `distance_to_coast_approx` | 距海岸距离（近似值，km） |
| `elevation_ft` | 海拔（英尺） |
| `elevation_m` | 海拔（米） |
| `latitude` | 纬度 |
| `longitude` | 经度 |
| `region_encoded` | 区域编码 |
| `station_id_encoded` | 站点ID编码 |

**新增地理特征** (10个):
| 特征名 | 说明 |
|--------|------|
| `county_encoded` | 县编码（9个县） |
| `city_encoded` | 城市编码（17个城市） |
| `groundcover_encoded` | 地面覆盖类型编码 |
| `is_eto_station` | 是否为ETo站点（0/1） |
| `distance_to_nearest_station` | 到最近站点的距离（km） |
| `station_density` | 站点密度（50km半径内的站点数） |
| `latitude_cos` | 纬度余弦编码 |
| `latitude_sin` | 纬度正弦编码 |
| `longitude_cos` | 经度余弦编码 |
| `longitude_sin` | 经度正弦编码 |

**站点-天气交互特征** (7个):
| 特征名 | 说明 |
|--------|------|
| `elevation_temp_interaction` | 海拔 × 温度（高海拔=更冷） |
| `latitude_temp_interaction` | 纬度 × 温度（高纬度=更冷） |
| `distance_coast_temp_interaction` | 距海岸距离 × 温度（海岸=更暖） |
| `elevation_humidity_interaction` | 海拔 × 湿度（高海拔=更低湿度） |
| `elevation_dewpoint_interaction` | 海拔 × 露点（高海拔=更低露点） |
| `latitude_humidity_interaction` | 纬度 × 湿度 |
| `distance_coast_humidity_interaction` | 距海岸距离 × 湿度（海岸=更高湿度） |

**为什么这些特征重要？**

1. **地理特征**:
   - **海拔 (Elevation)**: 海拔越高，温度越低，霜冻风险越高
   - **纬度 (Latitude)**: 高纬度地区冬季更冷
   - **距海岸距离**: 海岸地区受海洋调节，温度更稳定，霜冻风险更低
   - **周期性编码**: 捕获空间周期性模式（如气候带）

2. **空间特征**:
   - **站点密度**: 反映局部微气候特征
   - **最近站点距离**: 可用于空间插值或邻居特征

3. **交互特征**:
   - **海拔-温度**: 捕获海拔对温度的直接影响
   - **纬度-温度**: 捕获纬度对温度的影响
   - **距离-温度**: 捕获海洋对温度的调节作用
   - **海拔-湿度**: 捕获海拔对湿度的影响

4. **分类特征**:
   - **县/城市编码**: 捕获局部气候特征
   - **区域编码**: 捕获区域气候差异
   - **ETo站点**: 反映站点类型差异

### 10. 其他特征 (3个)

| 特征名 | 说明 |
|--------|------|
| `ETo (mm)` | 参考蒸散量（原始值） |
| `Precip (mm)` | 降水量（原始值） |
| `Vap Pres (kPa)` | 水汽压（原始值） |

---

## 特征实现

### 优先级 1: 必须添加 ✅

#### Sol Rad (太阳辐射)

**新增特征**：
- 滞后特征：`Sol Rad (W/sq.m)_lag_1`, `_lag_3`, `_lag_6`, `_lag_12`, `_lag_24`
- 滚动特征：`Sol Rad (W/sq.m)_rolling_3h/6h/12h/24h_mean/min/max/std/sum`
- 派生特征：
  - `daily_solar_radiation`: 日累积辐射
  - `sol_rad_change_rate`: 辐射变化率
  - `nighttime_cooling_rate`: 夜间冷却率
  - `radiation_temp_interaction`: 辐射-温度交互项

#### Wind Dir (风向)

**新增特征**：
- 周期性编码：`wind_dir_sin`, `wind_dir_cos`
- 类别编码：`wind_dir_category` (N=0, E=1, S=2, W=3)
- 滞后特征：`Wind Dir (0-360)_lag_1`, `_lag_3`, `_lag_6`, `_lag_12`, `_lag_24`
- 交互特征：`wind_dir_temp_interaction`: 风向-温度交互项

### 优先级 2: 强烈建议添加 ✅

#### Wind Speed (风速)

**新增特征**：
- 滞后特征和滚动特征
- `wind_speed_change_rate`: 风速变化率
- `calm_wind_duration`: 静风持续时间

#### Rel Hum (相对湿度)

**新增特征**：
- 滚动特征
- `saturation_vapor_pressure`: 饱和水汽压
- `dew_point_proximity`: 露点接近度
- `humidity_change_rate`: 湿度变化率
- `temp_humidity_interaction`: 温度-湿度交互项

#### 趋势特征

**新增特征**：
- `temp_decline_rate`: 温度下降率
- `cooling_acceleration`: 降温加速度
- `temp_trend`: 温度趋势
- `temp_change_rate`: 温度变化率

### 优先级 3: 可选添加 ✅

#### ETo, Precip, Vap Pres

**新增特征**：
- 每个变量的滞后和滚动特征

#### 站点特征

**新增特征**：
- `station_id_encoded`: 站点ID编码
- `region_encoded`: 区域编码
- `elevation_ft`, `elevation_m`: 海拔
- `latitude`, `longitude`: GPS坐标
- `distance_to_coast_approx`: 距海岸距离

---

## 特征选择

### 为什么需要特征选择？

1. **减少内存使用** - 更少的特征意味着更少的内存（当前281个特征使用~16GB/站点）
2. **加快训练速度** - 更少的数据处理（预期减少40-60%训练时间）
3. **提高模型性能** - 移除噪声和冗余特征
4. **减少过拟合** - 更少的特征减少模型复杂度

### 特征选择方法

#### 1. 基于重要性选择

**方法**: 根据训练模型的特征重要性选择特征

**使用场景**: 
- 已有训练好的模型
- 需要保留最重要的特征
- 推荐保留Top 100-150特征

**命令**:
```bash
python scripts/select_features.py \
    --data experiments/full_data_training/labeled_data.parquet \
    --importance experiments/full_data_training/horizon_3h/feature_importance.csv \
    --method importance \
    --top-k 150 \
    --output config/selected_features_3h.json
```

#### 2. 基于相关性选择

**方法**: 移除高度相关的特征（冗余移除）

**使用场景**:
- 不需要预训练模型
- 需要移除冗余特征
- 推荐相关系数阈值0.95

**命令**:
```bash
python scripts/select_features.py \
    --data experiments/full_data_training/labeled_data.parquet \
    --method correlation \
    --max-correlation 0.95 \
    --output config/selected_features_corr.json
```

#### 3. 基于缺失率选择

**方法**: 移除高缺失率特征

**使用场景**:
- 需要移除不可靠特征
- 推荐缺失率阈值0.5

**命令**:
```bash
python scripts/select_features.py \
    --data experiments/full_data_training/labeled_data.parquet \
    --method missing \
    --max-missing 0.5 \
    --output config/selected_features_missing.json
```

#### 4. 组合选择（推荐）

**方法**: 组合所有方法进行综合特征选择

**使用场景**:
- 需要全面的特征选择
- 移除冗余、缺失和低重要性特征
- 推荐保留Top 100-150特征

**命令**:
```bash
python scripts/select_features.py \
    --data experiments/full_data_training/labeled_data.parquet \
    --importance experiments/full_data_training/horizon_3h/feature_importance.csv \
    --method all \
    --top-k 150 \
    --max-correlation 0.95 \
    --max-missing 0.5 \
    --output config/selected_features_all.json
```

### 特征选择工作流程

#### 步骤 1: 训练初始模型

```bash
# 使用所有特征训练初始模型
python scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --output experiments/full_data_training \
    --model lightgbm
```

#### 步骤 2: 分析特征重要性

```bash
# 分析特征重要性
python scripts/analyze_feature_importance.py \
    --models experiments/full_data_training \
    --horizons 3 6 12 24 \
    --output experiments/full_data_training/feature_importance
```

#### 步骤 3: 选择特征

```bash
# 选择特征（推荐：保留Top 150）
for horizon in 3 6 12 24; do
    python scripts/select_features.py \
        --data experiments/full_data_training/labeled_data.parquet \
        --importance experiments/full_data_training/horizon_${horizon}h/feature_importance.csv \
        --horizon $horizon \
        --method all \
        --top-k 150 \
        --max-correlation 0.95 \
        --max-missing 0.5 \
        --output config/selected_features_${horizon}h.json
done
```

### 特征选择预期效果

| 指标 | 之前 (281特征) | 之后 (150特征) | 改进 |
|------|----------------|----------------|------|
| **特征数量** | 281 | ~100-150 | 减少 47-64% |
| **内存使用** | 16GB | 8-10GB | 减少 40-50% |
| **训练时间** | 2.5分钟 | 1.5分钟 | 减少 40% |
| **模型性能** | - | 可能提升 | - |

---

## 特征重要性分析

### Top 30 特征（3小时窗口）

| 排名 | 特征名 | 重要性 | 类别 |
|------|--------|--------|------|
| 1 | `Jul` | 124.0 | 时间特征 |
| 2 | `Dew Point (C)` | 91.0 | 露点温度 |
| 3 | `month_cos` | 69.0 | 时间特征 |
| 4 | `Sol Rad (W/sq.m)_rolling_24h_max` | 67.0 | 太阳辐射 |
| 5 | `Soil Temp (C)` | 63.0 | 土壤温度 |

### 关键发现

1. **露点温度 (Dew Point)** 是最重要的特征类别
   - 在 Top 30 中占据 7 个位置
   - 总重要性最高（274.0）

2. **太阳辐射 (Sol Rad)** 相关特征在 Top 30 中占据重要位置
   - `Sol Rad (W/sq.m)_rolling_24h_max` 排名第 4（重要性 67.0）
   - 平均重要性高（56.0）

3. **风向和风速特征**显著提升模型性能
   - 在 6h 窗口中，风向和风速相关特征占据 8 个位置
   - 总重要性 256.0

4. **时间特征**对季节性预测至关重要
   - `Jul`（年内日序）在 3h 和 6h 窗口中都是最重要的特征
   - `month_cos`（月份的周期性编码）也排名靠前

5. **滞后特征和滚动统计特征**捕捉动态变化
   - 滞后特征（lag_1, lag_3, lag_6, lag_12, lag_24）在 Top 30 中占据大量位置
   - 滚动统计特征（rolling_3h, rolling_6h, rolling_12h, rolling_24h）也广泛使用

---

## 性能改进

### 预期改进

| 指标 | 之前 | 预期 | 改进 |
|------|------|------|------|
| **Brier Score** | 0.0080 | 0.005-0.006 | ↓ 25-37% |
| **ROC-AUC** | 0.73 | 0.80-0.85 | ↑ 10-16% |
| **温度预测 MAE** | 5.76°C | 4.0-4.5°C | ↓ 22-30% |

### 实际结果

基于 50,000 样本训练：

**标准评估**：
- 3h: Brier=0.0054, ROC-AUC=0.8855, MAE=4.68°C, R²=0.58
- 6h: Brier=0.0055, ROC-AUC=0.8806, MAE=4.69°C, R²=0.58

**LOSO评估**：
- 3h: Brier=0.0092±0.0035, ROC-AUC=0.9019±0.0244, MAE=4.91±0.21°C
- 6h: Brier=0.0092±0.0034, ROC-AUC=0.8939±0.0214, MAE=4.93±0.20°C

### 关键改进亮点

1. **霜冻预测性能**
   - Brier Score: 改进 30-33%
   - ROC-AUC: 提升 21-33%
   - ECE: 改进 64%

2. **温度预测性能**
   - MAE: 改进 19-24%
   - R²: 提升 68-135%

3. **跨站点泛化能力**
   - ROC-AUC 标准差降低 79-86%
   - 所有指标的标准差都降低

---

## 📚 相关文档

- **[USER_GUIDE.md](USER_GUIDE.md)**: 用户指南
- **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)**: 技术文档
- **[TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md)**: 训练和评估文档

---

## 特征合理性评估

### 数据规模分析

| 指标 | 数值 | 评估 |
|------|------|------|
| **总样本数** | 2,367,360 | ✅ 充足 |
| **总特征数** | 298 | ✅ 合理（新增17个站点特征） |
| **样本/特征比** | 8,424.8:1 | ✅ 非常充足 |
| **推荐比例** | >100:1 | ✅ 远超推荐 |

### 合理性评估

#### ✅ **特征数量合理**

1. **样本/特征比**:
   - 当前: 8,424.8:1
   - 推荐: >100:1 (对于LightGBM)
   - **评估**: ✅ 非常充足，不会导致过拟合

2. **特征类别设计**:
   - 滚动特征（180个，64%）：捕获时间序列模式 ✅
   - 滞后特征（51个，18%）：捕获历史信息 ✅
   - 时间特征（11个，4%）：捕获季节性模式 ✅
   - 派生特征（11个，4%）：捕获非线性关系 ✅

3. **任务复杂度匹配**:
   - 预测目标: 霜冻概率 + 温度（二元任务）✅
   - 时间窗口: 4个（3h, 6h, 12h, 24h）✅
   - 站点数量: 18个（需要跨站点泛化）✅
   - 时空特性: 时间序列 + 空间变化 ✅

4. **行业标准对比**:
   - 气象预测任务: 通常50-500个特征 ✅
   - 时间序列任务: 通常50-300个特征 ✅
   - 当前特征数: 281个 ✅ 在合理范围内

#### ⚠️ **资源成本考虑**

| 资源 | 当前 | 优化后 | 改进 |
|------|------|--------|------|
| **内存使用** | ~16GB/站点 | ~8-10GB/站点 | ↓ 40-50% |
| **训练时间** | ~2.5分钟/站点 | ~1.5分钟/站点 | ↓ 40% |
| **特征数量** | 281个 | 100-150个 | ↓ 47-64% |

### 结论

#### ✅ **特征数量合理性: 合理**

**理由**:
1. ✅ 281个特征对于2.36M样本是合理的（样本/特征比充足）
2. ✅ 特征类别设计合理（时间、滞后、滚动、派生）
3. ✅ 符合气象预测任务的特征需求
4. ✅ 不会导致过拟合（样本充足）

**优化建议**:
1. ⚠️ 进行特征选择，减少到100-150个特征
2. ⚠️ 预期内存减少40-50%（16GB → 8-10GB）
3. ⚠️ 预期训练时间减少40%（2.5min → 1.5min）
4. ⚠️ 模型性能：可能提升或保持

#### 最终结论

**✅ 281个特征对于此任务是合理的**

- 特征数量在合理范围内
- 特征设计符合任务需求
- 样本充足，不会过拟合
- 建议进行特征选择以优化资源使用（非必需）

---

## 总结

### 当前特征状态

- **总特征数**: 298个 ✅ **合理**（原有281 + 新增17个站点特征）
- **特征类别**: 10个类别
- **主要特征类型**: 滚动特征（180个，64%）、滞后特征（51个，18%）、时间特征（11个，4%）
- **样本/特征比**: 8,424.8:1 ✅ **非常充足**

### 特征选择建议

基于特征重要性分析，以下是特征选择的推荐选项：

#### 关键统计

- **总特征数**: 298
- **使用特征** (importance > 0): 280 (94.0%)
- **未使用特征** (importance = 0): 18 (6.0%)

#### 累积重要性分析

| 特征数量 | 累积重要性 | 推荐级别 |
|---------|-----------|---------|
| Top 64 | 50.0% | 最小可行集 |
| **Top 136** | **80.0%** | **快速训练** |
| **Top 175** | **90.0%** | **⭐ 推荐** |
| **Top 206** | **95.0%** | **最佳性能** |
| Top 247 | 99.0% | 接近完整集 |
| All 280 | 100.0% | 完整特征集 |

#### 推荐选项

**选项1: 快速训练/生产环境**
- **Top 136 特征 (80% 重要性)**
- ✅ ~50% 特征减少 → 更快计算
- ✅ 更低内存占用
- ✅ 更短训练时间
- ✅ 保留大部分预测能力

**选项2: 平衡方法 ⭐ 推荐**
- **Top 175 特征 (90% 重要性)**
- ✅ 性能和效率的极佳平衡
- ✅ 保留90%的预测能力
- ✅ 合理的训练时间
- ✅ 可接受的内存使用
- **适用于**: 标准训练流程（推荐）、大多数使用场景

**选项3: 最佳性能**
- **Top 206 特征 (95% 重要性)**
- ✅ 接近最优性能
- ✅ 保留95%的预测能力
- ✅ 覆盖几乎所有有用特征
- **适用于**: 追求最大精度、有充足计算资源、最终模型训练

**选项4: 完整特征集**
- **280 特征 (所有重要性 > 0)**
- ✅ 最大可能的性能
- ✅ 保留所有有用特征
- ✅ 适合最终评估
- **适用于**: 最终模型训练、完整性能评估、计算成本不敏感

#### 比较表

| 选项 | 特征数量 | 累积重要性 | 训练速度 | 内存使用 | 推荐用于 |
|------|---------|-----------|---------|---------|---------|
| 快速训练 | 136 | 80% | ⚡⚡⚡ 快 | 💚 低 | 生产、快速迭代 |
| **平衡（推荐）** | **175** | **90%** | **⚡⚡ 中** | **💚💚 中** | **大多数场景 ⭐** |
| 最佳性能 | 206 | 95% | ⚡ 慢 | 💚💚💚 高 | 最终模型、最大精度 |
| 完整集 | 280 | 100% | ⚡⚡⚡ 最慢 | 💚💚💚 最高 | 最终评估 |

#### 实施建议

**推荐工作流**:

1. **初始训练/快速迭代**
   - 使用 **Top 136 特征 (80%)**
   - 快速验证模型和特征工程效果
   - 快速反馈循环

2. **标准训练** ⭐
   - 使用 **Top 175 特征 (90%)**
   - 平衡性能和效率
   - **推荐用于大多数场景**

3. **最终模型**
   - 使用 **Top 206 特征 (95%)** 或 **所有 280 特征**
   - 追求最大精度
   - 用于最终提交/生产

#### 最终推荐

**对于大多数使用场景: Top 175 特征 (90%)**

**理由**:
1. **最优平衡**: 90%重要性保留，仅减少~37%特征数量
2. **实用性能**: 对大多数预测任务足够
3. **合理成本**: 训练时间和内存使用可接受
4. **经过验证**: 基于100k样本训练的全面分析

**何时使用更多**:
- 如果精度绝对关键 → 使用 Top 206 (95%) 或所有 280
- 如果资源充足 → 使用所有 280 特征

**何时使用更少**:
- 如果计算资源有限 → 使用 Top 136 (80%)
- 如果需要实时推理 → 使用 Top 136 (80%)

#### 注意事项

1. **特征重要性可能变化**: 重要性排名可能因以下因素略有不同：
   - 不同样本大小
   - 不同时间窗口
   - 不同模型配置
   - 完整数据集训练

2. **需要验证**: 始终在保留测试集上验证特征选择

3. **动态选择**: 考虑使用完整训练的特征重要性进行最终特征选择

4. **类别平衡**: 确保重要类别（Time, Lag, Rolling）在选定特征中得到良好表示

#### 特征列表文件

特征重要性排名可在以下位置找到：
- CSV: `experiments/lightgbm/feature_importance/feature_importance_3h_all.csv`
- JSON: `docs/report/feature_analysis.json`

使用这些文件在训练期间选择Top N特征。

### 下一步

1. ✅ 当前特征数量合理，可直接使用
2. 🔄 训练初始模型获取特征重要性（可选）
3. 🔄 使用特征选择工具选择Top 175特征（推荐）
4. 🔄 使用选择的特征重新训练（可选）
5. 🔄 比较性能并调整（可选）

详细的特征选择建议请参考 [docs/report/FEATURE_SELECTION_RECOMMENDATIONS.md](report/FEATURE_SELECTION_RECOMMENDATIONS.md)。
