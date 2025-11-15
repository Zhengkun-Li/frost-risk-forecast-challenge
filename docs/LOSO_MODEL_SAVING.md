# LOSO模型保存功能说明

## 概述

LOSO（Leave-One-Station-Out）评估默认不保存模型权重，只保存评估指标。如果需要保存模型用于后续分析，可以使用以下可选参数。

## 命令行参数

### 1. `--save-loso-models`

保存所有LOSO模型（所有站点 × 所有时间窗口）。

**存储需求：**
- 18个站点 × 4个时间窗口 × 2个模型 = 144个模型文件
- 约 180 MB

**使用示例：**
```bash
python3 scripts/train/train_frost_forecast.py \
    --loso \
    --save-loso-models \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/lightgbm/top175_features
```

**保存位置：**
```
experiments/lightgbm/top175_features/loso/
  station_2/
    horizon_3h/
      frost_classifier/
      temp_regressor/
    horizon_6h/
      ...
  station_7/
    ...
```

### 2. `--save-loso-worst-n N`

只保存表现最差的N个站点的模型。

**工作原理：**
1. 训练时先保存所有模型（临时）
2. 评估完成后，计算每个站点的综合表现（基于Brier Score和MAE）
3. 识别最差的N个站点
4. 删除其他站点的模型，只保留最差N个站点的模型

**使用示例：**
```bash
python3 scripts/train/train_frost_forecast.py \
    --loso \
    --save-loso-worst-n 3 \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/lightgbm/top175_features
```

**输出：**
- 模型保存在 `loso/station_*/` 目录（只保留最差的3个站点）
- 最差站点信息保存在 `loso/worst_stations.json`

**最差站点评分方法：**
- 综合评分 = Brier Score × 10 + MAE
- 分数越高，表现越差
- 对所有时间窗口取平均

### 3. `--save-loso-horizons H1 H2 ...`

只保存指定时间窗口的模型。

**使用示例：**
```bash
# 只保存24h时间窗口的模型
python3 scripts/train/train_frost_forecast.py \
    --loso \
    --save-loso-horizons 24 \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/lightgbm/top175_features

# 只保存12h和24h时间窗口的模型
python3 scripts/train/train_frost_forecast.py \
    --loso \
    --save-loso-horizons 12 24 \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/lightgbm/top175_features
```

**存储需求：**
- 18个站点 × N个时间窗口 × 2个模型
- 例如：只保存24h → 18 × 1 × 2 = 36个模型文件，约 45 MB

## 参数组合

可以组合使用多个参数：

```bash
# 只保存最差的3个站点在24h时间窗口的模型
python3 scripts/train/train_frost_forecast.py \
    --loso \
    --save-loso-worst-n 3 \
    --save-loso-horizons 24 \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/lightgbm/top175_features
```

**注意：** 如果同时指定 `--save-loso-worst-n` 和 `--save-loso-horizons`，会先按时间窗口过滤，再按最差站点过滤。

## 使用建议

### 默认情况（不保存）
- **推荐用于：** 常规评估，只需要评估指标
- **优点：** 节省存储，避免文件管理复杂

### 保存特定时间窗口
- **推荐用于：** 分析最难的预测任务（如24h）
- **优点：** 平衡存储和分析需求
- **示例：** `--save-loso-horizons 24`

### 保存最差站点
- **推荐用于：** 分析表现最差的站点，找出问题原因
- **优点：** 只保存需要的模型，节省存储
- **示例：** `--save-loso-worst-n 3`

### 保存所有模型
- **推荐用于：** 完整的研究分析，需要所有模型
- **优点：** 完整的模型集合，可以进行各种分析
- **缺点：** 存储需求较大（180 MB）
- **示例：** `--save-loso-models`

## 模型文件结构

```
experiments/lightgbm/top175_features/loso/
├── checkpoint.json              # 训练检查点
├── station_metrics.csv          # 所有站点的指标
├── station_results.json         # 详细结果
├── summary.json                 # 汇总统计
├── worst_stations.json          # 最差站点信息（如果使用 --save-loso-worst-n）
└── station_*/                   # 站点模型目录（如果保存了模型）
    └── horizon_*h/
        ├── frost_classifier/
        │   ├── model.pkl
        │   └── config.json
        └── temp_regressor/
            ├── model.pkl
            └── config.json
```

## 注意事项

1. **存储空间：** 保存所有模型需要约180 MB，请确保有足够的存储空间
2. **训练时间：** 保存模型不会显著增加训练时间（主要是磁盘I/O）
3. **内存使用：** 保存模型不会增加内存使用（模型在保存后立即释放）
4. **最差站点识别：** `--save-loso-worst-n` 会在评估完成后自动删除不需要的模型
5. **恢复训练：** 使用 `--resume-loso` 时，已保存的模型不会被覆盖

## 示例场景

### 场景1：分析24h预测的困难
```bash
python3 scripts/train/train_frost_forecast.py \
    --loso \
    --save-loso-horizons 24 \
    --horizons 3 6 12 24
```
只保存24h时间窗口的模型，用于分析长期预测的挑战。

### 场景2：找出表现最差的站点
```bash
python3 scripts/train/train_frost_forecast.py \
    --loso \
    --save-loso-worst-n 3
```
保存最差的3个站点的所有模型，用于分析这些站点的问题。

### 场景3：完整的研究分析
```bash
python3 scripts/train/train_frost_forecast.py \
    --loso \
    --save-loso-models
```
保存所有模型，用于完整的模型分析和研究。

