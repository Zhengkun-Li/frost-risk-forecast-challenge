# 用户指南

**最后更新**: 2025-11-12

本指南涵盖从环境设置、快速开始到高级使用的所有内容。

## 📋 目录

1. [环境设置](#环境设置)
2. [快速开始](#快速开始)
3. [数据准备与加载](#数据准备与加载)
4. [完整流程指南](#完整流程指南)
5. [模型训练](#模型训练)
6. [模型评估](#模型评估)
7. [结果解读](#结果解读)
8. [常见问题](#常见问题)

---

## 环境设置

### 快速设置（推荐）

运行设置脚本：

```bash
./scripts/setup_venv.sh
```

这将：
1. 创建或修复虚拟环境 `.venv/`
2. 安装所有依赖包
3. 验证安装

### 手动设置

```bash
# 1. 创建虚拟环境
cd /home/zhengkun-li/frost-risk-forecast-challenge
python3 -m venv .venv

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 4. 验证安装
python3 -c "import pandas, numpy, lightgbm, xgboost; print('✅ All packages installed')"
```

### 使用虚拟环境运行脚本

**方法 1: 使用包装脚本（推荐）**

```bash
./scripts/run_with_venv.sh scripts/train/train_frost_forecast.py --sample-size 50000 --loso
```

**方法 2: 手动激活**

```bash
source .venv/bin/activate
python3 scripts/train/train_frost_forecast.py --sample-size 50000 --loso
deactivate  # 完成后退出虚拟环境
```

---

## 快速开始

### 最简单的使用方式

```bash
# 运行完整流程（使用默认设置）
./scripts/run_with_venv.sh scripts/train/train_frost_forecast.py --sample-size 50000 --horizons 3 6 12 24 --loso
```

这将自动完成：
1. 从 `stations/` 目录加载数据
2. 清洗数据并构建特征
3. 训练 LightGBM 模型（所有时间窗口）
4. 在 train/val/test 上评估
5. 执行 LOSO 评估
6. 生成可视化图表
7. 保存所有结果到 `experiments/` 目录

### 使用不同模型

```bash
# XGBoost
./scripts/run_with_venv.sh scripts/train/train_frost_forecast.py --model xgboost --sample-size 50000 --loso

# 指定输出目录（便于对比）
./scripts/run_with_venv.sh scripts/train/train_frost_forecast.py --model lightgbm --output experiments/runs/lightgbm_v1 --loso
```

---

## 数据准备与加载

### 数据位置

数据位于：`data/raw/frost-risk-forecast-challenge/stations/`

包含 18 个站点的 CSV 文件（每个约 14-15MB），系统会自动加载并合并。

### 自动加载（推荐）

系统按以下顺序自动检测数据：
1. `stations/` 目录（首选，18 个站点文件）
2. `cimis_all_stations.csv.gz`（备选）
3. `cimis_all_stations.csv`（最后）

**当前使用**: `stations/` 目录自动加载方法

### 数据加载过程

```
Loading 18 station files from stations/...
  Loaded 18/18 files...
Combining 18 station DataFrames...
Combined data: 2367360 rows, 26 columns
Stations: 18
```

### 性能说明

- **加载时间**: 18 个文件约需 10-30 秒
- **内存使用**: 合并后约 236 万行，内存占用约 500MB-1GB
- **文件大小**: 每个站点文件约 14-15MB，总计约 254MB

---

## 完整流程指南

### 数据流程图

```
原始数据 (CSV)
    ↓
[数据加载] → DataFrame (236万行, 26列)
    ↓
[QC过滤] → 低质量数据标记为 NaN
    ↓
[哨兵值处理] → -6999, -9999 → NaN
    ↓
[缺失值插补] → 前向填充
    ↓
[特征工程] → DataFrame (236万行, 300+列)
    ├─ 时间特征 (hour, month, season, ...)
    ├─ 滞后特征 (lag_1, lag_3, lag_6, ...)
    ├─ 滚动特征 (rolling_6h_mean, ...)
    ├─ 辐射特征 (Sol Rad相关)
    ├─ 风向特征 (Wind Dir周期性编码)
    └─ 派生特征 (temp_dew_diff, ...)
    ↓
[标签生成] → DataFrame (236万行, 300+特征列 + 8标签列)
    ├─ frost_3h, frost_6h, frost_12h, frost_24h
    └─ temp_3h, temp_6h, temp_12h, temp_24h
    ↓
[数据划分] → Train (70%) / Val (15%) / Test (15%)
    ↓
[模型训练] → 对每个时间窗口训练2个模型
    ├─ 分类模型 (frost probability)
    └─ 回归模型 (temperature)
    ↓
[模型评估] → 计算所有指标
    ↓
[模型保存] → .pkl 文件
```

### 关键步骤说明

#### 1. 数据清洗

系统自动执行以下清洗步骤：
- **QC 过滤**: 根据 QC 标记过滤低质量数据（保留空白和 `Y`，标记 `M/R/S/Q/P` 为 NaN）
- **哨兵值处理**: 将 `-6999`, `-9999` 等哨兵值替换为 `NaN`
- **缺失值插补**: 使用前向填充（按站点分组）

#### 2. 特征工程

系统自动创建以下特征：
- **时间特征**: hour, day_of_year, month, season, 周期性编码
- **滞后特征**: 1h, 3h, 6h, 12h, 24h 前的值
- **滚动统计**: 6h, 12h, 24h 窗口的 mean, min, max, std
- **辐射特征**: 日累积辐射、辐射变化率、夜间冷却率
- **风向特征**: 周期性编码、类别编码
- **派生特征**: 温度差、风寒指数、热指数等

详细说明请参考 [特征工程文档](FEATURE_ENGINEERING.md)。

#### 3. 标签生成

对每个预测时间窗口（3h, 6h, 12h, 24h），创建：
- **霜冻标签** (`frost_{h}h`): 未来温度是否 < 0°C（二分类）
- **温度标签** (`temp_{h}h`): 未来温度值（回归）

---

## 模型训练

### 使用训练脚本（推荐）

```bash
# 快速测试（推荐先运行）
./scripts/run_with_venv.sh scripts/train/train_frost_forecast.py \
    --sample-size 50000 \
    --horizons 3 6 12 24 \
    --output experiments/frost_forecast

# 完整数据训练
./scripts/run_with_venv.sh scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --loso \
    --output experiments/full_data_training \
    --model lightgbm
```

### 训练参数说明

- `--horizons`: 预测时间窗口（3, 6, 12, 24 小时）
- `--sample-size`: 采样数据量（用于快速测试，不指定则使用全部数据）
- `--loso`: 启用 LOSO（留站验证）评估
- `--output`: 输出目录
- `--model`: 模型类型（lightgbm, xgboost）

### 训练输出

每个时间窗口会训练两个模型：
1. **分类模型** (`frost_classifier`): 预测霜冻概率
2. **回归模型** (`temp_regressor`): 预测未来温度

---

## 模型评估

### 标准评估（自动包含）

每个模型自动在 train/val/test 三个数据集上评估：
- **指标**: MAE, RMSE, R², MAPE（回归）；Brier Score, ROC-AUC, PR-AUC（分类）
- **结果保存**: `{split}_metrics.json`
- **预测保存**: `predictions.json`

### LOSO 评估（可选，推荐）

使用 `--loso` 参数启用留站验证：

```bash
./scripts/run_with_venv.sh scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --loso \
    --output experiments/full_data_training
```

**LOSO 评估内容**:
- ✅ 每个站点的性能指标（18 个站点）
- ✅ 跨站点汇总统计（均值、标准差）
- ✅ 最佳/最差站点识别
- ✅ 结果保存到 `loso/` 目录

**LOSO 结果文件**:
- `loso/summary.json`: 汇总统计（mean_mae, std_mae, mean_rmse, mean_r2）
- `loso/station_metrics.csv`: 每个站点的详细指标

详细说明请参考 [训练和评估文档](TRAINING_AND_EVALUATION.md)。

---

## 结果解读

### 结果组织结构

```
experiments/full_data_training/
├── horizon_3h/
│   ├── frost_classifier/
│   │   ├── model.pkl
│   │   ├── metrics.json
│   │   └── feature_importance.csv
│   ├── temp_regressor/
│   │   ├── model.pkl
│   │   └── metrics.json
│   └── predictions.json
├── horizon_6h/
│   └── ...
├── loso/
│   ├── summary.json         # LOSO 摘要 ⭐
│   └── station_metrics.csv  # 每个站点指标
└── summary.json
```

### 关键指标位置

**单个模型的关键指标**:
- **测试集 MAE**: `test_metrics.json` → `mae`（越小越好）
- **测试集 R²**: `test_metrics.json` → `r2`（越接近 1 越好）
- **LOSO 平均 MAE**: `loso/summary.json` → `mean_mae`（泛化能力）
- **LOSO 标准差**: `loso/summary.json` → `std_mae`（稳定性）

### 结果质量判断

- **MAE < 1°C**: 预测精度高
- **R² > 0.9**: 模型拟合良好
- **Train vs Test 差异小**: 无过拟合
- **LOSO std_mae 小**: 跨站点稳定
- **LOSO mean_mae 接近 test_mae**: 泛化能力好

---

## 常见问题

### Q: 如何查看模型结果？

```bash
# 查看测试指标
cat experiments/full_data_training/horizon_3h/frost_classifier/metrics.json

# 查看 LOSO 摘要
cat experiments/full_data_training/loso/summary.json
```

### Q: LOSO 评估很慢怎么办？

使用采样数据：
```bash
./scripts/run_with_venv.sh scripts/train/train_frost_forecast.py \
    --sample-size 50000 \
    --horizons 3 6 12 24 \
    --loso
```

### Q: 如何自定义模型参数？

编辑训练脚本中的模型配置，或使用配置文件（如果支持）。

### Q: 结果会自动保存吗？

**是的！** 所有结果都会自动保存：
- 模型文件（.pkl）
- 评估指标（JSON）
- 预测结果（JSON）
- 特征重要性（CSV）
- LOSO 结果（如果启用）

### Q: 训练中断后可以继续吗？

如果使用 `--resume-loso` 参数，LOSO 评估可以从上次中断的地方继续。

---

## 📚 相关文档

- **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)**: 技术文档和 API 参考
- **[DATA_DOCUMENTATION.md](DATA_DOCUMENTATION.md)**: 数据说明和 QC 处理
- **[FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md)**: 特征工程详细说明
- **[TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md)**: 训练和评估详细指南
