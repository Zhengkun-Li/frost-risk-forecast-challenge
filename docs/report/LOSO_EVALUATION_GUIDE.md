# LOSO评估（空间泛化验证）指南

**Leave-One-Station-Out Cross-Validation**  
**留一站点交叉验证（空间泛化验证）**

---

## 什么是LOSO评估？

LOSO (Leave-One-Station-Out) 是一种交叉验证方法，专门用于评估模型在**空间上的泛化能力**。

### 基本原理

1. **依次将每个站点作为测试集**
2. **使用其余所有站点作为训练集**
3. **评估模型在新站点（从未见过的站点）上的表现**
4. **重复N次**（N = 站点数量，本项目中为18次）
5. **综合所有结果**评估空间泛化能力

### 为什么需要LOSO评估？

在气象预测中，不同地理位置的站点可能有：
- 不同的海拔高度
- 不同的地形特征
- 不同的微气候环境
- 不同的数据分布

标准的时间分割评估（70%训练，15%验证，15%测试）虽然能评估**时间泛化能力**，但**无法评估空间泛化能力**，因为所有站点的数据都混合在一起。

**LOSO评估解决了这个问题**，确保模型能够：
- ✅ 在新地点应用
- ✅ 适应不同的微气候环境
- ✅ 避免过拟合到特定站点的特征
- ✅ 真实反映生产环境性能

---

## LOSO vs 标准评估对比

### 标准评估（时间分割）

```
所有站点数据混合:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│  训练集 (70%) │  验证集 (15%) │  测试集 (15%) │
└───────────────┴───────────────┴───────────────┘
   时间上较早        时间上中间        时间上较晚

评估内容: 时间泛化能力（模型能否预测未来）
数据泄露风险: 低（时间独立）
空间泛化: ❌ 无法评估（所有站点混合）
```

### LOSO评估（站点分割）

```
18个站点，18次评估:

评估1:
┌─────────────────────┬──────────────┐
│  训练集 (17个站点)  │ 测试集 (站点1)│
└─────────────────────┴──────────────┘

评估2:
┌─────────────────────┬──────────────┐
│  训练集 (17个站点)  │ 测试集 (站点2)│
└─────────────────────┴──────────────┘

... (重复18次)

评估18:
┌─────────────────────┬──────────────┐
│  训练集 (17个站点)  │ 测试集 (站点18)│
└─────────────────────┴──────────────┘

评估内容: 空间泛化能力（模型能否在新地点应用）
数据泄露风险: 无（站点完全独立）
空间泛化: ✅ 充分评估（每个站点独立测试）
```

---

## LOSO评估流程

### Step 1: 数据准备

```python
# 加载所有站点数据
df = load_all_stations_data()  # 2.36M 样本，18个站点

# 创建LOSO分割
loso_splits = CrossValidator.leave_one_station_out(df)
# 返回: [(train_df_1, test_df_1), (train_df_2, test_df_2), ...]
# 共18个分割
```

### Step 2: 对每个站点进行评估

对于每个站点 i (i = 1, 2, ..., 18):

```
1. 准备数据
   - 训练集: 除了站点i之外的所有站点数据 (17个站点)
   - 测试集: 站点i的所有数据 (1个站点)

2. 特征工程
   - 只在训练集上拟合预处理器（scaler, imputer等）
   - 避免数据泄露（测试站点的统计信息不能泄露）

3. 模型训练
   - 使用训练集训练模型
   - 对于每个时间窗口 (3h, 6h, 12h, 24h) 分别训练

4. 模型评估
   - 在测试集（站点i）上评估
   - 计算指标: Brier Score, ROC-AUC, MAE, RMSE, R²等

5. 保存结果
   - 保存该站点的所有指标
   - 更新检查点
```

### Step 3: 综合结果

计算所有站点的统计量：
- **均值** (Mean): 平均性能
- **标准差** (Std): 性能稳定性
- **最小值/最大值**: 最差/最好站点的性能
- **站点特异性分析**: 哪些站点容易/难以预测

---

## 实现细节

### 代码位置

- **LOSO分割**: `src/evaluation/validators.py` → `leave_one_station_out()`
- **LOSO评估**: `scripts/train/train_frost_forecast.py` → `perform_loso_evaluation()`
- **检查点支持**: `scripts/train/train_frost_forecast.py` → `--resume-loso`

### 关键特性

#### 1. 无数据泄露保证

```python
# 预处理器只拟合训练站点的数据
X_train, X_test = preprocess_with_loso(
    train_df,  # 只有17个站点
    test_df,   # 1个测试站点
    feature_cols=feature_cols,
    scaling_method=None  # 树模型不需要标准化
)
```

**重要**: 测试站点的统计信息（均值、标准差等）**永远不会**用于预处理器的拟合。

#### 2. 内存优化

- **按站点处理**: 一次只处理一个站点
- **及时保存**: 每个站点完成后立即保存结果
- **释放内存**: 处理完一个站点后释放内存

```python
# 处理顺序
for station in stations:
    # 1. 加载该站点的数据（train + test）
    # 2. 处理所有时间窗口
    # 3. 保存结果
    # 4. 释放内存
    gc.collect()
```

#### 3. 检查点支持

支持中断后恢复：

```bash
# 初始运行
python scripts/train/train_frost_forecast.py --loso ...

# 如果中断，恢复运行
python scripts/train/train_frost_forecast.py --loso --resume-loso ...
```

检查点文件：
- `checkpoint.json`: 跟踪已完成站点
- `station_results.json`: 每个站点的结果

---

## 使用方法

### 基本用法

```bash
# 运行LOSO评估（所有时间窗口）
python scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --loso \
    --output experiments/loso_evaluation \
    --model lightgbm
```

### 恢复中断的运行

```bash
# 如果之前中断，恢复运行（跳过已完成的站点）
python scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --loso \
    --resume-loso \
    --output experiments/loso_evaluation \
    --model lightgbm
```

### 只评估特定时间窗口

```bash
# 只评估3小时和6小时窗口
python scripts/train/train_frost_forecast.py \
    --horizons 3 6 \
    --loso \
    --output experiments/loso_evaluation \
    --model lightgbm
```

---

## 输出结果

### 文件结构

```
experiments/loso_evaluation/
└── loso/
    ├── checkpoint.json          # 检查点（跟踪进度）
    ├── station_results.json     # 每个站点的详细结果
    ├── summary.json             # 汇总统计
    └── station_metrics.csv      # 扁平化的指标表（便于分析）
```

### 结果格式

#### `summary.json`

```json
{
  "3h": {
    "n_stations": 18,
    "frost_metrics": {
      "brier_score": {"mean": 0.0051, "std": 0.0012, "min": 0.0032, "max": 0.0089},
      "roc_auc": {"mean": 0.9132, "std": 0.0234, "min": 0.8654, "max": 0.9654},
      "ece": {"mean": 0.0033, "std": 0.0011, "min": 0.0012, "max": 0.0067}
    },
    "temp_metrics": {
      "mae": {"mean": 4.60, "std": 0.85, "min": 3.12, "max": 6.45},
      "rmse": {"mean": 5.60, "std": 1.02, "min": 3.89, "max": 7.89},
      "r2": {"mean": 0.5993, "std": 0.0845, "min": 0.4234, "max": 0.7891}
    }
  },
  ...
}
```

#### `station_metrics.csv`

| station_id | horizon | frost_brier_score | frost_roc_auc | temp_mae | temp_rmse | temp_r2 | ... |
|-----------|---------|-------------------|---------------|----------|-----------|---------|-----|
| 2 | 3h | 0.0032 | 0.9654 | 3.12 | 3.89 | 0.7891 | ... |
| 7 | 3h | 0.0045 | 0.9423 | 4.23 | 5.12 | 0.7123 | ... |
| ... | ... | ... | ... | ... | ... | ... | ... |

---

## 解读LOSO结果

### 1. 总体性能

查看 `summary.json` 中的均值：
- **Brier Score**: 越低越好（完美预测 = 0）
- **ROC-AUC**: 越高越好（完美预测 = 1.0）
- **MAE/RMSE**: 越低越好
- **R²**: 越高越好（完美预测 = 1.0）

### 2. 性能稳定性

查看标准差（std）：
- **小标准差**: 模型在所有站点上表现一致 ✅
- **大标准差**: 某些站点表现差，需要改进 ⚠️

### 3. 站点特异性

查看每个站点的结果：
- **表现好的站点**: 与其他站点相似的微气候
- **表现差的站点**: 可能需要更多数据或特殊处理

### 4. 时间窗口对比

对比不同时间窗口（3h, 6h, 12h, 24h）：
- **3h**: 通常表现最好（最近的数据）
- **24h**: 通常表现较差（预测更远的未来）

---

## LOSO评估 vs 标准评估对比表

| 方面 | 标准评估 | LOSO评估 |
|------|---------|----------|
| **数据分割方式** | 按时间分割（70/15/15） | 按站点分割（17训练/1测试） |
| **评估内容** | 时间泛化能力 | 空间泛化能力 |
| **数据泄露风险** | 低（时间独立） | 无（站点完全独立） |
| **训练数据量** | 70%总数据 | 94%总数据（17/18站点） |
| **测试数据量** | 15%总数据 | 6%总数据（1/18站点） |
| **评估次数** | 1次 | 18次（每个站点一次） |
| **训练时间** | 1倍 | ~4.5倍（18次 × 0.25复杂度） |
| **内存使用** | 高（所有数据） | 中等（一次一个站点） |
| **适用场景** | 时间序列预测评估 | 空间泛化能力评估 |
| **重要性** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 为什么LOSO评估很重要？

### 1. 生产环境真实性

在生产环境中，模型需要应用到**新地点**，而不仅仅是预测**新时间**。LOSO评估模拟了这种场景。

### 2. 避免过拟合

如果模型只在一个站点上表现好，但在其他站点上表现差，说明模型**过拟合**到特定站点的特征。LOSO评估能发现这个问题。

### 3. 特征工程验证

通过LOSO评估，可以验证：
- 站点特征（纬度、经度、海拔等）是否有效
- 特征工程是否具有通用性
- 模型是否真的学到了通用模式

### 4. 模型选择

不同的模型在LOSO评估上的表现可能不同。选择LOSO表现最好的模型，更有可能在新地点上表现好。

---

## LOSO评估的局限性

### 1. 计算成本高

- 需要训练18个模型（每个站点一次）
- 虽然每个模型复杂度较低（n_estimators=50 vs 200），但总时间仍然较长

### 2. 可能过于保守

- 如果所有站点都相似，LOSO评估可能过于保守
- 但在微气候变化大的情况下（如本项目），LOSO评估是必要的

### 3. 无法评估时间泛化

- LOSO评估只评估空间泛化
- 仍然需要标准评估来评估时间泛化能力

---

## 最佳实践

### 1. 先做标准评估

先做标准评估（快速），验证基本功能，然后再做LOSO评估（全面）。

### 2. 结合两种评估

```
标准评估 → 时间泛化 ✅
    +
LOSO评估 → 空间泛化 ✅
    =
完整评估 ✅
```

### 3. 分析站点差异

如果LOSO评估显示某些站点表现差，分析这些站点的特征，看看是否需要特殊处理。

### 4. 使用检查点

由于LOSO评估时间长，**一定要使用检查点**，避免中断后重新开始。

---

## 示例输出

```
============================================================
LOSO Evaluation: Processing 18 stations
   Memory optimization: One station at a time, save after each
   Horizons: [3, 6, 12, 24]
============================================================

[1/18] Processing Station 2
   Train stations: 17, Test station: 2
============================================================

  Processing 3h horizon...
    Train samples: 125000, Test samples: 7000
    ✅ Brier=0.0032, ECE=0.0012, ROC-AUC=0.9956, MAE=1.15°C

  Processing 6h horizon...
    Train samples: 125000, Test samples: 7000
    ✅ Brier=0.0045, ECE=0.0023, ROC-AUC=0.9923, MAE=1.58°C

  ...

  ✅ Station 2 completed and saved!
     Progress: 1/18 stations

============================================================
LOSO Summary Statistics
============================================================

3h Horizon:
  Stations evaluated: 18
  Brier Score: 0.0051 ± 0.0012 (min: 0.0032, max: 0.0089)
  ROC-AUC: 0.9132 ± 0.0234 (min: 0.8654, max: 0.9654)
  MAE: 4.60°C ± 0.85°C (min: 3.12°C, max: 6.45°C)
  R²: 0.5993 ± 0.0845 (min: 0.4234, max: 0.7891)

...
```

---

## 总结

LOSO评估是**空间泛化验证**的关键方法，特别适用于：

- ✅ 多站点数据集
- ✅ 微气候变化大的场景
- ✅ 需要评估新地点应用能力的任务
- ✅ 生产环境部署前的验证

**重要性**: ⭐⭐⭐⭐⭐

**建议**: 在完成标准评估后，**一定要进行LOSO评估**，确保模型具有良好的空间泛化能力。

---

---

## LOSO 检查点和恢复 (Checkpoint and Resume)

### 概述

LOSO评估已重构为**按站点逐个处理**，并在每个站点完成后**立即保存结果**。这可以防止进程崩溃时的数据丢失，并支持从检查点恢复。

### 处理顺序

**之前**:
- 按时间窗口处理（3h, 6h, 12h, 24h）
- 对每个时间窗口，处理所有站点
- 只在最后保存结果

**现在**:
- 按站点处理（站点1, 站点2, ..., 站点18）
- 对每个站点，处理所有时间窗口（3h, 6h, 12h, 24h）
- 每个站点完成后立即保存结果

### 检查点系统

**检查点文件**:
- `checkpoint.json`: 跟踪已完成的站点
- `station_results.json`: 存储所有已完成站点的结果
- `summary.json`: 最终汇总统计（在最后生成）

**检查点格式**:
```json
{
  "completed_stations": [2, 7, 15, ...],
  "total_stations": 18,
  "completed_count": 3
}
```

### 恢复功能

**恢复支持**:
- 自动跳过已完成的站点
- 从 `station_results.json` 加载现有结果
- 从下一个未完成的站点继续
- 实时跟踪进度

**使用方法**:
```bash
# 初始运行
python scripts/train/train_frost_forecast.py --loso ...

# 如果中断，恢复运行
python scripts/train/train_frost_forecast.py --loso --resume-loso ...
```

### 处理流程

#### Step 1: 处理站点1
1. 加载站点1（测试）和所有其他站点（训练）的数据
2. 处理所有时间窗口（3h, 6h, 12h, 24h）
3. 训练模型并评估
4. 保存结果到 `station_results.json`
5. 更新 `checkpoint.json`（标记站点1为已完成）
6. 释放内存

#### Step 2: 处理站点2
1. 检查检查点（如果站点1已完成则跳过）
2. 加载站点2（测试）和所有其他站点（训练）的数据
3. 处理所有时间窗口
4. 保存结果
5. 更新检查点
6. 释放内存

#### Step 3: 继续...
- 重复所有18个站点
- 每个站点独立处理
- 每个站点完成后立即保存结果

#### Step 4: 生成汇总
- 从所有站点结果计算汇总统计
- 保存到 `summary.json`
- 生成 `station_metrics.csv` 用于分析

### 内存优化

**之前（旧实现）**:
- 加载所有站点的所有数据
- 按时间窗口处理（一次处理所有站点）
- 高内存使用（64GB+）
- 崩溃风险高

**现在（新实现）**:
- 只加载当前站点的数据（训练+测试）
- 一次处理一个站点
- 每个站点处理后保存并释放内存
- 内存使用较低（~16GB 每个站点）
- 崩溃风险降低

### 错误处理

#### 站点级错误
- 如果站点失败，记录日志并跳过
- 其他站点继续处理
- 失败的站点可以稍后重试

#### 时间窗口级错误
- 如果某个时间窗口失败，其他时间窗口继续
- 站点仍标记为已完成（部分结果）
- 失败的时间窗口可以手动重试

### 检查点管理

#### 查看检查点状态
```bash
# 查看检查点文件
cat experiments/full_data_training/loso/checkpoint.json
```

#### 重置检查点
```bash
# 删除检查点以重新开始
rm experiments/full_data_training/loso/checkpoint.json
rm experiments/full_data_training/loso/station_results.json
```

#### 手动检查点
```python
# 手动加载检查点
import json
from pathlib import Path

checkpoint_file = Path("experiments/full_data_training/loso/checkpoint.json")
with open(checkpoint_file, "r") as f:
    checkpoint = json.load(f)

print(f"Completed: {checkpoint['completed_count']}/{checkpoint['total_stations']}")
print(f"Stations: {checkpoint['completed_stations']}")
```

### 监控进度

#### 实时进度

脚本在每个站点完成后打印进度：
```
✅ Station 2 completed and saved!
   Progress: 1/18 stations

✅ Station 7 completed and saved!
   Progress: 2/18 stations
```

#### 查看进度
```bash
# 查看检查点文件
cat experiments/full_data_training/loso/checkpoint.json | jq '.completed_count'

# 查看站点结果
cat experiments/full_data_training/loso/station_results.json | jq '. | length'
```

### 故障排除

#### 问题：检查点未找到

**解决方案**: 不使用 `--resume-loso` 重新运行，或手动创建检查点。

#### 问题：站点结果缺失

**解决方案**: 检查 `station_results.json` - 可能不完整。删除检查点以重新开始。

#### 问题：内存仍然很高

**解决方案**:
1. 减少模型参数中的 `n_jobs`
2. 一次处理更少的时间窗口
3. 使用更小的批处理大小

#### 问题：站点重复失败

**解决方案**:
1. 检查该站点的数据质量
2. 减少模型复杂度
3. 手动跳过有问题的站点

### 优势

1. **更低的内存使用**: 一次处理一个站点（~16GB vs 64GB+）
2. **崩溃恢复**: 每个站点完成后保存结果（无数据丢失）
3. **恢复支持**: 自动跳过已完成的站点
4. **进度跟踪**: 实时进度更新
5. **错误隔离**: 失败不影响其他站点
6. **快速恢复**: 从最后一个完成的站点恢复

---

**相关文档**:
- `docs/TRAINING_AND_EVALUATION.md`: 训练和评估完整指南
- `src/evaluation/validators.py`: LOSO实现代码

**最后更新**: 2025-11-13

