# 模型配置和定义指南

**最后更新**: 2025-01-XX  
**版本**: 2.0 (重构后模块化版本)

## 📍 模型配置和定义位置

模型配置和定义分为三个层次：

### 1️⃣ 模型参数配置

**位置**: `src/training/model_config.py`

这个文件包含所有模型的参数配置，包括：

- **`get_model_params()`** - 获取模型参数
  - 根据模型类型（lightgbm, xgboost, catboost, etc.）返回参数字典
  - 支持分类和回归任务
  - 支持标准训练和 LOSO 评估（自动调整参数）
  
- **`get_model_class()`** - 获取模型类
  - 根据模型类型返回对应的模型类
  - 自动导入相应的模型模块
  
- **`get_model_config()`** - 获取完整配置
  - 组合模型参数、任务类型、模型名称等
  - 添加模型特定的配置（如 ensemble 的 base_models）
  
- **`get_resource_aware_config()`** - 资源感知配置
  - 根据系统内存自动调整 LSTM 模型的参数
  - 用于深度学习模型的资源优化

### 2️⃣ 模型类定义

**位置**: `src/models/` 目录

所有模型类都在 `src/models/` 目录下，按照模型类型组织：

#### 机器学习模型 (`src/models/ml/`)

- **`lightgbm_model.py`** - LightGBM 模型
  - `LightGBMModel` 类
  - 继承自 `BaseModel`
  - 支持分类和回归任务

- **`xgboost_model.py`** - XGBoost 模型
  - `XGBoostModel` 类
  - 继承自 `BaseModel`
  - 支持分类和回归任务

- **`catboost_model.py`** - CatBoost 模型
  - `CatBoostModel` 类
  - 继承自 `BaseModel`
  - 支持分类和回归任务

- **`random_forest_model.py`** - Random Forest 模型
  - `RandomForestModel` 类
  - 继承自 `BaseModel`
  - 支持分类和回归任务

- **`ensemble_model.py`** - Ensemble 模型
  - `EnsembleModel` 类
  - 继承自 `BaseModel`
  - 组合 LightGBM、XGBoost 和 CatBoost
  - 支持分类和回归任务

#### 深度学习模型 (`src/models/deep/`)

- **`lstm_model.py`** - LSTM 模型
  - `LSTMForecastModel` 类
  - 继承自 `BaseModel`
  - 用于时间序列预测
  - 支持回归任务（温度预测）

- **`lstm_multitask_model.py`** - LSTM Multi-task 模型
  - `LSTMMultiTaskForecastModel` 类
  - 继承自 `BaseModel`
  - 同时预测温度和霜冻概率
  - 支持多任务学习

#### 传统时间序列模型 (`src/models/traditional/`)

- **`prophet_model.py`** - Prophet 模型
  - `ProphetModel` 类
  - 继承自 `BaseModel`
  - 用于时间序列预测
  - 支持回归任务（温度预测）

### 3️⃣ 基础模型接口

**位置**: `src/models/base.py`

所有模型都继承自 `BaseModel` 类，它定义了统一的接口：

- **`fit()`** - 训练模型
- **`predict()`** - 预测
- **`predict_proba()`** - 预测概率（分类任务）
- **`save()`** - 保存模型
- **`load()`** - 加载模型
- **`get_feature_importance()`** - 获取特征重要性

## 🔧 如何配置模型

### 修改现有模型的参数

1. **编辑 `src/training/model_config.py`**
2. **找到 `get_model_params()` 函数**
3. **修改对应模型类型的参数**

**示例**: 修改 LightGBM 的参数

```python
# 在 src/training/model_config.py 中
if model_type == "lightgbm":
    if task_type == "classification":
        return {
            "n_estimators": 300,  # 修改为 300
            "learning_rate": 0.01,  # 修改为 0.01
            "max_depth": 10,  # 修改为 10
            "num_leaves": 127,  # 修改为 127
            # ... 其他参数
        }
```

### 添加新模型

1. **在 `src/models/` 目录下创建新模型类**
   - 创建新文件（如 `src/models/ml/my_model.py`）
   - 实现 `MyModel` 类，继承自 `BaseModel`
   - 实现所有抽象方法：`fit()`, `predict()`, `predict_proba()`

2. **在 `src/training/model_config.py` 中添加模型参数配置**
   - 在 `get_model_params()` 函数中添加新模型类型的参数

3. **在 `src/training/model_config.py` 中添加模型类映射**
   - 在 `get_model_class()` 函数中添加新模型类型的类映射

4. **在主脚本中添加模型类型**
   - 在 `scripts/train/train_frost_forecast.py` 中添加新模型类型到 `argparse` 的 `choices`

**示例**: 添加新的模型类型

```python
# 1. 创建 src/models/ml/my_model.py
from src.models.base import BaseModel

class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化模型
    
    def fit(self, X, y, **kwargs):
        # 训练模型
        return self
    
    def predict(self, X):
        # 预测
        return predictions
    
    def predict_proba(self, X):
        # 预测概率
        return probabilities

# 2. 在 src/training/model_config.py 中添加参数配置
def get_model_params(model_type, task_type, max_workers, for_loso):
    # ... 其他模型 ...
    elif model_type == "my_model":
        return {
            "param1": value1,
            "param2": value2,
            # ... 其他参数
        }

# 3. 在 src/training/model_config.py 中添加类映射
def get_model_class(model_type):
    # ... 其他模型 ...
    elif model_type == "my_model":
        from src.models.ml.my_model import MyModel
        return MyModel

# 4. 在 scripts/train/train_frost_forecast.py 中添加模型类型
parser.add_argument(
    "--model",
    choices=["lightgbm", "xgboost", "my_model", ...],  # 添加 "my_model"
    default="lightgbm"
)
```

## 📊 模型参数配置详解

### LightGBM 参数

```python
{
    "n_estimators": 200,        # 树的数量
    "learning_rate": 0.05,      # 学习率
    "max_depth": 8,             # 最大深度
    "num_leaves": 63,           # 叶子节点数
    "random_state": 42,         # 随机种子
    "n_jobs": max_workers,      # 并行线程数
    "subsample": 0.8,           # 样本采样率
    "colsample_bytree": 0.8,    # 特征采样率
    "reg_alpha": 0.1,           # L1 正则化
    "reg_lambda": 0.1,          # L2 正则化
}
```

### XGBoost 参数

```python
{
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 8,
    "random_state": 42,
    "n_jobs": max_workers,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "tree_method": "hist",      # 树构建方法
    "objective": "binary:logistic" or "reg:squarederror",  # 目标函数
}
```

### CatBoost 参数

```python
{
    "iterations": 200,          # 迭代次数（CatBoost 使用 iterations 而不是 n_estimators）
    "learning_rate": 0.05,
    "depth": 8,                 # 深度（CatBoost 使用 depth 而不是 max_depth）
    "random_state": 42,
    "thread_count": max_workers,  # 线程数（CatBoost 使用 thread_count 而不是 n_jobs）
    "subsample": 0.8,
    "colsample_bylevel": 0.8,   # 特征采样率（CatBoost 使用 colsample_bylevel）
    "l2_leaf_reg": 0.1,         # L2 正则化（CatBoost 使用 l2_leaf_reg）
}
```

### Random Forest 参数

```python
{
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": max_workers,
}
```

### LSTM 参数

```python
{
    "sequence_length": 24,      # 序列长度（小时）
    "hidden_size": 64 or 128,   # 隐藏层大小（根据内存自动调整）
    "num_layers": 2,            # LSTM 层数
    "dropout": 0.2,             # Dropout 率
    "learning_rate": 0.001,     # 学习率
    "batch_size": 16 or 32 or 64,  # 批次大小（根据内存自动调整）
    "epochs": 100,              # 最大轮数
    "early_stopping": True,     # 早停机制
    "patience": 10,             # 早停耐心值
    "lr_scheduler": True,       # 学习率调度器
    "gradient_clip": 1.0,       # 梯度裁剪
    "save_best_model": True,    # 保存最佳模型
}
```

### Prophet 参数

```python
{
    "yearly_seasonality": True,   # 年度季节性
    "weekly_seasonality": True,   # 周季节性
    "daily_seasonality": True,    # 日季节性
    "seasonality_mode": "multiplicative",  # 季节性模式
}
```

## 🔍 资源感知配置

对于 LSTM 模型，系统会根据可用内存自动调整配置：

- **>= 32GB 内存**: `hidden_size=128`, `batch_size=64`
- **16-32GB 内存**: `hidden_size=128`, `batch_size=32`
- **< 16GB 内存**: `hidden_size=64`, `batch_size=16`

在 LOSO 评估中，配置会更小以节省内存（18 个站点 × 4 个时间范围 = 72 个模型）。

## 📝 配置示例

### 标准训练配置

```python
# 在 src/training/model_config.py 中
frost_config = get_model_config(
    model_type="lightgbm",
    horizon=3,
    task_type="classification",
    max_workers=8,
    for_loso=False
)
```

### LOSO 评估配置

```python
# 在 src/training/loso_evaluator.py 中
frost_config = get_model_config(
    model_type="lightgbm",
    horizon=3,
    task_type="classification",
    max_workers=8,
    for_loso=True,
    station_id=2
)
```

## 🚀 使用方式

配置好模型后，可以直接使用：

```bash
# 使用默认配置
python scripts/train/train_frost_forecast.py \
    --model lightgbm \
    --horizons 3 6 12 24

# 使用自定义配置（需要在 model_config.py 中修改）
python scripts/train/train_frost_forecast.py \
    --model xgboost \
    --horizons 3 6 12 24
```

## 📚 相关文档

- [训练脚本使用指南](../scripts/train/README.md)
- [模型比较指南](./MODEL_COMPARISON_GUIDE.md)
- [LSTM 和 Prophet 模型说明](./LSTM_AND_PROPHET_EXPLAINED.md)

## 🆘 故障排除

### 模型参数不生效

如果修改了 `src/training/model_config.py` 中的参数，但训练时没有使用新参数：

1. 检查是否正确修改了 `get_model_params()` 函数
2. 检查模型类型是否匹配
3. 检查任务类型是否匹配（classification 或 regression）

### 添加新模型后无法使用

如果添加了新模型但无法使用：

1. 检查模型类是否正确继承自 `BaseModel`
2. 检查是否正确实现了所有抽象方法
3. 检查是否在 `src/training/model_config.py` 的 `get_model_class()` 中添加了类映射
4. 检查是否在 `scripts/train/train_frost_forecast.py` 中添加了模型类型

### 资源不足错误

如果遇到内存不足的错误：

1. 对于 LSTM 模型，系统会自动调整配置
2. 对于树模型，可以减少 `n_estimators` 或 `max_depth`
3. 对于 LOSO 评估，系统会自动使用较小的配置

## 📝 更新日志

### v2.0 (2025-01-XX)

- ✅ 重构为模块化结构
- ✅ 添加资源感知配置
- ✅ 支持多种模型类型
- ✅ 改进的参数配置系统

### v1.0 (2025-11-12)

- 初始版本
- 基本的模型配置功能

