# 训练脚本使用指南

**最后更新**: 2025-01-XX  
**版本**: 2.0 (重构后模块化版本)

## 📁 模块结构

训练脚本已重构为模块化结构，提高代码可维护性和可扩展性：

```
src/training/                 - 训练模块 (库代码)
├── data_preparation.py      - 数据准备模块
│   ├── load_and_prepare_data()      - 数据加载和准备
│   ├── create_frost_labels()        - 创建霜冻标签
│   └── prepare_features_and_targets() - 准备特征和目标
│
├── model_config.py          - 模型配置模块
│   ├── get_model_params()           - 获取模型参数
│   ├── get_model_class()            - 获取模型类
│   ├── get_model_config()           - 获取完整配置
│   └── get_resource_aware_config()  - 资源感知配置
│
├── model_trainer.py         - 模型训练模块
│   ├── train_models_for_horizon()    - 训练模型 (主函数)
│   ├── train_frost_model()          - 训练霜冻模型
│   ├── train_temp_model()           - 训练温度模型
│   ├── train_multitask_model()      - 训练多任务模型
│   ├── evaluate_models()           - 评估模型
│   └── save_models_and_results()   - 保存模型和结果
│
└── loso_evaluator.py        - LOSO 评估模块
    ├── perform_loso_evaluation()    - LOSO 评估 (主函数)
    ├── train_loso_models_for_horizon() - LOSO 模型训练
    └── calculate_loso_summary()    - 计算 LOSO 摘要

scripts/train/                - 训练脚本
└── train_frost_forecast.py  - 主脚本
    └── main()                       - 参数解析和协调逻辑
```

## 🚀 快速开始

### 基本用法

```bash
# 训练 LightGBM 模型
python scripts/train/train_frost_forecast.py \
    --model lightgbm \
    --horizons 3 6 12 24 \
    --output experiments/lightgbm/top175_features

# 训练 XGBoost 模型
python scripts/train/train_frost_forecast.py \
    --model xgboost \
    --horizons 3 6 12 24 \
    --output experiments/xgboost/top175_features

# 训练 CatBoost 模型
python scripts/train/train_frost_forecast.py \
    --model catboost \
    --horizons 3 6 12 24 \
    --output experiments/catboost/top175_features

# 训练 Random Forest 模型
python scripts/train/train_frost_forecast.py \
    --model random_forest \
    --horizons 3 6 12 24 \
    --output experiments/random_forest/top175_features

# 训练 Ensemble 模型
python scripts/train/train_frost_forecast.py \
    --model ensemble \
    --horizons 3 6 12 24 \
    --output experiments/ensemble/top175_features

# 训练 LSTM 模型
python scripts/train/train_frost_forecast.py \
    --model lstm \
    --horizons 3 6 12 24 \
    --output experiments/lstm/top175_features

# 训练 LSTM Multi-task 模型
python scripts/train/train_frost_forecast.py \
    --model lstm_multitask \
    --horizons 3 6 12 24 \
    --output experiments/lstm_multitask/top175_features

# 训练 Prophet 模型
python scripts/train/train_frost_forecast.py \
    --model prophet \
    --horizons 3 6 12 24 \
    --output experiments/prophet/top175_features
```

### LOSO (Leave-One-Station-Out) 评估

```bash
# 运行 LOSO 评估并保存所有模型
python scripts/train/train_frost_forecast.py \
    --model lightgbm \
    --horizons 3 6 12 24 \
    --loso \
    --save-loso-models \
    --output experiments/lightgbm/top175_features

# 运行 LOSO 评估，只保存最差的 N 个站点的模型
python scripts/train/train_frost_forecast.py \
    --model lightgbm \
    --horizons 3 6 12 24 \
    --loso \
    --save-loso-worst-n 3 \
    --output experiments/lightgbm/top175_features

# 运行 LOSO 评估，只保存指定时间范围的模型
python scripts/train/train_frost_forecast.py \
    --model lightgbm \
    --horizons 3 6 12 24 \
    --loso \
    --save-loso-horizons 24 \
    --output experiments/lightgbm/top175_features

# 恢复 LOSO 评估（从检查点继续）
python scripts/train/train_frost_forecast.py \
    --model lightgbm \
    --horizons 3 6 12 24 \
    --loso \
    --resume-loso \
    --output experiments/lightgbm/top175_features
```

## 📋 参数说明

### 主要参数

- `--data`: 原始数据路径（默认：自动检测）
- `--output`: 输出目录（默认：自动生成时间戳目录）
- `--horizons`: 预测时间范围，单位：小时（默认：3 6 12 24）
- `--model`: 模型类型（可选：lightgbm, xgboost, catboost, random_forest, ensemble, lstm, lstm_multitask, prophet）
- `--frost-threshold`: 霜冻温度阈值，单位：°C（默认：0.0）
- `--sample-size`: 采样大小（用于快速测试，默认：使用所有数据）

### 特征选择参数

- `--feature-selection`: 特征选择配置文件路径（JSON 格式）
- `--top-k-features`: 使用重要性排名前 K 的特征（覆盖特征选择配置）

### LOSO 评估参数

- `--loso`: 执行 LOSO 评估
- `--resume-loso`: 从检查点恢复 LOSO 评估
- `--save-loso-models`: 保存所有 LOSO 模型
- `--save-loso-worst-n`: 只保存最差的 N 个站点的模型
- `--save-loso-horizons`: 只保存指定时间范围的模型

## 📊 输出结构

训练完成后，输出目录结构如下：

```
output_dir/
├── labeled_data.parquet          # 标注数据
├── full_training/                # 标准训练结果
│   ├── labeled_data.parquet      # 标注数据（副本）
│   ├── summary.json              # 训练摘要
│   └── horizon_{horizon}h/       # 每个时间范围的结果
│       ├── frost_classifier/     # 霜冻分类模型
│       │   ├── model.pkl         # 模型文件
│       │   └── model_metadata.json
│       ├── temp_regressor/       # 温度回归模型
│       │   ├── model.pkl         # 模型文件
│       │   └── model_metadata.json
│       ├── frost_metrics.json    # 霜冻评估指标
│       ├── temp_metrics.json     # 温度评估指标
│       ├── predictions.json      # 预测结果
│       └── reliability_diagram.png  # 可靠性图
│
└── loso/                         # LOSO 评估结果
    ├── checkpoint.json           # 检查点文件
    ├── station_results.json      # 站点结果
    ├── station_metrics.csv       # 站点指标（CSV 格式）
    ├── summary.json              # LOSO 摘要统计
    └── station_{station_id}/     # 每个站点的结果
        └── horizon_{horizon}h/   # 每个时间范围的结果
            ├── frost_classifier/ # 霜冻分类模型
            └── temp_regressor/   # 温度回归模型
```

## 🔧 支持的模型类型

### 树模型（Tree-based Models）

- **LightGBM**: 快速、高效的梯度提升框架
- **XGBoost**: 可扩展的梯度提升框架
- **CatBoost**: 自动处理类别特征的梯度提升框架
- **Random Forest**: 随机森林基准模型
- **Ensemble**: 集成模型（LightGBM + XGBoost + CatBoost 平均）

### 深度学习模型（Deep Learning Models）

- **LSTM**: 长短期记忆网络，用于时间序列预测
- **LSTM Multi-task**: 多任务 LSTM 模型，同时预测温度和霜冻概率

### 传统时间序列模型（Traditional Time Series Models）

- **Prophet**: Facebook 的时间序列预测框架，适用于趋势和季节性预测

## 📈 评估指标

### 分类指标（Frost Probability）

- **Brier Score**: 概率预测的均方误差（越低越好）
- **ECE (Expected Calibration Error)**: 预期校准误差（越低越好）
- **ROC-AUC**: ROC 曲线下面积（越高越好）
- **PR-AUC**: 精确率-召回率曲线下面积（越高越好）

### 回归指标（Temperature）

- **MAE (Mean Absolute Error)**: 平均绝对误差（越低越好）
- **RMSE (Root Mean Squared Error)**: 均方根误差（越低越好）
- **R² (R-squared)**: 决定系数（越高越好）

## 💡 使用示例

### 示例 1: 快速测试（使用采样数据）

```bash
python scripts/train/train_frost_forecast.py \
    --model lightgbm \
    --horizons 3 6 \
    --sample-size 100000 \
    --output experiments/test_run
```

### 示例 2: 完整训练（使用特征选择）

```bash
python scripts/train/train_frost_forecast.py \
    --model lightgbm \
    --horizons 3 6 12 24 \
    --top-k-features 175 \
    --output experiments/lightgbm/top175_features
```

### 示例 3: LOSO 评估（保存所有模型）

```bash
python scripts/train/train_frost_forecast.py \
    --model lightgbm \
    --horizons 3 6 12 24 \
    --loso \
    --save-loso-models \
    --output experiments/lightgbm/top175_features
```

### 示例 4: 恢复中断的 LOSO 评估

```bash
python scripts/train/train_frost_forecast.py \
    --model lightgbm \
    --horizons 3 6 12 24 \
    --loso \
    --resume-loso \
    --output experiments/lightgbm/top175_features
```

## 🔍 模块说明

### data_preparation.py

负责数据加载、清理、特征工程和标签创建。

**主要函数：**
- `load_and_prepare_data()`: 加载和准备数据
- `create_frost_labels()`: 创建霜冻标签
- `prepare_features_and_targets()`: 准备特征和目标

### model_config.py

负责模型参数配置和模型类选择。

**主要函数：**
- `get_model_params()`: 获取模型参数
- `get_model_class()`: 获取模型类
- `get_model_config()`: 获取完整配置
- `get_resource_aware_config()`: 资源感知配置（用于 LSTM 等深度学习模型）

### model_trainer.py

负责模型训练、评估和结果保存。

**主要函数：**
- `train_models_for_horizon()`: 训练模型（主函数）
- `train_frost_model()`: 训练霜冻模型
- `train_temp_model()`: 训练温度模型
- `train_multitask_model()`: 训练多任务模型
- `evaluate_models()`: 评估模型
- `save_models_and_results()`: 保存模型和结果

### loso_evaluator.py

负责 LOSO 评估、模型训练和摘要统计。

**主要函数：**
- `perform_loso_evaluation()`: LOSO 评估（主函数）
- `train_loso_models_for_horizon()`: LOSO 模型训练
- `calculate_loso_summary()`: 计算 LOSO 摘要

## 🐛 故障排除

### 内存不足

如果遇到内存不足的问题，可以：

1. **减少采样大小**：使用 `--sample-size` 参数
2. **减少时间范围**：只训练部分时间范围（如 `--horizons 3 6`）
3. **使用较小的模型**：LSTM 模型会自动根据系统内存调整配置

### LOSO 评估中断

如果 LOSO 评估中断，可以使用 `--resume-loso` 参数从检查点恢复：

```bash
python scripts/train/train_frost_forecast.py \
    --model lightgbm \
    --horizons 3 6 12 24 \
    --loso \
    --resume-loso \
    --output experiments/lightgbm/top175_features
```

### 模型文件不存在

如果模型文件不存在，训练脚本会自动训练新模型。如果模型已存在且想重新训练，可以：

1. 删除现有的模型文件
2. 使用 `--skip-if-exists False` 参数（如果支持）

## 📚 相关文档

- [项目 README](../README.md)
- [特征工程文档](../../docs/FEATURE_ENGINEERING.md)
- [模型比较指南](../../docs/MODEL_COMPARISON_GUIDE.md)
- [LSTM 和 Prophet 模型说明](../../docs/LSTM_AND_PROPHET_EXPLAINED.md)

## 🆘 获取帮助

如果遇到问题，可以：

1. 查看日志文件（位于输出目录）
2. 检查错误消息
3. 查看相关文档
4. 提交 Issue 到项目仓库

## 📝 更新日志

### v2.0 (2025-01-XX)

- ✅ 重构为模块化结构
- ✅ 支持多种模型类型（LightGBM, XGBoost, CatBoost, Random Forest, Ensemble, LSTM, LSTM Multi-task, Prophet）
- ✅ 改进的 LOSO 评估支持
- ✅ 资源感知配置（用于深度学习模型）
- ✅ 更好的错误处理和日志记录

### v1.0 (2025-11-12)

- 初始版本
- 支持 LightGBM 模型
- 基本的训练和评估功能
