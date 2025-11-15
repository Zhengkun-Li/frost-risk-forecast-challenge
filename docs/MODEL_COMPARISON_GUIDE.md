# 模型和方法对比指南

## 概述

本指南说明如何训练和对比不同的模型和方法，以找到最佳的性能配置。

## 当前支持的模型

1. **LightGBM** ✅ 已实现并训练完成
2. **XGBoost** ✅ 已实现，正在训练中

## 对比维度

### 1. 不同模型对比

#### LightGBM vs XGBoost

使用相同的配置（Top 175特征）训练两个模型，对比：
- 性能指标（ROC-AUC, Brier Score, MAE, RMSE, R²）
- 训练时间
- 模型大小
- 空间泛化能力（LOSO评估）

**训练XGBoost命令：**
```bash
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model xgboost \
    --output experiments/xgboost/top175_features \
    --top-k-features 175
```

### 2. 不同特征集对比

#### Top 175特征 vs 全部298特征

对比不同特征集对性能的影响。

**训练全部特征模型：**
```bash
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/lightgbm/all_features \
    # 不指定 --top-k-features，使用全部特征
```

### 3. 不同超参数对比

可以修改训练脚本中的超参数，对比：
- `n_estimators`: 树的数量
- `learning_rate`: 学习率
- `max_depth`: 最大深度
- `num_leaves`: 叶子节点数

### 4. 标准评估 vs LOSO评估

对比模型在标准评估和LOSO评估下的表现，评估空间泛化能力。

## 使用模型对比脚本

### 1. 配置对比模型

编辑 `config/model_comparison.json`：

```json
{
  "base_dir": "experiments/lightgbm",
  "horizons": ["3h", "6h", "12h", "24h"],
  "models": [
    {
      "name": "LightGBM (Top 175)",
      "experiment_dir": "top175_features",
      "method": "Top 175 features",
      "features": "175",
      "has_loso": true
    },
    {
      "name": "XGBoost (Top 175)",
      "experiment_dir": "xgboost/top175_features",
      "method": "Top 175 features",
      "features": "175",
      "has_loso": false
    }
  ]
}
```

### 2. 运行对比

```bash
python3 scripts/compare_models.py \
    --config config/model_comparison.json \
    --output docs/report/MODEL_COMPARISON.md \
    --csv docs/report/model_comparison.csv
```

### 3. 查看对比报告

生成的报告包含：
- 性能汇总表
- 按时间窗口的详细对比
- LOSO评估对比（如果可用）
- 最佳模型推荐

## 当前对比结果

### LightGBM性能分析

已生成 `docs/report/LIGHTGBM_ANALYSIS.md`，包含：
- 标准评估 vs LOSO评估对比
- 不同时间窗口的性能趋势
- 空间泛化能力分析

**关键发现：**
- 3h时间窗口：ROC-AUC = 0.9965 (标准) / 0.9974 (LOSO)
- 24h时间窗口：ROC-AUC = 0.9843 (标准) / 0.9878 (LOSO)
- LOSO评估显示良好的空间泛化能力

## 下一步

1. **等待XGBoost训练完成**
   - 监控：`tail -f experiments/xgboost/top175_features/xgboost_training.log`
   - 或：`bash scripts/monitor_xgboost_training.sh`

2. **运行模型对比**
   ```bash
   python3 scripts/compare_models.py \
       --config config/model_comparison.json
   ```

3. **可选：运行XGBoost的LOSO评估**
   ```bash
   python3 scripts/train/train_frost_forecast.py \
       --horizons 3 6 12 24 \
       --model xgboost \
       --loso \
       --output experiments/xgboost/top175_features \
       --top-k-features 175
   ```

4. **分析对比结果**
   - 查看生成的对比报告
   - 识别最佳模型
   - 分析不同模型的优缺点

## 对比指标说明

### 霜冻预测指标
- **ROC-AUC**: 越高越好，表示分类能力
- **Brier Score**: 越低越好，表示概率校准
- **ECE**: 越低越好，表示预期校准误差
- **PR-AUC**: 越高越好，表示在不平衡数据上的表现

### 温度预测指标
- **MAE**: 越低越好，平均绝对误差
- **RMSE**: 越低越好，均方根误差
- **R²**: 越高越好，决定系数

### 空间泛化指标
- **LOSO ROC-AUC**: 在未见过的站点上的表现
- **LOSO标准差**: 越小越好，表示稳定性

## 文件结构

```
experiments/
├── lightgbm/                     # LightGBM模型
│   ├── feature_importance/       # 特征重要性分析
│   └── top175_features/          # Top 175特征配置
│       ├── full_training/         # 标准评估
│       └── loso/                 # LOSO评估
└── xgboost/                      # XGBoost模型
    └── top175_features/          # Top 175特征配置
        └── full_training/         # 标准评估

docs/report/
├── LIGHTGBM_ANALYSIS.md          # LightGBM详细分析
└── MODEL_COMPARISON.md           # 模型对比报告（训练完成后生成）
```

