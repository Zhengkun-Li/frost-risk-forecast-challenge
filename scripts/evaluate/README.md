# 评估脚本使用指南

**最后更新**: 2025-11-12

## evaluate_model.py

评估已训练的模型。

### 基本用法

```bash
# 评估模型（使用配置中的数据路径）
python scripts/evaluate/evaluate_model.py experiments/runs/lightgbm_baseline_20250101_120000

# 指定测试数据
python scripts/evaluate/evaluate_model.py experiments/runs/lightgbm_baseline_20250101_120000 --data data/interim/features/test_features.parquet

# 指定输出目录
python scripts/evaluate/evaluate_model.py experiments/runs/lightgbm_baseline_20250101_120000 --output experiments/evaluations/eval_001
```

### 参数说明

- `model_dir`: 模型目录路径（必须）
- `--data`: 测试数据路径（可选，默认使用配置中的数据路径）
- `--output`: 评估结果输出目录（可选）

### 输出

- `evaluation_metrics.json`: 评估指标
- `predictions.csv`: 预测结果

## compare_models.py

对比多个训练好的模型。

### 基本用法

```bash
# 对比两个模型
python scripts/evaluate/compare_models.py \
    experiments/runs/model1 \
    experiments/runs/model2

# 对比多个模型并指定输出
python scripts/evaluate/compare_models.py \
    experiments/runs/model1 \
    experiments/runs/model2 \
    experiments/runs/model3 \
    --output experiments/comparisons/comparison_001
```

### 输出

- `comparison_table.csv`: 对比表格
- `comparison_summary.json`: 对比摘要（包含最佳模型）

### 示例输出

```
Best Models by Metric:
============================================================
MAE: lightgbm_baseline (1.2345)
RMSE: lightgbm_baseline (2.3456)
R2: lstm_model (0.9876)
ROC_AUC: lightgbm_baseline (0.9123)
============================================================
```

