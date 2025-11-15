# 特征分析脚本

**最后更新**: 2025-01-XX

这个目录包含用于特征分析和探索的脚本。

## 📁 脚本列表

### 1. `analyze_all_features.py`

分析数据集中的所有特征，生成统计报告。

**功能：**
- 计算所有特征的统计信息（均值、标准差、缺失率等）
- 分析特征相关性
- 生成特征分析报告

**使用示例：**
```bash
# 分析所有特征
python scripts/analysis/analyze_all_features.py \
    --data data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz \
    --output scripts/analysis/output

# 快速分析（使用采样数据）
python scripts/analysis/analyze_all_features.py \
    --sample-size 100000 \
    --output scripts/analysis/output
```

### 2. `analyze_feature_importance.py`

从训练好的模型中提取和分析特征重要性。

**功能：**
- 从训练好的模型中加载特征重要性
- 分析特征重要性排名
- 生成特征重要性报告和可视化

**使用示例：**
```bash
# 分析特征重要性
python scripts/analysis/analyze_feature_importance.py \
    --model-dir experiments/lightgbm/top175_features/full_training/horizon_3h \
    --model-type lightgbm \
    --task both \
    --output experiments/lightgbm/top175_features/feature_importance
```

### 3. `compare_feature_sets.py`

比较不同特征集的模型性能。

**功能：**
- 比较使用不同特征集训练的模型
- 分析不同特征集的性能差异
- 生成比较报告

**使用示例：**
```bash
# 比较不同特征集
python scripts/analysis/compare_feature_sets.py \
    --model-dirs experiments/lightgbm/top175_features experiments/lightgbm/all_features \
    --names top175 all_features \
    --output scripts/analysis/output
```

### 4. `compare_features.py`

比较不同模型中的特征重要性。

**功能：**
- 比较不同模型中的特征重要性
- 分析特征在不同任务中的表现
- 生成特征比较报告

**使用示例：**
```bash
# 比较特征重要性
python scripts/analysis/compare_features.py \
    --importance-files experiments/lightgbm/top175_features/feature_importance/frost_classifier.csv \
                      experiments/lightgbm/top175_features/feature_importance/temp_regressor.csv \
    --names frost_classifier temp_regressor \
    --output scripts/analysis/output
```

### 5. `generate_feature_report.py`

生成综合特征报告，包含统计信息、重要性和建议。

**功能：**
- 分析所有特征的统计信息
- 提取特征重要性（如果提供模型目录）
- 生成综合特征报告

**使用示例：**
```bash
# 生成综合特征报告
python scripts/analysis/generate_feature_report.py \
    --data data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz \
    --model-dir experiments/lightgbm/top175_features \
    --output scripts/analysis/output
```

## 📊 输出文件

所有脚本的输出文件保存在 `scripts/analysis/output/` 目录（或指定的输出目录）：

- `feature_statistics.csv` - 特征统计信息
- `feature_correlations.csv` - 特征相关性
- `feature_importance_*.csv` - 特征重要性
- `feature_analysis_report.md` - 特征分析报告
- `feature_importance_report_*.md` - 特征重要性报告
- `feature_sets_comparison.csv` - 特征集比较
- `comprehensive_feature_report.md` - 综合特征报告

## 🔧 依赖

这些脚本依赖于：
- `src.data` - 数据加载、清洗、特征工程
- `src.models` - 模型加载和特征重要性提取
- `src.visualization` - 可视化（可选）

## 📚 相关文档

- [特征工程文档](../../docs/FEATURE_ENGINEERING.md)
- [特征参考文档](../../docs/FEATURE_REFERENCE.md)
- [模型配置指南](../../docs/MODEL_CONFIGURATION_GUIDE.md)

