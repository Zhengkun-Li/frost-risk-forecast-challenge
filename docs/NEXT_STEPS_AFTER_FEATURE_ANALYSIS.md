# Next Steps After Feature Importance Analysis

**Date**: 2025-11-13  
**Status**: Ready to proceed with full data training

---

## ✅ Completed Work

1. ✅ **Feature Engineering**: 298 features created
2. ✅ **100k Sample Training**: Test training completed
3. ✅ **Feature Importance Analysis**: Comprehensive analysis completed
4. ✅ **Feature Selection Recommendations**: Top 175 features (90% importance) recommended

---

## 🎯 Recommended Next Steps

### Priority 1: Full Data Training ⭐ **RECOMMENDED**

**Purpose**: Train models with all available data for production use

#### Option A: Use All Features (Best Performance)

```bash
# Full training with all 298 features (all time horizons)
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/full_data_training \
    --n-jobs 8
```

**Expected Results**:
- Train models with all 298 features
- Maximum performance (100% feature importance)
- Training time: ~1.5-2.5 hours (without LOSO)
- Memory usage: ~16-32GB

#### Option B: Use Top 175 Features (Balanced - Recommended)

**Note**: Currently, the training script does not have built-in feature selection. You need to:

1. **Manual Feature Selection** (Recommended for now):
   - Load feature importance rankings from `experiments/lightgbm/feature_importance/feature_importance_3h_all.csv`
   - Select top 175 features
   - Modify training script to use only selected features

2. **Or use all features** and apply feature selection in post-processing

**Expected Results**:
- Train models with top 175 features (90% importance)
- Good performance with reduced memory/time
- Training time: ~1-1.5 hours (without LOSO)
- Memory usage: ~12-24GB

#### Training Commands

```bash
# Standard evaluation (time-based split)
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/full_data_training \
    --n-jobs 8

# Expected output:
# - experiments/full_data_training/horizon_3h/
# - experiments/full_data_training/horizon_6h/
# - experiments/full_data_training/horizon_12h/
# - experiments/full_data_training/horizon_24h/
```

**Expected Outcomes**:
- ✅ Models trained for all 4 horizons (3h, 6h, 12h, 24h)
- ✅ Performance metrics (Brier Score, ROC-AUC, MAE, RMSE, R²)
- ✅ Model files saved for inference
- ✅ Training logs and evaluation reports

---

### Priority 2: LOSO Evaluation ⭐ **REQUIRED**

**Purpose**: Validate spatial generalization across all stations

#### LOSO Evaluation Command

```bash
# LOSO evaluation (spatial generalization)
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --loso \
    --model lightgbm \
    --output experiments/full_data_training \
    --n-jobs 8
```

**Expected Results**:
- Evaluate model performance across all 18 stations
- Assess spatial generalization
- Training time: ~4-7 hours (with LOSO)
- Memory usage: ~16GB per station (processed one at a time)

#### Resume LOSO Evaluation

If LOSO evaluation is interrupted:

```bash
# Resume LOSO evaluation (skip completed stations)
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --loso \
    --resume-loso \
    --model lightgbm \
    --output experiments/full_data_training \
    --n-jobs 8
```

**Expected Outcomes**:
- ✅ LOSO results for all 18 stations
- ✅ Summary statistics (mean, std, min, max)
- ✅ Station-level performance metrics
- ✅ Spatial generalization assessment

**Output Files**:
- `experiments/full_data_training/loso/checkpoint.json`
- `experiments/full_data_training/loso/station_results.json`
- `experiments/full_data_training/loso/summary.json`
- `experiments/full_data_training/loso/station_metrics.csv`

---

### Priority 3: Feature Selection Optimization (Optional)

**Purpose**: Optimize feature set based on importance analysis

#### Current Status

The training script does not have built-in feature selection. Options:

#### Option A: Manual Feature Selection

1. **Load Feature Importance**:
   ```python
   import pandas as pd
   
   # Load feature importance rankings
   df_importance = pd.read_csv(
       "experiments/lightgbm/feature_importance/feature_importance_3h_all.csv"
   )
   
   # Select top 175 features
   top_features = df_importance.head(175)["feature"].tolist()
   ```

2. **Modify Training Script**:
   - Add feature selection parameter to `train_frost_forecast.py`
   - Filter features before training
   - Save selected features list

#### Option B: Post-Processing Feature Selection

1. Train with all features
2. Analyze feature importance from full training
3. Select top features for next iteration
4. Retrain with selected features

#### Recommended Approach

**For now**: Use all 298 features for full training, then:
1. Analyze feature importance from full training
2. Compare with 100k sample analysis
3. Select top features for optimization
4. Retrain if needed

---

### Priority 4: Model Deployment Testing (Optional)

**Purpose**: Test inference pipeline and validate output format

#### Inference Testing

```bash
# Test inference with trained models
python3 scripts/inference/predict_frost.py \
    --models experiments/full_data_training \
    --horizons 3 6 12 24 \
    --index 0

# Batch predictions
python3 scripts/inference/predict_frost.py \
    --models experiments/full_data_training \
    --horizons 3 6 12 24 \
    --sample-size 100 \
    --output predictions.csv
```

**Expected Outcomes**:
- ✅ Inference pipeline works correctly
- ✅ Output format matches challenge requirements
- ✅ Predictions are reasonable
- ✅ Model files load correctly

---

## 📋 Step-by-Step Workflow

### Step 1: Full Data Training (Standard Evaluation)

```bash
# 1. Train models with all data (standard evaluation)
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/full_data_training \
    --n-jobs 8

# Expected time: ~1.5-2.5 hours
# Expected output: models for all 4 horizons
```

### Step 2: LOSO Evaluation

```bash
# 2. Run LOSO evaluation (spatial generalization)
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --loso \
    --model lightgbm \
    --output experiments/full_data_training \
    --n-jobs 8

# Expected time: ~4-7 hours
# Expected output: LOSO results for all 18 stations
```

### Step 3: Analyze Results

```bash
# 3. Analyze training results
# Check performance metrics in:
# - experiments/full_data_training/horizon_*/metrics.json
# - experiments/full_data_training/loso/summary.json
```

### Step 4: Feature Selection (Optional)

```bash
# 4. Analyze feature importance from full training
python3 scripts/analyze_all_features.py \
    --models experiments/full_data_training \
    --horizons 3 6 12 24 \
    --output docs/report/feature_importance_full_training.md

# Compare with 100k sample analysis
# Decide on feature selection strategy
```

### Step 5: Model Deployment Testing

```bash
# 5. Test inference pipeline
python3 scripts/inference/predict_frost.py \
    --models experiments/full_data_training \
    --horizons 3 6 12 24 \
    --index 0

# Verify output format
# Test batch predictions
```

---

## 🎯 Recommended Priority Order

1. **⭐ Full Data Training** (Standard Evaluation)
   - Train models with all data
   - Get baseline performance
   - Time: ~1.5-2.5 hours

2. **⭐ LOSO Evaluation** (Spatial Generalization)
   - Validate spatial generalization
   - Assess cross-station performance
   - Time: ~4-7 hours

3. **Optional: Feature Selection**
   - Analyze feature importance from full training
   - Select top features if needed
   - Retrain with selected features

4. **Optional: Model Deployment**
   - Test inference pipeline
   - Validate output format
   - Prepare for production

---

## 📊 Expected Performance

Based on 100k sample analysis:

### Standard Evaluation (Expected)

| Horizon | Brier Score | ROC-AUC | MAE (°C) | RMSE (°C) | R² |
|---------|-------------|---------|----------|-----------|-----|
| 3h | 0.0028-0.0051 | 0.91-0.99 | 1.2-4.6 | 1.6-5.6 | 0.60-0.97 |
| 6h | 0.0042-0.0055 | 0.88-0.99 | 1.5-4.9 | 2.0-5.9 | 0.54-0.95 |
| 12h | 0.0046-0.0060 | 0.87-0.99 | 1.8-5.2 | 2.3-6.2 | 0.52-0.93 |
| 24h | 0.0063-0.0070 | 0.83-0.98 | 2.0-5.5 | 2.6-6.5 | 0.50-0.92 |

### LOSO Evaluation (Expected)

| Horizon | ROC-AUC (mean ± std) | MAE (mean ± std, °C) | R² (mean ± std) |
|---------|---------------------|---------------------|-----------------|
| 3h | 0.90-0.92 ± 0.02-0.03 | 4.6-4.9 ± 0.8-0.9 | 0.54-0.60 ± 0.05-0.08 |
| 6h | 0.88-0.90 ± 0.02-0.03 | 4.9-5.0 ± 0.8-0.9 | 0.54-0.58 ± 0.05-0.08 |
| 12h | 0.86-0.88 ± 0.02-0.03 | 5.2-5.4 ± 0.9-1.0 | 0.52-0.56 ± 0.05-0.08 |
| 24h | 0.83-0.86 ± 0.02-0.03 | 5.5-5.8 ± 0.9-1.0 | 0.50-0.54 ± 0.05-0.08 |

---

## ⚠️ Important Notes

### 1. Feature Selection

**Current Status**: Training script does not have built-in feature selection.

**Recommendation**: 
- Use all 298 features for full training
- Analyze feature importance from full training
- Compare with 100k sample analysis
- Implement feature selection if needed

### 2. Memory Usage

**Expected Memory Usage**:
- Standard evaluation: ~16-32GB
- LOSO evaluation: ~16GB per station (processed one at a time)

**Optimization**:
- Use `--n-jobs 8` to limit CPU cores
- LOSO evaluation processes one station at a time (memory efficient)
- Feature selection can reduce memory usage by ~40-50%

### 3. Training Time

**Expected Training Time**:
- Standard evaluation: ~1.5-2.5 hours
- LOSO evaluation: ~4-7 hours (18 stations × 4 horizons)
- Total: ~5.5-9.5 hours

**Optimization**:
- Use `--n-jobs 8` to balance CPU usage
- Feature selection can reduce training time by ~40-60%
- LOSO evaluation supports resume (`--resume-loso`)

### 4. Model Performance

**Expected Performance**:
- Full data training should perform better than 100k sample
- LOSO evaluation should show good spatial generalization
- Feature selection may slightly reduce performance (~5-10%)

---

## 🚀 Quick Start Commands

### Full Data Training (Recommended)

```bash
# Standard evaluation
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/full_data_training \
    --n-jobs 8

# LOSO evaluation
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --loso \
    --model lightgbm \
    --output experiments/full_data_training \
    --n-jobs 8
```

### Monitor Training

```bash
# Monitor training progress
./scripts/monitor_training_detailed.sh

# Or check logs
tail -f /tmp/full_training.log
```

### Check Results

```bash
# Check standard evaluation results
cat experiments/full_data_training/horizon_3h/metrics.json | jq

# Check LOSO evaluation results
cat experiments/full_data_training/loso/summary.json | jq
```

---

## 📚 Related Documents

- **[TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md)**: Training and evaluation guide
- **[FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md)**: Feature engineering documentation
- **[docs/report/feature_importance_report.md](report/feature_importance_report.md)**: Feature importance analysis report
- **[docs/report/FEATURE_SELECTION_RECOMMENDATIONS.md](report/FEATURE_SELECTION_RECOMMENDATIONS.md)**: Feature selection recommendations
- **[docs/report/LOSO_EVALUATION_GUIDE.md](report/LOSO_EVALUATION_GUIDE.md)**: LOSO evaluation guide

---

## 🎯 Summary

**Next Steps**:

1. **⭐ Full Data Training** (Priority 1)
   - Train models with all 298 features
   - Standard evaluation: ~1.5-2.5 hours
   - Get baseline performance

2. **⭐ LOSO Evaluation** (Priority 2)
   - Validate spatial generalization
   - LOSO evaluation: ~4-7 hours
   - Assess cross-station performance

3. **Optional: Feature Selection** (Priority 3)
   - Analyze feature importance from full training
   - Select top features if needed
   - Retrain with selected features

4. **Optional: Model Deployment** (Priority 4)
   - Test inference pipeline
   - Validate output format
   - Prepare for production

**Recommendation**: Start with full data training (standard evaluation), then proceed with LOSO evaluation.

---

**Last Updated**: 2025-11-13  
**Status**: Ready to proceed with full data training

