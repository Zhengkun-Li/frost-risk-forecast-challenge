# Full Data Training Execution Plan

**Date**: 2025-11-13  
**Status**: Ready to execute  
**Strategy**: Use all 298 features first, evaluate, then decide on feature reduction

---

## 🎯 Execution Plan

### Phase 1: Full Data Training with All 298 Features

**Purpose**: Get baseline performance and actual resource usage

---

## 📋 Step-by-Step Commands

### Step 1: Standard Evaluation (All Features)

**Command**:
```bash
# Full data training with all 298 features (standard evaluation)
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/full_data_training_all_features \
    --n-jobs 8
```

**Expected Results**:
- Training time: ~1.5-2.5 hours
- Memory usage: ~16-32GB
- Output: Models for all 4 horizons (3h, 6h, 12h, 24h)

**Output Files**:
```
experiments/full_data_training_all_features/
├── horizon_3h/
│   ├── frost_classifier/
│   │   ├── model.pkl
│   │   └── metrics.json
│   ├── temp_regressor/
│   │   ├── model.pkl
│   │   └── metrics.json
│   └── metrics.json
├── horizon_6h/
├── horizon_12h/
├── horizon_24h/
├── labeled_data.parquet
└── training.log
```

**Expected Performance** (based on 100k sample):
- 3h: ROC-AUC 0.91-0.99, MAE 1.2-4.6°C, R² 0.60-0.97
- 6h: ROC-AUC 0.88-0.99, MAE 1.5-4.9°C, R² 0.54-0.95
- 12h: ROC-AUC 0.87-0.99, MAE 1.8-5.2°C, R² 0.52-0.93
- 24h: ROC-AUC 0.83-0.98, MAE 2.0-5.5°C, R² 0.50-0.92

---

### Step 2: Monitor Training Progress

**Monitor Script**:
```bash
# Detailed monitoring (recommended)
./scripts/monitor_training_detailed.sh

# Or simple monitoring
./scripts/monitor_training.sh
```

**Manual Monitoring**:
```bash
# Check training log
tail -f experiments/full_data_training_all_features/training.log

# Check process status
ps aux | grep train_frost_forecast | grep -v grep

# Check resource usage
htop  # or top
```

**What to Monitor**:
- ✅ Training progress (current step, percentage)
- ✅ Memory usage (should stay < 32GB)
- ✅ CPU usage (should be controlled by n-jobs=8)
- ✅ Estimated time remaining (ETA)
- ✅ Current training stage (data loading, cleaning, feature engineering, training)

---

### Step 3: Check Training Results

**After Training Completes**:

```bash
# Check results summary
cat experiments/full_data_training_all_features/horizon_3h/metrics.json | jq

# Check all horizons
for h in 3 6 12 24; do
    echo "=== Horizon ${h}h ==="
    cat experiments/full_data_training_all_features/horizon_${h}h/metrics.json | jq
done
```

**Expected Metrics to Check**:
- **Frost Classification**: Brier Score, ROC-AUC, ECE, PR-AUC
- **Temperature Regression**: MAE, RMSE, R², MAPE
- **Training Time**: Check training.log for timing
- **Memory Usage**: Check system logs or training.log

---

### Step 4: Evaluate Resource Usage

**Check Training Time**:
```bash
# From training.log, check timing
grep -E "(Step|Training|Time|Elapsed)" experiments/full_data_training_all_features/training.log
```

**Check Memory Usage**:
```bash
# From training.log, check memory
grep -i "memory" experiments/full_data_training_all_features/training.log
```

**Evaluate Results**:
1. ✅ **Performance**: Compare with 100k sample results
2. ✅ **Memory**: Check if within limits (< 32GB)
3. ✅ **Time**: Check if acceptable (< 3 hours)
4. ✅ **Overfitting**: Check train/test performance gap

---

### Step 5: Decide on Feature Reduction

#### Decision Criteria

**✅ No Feature Reduction Needed** if:
- Performance is good (meets expectations)
- Memory usage < 32GB
- Training time < 3 hours
- No overfitting issues
- Resources are sufficient

**⚠️ Feature Reduction Recommended** if:
- Memory usage > 32GB
- Training time > 3 hours
- Resources are constrained
- Real-time inference needed

#### If Feature Reduction Needed

**Option 1: Top 175 Features (90% Importance) - Recommended**
```bash
# Note: Feature selection needs to be implemented in training script
# For now, use all features and apply selection later if needed
```

**Option 2: Top 136 Features (80% Importance)**
- For resource-constrained environments
- Faster training (~1 hour)
- Lower memory (~16GB)

---

### Step 6: LOSO Evaluation (After Standard Evaluation)

**After standard evaluation completes successfully**, proceed with LOSO evaluation:

```bash
# LOSO evaluation (spatial generalization)
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --loso \
    --model lightgbm \
    --output experiments/full_data_training_all_features \
    --n-jobs 8
```

**Expected Results**:
- Training time: ~4-7 hours (with checkpointing)
- Memory usage: ~16GB per station (processed one at a time)
- Output: LOSO results for all 18 stations

**If Interrupted**:
```bash
# Resume LOSO evaluation
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --loso \
    --resume-loso \
    --model lightgbm \
    --output experiments/full_data_training_all_features \
    --n-jobs 8
```

---

## 📊 Performance Evaluation Checklist

### After Standard Evaluation

- [ ] Check Brier Score (lower is better, target < 0.01)
- [ ] Check ROC-AUC (higher is better, target > 0.90)
- [ ] Check MAE (lower is better, target < 5°C)
- [ ] Check R² (higher is better, target > 0.50)
- [ ] Check training time (note actual time)
- [ ] Check memory usage (note peak usage)
- [ ] Compare with 100k sample results
- [ ] Check for overfitting (train/test gap)

### After LOSO Evaluation

- [ ] Check mean ROC-AUC across stations (target > 0.85)
- [ ] Check ROC-AUC std (lower is better, target < 0.05)
- [ ] Check mean MAE across stations (target < 6°C)
- [ ] Check MAE std (lower is better, target < 1.0)
- [ ] Identify best/worst performing stations
- [ ] Check spatial generalization quality

---

## 🔍 Resource Usage Analysis

### What to Record

1. **Training Time**:
   - Data loading: __ minutes
   - Data cleaning: __ minutes
   - Feature engineering: __ minutes
   - Model training (per horizon): __ minutes
   - Total time: __ hours

2. **Memory Usage**:
   - Peak memory: __ GB
   - Average memory: __ GB
   - Memory per feature: __ MB

3. **CPU Usage**:
   - Average CPU: __ %
   - Peak CPU: __ %
   - CPU cores used: __

4. **Disk Usage**:
   - Model files: __ GB
   - Data files: __ GB
   - Log files: __ MB

---

## 📈 Performance Comparison

### Compare with 100k Sample Results

| Metric | 100k Sample | Full Data (Expected) | Full Data (Actual) |
|--------|-------------|---------------------|-------------------|
| **3h ROC-AUC** | 0.91 | 0.91-0.99 | __ |
| **3h MAE** | 4.6°C | 1.2-4.6°C | __ |
| **6h ROC-AUC** | 0.88 | 0.88-0.99 | __ |
| **6h MAE** | 4.9°C | 1.5-4.9°C | __ |
| **12h ROC-AUC** | 0.87 | 0.87-0.99 | __ |
| **12h MAE** | 5.2°C | 1.8-5.2°C | __ |
| **24h ROC-AUC** | 0.83 | 0.83-0.98 | __ |
| **24h MAE** | 5.5°C | 2.0-5.5°C | __ |

---

## 🎯 Decision Points

### After Standard Evaluation

**Decision 1: Feature Reduction Needed?**
- ✅ No → Proceed with LOSO evaluation using all features
- ⚠️ Yes → Implement feature selection, retrain, then LOSO

**Decision 2: Performance Acceptable?**
- ✅ Yes → Proceed with LOSO evaluation
- ⚠️ No → Analyze issues, adjust model parameters, retrain

**Decision 3: Resources Sufficient?**
- ✅ Yes → Proceed with LOSO evaluation
- ⚠️ No → Reduce features (Top 175 or Top 136)

---

## 📝 Record Keeping Template

### Training Session Log

```
Date: 2025-11-13
Command: python3 scripts/train/train_frost_forecast.py --horizons 3 6 12 24 --model lightgbm --output experiments/full_data_training_all_features --n-jobs 8

Training Time:
- Start: __:__
- End: __:__
- Duration: __ hours __ minutes

Resource Usage:
- Peak Memory: __ GB
- Average Memory: __ GB
- Average CPU: __ %
- CPU Cores Used: 8

Performance:
- 3h ROC-AUC: __
- 3h MAE: __°C
- 6h ROC-AUC: __
- 6h MAE: __°C
- 12h ROC-AUC: __
- 12h MAE: __°C
- 24h ROC-AUC: __
- 24h MAE: __°C

Issues: 
- [ ] None
- [ ] Memory issues
- [ ] Time issues
- [ ] Performance issues
- [ ] Other: __

Decision:
- [ ] No feature reduction needed
- [ ] Feature reduction needed (Top __ features)
- [ ] Other: __
```

---

## 🚀 Quick Start Commands

### Full Training (All Features)

```bash
# 1. Start training (standard evaluation)
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/full_data_training_all_features \
    --n-jobs 8

# 2. Monitor training (in another terminal)
./scripts/monitor_training_detailed.sh

# 3. Check results (after training)
cat experiments/full_data_training_all_features/horizon_3h/metrics.json | jq
```

### LOSO Evaluation (After Standard Evaluation)

```bash
# LOSO evaluation
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --loso \
    --model lightgbm \
    --output experiments/full_data_training_all_features \
    --n-jobs 8

# Monitor LOSO progress
./scripts/monitor_training_detailed.sh
```

---

## 📚 Related Documents

- **[TRAINING_AND_EVALUATION.md](../TRAINING_AND_EVALUATION.md)**: Training and evaluation guide
- **[docs/report/FEATURE_REDUCTION_RECOMMENDATION.md](FEATURE_REDUCTION_RECOMMENDATION.md)**: Feature reduction analysis
- **[docs/report/FEATURE_SELECTION_RECOMMENDATIONS.md](FEATURE_SELECTION_RECOMMENDATIONS.md)**: Feature selection guide
- **[docs/report/LOSO_EVALUATION_GUIDE.md](LOSO_EVALUATION_GUIDE.md)**: LOSO evaluation guide

---

## ✅ Checklist

### Before Training
- [ ] Data available in `data/raw/frost-risk-forecast-challenge/stations/`
- [ ] Sufficient disk space (> 50GB free)
- [ ] Sufficient memory (> 32GB available)
- [ ] Training script tested (100k sample)
- [ ] Monitoring script ready

### During Training
- [ ] Monitor training progress
- [ ] Check resource usage
- [ ] Watch for errors or warnings
- [ ] Note any issues

### After Training
- [ ] Check all metrics files
- [ ] Evaluate performance
- [ ] Record resource usage
- [ ] Compare with 100k sample
- [ ] Decide on feature reduction
- [ ] Plan LOSO evaluation

---

## 🎯 Next Steps After Full Training

1. ✅ **Evaluate Results**: Check performance metrics
2. ✅ **Resource Analysis**: Check memory and time usage
3. ✅ **Decision**: Decide on feature reduction (if needed)
4. ✅ **LOSO Evaluation**: Proceed with spatial generalization validation
5. ✅ **Documentation**: Update training results documentation

---

**Last Updated**: 2025-11-13  
**Status**: Ready to execute  
**Strategy**: Use all 298 features → Evaluate → Decide → Optimize (if needed)

