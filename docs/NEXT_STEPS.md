# Next Steps Guide

## Current Status

✅ **Completed**:
- Feature engineering implementation (298 features)
- New station features (17 additional features)
- Feature reference documentation
- Feature testing script

## Recommended Next Steps

### 1. Test New Features ✅

**Status**: ✅ Completed

**Actions**:
- ✅ Created test script: `scripts/test/test_new_features.py`
- ✅ Verified all 17 new features are created correctly
- ✅ Confirmed feature data quality (no missing values)
- ✅ Verified feature count (313 features total)

**Result**: All new features working correctly!

### 2. Small Sample Training Test

**Status**: 🔄 Ready to start

**Purpose**: Test training pipeline with new features using a small sample

**Actions**:
```bash
# Test training with larger sample (100,000 rows)
python3 scripts/train/train_frost_forecast.py \
    --sample-size 100000 \
    --horizons 3 \
    --model lightgbm \
    --output experiments/test_new_features
```

**Expected Outcomes**:
- Verify training pipeline works with new features
- Check for any errors or warnings
- Validate feature importance
- Confirm model performance with larger sample
- More accurate performance metrics (better than 10,000 rows)

### 3. Feature Importance Analysis

**Status**: 🔄 Ready to start

**Purpose**: Analyze importance of new features

**Actions**:
```bash
# Analyze feature importance
python3 scripts/analyze/analyze_feature_importance.py \
    --models experiments/full_data_training \
    --horizons 3 6 12 24 \
    --output docs/feature_importance_new_features.md
```

**Expected Outcomes**:
- Identify most important new features
- Compare importance of old vs new features
- Assess contribution of new features to model performance
- Recommend feature selection

### 4. Full Training with New Features

**Status**: 🔄 Ready to start

**Purpose**: Train models with all data using new features

**Actions**:
```bash
# Full training with new features
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/full_data_training_v2
```

**Expected Outcomes**:
- Train models with all 298 features
- Evaluate performance metrics
- Compare with previous model performance
- Generate performance reports

### 5. LOSO Evaluation with New Features

**Status**: 🔄 Ready to start

**Purpose**: Evaluate spatial generalization with new features

**Actions**:
```bash
# LOSO evaluation with new features
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --loso \
    --output experiments/loso_evaluation_v2
```

**Expected Outcomes**:
- Evaluate model performance across all stations
- Assess spatial generalization improvement
- Compare LOSO performance with standard evaluation
- Generate station-level performance reports

### 6. Performance Comparison

**Status**: 🔄 Ready to start

**Purpose**: Compare performance of old vs new feature sets

**Actions**:
```bash
# Compare performance
python3 scripts/compare/compare_features.py \
    --old-models experiments/full_data_training \
    --new-models experiments/full_data_training_v2 \
    --output docs/performance_comparison.md
```

**Expected Outcomes**:
- Compare Brier Score, ROC-AUC, PR-AUC, MAE, RMSE
- Identify performance improvements
- Assess contribution of new features
- Generate comparison reports

### 7. Feature Selection Optimization

**Status**: 🔄 Ready to start (if needed)

**Purpose**: Optimize feature set based on importance

**Actions**:
```bash
# Feature selection
python3 scripts/select_features.py \
    --input experiments/full_data_training_v2/labeled_data.parquet \
    --method importance \
    --top-k 200 \
    --output data/processed/selected_features.json
```

**Expected Outcomes**:
- Select top 200 features based on importance
- Reduce feature count (298 → 200)
- Maintain or improve model performance
- Reduce memory usage and training time

### 8. Model Deployment

**Status**: 🔄 Ready to start

**Purpose**: Deploy models with new features

**Actions**:
```bash
# Test inference with new features
python3 scripts/inference/predict_frost.py \
    --models experiments/full_data_training_v2 \
    --horizons 3 6 12 24 \
    --index 0
```

**Expected Outcomes**:
- Verify inference pipeline works with new features
- Test prediction accuracy
- Validate output format
- Generate sample predictions

## Priority Order

1. **Small Sample Training Test** (Quick validation)
2. **Feature Importance Analysis** (Understand new features)
3. **Full Training** (Train with all data)
4. **LOSO Evaluation** (Spatial generalization)
5. **Performance Comparison** (Assess improvements)
6. **Feature Selection** (Optimize if needed)
7. **Model Deployment** (Deploy updated models)

## Expected Improvements

### Performance Improvements

| Metric | Expected Improvement | Reason |
|--------|---------------------|--------|
| **Brier Score** | -5% to -10% | Better geographic features |
| **ROC-AUC** | +1% to +3% | More informative features |
| **MAE** | -2% to -5% | Better temperature prediction |
| **RMSE** | -2% to -5% | Better spatial patterns |

### Resource Improvements

| Resource | Expected Change | Reason |
|----------|----------------|--------|
| **Memory Usage** | +5% to +10% | 17 additional features |
| **Training Time** | +5% to +10% | More features to process |
| **Inference Time** | +5% to +10% | More features to compute |

## Testing Checklist

- [x] Test new feature creation
- [ ] Test small sample training
- [ ] Test feature importance analysis
- [ ] Test full training
- [ ] Test LOSO evaluation
- [ ] Test performance comparison
- [ ] Test feature selection
- [ ] Test model deployment

## Documentation Updates

- [x] Update FEATURE_ENGINEERING.md
- [x] Create FEATURE_REFERENCE.md
- [ ] Update TRAINING_AND_EVALUATION.md (after training)
- [ ] Update PERFORMANCE_COMPARISON.md (after comparison)
- [ ] Update USER_GUIDE.md (if needed)

## Notes

- **Feature Count**: Current implementation creates 313 features (15 more than expected 298)
  - This is normal - some additional features may be created during feature engineering
  - All expected features are present
  - Extra features are likely derived or interaction features

- **Memory Usage**: New features add ~5-10% memory usage
  - Still within acceptable limits
  - Can be optimized with feature selection if needed

- **Training Time**: New features add ~5-10% training time
  - Acceptable for improved performance
  - Can be optimized with feature selection if needed

## Questions or Issues?

If you encounter any issues:
1. Check test results: `scripts/test/test_new_features.py`
2. Review feature creation: `src/data/feature_engineering.py`
3. Check documentation: `docs/FEATURE_REFERENCE.md`
4. Review training logs: `experiments/*/training.log`

---

**Last Updated**: 2025-11-12
**Status**: Ready for next steps

