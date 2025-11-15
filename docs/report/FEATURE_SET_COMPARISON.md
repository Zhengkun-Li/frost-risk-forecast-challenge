# Feature Set Comparison Report

## 特征集对比报告

**Comparison between:**
- Full feature set: 298 features
- Top 175 features: 90% cumulative importance

---

## Performance Metrics Comparison

### Summary Table

| Horizon | ROC-AUC (Full) | ROC-AUC (Top175) | ROC-AUC (Diff) | MAE (Full) | MAE (Top175) | MAE (Diff) | R² (Full) | R² (Top175) | R² (Diff) |
|---------|----------------|------------------|----------------|------------|--------------|------------|-----------|-------------|-----------|
| 3h | 0.9965 | 0.9965 | +0.0000 | 1.1548 | 1.1438 | -0.0110 | 0.9698 | 0.9703 | +0.0006 |
| 6h | 0.9928 | 0.9926 | -0.0002 | 1.5853 | 1.5454 | -0.0399 | 0.9458 | 0.9481 | +0.0023 |
| 12h | 0.9892 | 0.9892 | +0.0000 | 1.8439 | 1.7925 | -0.0514 | 0.9270 | 0.9304 | +0.0034 |
| 24h | 0.9827 | 0.9843 | +0.0016 | 1.9562 | 1.9287 | -0.0276 | 0.9171 | 0.9196 | +0.0025 |

---

## Detailed Metrics

### 3h Horizon

#### Frost Classification

- **ROC-AUC**: 0.9965 (Full) → 0.9965 (Top175) (+0.0000, +0.00%)
- **Brier Score**: 0.0029 (Full) → 0.0028 (Top175) (-0.0000)
- **F1 Score**: 0.6325 (Full) → 0.6491 (Top175) (+0.0165)

#### Temperature Regression

- **MAE**: 1.1548°C (Full) → 1.1438°C (Top175) (-0.0110°C, -0.95%)
- **RMSE**: 1.5383°C (Full) → 1.5235°C (Top175) (-0.0148°C)
- **R²**: 0.9698 (Full) → 0.9703 (Top175) (+0.0006, +0.06%)

### 6h Horizon

#### Frost Classification

- **ROC-AUC**: 0.9928 (Full) → 0.9926 (Top175) (-0.0002, -0.02%)
- **Brier Score**: 0.0039 (Full) → 0.0040 (Top175) (+0.0000)
- **F1 Score**: 0.5118 (Full) → 0.5117 (Top175) (-0.0001)

#### Temperature Regression

- **MAE**: 1.5853°C (Full) → 1.5454°C (Top175) (-0.0399°C, -2.52%)
- **RMSE**: 2.0600°C (Full) → 2.0158°C (Top175) (-0.0441°C)
- **R²**: 0.9458 (Full) → 0.9481 (Top175) (+0.0023, +0.24%)

### 12h Horizon

#### Frost Classification

- **ROC-AUC**: 0.9892 (Full) → 0.9892 (Top175) (+0.0000, +0.00%)
- **Brier Score**: 0.0045 (Full) → 0.0043 (Top175) (-0.0002)
- **F1 Score**: 0.4362 (Full) → 0.4605 (Top175) (+0.0243)

#### Temperature Regression

- **MAE**: 1.8439°C (Full) → 1.7925°C (Top175) (-0.0514°C, -2.79%)
- **RMSE**: 2.3895°C (Full) → 2.3335°C (Top175) (-0.0560°C)
- **R²**: 0.9270 (Full) → 0.9304 (Top175) (+0.0034, +0.36%)

### 24h Horizon

#### Frost Classification

- **ROC-AUC**: 0.9827 (Full) → 0.9843 (Top175) (+0.0016, +0.16%)
- **Brier Score**: 0.0068 (Full) → 0.0060 (Top175) (-0.0007)
- **F1 Score**: 0.3362 (Full) → 0.3589 (Top175) (+0.0227)

#### Temperature Regression

- **MAE**: 1.9562°C (Full) → 1.9287°C (Top175) (-0.0276°C, -1.41%)
- **RMSE**: 2.5469°C (Full) → 2.5081°C (Top175) (-0.0389°C)
- **R²**: 0.9171 (Full) → 0.9196 (Top175) (+0.0025, +0.27%)

---

## Analysis

### Average Performance Changes

- **ROC-AUC**: +0.0003 (平均变化)
- **MAE**: -0.0325°C (平均变化)
- **R²**: +0.0022 (平均变化)

### Performance Impact

✅ **ROC-AUC**: 性能下降 < 1%，可接受
✅ **MAE**: 误差增加 < 0.1°C，可接受
✅ **R²**: 性能下降 < 1%，可接受

---

## Conclusion

✅ **Recommendation**: Top 175 features provide excellent performance with reduced complexity.
   - Performance impact is minimal (< 1%)
   - Significant reduction in feature count (298 → 175, 41% reduction)
   - Faster training and inference
   - Lower memory usage
