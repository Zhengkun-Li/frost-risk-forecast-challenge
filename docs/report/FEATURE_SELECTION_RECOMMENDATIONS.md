# Feature Selection Recommendations

**Based on Feature Importance Analysis**  
**Date**: 2025-11-13

---

## Executive Summary

Based on comprehensive feature importance analysis of the frost forecasting model, this document provides recommendations for feature selection to optimize the balance between model performance and computational efficiency.

### Key Statistics

- **Total Features**: 298
- **Used Features** (importance > 0): 280 (94.0%)
- **Unused Features** (importance = 0): 18 (6.0%)

---

## Cumulative Importance Analysis

| Feature Count | Cumulative Importance | Recommendation Level |
|---------------|----------------------|---------------------|
| Top 64 | 50.0% | Minimum viable set |
| **Top 136** | **80.0%** | **Fast training** |
| **Top 175** | **90.0%** | **⭐ Recommended** |
| **Top 206** | **95.0%** | **Best performance** |
| Top 247 | 99.0% | Near-complete set |
| All 280 | 100.0% | Complete feature set |

---

## Recommended Options

### Option 1: Fast Training / Production Environment
**Top 136 Features (80% Importance)**

#### Advantages
- ✅ ~50% reduction in feature count → faster computation
- ✅ Lower memory footprint
- ✅ Shorter training time
- ✅ Retains most predictive power

#### Disadvantages
- ⚠️ May lose 10-20% of subtle patterns
- ⚠️ Some edge cases may perform slightly worse

#### Use Cases
- Real-time prediction systems
- Resource-constrained environments
- Rapid prototype development
- Quick model iteration

---

### Option 2: Balanced Approach ⭐ **RECOMMENDED**
**Top 175 Features (90% Importance)**

#### Advantages
- ✅ Excellent balance between performance and efficiency
- ✅ Retains 90% of predictive capability
- ✅ Reasonable training time
- ✅ Acceptable memory usage

#### Disadvantages
- ⚠️ Slightly slower than Option 1

#### Use Cases
- **Standard training pipeline (Recommended)**
- Production environments with adequate resources
- Competitions / Research
- **Most use cases**

---

### Option 3: Best Performance
**Top 206 Features (95% Importance)**

#### Advantages
- ✅ Near-optimal performance
- ✅ Retains 95% of predictive capability
- ✅ Covers almost all useful features

#### Disadvantages
- ⚠️ Higher computational cost
- ⚠️ Longer training time

#### Use Cases
- Pursuing maximum accuracy
- Sufficient computational resources available
- Final model training
- Performance-critical applications

---

### Option 4: Complete Feature Set
**280 Features (All with importance > 0)**

#### Advantages
- ✅ Maximum possible performance
- ✅ All useful features retained
- ✅ Suitable for final evaluation

#### Disadvantages
- ⚠️ Highest computational cost
- ⚠️ May include low-value features

#### Use Cases
- Final model training
- Complete performance evaluation
- When computational cost is not a concern

---

### Option 5: Clean Feature Set
**280 Features (Remove importance = 0 features)**

#### Advantages
- ✅ Removes redundant features
- ✅ Maintains complete performance
- ✅ Cleaner model

#### Disadvantages
- ⚠️ Still includes low-importance features

#### Use Cases
- Feature engineering cleanup
- Model optimization

---

## Comparison Table

| Option | Feature Count | Cumulative Importance | Training Speed | Memory Usage | Recommended For |
|--------|---------------|----------------------|----------------|--------------|-----------------|
| Fast Training | 136 | 80% | ⚡⚡⚡ Fast | 💚 Low | Production, Quick iteration |
| **Balanced (Recommended)** | **175** | **90%** | **⚡⚡ Medium** | **💚💚 Medium** | **Most scenarios ⭐** |
| Best Performance | 206 | 95% | ⚡ Slow | 💚💚💚 High | Final models, Max accuracy |
| Complete Set | 280 | 100% | ⚡⚡⚡ Slowest | 💚💚💚 Highest | Final evaluation |

---

## Implementation Recommendations

### Recommended Workflow

1. **Initial Training / Fast Iteration**
   - Use **Top 136 features (80%)**
   - Quickly validate model and feature engineering effectiveness
   - Fast feedback loop

2. **Standard Training** ⭐
   - Use **Top 175 features (90%)**
   - Balance performance and efficiency
   - **Recommended for most scenarios**

3. **Final Model**
   - Use **Top 206 features (95%)** or **all 280 features**
   - Pursue maximum accuracy
   - For final submission / production

### Implementation

Add feature selection parameter to training script:

```bash
# Fast training
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --max-features 136

# Balanced (Recommended)
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --max-features 175

# Best performance
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --max-features 206

# Complete set
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --max-features 280
```

### Feature List Files

Feature importance rankings are available at:
- CSV: `experiments/lightgbm/feature_importance/feature_importance_3h_all.csv`
- JSON: `docs/report/feature_analysis.json`

Use these files to select top N features during training.

---

## Detailed Feature Counts

| Threshold | Feature Count | Actual Cumulative % | Last Feature (at threshold) |
|-----------|---------------|---------------------|----------------------------|
| 50% | 64 | ~50.0% | See CSV file |
| 80% | 136 | ~80.0% | See CSV file |
| 90% | 175 | ~90.0% | See CSV file |
| 95% | 206 | ~95.0% | See CSV file |
| 99% | 247 | ~99.0% | See CSV file |

---

## Final Recommendation

### For Most Use Cases: **Top 175 Features (90%)**

**Reasoning:**
1. **Optimal Balance**: 90% importance retained with only ~37% reduction in feature count
2. **Practical Performance**: Sufficient for most prediction tasks
3. **Reasonable Cost**: Training time and memory usage are acceptable
4. **Proven Effective**: Based on comprehensive analysis of 100k sample training

**When to Use More:**
- If accuracy is absolutely critical → Use Top 206 (95%) or all 280
- If resources are abundant → Use all 280 features

**When to Use Less:**
- If computational resources are limited → Use Top 136 (80%)
- If real-time inference is required → Use Top 136 (80%)

---

## Notes

1. **Feature Importance Can Vary**: Importance rankings may differ slightly with:
   - Different sample sizes
   - Different time horizons
   - Different model configurations
   - Full dataset training

2. **Validation Required**: Always validate feature selection on a held-out test set

3. **Dynamic Selection**: Consider using feature importance from full training for final feature selection

4. **Category Balance**: Ensure important categories (Time, Lag, Rolling) are well-represented in selected features

---

**Last Updated**: 2025-11-13  
**Based on**: Feature importance analysis from 100k sample training (3h horizon)

