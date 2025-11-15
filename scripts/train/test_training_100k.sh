#!/bin/bash
# Test training with 100,000 sample size

echo "======================================================================"
echo "Testing Training with New Features (100,000 samples)"
echo "======================================================================"

# Test training with 100,000 sample
python3 scripts/train/train_frost_forecast.py \
    --sample-size 100000 \
    --horizons 3 \
    --model lightgbm \
    --output experiments/lightgbm/feature_importance

echo ""
echo "======================================================================"
echo "Training test completed!"
echo "Output: experiments/lightgbm/feature_importance"
echo "======================================================================"
