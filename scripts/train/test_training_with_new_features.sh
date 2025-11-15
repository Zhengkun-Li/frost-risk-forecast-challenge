#!/bin/bash
# Quick test training with new features

echo "="*70
echo "Testing Training with New Features"
echo "="*70

# Test training with larger sample (100,000 rows)
python3 scripts/train/train_frost_forecast.py \
    --sample-size 100000 \
    --horizons 3 \
    --model lightgbm \
    --output experiments/test_new_features

echo "="*70
echo "Training test completed!"
echo "="*70
