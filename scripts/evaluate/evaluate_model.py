#!/usr/bin/env python3
"""Evaluate a trained model on test data."""

import sys
import argparse
from pathlib import Path
import json
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.data.loaders import DataLoader
from src.models.ml.lightgbm_model import LightGBMModel
from src.evaluation.metrics import MetricsCalculator
from src.utils.path_utils import ensure_dir


def load_model(model_dir: Path, config: dict = None):
    """Load trained model from directory."""
    if config is None:
        config_path = model_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    model_type = config.get("model_type", "lightgbm")
    
    if model_type == "lightgbm":
        model = LightGBMModel.load(model_dir)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, config


def evaluate_on_data(model, X: pd.DataFrame, y: pd.Series, config: dict, split_name: str = "test"):
    """Evaluate model on given data."""
    print(f"\nEvaluating on {split_name} set ({len(X)} samples)...")
    
    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    task_type = config.get("task_type", "regression")
    metrics = MetricsCalculator.calculate_all_metrics(
        y.values, y_pred, y_proba, task_type=task_type
    )
    
    # Format and print
    print(f"\n{split_name.upper()} Metrics:")
    print(MetricsCalculator.format_metrics(metrics))
    
    return metrics, y_pred, y_proba


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Path to model directory (contains model.pkl and config.yaml)"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to test data (default: use from config)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for evaluation results"
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load model and config
    print(f"Loading model from {model_dir}...")
    model, config = load_model(model_dir)
    print("Model loaded successfully!")
    
    # Load data
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = project_root / config["data"]["input_path"]
    
    print(f"Loading data from {data_path}...")
    df = DataLoader.load_processed_data(data_path)
    
    # Prepare features
    exclude_cols = set(config["data"].get("exclude_columns", []))
    exclude_cols.add(config["data"]["target_column"])
    
    if config["data"].get("feature_columns"):
        feature_cols = [col for col in config["data"]["feature_columns"] if col in df.columns]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[config["data"]["target_column"]].copy()
    
    # Remove missing
    mask = ~y.isna()
    X = X[mask].copy()
    y = y[mask].copy()
    
    print(f"Evaluating on {len(X)} samples with {len(feature_cols)} features")
    
    # Evaluate
    metrics, y_pred, y_proba = evaluate_on_data(model, X, y, config, "test")
    
    # Save results
    if args.output:
        output_dir = Path(args.output)
        ensure_dir(output_dir)
        
        # Save metrics
        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save predictions
        pred_df = pd.DataFrame({
            "y_true": y.values,
            "y_pred": y_pred,
        })
        if y_proba is not None:
            pred_df["y_proba"] = y_proba
        
        pred_path = output_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        
        print(f"\nResults saved to {output_dir}")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

