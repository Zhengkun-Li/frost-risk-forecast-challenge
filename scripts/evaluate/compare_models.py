#!/usr/bin/env python3
"""Compare multiple trained models and generate comparison report."""

import sys
import argparse
from pathlib import Path
import json
import yaml
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.data.loaders import DataLoader
from src.evaluation.metrics import MetricsCalculator
from src.utils.path_utils import ensure_dir


def load_model_metrics(model_dir: Path) -> Dict:
    """Load metrics from a model directory."""
    metrics_path = model_dir / "metrics.json"
    config_path = model_dir / "config.yaml"
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    
    return {
        "metrics": metrics,
        "config": config,
        "model_dir": str(model_dir),
        "model_name": config.get("model_name", model_dir.name)
    }


def create_comparison_table(model_results: List[Dict], output_path: Path):
    """Create comparison table from model results."""
    # Extract test metrics
    comparison_data = []
    
    for result in model_results:
        metrics = result["metrics"]
        model_name = result["model_name"]
        
        # Extract test metrics
        test_metrics = {k.replace("test_", ""): v for k, v in metrics.items() if k.startswith("test_")}
        
        row = {"model": model_name}
        row.update(test_metrics)
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    comparison_df.to_csv(output_path, index=False)
    
    print(f"\nComparison table saved to {output_path}")
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def find_best_model(model_results: List[Dict], metric: str = "mae", lower_is_better: bool = True) -> Dict:
    """Find best model based on a metric."""
    best_value = float("inf") if lower_is_better else float("-inf")
    best_model = None
    
    for result in model_results:
        metrics = result["metrics"]
        test_metric_key = f"test_{metric}"
        
        if test_metric_key in metrics:
            value = metrics[test_metric_key]
            
            if lower_is_better:
                if value < best_value:
                    best_value = value
                    best_model = result
            else:
                if value > best_value:
                    best_value = value
                    best_model = result
    
    return best_model


def generate_comparison_report(model_results: List[Dict], output_dir: Path):
    """Generate HTML comparison report."""
    ensure_dir(output_dir)
    
    # Create comparison table
    comparison_df = create_comparison_table(
        model_results,
        output_dir / "comparison_table.csv"
    )
    
    # Find best models for different metrics
    best_models = {}
    for metric, lower_is_better in [("mae", True), ("rmse", True), ("r2", False), ("roc_auc", False)]:
        best = find_best_model(model_results, metric, lower_is_better)
        if best:
            best_models[metric] = {
                "model": best["model_name"],
                "value": best["metrics"].get(f"test_{metric}", "N/A")
            }
    
    # Generate summary
    summary = {
        "total_models": len(model_results),
        "best_models": best_models,
        "comparison_table": comparison_df.to_dict(orient="records")
    }
    
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nComparison summary saved to {summary_path}")
    
    # Print best models
    print("\n" + "="*60)
    print("Best Models by Metric:")
    print("="*60)
    for metric, info in best_models.items():
        print(f"{metric.upper()}: {info['model']} ({info['value']:.4f})")
    print("="*60)
    
    return summary


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare multiple trained models")
    parser.add_argument(
        "model_dirs",
        type=Path,
        nargs="+",
        help="Paths to model directories to compare"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for comparison results"
    )
    
    args = parser.parse_args()
    
    # Load all model results
    model_results = []
    for model_dir in args.model_dirs:
        model_dir = Path(model_dir)
        if not model_dir.exists():
            print(f"Warning: Model directory not found: {model_dir}")
            continue
        
        try:
            result = load_model_metrics(model_dir)
            model_results.append(result)
            print(f"Loaded: {result['model_name']}")
        except Exception as e:
            print(f"Error loading {model_dir}: {e}")
            continue
    
    if len(model_results) < 2:
        raise ValueError("Need at least 2 models to compare")
    
    print(f"\nComparing {len(model_results)} models...")
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "experiments" / "comparisons" / "latest"
    
    ensure_dir(output_dir)
    
    # Generate comparison
    summary = generate_comparison_report(model_results, output_dir)
    
    print(f"\nComparison completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

