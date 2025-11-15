#!/usr/bin/env python3
"""Analyze feature importance from trained models.

This script:
1. Loads trained models
2. Extracts feature importance
3. Analyzes and visualizes feature importance
4. Generates feature importance report
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional

from src.models.ml.lightgbm_model import LightGBMModel
from src.models.ml.xgboost_model import XGBoostModel
from src.models.ml.catboost_model import CatBoostModel
from src.models.ml.random_forest_model import RandomForestModel
from src.utils.path_utils import ensure_dir

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_feature_importance_from_model(
    model_path: Path,
    model_type: str = "lightgbm"
) -> Optional[pd.DataFrame]:
    """Load feature importance from a trained model.
    
    Args:
        model_path: Path to model directory or file.
        model_type: Model type (lightgbm, xgboost, catboost, random_forest).
    
    Returns:
        DataFrame with feature importance, or None if not available.
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        return None
    
    try:
        # Load model based on type
        if model_type == "lightgbm":
            model = LightGBMModel.load(model_path)
        elif model_type == "xgboost":
            model = XGBoostModel.load(model_path)
        elif model_type == "catboost":
            model = CatBoostModel.load(model_path)
        elif model_type == "random_forest":
            model = RandomForestModel.load(model_path)
        else:
            print(f"⚠️  Unsupported model type: {model_type}")
            return None
        
        # Get feature importance
        importance_df = model.get_feature_importance()
        
        if importance_df is not None and len(importance_df) > 0:
            return importance_df
        else:
            print(f"⚠️  No feature importance available for {model_path}")
            return None
            
    except Exception as e:
        print(f"⚠️  Error loading model {model_path}: {e}")
        return None


def aggregate_feature_importance(
    importance_dfs: List[pd.DataFrame]
) -> pd.DataFrame:
    """Aggregate feature importance from multiple models.
    
    Args:
        importance_dfs: List of feature importance DataFrames.
    
    Returns:
        Aggregated feature importance DataFrame.
    """
    if not importance_dfs:
        return pd.DataFrame()
    
    # Combine all importance DataFrames
    all_importance = pd.concat(importance_dfs, ignore_index=True)
    
    # Group by feature and aggregate
    aggregated = all_importance.groupby("feature")["importance"].agg([
        "mean", "std", "min", "max", "count"
    ]).reset_index()
    
    aggregated = aggregated.sort_values("mean", ascending=False)
    aggregated.columns = ["feature", "mean_importance", "std_importance", "min_importance", "max_importance", "count"]
    
    return aggregated


def generate_importance_report(
    importance_df: pd.DataFrame,
    output_dir: Path,
    model_name: str = "model"
) -> None:
    """Generate feature importance report.
    
    Args:
        importance_df: Feature importance DataFrame.
        output_dir: Output directory.
        model_name: Model name for report.
    """
    ensure_dir(output_dir)
    
    # Save CSV
    csv_path = output_dir / f"feature_importance_{model_name}.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"✅ Saved feature importance to {csv_path}")
    
    # Generate summary report
    report_lines = [
        f"# Feature Importance Report - {model_name}",
        "",
        f"Generated: {pd.Timestamp.now()}",
        "",
        "## Summary",
        f"- Total features: {len(importance_df)}",
        "",
        "## Top 20 Features",
        "",
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ]
    
    top_20 = importance_df.head(20)
    for idx, (_, row) in enumerate(top_20.iterrows(), 1):
        importance_val = row.get("importance", row.get("mean_importance", 0))
        report_lines.append(f"| {idx} | {row['feature']} | {importance_val:.4f} |")
    
    report_content = "\n".join(report_lines)
    report_path = output_dir / f"feature_importance_report_{model_name}.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(f"✅ Saved feature importance report to {report_path}")
    
    # Generate visualization if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        try:
            top_30 = importance_df.head(30)
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_30)), top_30["importance"].values if "importance" in top_30.columns else top_30["mean_importance"].values)
            plt.yticks(range(len(top_30)), top_30["feature"].values)
            plt.xlabel("Importance")
            plt.title(f"Top 30 Feature Importance - {model_name}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            fig_path = output_dir / f"feature_importance_{model_name}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved feature importance plot to {fig_path}")
        except Exception as e:
            print(f"⚠️  Could not generate plot: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze feature importance from trained models"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to model directory (e.g., experiments/lightgbm/top175_features/full_training/horizon_3h)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "xgboost", "catboost", "random_forest"],
        help="Model type (default: lightgbm)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["frost", "temp", "both"],
        help="Which task to analyze (default: both)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: model_dir/feature_importance/)"
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return 1
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = model_dir / "feature_importance"
    ensure_dir(output_dir)
    
    print(f"Analyzing feature importance from: {model_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Task: {args.task}")
    
    importance_dfs = []
    
    # Load frost classifier importance
    if args.task in ["frost", "both"]:
        frost_model_path = model_dir / "frost_classifier"
        if frost_model_path.exists():
            print(f"Loading frost classifier from: {frost_model_path}")
            frost_importance = load_feature_importance_from_model(
                frost_model_path, args.model_type
            )
            if frost_importance is not None:
                frost_importance["task"] = "frost"
                importance_dfs.append(frost_importance)
                generate_importance_report(frost_importance, output_dir, "frost_classifier")
    
    # Load temp regressor importance
    if args.task in ["temp", "both"]:
        temp_model_path = model_dir / "temp_regressor"
        if temp_model_path.exists():
            print(f"Loading temp regressor from: {temp_model_path}")
            temp_importance = load_feature_importance_from_model(
                temp_model_path, args.model_type
            )
            if temp_importance is not None:
                temp_importance["task"] = "temp"
                importance_dfs.append(temp_importance)
                generate_importance_report(temp_importance, output_dir, "temp_regressor")
    
    # Aggregate if both tasks
    if args.task == "both" and len(importance_dfs) > 1:
        print("Aggregating feature importance from both tasks...")
        aggregated = aggregate_feature_importance(importance_dfs)
        if len(aggregated) > 0:
            generate_importance_report(aggregated, output_dir, "aggregated")
    
    print(f"\n✅ Feature importance analysis complete!")
    print(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

