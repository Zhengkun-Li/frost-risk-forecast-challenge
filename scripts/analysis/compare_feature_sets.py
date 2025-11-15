#!/usr/bin/env python3
"""Compare different feature sets.

This script:
1. Loads results from models trained with different feature sets
2. Compares performance metrics
3. Generates comparison report
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
from typing import Dict, List

from src.utils.path_utils import ensure_dir
from typing import Optional

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_model_metrics(model_dir: Path) -> Optional[Dict]:
    """Load metrics from a model directory.
    
    Args:
        model_dir: Path to model directory.
    
    Returns:
        Dictionary with metrics, or None if not found.
    """
    model_dir = Path(model_dir)
    
    frost_metrics_path = model_dir / "frost_metrics.json"
    temp_metrics_path = model_dir / "temp_metrics.json"
    
    metrics = {}
    
    if frost_metrics_path.exists():
        with open(frost_metrics_path, "r") as f:
            metrics["frost"] = json.load(f)
    
    if temp_metrics_path.exists():
        with open(temp_metrics_path, "r") as f:
            metrics["temp"] = json.load(f)
    
    return metrics if metrics else None


def compare_feature_sets(
    model_dirs: List[Path],
    feature_set_names: List[str],
    output_dir: Path
) -> None:
    """Compare different feature sets.
    
    Args:
        model_dirs: List of model directories.
        feature_set_names: List of feature set names.
        output_dir: Output directory.
    """
    ensure_dir(output_dir)
    
    comparison_data = []
    
    for model_dir, name in zip(model_dirs, feature_set_names):
        # Try to find metrics in horizon directories
        for horizon in [3, 6, 12, 24]:
            horizon_dir = model_dir / "full_training" / f"horizon_{horizon}h"
            if not horizon_dir.exists():
                continue
            
            metrics = load_model_metrics(horizon_dir)
            if metrics:
                row = {
                    "feature_set": name,
                    "horizon": f"{horizon}h",
                }
                
                if "frost" in metrics:
                    frost_metrics = metrics["frost"]
                    row.update({
                        "frost_brier": frost_metrics.get("brier_score"),
                        "frost_ece": frost_metrics.get("ece"),
                        "frost_roc_auc": frost_metrics.get("roc_auc"),
                        "frost_pr_auc": frost_metrics.get("pr_auc"),
                    })
                
                if "temp" in metrics:
                    temp_metrics = metrics["temp"]
                    row.update({
                        "temp_mae": temp_metrics.get("mae"),
                        "temp_rmse": temp_metrics.get("rmse"),
                        "temp_r2": temp_metrics.get("r2"),
                    })
                
                comparison_data.append(row)
    
    if not comparison_data:
        print("⚠️  No metrics found in any model directories")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "feature_sets_comparison.csv", index=False)
    print(f"✅ Saved comparison to {output_dir / 'feature_sets_comparison.csv'}")
    
    # Generate summary report
    report_lines = [
        "# Feature Sets Comparison Report",
        "",
        f"Generated: {pd.Timestamp.now()}",
        "",
        "## Summary",
    ]
    
    for horizon in ["3h", "6h", "12h", "24h"]:
        horizon_data = comparison_df[comparison_df["horizon"] == horizon]
        if len(horizon_data) == 0:
            continue
        
        report_lines.extend([
            f"### {horizon} Horizon",
            "",
        ])
        
        if "frost_roc_auc" in horizon_data.columns:
            best_frost = horizon_data.loc[horizon_data["frost_roc_auc"].idxmax()]
            report_lines.append(f"- Best Frost ROC-AUC: {best_frost['feature_set']} ({best_frost['frost_roc_auc']:.4f})")
        
        if "temp_r2" in horizon_data.columns:
            best_temp = horizon_data.loc[horizon_data["temp_r2"].idxmax()]
            report_lines.append(f"- Best Temp R²: {best_temp['feature_set']} ({best_temp['temp_r2']:.4f})")
        
        report_lines.append("")
    
    report_content = "\n".join(report_lines)
    report_path = output_dir / "feature_sets_comparison_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(f"✅ Saved comparison report to {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare different feature sets"
    )
    parser.add_argument(
        "--model-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to model directories (e.g., experiments/lightgbm/top175_features experiments/lightgbm/all_features)"
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        required=True,
        help="Names for each feature set (e.g., top175 all_features)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: scripts/analysis/output/)"
    )
    
    args = parser.parse_args()
    
    if len(args.model_dirs) != len(args.names):
        print("❌ Number of model directories must match number of names")
        return 1
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "scripts" / "analysis" / "output"
    ensure_dir(output_dir)
    
    print(f"Comparing {len(args.model_dirs)} feature sets...")
    for model_dir, name in zip(args.model_dirs, args.names):
        print(f"  - {name}: {model_dir}")
    
    compare_feature_sets(args.model_dirs, args.names, output_dir)
    
    print(f"\n✅ Feature sets comparison complete!")
    print(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

