#!/usr/bin/env python3
"""Compare individual features.

This script:
1. Analyzes individual feature performance
2. Compares features across different models
3. Generates feature comparison report
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.utils.path_utils import ensure_dir


def compare_features_from_importance(
    importance_files: List[Path],
    feature_names: List[str],
    output_dir: Path
) -> None:
    """Compare features from importance files.
    
    Args:
        importance_files: List of feature importance CSV files.
        feature_names: List of names for each importance file.
        output_dir: Output directory.
    """
    ensure_dir(output_dir)
    
    all_importance = []
    
    for importance_file, name in zip(importance_files, feature_names):
        if not importance_file.exists():
            print(f"⚠️  Importance file not found: {importance_file}")
            continue
        
        df = pd.read_csv(importance_file)
        df["source"] = name
        all_importance.append(df)
    
    if not all_importance:
        print("❌ No importance files found")
        return
    
    combined_df = pd.concat(all_importance, ignore_index=True)
    
    # Pivot to compare features across sources
    if "importance" in combined_df.columns:
        pivot_df = combined_df.pivot_table(
            index="feature",
            columns="source",
            values="importance",
            aggfunc="mean"
        )
    elif "mean_importance" in combined_df.columns:
        pivot_df = combined_df.pivot_table(
            index="feature",
            columns="source",
            values="mean_importance",
            aggfunc="mean"
        )
    else:
        print("❌ Could not find importance column")
        return
    
    pivot_df = pivot_df.sort_values(pivot_df.columns[0], ascending=False)
    pivot_df.to_csv(output_dir / "features_comparison.csv")
    print(f"✅ Saved feature comparison to {output_dir / 'features_comparison.csv'}")
    
    # Generate summary report
    report_lines = [
        "# Feature Comparison Report",
        "",
        f"Generated: {pd.Timestamp.now()}",
        "",
        "## Top 20 Features (Average Importance)",
        "",
    ]
    
    pivot_df["average"] = pivot_df.mean(axis=1)
    top_20 = pivot_df.nlargest(20, "average")
    
    # Create table header
    header = "| Feature | " + " | ".join(pivot_df.columns) + " | Average |"
    separator = "|" + "|".join(["---"] * (len(pivot_df.columns) + 2)) + "|"
    report_lines.extend([header, separator])
    
    for feature, row in top_20.iterrows():
        values = " | ".join([f"{row[col]:.4f}" if pd.notna(row[col]) else "N/A" for col in pivot_df.columns])
        report_lines.append(f"| {feature} | {values} | {row['average']:.4f} |")
    
    report_content = "\n".join(report_lines)
    report_path = output_dir / "features_comparison_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(f"✅ Saved feature comparison report to {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare individual features"
    )
    parser.add_argument(
        "--importance-files",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to feature importance CSV files"
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        required=True,
        help="Names for each importance file (e.g., frost_classifier temp_regressor)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: scripts/analysis/output/)"
    )
    
    args = parser.parse_args()
    
    if len(args.importance_files) != len(args.names):
        print("❌ Number of importance files must match number of names")
        return 1
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "scripts" / "analysis" / "output"
    ensure_dir(output_dir)
    
    print(f"Comparing {len(args.importance_files)} feature importance files...")
    
    compare_features_from_importance(args.importance_files, args.names, output_dir)
    
    print(f"\n✅ Feature comparison complete!")
    print(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

