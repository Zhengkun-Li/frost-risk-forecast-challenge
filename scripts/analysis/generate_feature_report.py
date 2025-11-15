#!/usr/bin/env python3
"""Generate comprehensive feature report.

This script:
1. Analyzes all features
2. Extracts feature importance from models
3. Generates comprehensive feature report with statistics, importance, and recommendations
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

from src.data.loaders import DataLoader
from src.data.cleaners import DataCleaner
from src.data.feature_engineering import FeatureEngineer
from src.utils.path_utils import ensure_dir


def generate_comprehensive_feature_report(
    data_path: Path,
    model_dir: Optional[Path] = None,
    output_dir: Path = None,
    sample_size: Optional[int] = None
) -> None:
    """Generate comprehensive feature report.
    
    Args:
        data_path: Path to data file.
        model_dir: Optional path to model directory for feature importance.
        output_dir: Output directory.
        sample_size: Optional sample size for quick analysis.
    """
    ensure_dir(output_dir)
    
    print("="*60)
    print("Generating Comprehensive Feature Report")
    print("="*60)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    loader = DataLoader()
    df_raw = loader.load(data_path)
    
    if sample_size:
        df_raw = df_raw.head(sample_size)
        print(f"Using sample size: {sample_size}")
    
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean(df_raw)
    
    engineer = FeatureEngineer()
    feature_config = {
        "create_time_features": True,
        "create_lag_features": True,
        "create_rolling_features": True,
        "lag_periods": [1, 3, 6, 12, 24],
        "rolling_windows": [3, 6, 12, 24],
    }
    df_features = engineer.engineer_features(df_cleaned, feature_config)
    
    print(f"Total features: {len(df_features.columns)}")
    print(f"Total samples: {len(df_features)}")
    
    # Feature statistics
    print("\n2. Computing feature statistics...")
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["Stn Id"]]
    
    stats = []
    for col in numeric_cols:
        col_data = df_features[col]
        stats.append({
            "feature": col,
            "mean": col_data.mean(),
            "std": col_data.std(),
            "min": col_data.min(),
            "max": col_data.max(),
            "missing_rate": col_data.isna().mean(),
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / "feature_statistics.csv", index=False)
    print(f"✅ Saved feature statistics to {output_dir / 'feature_statistics.csv'}")
    
    # Feature importance (if model provided)
    importance_info = {}
    if model_dir:
        print("\n3. Loading feature importance from models...")
        model_dir = Path(model_dir)
        
        for horizon in [3, 6, 12, 24]:
            horizon_dir = model_dir / "full_training" / f"horizon_{horizon}h"
            if not horizon_dir.exists():
                continue
            
            frost_importance_path = horizon_dir / "frost_classifier" / "feature_importance.csv"
            if frost_importance_path.exists():
                importance_info[f"{horizon}h_frost"] = pd.read_csv(frost_importance_path)
                print(f"  ✅ Loaded {horizon}h frost importance")
    
    # Generate comprehensive report
    print("\n4. Generating comprehensive report...")
    
    report_lines = [
        "# Comprehensive Feature Report",
        "",
        f"Generated: {pd.Timestamp.now()}",
        "",
        "## Dataset Overview",
        f"- Total features: {len(df_features.columns)}",
        f"- Total samples: {len(df_features)}",
        f"- Numeric features: {len(numeric_cols)}",
        "",
        "## Feature Statistics",
        "",
        "### Features with High Missing Rates (>10%)",
    ]
    
    high_missing = stats_df[stats_df["missing_rate"] > 0.1].sort_values("missing_rate", ascending=False)
    if len(high_missing) > 0:
        report_lines.append("")
        report_lines.append("| Feature | Missing Rate | Mean | Std |")
        report_lines.append("|---------|--------------|------|-----|")
        for _, row in high_missing.head(20).iterrows():
            report_lines.append(f"| {row['feature']} | {row['missing_rate']:.2%} | {row['mean']:.4f} | {row['std']:.4f} |")
    else:
        report_lines.append("None")
    
    # Add feature importance section if available
    if importance_info:
        report_lines.extend([
            "",
            "## Feature Importance",
            "",
        ])
        
        for key, importance_df in importance_info.items():
            report_lines.extend([
                f"### {key}",
                "",
                "| Rank | Feature | Importance |",
                "|------|---------|------------|",
            ])
            
            top_20 = importance_df.head(20)
            for idx, (_, row) in enumerate(top_20.iterrows(), 1):
                importance_val = row.get("importance", row.get("mean_importance", 0))
                report_lines.append(f"| {idx} | {row['feature']} | {importance_val:.4f} |")
            report_lines.append("")
    
    # Recommendations
    report_lines.extend([
        "",
        "## Recommendations",
        "",
        "1. **Feature Selection**: Use top N features based on importance",
        "2. **Missing Values**: Consider imputation for features with high missing rates",
        "3. **Feature Engineering**: Continue exploring derived features",
        "",
    ])
    
    report_content = "\n".join(report_lines)
    report_path = output_dir / "comprehensive_feature_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(f"✅ Saved comprehensive report to {report_path}")
    
    print(f"\n✅ Feature report generation complete!")
    print(f"Results saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive feature report"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to data file (default: auto-detect)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to model directory for feature importance (optional)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: scripts/analysis/output/)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for quick analysis (default: use all data)"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "scripts" / "analysis" / "output"
    ensure_dir(output_dir)
    
    # Find data path
    if args.data:
        data_path = Path(args.data)
    else:
        raw_dir = project_root / "data" / "raw" / "frost-risk-forecast-challenge"
        stations_dir = raw_dir / "stations"
        if stations_dir.exists() and stations_dir.is_dir():
            data_path = stations_dir
        else:
            data_path = raw_dir / "cimis_all_stations.csv.gz"
            if not data_path.exists():
                data_path = raw_dir / "cimis_all_stations.csv"
    
    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        return 1
    
    generate_comprehensive_feature_report(
        data_path,
        args.model_dir,
        output_dir,
        args.sample_size
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

