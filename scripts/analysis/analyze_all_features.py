#!/usr/bin/env python3
"""Analyze all features in the dataset.

This script:
1. Loads the dataset
2. Computes statistics for all features
3. Analyzes distributions, missing values, correlations
4. Generates feature analysis report
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict, List

from src.data.loaders import DataLoader
from src.data.cleaners import DataCleaner
from src.data.feature_engineering import FeatureEngineer
from src.utils.path_utils import ensure_dir

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def analyze_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics for all features.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        DataFrame with statistics for each feature.
    """
    stats = []
    
    for col in df.columns:
        if col in ["Date", "Stn Id"]:
            continue
        
        col_data = df[col]
        
        stat = {
            "feature": col,
            "dtype": str(col_data.dtype),
            "count": col_data.count(),
            "missing": col_data.isna().sum(),
            "missing_rate": col_data.isna().mean(),
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            stat.update({
                "mean": col_data.mean(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "median": col_data.median(),
                "q25": col_data.quantile(0.25),
                "q75": col_data.quantile(0.75),
                "skewness": col_data.skew(),
                "kurtosis": col_data.kurtosis(),
            })
        else:
            stat.update({
                "n_unique": col_data.nunique(),
                "most_frequent": col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None,
            })
        
        stats.append(stat)
    
    return pd.DataFrame(stats)


def analyze_feature_correlations(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Analyze feature correlations.
    
    Args:
        df: Input DataFrame.
        top_n: Number of top correlations to return.
    
    Returns:
        DataFrame with top correlations.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["Stn Id"]]
    
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    
    corr_matrix = df[numeric_cols].corr()
    
    # Get upper triangle (avoid duplicates)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_pairs.append({
                "feature1": corr_matrix.columns[i],
                "feature2": corr_matrix.columns[j],
                "correlation": corr_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs)
    corr_df = corr_df.sort_values("correlation", key=abs, ascending=False)
    
    return corr_df.head(top_n)


def generate_feature_analysis_report(
    df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Generate comprehensive feature analysis report.
    
    Args:
        df: Input DataFrame.
        output_dir: Output directory for reports.
    """
    ensure_dir(output_dir)
    
    print("Computing feature statistics...")
    stats_df = analyze_feature_statistics(df)
    stats_df.to_csv(output_dir / "feature_statistics.csv", index=False)
    print(f"✅ Saved feature statistics to {output_dir / 'feature_statistics.csv'}")
    
    print("Analyzing feature correlations...")
    corr_df = analyze_feature_correlations(df, top_n=50)
    if len(corr_df) > 0:
        corr_df.to_csv(output_dir / "feature_correlations.csv", index=False)
        print(f"✅ Saved feature correlations to {output_dir / 'feature_correlations.csv'}")
    
    # Generate summary report
    report_lines = [
        "# Feature Analysis Report",
        "",
        f"Generated: {pd.Timestamp.now()}",
        "",
        "## Dataset Overview",
        f"- Total features: {len(df.columns)}",
        f"- Total samples: {len(df)}",
        f"- Numeric features: {len(df.select_dtypes(include=[np.number]).columns)}",
        f"- Categorical features: {len(df.select_dtypes(exclude=[np.number]).columns)}",
        "",
        "## Feature Statistics",
        "",
        "### Features with High Missing Rates (>10%)",
    ]
    
    high_missing = stats_df[stats_df["missing_rate"] > 0.1].sort_values("missing_rate", ascending=False)
    if len(high_missing) > 0:
        report_lines.append("")
        report_lines.append("| Feature | Missing Rate | Count | Missing |")
        report_lines.append("|---------|--------------|-------|---------|")
        for _, row in high_missing.iterrows():
            report_lines.append(f"| {row['feature']} | {row['missing_rate']:.2%} | {row['count']} | {row['missing']} |")
    else:
        report_lines.append("None")
    
    report_lines.extend([
        "",
        "### Highly Correlated Features (|correlation| > 0.9)",
    ])
    
    high_corr = corr_df[corr_df["correlation"].abs() > 0.9]
    if len(high_corr) > 0:
        report_lines.append("")
        report_lines.append("| Feature 1 | Feature 2 | Correlation |")
        report_lines.append("|-----------|-----------|-------------|")
        for _, row in high_corr.iterrows():
            report_lines.append(f"| {row['feature1']} | {row['feature2']} | {row['correlation']:.4f} |")
    else:
        report_lines.append("None")
    
    report_content = "\n".join(report_lines)
    report_path = output_dir / "feature_analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(f"✅ Saved feature analysis report to {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze all features in the dataset"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to data file (default: auto-detect)"
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
    
    # Load data
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
    
    print(f"Loading data from: {data_path}")
    loader = DataLoader()
    df_raw = loader.load(data_path)
    
    if args.sample_size:
        df_raw = df_raw.head(args.sample_size)
        print(f"Using sample size: {args.sample_size}")
    
    # Clean data
    print("Cleaning data...")
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean(df_raw)
    
    # Engineer features
    print("Engineering features...")
    engineer = FeatureEngineer()
    feature_config = {
        "create_time_features": True,
        "create_lag_features": True,
        "create_rolling_features": True,
        "lag_periods": [1, 3, 6, 12, 24],
        "rolling_windows": [3, 6, 12, 24],
    }
    df_features = engineer.engineer_features(df_cleaned, feature_config)
    
    print(f"Analyzing {len(df_features.columns)} features...")
    
    # Generate analysis report
    generate_feature_analysis_report(df_features, output_dir)
    
    print(f"\n✅ Feature analysis complete!")
    print(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

