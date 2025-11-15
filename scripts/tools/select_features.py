#!/usr/bin/env python3
"""Feature selection script.

Select features based on:
1. Feature importance (from trained models)
2. Correlation (remove highly correlated features)
3. Missing rate (remove features with high missing rates)
4. Variance (remove low-variance features)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.data.feature_selection import FeatureSelector, select_features_from_importance
from src.data.loaders import DataLoader


def main():
    """Main feature selection function."""
    parser = argparse.ArgumentParser(
        description="Select features for frost forecasting"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to labeled data (Parquet file)"
    )
    parser.add_argument(
        "--importance",
        type=Path,
        help="Path to feature importance CSV file"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        choices=[3, 6, 12, 24],
        help="Forecast horizon in hours (default: 3)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["importance", "correlation", "missing", "variance", "all"],
        help="Feature selection method (default: all)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Keep top K features (for importance method)"
    )
    parser.add_argument(
        "--min-importance",
        type=float,
        default=0.01,
        help="Minimum importance ratio to keep (default: 0.01)"
    )
    parser.add_argument(
        "--max-correlation",
        type=float,
        default=0.95,
        help="Maximum correlation to keep (default: 0.95)"
    )
    parser.add_argument(
        "--max-missing",
        type=float,
        default=0.5,
        help="Maximum missing rate to keep (default: 0.5)"
    )
    parser.add_argument(
        "--min-variance",
        type=float,
        default=0.0,
        help="Minimum variance to keep (default: 0.0)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for selected features list (JSON)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Sample size for analysis (default: use all data)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} rows")
    
    # Sample if requested
    if args.sample_size and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=42)
        print(f"Sampled to {len(df)} rows")
    
    # Prepare features
    from scripts.train.train_frost_forecast import prepare_features_and_targets
    
    print(f"\nPreparing features for {args.horizon}h horizon...")
    X, y_frost, y_temp = prepare_features_and_targets(df, args.horizon)
    print(f"Features: {len(X.columns)}")
    print(f"Samples: {len(X)}")
    
    # Apply feature selection
    print(f"\nApplying feature selection (method: {args.method})...")
    
    selector = FeatureSelector(
        min_importance=args.min_importance,
        max_correlation=args.max_correlation,
        max_missing_rate=args.max_missing,
        min_variance=args.min_variance
    )
    
    # Load feature importance if provided
    feature_importance = None
    if args.importance:
        if args.importance.exists():
            feature_importance = pd.read_csv(args.importance)
            print(f"Loaded feature importance from {args.importance}")
        else:
            print(f"⚠️  Feature importance file not found: {args.importance}")
    
    # Apply feature selection
    method = args.method
    remove_correlated = method in ["correlation", "all"]
    remove_high_missing = method in ["missing", "all"]
    remove_low_variance = method in ["variance", "all"]
    use_importance = method in ["importance", "all"] and feature_importance is not None
    
    X_selected = selector.select_features(
        X,
        feature_importance=feature_importance,
        top_k=args.top_k,
        min_importance=args.min_importance,
        remove_correlated=remove_correlated,
        remove_high_missing=remove_high_missing,
        remove_low_variance=remove_low_variance
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("Feature Selection Results")
    print(f"{'='*60}")
    print(f"Original features: {len(X.columns)}")
    print(f"Selected features: {len(X_selected.columns)}")
    print(f"Removed features: {len(X.columns) - len(X_selected.columns)}")
    print(f"Reduction: {(1 - len(X_selected.columns) / len(X.columns)) * 100:.1f}%")
    
    # Print removed features by category
    report = selector.get_selection_report()
    print(f"\nRemoved features by category:")
    for category, count in report["removed_features"].items():
        print(f"  {category}: {count}")
    
    # Save selected features
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save selected features list
        selected_features = list(X_selected.columns)
        with open(output_path, "w") as f:
            json.dump({
                "selected_features": selected_features,
                "n_selected": len(selected_features),
                "n_original": len(X.columns),
                "reduction_rate": 1 - len(selected_features) / len(X.columns),
                "removed_features": report["removed_features"],
                "config": {
                    "method": args.method,
                    "top_k": args.top_k,
                    "min_importance": args.min_importance,
                    "max_correlation": args.max_correlation,
                    "max_missing": args.max_missing,
                    "min_variance": args.min_variance
                }
            }, f, indent=2)
        
        print(f"\n✅ Selected features saved to {output_path}")
    
    # Save selection report
    report_path = args.output.parent / f"{args.output.stem}_report.json" if args.output else Path("feature_selection_report.json")
    selector.save_selection_report(report_path)
    print(f"✅ Selection report saved to {report_path}")
    
    # Print top selected features
    print(f"\nTop 20 selected features:")
    if feature_importance is not None:
        top_features = feature_importance[feature_importance['feature'].isin(X_selected.columns)]
        top_features = top_features.sort_values('importance', ascending=False).head(20)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:50s} (importance: {row['importance']:.2f})")
    else:
        for i, feature in enumerate(X_selected.columns[:20], 1):
            print(f"  {i:2d}. {feature}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

