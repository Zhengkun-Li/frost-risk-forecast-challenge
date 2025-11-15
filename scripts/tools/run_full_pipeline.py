#!/usr/bin/env python3
"""Run full pipeline on real CIMIS data.

This script demonstrates the complete workflow:
1. Load raw CIMIS data
2. Clean data (QC filtering, sentinel values, imputation)
3. Feature engineering
4. Train model
5. Evaluate model
6. Generate visualizations

Note: This script should be run within the project's virtual environment.
To ensure virtual environment is activated, use: ./scripts/run_with_venv.sh scripts/run_full_pipeline.py [args]
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
import os

# Check if running in virtual environment
venv_path = Path(__file__).parent.parent / ".venv"
if venv_path.exists():
    venv_python = venv_path / "bin" / "python3"
    if venv_python.exists() and sys.executable != str(venv_python):
        print("⚠️  Warning: Not running in virtual environment!")
        print(f"   Current Python: {sys.executable}")
        print(f"   Expected Python: {venv_python}")
        print("   Consider using: ./scripts/run_with_venv.sh scripts/run_full_pipeline.py [args]")
        print()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yaml

from src.data.loaders import DataLoader
from src.data.cleaners import DataCleaner
from src.data.feature_engineering import FeatureEngineer
from src.models.ml.lightgbm_model import LightGBMModel
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.validators import CrossValidator
from src.visualization.plots import Plotter
from src.utils.path_utils import ensure_dir


def main():
    """Run full pipeline."""
    parser = argparse.ArgumentParser(description="Run full pipeline on real CIMIS data")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to raw CIMIS data (default: search in data/raw/)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: experiments/full_pipeline_YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "xgboost"],
        help="Model type to use"
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip data cleaning (use pre-cleaned data)"
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering (use pre-engineered features)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Use only a sample of data (for quick testing)"
    )
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Perform LOSO (Leave-One-Station-Out) evaluation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "experiments" / f"full_pipeline_{timestamp}"
    
    ensure_dir(output_dir)
    print(f"Output directory: {output_dir}")
    
    # Step 1: Load data
    print("\n" + "="*60)
    print("Step 1: Loading Data")
    print("="*60)
    
    if args.data:
        data_path = Path(args.data)
    else:
        # Search for CIMIS data - prefer stations directory
        raw_dir = project_root / "data" / "raw" / "frost-risk-forecast-challenge"
        stations_dir = raw_dir / "stations"
        
        # Try stations directory first
        if stations_dir.exists() and stations_dir.is_dir():
            data_path = stations_dir
        else:
            # Fall back to merged file
            data_path = raw_dir / "cimis_all_stations.csv.gz"
            if not data_path.exists():
                data_path = raw_dir / "cimis_all_stations.csv"
    
    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        print("Please provide data path with --data option")
        print("Expected: stations directory or cimis_all_stations.csv.gz")
        return 1
    
    print(f"Loading data from {data_path}...")
    df = DataLoader.load_raw_data(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Sample data if requested
    if args.sample_size:
        print(f"Sampling {args.sample_size} rows...")
        df = df.sample(n=min(args.sample_size, len(df)), random_state=42)
        print(f"Using {len(df)} rows")
    
    # Step 2: Clean data
    if not args.skip_cleaning:
        print("\n" + "="*60)
        print("Step 2: Cleaning Data")
        print("="*60)
        
        cleaner = DataCleaner()
        df_cleaned = cleaner.clean_pipeline(df)
        
        print(f"After cleaning: {len(df_cleaned)} rows")
        print(f"Missing values: {df_cleaned.isna().sum().sum()} total")
        
        # Save cleaned data
        cleaned_path = output_dir / "cleaned_data.parquet"
        df_cleaned.to_parquet(cleaned_path)
        print(f"Saved cleaned data to {cleaned_path}")
    else:
        print("\nSkipping data cleaning...")
        df_cleaned = df
    
    # Step 3: Feature engineering
    if not args.skip_features:
        print("\n" + "="*60)
        print("Step 3: Feature Engineering")
        print("="*60)
        
        engineer = FeatureEngineer()
        config = {
            "time_features": True,
            "lag_features": {
                "enabled": True,
                "columns": [
                    "Air Temp (C)", 
                    "Dew Point (C)", 
                    "Rel Hum (%)",
                    "Sol Rad (W/sq.m)",      # Priority 1: 新增
                    "Wind Speed (m/s)",      # Priority 2: 新增
                    "Wind Dir (0-360)",      # Priority 1: 新增
                    "Soil Temp (C)",         # Priority 2: 新增
                    "ETo (mm)",              # Priority 3: 新增
                    "Precip (mm)",           # Priority 3: 新增
                    "Vap Pres (kPa)"         # Priority 3: 新增
                ],
                "lags": [1, 3, 6, 12, 24]
            },
            "rolling_features": {
                "enabled": True,
                "columns": [
                    "Air Temp (C)", 
                    "Dew Point (C)",
                    "Rel Hum (%)",           # Priority 2: 新增
                    "Sol Rad (W/sq.m)",      # Priority 1: 新增
                    "Wind Speed (m/s)",      # Priority 2: 新增
                    "Soil Temp (C)",         # Priority 2: 新增
                    "ETo (mm)",              # Priority 3: 新增
                    "Precip (mm)",           # Priority 3: 新增（使用sum）
                    "Vap Pres (kPa)"         # Priority 3: 新增
                ],
                "windows": [3, 6, 12, 24],
                "functions": ["mean", "min", "max", "std", "sum"]  # 新增sum用于Precip
            },
            "derived_features": True,
            "radiation_features": True,      # Priority 1: 新增
            "wind_features": True,            # Priority 1: 新增
            "humidity_features": True,        # Priority 2: 新增
            "trend_features": True,            # Priority 2: 新增
            "station_features": True,         # Priority 3: 新增
            "station_metadata_path": "data/external/cimis_station_metadata.csv"
        }
        
        df_features = engineer.build_feature_set(df_cleaned, config)
        print(f"After feature engineering: {len(df_features)} rows, {len(df_features.columns)} columns")
        
        # Save features
        features_path = output_dir / "features.parquet"
        df_features.to_parquet(features_path)
        print(f"Saved features to {features_path}")
    else:
        print("\nSkipping feature engineering...")
        df_features = df_cleaned
    
    # Step 4: Prepare training data
    print("\n" + "="*60)
    print("Step 4: Preparing Training Data")
    print("="*60)
    
    # Ensure Date is datetime
    if "Date" in df_features.columns:
        df_features["Date"] = pd.to_datetime(df_features["Date"])
    
    # Select features
    exclude_cols = {
        "Stn Id", "Stn Name", "CIMIS Region", "Date", "Hour (PST)", "Jul",
        "Air Temp (C)", "qc", "qc.1", "qc.2", "qc.3", "qc.4", "qc.5",
        "qc.6", "qc.7", "qc.8", "qc.9"
    }
    
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df_features[feature_cols].copy()
    y = df_features["Air Temp (C)"].copy()
    
    # Remove missing target
    mask = ~y.isna()
    X = X[mask].copy()
    y = y[mask].copy()
    df_features = df_features.loc[X.index]
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Split data (use df_features which has the same index as X and y)
    train_df, val_df, test_df = CrossValidator.time_split(
        df_features,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # Get indices from split DataFrames
    train_idx = train_df.index.intersection(X.index)
    val_idx = val_df.index.intersection(X.index)
    test_idx = test_df.index.intersection(X.index)
    
    X_train = X.loc[train_idx]
    X_val = X.loc[val_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_val = y.loc[val_idx]
    y_test = y.loc[test_idx]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Step 5: Train model
    print("\n" + "="*60)
    print(f"Step 5: Training {args.model.upper()} Model")
    print("="*60)
    
    if args.model == "lightgbm":
        from src.models.ml.lightgbm_model import LightGBMModel
        model_class = LightGBMModel
        model_config = {
            "model_name": "lightgbm",
            "model_type": "lightgbm",
            "task_type": "regression",
            "model_params": {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31,
                "random_state": 42,
                "verbose": -1
            }
        }
    elif args.model == "xgboost":
        from src.models.ml.xgboost_model import XGBoostModel
        model_class = XGBoostModel
        model_config = {
            "model_name": "xgboost",
            "model_type": "xgboost",
            "task_type": "regression",
            "model_params": {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "random_state": 42,
                "verbosity": 0
            }
        }
    
    model = model_class(model_config)
    model.fit(X_train, y_train, eval_set=[(X_val.values, y_val.values)])
    
    # Save model
    model_dir = output_dir / "model"
    model.save(model_dir)
    print(f"Model saved to {model_dir}")
    
    # Step 6: Evaluate
    print("\n" + "="*60)
    print("Step 6: Evaluating Model")
    print("="*60)
    
    all_results = {}
    all_predictions = {}
    
    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test)
    ]:
        y_pred = model.predict(X_split)
        metrics = MetricsCalculator.calculate_all_metrics(
            y_split.values, y_pred, task_type="regression"
        )
        
        all_results[split_name] = metrics
        all_predictions[split_name] = {
            "y_true": y_split.values.tolist(),
            "y_pred": y_pred.tolist()
        }
        
        print(f"\n{split_name.upper()} Metrics:")
        print(MetricsCalculator.format_metrics(metrics))
        
        # Save metrics
        import json
        metrics_path = output_dir / f"{split_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
    
    # Save all predictions
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(all_predictions, f, indent=2, default=str)
    
    # Save summary
    summary = {
        "model_name": model_config.get("model_name", "unknown"),
        "model_type": model_config.get("model_type", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "test_metrics": all_results.get("test", {})
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Step 7: Visualizations
    print("\n" + "="*60)
    print("Step 7: Generating Visualizations")
    print("="*60)
    
    plots_dir = output_dir / "plots"
    ensure_dir(plots_dir)
    
    plotter = Plotter(style="matplotlib", figsize=(14, 8))
    
    # Predictions plot (on test set)
    y_test_pred = model.predict(X_test)
    test_dates = df_features.loc[X_test.index, "Date"] if "Date" in df_features.columns else None
    
    plotter.plot_predictions(
        y_test.values,
        y_test_pred,
        dates=test_dates,
        title="Test Set Predictions",
        save_path=plots_dir / "predictions.png",
        show=False
    )
    print(f"Saved predictions plot to {plots_dir / 'predictions.png'}")
    
    # Feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        plotter.plot_feature_importance(
            importance,
            top_n=20,
            title="Top 20 Feature Importance",
            save_path=plots_dir / "feature_importance.png",
            show=False
        )
        print(f"Saved feature importance plot to {plots_dir / 'feature_importance.png'}")
        
        # Save importance CSV
        importance_path = output_dir / "feature_importance.csv"
        importance.to_csv(importance_path, index=False)
        print(f"Saved feature importance to {importance_path}")
    
    # Step 8: LOSO Evaluation (optional)
    if args.loso:
        print("\n" + "="*60)
        print("Step 8: LOSO (Leave-One-Station-Out) Evaluation")
        print("="*60)
        
        loso_splits = CrossValidator.leave_one_station_out(df_features)
        loso_results = []
        
        print(f"Evaluating on {len(loso_splits)} stations...")
        
        for i, (train_df, test_df) in enumerate(loso_splits, 1):
            test_station = test_df["Stn Id"].iloc[0]
            print(f"\n[{i}/{len(loso_splits)}] Testing on station {test_station}...")
            
            train_idx = train_df.index.intersection(X.index)
            test_idx = test_df.index.intersection(X.index)
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                print(f"  Skipping (insufficient data)")
                continue
            
            X_train_loso = X.loc[train_idx]
            y_train_loso = y.loc[train_idx]
            X_test_loso = X.loc[test_idx]
            y_test_loso = y.loc[test_idx]
            
            # Train model
            model_loso = model_class(model_config)
            try:
                model_loso.fit(X_train_loso, y_train_loso)
                y_pred_loso = model_loso.predict(X_test_loso)
                metrics_loso = MetricsCalculator.calculate_all_metrics(
                    y_test_loso.values, y_pred_loso, task_type="regression"
                )
                
                metrics_loso["station_id"] = int(test_station)
                loso_results.append(metrics_loso)
                
                print(f"  MAE: {metrics_loso['mae']:.4f}, RMSE: {metrics_loso['rmse']:.4f}, R²: {metrics_loso['r2']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if loso_results:
            loso_df = pd.DataFrame(loso_results)
            loso_dir = output_dir / "loso"
            ensure_dir(loso_dir)
            
            loso_df.to_csv(loso_dir / "station_metrics.csv", index=False)
            
            loso_summary = {
                "mean_mae": float(loso_df["mae"].mean()),
                "std_mae": float(loso_df["mae"].std()),
                "mean_rmse": float(loso_df["rmse"].mean()),
                "std_rmse": float(loso_df["rmse"].std()),
                "mean_r2": float(loso_df["r2"].mean()),
                "std_r2": float(loso_df["r2"].std()),
                "n_stations": len(loso_df)
            }
            
            with open(loso_dir / "summary.json", "w") as f:
                json.dump(loso_summary, f, indent=2)
            
            print(f"\nLOSO Summary:")
            print(f"  Mean MAE: {loso_summary['mean_mae']:.4f} ± {loso_summary['std_mae']:.4f}")
            print(f"  Mean RMSE: {loso_summary['mean_rmse']:.4f} ± {loso_summary['std_rmse']:.4f}")
            print(f"  Mean R²: {loso_summary['mean_r2']:.4f} ± {loso_summary['std_r2']:.4f}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

