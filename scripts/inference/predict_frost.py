#!/usr/bin/env python3
"""Frost forecast inference script.

Load trained models and generate formatted predictions:
"There is a 30% chance of frost in the next 3 hours, predicted temperature: 4.50 °C"
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.models.base import BaseModel
from src.models.ml.lightgbm_model import LightGBMModel
from src.models.ml.xgboost_model import XGBoostModel
from src.data.loaders import DataLoader
from scripts.train.train_frost_forecast import (
    load_and_prepare_data,
    prepare_features_and_targets
)


def load_model(model_path: Path, model_type: str = "lightgbm") -> BaseModel:
    """Load a trained model from disk.
    
    Args:
        model_path: Path to model directory containing model.pkl and config.json.
        model_type: Type of model ("lightgbm" or "xgboost").
    
    Returns:
        Loaded model instance.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Load config to determine model type
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        model_type = config.get("model_type", model_type)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load model based on type
    if model_type == "lightgbm":
        model = LightGBMModel.load(model_path)
    elif model_type == "xgboost":
        model = XGBoostModel.load(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


def format_prediction(
    frost_probability: float,
    predicted_temperature: float,
    horizon: int
) -> str:
    """Format prediction according to challenge requirements.
    
    Args:
        frost_probability: Frost probability (0-1).
        predicted_temperature: Predicted temperature in °C.
        horizon: Forecast horizon in hours.
    
    Returns:
        Formatted prediction string.
    """
    # Convert probability to percentage
    prob_percent = frost_probability * 100
    
    # Format: "There is a 30% chance of frost in the next 3 hours, predicted temperature: 4.50 °C"
    formatted = (
        f"There is a {prob_percent:.1f}% chance of frost in the next {horizon} hours, "
        f"predicted temperature: {predicted_temperature:.2f} °C"
    )
    
    return formatted


def predict_single(
    models_dir: Path,
    horizon: int,
    features: pd.DataFrame,
    model_type: str = "lightgbm"
) -> Tuple[float, float, str]:
    """Make predictions for a single sample.
    
    Args:
        models_dir: Directory containing trained models.
        horizon: Forecast horizon in hours (3, 6, 12, or 24).
        features: Feature DataFrame (single row or multiple rows).
        model_type: Type of model ("lightgbm" or "xgboost").
    
    Returns:
        Tuple of (frost_probability, predicted_temperature, formatted_string).
    """
    # Load models
    frost_model_path = models_dir / f"horizon_{horizon}h" / "frost_classifier"
    temp_model_path = models_dir / f"horizon_{horizon}h" / "temp_regressor"
    
    if not frost_model_path.exists() or not temp_model_path.exists():
        raise FileNotFoundError(
            f"Models not found for {horizon}h horizon: {frost_model_path} or {temp_model_path}"
        )
    
    # Load models
    frost_model = load_model(frost_model_path, model_type)
    temp_model = load_model(temp_model_path, model_type)
    
    # Ensure features match model expectations
    # Models expect specific feature columns - we need to align
    if hasattr(frost_model, 'feature_names') and frost_model.feature_names:
        # Reorder features to match model
        missing_features = set(frost_model.feature_names) - set(features.columns)
        if missing_features:
            raise ValueError(
                f"Missing features: {missing_features}. "
                f"Available features: {list(features.columns)}"
            )
        features_aligned = features[frost_model.feature_names]
    else:
        features_aligned = features
    
    # Make predictions
    frost_proba = frost_model.predict_proba(features_aligned)
    temp_pred = temp_model.predict(features_aligned)
    
    # Handle single sample vs batch
    if len(features_aligned) == 1:
        frost_proba = frost_proba[0] if isinstance(frost_proba, np.ndarray) else frost_proba
        temp_pred = temp_pred[0] if isinstance(temp_pred, np.ndarray) else temp_pred
    else:
        # Return arrays for batch predictions
        pass
    
    # Format prediction
    formatted = format_prediction(frost_proba, temp_pred, horizon)
    
    return frost_proba, temp_pred, formatted


def predict_batch(
    models_dir: Path,
    horizons: List[int],
    features: pd.DataFrame,
    model_type: str = "lightgbm",
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Make predictions for a batch of samples.
    
    Args:
        models_dir: Directory containing trained models.
        horizons: List of forecast horizons in hours.
        features: Feature DataFrame (multiple rows).
        model_type: Type of model ("lightgbm" or "xgboost").
        output_path: Optional path to save predictions.
    
    Returns:
        DataFrame with predictions for each horizon.
    """
    results = []
    
    for horizon in horizons:
        print(f"Predicting for {horizon}h horizon...")
        
        try:
            # Load models
            frost_model_path = models_dir / f"horizon_{horizon}h" / "frost_classifier"
            temp_model_path = models_dir / f"horizon_{horizon}h" / "temp_regressor"
            
            if not frost_model_path.exists() or not temp_model_path.exists():
                print(f"⚠️  Models not found for {horizon}h horizon, skipping...")
                continue
            
            frost_model = load_model(frost_model_path, model_type)
            temp_model = load_model(temp_model_path, model_type)
            
            # Align features
            if hasattr(frost_model, 'feature_names') and frost_model.feature_names:
                missing_features = set(frost_model.feature_names) - set(features.columns)
                if missing_features:
                    print(f"⚠️  Missing features for {horizon}h: {missing_features}, skipping...")
                    continue
                features_aligned = features[frost_model.feature_names]
            else:
                features_aligned = features
            
            # Make predictions
            frost_proba = frost_model.predict_proba(features_aligned)
            temp_pred = temp_model.predict(features_aligned)
            
            # Store results
            for i in range(len(features_aligned)):
                result = {
                    "horizon": horizon,
                    "frost_probability": float(frost_proba[i]),
                    "predicted_temperature": float(temp_pred[i]),
                    "formatted_prediction": format_prediction(
                        frost_proba[i], temp_pred[i], horizon
                    )
                }
                results.append(result)
        
        except Exception as e:
            print(f"❌ Error predicting for {horizon}h horizon: {e}")
            continue
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")
    
    return results_df


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Frost forecast inference - generate formatted predictions"
    )
    parser.add_argument(
        "--models",
        type=Path,
        required=True,
        help="Path to directory containing trained models"
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to data file (CSV or Parquet). If not provided, uses labeled_data.parquet from models directory"
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[3, 6, 12, 24],
        help="Forecast horizons in hours (default: 3 6 12 24)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "xgboost"],
        help="Model type (default: lightgbm)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for predictions (CSV). If not provided, prints to stdout"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of samples to predict (for testing)"
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Index of single sample to predict (for testing)"
    )
    
    args = parser.parse_args()
    
    # Check models directory
    if not args.models.exists():
        print(f"❌ Models directory not found: {args.models}")
        return 1
    
    # Load data
    if args.data:
        data_path = args.data
    else:
        # Try to use labeled_data.parquet from models directory
        labeled_data_path = args.models / "labeled_data.parquet"
        if labeled_data_path.exists():
            data_path = labeled_data_path
        else:
            print(f"❌ Data file not found. Please provide --data or ensure labeled_data.parquet exists in {args.models}")
            return 1
    
    print(f"Loading data from {data_path}...")
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} rows")
    
    # Prepare features for first horizon (all horizons use same features)
    horizon = args.horizons[0]
    print(f"Preparing features for {horizon}h horizon...")
    
    try:
        X, _, _ = prepare_features_and_targets(df, horizon)
        print(f"Prepared {len(X)} samples with {len(X.columns)} features")
    except Exception as e:
        print(f"❌ Error preparing features: {e}")
        return 1
    
    # Subset data if requested
    if args.index is not None:
        if args.index >= len(X):
            print(f"❌ Index {args.index} out of range (max: {len(X)-1})")
            return 1
        X = X.iloc[[args.index]]
        print(f"Predicting for single sample at index {args.index}")
    elif args.sample_size:
        X = X.head(args.sample_size)
        print(f"Predicting for {len(X)} samples")
    
    # Make predictions
    if args.index is not None or len(X) == 1:
        # Single prediction
        horizon = args.horizons[0]
        try:
            frost_proba, temp_pred, formatted = predict_single(
                args.models, horizon, X, args.model_type
            )
            print("\n" + "="*60)
            print("PREDICTION")
            print("="*60)
            print(formatted)
            print("="*60)
            print(f"Frost Probability: {frost_proba:.4f}")
            print(f"Predicted Temperature: {temp_pred:.2f} °C")
            
            # If multiple horizons requested, predict for all
            if len(args.horizons) > 1:
                print("\n" + "="*60)
                print("PREDICTIONS FOR ALL HORIZONS")
                print("="*60)
                for h in args.horizons:
                    try:
                        _, _, formatted = predict_single(
                            args.models, h, X, args.model_type
                        )
                        print(formatted)
                    except Exception as e:
                        print(f"❌ Error predicting for {h}h: {e}")
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        # Batch predictions
        results_df = predict_batch(
            args.models, args.horizons, X, args.model_type, args.output
        )
        
        if args.output:
            print(f"✅ Saved {len(results_df)} predictions to {args.output}")
        else:
            # Print sample predictions
            print("\n" + "="*60)
            print("SAMPLE PREDICTIONS (first 5)")
            print("="*60)
            for _, row in results_df.head(5).iterrows():
                print(row["formatted_prediction"])
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

