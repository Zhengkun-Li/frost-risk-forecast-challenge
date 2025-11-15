"""Data preprocessing utilities with proper train/test separation to prevent data leakage."""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class FeaturePreprocessor:
    """Preprocess features with proper train/test separation.
    
    This ensures no data leakage: scalers and imputers are fitted only on training data.
    """
    
    def __init__(
        self,
        scaling_method: Optional[str] = None,
        imputation_strategy: str = "forward_fill",
        handle_outliers: bool = False
    ):
        """Initialize preprocessor.
        
        Args:
            scaling_method: Scaling method ("standard", "minmax", "robust", or None).
            imputation_strategy: Imputation strategy (only used if sklearn available).
            handle_outliers: Whether to handle outliers (not implemented yet).
        """
        self.scaling_method = scaling_method
        self.imputation_strategy = imputation_strategy
        self.handle_outliers = handle_outliers
        
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X_train: pd.DataFrame) -> "FeaturePreprocessor":
        """Fit preprocessor on training data only.
        
        Args:
            X_train: Training feature DataFrame.
        
        Returns:
            Self for method chaining.
        """
        self.feature_names = list(X_train.columns)
        
        # Fit scaler if needed
        if self.scaling_method and SKLEARN_AVAILABLE:
            if self.scaling_method == "standard":
                self.scaler = StandardScaler()
            elif self.scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            elif self.scaling_method == "robust":
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")
            
            # Fit scaler on training data
            self.scaler.fit(X_train.values)
        
        # Fit imputer if needed (for sklearn-based imputation)
        if self.imputation_strategy in ["mean", "median", "most_frequent"] and SKLEARN_AVAILABLE:
            self.imputer = SimpleImputer(strategy=self.imputation_strategy)
            self.imputer.fit(X_train.values)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor.
        
        Args:
            X: Feature DataFrame (train or test).
        
        Returns:
            Transformed DataFrame.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Ensure same columns and order
        X_aligned = X[self.feature_names].copy()
        
        # Apply scaling
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_aligned.values)
            X_aligned = pd.DataFrame(
                X_scaled,
                index=X_aligned.index,
                columns=X_aligned.columns
            )
        
        # Apply imputation (if using sklearn imputer)
        if self.imputer is not None:
            X_imputed = self.imputer.transform(X_aligned.values)
            X_aligned = pd.DataFrame(
                X_imputed,
                index=X_aligned.index,
                columns=X_aligned.columns
            )
        
        return X_aligned
    
    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform training data.
        
        Args:
            X_train: Training feature DataFrame.
        
        Returns:
            Transformed DataFrame.
        """
        return self.fit(X_train).transform(X_train)


def preprocess_with_loso(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaling_method: Optional[str] = None
) -> tuple:
    """Preprocess features for LOSO with no data leakage.
    
    This function ensures that:
    1. Scaler is fitted only on training stations
    2. Scaler is applied to test station
    3. No information from test station leaks into preprocessing
    
    Args:
        train_df: Training DataFrame (all stations except test station).
        test_df: Test DataFrame (single station).
        feature_cols: List of feature column names.
        scaling_method: Scaling method ("standard", "minmax", "robust", or None).
    
    Returns:
        Tuple of (X_train, X_test) as DataFrames.
    """
    # Extract features
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    # Create and fit preprocessor on training data only
    preprocessor = FeaturePreprocessor(scaling_method=scaling_method)
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform test data using fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed

