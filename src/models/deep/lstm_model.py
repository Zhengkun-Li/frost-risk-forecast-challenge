"""LSTM model implementation for frost forecasting."""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import os
import pandas as pd
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..base import BaseModel


class TimeSeriesDataset(Dataset):
    """Dataset for time series data with station grouping support."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 24, 
                 station_ids: np.ndarray = None):
        """Initialize dataset.
        
        Args:
            X: Feature array.
            y: Target array.
            sequence_length: Length of input sequences.
            station_ids: Optional array of station IDs. If provided, sequences will not cross station boundaries.
        """
        self.sequence_length = sequence_length
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.station_ids = station_ids
        
        # If station_ids provided, create valid indices that don't cross station boundaries
        if station_ids is not None:
            self.valid_indices = []
            unique_stations = np.unique(station_ids)
            for station_id in unique_stations:
                station_mask = (station_ids == station_id)
                station_indices = np.where(station_mask)[0]
                # Only add indices where we have enough data for a full sequence
                for i in range(len(station_indices) - self.sequence_length + 1):
                    self.valid_indices.append(station_indices[i])
            self.valid_indices = np.array(self.valid_indices)
        else:
            # No station grouping, use all indices (original behavior)
            self.valid_indices = np.arange(len(self.X) - self.sequence_length + 1)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        return (
            self.X[actual_idx:actual_idx + self.sequence_length],
            self.y[actual_idx + self.sequence_length - 1]
        )


class LSTMModel(nn.Module):
    """LSTM neural network model."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """Initialize LSTM model.
        
        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence, features).
        
        Returns:
            Output tensor of shape (batch, 1).
        """
        lstm_out, _ = self.lstm(x)
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output.squeeze()


class LSTMForecastModel(BaseModel):
    """LSTM model for frost forecasting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LSTM model.
        
        Args:
            config: Model configuration dictionary.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        super().__init__(config)
        
        model_params = config.get("model_params", {})
        self.sequence_length = model_params.get("sequence_length", 24)
        self.hidden_size = model_params.get("hidden_size", 64)
        self.num_layers = model_params.get("num_layers", 2)
        self.dropout = model_params.get("dropout", 0.2)
        self.learning_rate = model_params.get("learning_rate", 0.001)
        self.batch_size = model_params.get("batch_size", 32)
        self.epochs = model_params.get("epochs", 50)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Optimization parameters (using PyTorch built-in features)
        self.use_early_stopping = model_params.get("early_stopping", True)
        self.patience = model_params.get("patience", 10)
        self.min_delta = model_params.get("min_delta", 1e-6)
        self.use_lr_scheduler = model_params.get("lr_scheduler", True)
        self.lr_scheduler_patience = model_params.get("lr_scheduler_patience", 5)
        self.lr_scheduler_factor = model_params.get("lr_scheduler_factor", 0.5)
        self.gradient_clip_value = model_params.get("gradient_clip", None)  # None means no clipping
        self.save_best_model = model_params.get("save_best_model", True)
        self.val_frequency = model_params.get("val_frequency", 1)  # Validate every N epochs (default: every epoch)
        self.checkpoint_frequency = model_params.get("checkpoint_frequency", 10)  # Save checkpoint every N epochs (0 = disabled)
        
        # Training utilities will be set up via setup_training_tools or in fit()
        # Keep checkpoint_dir for backward compatibility
        self.checkpoint_dir = None
        
        # Station grouping support
        self.station_column = config.get("station_column", "Stn Id")  # Column name for station ID
        
        # Model will be initialized in fit() when we know input_size
        self.model = None
        self.input_size = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "LSTMForecastModel":
        """Train the LSTM model.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            **kwargs: Additional arguments:
                - checkpoint_dir: Optional directory for saving checkpoints
                - resume_from_checkpoint: Optional epoch number or 'latest' to resume from
                - station_ids: Optional array of station IDs for sequence grouping
        
        Returns:
            Self for method chaining.
        """
        # Validate configuration
        from src.models.utils import ConfigValidator
        is_valid, error_msg = ConfigValidator.validate_model_config("lstm", self.config)
        if not is_valid:
            raise ValueError(f"Invalid LSTM configuration: {error_msg}")
        
        self.feature_names = list(X.columns)
        self.input_size = len(self.feature_names)
        
        # Initialize model
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Prepare data - convert to numpy arrays and free original DataFrame memory
        X_array = X.values.astype(np.float32)
        y_array = y.values.astype(np.float32)
        
        # Free original DataFrame memory immediately
        del X, y
        import gc
        gc.collect()
        
        # Get station IDs if available (to avoid cross-station sequences)
        station_ids = kwargs.get('station_ids', None)
        
        # Check for NaN values and handle them
        if np.isnan(X_array).any():
            print(f"  ⚠️  Warning: Found NaN values in features, filling with 0", flush=True)
            X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(y_array).any():
            print(f"  ⚠️  Warning: Found NaN values in targets, removing rows", flush=True)
            # Remove rows with NaN targets
            valid_mask = ~np.isnan(y_array)
            X_array = X_array[valid_mask]
            y_array = y_array[valid_mask]
            if station_ids is not None:
                station_ids = np.asarray(station_ids)[valid_mask]
            print(f"  Removed {np.sum(~valid_mask)} rows with NaN targets, remaining: {len(y_array)}", flush=True)
            del valid_mask
            gc.collect()
        
        # Check for infinite values
        if np.isinf(X_array).any():
            print(f"  ⚠️  Warning: Found infinite values in features, clipping to finite range", flush=True)
            X_array = np.clip(X_array, -1e6, 1e6)
        
        if np.isinf(y_array).any():
            print(f"  ⚠️  Warning: Found infinite values in targets, clipping to finite range", flush=True)
            y_array = np.clip(y_array, -1e6, 1e6)
        
        # Validate station_ids if provided
        if station_ids is not None:
            station_ids = np.asarray(station_ids)
            if len(station_ids) != len(X_array):
                print(f"  ⚠️  Warning: station_ids length ({len(station_ids)}) doesn't match X length ({len(X_array)}). Ignoring station_ids.", flush=True)
                station_ids = None
        
        # Create datasets with station grouping support
        dataset = TimeSeriesDataset(X_array, y_array, self.sequence_length, station_ids=station_ids)
        
        # Free numpy arrays after dataset creation (dataset will keep its own copies)
        del X_array, y_array
        gc.collect()
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Optimize DataLoader for GPU: use multiple workers and pin memory
        # Efficiency optimization: use more workers for faster data loading
        num_workers = min(8, (os.cpu_count() or 1) // 2)  # Use up to 8 workers (half of CPU cores)
        pin_memory = torch.cuda.is_available()  # Pin memory if GPU available
        prefetch_factor = 2 if num_workers > 0 else None  # Prefetch 2 batches per worker
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,  # Keep workers alive between epochs
            prefetch_factor=prefetch_factor,  # Limit prefetch to reduce memory
            drop_last=True  # Drop last incomplete batch to ensure consistent memory usage
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor,
            drop_last=False  # Keep all validation data
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Mixed precision training for faster training and lower memory usage
        use_amp = self.use_amp and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        if use_amp:
            print("  ✅ Using Mixed Precision Training (AMP) for faster training", flush=True)
        
        # Learning rate scheduler (using PyTorch built-in)
        scheduler = None
        if self.use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                min_lr=1e-6
            )
        
        # Restore from checkpoint if available (after optimizer/scheduler creation)
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        if resume_checkpoint_data:
            if 'model_state_dict' in resume_checkpoint_data:
                self.model.load_state_dict(resume_checkpoint_data['model_state_dict'])
            if 'optimizer_state_dict' in resume_checkpoint_data:
                optimizer.load_state_dict(resume_checkpoint_data['optimizer_state_dict'])
            if 'scheduler_state_dict' in resume_checkpoint_data and scheduler:
                scheduler.load_state_dict(resume_checkpoint_data['scheduler_state_dict'])
            if 'scaler_state_dict' in resume_checkpoint_data and scaler:
                scaler.load_state_dict(resume_checkpoint_data['scaler_state_dict'])
            if 'training_history' in resume_checkpoint_data:
                # Restore training history
                history_data = resume_checkpoint_data['training_history']
                if isinstance(history_data, dict):
                    self.training_history.history = history_data
            best_val_loss = resume_checkpoint_data.get('best_val_loss', float('inf'))
            patience_counter = resume_checkpoint_data.get('patience_counter', 0)
            print(f"  ✅ Resumed training from checkpoint at epoch {start_epoch}", flush=True)
        
        # Setup training utilities if not already set up
        checkpoint_dir = kwargs.get('checkpoint_dir', None)
        resume_from_checkpoint = kwargs.get('resume_from_checkpoint', None)
        
        # Validate training arguments
        from src.models.utils import ConfigValidator
        is_valid, error_msg = ConfigValidator.validate_training_args(
            "lstm", checkpoint_dir=checkpoint_dir
        )
        if not is_valid:
            raise ValueError(f"Invalid training arguments: {error_msg}")
        
        if checkpoint_dir and not self.checkpoint_manager:
            # Setup training tools if checkpoint_dir provided
            self.setup_training_tools(
                checkpoint_dir=checkpoint_dir,
                checkpoint_frequency=self.checkpoint_frequency,
                save_best=self.save_best_model,
                best_metric="val_loss",
                best_mode="min"
            )
            self.checkpoint_dir = Path(checkpoint_dir)  # For backward compatibility
        
        # Handle resume from checkpoint (will be restored after optimizer/scheduler creation)
        resume_checkpoint_data = None
        start_epoch = 0
        if resume_from_checkpoint and self.checkpoint_manager:
            if resume_from_checkpoint == "latest":
                resume_checkpoint_data = self.checkpoint_manager.get_latest_checkpoint()
            elif isinstance(resume_from_checkpoint, int):
                resume_checkpoint_data = self.checkpoint_manager.load_checkpoint(resume_from_checkpoint)
            
            if resume_checkpoint_data:
                start_epoch = resume_checkpoint_data.get('epoch', 0)
                print(f"  ✅ Will resume training from checkpoint at epoch {start_epoch}", flush=True)
            else:
                print(f"  ⚠️  Checkpoint not found, starting from scratch", flush=True)
        
        # Initialize training history if not already set up
        if not self.training_history:
            from src.models.utils import TrainingHistory
            self.training_history = TrainingHistory()
        
        # Initialize progress logger if not already set up
        if not self.progress_logger:
            from src.models.utils import ProgressLogger
            self.progress_logger = ProgressLogger()
        
        # Start training history tracking
        self.training_history.start_training()
        
        # Log training start
        self.progress_logger.log_training_start(
            model_name="LSTM",
            device=str(self.device),
            config={
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "sequence_length": self.sequence_length
            }
        )
        
        # Training loop with progress bar
        try:
            from tqdm import tqdm
            USE_TQDM = True
        except ImportError:
            USE_TQDM = False
            print("  ⚠️  tqdm not available, using simple progress display", flush=True)
        
        import time
        import sys
        epoch_start_time = time.time()
        
        # Print initial training info
        print(f"\n  🚀 Starting LSTM training")
        print(f"     Device: {self.device}")
        print(f"     Input size: {self.input_size}")
        print(f"     Hidden size: {self.hidden_size}")
        print(f"     Batch size: {self.batch_size}")
        print(f"     Epochs: {self.epochs}")
        print(f"     Sequence length: {self.sequence_length}")
        sys.stdout.flush()
        
        self.model.train()
        epoch_pbar = tqdm(range(start_epoch, self.epochs), desc="Training", unit="epoch", disable=not USE_TQDM, initial=start_epoch) if USE_TQDM else range(start_epoch, self.epochs)
        
        for epoch in epoch_pbar:
            epoch_iter_start = time.time()
            
            # Training phase with progress bar
            train_loss = 0.0
            train_batches = 0
            
            if USE_TQDM:
                train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]", 
                                 leave=False, unit="batch")
            else:
                train_pbar = train_loader
            
            for batch_X, batch_y in train_pbar:
                batch_X = batch_X.to(self.device, non_blocking=True)  # Non-blocking transfer
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Mixed precision training
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping (using PyTorch built-in)
                    if self.gradient_clip_value is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping (using PyTorch built-in)
                    if self.gradient_clip_value is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                    
                    optimizer.step()
                
                # Check for NaN loss
                loss_value = loss.item()
                if np.isnan(loss_value) or np.isinf(loss_value):
                    print(f"\n  ⚠️  Warning: Invalid loss value ({loss_value}) at batch {train_batches}", flush=True)
                    print(f"     This may indicate numerical instability. Consider:", flush=True)
                    print(f"     - Reducing learning rate", flush=True)
                    print(f"     - Checking for NaN/inf in input data", flush=True)
                    print(f"     - Using gradient clipping", flush=True)
                    # Skip this batch's loss
                    continue
                
                train_loss += loss_value
                train_batches += 1
                
                # Memory optimization: delete batch data immediately after use
                del batch_X, batch_y, outputs, loss
                
                # Update progress bar
                if USE_TQDM and train_batches % 10 == 0:
                    train_pbar.set_postfix({'loss': f'{loss_value:.6f}'})
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
                torch.cuda.empty_cache()
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
            
            # Validation phase (every N epochs for efficiency, but always on first and last epoch)
            should_validate = (epoch + 1) % self.val_frequency == 0 or epoch == 0 or epoch == self.epochs - 1
            
            if should_validate:
                val_loss = 0.0
                val_batches = 0
                self.model.eval()
                
                if USE_TQDM:
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]", 
                                   leave=False, unit="batch")
                else:
                    val_pbar = val_loader
                
                with torch.no_grad():
                    for batch_X, batch_y in val_pbar:
                        batch_X = batch_X.to(self.device, non_blocking=True)
                        batch_y = batch_y.to(self.device, non_blocking=True)
                        
                        # Use mixed precision for validation too
                        if scaler is not None:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(batch_X)
                                batch_val_loss = criterion(outputs, batch_y).item()
                        else:
                            outputs = self.model(batch_X)
                            batch_val_loss = criterion(outputs, batch_y).item()
                        
                        val_loss += batch_val_loss
                        val_batches += 1
                        
                        # Memory optimization: delete batch data immediately
                        del batch_X, batch_y, outputs
                        
                        # Update progress bar
                        if USE_TQDM and val_batches % 10 == 0:
                            val_pbar.set_postfix({'val_loss': f'{batch_val_loss:.6f}'})
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            else:
                # Skip validation, use previous validation loss for early stopping
                avg_val_loss = best_val_loss if best_val_loss != float('inf') else float('inf')
            self.model.train()
            
            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step(avg_val_loss)
                new_lr = optimizer.param_groups[0]['lr']
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_iter_start
            elapsed_time = time.time() - epoch_start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = self.epochs - (epoch + 1)
            estimated_remaining = avg_epoch_time * remaining_epochs
            
            # Record epoch in training history
            self.training_history.record_epoch(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss if should_validate else None,
                learning_rate=current_lr,
                epoch_time=epoch_time
            )
            
            # Save checkpoint periodically
            if self.checkpoint_manager and self.checkpoint_manager.should_save_checkpoint(epoch + 1):
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch + 1,
                    model_state=self.model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler else None,
                    scaler_state=scaler.state_dict() if scaler else None,
                    metrics={'train_loss': avg_train_loss, 'val_loss': avg_val_loss},
                    training_history=self.training_history
                )
                self.progress_logger.log(f"  💾 Checkpoint saved: epoch {epoch + 1}", flush=True)
            
            # Early stopping logic
            if self.use_early_stopping:
                improved = False
                if avg_val_loss < best_val_loss - self.min_delta:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    improved = True
                    # Save best model state
                    if self.save_best_model:
                        best_model_state = self.model.state_dict().copy()
                        # Also save via checkpoint manager if available
                        if self.checkpoint_manager:
                            self.checkpoint_manager.save_best_checkpoint(
                                epoch=epoch + 1,
                                model_state=best_model_state,
                                metric_value=avg_val_loss
                            )
                        # Log improvement
                        self.progress_logger.log_improvement("val_loss", avg_val_loss, best_val_loss)
                else:
                    patience_counter += 1
                
                # Update progress bar or print
                progress_info = (
                    f"Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}, "
                    f"LR: {current_lr:.6f}, Patience: {patience_counter}/{self.patience}"
                )
                
                if USE_TQDM:
                    epoch_pbar.set_postfix({
                        'train_loss': f'{avg_train_loss:.6f}',
                        'val_loss': f'{avg_val_loss:.6f}',
                        'lr': f'{current_lr:.6f}',
                        'patience': f'{patience_counter}/{self.patience}',
                        'ETA': f'{estimated_remaining/60:.1f}m'
                    })
                    if improved:
                        epoch_pbar.write(f"  ✅ Improved! Val Loss: {avg_val_loss:.6f} (Best: {best_val_loss:.6f})")
                else:
                    # Print every epoch or on improvement
                    if epoch % 5 == 0 or improved or epoch == 0:
                        print(f"  Epoch {epoch+1}/{self.epochs} - {progress_info} - Time: {epoch_time:.1f}s, ETA: {estimated_remaining/60:.1f}m")
                
                # Early stopping
                if patience_counter >= self.patience:
                    self.progress_logger.log_early_stopping(epoch + 1, self.patience)
                    if self.save_best_model and best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
            else:
                # Print progress every epoch
                progress_info = f"Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}, LR: {current_lr:.6f}"
                if USE_TQDM:
                    epoch_pbar.set_postfix({
                        'train_loss': f'{avg_train_loss:.6f}',
                        'val_loss': f'{avg_val_loss:.6f}',
                        'lr': f'{current_lr:.6f}',
                        'ETA': f'{estimated_remaining/60:.1f}m'
                    })
                else:
                    if epoch % 5 == 0 or epoch == 0:
                        print(f"  Epoch {epoch+1}/{self.epochs} - {progress_info} - Time: {epoch_time:.1f}s, ETA: {estimated_remaining/60:.1f}m", flush=True)
        
        # Save training artifacts using unified utilities
        if self.checkpoint_dir and self.training_history and len(self.training_history) > 0:
            self.save_training_artifacts(self.checkpoint_dir)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make point predictions.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Predictions array.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        X_array = X.values.astype(np.float32)
        
        # Create sequences
        predictions = []
        with torch.no_grad():
            for i in range(len(X_array) - self.sequence_length + 1):
                sequence = X_array[i:i + self.sequence_length]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                pred = self.model(sequence_tensor)
                predictions.append(pred.cpu().numpy())
        
        # Pad predictions for initial sequence_length - 1 samples
        if len(predictions) < len(X_array):
            padding = [predictions[0]] * (self.sequence_length - 1)
            predictions = padding + predictions
        
        return np.array(predictions).flatten()
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """LSTM doesn't support probability predictions for regression.
        
        Returns:
            None (LSTM is regression-only in this implementation).
        """
        return None
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """LSTM doesn't provide direct feature importance.
        
        Returns:
            None (LSTM uses different importance metrics).
        """
        return None
    
    def save(self, path: Path) -> None:
        """Save model to disk.
        
        Args:
            path: Directory path to save model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        model_path = path / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        
        metadata = {
            "model_type": "lstm",
            "input_size": self.input_size,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "feature_names": self.feature_names,
            "config": self.config
        }
        
        metadata_path = path / "metadata.pkl"
        import pickle
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, path: Path) -> "LSTMForecastModel":
        """Load model from disk.
        
        Args:
            path: Directory path containing saved model.
        
        Returns:
            Loaded model instance.
        """
        import pickle
        path = Path(path)
        
        model_path = path / "model.pth"
        metadata_path = path / "metadata.pkl"
        
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        instance = cls(metadata["config"])
        instance.input_size = metadata["input_size"]
        instance.sequence_length = metadata["sequence_length"]
        instance.hidden_size = metadata["hidden_size"]
        instance.num_layers = metadata["num_layers"]
        instance.dropout = metadata["dropout"]
        instance.feature_names = metadata.get("feature_names", [])
        
        instance.model = LSTMModel(
            input_size=instance.input_size,
            hidden_size=instance.hidden_size,
            num_layers=instance.num_layers,
            dropout=instance.dropout
        ).to(instance.device)
        
        instance.model.load_state_dict(torch.load(model_path, map_location=instance.device))
        instance.model.eval()
        instance.is_fitted = True
        
        return instance

