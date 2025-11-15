"""Multi-task LSTM model for simultaneous temperature and frost probability prediction."""

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
    
    def __init__(self, X: np.ndarray, y_temp: np.ndarray, y_frost: np.ndarray, 
                 sequence_length: int = 24, station_ids: np.ndarray = None):
        """Initialize dataset.
        
        Args:
            X: Feature array.
            y_temp: Temperature target array.
            y_frost: Frost target array (0/1).
            sequence_length: Length of input sequences.
            station_ids: Optional array of station IDs. If provided, sequences will not cross station boundaries.
        """
        self.sequence_length = sequence_length
        self.X = torch.FloatTensor(X)
        self.y_temp = torch.FloatTensor(y_temp)
        self.y_frost = torch.FloatTensor(y_frost)
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
            self.y_temp[actual_idx + self.sequence_length - 1],
            self.y_frost[actual_idx + self.sequence_length - 1]
        )


class LSTMMultiTaskModel(nn.Module):
    """Multi-task LSTM neural network model.
    
    This model has two output heads:
    1. Temperature prediction (regression)
    2. Frost probability prediction (classification)
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """Initialize multi-task LSTM model.
        
        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super(LSTMMultiTaskModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Shared LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
        # Two output heads
        self.fc_temp = nn.Linear(hidden_size, 1)      # Temperature (regression)
        self.fc_frost = nn.Linear(hidden_size, 1)     # Frost probability (classification)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence, features).
        
        Returns:
            Tuple of (temperature_pred, frost_prob_pred):
            - temperature_pred: Tensor of shape (batch,)
            - frost_prob_pred: Tensor of shape (batch,)
        """
        lstm_out, _ = self.lstm(x)
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        
        # Two output heads
        temp_pred = self.fc_temp(lstm_out).squeeze()           # Temperature (regression)
        frost_logit = self.fc_frost(lstm_out).squeeze()        # Frost logit (classification)
        frost_prob = torch.sigmoid(frost_logit)                # Convert to probability
        
        return temp_pred, frost_prob


class LSTMMultiTaskForecastModel(BaseModel):
    """Multi-task LSTM forecast model wrapper.
    
    This model predicts both temperature and frost probability simultaneously.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-task LSTM forecast model.
        
        Args:
            config: Configuration dictionary containing:
                - model_params: Model-specific parameters
                    - sequence_length: Length of input sequences (default: 24)
                    - hidden_size: Number of hidden units (default: 128)
                    - num_layers: Number of LSTM layers (default: 2)
                    - dropout: Dropout rate (default: 0.2)
                    - learning_rate: Learning rate (default: 0.001)
                    - batch_size: Batch size (default: 32)
                    - epochs: Number of training epochs (default: 100)
                    - loss_weight_temp: Weight for temperature loss (default: 0.5)
                    - loss_weight_frost: Weight for frost loss (default: 0.5)
                    - early_stopping: Use early stopping (default: True)
                    - patience: Early stopping patience (default: 10)
                    - lr_scheduler: Use learning rate scheduler (default: True)
                    - gradient_clip: Gradient clipping value (default: 1.0)
                    - save_best_model: Save best model during training (default: True)
        """
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM models. Install with: pip install torch")
        
        model_params = config.get("model_params", {})
        
        # Model architecture parameters
        self.sequence_length = model_params.get("sequence_length", 24)
        self.hidden_size = model_params.get("hidden_size", 128)
        self.num_layers = model_params.get("num_layers", 2)
        self.dropout = model_params.get("dropout", 0.2)
        
        # Training parameters
        self.learning_rate = model_params.get("learning_rate", 0.001)
        self.batch_size = model_params.get("batch_size", 32)
        self.epochs = model_params.get("epochs", 100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Loss weights for multi-task learning
        self.loss_weight_temp = model_params.get("loss_weight_temp", 0.5)
        self.loss_weight_frost = model_params.get("loss_weight_frost", 0.5)
        
        # Optimization parameters
        self.use_early_stopping = model_params.get("early_stopping", True)
        self.patience = model_params.get("patience", 10)
        self.min_delta = model_params.get("min_delta", 1e-6)
        self.use_lr_scheduler = model_params.get("lr_scheduler", True)
        self.lr_scheduler_patience = model_params.get("lr_scheduler_patience", 5)
        self.lr_scheduler_factor = model_params.get("lr_scheduler_factor", 0.5)
        self.gradient_clip_value = model_params.get("gradient_clip", 1.0)
        self.save_best_model = model_params.get("save_best_model", True)
        
        # Station grouping support
        self.station_column = config.get("station_column", "Stn Id")
        
        # Model will be initialized in fit() when we know input_size
        self.model = None
        self.input_size = None
    
    def fit(self, X: pd.DataFrame, y_temp: pd.Series, y_frost: pd.Series, **kwargs) -> "LSTMMultiTaskForecastModel":
        """Train the multi-task LSTM model.
        
        Args:
            X: Feature DataFrame.
            y_temp: Temperature target Series.
            y_frost: Frost target Series (0/1).
            **kwargs: Additional arguments (e.g., station_ids).
        
        Returns:
            Self for method chaining.
        """
        self.feature_names = list(X.columns)
        self.input_size = len(self.feature_names)
        
        # Initialize model
        self.model = LSTMMultiTaskModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Prepare data - convert to numpy arrays and free original DataFrame memory
        X_array = X.values.astype(np.float32)
        y_temp_array = y_temp.values.astype(np.float32)
        y_frost_array = y_frost.values.astype(np.float32)
        
        # Free original DataFrame memory immediately
        del X, y_temp, y_frost
        import gc
        gc.collect()
        
        # Get station IDs if available
        station_ids = kwargs.get('station_ids', None)
        
        # Check for NaN values and handle them
        if np.isnan(X_array).any():
            print(f"  ⚠️  Warning: Found NaN values in features, filling with 0")
            X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Remove rows with NaN targets
        valid_mask = ~(np.isnan(y_temp_array) | np.isnan(y_frost_array))
        if not valid_mask.all():
            X_array = X_array[valid_mask]
            y_temp_array = y_temp_array[valid_mask]
            y_frost_array = y_frost_array[valid_mask]
            if station_ids is not None:
                station_ids = np.asarray(station_ids)[valid_mask]
            print(f"  Removed {np.sum(~valid_mask)} rows with NaN targets, remaining: {len(y_temp_array)}")
        
        # Check for infinite values
        if np.isinf(X_array).any():
            print(f"  ⚠️  Warning: Found infinite values in features, clipping to finite range")
            X_array = np.clip(X_array, -1e6, 1e6)
        
        if np.isinf(y_temp_array).any() or np.isinf(y_frost_array).any():
            print(f"  ⚠️  Warning: Found infinite values in targets, clipping to finite range")
            y_temp_array = np.clip(y_temp_array, -1e6, 1e6)
            y_frost_array = np.clip(y_frost_array, -1.0, 1.0)  # Frost is probability, should be [0, 1]
        if station_ids is not None:
            station_ids = np.asarray(station_ids)
            if len(station_ids) != len(X):
                print(f"  ⚠️  Warning: station_ids length ({len(station_ids)}) doesn't match X length ({len(X)}). Ignoring station_ids.")
                station_ids = None
        
        # Create datasets with station grouping support
        dataset = TimeSeriesDataset(X_array, y_temp_array, y_frost_array, 
                                   self.sequence_length, station_ids=station_ids)
        
        # Free numpy arrays after dataset creation (dataset will keep its own copies)
        del X_array, y_temp_array, y_frost_array
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
        criterion_temp = nn.MSELoss()
        criterion_frost = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = None
        if self.use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                min_lr=1e-6
            )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training history for logging
        training_history = {
            'epoch': [],
            'train_loss_total': [],
            'train_loss_temp': [],
            'train_loss_frost': [],
            'val_loss_total': [],
            'val_loss_temp': [],
            'val_loss_frost': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Setup training utilities if not already set up
        checkpoint_dir = kwargs.get('checkpoint_dir', None)
        checkpoint_frequency = model_params.get("checkpoint_frequency", 10)
        if checkpoint_dir and not self.checkpoint_manager:
            # Setup training tools if checkpoint_dir provided
            self.setup_training_tools(
                checkpoint_dir=checkpoint_dir,
                checkpoint_frequency=checkpoint_frequency,
                save_best=self.save_best_model,
                best_metric="val_loss_total",
                best_mode="min"
            )
            self.checkpoint_dir = Path(checkpoint_dir)  # For backward compatibility
        
        # Initialize training history if not already set up
        if not self.training_history:
            from src.models.utils import TrainingHistory
            # Multi-task model needs additional metrics
            self.training_history = TrainingHistory(metrics=[
                'train_loss_total', 'train_loss_temp', 'train_loss_frost',
                'val_loss_total', 'val_loss_temp', 'val_loss_frost',
                'learning_rate', 'epoch_time'
            ])
        
        # Initialize progress logger if not already set up
        if not self.progress_logger:
            from src.models.utils import ProgressLogger
            self.progress_logger = ProgressLogger()
        
        # Start training history tracking
        self.training_history.start_training()
        
        # Log training start
        self.progress_logger.log_training_start(
            model_name="LSTM Multi-task",
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
            print("  ⚠️  tqdm not available, using simple progress display")
        
        import time
        epoch_start_time = time.time()
        
        self.model.train()
        epoch_pbar = tqdm(range(self.epochs), desc="Training", unit="epoch", disable=not USE_TQDM) if USE_TQDM else range(self.epochs)
        
        for epoch in epoch_pbar:
            epoch_iter_start = time.time()
            # Training phase with progress bar
            train_loss_total = 0.0
            train_loss_temp = 0.0
            train_loss_frost = 0.0
            train_batches = 0
            
            if USE_TQDM:
                train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]", 
                                 leave=False, unit="batch")
            else:
                train_pbar = train_loader
            
            for batch_X, batch_y_temp, batch_y_frost in train_pbar:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y_temp = batch_y_temp.to(self.device, non_blocking=True)
                batch_y_frost = batch_y_frost.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                temp_pred, frost_pred = self.model(batch_X)
                
                # Calculate losses
                loss_temp = criterion_temp(temp_pred, batch_y_temp)
                loss_frost = criterion_frost(frost_pred, batch_y_frost)
                
                # Combined loss with weights
                loss_total = self.loss_weight_temp * loss_temp + self.loss_weight_frost * loss_frost
                
                loss_total.backward()
                
                # Gradient clipping
                if self.gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                
                optimizer.step()
                
                # Check for NaN loss (before deleting)
                loss_total_value = loss_total.item()
                loss_temp_value = loss_temp.item()
                loss_frost_value = loss_frost.item()
                
                # Memory optimization: delete batch data immediately after use
                del batch_X, batch_y_temp, batch_y_frost, temp_pred, frost_pred, loss_temp, loss_frost, loss_total
                
                if np.isnan(loss_total_value) or np.isinf(loss_total_value):
                    print(f"\n  ⚠️  Warning: Invalid loss value (total={loss_total_value}, temp={loss_temp_value}, frost={loss_frost_value}) at batch {train_batches}")
                    print(f"     This may indicate numerical instability. Consider:")
                    print(f"     - Reducing learning rate")
                    print(f"     - Checking for NaN/inf in input data")
                    print(f"     - Using gradient clipping")
                    # Skip this batch's loss
                    continue
                
                train_loss_total += loss_total_value
                train_loss_temp += loss_temp_value
                train_loss_frost += loss_frost_value
                train_batches += 1
                
                # Update progress bar
                if USE_TQDM and train_batches % 10 == 0:
                    train_pbar.set_postfix({
                        'total': f'{loss_total_value:.6f}',
                        'temp': f'{loss_temp_value:.6f}',
                        'frost': f'{loss_frost_value:.6f}'
                    })
            
            avg_train_loss_total = train_loss_total / train_batches
            avg_train_loss_temp = train_loss_temp / train_batches
            avg_train_loss_frost = train_loss_frost / train_batches
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
                torch.cuda.empty_cache()
            
            # Validation phase with progress bar
            self.model.eval()
            val_loss_total = 0.0
            val_loss_temp = 0.0
            val_loss_frost = 0.0
            val_batches = 0
            
            if USE_TQDM:
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]", 
                               leave=False, unit="batch")
            else:
                val_pbar = val_loader
            
            with torch.no_grad():
                for batch_X, batch_y_temp, batch_y_frost in val_pbar:
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y_temp = batch_y_temp.to(self.device, non_blocking=True)
                    batch_y_frost = batch_y_frost.to(self.device, non_blocking=True)
                    
                    temp_pred, frost_pred = self.model(batch_X)
                    
                    loss_temp = criterion_temp(temp_pred, batch_y_temp)
                    loss_frost = criterion_frost(frost_pred, batch_y_frost)
                    loss_total = self.loss_weight_temp * loss_temp + self.loss_weight_frost * loss_frost
                    
                    val_loss_total += loss_total.item()
                    val_loss_temp += loss_temp.item()
                    val_loss_frost += loss_frost.item()
                    val_batches += 1
                    
                    # Memory optimization: delete batch data immediately
                    del batch_X, batch_y_temp, batch_y_frost, temp_pred, frost_pred, loss_temp, loss_frost, loss_total
                    
                    # Update progress bar
                    if USE_TQDM and val_batches % 10 == 0:
                        val_pbar.set_postfix({
                            'total': f'{val_loss_total/val_batches:.6f}',
                            'temp': f'{val_loss_temp/val_batches:.6f}',
                            'frost': f'{val_loss_frost/val_batches:.6f}'
                        })
            
            avg_val_loss_total = val_loss_total / val_batches
            avg_val_loss_temp = val_loss_temp / val_batches
            avg_val_loss_frost = val_loss_frost / val_batches
            
            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step(avg_val_loss_total)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_iter_start
            elapsed_time = time.time() - epoch_start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = self.epochs - (epoch + 1)
            estimated_remaining = avg_epoch_time * remaining_epochs
            
            # Record epoch in training history
            self.training_history.record_epoch(
                epoch=epoch + 1,
                train_loss_total=avg_train_loss_total,
                train_loss_temp=avg_train_loss_temp,
                train_loss_frost=avg_train_loss_frost,
                val_loss_total=avg_val_loss_total,
                val_loss_temp=avg_val_loss_temp,
                val_loss_frost=avg_val_loss_frost,
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
                    scaler_state=None,  # Multi-task model doesn't use scaler yet
                    metrics={
                        'train_loss_total': avg_train_loss_total,
                        'val_loss_total': avg_val_loss_total
                    },
                    training_history=self.training_history
                )
                self.progress_logger.log(f"  💾 Checkpoint saved: epoch {epoch + 1}", flush=True)
            
            # Early stopping logic
            if self.use_early_stopping:
                improved = False
                if avg_val_loss_total < best_val_loss - self.min_delta:
                    best_val_loss = avg_val_loss_total
                    patience_counter = 0
                    improved = True
                    if self.save_best_model:
                        best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # Update progress bar or print
                if USE_TQDM:
                    epoch_pbar.set_postfix({
                        'train': f'{avg_train_loss_total:.6f}',
                        'val': f'{avg_val_loss_total:.6f}',
                        'temp': f'{avg_val_loss_temp:.6f}',
                        'frost': f'{avg_val_loss_frost:.6f}',
                        'lr': f'{current_lr:.6f}',
                        'patience': f'{patience_counter}/{self.patience}',
                        'ETA': f'{estimated_remaining/60:.1f}m'
                    })
                    if improved:
                        epoch_pbar.write(f"  ✅ Improved! Val Loss: {avg_val_loss_total:.6f} (Best: {best_val_loss:.6f})")
                else:
                    # Print every 5 epochs or on improvement
                    if epoch % 5 == 0 or improved or epoch == 0:
                        print(f"  Epoch {epoch+1}/{self.epochs} - "
                              f"Train: {avg_train_loss_total:.6f} (Temp: {avg_train_loss_temp:.6f}, Frost: {avg_train_loss_frost:.6f}) - "
                              f"Val: {avg_val_loss_total:.6f} (Temp: {avg_val_loss_temp:.6f}, Frost: {avg_val_loss_frost:.6f}) - "
                              f"LR: {current_lr:.6f}, Patience: {patience_counter}/{self.patience} - "
                              f"Time: {epoch_time:.1f}s, ETA: {estimated_remaining/60:.1f}m")
                
                # Early stopping
                if patience_counter >= self.patience:
                    if self.progress_logger:
                        self.progress_logger.log_early_stopping(epoch + 1, self.patience)
                    if self.save_best_model and best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
            else:
                # Print progress every epoch
                if USE_TQDM:
                    epoch_pbar.set_postfix({
                        'train': f'{avg_train_loss_total:.6f}',
                        'val': f'{avg_val_loss_total:.6f}',
                        'temp': f'{avg_val_loss_temp:.6f}',
                        'frost': f'{avg_val_loss_frost:.6f}',
                        'lr': f'{current_lr:.6f}',
                        'ETA': f'{estimated_remaining/60:.1f}m'
                    })
                else:
                    if epoch % 5 == 0 or epoch == 0:
                        print(f"  Epoch {epoch+1}/{self.epochs} - "
                              f"Train: {avg_train_loss_total:.6f} (Temp: {avg_train_loss_temp:.6f}, Frost: {avg_train_loss_frost:.6f}) - "
                              f"Val: {avg_val_loss_total:.6f} (Temp: {avg_val_loss_temp:.6f}, Frost: {avg_val_loss_frost:.6f}) - "
                              f"Time: {epoch_time:.1f}s, ETA: {estimated_remaining/60:.1f}m")
        
        # Save training artifacts using unified utilities
        training_time = time.time() - epoch_start_time
        if self.checkpoint_dir and self.training_history and len(self.training_history) > 0:
            self.save_training_artifacts(self.checkpoint_dir)
        self.progress_logger.log_training_complete(training_time, len(self.training_history))
        
        # Final memory cleanup
        del train_loader, val_loader, train_dataset, val_dataset, dataset
        if 'station_ids' in locals():
            del station_ids
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_fitted = True
        return self
    
    def predict_temp(self, X: pd.DataFrame) -> np.ndarray:
        """Predict temperature.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Temperature predictions array.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        X_array = X.values.astype(np.float32)
        
        predictions = []
        with torch.no_grad():
            for i in range(len(X_array) - self.sequence_length + 1):
                sequence = X_array[i:i + self.sequence_length]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                temp_pred, _ = self.model(sequence_tensor)
                predictions.append(temp_pred.cpu().numpy())
        
        # Pad predictions for initial sequence_length - 1 samples
        if len(predictions) < len(X_array):
            padding = [predictions[0]] * (self.sequence_length - 1)
            predictions = padding + predictions
        
        return np.array(predictions).flatten()
    
    def predict_frost_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict frost probability.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Frost probability predictions array.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        X_array = X.values.astype(np.float32)
        
        predictions = []
        with torch.no_grad():
            for i in range(len(X_array) - self.sequence_length + 1):
                sequence = X_array[i:i + self.sequence_length]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                _, frost_pred = self.model(sequence_tensor)
                predictions.append(frost_pred.cpu().numpy())
        
        # Pad predictions for initial sequence_length - 1 samples
        if len(predictions) < len(X_array):
            padding = [predictions[0]] * (self.sequence_length - 1)
            predictions = padding + predictions
        
        return np.array(predictions).flatten()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions (returns temperature for compatibility).
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Temperature predictions array.
        """
        return self.predict_temp(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict frost probability.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Frost probability predictions array.
        """
        return self.predict_frost_proba(X)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Multi-task LSTM doesn't provide direct feature importance.
        
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
        
        config_path = path / "config.json"
        config_to_save = {
            "feature_names": self.feature_names,
            "input_size": self.input_size,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "model_params": self.config.get("model_params", {}),
        }
        
        with open(config_path, 'w') as f:
            import json
            json.dump(config_to_save, f, indent=2)
        
        print(f"  ✅ Multi-task LSTM model saved to {path}")
    
    def load(self, path: Path) -> "LSTMMultiTaskForecastModel":
        """Load model from disk.
        
        Args:
            path: Directory path to load model from.
        
        Returns:
            Self for method chaining.
        """
        path = Path(path)
        
        config_path = path / "config.json"
        with open(config_path, 'r') as f:
            import json
            saved_config = json.load(f)
        
        self.feature_names = saved_config["feature_names"]
        self.input_size = saved_config["input_size"]
        self.sequence_length = saved_config["sequence_length"]
        self.hidden_size = saved_config["hidden_size"]
        self.num_layers = saved_config["num_layers"]
        self.dropout = saved_config["dropout"]
        
        # Initialize model
        self.model = LSTMMultiTaskModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Load weights
        model_path = path / "model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.is_fitted = True
        print(f"  ✅ Multi-task LSTM model loaded from {path}")
        
        return self

