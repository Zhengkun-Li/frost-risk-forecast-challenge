"""Training history tracking utility."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import time


class TrainingHistory:
    """Track and manage training history across epochs.
    
    This class provides a unified interface for recording training metrics
    that can be used by any model type (tree models, neural networks, etc.).
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """Initialize training history tracker.
        
        Args:
            metrics: List of metric names to track. If None, uses default metrics.
        """
        if metrics is None:
            metrics = ['train_loss', 'val_loss', 'learning_rate', 'epoch_time']
        
        self.metrics = metrics
        self.history: Dict[str, List[Any]] = {
            'epoch': [],
            **{metric: [] for metric in metrics}
        }
        self.start_time: Optional[float] = None
        self.current_epoch: int = 0
    
    def start_training(self) -> None:
        """Mark the start of training."""
        self.start_time = time.time()
    
    def record_epoch(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        **kwargs
    ) -> None:
        """Record metrics for a single epoch.
        
        Args:
            epoch: Epoch number (1-indexed).
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch.
            learning_rate: Learning rate for this epoch.
            **kwargs: Additional metrics to record.
        """
        self.current_epoch = epoch
        self.history['epoch'].append(epoch)
        
        # Record standard metrics
        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if learning_rate is not None:
            self.history['learning_rate'].append(learning_rate)
        
        # Record epoch time
        epoch_time = kwargs.pop('epoch_time', None)
        if epoch_time is not None:
            self.history['epoch_time'].append(epoch_time)
        
        # Record any additional metrics
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_history(self) -> Dict[str, List[Any]]:
        """Get the complete training history.
        
        Returns:
            Dictionary mapping metric names to lists of values.
        """
        return self.history.copy()
    
    def get_latest(self, metric: str) -> Optional[Any]:
        """Get the latest value for a metric.
        
        Args:
            metric: Name of the metric.
        
        Returns:
            Latest value, or None if not available.
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return None
        return self.history[metric][-1]
    
    def save(self, path: Path) -> None:
        """Save training history to JSON file.
        
        Args:
            path: Path to save the history JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add training duration if available
        history_data = self.history.copy()
        if self.start_time is not None:
            history_data['training_duration_seconds'] = time.time() - self.start_time
            history_data['total_epochs'] = len(self.history['epoch'])
        
        with open(path, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> "TrainingHistory":
        """Load training history from JSON file.
        
        Args:
            path: Path to the history JSON file.
        
        Returns:
            TrainingHistory instance with loaded data.
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Extract metrics (all keys except special ones)
        special_keys = {'epoch', 'training_duration_seconds', 'total_epochs'}
        metrics = [k for k in data.keys() if k not in special_keys]
        
        instance = cls(metrics=metrics)
        instance.history = data
        if 'epoch' in data and len(data['epoch']) > 0:
            instance.current_epoch = data['epoch'][-1]
        
        return instance
    
    def __len__(self) -> int:
        """Return the number of recorded epochs."""
        return len(self.history['epoch'])

