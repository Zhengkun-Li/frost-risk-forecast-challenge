"""Checkpoint management utility for model training."""

from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import json
import time


class CheckpointManager:
    """Manage model checkpoints during training.
    
    This class provides a unified interface for saving and loading checkpoints
    that can be used by any model type.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_frequency: int = 10,
        save_best: bool = True,
        best_metric: str = "val_loss",
        best_mode: str = "min"
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
            checkpoint_frequency: Save checkpoint every N epochs (0 = disabled).
            save_best: Whether to save the best model based on metric.
            best_metric: Metric name to use for determining best model.
            best_mode: "min" or "max" - whether lower or higher is better.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_frequency = checkpoint_frequency
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.checkpoint_count = 0
    
    def should_save_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint should be saved for this epoch.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            True if checkpoint should be saved.
        """
        if self.checkpoint_frequency <= 0:
            return False
        return epoch % self.checkpoint_frequency == 0
    
    def is_best(self, metric_value: float) -> bool:
        """Check if the current metric value is the best so far.
        
        Args:
            metric_value: Current metric value.
        
        Returns:
            True if this is the best value seen so far.
        """
        if self.best_value is None:
            return True
        
        if self.best_mode == "min":
            return metric_value < self.best_value
        else:
            return metric_value > self.best_value
    
    def update_best(self, epoch: int, metric_value: float) -> bool:
        """Update best metric value and epoch.
        
        Args:
            epoch: Current epoch number.
            metric_value: Current metric value.
        
        Returns:
            True if this is a new best value.
        """
        if self.is_best(metric_value):
            self.best_value = metric_value
            self.best_epoch = epoch
            return True
        return False
    
    def save_checkpoint(
        self,
        epoch: int,
        model_state: Any,
        optimizer_state: Optional[Any] = None,
        scheduler_state: Optional[Any] = None,
        scaler_state: Optional[Any] = None,
        metrics: Optional[Dict[str, Any]] = None,
        training_history: Optional[Any] = None,
        custom_data: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save a checkpoint.
        
        Args:
            epoch: Epoch number.
            model_state: Model state to save (dict, state_dict, etc.).
            optimizer_state: Optimizer state (optional).
            scheduler_state: Learning rate scheduler state (optional).
            scaler_state: Mixed precision scaler state (optional).
            metrics: Dictionary of current metrics.
            training_history: Training history object or dict.
            custom_data: Additional custom data to save.
        
        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'timestamp': time.time(),
            'model_state': model_state,
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state'] = optimizer_state
        if scheduler_state is not None:
            checkpoint['scheduler_state'] = scheduler_state
        if scaler_state is not None:
            checkpoint['scaler_state'] = scaler_state
        if metrics is not None:
            checkpoint['metrics'] = metrics
        if training_history is not None:
            if hasattr(training_history, 'get_history'):
                checkpoint['training_history'] = training_history.get_history()
            else:
                checkpoint['training_history'] = training_history
        if custom_data is not None:
            checkpoint.update(custom_data)
        
        # Save checkpoint (using pickle for PyTorch models, JSON for others)
        import pickle
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.checkpoint_count += 1
        return checkpoint_path
    
    def save_best_checkpoint(
        self,
        epoch: int,
        model_state: Any,
        metric_value: float,
        **kwargs
    ) -> Optional[Path]:
        """Save checkpoint if this is the best model so far.
        
        Args:
            epoch: Epoch number.
            model_state: Model state to save.
            metric_value: Current metric value.
            **kwargs: Additional arguments passed to save_checkpoint.
        
        Returns:
            Path to saved checkpoint if saved, None otherwise.
        """
        if not self.save_best:
            return None
        
        if self.update_best(epoch, metric_value):
            best_path = self.checkpoint_dir / "best_model.pth"
            checkpoint = {
                'epoch': epoch,
                'timestamp': time.time(),
                'model_state': model_state,
                'best_metric': self.best_metric,
                'best_value': metric_value,
                **kwargs
            }
            
            import pickle
            with open(best_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            return best_path
        return None
    
    def get_checkpoint_path(self, epoch: Optional[int] = None) -> Path:
        """Get path to a checkpoint file.
        
        Args:
            epoch: Epoch number. If None, returns best model path.
        
        Returns:
            Path to checkpoint file.
        """
        if epoch is None:
            return self.checkpoint_dir / "best_model.pth"
        return self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoint files.
        
        Returns:
            List of checkpoint file paths, sorted by epoch.
        """
        checkpoints = []
        for path in self.checkpoint_dir.glob("checkpoint_epoch_*.pth"):
            checkpoints.append(path)
        return sorted(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    
    def load_checkpoint(self, epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Load a checkpoint from disk.
        
        Args:
            epoch: Epoch number to load. If None, loads the best model.
        
        Returns:
            Dictionary containing checkpoint data, or None if not found.
        """
        checkpoint_path = self.get_checkpoint_path(epoch)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            import pickle
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Update internal state
            if 'epoch' in checkpoint:
                self.best_epoch = checkpoint.get('epoch', 0)
            if 'best_value' in checkpoint:
                self.best_value = checkpoint.get('best_value')
            
            return checkpoint
        except Exception as e:
            print(f"  ⚠️  Failed to load checkpoint from {checkpoint_path}: {e}", flush=True)
            return None
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint available.
        
        Returns:
            Dictionary containing latest checkpoint data, or None if not found.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        
        # Get the latest checkpoint (last in sorted list)
        latest_path = checkpoints[-1]
        try:
            import pickle
            with open(latest_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  ⚠️  Failed to load latest checkpoint from {latest_path}: {e}", flush=True)
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get checkpoint manager information.
        
        Returns:
            Dictionary with checkpoint manager state.
        """
        return {
            'checkpoint_dir': str(self.checkpoint_dir),
            'checkpoint_frequency': self.checkpoint_frequency,
            'save_best': self.save_best,
            'best_metric': self.best_metric,
            'best_mode': self.best_mode,
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'checkpoint_count': self.checkpoint_count,
        }

