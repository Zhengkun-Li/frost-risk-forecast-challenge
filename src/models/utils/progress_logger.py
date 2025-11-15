"""Progress logging utility for model training."""

from typing import Optional, Dict, Any
import sys
import time
from pathlib import Path


class ProgressLogger:
    """Log training progress in a unified way.
    
    This class provides a unified interface for logging training progress
    that can be used by any model type.
    """
    
    def __init__(
        self,
        log_file: Optional[Path] = None,
        use_tqdm: bool = True,
        flush_interval: int = 1
    ):
        """Initialize progress logger.
        
        Args:
            log_file: Optional path to log file.
            use_tqdm: Whether to use tqdm for progress bars.
            flush_interval: Flush output every N messages.
        """
        self.log_file = Path(log_file) if log_file else None
        self.use_tqdm = use_tqdm
        self.flush_interval = flush_interval
        self.message_count = 0
        
        # Check tqdm availability
        self._tqdm_available = False
        if use_tqdm:
            try:
                import tqdm
                self._tqdm_available = True
            except ImportError:
                self._tqdm_available = False
        
        # Setup log file
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str, flush: bool = False) -> None:
        """Log a message.
        
        Args:
            message: Message to log.
            flush: Whether to flush immediately.
        """
        # Print to stdout
        print(message, flush=flush or (self.message_count % self.flush_interval == 0))
        
        # Write to log file if available
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(message + '\n')
                    if flush:
                        f.flush()
            except Exception as e:
                # Don't fail if logging fails
                pass
        
        self.message_count += 1
        if flush:
            sys.stdout.flush()
    
    def log_training_start(
        self,
        model_name: str,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log training start information.
        
        Args:
            model_name: Name of the model.
            device: Device being used (e.g., "cuda", "cpu").
            config: Model configuration dictionary.
        """
        self.log(f"\n  🚀 Starting {model_name} training", flush=True)
        if device:
            self.log(f"     Device: {device}", flush=True)
        if config:
            for key, value in config.items():
                if isinstance(value, (int, float, str, bool)):
                    self.log(f"     {key}: {value}", flush=True)
    
    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        epoch_time: Optional[float] = None,
        eta: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log epoch information.
        
        Args:
            epoch: Current epoch number.
            total_epochs: Total number of epochs.
            train_loss: Training loss.
            val_loss: Validation loss.
            learning_rate: Current learning rate.
            epoch_time: Time taken for this epoch.
            eta: Estimated time remaining.
            **kwargs: Additional metrics to log.
        """
        parts = [f"Epoch {epoch}/{total_epochs}"]
        
        if train_loss is not None:
            parts.append(f"Train: {train_loss:.6f}")
        if val_loss is not None:
            parts.append(f"Val: {val_loss:.6f}")
        if learning_rate is not None:
            parts.append(f"LR: {learning_rate:.6f}")
        if epoch_time is not None:
            parts.append(f"Time: {epoch_time:.1f}s")
        if eta is not None:
            parts.append(f"ETA: {eta/60:.1f}m")
        
        # Add any additional metrics
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                parts.append(f"{key}: {value:.6f}")
            else:
                parts.append(f"{key}: {value}")
        
        message = "  " + " - ".join(parts)
        self.log(message, flush=True)
    
    def log_improvement(
        self,
        metric_name: str,
        current_value: float,
        best_value: float
    ) -> None:
        """Log when a metric improves.
        
        Args:
            metric_name: Name of the metric.
            current_value: Current metric value.
            best_value: Best metric value so far.
        """
        self.log(f"  ✅ Improved! {metric_name}: {current_value:.6f} (Best: {best_value:.6f})", flush=True)
    
    def log_early_stopping(
        self,
        epoch: int,
        patience: int
    ) -> None:
        """Log early stopping.
        
        Args:
            epoch: Epoch where training stopped.
            patience: Patience value used.
        """
        self.log(f"  Early stopping at epoch {epoch} (patience={patience})", flush=True)
    
    def log_training_complete(
        self,
        total_time: float,
        total_epochs: int
    ) -> None:
        """Log training completion.
        
        Args:
            total_time: Total training time in seconds.
            total_epochs: Total number of epochs completed.
        """
        self.log(f"  ✅ Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)", flush=True)
        self.log(f"     Total epochs: {total_epochs}", flush=True)
    
    def get_tqdm(self, iterable, desc: str = "", **kwargs):
        """Get a tqdm progress bar if available.
        
        Args:
            iterable: Iterable to wrap.
            desc: Description for the progress bar.
            **kwargs: Additional arguments for tqdm.
        
        Returns:
            tqdm progress bar or the original iterable.
        """
        if self._tqdm_available and self.use_tqdm:
            from tqdm import tqdm
            return tqdm(iterable, desc=desc, **kwargs)
        return iterable

