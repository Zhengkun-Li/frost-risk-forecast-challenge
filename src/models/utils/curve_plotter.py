"""Training curve plotting utility."""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys


class TrainingCurvePlotter:
    """Plot training curves from training history.
    
    This class provides a unified interface for plotting training metrics
    that can be used by any model type.
    """
    
    def __init__(self, backend: str = "matplotlib"):
        """Initialize curve plotter.
        
        Args:
            backend: Plotting backend ("matplotlib" or "plotly").
        """
        self.backend = backend
        self._matplotlib_available = False
        self._plotly_available = False
        
        # Check availability
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            self._matplotlib_available = True
        except ImportError:
            pass
        
        try:
            import plotly.graph_objects as go
            self._plotly_available = True
        except ImportError:
            pass
    
    def plot(
        self,
        history: Dict[str, List[Any]],
        save_path: Path,
        title: str = "Training Curves",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 150
    ) -> bool:
        """Plot training curves from history.
        
        Args:
            history: Training history dictionary with metric names as keys.
            save_path: Path to save the plot.
            title: Plot title.
            figsize: Figure size (width, height).
            dpi: Resolution for saved figure.
        
        Returns:
            True if plot was saved successfully, False otherwise.
        """
        if self.backend == "matplotlib" and self._matplotlib_available:
            return self._plot_matplotlib(history, save_path, title, figsize, dpi)
        elif self.backend == "plotly" and self._plotly_available:
            return self._plot_plotly(history, save_path, title)
        else:
            print(f"  ⚠️  Plotting backend '{self.backend}' not available, skipping plot", flush=True)
            return False
    
    def _plot_matplotlib(
        self,
        history: Dict[str, List[Any]],
        save_path: Path,
        title: str,
        figsize: Tuple[int, int],
        dpi: int
    ) -> bool:
        """Plot using matplotlib.
        
        Args:
            history: Training history dictionary.
            save_path: Path to save the plot.
            title: Plot title.
            figsize: Figure size.
            dpi: Resolution.
        
        Returns:
            True if successful.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Determine number of subplots based on available metrics
            metrics_to_plot = []
            if 'train_loss' in history or 'val_loss' in history:
                metrics_to_plot.append(('loss', ['train_loss', 'val_loss']))
            if 'learning_rate' in history:
                metrics_to_plot.append(('lr', ['learning_rate']))
            
            if len(metrics_to_plot) == 0:
                print("  ⚠️  No metrics to plot", flush=True)
                return False
            
            n_plots = len(metrics_to_plot)
            fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
            if n_plots == 1:
                axes = [axes]
            
            epochs = history.get('epoch', list(range(1, len(history.get('train_loss', [])) + 1)))
            
            for idx, (plot_type, metric_names) in enumerate(metrics_to_plot):
                ax = axes[idx]
                
                if plot_type == 'loss':
                    if 'train_loss' in history:
                        train_loss = [v for v in history['train_loss'] if v != float('inf')]
                        train_epochs = epochs[:len(train_loss)]
                        ax.plot(train_epochs, train_loss, label='Train Loss', 
                               marker='o', markersize=3, linewidth=1.5)
                    
                    if 'val_loss' in history:
                        val_loss = [(e, v) for e, v in zip(epochs, history['val_loss']) 
                                   if v != float('inf')]
                        if val_loss:
                            val_epochs, val_losses = zip(*val_loss)
                            ax.plot(val_epochs, val_losses, label='Val Loss', 
                                   marker='s', markersize=3, linewidth=1.5)
                    
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title('Training and Validation Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                elif plot_type == 'lr':
                    if 'learning_rate' in history:
                        lr = history['learning_rate']
                        ax.plot(epochs[:len(lr)], lr, label='Learning Rate', 
                               marker='o', markersize=3, linewidth=1.5, color='green')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Learning Rate')
                        ax.set_title('Learning Rate Schedule')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.set_yscale('log')
            
            plt.suptitle(title, fontsize=14, y=0.995)
            plt.tight_layout()
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"  ⚠️  Failed to plot training curves: {e}", flush=True)
            return False
    
    def _plot_plotly(
        self,
        history: Dict[str, List[Any]],
        save_path: Path,
        title: str
    ) -> bool:
        """Plot using plotly (interactive).
        
        Args:
            history: Training history dictionary.
            save_path: Path to save the plot.
            title: Plot title.
        
        Returns:
            True if successful.
        """
        # Placeholder for plotly implementation
        # Can be extended in the future
        return False
    
    def plot_multitask(
        self,
        history: Dict[str, List[Any]],
        save_path: Path,
        title: str = "Multi-task Training Curves",
        figsize: Tuple[int, int] = (10, 12),
        dpi: int = 150
    ) -> bool:
        """Plot training curves for multi-task models.
        
        Args:
            history: Training history dictionary with multi-task metrics.
            save_path: Path to save the plot.
            title: Plot title.
            figsize: Figure size.
            dpi: Resolution.
        
        Returns:
            True if successful.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            epochs = history.get('epoch', list(range(1, len(history.get('train_loss_total', [])) + 1)))
            
            fig, axes = plt.subplots(3, 1, figsize=figsize)
            
            # Plot 1: Total loss
            if 'train_loss_total' in history and 'val_loss_total' in history:
                axes[0].plot(epochs, history['train_loss_total'], 
                           label='Train Loss (Total)', marker='o', markersize=3, linewidth=1.5)
                axes[0].plot(epochs, history['val_loss_total'], 
                           label='Val Loss (Total)', marker='s', markersize=3, linewidth=1.5)
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Total Loss (Temperature + Frost)')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Task-specific losses
            if 'train_loss_temp' in history:
                axes[1].plot(epochs, history['train_loss_temp'], 
                           label='Train Temp Loss', marker='o', markersize=3, linewidth=1.5, color='orange')
            if 'val_loss_temp' in history:
                axes[1].plot(epochs, history['val_loss_temp'], 
                           label='Val Temp Loss', marker='s', markersize=3, linewidth=1.5, color='orange', linestyle='--')
            if 'train_loss_frost' in history:
                axes[1].plot(epochs, history['train_loss_frost'], 
                           label='Train Frost Loss', marker='o', markersize=3, linewidth=1.5, color='blue')
            if 'val_loss_frost' in history:
                axes[1].plot(epochs, history['val_loss_frost'], 
                           label='Val Frost Loss', marker='s', markersize=3, linewidth=1.5, color='blue', linestyle='--')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Task-Specific Losses')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Learning rate
            if 'learning_rate' in history:
                axes[2].plot(epochs, history['learning_rate'], 
                           label='Learning Rate', marker='o', markersize=3, linewidth=1.5, color='green')
                axes[2].set_xlabel('Epoch')
                axes[2].set_ylabel('Learning Rate')
                axes[2].set_title('Learning Rate Schedule')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                axes[2].set_yscale('log')
            
            plt.suptitle(title, fontsize=14, y=0.995)
            plt.tight_layout()
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"  ⚠️  Failed to plot multi-task training curves: {e}", flush=True)
            return False

