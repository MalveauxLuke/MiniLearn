import torch
import matplotlib.pyplot as plt
from operator import attrgetter
import copy
from torcheval.metrics import Mean
from tqdm import tqdm  
from pathlib import Path

from .metrics import MetricsCB
from .utils import to_device, to_cpu  

class CancelFitException(Exception):
    """Raised to cancel the fit phase."""
    pass
    
class CancelBatchException(Exception):
    """Raised to skip the rest of a batch during training."""
    pass
    
class CancelEpochException(Exception): 
    """Raised to skip the rest of the current epoch."""
    pass

# === Callback: Device Transfer ===

class DeviceCB():
    """
    Callback that moves model and batches to a specified device.

    Attributes:
        device (torch.device): The target device.
    """
    order = 0 # Run early in callback order
    
    def __init__(self, device=(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))): 
        self.device = device
        
    def before_fit(self, learn):
        """Move model to target device at the start of training."""
        learn.model.to(self.device)

    def before_batch(self, learn):
        """Move batch to target device before processing."""
        learn.batch = to_device(learn.batch, device=self.device)

# === Callback: Progress Bar/Loss Plotting === 

class ProgressCB():
    """
    Callback that displays progress bars using tqdm and optionally plots training loss.

    Args:
        plot (bool): Whether to plot training loss after fit.
        displayed_plot (bool): Flag that displays whether plot was displayed or not. Used for plotting loss if fit is cancelled
    """
    order = MetricsCB.order + 1 # Ensure runs after metrics

    def __init__(self, plot=False):
        self.plot = plot
        self.losses = []
        self.displayed_plot = False

    def before_fit(self, learn):
        """Reset loss tracker  at the beginning of training"""
        self.losses = []

    def before_epoch(self, learn):
        """Initialize tqdm progress bar for each epioch. """
        phase = "Train" if learn.model.training else "Valid"
        self.pbar = tqdm(learn.dl, desc=f"{phase} Epoch {learn.epoch+1}/{learn.n_epochs}", leave=True, position=0, dynamic_ncols=True)

    def after_batch(self, learn):
        """Update progress bar and log loss during training if plotting is enabled."""
        self.pbar.set_postfix(loss=f"{learn.loss:.4f}")
        self.pbar.update(1)
        if self.plot and learn.model.training:
            self.losses.append(learn.loss.item())

    def after_epoch(self, learn):
        """Cleanup progress bar after every epoch"""
        self.pbar.close()

    def after_fit(self, learn):
        """Display loss plot if enabled after training ends."""
        if self.plot:
            plt.figure(figsize=(8, 4))
            plt.plot(self.losses, label="Train Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss Over Time")
            plt.grid(True)
            plt.legend()
            plt.show()
            self.displayed_plot = True 
            
    def cleanup_fit(self, learn):
        """Display loss plot if enabled even if training is interupted."""
        if self.plot and not self.displayed_plot:
            plt.figure(figsize=(8, 4))
            plt.plot(self.losses, label="Train Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss Over Time")
            plt.grid(True)
            plt.legend()
            plt.show()
            
# === Callback: Checkpoints ===

class CheckpointsCB():
    """
    Callback that handles model checkpointing during validation.

    Saves the best model according to a provided metric, uses interval-based checkpoints every N epochs (optional),
    and uses a cooldown to avoid saving too frequently even when improved.

    Args:
        monitor (str): Metric name to monitor 
        min_delta (float): Minimum change in metric to count as an improvement
        save_interval (int, optional): Save a checkpoint every N epochs (regardless of performance)
        cooldown (int, optional): Number of epochs to wait before allowing another best model save
        mode (str): One of 'min' or 'max'. Use 'min' for loss, 'max' for accuracy metrics
    """
    order = ProgressCB.order+1

    def __init__(self, monitor='loss', min_delta=1e-3, save_interval=None, cooldown=None, mode='min'):
        self.monitor = monitor
        self.min_delta = min_delta
        self.save_interval = save_interval
        self.cooldown = cooldown
        self.mode = mode

        self.best_val = None
        self.best_epoch = -1
        self.best_weights = None
        self.last_save_epoch = -999  # Ensures first save is allowed

        # Create top level directory
        self.checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialized in before_fit
        self.model_checkpoint_dir = None
        self.model_interval_checkpoints_dir = None
        self.model_best_dir = None

    def before_fit(self, learn):
        self.learn = learn
        model_name = learn.model.__class__.__name__.lower()

        # Create model-specific directories
        self.model_checkpoint_dir = self.checkpoint_dir / model_name
        self.model_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model_best_dir = self.model_checkpoint_dir / "best"
        self.model_best_dir.mkdir(parents=True, exist_ok=True)

        if self.save_interval is not None:
            self.model_interval_checkpoints_dir = self.model_checkpoint_dir / "interval_checkpoints"
            self.model_interval_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def after_epoch(self, learn):
        if learn.model.training:
            return

        # Interval checkpointing
        if self.save_interval is not None and (learn.epoch % self.save_interval == 0):
            save_path = self.model_interval_checkpoints_dir / f"checkpoint_epoch_{learn.epoch}.pth"
            torch.save(learn.model.state_dict(), save_path)
            print(f"[CheckpointsCB] Interval checkpoint saved to: {save_path}")

        # Performance checkpointing
        metric = learn.metrics.all_metrics[self.monitor].compute()

        improved = (
            (self.mode == 'min' and (self.best_val is None or metric < self.best_val - self.min_delta)) or
            (self.mode == 'max' and (self.best_val is None or metric > self.best_val + self.min_delta))
        )

        if improved:
            self.best_val = metric
            self.best_weights = copy.deepcopy(learn.model.state_dict())
            self.best_epoch = learn.epoch

            cooldown_passed = (learn.epoch - self.last_save_epoch >= self.cooldown) if self.cooldown else True
            if cooldown_passed:
                save_path = self.model_best_dir / "best_model.pth"
                torch.save(self.best_weights, save_path)
                self.last_save_epoch = learn.epoch

    def cleanup_fit(self, learn):
        if self.best_weights:
            save_path = self.model_best_dir / "best_model.pth"
            print(f"[CheckpointsCB] Best model saved to: {save_path}")
            print(f"[CheckpointsCB] Metric: '{self.monitor}' = {self.best_val:.4f} (epoch {self.best_epoch + 1})")
        else:
            print("[CheckpointsCB] No best model was saved during training.")

# == Callback: Early Stopping ===

class EarlyStoppingCB():
    """
    Callback that stops training if validation doesn't improve.

    Args:
        monitor (str): Metric to track
        min_delta (float): Minimum change to qualify as an improvement
        patience (int): Number of epochs to wait before stopping
        restore_best (bool): Whether to load the best weights before stopping
        mode (str): 'min' or 'max' for loss or accuracy-style metrics
    """
    order = CheckpointsCB.order + 1
    
    def __init__(self, monitor='loss', min_delta=0.0, patience=3, restore_best=True, mode='min'):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.restore_best = restore_best
        self.mode = mode
        self.best_val = None
        self.best_epoch = -1
        self.wait = 0
        self.best_weights = None

    def before_fit(self, learn):
        """
        Allows us to call learn without using self. every time
        """
        self.learn = learn

    def after_epoch(self,learn):
        """
        A function that runs after every epoch that checks for improvement in the specified metric. 

        Skips training phase and only evaluates after the validation epoch. 
        
        Args:
        learn (Learner): The specific learner object.
        """
        if learn.model.training:
            return # Skip train phase
            
        # Get metric
        metric = learn.metrics.all_metrics[self.monitor].compute()

        # Checks for improvements
        improved = (
            (self.mode == 'min' and (self.best_val is None or metric < self.best_val - self.min_delta)) or
            (self.mode == 'max' and (self.best_val is None or metric > self.best_val + self.min_delta))
        )

        if improved:
            self.best_val = metric
            self.best_epoch = learn.epoch
            self.wait = 0
            if self.restore_best:
                self.best_weights = copy.deepcopy(learn.model.state_dict())
        else:
            self.wait += 1
            print(f"No improvement in '{self.monitor}' for {self.wait} epoch(s)...")
            
            if self.wait >= self.patience:
                print(f"Early stopping at epoch {learn.epoch + 1}")
                if self.restore_best and self.best_weights:
                    learn.model.load_state_dict(self.best_weights)
                    print("Restored best model weights.")
                print(f"Best '{self.monitor}': {self.best_val:.3f} at epoch {self.best_epoch + 1}")
                raise CancelFitException()
        
    def after_fit(self, learn):
        """
        Called at the end of training to report the best value and when it occurred.

        Args:
        learn (Learner): The learner.
        """
        print(f"Best '{self.monitor}': {self.best_val:.4f} at epoch {self.best_epoch + 1}")

# === Callback Runner ===

def run_cbs(cbs, method_nm, learn = None):
    """
    Runs a specific callback method across all callbacks in order.

    Args:
        cbs (list): List of callbacks.
        method_nm (str): The callback method to call.
        learn (Learner): Instance of learner framework.
    """
    for cb in sorted(cbs, key=attrgetter('order')):
        method = getattr(cb, method_nm, None)
        if method is not None:
            method(learn)
