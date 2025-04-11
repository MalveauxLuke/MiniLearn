import math
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

from .utils import to_cpu
from .callbacks import CancelFitException, CancelEpochException

class LRFinderCB():
    """
    A callback for performing learning rate range test using exponential scheduling.

    Gradually increases the learning rate after every batch and monitors the loss.
    Stops training once the loss diverges or exceeds a multiple of lowest recorded loss.

    Attributes:
        gamma (float): Learning rate multiplier to be applied after each batch.
        max_mult (float): Multiplier above the minimum loss that triggers early stop.
        sched (ExponentialLR): PyTorch learning rate scheduler.
        lrs (list): List of learning rates.
        losses (list): Recorded losses.
        min (float): Minimum loss observed so far.
    """
    order = 0 # Run early in callback roder
    
    def __init__(self, gamma=1.3, max_mult=3): 
        """
        Initializes the learning rate finder.

        Args:
            gamma (float): LR multiplier applied after each batch.
            max_mult (float): Max allowable multiple of min loss before stopping.
        """
        self.gamma = gamma
        self.max_mult = max_mult
        
    def before_fit(self, learn):
        """
        Called before training begins. Initializes scheduler and begins record keeping.
        """
        self.sched =  ExponentialLR(learn.opt, self.gamma)
        self.lrs,self.losses = [],[]
        self.min = math.inf
        
    def after_batch(self,learn):
        """
        Called after every batch during training

        Records current LR and loss and checks for stopping conditions.
        """
        if not learn.model.training:
            # Only need to adjust learning rate and collect stats if training
            raise CancelEpochException()
        # Record current learning rate and loss
        self.lrs.append(learn.opt.param_groups[0]['lr'])
        loss = to_cpu(learn.loss)
        self.losses.append(loss)

        #Update minimum loss
        if loss < self.min:
            self.min = loss

        # Cancel if any of the conditions are met. Loss spikes or becomes NaN
        if math.isnan(loss) or (loss > self.min*self.max_mult):
            raise CancelFitException() 
        # Step scheduler
        self.sched.step()
        
    def cleanup_fit(self, learn):
        """
        Called at the end of training to plot the LR vs loss curve.
        """           
        plt.plot(self.lrs, self.losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.title("Learning Rate Finder")
        plt.grid(True)