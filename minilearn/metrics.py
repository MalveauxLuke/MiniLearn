from copy import copy
from torcheval.metrics import Mean
from .utils import to_cpu

class MetricsCB():
    """
    Callback for computing and logging performance metrics during training and validation.

    Accepts any number of torcheval metric objects and logs them after each epoch.

    Attributes:
        metrics (dict): User-provided metrics keyed by class name.
        all_metrics (dict): All metrics.
        loss (Mean): Mean object to track loss per batch.
    """
    order = 0 # Run early in callback list
        
    def __init__(self, *ms):
        """
        Initializes the metric callback wit torcheval Metric instances.

        Args:
            *ms: Torcheval metric(s)
        """
        self.metrics = {type(o).__name__: o for o in ms}
        # Create a copy to avoid modifying user-passed metric instances
        self.all_metrics = copy(self.metrics)
        self.all_metrics['loss'] = self.loss = Mean()
        
    def before_fit(self, learn):
        """
        Adds the metrics callback to the learner so it can be accessed externally.
        """
        learn.metrics = self
        
    def before_epoch(self, learn):
        """
        Resets all metrics before every epoch.
        """
        for o in self.all_metrics.values():
            o.reset()
            
    def after_epoch(self, learn):
        """
        Computes and logs all metrics after each epoch.

        Outputs a dictionary including each metrics computed value, epoch number, and phase.
        """
        log = {k:f'{v.compute():.4f}' for k,v in self.all_metrics.items()}
        log['epoch'] = learn.epoch +1
        if learn.model.training:
            log['train'] = 'train'
        else:
            log['train'] = 'eval'
        print(log)
    def after_batch(self, learn):
        """
        Updates metrics and loss tracker after each batch.

        Moves batch and predictions to CPU to reduce GPU use, updates each metric with predictions and 
        targets, and updates running loss with batch loss.
        """
        x,y,*_ = to_cpu(learn.batch)
        
        for m in self.metrics.values():
            m.update(to_cpu(learn.preds), y)
            
        self.loss.update(to_cpu(learn.loss), weight=len(x))
    