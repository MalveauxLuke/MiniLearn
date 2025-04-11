import torch
import torch.nn.functional as F
from torch import optim
from functools import partial
import copy
from .callbacks import run_cbs, CancelFitException, CancelBatchException, CancelEpochException


class with_cbs:
    """

    Decorator class for wrapping Learner methods with callbacks

    Executes, 'before_<phase>', the original method, and after<phase>', 
    while ensuring 'cleanup<phase>' runs in any case. Supports control flow
    through phase specific exceptions (e.g., CancelFitException).
    """
    def __init__(self, nm): self.nm = nm
    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                 # Trigger callbacks before phase
                 o.callback(f'before_{self.nm}')
                 # Run the method 
                 f(o, *args, **kwargs)
                 # Trigger callbacks after phase
                 o.callback(f'after_{self.nm}')
            except globals()[f'Cancel{self.nm.title()}Exception']: 
                # If a callback cancels the phase, skip 
                pass
            finally: 
                # Always run cleanup even if phase skipped
                o.callback(f'cleanup_{self.nm}')
        return _f
        
class Learner:
    """
    Core class that manages the models training, data, loss, optomizer and callbacks.

    Supports a flexible training loop via callbacks before every phase.
    """
    def __init__(self, model, tdl, vdl, loss_func= F.mse_loss , lr = 0.1 ,cbs= None, opt_func = optim.SGD):
        """
        Initializes the Learner.

        Args:
            model (torch.nn.Module): The model to train.
            tdl (DataLoader): Training dataloader.
            vdl (DataLoader): Validation dataloader.
            loss_func (callable): Loss function to use.
            lr (float): Learning rate.
            cbs (list): List of callbacks.
            opt_func (callable): Optimizer to use (default: SGD).
            self.init_weights 
        Attributes:
            init_weights (dict): A copy of the model's initial weights for reset purposes.
        """
        if cbs is None: 
            cbs = []
        cbs = list(cbs)
        self.cbs = cbs
        self.model = model
        self.tdl = tdl
        self.vdl = vdl
        self.loss_func = loss_func
        self.lr = lr
        self.opt_func = opt_func
        self.init_weights = copy.deepcopy(model.state_dict())
        
    def fit(self, n_epochs=1, train=True, valid=True, cbs=None, lr=None, reset = False, reset_exact=False):
        """
        Method to start training and/or validation

        Adds any extra callbacks, optionally resets the weights, sets up the optomizer, sets the number of epochs,
        and runs the internal _fit loop.

        Args:
            n_epochs (int): Number of training epochs.
            train (bool): Whether to run the training loop.
            valid (bool): Whether to run validation loop.
            cbs (list): Additional callbacks to use temporarily.
            lr (float): Optional learning rate change.
            reset (bool): Optional reset weights randomly.
            reset_exact (bool): Optionally resets weights exactly as it had at init.
        """
        if cbs is None: 
            cbs = []
        cbs = list(cbs)
        for cb in cbs:
            self.cbs.append(cb)

        if reset:
            if reset_exact:
                if hasattr(self, 'init_weights'):
                    # Restore the model to its original initialization
                    self.model.load_state_dict(self.init_weights)
                else:
                    raise ValueError("Exact reset requested, but 'init_weights' not found. Store it in __init__ or manually.")
            else:
                # Random reinitialization using reset_parameters
                self._reset_model_weights()
             
        try:
            self.n_epochs = n_epochs
            if lr is None: 
                lr = self.lr
            self.opt = self.opt_func(self.model.parameters(), lr)
            self._fit(train, valid) 
        finally:
            # Remove temporary callbacks regardless of if training finishes crashes or skips
            for cb in cbs:
                self.cbs.remove(cb)  
                 
    @with_cbs('fit')
    def _fit(self, train, valid):
        """
        Internal training loop wrapped in with callbacks

        Decorate _fit (not the outer fit method) so that the callback cycle 
        (before_fit/fater_fit) surroudns only the training process. This ensures
        that the setup in 'fit' runs regardless of any exceptions thrown to skip 
        training. 
        """
        for self.epoch in range(self.n_epochs):
            # Run training loop if train enabled
            if train: 
                self.one_epoch(True)
            # Run valid loop if validation enabled
            if valid: 
                with torch.no_grad():
                    self.one_epoch(False)
                    
    def one_epoch(self, train):
        """
        Sets the model mode, assigns appropriate dataloader, and runs one epoch
        through internal method '_one_epoch()'.
        
        Args:
            train (bool): Whether to run training or validation phase.
        """
        if train:
            self.model.train()
            self.dl = self.tdl
        else:
            self.model.eval()
            self.dl = self.vdl
        self._one_epoch()
    
    @with_cbs('epoch')
    def _one_epoch(self):
        """
        Runs one complete pass over the current DataLoader.

        Each batch is sent through '_one_batch()' and wrapped in before_batch/after_batch callbacks.
        """
        for self.num, self.batch in enumerate(self.dl):
            self._one_batch()
    
    
    @with_cbs('batch')
    def _one_batch(self):
        """
        Computes a single batch.

        Goes through forward pass, loss calculation, and backwards pass if in training mode.
        """
        self.preds = self.model(self.batch[0])
        self.loss = self.loss_func(self.preds, self.batch[1])
        if self.model.training:
            self.loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            
    def callback(self, method_nm): 
        """
        Executes the specified callback method on all active callbacks.

        Args:
            method_nm (str): Name of the method to call.
        """
        run_cbs(self.cbs, method_nm, self)