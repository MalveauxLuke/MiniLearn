import torch
from functools import partial
from .utils import to_cpu

class Hook():
    """
    Wrapper for PyTorch's hooks that stores output statistics. 

    Attributes:
        hook (RemovableHandle): The handle to the registered forward hook.
    """
    def __init__(self, m, f):
        """
        Registers a forward hook on the current module.

        Args:
            m (torch.nn.Module): The module to hook into.
            f (function): The hook function, partially applied with the hook instance.
        """
        self.hook = m.register_forward_hook(partial(f, self)) 
    def remove(self): 
        """remove forward hook."""
        self.hook.remove()
    def __del__(self): 
        """Ensure the hook is removed when garbage collection occurs."""        
        self.remove()
        
def append_stats(hook, mod, inp, outp):
    """
    Hook function to get output statistics during forward pass.

    The function tracks the mean of activations, the standard deviation, and a histogram of absolute values 
    (40 bins from 0 to 10)

    Args:
        hook (Hook): The Hook instance for storing stats.
        mod (torch.nn.Module): The hooked module (unused here).
        inp (tuple): Input to the module (unused here).
        outp (torch.Tensor): Output of the module.
    """
    if not hasattr(hook,'stats'): 
        hook.stats = ([],[],[])
    acts = to_cpu(outp)
    hook.stats[0].append(acts.mean())
    hook.stats[1].append(acts.std())
    hook.stats[2].append(acts.abs().histc(40,0,10))

class Hooks(list):
    """
    A container that inherits from list and is sued for managing multiple forward hooks on a model.

    The class registers a forward hook on each module using the given hookfunction, 
    automatically removes hooks when exiting and supports standard list operations.
    """
    def __init__(self, ms, f):
        """
        Registers hooks on a list of modules.

        Args:
            ms (list): List of torch.nn.Module.
            f (function): Hook function to apply to each module.
        """
        super().__init__([Hook(m,f) for m in ms])

    def __enter__(self, *args): 
        """Allows use in a 'with' statement."""
        return self
        
    def __exit__ (self, *args):
        """Ensures hooks are removed when exiting context."""
        self.remove()
        
    def __delitem__(self, i):
        """Removes a specific hook and deletes from list"""
        self[i].remove()
        super().__delitem__(i)
        
    def remove(self):
        """Removes all hooks from modules."""
        for h in self: 
            h.remove()
        