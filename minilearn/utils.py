import torch

def to_cpu(x):
    """
    Moves a tensor or other structure to CPU:

    Supports:
        - Single tensors
        - Lists of tensors
        - Tuples of tensors

    Args:
        x (Any): Tensor or collection of tensors.
    Returns:
        The same structure with all tensors on CPU.
    """
    if isinstance(x, list): 
        return [to_cpu(o) for o in x]
    if isinstance(x, tuple): 
        return tuple(to_cpu(list(x)))
        
    res = x.detach().cpu()
    return res.float() if res.dtype==torch.float16 else res
    
def to_device(data, device): 
    """
    Recursively moves a tensor to the specified device.

    Supports: 
        - Single tensors
        - Lists, tuples, and dicts containing tensors

    Args: 
        data (Any): Tensor or structure containing tensors.
        device (torch.device): Target device.

    Returns:
        The structure with all tensors moved to the specified device.
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
        
    return data.to(device)