"""Common utility functions."""

import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def init_random_seed(seed: int, deterministic: bool = False):
    """Initialize random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic mode for CUDA (may impact performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        gpu_id: Specific GPU ID to use. If None, uses GPU 0 if available.
        
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    additional_info: Optional[Dict] = None
):
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current loss value
        filepath: Path to save checkpoint
        additional_info: Additional information to save in checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load checkpoint to
        
    Returns:
        Dictionary containing checkpoint information
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint