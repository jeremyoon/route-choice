"""Metrics for evaluating route choice models."""

import torch
import torch.nn.functional as F


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate classification accuracy.
    
    Args:
        predictions: Predicted class indices
        targets: True class indices
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def calculate_log_likelihood(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate average log-likelihood.
    
    Args:
        logits: Model output logits
        targets: True class indices
        
    Returns:
        Average log-likelihood
    """
    log_probs = F.log_softmax(logits, dim=1)
    log_likelihood = -F.nll_loss(log_probs, targets, reduction='mean')
    return log_likelihood.item()


def calculate_rho_squared(
    log_likelihood: float, 
    log_likelihood_null: float,
    num_parameters: int = 0
) -> float:
    """Calculate McFadden's rho-squared (with optional adjustment).
    
    Args:
        log_likelihood: Log-likelihood of the estimated model
        log_likelihood_null: Log-likelihood of the null model
        num_parameters: Number of parameters (for adjusted rho-squared)
        
    Returns:
        Rho-squared value
    """
    if log_likelihood_null == 0:
        return 0.0
    
    rho_squared = 1 - (log_likelihood / log_likelihood_null)
    
    if num_parameters > 0:
        # Adjusted rho-squared
        rho_squared = 1 - ((log_likelihood - num_parameters) / log_likelihood_null)
    
    return rho_squared