"""Training utilities for DeepLogit models."""

from typing import Optional, Dict, Any, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 100
) -> Tuple[float, float]:
    """Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to run on
        epoch: Current epoch number
        log_interval: How often to log progress
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        routes = batch['route'].to(device)
        choices = batch['choice'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(routes)
        
        # Compute loss
        loss = F.cross_entropy(logits, choices)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == choices).sum().item()
        total += choices.size(0)
        
        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Validate model on validation set.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device to run on
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            routes = batch['route'].to(device)
            choices = batch['choice'].to(device)
            
            # Forward pass
            logits = model(routes)
            
            # Compute loss
            loss = F.cross_entropy(logits, choices)
            
            # Track metrics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == choices).sum().item()
            total += choices.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    save_path: Optional[str] = None,
    early_stopping_patience: int = 5
) -> Dict[str, Any]:
    """Full training loop with validation and early stopping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to run on
        num_epochs: Number of epochs to train
        save_path: Path to save best model (optional)
        early_stopping_patience: Patience for early stopping
        
    Returns:
        Dictionary containing training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_epoch': 0
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history['best_epoch'] = epoch
            patience_counter = 0
            
            # Save best model
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, save_path)
                print(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    return history