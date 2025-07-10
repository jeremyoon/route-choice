"""Test script for CNN1 model."""

import sys
sys.path.insert(0, '/mnt/md0/route-choice/DeepLogit')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.cnn1 import CNN1
from data.dataset import CustomDataset
from utils.common import init_random_seed, count_parameters, get_device
from utils.training import train_epoch, validate


def create_synthetic_data(n_samples=1000, n_routes=6, n_features=4):
    """Create synthetic route choice data for testing."""
    # Create random route features
    routes = torch.randn(n_samples, n_routes, n_features, 1)
    
    # Create synthetic choices based on a simple utility function
    # Utility = sum of features with some weights
    weights = torch.tensor([0.5, -0.3, 0.2, 0.8])
    utilities = torch.zeros(n_samples, n_routes)
    
    for i in range(n_samples):
        for j in range(n_routes):
            utilities[i, j] = torch.sum(routes[i, j, :, 0] * weights)
    
    # Add some noise and select choices
    utilities += torch.randn_like(utilities) * 0.1
    choices = torch.argmax(utilities, dim=1)
    
    return routes, choices


def main():
    """Main test function."""
    print("Testing CNN1 model for route choice prediction")
    print("=" * 50)
    
    # Set random seed for reproducibility
    init_random_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create synthetic data
    print("\nCreating synthetic data...")
    train_routes, train_choices = create_synthetic_data(n_samples=800)
    val_routes, val_choices = create_synthetic_data(n_samples=200)
    
    # Create datasets and dataloaders
    train_dataset = CustomDataset(train_routes, train_choices)
    val_dataset = CustomDataset(val_routes, val_choices)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("\nCreating CNN1 model...")
    model = CNN1(num_features=4).to(device)
    print(f"Model parameters: {count_parameters(model)}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    sample_batch = next(iter(train_loader))
    sample_routes = sample_batch['route'].to(device)
    sample_choices = sample_batch['choice'].to(device)
    
    with torch.no_grad():
        logits = model(sample_routes)
        print(f"Input shape: {sample_routes.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Sample logits: {logits[0].cpu().numpy()}")
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train for a few epochs
    print("\nTraining for 5 epochs...")
    for epoch in range(1, 6):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, device)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.4f}")
    
    # Test model functions
    print("\nTesting model utility functions...")
    with torch.no_grad():
        # Get feature weights
        weights = model.get_feature_weights()
        print(f"Learned feature weights: {weights.cpu().numpy()}")
        
        # Get choice probabilities
        probs = model.get_choice_probabilities(sample_routes[:1])
        print(f"Choice probabilities for first sample: {probs[0].cpu().numpy()}")
        
        # Make predictions
        preds = model.predict(sample_routes)
        print(f"Predictions shape: {preds.shape}")
        print(f"Accuracy on batch: {(preds == sample_choices).float().mean():.4f}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()