"""Example script for training SimpleCNN on route choice data."""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.simple_cnn import SimpleCNN
from data.dataset import RouteChoiceDataset
from utils.common import init_random_seed, count_parameters, get_device
from utils.training import train_model


def main(args):
    """Main training function."""
    # Set random seed
    init_random_seed(args.seed)
    
    # Get device
    device = get_device(args.gpu_id)
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    dataset = RouteChoiceDataset(
        args.data_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=args.seed
    )
    
    # Print dataset statistics
    stats = dataset.get_data_stats()
    print(f"\nDataset statistics:")
    print(f"Total samples: {stats['n_samples']}")
    print(f"Train samples: {stats['n_train']}")
    print(f"Val samples: {stats['n_val']}")
    print(f"Test samples: {stats['n_test']}")
    print(f"Number of features: {stats['n_features']}")
    print(f"Maximum routes per choice: {stats['max_routes']}")
    
    # Create data loaders
    train_loader = DataLoader(
        dataset.get_train_dataset(),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        dataset.get_val_dataset(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        dataset.get_test_dataset(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    print(f"\nCreating SimpleCNN model...")
    model = SimpleCNN(
        num_features=stats['n_features'],
        use_bias=args.use_bias
    ).to(device)
    print(f"Model parameters: {count_parameters(model)}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    print(f"\nTraining for {args.epochs} epochs...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        save_path=args.save_path,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    from utils.training import validate
    test_loss, test_acc = validate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Print learned weights
    weights = model.get_feature_weights()
    print(f"\nLearned feature weights:")
    for i, w in enumerate(weights):
        print(f"  Feature {i}: {w:.4f}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimpleCNN on route choice data")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                      help="Path to route choice data pickle file")
    
    # Model arguments
    parser.add_argument("--use_bias", action="store_true",
                      help="Use bias in convolutional layer")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=512,
                      help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Learning rate")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                      help="Patience for early stopping")
    
    # Other arguments
    parser.add_argument("--save_path", type=str, default="best_model.pt",
                      help="Path to save best model")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--gpu_id", type=int, default=None,
                      help="GPU ID to use (None for automatic selection)")
    parser.add_argument("--num_workers", type=int, default=0,
                      help="Number of data loading workers")
    
    args = parser.parse_args()
    main(args)