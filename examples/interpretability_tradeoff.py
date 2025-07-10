"""Example demonstrating the interpretability-accuracy tradeoff in DeepLogit models."""

import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models import SimpleCNN, TransformerConstrained, TransformerUnconstrained, TransformerSeparated
from data.dataset import RouteChoiceDataset
from utils.common import init_random_seed, count_parameters, get_device
from utils.training import train_model


def compare_models(data_path: str, num_epochs: int = 20):
    """Compare different models on the interpretability-accuracy spectrum.
    
    Args:
        data_path: Path to route choice data
        num_epochs: Number of training epochs
    """
    # Set random seed
    init_random_seed(42)
    device = get_device()
    
    # Load data
    print("Loading data...")
    dataset = RouteChoiceDataset(data_path)
    stats = dataset.get_data_stats()
    
    print(f"Dataset info:")
    print(f"  Total samples: {stats['n_samples']}")
    print(f"  Features: {stats['n_features']}")
    print(f"  Max routes: {stats['max_routes']}")
    
    # Create data loaders
    train_loader = DataLoader(dataset.get_train_dataset(), batch_size=256, shuffle=True)
    val_loader = DataLoader(dataset.get_val_dataset(), batch_size=256, shuffle=False)
    test_loader = DataLoader(dataset.get_test_dataset(), batch_size=256, shuffle=False)
    
    # Initialize results storage
    results = {}
    
    # 1. Train SimpleCNN (baseline)
    print("\n" + "="*50)
    print("Training SimpleCNN (MNL equivalent)...")
    print("="*50)
    
    simple_cnn = SimpleCNN(num_features=4).to(device)
    optimizer = optim.Adam(simple_cnn.parameters(), lr=0.001)
    
    history_simple = train_model(
        simple_cnn, train_loader, val_loader, optimizer, 
        device, num_epochs=num_epochs
    )
    
    # Get interpretable weights
    simple_weights = simple_cnn.get_feature_weights()
    print(f"\nSimpleCNN learned weights:")
    feature_names = ['Travel Time', 'Cost', 'Walk Time', 'Num Transfers']
    for i, (name, weight) in enumerate(zip(feature_names, simple_weights)):
        print(f"  {name}: {weight:.4f}")
    
    results['SimpleCNN'] = {
        'model': simple_cnn,
        'history': history_simple,
        'weights': simple_weights,
        'params': count_parameters(simple_cnn)
    }
    
    # 2. Train TransformerConstrained
    print("\n" + "="*50)
    print("Training TransformerConstrained...")
    print("="*50)
    
    # Prepare constrained weights tensor
    constrained_weights = torch.zeros(1, 1, 1, stats['n_features'])
    constrained_weights[0, 0, 0, :4] = simple_weights
    
    transformer_const = TransformerConstrained(
        num_features=stats['n_features'],
        num_constrained_features=4,
        pretrained_weights=constrained_weights
    ).to(device)
    
    optimizer = optim.Adam(transformer_const.parameters(), lr=0.001)
    
    # Custom training loop that enforces constraints
    history_const = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Train with constraint enforcement
        transformer_const.train()
        for batch in train_loader:
            routes = batch['route'].to(device)
            choices = batch['choice'].to(device)
            
            optimizer.zero_grad()
            logits = transformer_const(routes)
            loss = torch.nn.functional.cross_entropy(logits, choices)
            loss.backward()
            optimizer.step()
            
            # Enforce constraints after each step
            transformer_const.enforce_constraints(constrained_weights)
        
        # Validation
        transformer_const.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                routes = batch['route'].to(device)
                choices = batch['choice'].to(device)
                logits = transformer_const(routes)
                pred = logits.argmax(dim=1)
                val_correct += (pred == choices).sum().item()
                val_total += choices.size(0)
        
        val_acc = val_correct / val_total
        history_const['val_acc'].append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")
    
    # Verify constraints are maintained
    const_weights = transformer_const.get_constrained_weights()
    print(f"\nConstrained weights (should match SimpleCNN):")
    for i, (name, weight) in enumerate(zip(feature_names, const_weights)):
        print(f"  {name}: {weight:.4f}")
    
    results['TransformerConstrained'] = {
        'model': transformer_const,
        'history': history_const,
        'weights': const_weights,
        'params': count_parameters(transformer_const)
    }
    
    # 3. Train TransformerUnconstrained
    print("\n" + "="*50)
    print("Training TransformerUnconstrained...")
    print("="*50)
    
    transformer_unconst = TransformerUnconstrained(
        num_features=stats['n_features']
    ).to(device)
    
    optimizer = optim.Adam(transformer_unconst.parameters(), lr=0.001)
    
    history_unconst = train_model(
        transformer_unconst, train_loader, val_loader, optimizer,
        device, num_epochs=num_epochs
    )
    
    results['TransformerUnconstrained'] = {
        'model': transformer_unconst,
        'history': history_unconst,
        'params': count_parameters(transformer_unconst)
    }
    
    # 4. Train TransformerSeparated
    print("\n" + "="*50)
    print("Training TransformerSeparated...")
    print("="*50)
    
    transformer_sep = TransformerSeparated(
        num_features=stats['n_features'],
        num_interpretable=4
    ).to(device)
    
    optimizer = optim.Adam(transformer_sep.parameters(), lr=0.001)
    
    history_sep = train_model(
        transformer_sep, train_loader, val_loader, optimizer,
        device, num_epochs=num_epochs
    )
    
    sep_weights = transformer_sep.get_interpretable_weights()
    print(f"\nSeparated model interpretable weights:")
    for i, (name, weight) in enumerate(zip(feature_names, sep_weights)):
        print(f"  {name}: {weight:.4f}")
    
    results['TransformerSeparated'] = {
        'model': transformer_sep,
        'history': history_sep,
        'weights': sep_weights,
        'params': count_parameters(transformer_sep)
    }
    
    # Evaluate all models on test set
    print("\n" + "="*50)
    print("Final Test Set Evaluation")
    print("="*50)
    
    from utils.training import validate
    
    for model_name, model_info in results.items():
        test_loss, test_acc = validate(model_info['model'], test_loader, device)
        results[model_name]['test_acc'] = test_acc
        results[model_name]['test_loss'] = test_loss
        
        print(f"\n{model_name}:")
        print(f"  Parameters: {model_info['params']:,}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
    
    # Plot comparison
    plot_comparison(results)
    
    return results


def plot_comparison(results):
    """Plot model comparison results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    models = list(results.keys())
    accuracies = [results[m]['test_acc'] for m in models]
    params = [results[m]['params'] for m in models]
    
    colors = ['blue', 'green', 'red', 'orange']
    ax1.bar(models, accuracies, color=colors)
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(min(accuracies) * 0.95, max(accuracies) * 1.02)
    
    # Add value labels
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        ax1.text(i, acc + 0.002, f'{acc:.3f}', ha='center')
    
    # Parameters vs Accuracy
    ax2.scatter(params, accuracies, c=colors, s=100)
    for i, model in enumerate(models):
        ax2.annotate(model, (params[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Number of Parameters')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Model Complexity vs Accuracy')
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('interpretability_tradeoff.png', dpi=150)
    plt.show()
    
    # Create interpretability-accuracy plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define interpretability scores (subjective but meaningful)
    interpretability = {
        'SimpleCNN': 1.0,
        'TransformerConstrained': 0.8,
        'TransformerSeparated': 0.6,
        'TransformerUnconstrained': 0.2
    }
    
    interp_scores = [interpretability[m] for m in models]
    
    ax.scatter(interp_scores, accuracies, c=colors, s=200, alpha=0.7)
    
    for i, model in enumerate(models):
        ax.annotate(model, (interp_scores[i], accuracies[i]),
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Interpretability Score')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Interpretability-Accuracy Tradeoff')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('interpretability_accuracy_tradeoff.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DeepLogit models")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to route choice data")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    
    args = parser.parse_args()
    results = compare_models(args.data_path, args.epochs)