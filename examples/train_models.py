"""Example script for training DeepLogit models."""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models import CNN1, CNN2U, CNN2S, CNN2C, TFMU, TFMC
from data.dataset import RouteChoiceDataset
from utils.common import init_random_seed, get_device
from utils.training import train_model


def main():
    # Setup
    init_random_seed(42)
    device = get_device()
    
    # Load data (example - adjust path)
    dataset = RouteChoiceDataset("path/to/data.pkl")
    train_loader = DataLoader(dataset.get_train_dataset(), batch_size=512, shuffle=True)
    val_loader = DataLoader(dataset.get_val_dataset(), batch_size=512, shuffle=False)
    
    # Step 1: Train CNN1 with 4 features to get interpretable parameters
    print("Training CNN1 (4 features)...")
    cnn1_4 = CNN1(num_features=4).to(device)
    optimizer = optim.Adam(cnn1_4.parameters(), lr=0.001)
    train_model(cnn1_4, train_loader, val_loader, optimizer, device, num_epochs=50)
    
    # Extract learned weights
    base_weights = cnn1_4.get_feature_weights()
    print(f"Learned weights: {base_weights}")
    
    # Step 2: Train constrained models with 97 features
    print("\nTraining CNN2C (97 features, constrained)...")
    cnn2c = CNN2C(num_features=97, cnn1_weights=base_weights).to(device)
    optimizer = optim.Adam(cnn2c.parameters(), lr=0.001)
    
    # Custom training loop that maintains constraints
    for epoch in range(50):
        cnn2c.train()
        for batch in train_loader:
            routes = batch['route'].to(device)
            choices = batch['choice'].to(device)
            
            optimizer.zero_grad()
            logits = cnn2c(routes)
            loss = torch.nn.functional.cross_entropy(logits, choices)
            loss.backward()
            optimizer.step()
            
            # Re-apply constraints after each update
            cnn2c.apply_constraint(base_weights)
    
    print("\nTraining TFMC (97 features, constrained transformer)...")
    tfmc = TFMC(num_features=97, cnn1_weights=base_weights).to(device)
    optimizer = optim.Adam(tfmc.parameters(), lr=0.001)
    
    # Similar training with constraints
    for epoch in range(50):
        tfmc.train()
        for batch in train_loader:
            routes = batch['route'].to(device)
            choices = batch['choice'].to(device)
            
            optimizer.zero_grad()
            logits = tfmc(routes)
            loss = torch.nn.functional.cross_entropy(logits, choices)
            loss.backward()
            optimizer.step()
            
            # Re-apply constraints
            tfmc.apply_constraint(base_weights)
    
    print("Training complete!")


if __name__ == "__main__":
    main()