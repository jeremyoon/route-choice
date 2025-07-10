"""Dataset classes for route choice modeling."""

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """PyTorch Dataset for route choice data.
    
    Args:
        routes: Tensor of shape (n_samples, n_routes, n_features, 1)
        labels: Tensor of shape (n_samples,) containing chosen route indices
    """
    
    def __init__(self, routes: torch.Tensor, labels: torch.Tensor):
        self.routes = routes
        self.choice = labels

    def __len__(self) -> int:
        return len(self.choice)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "route": self.routes[idx], 
            "choice": self.choice[idx]
        }


class RouteChoiceDataset:
    """Main dataset class for loading and splitting route choice data.
    
    This class handles loading route choice data from pickle files,
    creating train/validation/test splits, and providing PyTorch datasets.
    
    Args:
        data_path: Path to the pickle file containing route choice data
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        random_state: Random seed for reproducibility (default: 42)
    """
    
    def __init__(
        self,
        data_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Train, validation and test ratios must sum to 1"
        
        self.data_path = Path(data_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Load data
        self._load_data()
        
        # Create splits
        self._create_splits()
    
    def _load_data(self):
        """Load route choice data from pickle file."""
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract routes and choices
        # Expected format: data contains route features and choice labels
        if isinstance(data, dict):
            self.routes = data.get('routes', data.get('X', None))
            self.choices = data.get('choices', data.get('y', None))
        elif isinstance(data, tuple) and len(data) == 2:
            self.routes, self.choices = data
        else:
            raise ValueError("Unsupported data format in pickle file")
        
        # Convert to tensors if needed
        if not isinstance(self.routes, torch.Tensor):
            self.routes = torch.tensor(self.routes, dtype=torch.float32)
        if not isinstance(self.choices, torch.Tensor):
            self.choices = torch.tensor(self.choices, dtype=torch.long)
    
    def _create_splits(self):
        """Create train/validation/test splits."""
        n_samples = len(self.choices)
        indices = np.arange(n_samples)
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_ratio,
            random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=self.random_state
        )
        
        # Store indices
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
    
    def get_train_dataset(self) -> CustomDataset:
        """Get training dataset."""
        return CustomDataset(
            self.routes[self.train_idx],
            self.choices[self.train_idx]
        )
    
    def get_val_dataset(self) -> CustomDataset:
        """Get validation dataset."""
        return CustomDataset(
            self.routes[self.val_idx],
            self.choices[self.val_idx]
        )
    
    def get_test_dataset(self) -> CustomDataset:
        """Get test dataset."""
        return CustomDataset(
            self.routes[self.test_idx],
            self.choices[self.test_idx]
        )
    
    def get_feature_dim(self) -> int:
        """Get number of features per route."""
        return self.routes.shape[2]
    
    def get_num_routes(self) -> int:
        """Get maximum number of routes per choice situation."""
        return self.routes.shape[1]
    
    def get_data_stats(self) -> Dict[str, any]:
        """Get dataset statistics."""
        return {
            "n_samples": len(self.choices),
            "n_train": len(self.train_idx),
            "n_val": len(self.val_idx),
            "n_test": len(self.test_idx),
            "n_features": self.get_feature_dim(),
            "max_routes": self.get_num_routes(),
            "choice_distribution": torch.bincount(self.choices).tolist()
        }