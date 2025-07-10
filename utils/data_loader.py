"""Data loading utilities with proper class resolution."""

import os
import sys
import torch
import pickle

# Ensure the CustomDataset can be found when unpickling
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import CustomDataset, WeightedBucketSampler

# Also add the original path for backward compatibility
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src/vit_tabtransformer'))
try:
    from dataset import CustomDataset as OriginalCustomDataset
except:
    OriginalCustomDataset = CustomDataset


class DataLoader:
    """Custom unpickler that properly resolves CustomDataset class."""
    
    @staticmethod
    def load_data(filepath):
        """Load pickled data with proper class resolution."""
        # Register both possible locations of CustomDataset
        import sys
        import dataset as original_dataset
        original_dataset.CustomDataset = CustomDataset
        sys.modules['dataset'] = original_dataset
        
        # Now load the data
        return torch.load(filepath, map_location='cpu')