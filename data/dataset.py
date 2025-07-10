"""Dataset classes for DeepLogit models."""

import torch
from torch.utils.data import Dataset, Sampler
from typing import Iterator, List
from collections import defaultdict


class CustomDataset(Dataset):
    """Custom dataset for route choice data.
    
    This dataset handles the specific format of the route choice data where
    each sample contains multiple routes and a choice indicator.
    """
    
    def __init__(self, data):
        """Initialize the dataset.
        
        Args:
            data: List of tuples (routes, choices) where each element contains
                  multiple samples
        """
        self.routes = dict()
        self.choice = dict()
        
        counter = 0
        for i in range(len(data)):
            for j in range(len(data[i][0])):
                self.routes[counter] = data[i][0][j]
                self.choice[counter] = data[i][1][j]
                counter += 1
            
    def __len__(self):
        return len(self.choice)
        
    def __getitem__(self, idx):
        """Get a single sample.
        
        Returns:
            tuple: (routes, choice) where:
                - routes: tensor of shape (num_routes, num_features, 1)
                - choice: scalar tensor indicating chosen route
        """
        choice = self.choice[idx]
        routes = self.routes[idx]
        return routes, choice


class WeightedBucketSampler(Sampler[List[int]]):
    """Weighted bucket sampler for handling different choice set sizes.
    
    This sampler groups samples by the number of available routes and samples
    from these groups according to specified weights.
    """
    
    def __init__(self, data, weights, num_samples: int,
                 replacement: bool = True, generator=None, 
                 shuffle=True, batch_size=32, drop_last=False):
        super().__init__(data)
        
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.batch_size = batch_size
        self.buckets = defaultdict(list)
        
        # Group data by number of routes (bucket index = num_routes)
        counter = 0
        for i in range(len(data)):
            # Assuming bucket index is based on number of routes
            # Adjust this logic based on your specific bucketing strategy
            self.buckets[i+2] += [data[i][0], data[i][1]]
            counter += len(data[i][0])
        self.length = counter    

    def __iter__(self) -> Iterator[int]:
        # Choose a bucket depending on the weighted sample
        rand_bucket = torch.multinomial(self.weights, self.num_samples, 
                                       self.replacement, generator=self.generator)   
        
        batch = [0] * self.batch_size
        idx_in_batch = 0
        
        for bucket_idx in rand_bucket.tolist():
            bucketsample_count = 0
            shifter = sum([len(self.buckets[i+2][0]) for i in range(bucket_idx)])
            
            # Generate random indices from the bucket and shift them
            rand_tensor = torch.randperm(len(self.buckets[bucket_idx+2][0]), 
                                        generator=self.generator)
            
            for idx in rand_tensor.tolist():
                batch[idx_in_batch] = idx + shifter
                idx_in_batch += 1
                
                if idx_in_batch == self.batch_size:
                    bucketsample_count += self.batch_size
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
                    
            if idx_in_batch > 0 and not self.drop_last:
                bucketsample_count += idx_in_batch
                yield batch[:idx_in_batch]
                idx_in_batch = 0
                batch = [0] * self.batch_size
        
    def __len__(self):
        return (self.length + (self.batch_size - 1)) // self.batch_size