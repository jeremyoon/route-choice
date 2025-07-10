"""CNN1 model - Linear CNN equivalent to multinomial logit."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1(nn.Module):
    """Simple 1D CNN for route choice modeling.
    
    This model is mathematically equivalent to a multinomial logit model.
    It uses a 1D convolution with kernel size equal to the number of features,
    effectively performing a linear transformation across features for each route.
    
    Args:
        num_features: Number of features per route
        use_bias: Whether to use bias in the convolutional layer (default: False)
    
    Example:
        >>> model = SimpleCNN(num_features=4)
        >>> # Input: (batch_size, num_routes, num_features, 1)
        >>> x = torch.randn(32, 6, 4, 1)
        >>> logits = model(x)  # Output: (32, 6)
    """
    
    def __init__(self, num_features: int, use_bias: bool = False):
        super(CNN1, self).__init__()
        self.num_features = num_features
        
        # 1D convolution that acts across features
        # Input channels = 1, Output channels = 1
        # Kernel size = num_features (covers all features)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, num_features),
            padding=0,
            bias=use_bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, num_routes, num_features, 1)
            
        Returns:
            Logits tensor of shape (batch_size, num_routes)
        """
        batch_size, num_routes, num_features, _ = x.size()
        
        # Reshape to add channel dimension: (batch, 1, routes, features)
        x = x.view(batch_size, 1, num_routes, num_features)
        
        # Apply convolution: (batch, 1, routes, 1)
        x = self.conv(x)
        
        # Remove extra dimensions: (batch, routes)
        x = x.squeeze(1).squeeze(-1)
        
        return x
    
    def get_choice_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get choice probabilities using softmax.
        
        Args:
            x: Input tensor of shape (batch_size, num_routes, num_features, 1)
            
        Returns:
            Probability tensor of shape (batch_size, num_routes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the most likely route choice.
        
        Args:
            x: Input tensor of shape (batch_size, num_routes, num_features, 1)
            
        Returns:
            Predicted route indices of shape (batch_size,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def get_feature_weights(self) -> torch.Tensor:
        """Get the learned feature weights.
        
        Returns:
            Feature weights tensor of shape (num_features,)
        """
        return self.conv.weight.squeeze().detach()
    
    def compute_utilities(self, x: torch.Tensor) -> torch.Tensor:
        """Compute utilities for each route (before softmax).
        
        This is equivalent to the systematic utility in discrete choice models.
        
        Args:
            x: Input tensor of shape (batch_size, num_routes, num_features, 1)
            
        Returns:
            Utilities tensor of shape (batch_size, num_routes)
        """
        return self.forward(x)