"""CNN2 models with quadratic features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN2U(nn.Module):
    """CNN2 Unconstrained - includes quadratic features without constraints.
    
    Parameters depend on number of features:
    - 4 features: 4 + 10 = 14 parameters (4 linear + 4x5/2 quadratic)
    - 97 features: 97 + 4753 = 4850 parameters
    """
    
    def __init__(self, num_features: int):
        super(CNN2U, self).__init__()
        self.num_features = num_features
        
        # Calculate number of quadratic features (upper triangle including diagonal)
        num_quad = num_features * (num_features + 1) // 2
        total_features = num_features + num_quad
        
        # Single convolution layer
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, total_features),
            padding=0,
            bias=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quadratic feature expansion."""
        batch_size, num_routes, num_features, _ = x.size()
        
        # Extract features
        x_features = x.view(batch_size * num_routes, num_features, 1)
        
        # Create quadratic features
        x_quad = torch.matmul(x_features, x_features.transpose(1, 2))
        
        # Extract upper triangle (including diagonal)
        upper_idx = torch.triu_indices(num_features, num_features, offset=0)
        x_quad_upper = x_quad[:, upper_idx[0], upper_idx[1]]
        
        # Combine linear and quadratic
        x_linear = x_features.view(batch_size * num_routes, num_features)
        x_combined = torch.cat([x_linear, x_quad_upper], dim=1)
        
        # Reshape for convolution
        x_combined = x_combined.view(batch_size, 1, num_routes, -1)
        
        # Apply convolution
        out = self.conv(x_combined)
        logits = out.squeeze(1).squeeze(-1)
        
        return logits


class CNN2S(nn.Module):
    """CNN2 Separated - quadratic features computed separately for interpretable and non-interpretable.
    
    For 97 features (4 interpretable + 93 non-interpretable):
    - Interpretable: 4 + 10 = 14 parameters
    - Non-interpretable: 93 + 4371 = 4464 parameters  
    - Total: 14 + 4464 = 4478 parameters
    
    Note: No cross-terms between interpretable and non-interpretable features.
    """
    
    def __init__(self, num_features: int, num_interpretable: int = 4):
        super(CNN2S, self).__init__()
        self.num_features = num_features
        self.num_interpretable = num_interpretable
        self.num_non_interpretable = num_features - num_interpretable
        
        # Calculate quadratic features for each group
        num_quad_interp = num_interpretable * (num_interpretable + 1) // 2
        num_quad_non_interp = self.num_non_interpretable * (self.num_non_interpretable + 1) // 2
        
        # Total features = linear + quadratic (no cross terms)
        total_features = num_interpretable + num_quad_interp + self.num_non_interpretable + num_quad_non_interp
        
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, total_features),
            padding=0,
            bias=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with separated quadratic features."""
        batch_size, num_routes, num_features, _ = x.size()
        
        # Separate features
        x_flat = x.view(batch_size * num_routes, num_features, 1)
        x_interp = x_flat[:, :self.num_interpretable, :]
        x_non_interp = x_flat[:, self.num_interpretable:, :]
        
        # Quadratic features for interpretable
        x_quad_interp = torch.matmul(x_interp, x_interp.transpose(1, 2))
        upper_idx_interp = torch.triu_indices(self.num_interpretable, self.num_interpretable, offset=0)
        x_quad_interp_upper = x_quad_interp[:, upper_idx_interp[0], upper_idx_interp[1]]
        
        # Quadratic features for non-interpretable
        x_quad_non_interp = torch.matmul(x_non_interp, x_non_interp.transpose(1, 2))
        upper_idx_non_interp = torch.triu_indices(self.num_non_interpretable, self.num_non_interpretable, offset=0)
        x_quad_non_interp_upper = x_quad_non_interp[:, upper_idx_non_interp[0], upper_idx_non_interp[1]]
        
        # Combine all features
        x_interp_linear = x_interp.view(batch_size * num_routes, self.num_interpretable)
        x_non_interp_linear = x_non_interp.view(batch_size * num_routes, self.num_non_interpretable)
        
        x_combined = torch.cat([
            x_interp_linear,
            x_quad_interp_upper,
            x_non_interp_linear,
            x_quad_non_interp_upper
        ], dim=1)
        
        # Reshape and convolve
        x_combined = x_combined.view(batch_size, 1, num_routes, -1)
        out = self.conv(x_combined)
        logits = out.squeeze(1).squeeze(-1)
        
        return logits


class CNN2C(nn.Module):
    """CNN2 Constrained - fixes weights for first 4 features from CNN1.
    
    Same architecture as CNN2S but with weight constraints on interpretable parameters.
    """
    
    def __init__(self, num_features: int, num_interpretable: int = 4, cnn1_weights: torch.Tensor = None):
        super(CNN2C, self).__init__()
        self.num_features = num_features
        self.num_interpretable = num_interpretable
        self.num_non_interpretable = num_features - num_interpretable
        self.cnn1_weights = cnn1_weights
        
        # Same architecture as CNN2S
        num_quad_interp = num_interpretable * (num_interpretable + 1) // 2
        num_quad_non_interp = self.num_non_interpretable * (self.num_non_interpretable + 1) // 2
        total_features = num_interpretable + num_quad_interp + self.num_non_interpretable + num_quad_non_interp
        
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, total_features),
            padding=0,
            bias=False
        )
        
        # Apply initial constraint if weights provided
        if cnn1_weights is not None:
            self.apply_constraint(cnn1_weights)
    
    def apply_constraint(self, cnn1_weights: torch.Tensor):
        """Fix the weights for interpretable features (first 4).
        
        Args:
            cnn1_weights: Weights from CNN1 model for the first 4 features
        """
        with torch.no_grad():
            # Set the linear weights for interpretable features
            self.conv.weight.data[0, 0, 0, :self.num_interpretable] = cnn1_weights.flatten()[:self.num_interpretable]
            # Ensure fare weight is clamped to -1
            if self.num_interpretable >= 2:
                self.conv.weight.data[0, 0, 0, 1] = -1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Same forward pass as CNN2S."""
        batch_size, num_routes, num_features, _ = x.size()
        
        # Separate features
        x_flat = x.view(batch_size * num_routes, num_features, 1)
        x_interp = x_flat[:, :self.num_interpretable, :]
        x_non_interp = x_flat[:, self.num_interpretable:, :]
        
        # Quadratic features for interpretable
        x_quad_interp = torch.matmul(x_interp, x_interp.transpose(1, 2))
        upper_idx_interp = torch.triu_indices(self.num_interpretable, self.num_interpretable, offset=0)
        x_quad_interp_upper = x_quad_interp[:, upper_idx_interp[0], upper_idx_interp[1]]
        
        # Quadratic features for non-interpretable  
        x_quad_non_interp = torch.matmul(x_non_interp, x_non_interp.transpose(1, 2))
        upper_idx_non_interp = torch.triu_indices(self.num_non_interpretable, self.num_non_interpretable, offset=0)
        x_quad_non_interp_upper = x_quad_non_interp[:, upper_idx_non_interp[0], upper_idx_non_interp[1]]
        
        # Combine all features
        x_interp_linear = x_interp.view(batch_size * num_routes, self.num_interpretable)
        x_non_interp_linear = x_non_interp.view(batch_size * num_routes, self.num_non_interpretable)
        
        x_combined = torch.cat([
            x_interp_linear,
            x_quad_interp_upper,
            x_non_interp_linear,
            x_quad_non_interp_upper
        ], dim=1)
        
        # Reshape and convolve
        x_combined = x_combined.view(batch_size, 1, num_routes, -1)
        out = self.conv(x_combined)
        logits = out.squeeze(1).squeeze(-1)
        
        return logits