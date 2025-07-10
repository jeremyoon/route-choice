"""Transformer model with constrained parameters for interpretability."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightConstraint:
    """Constraint to fix specific parameter values during training.
    
    This is used to maintain interpretability of key parameters (e.g., travel time,
    cost, etc.) while allowing the model to learn complex interactions through
    the transformer layers.
    """
    
    def __init__(self, start_idx: int, end_idx: int, fixed_weights: torch.Tensor):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.fixed_weights = fixed_weights
    
    def __call__(self, module: nn.Module):
        """Apply constraint to module weights."""
        with torch.no_grad():
            module.weight.data[0, 0, 0, self.start_idx:self.end_idx] = \
                self.fixed_weights[0, 0, 0, self.start_idx:self.end_idx]


class TransformerConstrained(nn.Module):
    """Transformer model with constrained parameters for route choice.
    
    This model combines:
    1. Fixed parameters for interpretable features (from a pre-trained linear model)
    2. Transformer encoder for learning complex interactions from other features
    3. Final convolution to combine both components
    
    The key innovation is maintaining interpretability of important parameters
    (e.g., travel time, cost) while leveraging deep learning for better accuracy.
    
    Args:
        num_features: Total number of features
        num_constrained_features: Number of features with constrained parameters
        pretrained_weights: Weights from pre-trained linear model to constrain
        d_model: Dimension of transformer model
        nhead: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        hidden_size: Hidden layer size
    """
    
    def __init__(
        self,
        num_features: int,
        num_constrained_features: int,
        pretrained_weights: torch.Tensor,
        d_model: int = 512,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        hidden_size: int = 512
    ):
        super(TransformerConstrained, self).__init__()
        
        self.num_features = num_features
        self.num_constrained_features = num_constrained_features
        self.d_model = d_model
        self.hidden_size = hidden_size
        
        # Transformer encoder for unconstrained features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Linear projection after transformer
        self.fc = nn.Linear(d_model, dim_feedforward)
        
        # Final convolution combining constrained and unconstrained features
        conv_kernel_size = num_constrained_features + dim_feedforward
        self.conv_final = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, conv_kernel_size),
            padding=0
        )
        
        # Adaptive pooling for dimension reduction
        self.avgpool = nn.AdaptiveAvgPool1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Apply constraint to preserve interpretable parameters
        if pretrained_weights is not None:
            self._apply_weight_constraint(pretrained_weights)
    
    def _apply_weight_constraint(self, pretrained_weights: torch.Tensor):
        """Apply constraint to keep certain weights fixed."""
        constraint = WeightConstraint(
            start_idx=0,
            end_idx=self.num_constrained_features,
            fixed_weights=pretrained_weights
        )
        self.conv_final.apply(constraint)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, num_routes, num_features, 1)
            
        Returns:
            Logits tensor of shape (batch_size, num_routes)
        """
        batch_size, num_routes, num_features, _ = x.size()
        
        # Separate constrained and unconstrained features
        x_flat = x.view(batch_size * num_routes, num_features, 1)
        x_constrained = x_flat[:, :self.num_constrained_features, :]
        x_unconstrained = x_flat[:, self.num_constrained_features:, :]
        
        # Process unconstrained features with quadratic interactions
        # Create quadratic features through outer product
        y_unconstrained = torch.matmul(x_unconstrained, x_unconstrained.transpose(1, 2))
        
        # Extract upper triangle to avoid redundancy
        n_unconstr = self.num_features - self.num_constrained_features
        upper_idx = torch.triu_indices(n_unconstr, n_unconstr, offset=0)
        y_upper = y_unconstrained[..., upper_idx[0], upper_idx[1]]
        
        # Combine linear and quadratic unconstrained features
        x_unconstr_flat = x_unconstrained.view(batch_size * num_routes, n_unconstr)
        y_unconstrained = torch.cat([x_unconstr_flat, y_upper], dim=1)
        
        # Apply dropout and pooling
        y_unconstrained = self.dropout(y_unconstrained)
        y_unconstrained = self.avgpool(y_unconstrained)
        y_unconstrained = y_unconstrained.view(batch_size * num_routes, self.hidden_size)
        y_unconstrained = F.relu(y_unconstrained)
        
        # Pass through transformer encoder
        y_transformed = self.transformer_encoder(y_unconstrained.unsqueeze(0))
        y_transformed = y_transformed.squeeze(0)
        y_transformed = self.fc(y_transformed)
        
        # Combine constrained and transformed features
        x_constrained_flat = x_constrained.view(batch_size * num_routes, self.num_constrained_features)
        y_combined = torch.cat([x_constrained_flat, y_transformed], dim=1)
        
        # Final convolution
        y_combined = y_combined.view(batch_size, 1, num_routes, -1)
        y_out = self.conv_final(y_combined)
        
        # Flatten to get logits
        logits = y_out.squeeze(1).squeeze(-1)
        
        return logits
    
    def get_constrained_weights(self) -> torch.Tensor:
        """Get the constrained (interpretable) weights.
        
        Returns:
            Tensor containing the constrained parameter values
        """
        return self.conv_final.weight[0, 0, 0, :self.num_constrained_features].detach()
    
    def enforce_constraints(self, pretrained_weights: torch.Tensor):
        """Re-apply weight constraints (call after each optimization step).
        
        Args:
            pretrained_weights: Fixed weights to enforce
        """
        self._apply_weight_constraint(pretrained_weights)