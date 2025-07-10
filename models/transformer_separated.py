"""Transformer model with separated interpretable and non-interpretable features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerSeparated(nn.Module):
    """Transformer model that separates interpretable from non-interpretable features.
    
    This model implements a novel approach where:
    1. Interpretable features (e.g., travel time, cost) are processed linearly
    2. Non-interpretable features (e.g., land use) are processed through transformers
    3. Both components are combined for final predictions
    
    This separation allows maintaining interpretability for key policy variables
    while leveraging deep learning for complex interactions in other features.
    
    Args:
        num_features: Total number of features
        num_interpretable: Number of interpretable features (processed linearly)
        d_model: Dimension of transformer model
        nhead: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        hidden_size: Hidden size after pooling
        use_quadratic: Whether to use quadratic interactions for non-interpretable features
    """
    
    def __init__(
        self,
        num_features: int,
        num_interpretable: int = 4,
        d_model: int = 748,
        nhead: int = 2,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 748,
        dropout: float = 0.2,
        hidden_size: int = 748,
        use_quadratic: bool = True
    ):
        super(TransformerSeparated, self).__init__()
        
        self.num_features = num_features
        self.num_interpretable = num_interpretable
        self.num_non_interpretable = num_features - num_interpretable
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.use_quadratic = use_quadratic
        
        # Transformer encoder for non-interpretable features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Linear projection after transformer
        self.fc = nn.Linear(d_model, dim_feedforward)
        
        # Pooling for dimension reduction
        self.maxpool = nn.AdaptiveMaxPool1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Calculate final convolution kernel size
        if use_quadratic:
            # Interpretable: linear + quadratic terms
            interpretable_size = num_interpretable + (num_interpretable * (num_interpretable + 1)) // 2
        else:
            interpretable_size = num_interpretable
        
        # Total kernel size = interpretable features + transformer output
        conv_kernel_size = interpretable_size + dim_feedforward
        
        # Final convolution to combine all features
        self.conv_final = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, conv_kernel_size),
            padding=0
        )
    
    def _create_quadratic_features(self, x: torch.Tensor) -> torch.Tensor:
        """Create quadratic interaction features.
        
        Args:
            x: Input tensor of shape (batch_size, num_features, 1)
            
        Returns:
            Tensor with linear and quadratic features concatenated
        """
        batch_size = x.shape[0]
        num_feat = x.shape[1]
        
        # Create outer product for quadratic features
        x_quad = torch.matmul(x, x.transpose(1, 2))
        
        # Extract upper triangle (including diagonal)
        upper_idx = torch.triu_indices(num_feat, num_feat, offset=0)
        x_upper = x_quad[..., upper_idx[0], upper_idx[1]]
        
        # Combine with linear features
        x_linear = x.view(batch_size, num_feat)
        x_combined = torch.cat([x_linear, x_upper], dim=1)
        
        return x_combined
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, num_routes, num_features, 1)
            
        Returns:
            Logits tensor of shape (batch_size, num_routes)
        """
        batch_size, num_routes, num_features, _ = x.size()
        
        # Reshape and separate features
        x_reshaped = x.view(batch_size * num_routes, num_features, 1)
        x_interpretable = x_reshaped[:, :self.num_interpretable, :]
        x_non_interpretable = x_reshaped[:, self.num_interpretable:, :]
        
        # Process interpretable features
        if self.use_quadratic:
            # Create quadratic features for interpretable variables
            y_interpretable = self._create_quadratic_features(x_interpretable)
        else:
            # Just use linear features
            y_interpretable = x_interpretable.view(batch_size * num_routes, self.num_interpretable)
        
        # Process non-interpretable features with transformer
        # First create quadratic features
        y_non_interp_quad = torch.matmul(x_non_interpretable, x_non_interpretable.transpose(1, 2))
        y_non_interp_quad = y_non_interp_quad.view(batch_size * num_routes, -1)
        
        # Combine with linear features
        x_non_interp_linear = x_non_interpretable.view(batch_size * num_routes, self.num_non_interpretable)
        y_non_interp = torch.cat([y_non_interp_quad, x_non_interp_linear], dim=1)
        
        # Apply dropout and pooling
        y_non_interp = self.dropout(y_non_interp)
        y_non_interp = self.maxpool(y_non_interp)
        y_non_interp = y_non_interp.view(batch_size * num_routes, self.hidden_size)
        y_non_interp = F.relu(y_non_interp)
        
        # Prepare for transformer (add sequence dimension)
        y_non_interp = y_non_interp.unsqueeze(1)
        y_non_interp = self.layer_norm(y_non_interp)
        
        # Pass through transformer
        y_transformed = self.transformer_encoder(y_non_interp)
        y_transformed = y_transformed.squeeze(1)
        y_transformed = self.fc(y_transformed)
        
        # Combine interpretable and transformed non-interpretable features
        y_combined = torch.cat([y_interpretable, y_transformed], dim=1)
        
        # Reshape for final convolution
        y_combined = y_combined.view(batch_size, 1, num_routes, -1)
        
        # Final convolution
        y_out = self.conv_final(y_combined)
        
        # Get logits
        logits = y_out.squeeze(1).squeeze(-1)
        
        return logits
    
    def get_interpretable_weights(self) -> torch.Tensor:
        """Get weights for interpretable features.
        
        Returns:
            Weights corresponding to interpretable features
        """
        # Extract weights from convolution layer
        conv_weights = self.conv_final.weight.squeeze()
        
        # Return weights for interpretable features (linear terms only)
        return conv_weights[:self.num_interpretable].detach()
    
    def get_feature_separation_info(self) -> dict:
        """Get information about feature separation.
        
        Returns:
            Dictionary with feature separation details
        """
        return {
            "num_interpretable": self.num_interpretable,
            "num_non_interpretable": self.num_non_interpretable,
            "interpretable_indices": list(range(self.num_interpretable)),
            "non_interpretable_indices": list(range(self.num_interpretable, self.num_features)),
            "uses_quadratic_interpretable": self.use_quadratic
        }