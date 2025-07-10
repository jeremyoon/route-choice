"""Transformer model without parameter constraints for maximum flexibility."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerUnconstrained(nn.Module):
    """Unconstrained transformer model for route choice prediction.
    
    This model uses a transformer architecture without any parameter constraints,
    allowing it to learn freely from the data. While this provides maximum
    flexibility and potentially better accuracy, it sacrifices parameter
    interpretability.
    
    Architecture:
    1. Quadratic feature expansion for all features
    2. Transformer encoder to learn complex interactions
    3. Final convolution to produce route utilities
    
    Args:
        num_features: Total number of input features
        d_model: Dimension of transformer model
        nhead: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        hidden_size: Hidden layer size after pooling
    """
    
    def __init__(
        self,
        num_features: int,
        d_model: int = 512,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        hidden_size: int = 512
    ):
        super(TransformerUnconstrained, self).__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.hidden_size = hidden_size
        
        # Transformer encoder
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
        
        # Final convolution
        # Kernel size = original features + transformer output
        conv_kernel_size = num_features + dim_feedforward
        self.conv_final = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, conv_kernel_size),
            padding=0
        )
        
        # Adaptive pooling for dimension reduction
        self.avgpool = nn.AdaptiveAvgPool1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, num_routes, num_features, 1)
            
        Returns:
            Logits tensor of shape (batch_size, num_routes)
        """
        batch_size, num_routes, num_features, _ = x.size()
        
        # Reshape for processing
        x = x.view(batch_size * num_routes, num_features, 1)
        
        # Create quadratic features through outer product
        quadratic_features = torch.matmul(x, x.transpose(1, 2))
        
        # Extract upper triangle (including diagonal) to avoid redundancy
        upper_idx = torch.triu_indices(num_features, num_features, offset=0)
        quad_upper = quadratic_features[..., upper_idx[0], upper_idx[1]]
        
        # Combine linear and quadratic features
        x_linear = x.view(batch_size * num_routes, num_features)
        x_combined = torch.cat([x_linear, quad_upper], dim=1)
        
        # Apply dropout and pooling
        x_combined = self.dropout(x_combined)
        x_pooled = self.avgpool(x_combined)
        x_pooled = x_pooled.view(batch_size * num_routes, self.hidden_size)
        x_pooled = F.relu(x_pooled)
        
        # Reshape for transformer (batch, seq_len=1, features)
        x_transformer = x_pooled.unsqueeze(1)
        
        # Apply layer normalization
        x_transformer = self.layer_norm(x_transformer)
        
        # Pass through transformer encoder
        y_transformed = self.transformer_encoder(x_transformer)
        y_transformed = y_transformed.squeeze(1)
        
        # Final projection
        y_transformed = self.fc(y_transformed)
        
        # Combine original features with transformed features
        y_combined = torch.cat([x_linear, y_transformed], dim=1)
        
        # Reshape for convolution
        y_combined = y_combined.view(batch_size, 1, num_routes, -1)
        
        # Final convolution to get utilities
        y_out = self.conv_final(y_combined)
        
        # Get logits
        logits = y_out.squeeze(1).squeeze(-1)
        
        return logits
    
    def get_feature_importance(self) -> torch.Tensor:
        """Get feature importance scores from the final convolution layer.
        
        Note: These are not directly interpretable as in linear models,
        but can give some indication of feature contribution.
        
        Returns:
            Feature importance scores
        """
        # Get weights from final convolution
        conv_weights = self.conv_final.weight.squeeze()
        
        # Extract weights corresponding to original features
        feature_weights = conv_weights[:self.num_features]
        
        # Return absolute values as importance scores
        return torch.abs(feature_weights).detach()
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for input (useful for interpretation).
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights from the transformer
        """
        # This would require modifying the transformer to return attention weights
        # Left as a placeholder for future implementation
        raise NotImplementedError("Attention weight extraction not yet implemented")