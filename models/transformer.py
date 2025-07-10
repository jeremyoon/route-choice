"""Transformer models for route choice."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TFMU(nn.Module):
    """Transformer Unconstrained - no parameter constraints."""
    
    def __init__(
        self,
        num_features: int = 97,
        d_model: int = 512,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        hidden_size: int = 512
    ):
        super(TFMU, self).__init__()
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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Projection after transformer
        self.fc = nn.Linear(d_model, dim_feedforward)
        
        # Final convolution
        conv_kernel_size = num_features + dim_feedforward
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, conv_kernel_size), padding=0)
        
        # Pooling and normalization
        self.avgpool = nn.AdaptiveAvgPool1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_routes, num_features, _ = x.size()
        
        # Flatten batch and routes
        x = x.view(batch_size * num_routes, num_features, 1)
        
        # Create quadratic features
        x_quad = torch.matmul(x, x.transpose(1, 2))
        upper_idx = torch.triu_indices(num_features, num_features, offset=0)
        x_quad_upper = x_quad[:, upper_idx[0], upper_idx[1]]
        
        # Combine linear and quadratic
        x_linear = x.view(batch_size * num_routes, num_features)
        x_combined = torch.cat([x_linear, x_quad_upper], dim=1)
        
        # Pooling and activation
        x_combined = self.dropout(x_combined)
        x_pooled = self.avgpool(x_combined)
        x_pooled = x_pooled.view(batch_size * num_routes, self.hidden_size)
        x_pooled = F.relu(x_pooled)
        
        # Transformer processing
        x_pooled = x_pooled.unsqueeze(1)  # Add sequence dimension
        x_pooled = self.layer_norm(x_pooled)
        x_transformed = self.transformer_encoder(x_pooled)
        x_transformed = x_transformed.squeeze(1)
        x_transformed = self.fc(x_transformed)
        
        # Combine with original features
        y = torch.cat([x_linear, x_transformed], dim=1)
        y = y.view(batch_size, 1, num_routes, -1)
        
        # Final convolution
        out = self.conv(y)
        logits = out.squeeze(1).squeeze(-1)
        
        return logits


class TFMC(nn.Module):
    """Transformer Constrained - fixes weights for first 4 features from CNN1."""
    
    def __init__(
        self,
        num_features: int = 97,
        cnn1_weights: torch.Tensor = None,
        d_model: int = 512,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        hidden_size: int = 512
    ):
        super(TFMC, self).__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_constrained = 4  # First 4 features are constrained
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Projection after transformer
        self.fc = nn.Linear(d_model, dim_feedforward)
        
        # Final convolution
        conv_kernel_size = 4 + dim_feedforward  # 4 constrained features + transformer output
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, conv_kernel_size), padding=0)
        
        # Pooling and normalization
        self.avgpool = nn.AdaptiveAvgPool1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Apply initial constraints if provided
        if cnn1_weights is not None:
            self.apply_constraint(cnn1_weights)
    
    def apply_constraint(self, cnn1_weights: torch.Tensor):
        """Fix weights for first 4 features."""
        with torch.no_grad():
            self.conv.weight.data[0, 0, 0, :self.num_constrained] = cnn1_weights.flatten()[:self.num_constrained]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_routes, num_features, _ = x.size()
        
        # Separate constrained and unconstrained features
        x = x[:, :, :self.num_features, :]  # Use only specified features
        x_flat = x.view(batch_size * num_routes, self.num_features, 1)
        x_constrained = x_flat[:, :self.num_constrained, :]
        x_unconstrained = x_flat[:, self.num_constrained:, :]
        
        # Process unconstrained features with quadratic expansion
        x_quad = torch.matmul(x_unconstrained, x_unconstrained.transpose(1, 2))
        n_unconstr = self.num_features - self.num_constrained
        upper_idx = torch.triu_indices(n_unconstr, n_unconstr, offset=0)
        x_quad_upper = x_quad[:, upper_idx[0], upper_idx[1]]
        
        # Combine linear and quadratic unconstrained features
        x_unconstr_linear = x_unconstrained.view(batch_size * num_routes, n_unconstr)
        x_unconstr_combined = torch.cat([x_unconstr_linear, x_quad_upper], dim=1)
        
        # Pooling and processing
        x_unconstr_combined = self.dropout(x_unconstr_combined)
        x_pooled = self.avgpool(x_unconstr_combined)
        x_pooled = x_pooled.view(batch_size * num_routes, self.hidden_size)
        x_pooled = F.relu(x_pooled)
        
        # Transformer processing
        x_pooled = x_pooled.unsqueeze(1)
        x_pooled = self.layer_norm(x_pooled)
        x_transformed = self.transformer_encoder(x_pooled)
        x_transformed = x_transformed.squeeze(1)
        x_transformed = self.fc(x_transformed)
        
        # Combine constrained features with transformer output
        x_constrained_flat = x_constrained.view(batch_size * num_routes, self.num_constrained)
        y = torch.cat([x_constrained_flat, x_transformed], dim=1)
        y = y.view(batch_size, 1, num_routes, -1)
        
        # Final convolution
        out = self.conv(y)
        logits = out.squeeze(1).squeeze(-1)
        
        return logits