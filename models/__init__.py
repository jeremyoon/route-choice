"""Neural network models for route choice prediction."""

from .simple_cnn import SimpleCNN

# Import transformers only if they exist
try:
    from .transformer_constrained import TransformerConstrained
    from .transformer_unconstrained import TransformerUnconstrained
    from .transformer_separated import TransformerSeparated
    
    __all__ = [
        "SimpleCNN",
        "TransformerConstrained",
        "TransformerUnconstrained", 
        "TransformerSeparated"
    ]
except ImportError:
    __all__ = ["SimpleCNN"]