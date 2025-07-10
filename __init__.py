"""
DeepLogit: A sequentially constrained deep learning modelling approach for transport policy analysis.

This package provides neural network models for route choice prediction, including:
- Simple CNN models (mathematically equivalent to logit)
- CNN models with geographic features
- Transformer-based models (ViT, TabTransformer)
"""

__version__ = "0.1.0"
__author__ = "DeepLogit Contributors"

from . import models
from . import data
from . import utils

__all__ = ["models", "data", "utils"]