# DeepLogit

A sequentially constrained deep learning modelling approach for transport policy analysis. This library implements various neural network architectures for route choice modeling as a classification problem.

## Overview

DeepLogit provides implementations of:
- Simple CNN model that is mathematically equivalent to multinomial logit
- Transformer with Constrained parameters (TransformerConstrained) - maintains interpretability of key features
- Transformer Unconstrained (TransformerUnconstrained) - maximum flexibility without constraints
- Transformer with Separated features (TransformerSeparated) - separates interpretable from non-interpretable features

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from deeplogit.models import SimpleCNN, TransformerConstrained
from deeplogit.data import RouteChoiceDataset
from deeplogit.utils.training import train_model

# Load your data
dataset = RouteChoiceDataset(data_path="path/to/data.pkl")

# Train a simple CNN (equivalent to MNL)
simple_model = SimpleCNN(num_features=4)

# Train a constrained transformer
# First get weights from simple model
simple_weights = simple_model.get_feature_weights()

# Create transformer with constraints
transformer = TransformerConstrained(
    num_features=97,
    num_constrained_features=4,
    pretrained_weights=simple_weights
)
```

## Models

### 1. SimpleCNN
- Mathematically equivalent to multinomial logit (MNL)
- Uses 1D convolution over route features
- Provides interpretable parameters
- Serves as baseline and for pre-training constrained models

### 2. TransformerConstrained
- Maintains interpretability of key parameters (e.g., travel time, cost)
- Uses pre-trained weights from SimpleCNN for important features
- Applies transformer to additional features (e.g., land use)
- Balances interpretability with predictive power

### 3. TransformerUnconstrained
- No parameter constraints - maximum flexibility
- Learns all parameters freely from data
- Highest potential accuracy but loses interpretability
- Useful for pure prediction tasks

### 4. TransformerSeparated
- Novel approach separating interpretable from non-interpretable features
- Linear processing for policy-relevant variables
- Transformer processing for complex features (e.g., geographic data)
- Maintains clear interpretation of key parameters

## Data Format

### Transit Route Choice Data

DeepLogit is designed for transit route choice modeling using smart card (AFC) data. The framework was developed using data from Singapore's public transit system, which includes:

- **Heavy Rail**: Mass Rapid Transit (MRT)
- **Light Rail**: Light Rail Transit (LRT)  
- **Bus Services**: Extensive bus network

### Route Features

The models use two types of features:

#### 1. Core Route Attributes (Interpretable)
- **IVTT**: In-Vehicle Travel Time (minutes)
- **NoT**: Number of Transfers
- **Fare**: Public transit fare (S$)
- **WT**: Walk time for transfers (minutes)

#### 2. Geographic/Land Use Features (93 features)
Based on Singapore's Master Plan 2014, including 30 zoning categories:
- Residential, Commercial, Business districts
- Transport facilities, MRT/LRT stations
- Recreation, Parks, Open spaces
- Special use areas (airports, ports, etc.)
- Mixed-use zones

### Data Scale

The framework was developed using large-scale AFC data:
- **Daily volume**: ~6.2 million trips, ~4.5 million journeys
- **Journey definition**: Series of trips with up to 5 transfers within 2 hours
- **Transfer rules**: Maximum 45 minutes between bus/rail transfers

### Choice Set Generation

For each origin-destination pair, the choice set includes the fastest 5 routes from each category:
1. Bus only
2. Bus-Bus (one transfer)
3. Rail only
4. Bus-Rail
5. Rail-Bus
6. Bus-Rail-Bus

### Data Structure

The library expects data in the following format:
- **Routes**: tensor of shape `(n_samples, max_routes, n_features, 1)`
  - `n_samples`: number of choice situations
  - `max_routes`: maximum routes per choice (typically 6-15)
  - `n_features`: 4 core + geographic features (e.g., 97 total)
- **Choices**: tensor of shape `(n_samples,)` containing chosen route indices

### Example Data Loading

```python
import pickle
import torch

# Load preprocessed data
with open('route_choice_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Expected structure
routes = data['routes']  # Shape: (n_samples, max_routes, 97, 1)
choices = data['choices']  # Shape: (n_samples,)

# First 4 features are interpretable
interpretable_features = routes[:, :, :4, :]  # IVTT, Fare, WT, NoT
geographic_features = routes[:, :, 4:, :]  # Land use categories
```

## Citation

If you use DeepLogit in your research, please cite:

```
@article{deeplogit2024,
  title={DeepLogit: A sequentially constraint deep learning modelling approach for transport policy analysis},
  author={Jeremy Oon, Rakhi Manohar Mepparambatha, Ling Feng},
  year={2025}
}
```

## License

MIT License