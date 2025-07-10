# DeepLogit

A sequentially constrained deep learning modelling approach for transport policy analysis.

## Overview

DeepLogit implements the models from the paper "DeepLogit: A sequentially constraint deep learning modelling approach for transport policy analysis". The key innovation is maintaining parameter interpretability while improving predictive accuracy.

## Models

### CNN1 (Linear Model)
- **CNN1 (4 features)**: Equivalent to multinomial logit, provides interpretable parameters
- **CNN1 (97 features)**: Same model with geographic features added

### CNN2 (Quadratic Models)
- **CNN2U (4 features)**: 14 parameters - unconstrained quadratic
- **CNN2U (97 features)**: 4850 parameters - unconstrained quadratic  
- **CNN2S (97 features)**: 4478 parameters - separated quadratic (no cross-terms between interpretable/non-interpretable)
- **CNN2C (97 features)**: 4478 parameters - constrained quadratic (fixes first 4 weights from CNN1)

### Transformer Models
- **TFMU**: Unconstrained transformer
- **TFMC**: Constrained transformer (fixes first 4 weights from CNN1)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from deeplogit.models import CNN1, CNN2C, TFMC

# Train CNN1 to get interpretable parameters
cnn1 = CNN1(num_features=4)
# ... train model ...
base_weights = cnn1.get_feature_weights()

# Use constrained CNN2
cnn2c = CNN2C(num_features=97, cnn1_weights=base_weights)

# Or constrained transformer
tfmc = TFMC(num_features=97, cnn1_weights=base_weights)
```

## Data Format

### Features
1. **Interpretable features (first 4)**:
   - IVTT: In-Vehicle Travel Time (minutes)
   - Fare: Public transit fare (S$)
   - WT: Walk time for transfers (minutes)
   - NoT: Number of Transfers

2. **Geographic features (remaining 93)**: Land use categories from Singapore Master Plan

### Input Shape
- Routes: `(batch_size, num_routes, num_features, 1)`
- Choices: `(batch_size,)` with chosen route indices

## Key Results

From the paper using Singapore transit data:
- CNN1 (4 features): 75% accuracy (baseline)
- CNN2C (97 features): 82% accuracy (constrained)
- CNN2U (97 features): 84% accuracy (unconstrained)
- Cost of interpretability: ~2%

## Citation

```
@article{deeplogit2025,
  title={DeepLogit: A sequentially constraint deep learning modelling approach for transport policy analysis},
  author={Jeremy Oon, Rakhi Manohar Mepparambatha, Ling Feng},
  year={2025}
}
```