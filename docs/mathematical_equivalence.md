# Mathematical Equivalence: CNN and Multinomial Logit

This document explains how the SimpleCNN model is mathematically equivalent to a multinomial logit model.

## Multinomial Logit Model

In a standard multinomial logit model for route choice, the utility of route *j* for individual *n* is:

```
U_nj = V_nj + ε_nj = β'X_nj + ε_nj
```

Where:
- `X_nj` is a vector of observable attributes for route *j*
- `β` is a vector of parameters to be estimated
- `ε_nj` is a random error term (following Gumbel distribution)

The probability of choosing route *j* is:

```
P_nj = exp(V_nj) / Σ_k exp(V_nk)
```

## CNN Architecture

The SimpleCNN model uses a 1D convolution with:
- Input channels: 1
- Output channels: 1
- Kernel size: (1, num_features)
- No bias (optional)

### Forward Pass

1. **Input**: Tensor of shape `(batch_size, num_routes, num_features, 1)`
2. **Reshape**: Add channel dimension → `(batch_size, 1, num_routes, num_features)`
3. **Convolution**: Apply 1D convolution → `(batch_size, 1, num_routes, 1)`
4. **Output**: Remove extra dimensions → `(batch_size, num_routes)`

### Mathematical Operation

The convolution operation for route *j* is:

```
output_j = Σ_i (w_i × x_ji)
```

Where:
- `w_i` is the weight for feature *i* (learned parameter)
- `x_ji` is feature *i* of route *j*

This is exactly equivalent to the linear utility function `V_nj = β'X_nj` in the logit model!

## Equivalence Proof

1. **Linear Transformation**: The CNN performs `V_j = W'X_j` where W are the convolutional weights
2. **Softmax**: We apply softmax to get probabilities: `P_j = exp(V_j) / Σ_k exp(V_k)`
3. **Loss Function**: Cross-entropy loss is equivalent to negative log-likelihood in MNL

Therefore:
- CNN weights = MNL β parameters
- CNN output (before softmax) = MNL utilities
- Softmax(CNN output) = MNL choice probabilities

## Advantages of CNN Formulation

1. **Computational Efficiency**: Leverages optimized convolution operations
2. **GPU Acceleration**: Natural parallelization across batch and routes
3. **Framework Integration**: Easy to extend with more complex architectures
4. **Automatic Differentiation**: No need to derive gradients manually

## Code Example

```python
# Multinomial Logit (traditional)
utilities = np.dot(route_features, beta)  # Shape: (n_routes,)
probabilities = np.exp(utilities) / np.sum(np.exp(utilities))

# CNN (equivalent)
model = SimpleCNN(num_features=4)
logits = model(route_features)  # Convolution operation
probabilities = F.softmax(logits, dim=1)
```

Both approaches yield identical results when trained on the same data!