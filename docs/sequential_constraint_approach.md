# Sequential Constraint Approach in DeepLogit

## Overview

DeepLogit introduces a novel **sequential constraint approach** for combining the interpretability of discrete choice models with the predictive power of deep learning. This approach addresses a key challenge in transport policy analysis: maintaining parameter interpretability while improving model accuracy.

## The Challenge

Traditional discrete choice models (DCMs) like multinomial logit provide:
- ✅ Clear parameter interpretation (e.g., value of time)
- ✅ Policy-relevant insights
- ❌ Limited flexibility in capturing complex patterns
- ❌ Lower predictive accuracy

Deep learning models provide:
- ✅ High predictive accuracy
- ✅ Ability to capture complex interactions
- ❌ Black-box nature
- ❌ No clear parameter interpretation

## The Sequential Constraint Approach

Our approach works in two sequential steps:

### Step 1: Train Interpretable Base Model
First, we train a simple CNN that is mathematically equivalent to multinomial logit:

```python
# Train base model on core features
simple_cnn = SimpleCNN(num_features=4)  # IVTT, Cost, Walk Time, Transfers
simple_cnn.fit(data)

# Extract interpretable parameters
interpretable_weights = simple_cnn.get_feature_weights()
# e.g., [-2.46, -1.0, -2.98, -3.67] for [time, cost, walk, transfers]
```

### Step 2: Extend with Constrained Deep Learning
Next, we build more complex models that:
1. **Fix** the parameters from Step 1 for interpretable features
2. **Learn** additional parameters for other features (e.g., land use)
3. **Combine** both using deep learning architectures

```python
# Create constrained transformer
transformer = TransformerConstrained(
    num_features=97,  # 4 core + 93 land use features
    num_constrained_features=4,
    pretrained_weights=interpretable_weights
)

# During training, core parameters remain fixed
# Only additional features are learned
```

## Key Benefits

1. **Interpretability Preservation**: Core policy parameters (time, cost) maintain their economic interpretation
2. **Improved Accuracy**: Additional features and non-linear interactions improve predictions
3. **Flexibility**: Can choose which parameters to constrain based on policy needs
4. **Theoretical Grounding**: Base model follows random utility theory

## Model Variants

### 1. TransformerConstrained
- Fixes first N parameters from pre-trained model
- Applies transformer to remaining features
- Combines both in final layer

### 2. TransformerSeparated
- Processes interpretable features linearly
- Processes other features with transformer
- No pre-training needed, learns separation jointly

### 3. TransformerUnconstrained
- No constraints - baseline for comparison
- Shows maximum achievable accuracy
- Helps quantify "cost of interpretability"

## Mathematical Foundation

For a route choice problem with alternatives r and decision maker n:

**Traditional MNL**:
```
U_rn = β₁X₁_rn + β₂X₂_rn + ... + βₖXₖ_rn + ε_rn
P_rn = exp(U_rn) / Σ exp(U_jn)
```

**Sequential Constraint Approach**:
```
Step 1: U_rn^base = β₁*X₁_rn + ... + β₄*X₄_rn
Step 2: U_rn^full = [β₁*X₁_rn + ... + β₄*X₄_rn]_fixed + f_DL(X₅_rn, ..., Xₖ_rn)
```

Where:
- β₁* to β₄* are fixed from Step 1
- f_DL is a deep learning function (transformer)
- X₁ to X₄ are interpretable features
- X₅ to Xₖ are additional features

## Practical Example

```python
# Singapore transit route choice
# Interpretable features: travel time, cost, walk time, transfers
# Additional features: 93 land use categories

# Step 1: Train base model
base_accuracy = 0.75  # Typical MNL accuracy
time_parameter = -2.46  # Clear interpretation: disutility per minute

# Step 2: Add constraints and land use
constrained_accuracy = 0.82  # Improved accuracy
time_parameter = -2.46  # SAME interpretation preserved!

# Compare to unconstrained
unconstrained_accuracy = 0.84  # Slightly better
time_parameter = ???  # Lost interpretation

# Cost of interpretability = 0.84 - 0.82 = 0.02 (2% accuracy)
# Benefit of learning = 0.82 - 0.75 = 0.07 (7% accuracy)
```

## When to Use This Approach

Use sequential constraints when:
- Policy interpretation is critical
- You have additional complex features
- Base model theory is well-established
- Small accuracy sacrifice is acceptable

Don't use when:
- Pure prediction is the only goal
- No established theory exists
- All features need interpretation
- Maximum accuracy is required

## Conclusion

The sequential constraint approach provides a principled way to combine theory-driven discrete choice models with data-driven deep learning, maintaining the strengths of both approaches for transport policy analysis.