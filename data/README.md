# DeepLogit Data Directory

This directory contains the processed data files required for training DeepLogit models.

## Directory Structure

```
data/
├── __init__.py          # Package initialization
├── dataset.py           # CustomDataset and WeightedBucketSampler classes
├── README.md           # This file
└── processed/          # Processed data files (not included in repository)
    ├── train_datasets_{combi}_cpu.pkl
    ├── valid_datasets_{combi}_cpu.pkl
    ├── test_datasets_{combi}_cpu.pkl
    ├── sampler_train_{combi}_cpu.pkl
    ├── sampler_valid_{combi}_cpu.pkl
    └── sampler_test_{combi}_cpu.pkl
```

## Data Files

The processed data files are not included in the repository due to size and privacy constraints. 

### File Naming Convention

Files follow the pattern: `{type}_datasets_{combi}_cpu.pkl` where:
- `type`: train, valid, or test
- `combi`: Cross-validation fold combination (e.g., "0 1 2 3_4")

### Available Combinations

- `0 1 2 3_4`: Train on folds 0,1,2,3; validate on fold 4
- `0 1 2 4_3`: Train on folds 0,1,2,4; validate on fold 3
- `0 1 3 4_2`: Train on folds 0,1,3,4; validate on fold 2
- `0 2 3 4_1`: Train on folds 0,2,3,4; validate on fold 1
- `1 2 3 4_0`: Train on folds 1,2,3,4; validate on fold 0

## Data Format

Each pickle file contains:
- **Dataset files**: List of CustomDataset objects
- **Sampler files**: List of WeightedBucketSampler objects

The data represents Singapore public transit route choice data with:
- 4 interpretable features: IVTT, Fare, WT, NoT
- 93 additional geographic/land use features
- Variable number of route alternatives per choice situation

## CustomDataset Class

The `CustomDataset` class handles the specific format where each sample contains:
- `routes`: Tensor of shape (num_routes, num_features, 1)
- `choice`: Scalar tensor indicating the chosen route

## Note

The data files are large (>90GB total) and contain proprietary data. They are excluded from version control via `.gitignore`.