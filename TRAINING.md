# DeepLogit Training Guide

## Quick Start

To train all DeepLogit models with the full experimental setup:

```bash
cd DeepLogit
./run_full_training.sh
```

This will train all 8 model variants across 5 cross-validation folds.

## Training Script

The main training script `train_all_experiments.py` trains all models from the DeepLogit paper:

### Models

| Model | Features | Parameters | Batch Size | Epochs | Constraints |
|-------|----------|------------|------------|--------|-------------|
| CNN1 | 4 | 4 | 512 | 50 | Fare β = -1 |
| CNN1 | 97 | 97 | 512 | 50 | Fare β = -1 |
| CNN2U | 4 | 14 | 1024 | 20,000 | None |
| CNN2U | 97 | 4,850 | 1536 | 20,000 | None |
| CNN2S | 97 | 4,478 | 1664 | 20,000 | Fare β = -1 |
| CNN2C | 97 | 4,478 | 2048 | 20,000 | Inherits from CNN1 |
| TFMU | 97 | 532,993 | 256 | 20,000 | None |
| TFMC | 97 | 527,873 | 384 | 20,000 | Inherits from CNN1 |

### Usage

```bash
python train_all_experiments.py [options]

Options:
  --combinations    Train/validation splits (default: all 5 combinations)
  --epochs_cnn1     Epochs for CNN1 models (default: 50)
  --epochs_cnn2     Epochs for CNN2 models (default: 20000)
  --epochs_tfm      Epochs for Transformer models (default: 20000)
  --test_run        Quick test with 1 epoch per model
```

### Quick Test

Before running the full training:

```bash
python train_all_experiments.py --test_run --combinations "0 1 2 3_4"
```

## Data Requirements

The script expects processed data files in `../data/processed/`:
- `train_datasets_{combi}_cpu.pkl`
- `valid_datasets_{combi}_cpu.pkl`
- `test_datasets_{combi}_cpu.pkl`
- `sampler_train_{combi}_cpu.pkl`
- `sampler_valid_{combi}_cpu.pkl`
- `sampler_test_{combi}_cpu.pkl`

Where `{combi}` is one of: `0 1 2 3_4`, `0 1 2 4_3`, `0 1 3 4_2`, `0 2 3 4_1`, `1 2 3 4_0`

## Output

Results are saved to `../results/`:

### Model Checkpoints
- `{model}_best_{combi}.pt` - Best model based on validation accuracy
- `{model}_epoch{N}_{combi}.pt` - Checkpoints every 1000 epochs

### Training History
- `{model}_train_loss_history_{combi}.npy` - Training loss per epoch
- `{model}_train_acc_history_{combi}.npy` - Training accuracy per epoch
- `{model}_valid_loss_history_{combi}.npy` - Validation loss
- `{model}_valid_acc_history_{combi}.npy` - Validation accuracy
- `{model}_valid_epochs_{combi}.npy` - Epochs when validation was performed
- `{model}_history_{combi}.pkl` - Complete history dictionary

### LaTeX Tables
- `latex_tables.txt` - Tables Tb_NN and Tb_Compare for the paper

## Plotting Training Curves

After training completes:

```bash
python plot_training_curves.py --comparison
```

This creates:
- Individual training curves for each model
- Comparison plots across models
- Zoomed plots for the last 10% of training (for long runs)

## Sequential Constraint Approach

The training implements the sequential constraint methodology:

1. **Train CNN1 (4 features)** to learn interpretable parameters
2. **Extract weights** with fare β clamped to -1
3. **Apply constraints** to CNN2C and TFMC models
4. **Compare** constrained vs unconstrained performance

## Notes

- Batch sizes are optimized per model to avoid OOM errors
- Validation is performed every 100 epochs for long training
- Early stopping is not used to allow full convergence analysis
- All models use SGD optimizer with learning rate 0.001