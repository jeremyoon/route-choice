#!/bin/bash

# Script to run full training
# This will train all models with the full epoch counts

echo "Starting full DeepLogit training at $(date)"
echo "This will run for approximately 2 weeks"
echo "============================================"

# Create log directory
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# Run training with nohup to keep running after logout
# Note: Batch sizes are optimized per model to avoid OOM
nohup python train_all_experiments.py \
    --epochs_cnn1 50 \
    --epochs_cnn2 20000 \
    --epochs_tfm 20000 \
    > $LOG_DIR/training_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: $PID"
echo "Log file: $LOG_DIR/training_full_$(date +%Y%m%d_%H%M%S).log"

# Save PID to file for later reference
echo $PID > $LOG_DIR/training.pid

echo ""
echo "To check progress:"
echo "  tail -f $LOG_DIR/training_full_*.log"
echo ""
echo "To check if still running:"
echo "  ps -p $PID"
echo ""
echo "To stop training:"
echo "  kill $PID"
echo ""
echo "Training curves can be plotted with:"
echo "  python plot_training_curves.py --comparison"