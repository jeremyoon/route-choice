"""Plot training curves from saved history files."""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse

RESULTS_DIR = './results/'

def plot_training_curves(model_name, combi, save_dir=RESULTS_DIR):
    """Plot training and validation curves for a model."""
    
    # Load history
    history_file = f'{save_dir}/{model_name}_history_{combi}.pkl'
    if not os.path.exists(history_file):
        print(f"History file not found: {history_file}")
        return
    
    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    
    train_loss = history['train_loss_history']
    train_acc = history['train_acc_history']
    valid_loss = history['valid_loss_history']
    valid_acc = history['valid_acc_history']
    valid_epochs = history['valid_epochs']
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss
    ax1.plot(range(1, len(train_loss) + 1), train_loss, 'b-', label='Train Loss', alpha=0.7)
    if valid_loss:
        ax1.plot(valid_epochs, valid_loss, 'r-', label='Valid Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss ({combi})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(range(1, len(train_acc) + 1), train_acc, 'b-', label='Train Acc', alpha=0.7)
    if valid_acc:
        ax2.plot(valid_epochs, valid_acc, 'r-', label='Valid Acc', linewidth=2)
        # Mark best validation accuracy
        best_idx = np.argmax(valid_acc)
        best_epoch = valid_epochs[best_idx]
        best_acc = valid_acc[best_idx]
        ax2.plot(best_epoch, best_acc, 'g*', markersize=15, label=f'Best: {best_acc:.4f} @ {best_epoch}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} - Accuracy ({combi})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_curves_{combi}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {save_dir}/{model_name}_curves_{combi}.png")
    
    # Also create a zoomed version for long training
    if len(train_loss) > 1000:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot last 10% of epochs
        start_idx = int(0.9 * len(train_loss))
        
        ax1.plot(range(start_idx, len(train_loss) + 1), train_loss[start_idx-1:], 'b-', label='Train Loss', alpha=0.7)
        valid_epochs_zoomed = [e for e in valid_epochs if e >= start_idx]
        valid_loss_zoomed = [valid_loss[i] for i, e in enumerate(valid_epochs) if e >= start_idx]
        if valid_loss_zoomed:
            ax1.plot(valid_epochs_zoomed, valid_loss_zoomed, 'r-', label='Valid Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name} - Loss (Last 10%) ({combi})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(range(start_idx, len(train_acc) + 1), train_acc[start_idx-1:], 'b-', label='Train Acc', alpha=0.7)
        valid_acc_zoomed = [valid_acc[i] for i, e in enumerate(valid_epochs) if e >= start_idx]
        if valid_acc_zoomed:
            ax2.plot(valid_epochs_zoomed, valid_acc_zoomed, 'r-', label='Valid Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{model_name} - Accuracy (Last 10%) ({combi})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{model_name}_curves_zoomed_{combi}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved zoomed plot: {save_dir}/{model_name}_curves_zoomed_{combi}.png")


def plot_model_comparison(models, combi, save_dir=RESULTS_DIR):
    """Plot comparison of multiple models."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for model_name in models:
        history_file = f'{save_dir}/{model_name}_history_{combi}.pkl'
        if not os.path.exists(history_file):
            print(f"Skipping {model_name} - history file not found")
            continue
        
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
        
        valid_loss = history['valid_loss_history']
        valid_acc = history['valid_acc_history']
        valid_epochs = history['valid_epochs']
        
        if valid_loss:
            ax1.plot(valid_epochs, valid_loss, label=model_name, linewidth=2)
        if valid_acc:
            ax2.plot(valid_epochs, valid_acc, label=model_name, linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title(f'Model Comparison - Validation Loss ({combi})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title(f'Model Comparison - Validation Accuracy ({combi})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison_{combi}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {save_dir}/model_comparison_{combi}.png")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['CNN1_4', 'CNN1_97', 'CNN2U_4', 'CNN2U_97', 'CNN2S_97', 'CNN2C_97', 'TFMU_97', 'TFMC_97'],
                        help='Models to plot')
    parser.add_argument('--combinations', type=str, nargs='+',
                        default=['0 1 2 3_4'],
                        help='Combinations to plot')
    parser.add_argument('--comparison', action='store_true',
                        help='Create comparison plots')
    args = parser.parse_args()
    
    # Plot individual curves
    for combi in args.combinations:
        print(f"\nPlotting curves for combination: {combi}")
        for model in args.models:
            plot_training_curves(model, combi)
        
        # Plot comparison
        if args.comparison:
            plot_model_comparison(args.models, combi)


if __name__ == "__main__":
    main()