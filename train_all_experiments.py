"""Train all DeepLogit models and generate LaTeX tables.

This script trains all models from the DeepLogit paper and generates
the performance comparison tables in LaTeX format.
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
from collections import defaultdict
import argparse
from datetime import datetime

# Fix for loading pickled CustomDataset objects
import data.dataset
sys.modules['dataset'] = data.dataset

from models import CNN1, CNN2U, CNN2S, CNN2C, TFMU, TFMC
from utils.common import init_random_seed, count_parameters, get_device
from data.dataset import CustomDataset

# Data directories (relative to DeepLogit folder)
DATA_DIR = './data/'
PROCESSED_DIR = './data/processed/'
RESULTS_DIR = './results/'


class FareClampedCNN1(CNN1):
    """CNN1 with fare beta clamped to -1."""
    
    def __init__(self, num_features: int, use_bias: bool = False):
        super().__init__(num_features, use_bias)
        # After initialization, clamp fare weight
        if num_features >= 2:
            with torch.no_grad():
                self.conv.weight.data[0, 0, 0, 1] = -1.0  # Fare is the second feature
    
    def clamp_fare_weight(self):
        """Clamp fare weight to -1 after each optimization step."""
        with torch.no_grad():
            if self.num_features >= 2:
                self.conv.weight.data[0, 0, 0, 1] = -1.0


def load_processed_data(combi="0 1 2 3_4"):
    """Load the processed data for a given train/valid split."""
    print(f"Loading data for combination: {combi}")
    
    # Check if files exist (with or without _cpu suffix)
    train_file = f'{PROCESSED_DIR}/train_datasets_{combi}.pkl'
    valid_file = f'{PROCESSED_DIR}/valid_datasets_{combi}.pkl'
    test_file = f'{PROCESSED_DIR}/test_datasets_{combi}.pkl'
    
    sampler_train_file = f'{PROCESSED_DIR}/sampler_train_{combi}.pkl'
    sampler_valid_file = f'{PROCESSED_DIR}/sampler_valid_{combi}.pkl'
    sampler_test_file = f'{PROCESSED_DIR}/sampler_test_{combi}.pkl'
    
    # Check for _cpu suffix versions
    if not os.path.exists(train_file):
        train_file = f'{PROCESSED_DIR}/train_datasets_{combi}_cpu.pkl'
        valid_file = f'{PROCESSED_DIR}/valid_datasets_{combi}_cpu.pkl'
        test_file = f'{PROCESSED_DIR}/test_datasets_{combi}_cpu.pkl'
        
        sampler_train_file = f'{PROCESSED_DIR}/sampler_train_{combi}_cpu.pkl'
        sampler_valid_file = f'{PROCESSED_DIR}/sampler_valid_{combi}_cpu.pkl'
        sampler_test_file = f'{PROCESSED_DIR}/sampler_test_{combi}_cpu.pkl'
    
    # Check for required files
    required_files = [train_file, valid_file, test_file, sampler_train_file, sampler_valid_file, sampler_test_file]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("\nError: The following required data files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure you have run the data preprocessing pipeline first.")
        raise FileNotFoundError(f"Missing required data files: {missing_files}")
    
    # Load datasets using torch.load
    train_datasets = torch.load(train_file, map_location='cpu')
    valid_datasets = torch.load(valid_file, map_location='cpu')
    test_datasets = torch.load(test_file, map_location='cpu')
    
    # Load samplers
    sampler_train = torch.load(sampler_train_file, map_location='cpu')
    sampler_valid = torch.load(sampler_valid_file, map_location='cpu')
    sampler_test = torch.load(sampler_test_file, map_location='cpu')
    
    return train_datasets, valid_datasets, test_datasets, sampler_train, sampler_valid, sampler_test


def train_model(model, train_datasets, valid_datasets, sampler_train, sampler_valid, 
                device, n_epochs, batch_size, lr, model_name, use_fare_clamp=False,
                save_dir=RESULTS_DIR, combi=""):
    """Train a model and return training history."""
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []
    valid_epochs = []  # Track which epochs we validated on
    
    best_valid_acc = 0
    best_model_state = None
    best_epoch = 0
    
    for epoch in tqdm.tqdm(range(1, n_epochs + 1), desc=f"Training {model_name}"):
        # Train
        model.train()
        train_loss_epoch = []
        train_correct_epoch = []
        train_total_epoch = []
        
        for k in range(len(train_datasets)):
            train_loader = DataLoader(train_datasets[k], batch_size=batch_size, sampler=sampler_train[k])
            
            for batch in train_loader:
                # Handle different feature counts
                if '4' in model_name and 'CNN' in model_name:
                    routes = batch[0][:, :, :4, :].float().to(device)  # Only first 4 features
                else:
                    routes = batch[0].float().to(device)  # All features
                    
                choices = batch[1].to(device)
                
                optimizer.zero_grad()
                outputs = model(routes)
                loss = criterion(outputs, choices)
                loss.backward()
                optimizer.step()
                
                # Apply fare clamping if needed
                if use_fare_clamp and hasattr(model, 'clamp_fare_weight'):
                    model.clamp_fare_weight()
                elif use_fare_clamp and hasattr(model, 'apply_constraint'):
                    # For CNN2C/TFMC
                    base_weights = model.base_weights.clone()
                    base_weights[1] = -1.0  # Clamp fare
                    model.apply_constraint(base_weights)
                
                train_loss_epoch.append(loss.item() * choices.size(0))
                _, predicted = outputs.max(1)
                train_correct_epoch.append(predicted.eq(choices).sum().item())
                train_total_epoch.append(choices.size(0))
        
        train_loss = np.sum(train_loss_epoch) / np.sum(train_total_epoch)
        train_acc = np.sum(train_correct_epoch) / np.sum(train_total_epoch)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        
        # Validate every 100 epochs for long training, or every 5 for short
        validate_freq = 100 if n_epochs > 1000 else 5
        if epoch % validate_freq == 0 or epoch == n_epochs:
            model.eval()
            valid_loss_epoch = []
            valid_correct_epoch = []
            valid_total_epoch = []
            
            with torch.no_grad():
                for k in range(len(valid_datasets)):
                    valid_loader = DataLoader(valid_datasets[k], batch_size=batch_size, sampler=sampler_valid[k])
                    
                    for batch in valid_loader:
                        if '4' in model_name and 'CNN' in model_name:
                            routes = batch[0][:, :, :4, :].float().to(device)
                        else:
                            routes = batch[0].float().to(device)
                            
                        choices = batch[1].to(device)
                        outputs = model(routes)
                        loss = criterion(outputs, choices)
                        
                        valid_loss_epoch.append(loss.item() * choices.size(0))
                        _, predicted = outputs.max(1)
                        valid_correct_epoch.append(predicted.eq(choices).sum().item())
                        valid_total_epoch.append(choices.size(0))
            
            valid_loss = np.sum(valid_loss_epoch) / np.sum(valid_total_epoch)
            valid_acc = np.sum(valid_correct_epoch) / np.sum(valid_total_epoch)
            valid_loss_history.append(valid_loss)
            valid_acc_history.append(valid_acc)
            valid_epochs.append(epoch)
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_epoch = epoch
                best_model_state = model.state_dict()
                # Save best model
                torch.save(model.state_dict(), f'{save_dir}/{model_name}_best_{combi}.pt')
            
            # Print progress
            print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
            print(f"         Valid Loss={valid_loss:.4f}, Valid Acc={valid_acc:.4f} (Best: {best_valid_acc:.4f} @ epoch {best_epoch})")
            
            # Save checkpoint every 1000 epochs
            if epoch % 1000 == 0:
                torch.save(model.state_dict(), f'{save_dir}/{model_name}_epoch{epoch}_{combi}.pt')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save training history
    history = {
        'train_loss_history': train_loss_history,
        'train_acc_history': train_acc_history,
        'valid_loss_history': valid_loss_history,
        'valid_acc_history': valid_acc_history,
        'valid_epochs': valid_epochs,
        'best_epoch': best_epoch,
        'best_valid_acc': best_valid_acc
    }
    
    # Save history to numpy files
    np.save(f'{save_dir}/{model_name}_train_loss_history_{combi}.npy', train_loss_history)
    np.save(f'{save_dir}/{model_name}_train_acc_history_{combi}.npy', train_acc_history)
    np.save(f'{save_dir}/{model_name}_valid_loss_history_{combi}.npy', valid_loss_history)
    np.save(f'{save_dir}/{model_name}_valid_acc_history_{combi}.npy', valid_acc_history)
    np.save(f'{save_dir}/{model_name}_valid_epochs_{combi}.npy', valid_epochs)
    
    # Also save as pickle for complete history
    with open(f'{save_dir}/{model_name}_history_{combi}.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    return {
        'train_loss': train_loss_history[-1],
        'train_acc': train_acc_history[-1],
        'valid_loss': valid_loss_history[-1] if valid_loss_history else train_loss_history[-1],
        'valid_acc': valid_acc_history[-1] if valid_acc_history else train_acc_history[-1],
        'model': model,
        'history': history
    }


def evaluate_model(model, test_datasets, sampler_test, device, batch_size, model_name):
    """Evaluate model on test set."""
    
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    test_loss_total = []
    test_correct_total = []
    test_total = []
    
    with torch.no_grad():
        for k in range(len(test_datasets)):
            test_loader = DataLoader(test_datasets[k], batch_size=batch_size, sampler=sampler_test[k])
            
            for batch in test_loader:
                if '4' in model_name and 'CNN' in model_name:
                    routes = batch[0][:, :, :4, :].float().to(device)
                else:
                    routes = batch[0].float().to(device)
                    
                choices = batch[1].to(device)
                outputs = model(routes)
                loss = criterion(outputs, choices)
                
                test_loss_total.append(loss.item() * choices.size(0))
                _, predicted = outputs.max(1)
                test_correct_total.append(predicted.eq(choices).sum().item())
                test_total.append(choices.size(0))
    
    test_loss = np.sum(test_loss_total) / np.sum(test_total)
    test_acc = np.sum(test_correct_total) / np.sum(test_total)
    
    return test_loss, test_acc


def main(combinations, n_epochs_cnn1=50, n_epochs_cnn2=20000, n_epochs_tfm=20000):
    """Run all experiments and generate LaTeX tables."""
    
    # Initialize
    init_random_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Start time: {datetime.now()}")
    print(f"CNN1 epochs: {n_epochs_cnn1}")
    print(f"CNN2 epochs: {n_epochs_cnn2}")
    print(f"Transformer epochs: {n_epochs_tfm}")
    lr = 0.001
    
    # Model-specific batch sizes to avoid OOM
    batch_sizes = {
        'CNN1_4': 512,      # CNN1 with 4 features
        'CNN1_97': 512,     # CNN1 with 97 features
        'CNN2U_4': 1024,    # CNN2U with 4 features
        'CNN2U_97': 1536,   # CNN2U with 97 features (quadratic features)
        'CNN2S_97': 1664,   # CNN2S with 97 features (separated)
        'CNN2C_97': 2048,   # CNN2C with 97 features (constrained)
        'TFMU_97': 256,     # Transformer unconstrained
        'TFMC_97': 384      # Transformer constrained
    }
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Store results for all models and combinations
    all_results = defaultdict(dict)
    model_params = {}
    interpretable_weights = {}
    
    for combi in combinations:
        print(f"\n{'='*70}")
        print(f"Training models for combination: {combi}")
        print(f"{'='*70}")
        
        # Load data
        train_datasets, valid_datasets, test_datasets, sampler_train, sampler_valid, sampler_test = load_processed_data(combi)
        
        # 1. Train CNN1 (4 features) with fare clamping
        print(f"\nTraining CNN1 (4 features) - batch_size={batch_sizes['CNN1_4']}...")
        cnn1_4 = FareClampedCNN1(num_features=4).to(device)
        results = train_model(cnn1_4, train_datasets, valid_datasets, sampler_train, sampler_valid,
                            device, n_epochs_cnn1, batch_sizes['CNN1_4'], lr, "CNN1_4", use_fare_clamp=True, combi=combi)
        test_loss, test_acc = evaluate_model(results['model'], test_datasets, sampler_test, 
                                            device, batch_sizes['CNN1_4'], "CNN1_4")
        
        all_results[combi]['CNN1_4'] = {
            'train_loss': results['train_loss'],
            'train_acc': results['train_acc'],
            'valid_loss': results['valid_loss'],
            'valid_acc': results['valid_acc'],
            'test_loss': test_loss,
            'test_acc': test_acc
        }
        model_params['CNN1_4'] = count_parameters(cnn1_4)
        
        # Get interpretable weights
        base_weights = cnn1_4.get_feature_weights()
        base_weights[1] = -1.0  # Ensure fare is -1
        interpretable_weights[combi] = base_weights.clone()
        
        print(f"Learned weights: IVTT={base_weights[0]:.4f}, Fare={base_weights[1]:.4f}, "
              f"WT={base_weights[2]:.4f}, NoT={base_weights[3]:.4f}")
        
        # 2. Train CNN1 (97 features) with fare clamping
        print(f"\nTraining CNN1 (97 features) - batch_size={batch_sizes['CNN1_97']}...")
        cnn1_97 = FareClampedCNN1(num_features=97).to(device)
        results = train_model(cnn1_97, train_datasets, valid_datasets, sampler_train, sampler_valid,
                            device, n_epochs_cnn1, batch_sizes['CNN1_97'], lr, "CNN1_97", use_fare_clamp=True, combi=combi)
        test_loss, test_acc = evaluate_model(results['model'], test_datasets, sampler_test, 
                                           device, batch_sizes['CNN1_97'], "CNN1_97")
        
        all_results[combi]['CNN1_97'] = {
            'train_loss': results['train_loss'],
            'train_acc': results['train_acc'],
            'valid_loss': results['valid_loss'],
            'valid_acc': results['valid_acc'],
            'test_loss': test_loss,
            'test_acc': test_acc
        }
        model_params['CNN1_97'] = count_parameters(cnn1_97)
        
        # 3. Train CNN2U (4 features) - unconstrained
        print(f"\nTraining CNN2U (4 features) - batch_size={batch_sizes['CNN2U_4']}...")
        cnn2u_4 = CNN2U(num_features=4).to(device)
        results = train_model(cnn2u_4, train_datasets, valid_datasets, sampler_train, sampler_valid,
                            device, n_epochs_cnn2, batch_sizes['CNN2U_4'], lr, "CNN2U_4", use_fare_clamp=False, combi=combi)
        test_loss, test_acc = evaluate_model(results['model'], test_datasets, sampler_test, 
                                           device, batch_sizes['CNN2U_4'], "CNN2U_4")
        
        all_results[combi]['CNN2U_4'] = {
            'train_loss': results['train_loss'],
            'train_acc': results['train_acc'],
            'valid_loss': results['valid_loss'],
            'valid_acc': results['valid_acc'],
            'test_loss': test_loss,
            'test_acc': test_acc
        }
        model_params['CNN2U_4'] = count_parameters(cnn2u_4)
        
        # 4. Train CNN2U (97 features) - unconstrained
        print(f"\nTraining CNN2U (97 features) - batch_size={batch_sizes['CNN2U_97']}...")
        cnn2u_97 = CNN2U(num_features=97).to(device)
        results = train_model(cnn2u_97, train_datasets, valid_datasets, sampler_train, sampler_valid,
                            device, n_epochs_cnn2, batch_sizes['CNN2U_97'], lr, "CNN2U_97", use_fare_clamp=False, combi=combi)
        test_loss, test_acc = evaluate_model(results['model'], test_datasets, sampler_test, 
                                           device, batch_sizes['CNN2U_97'], "CNN2U_97")
        
        all_results[combi]['CNN2U_97'] = {
            'train_loss': results['train_loss'],
            'train_acc': results['train_acc'],
            'valid_loss': results['valid_loss'],
            'valid_acc': results['valid_acc'],
            'test_loss': test_loss,
            'test_acc': test_acc
        }
        model_params['CNN2U_97'] = count_parameters(cnn2u_97)
        
        # 5. Train CNN2S (97 features) - separated, with fare clamping
        print(f"\nTraining CNN2S (97 features) - batch_size={batch_sizes['CNN2S_97']}...")
        cnn2s = CNN2S(num_features=97).to(device)
        # Add method to clamp fare weight
        def clamp_fare_weight_cnn2s(self):
            with torch.no_grad():
                if self.num_features >= 2:
                    self.conv.weight.data[0, 0, 0, 1] = -1.0
        cnn2s.clamp_fare_weight = lambda: clamp_fare_weight_cnn2s(cnn2s)
        cnn2s.clamp_fare_weight()  # Apply initial clamp
        results = train_model(cnn2s, train_datasets, valid_datasets, sampler_train, sampler_valid,
                            device, n_epochs_cnn2, batch_sizes['CNN2S_97'], lr, "CNN2S_97", use_fare_clamp=True, combi=combi)
        test_loss, test_acc = evaluate_model(results['model'], test_datasets, sampler_test, 
                                           device, batch_sizes['CNN2S_97'], "CNN2S_97")
        
        all_results[combi]['CNN2S_97'] = {
            'train_loss': results['train_loss'],
            'train_acc': results['train_acc'],
            'valid_loss': results['valid_loss'],
            'valid_acc': results['valid_acc'],
            'test_loss': test_loss,
            'test_acc': test_acc
        }
        model_params['CNN2S_97'] = count_parameters(cnn2s)
        
        # 6. Train CNN2C (97 features) - constrained with interpretable weights
        print(f"\nTraining CNN2C (97 features) - batch_size={batch_sizes['CNN2C_97']}...")
        cnn2c = CNN2C(num_features=97, cnn1_weights=base_weights).to(device)
        cnn2c.base_weights = base_weights  # Store for constraint application
        results = train_model(cnn2c, train_datasets, valid_datasets, sampler_train, sampler_valid,
                            device, n_epochs_cnn2, batch_sizes['CNN2C_97'], lr, "CNN2C_97", use_fare_clamp=True, combi=combi)
        test_loss, test_acc = evaluate_model(results['model'], test_datasets, sampler_test, 
                                           device, batch_sizes['CNN2C_97'], "CNN2C_97")
        
        all_results[combi]['CNN2C_97'] = {
            'train_loss': results['train_loss'],
            'train_acc': results['train_acc'],
            'valid_loss': results['valid_loss'],
            'valid_acc': results['valid_acc'],
            'test_loss': test_loss,
            'test_acc': test_acc
        }
        model_params['CNN2C_97'] = count_parameters(cnn2c)
        
        # 7. Train TFMU (97 features) - unconstrained transformer
        print(f"\nTraining TFMU (97 features) - batch_size={batch_sizes['TFMU_97']}...")
        tfmu = TFMU(num_features=97).to(device)
        results = train_model(tfmu, train_datasets, valid_datasets, sampler_train, sampler_valid,
                            device, n_epochs_tfm, batch_sizes['TFMU_97'], lr, "TFMU_97", use_fare_clamp=False, combi=combi)
        test_loss, test_acc = evaluate_model(results['model'], test_datasets, sampler_test, 
                                           device, batch_sizes['TFMU_97'], "TFMU_97")
        
        all_results[combi]['TFMU_97'] = {
            'train_loss': results['train_loss'],
            'train_acc': results['train_acc'],
            'valid_loss': results['valid_loss'],
            'valid_acc': results['valid_acc'],
            'test_loss': test_loss,
            'test_acc': test_acc
        }
        model_params['TFMU_97'] = count_parameters(tfmu)
        
        # 8. Train TFMC (97 features) - constrained transformer
        print(f"\nTraining TFMC (97 features) - batch_size={batch_sizes['TFMC_97']}...")
        tfmc = TFMC(num_features=97, cnn1_weights=base_weights).to(device)
        tfmc.base_weights = base_weights  # Store for constraint application
        results = train_model(tfmc, train_datasets, valid_datasets, sampler_train, sampler_valid,
                            device, n_epochs_tfm, batch_sizes['TFMC_97'], lr, "TFMC_97", use_fare_clamp=True, combi=combi)
        test_loss, test_acc = evaluate_model(results['model'], test_datasets, sampler_test, 
                                           device, batch_sizes['TFMC_97'], "TFMC_97")
        
        all_results[combi]['TFMC_97'] = {
            'train_loss': results['train_loss'],
            'train_acc': results['train_acc'],
            'valid_loss': results['valid_loss'],
            'valid_acc': results['valid_acc'],
            'test_loss': test_loss,
            'test_acc': test_acc
        }
        model_params['TFMC_97'] = count_parameters(tfmc)
    
    # Generate LaTeX tables
    print("\n" + "="*70)
    print("GENERATING LATEX TABLES")
    print("="*70)
    
    # Table 1: Performance comparison (Tb_NN)
    print("\n% Table Tb_NN: Neural Network Performance Comparison")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Neural Network Model Performance Comparison}")
    print("\\label{tab:Tb_NN}")
    print("\\begin{tabular}{l c c c c c c}")
    print("\\toprule")
    print("Model & Parameters & Features & Train Loss & Val Loss & Train Acc & Val Acc \\\\")
    print("\\midrule")
    
    # Average results across all combinations for each model
    models = ['CNN1_4', 'CNN1_97', 'CNN2U_4', 'CNN2U_97', 'CNN2S_97', 'CNN2C_97', 'TFMU_97', 'TFMC_97']
    model_display_names = {
        'CNN1_4': 'CNN 1',
        'CNN1_97': 'CNN 1',
        'CNN2U_4': 'CNN 2U',
        'CNN2U_97': 'CNN 2U',
        'CNN2S_97': 'CNN 2S',
        'CNN2C_97': 'CNN 2C',
        'TFMU_97': 'TFM U',
        'TFMC_97': 'TFM C'
    }
    
    for model in models:
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        
        for combi in combinations:
            if model in all_results[combi]:
                train_losses.append(all_results[combi][model]['train_loss'])
                valid_losses.append(all_results[combi][model]['valid_loss'])
                train_accs.append(all_results[combi][model]['train_acc'])
                valid_accs.append(all_results[combi][model]['valid_acc'])
        
        if train_losses:
            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_train_acc = np.mean(train_accs)
            avg_valid_acc = np.mean(valid_accs)
            
            features = int(model.split('_')[1])
            display_name = model_display_names[model]
            
            print(f"{display_name} & {model_params[model]} & {features} & "
                  f"{avg_train_loss:.4f} & {avg_valid_loss:.4f} & "
                  f"{avg_train_acc:.4f} & {avg_valid_acc:.4f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Table 2: Interpretable weights comparison (Tb_Compare)
    print("\n% Table Tb_Compare: Interpretable Weights Comparison")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Learned Interpretable Parameters}")
    print("\\label{tab:Tb_Compare}")
    print("\\begin{tabular}{l c c c c c}")
    print("\\toprule")
    print("Model & Feature Cnt & $\\beta_{IVTT}$ & $\\beta_{Fare}$ & $\\beta_{WT}$ & $\\beta_{NoT}$ \\\\")
    print("\\midrule")
    
    # Models with interpretable weights
    interp_models = [
        ('CNN1_4', 'CNN 1', 4),
        ('CNN1_97', 'CNN 1', 97),
        ('CNN2C_97', 'CNN 2C', 97),
        ('TFMC_97', 'TFM C', 97)
    ]
    
    for _, display_name, features in interp_models:
        # Get average weights across combinations
        weights_list = []
        for combi in combinations:
            if combi in interpretable_weights:
                weights_list.append(interpretable_weights[combi].cpu().numpy())
        
        if weights_list:
            avg_weights = np.mean(weights_list, axis=0)
            print(f"{display_name} & {features} & {avg_weights[0]:.4f} & "
                  f"{avg_weights[1]:.4f} & {avg_weights[2]:.4f} & {avg_weights[3]:.4f} \\\\")
    
    # Add unconstrained models (these don't have fixed interpretable weights)
    print("CNN 2U & 4 & - & - & - & - \\\\")
    print("CNN 2U & 97 & - & - & - & - \\\\")
    print("CNN 2S & 97 & - & - & - & - \\\\")
    print("TFM U & 97 & - & - & - & - \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Save detailed results per combination
    print("\n% Detailed results per combination")
    for combi in combinations:
        print(f"\n% Results for combination {combi}")
        for model in models:
            if model in all_results[combi]:
                result = all_results[combi][model]
                print(f"% {model}: Test Loss={result['test_loss']:.4f}, Test Acc={result['test_acc']:.4f}")
    
    # Save results to file
    with open(f"{RESULTS_DIR}/latex_tables.txt", "w") as f:
        # Write Tb_NN table
        f.write("% Table Tb_NN: Neural Network Performance Comparison\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Neural Network Model Performance Comparison}\n")
        f.write("\\label{tab:Tb_NN}\n")
        f.write("\\begin{tabular}{l c c c c c c}\n")
        f.write("\\toprule\n")
        f.write("Model & Parameters & Features & Train Loss & Val Loss & Train Acc & Val Acc \\\\\n")
        f.write("\\midrule\n")
        
        for model in models:
            train_losses = []
            valid_losses = []
            train_accs = []
            valid_accs = []
            
            for combi in combinations:
                if model in all_results[combi]:
                    train_losses.append(all_results[combi][model]['train_loss'])
                    valid_losses.append(all_results[combi][model]['valid_loss'])
                    train_accs.append(all_results[combi][model]['train_acc'])
                    valid_accs.append(all_results[combi][model]['valid_acc'])
            
            if train_losses:
                avg_train_loss = np.mean(train_losses)
                avg_valid_loss = np.mean(valid_losses)
                avg_train_acc = np.mean(train_accs)
                avg_valid_acc = np.mean(valid_accs)
                
                features = int(model.split('_')[1])
                display_name = model_display_names[model]
                
                f.write(f"{display_name} & {model_params[model]} & {features} & "
                      f"{avg_train_loss:.4f} & {avg_valid_loss:.4f} & "
                      f"{avg_train_acc:.4f} & {avg_valid_acc:.4f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Write Tb_Compare table
        f.write("% Table Tb_Compare: Interpretable Weights Comparison\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Learned Interpretable Parameters}\n")
        f.write("\\label{tab:Tb_Compare}\n")
        f.write("\\begin{tabular}{l c c c c c}\n")
        f.write("\\toprule\n")
        f.write("Model & Feature Cnt & $\\beta_{IVTT}$ & $\\beta_{Fare}$ & $\\beta_{WT}$ & $\\beta_{NoT}$ \\\\\n")
        f.write("\\midrule\n")
        
        for _, display_name, features in interp_models:
            weights_list = []
            for combi in combinations:
                if combi in interpretable_weights:
                    weights_list.append(interpretable_weights[combi].cpu().numpy())
            
            if weights_list:
                avg_weights = np.mean(weights_list, axis=0)
                f.write(f"{display_name} & {features} & {avg_weights[0]:.4f} & "
                      f"{avg_weights[1]:.4f} & {avg_weights[2]:.4f} & {avg_weights[3]:.4f} \\\\\n")
        
        f.write("CNN 2U & 4 & - & - & - & - \\\\\n")
        f.write("CNN 2U & 97 & - & - & - & - \\\\\n")
        f.write("CNN 2S & 97 & - & - & - & - \\\\\n")
        f.write("TFM U & 97 & - & - & - & - \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\nTables saved to: {RESULTS_DIR}/latex_tables.txt")
    print(f"\nTraining completed at: {datetime.now()}")
    print(f"All training histories saved to: {RESULTS_DIR}")
    
    # Print summary of saved files
    print("\nSaved files for each model and combination:")
    print("- {model}_best_{combi}.pt - Best model checkpoint")
    print("- {model}_train_loss_history_{combi}.npy - Training loss history")
    print("- {model}_train_acc_history_{combi}.npy - Training accuracy history")
    print("- {model}_valid_loss_history_{combi}.npy - Validation loss history")
    print("- {model}_valid_acc_history_{combi}.npy - Validation accuracy history")
    print("- {model}_valid_epochs_{combi}.npy - Epochs when validation was performed")
    print("- {model}_history_{combi}.pkl - Complete history dictionary")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train all DeepLogit models and generate LaTeX tables')
    parser.add_argument('--combinations', type=str, nargs='+', 
                        default=['0 1 2 3_4', '0 1 2 4_3', '0 1 3 4_2', '0 2 3 4_1', '1 2 3 4_0'],
                        help='List of train/validation splits')
    parser.add_argument('--epochs_cnn1', type=int, default=50,
                        help='Number of epochs for CNN1 models (default: 50)')
    parser.add_argument('--epochs_cnn2', type=int, default=20000,
                        help='Number of epochs for CNN2 models (default: 20000)')
    parser.add_argument('--epochs_tfm', type=int, default=20000,
                        help='Number of epochs for Transformer models (default: 20000)')
    parser.add_argument('--test_run', action='store_true',
                        help='Run a quick test with 1 epoch for each model')
    args = parser.parse_args()
    
    # Override epochs for test run
    if args.test_run:
        print("TEST RUN MODE: Using 1 epoch for all models")
        args.epochs_cnn1 = 1
        args.epochs_cnn2 = 1
        args.epochs_tfm = 1
    
    # Pass arguments to main
    main(args.combinations, args.epochs_cnn1, args.epochs_cnn2, args.epochs_tfm)