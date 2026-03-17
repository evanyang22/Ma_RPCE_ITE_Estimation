"""
Example Usage: Training and Evaluating TSPF Models

This script demonstrates how to use the TSPF models for treatment effect estimation.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tspf_implementation import (
    TSPFDataset, TSPFModel, TSPFModelOT,
    train_stage1, train_stage2, initialize_stage2_from_stage1,
    evaluate_model
)


# ============================================================================
# Example 1: Basic Usage with Standard TSPF
# ============================================================================

def example_standard_tspf():
    """Train and evaluate standard TSPF model"""
    print("="*80)
    print("EXAMPLE 1: Standard TSPF Model")
    print("="*80)
    
    # Load data
    train_data = np.load('jobs_DW_bin_new_10_train.npz')
    test_data = np.load('jobs_DW_bin_new_10_test.npz')
    
    # Create datasets
    # Assume first 70% of training data is observational, rest is RCT
    x_train = train_data['x']
    t_train = train_data['t']
    yf_train = train_data['yf']
    n_samples = x_train.shape[0]
    n_obs = int(0.7 * n_samples)
    
    e_train = np.zeros_like(t_train)
    e_train[n_obs:, :] = 1  # Mark RCT data
    
    train_dataset = TSPFDataset(x_train, t_train, yf_train, e_train)
    test_dataset = TSPFDataset(test_data['x'], test_data['t'], test_data['yf'])
    
    # Split train/val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Get input dimension
    input_dim = x_train.shape[1]  # 17 features
    
    # ========== Stage 1 ==========
    print("\n--- Stage 1: Pretraining ---")
    model_s1 = TSPFModel(input_dim=input_dim, repr_dims=[64, 32], stage=1)
    
    history_s1 = train_stage1(
        model=model_s1,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        lr=1e-3,
        weight_decay=1e-4,
        lambda_rec=0.1,
        use_ot=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # ========== Stage 2 ==========
    print("\n--- Stage 2: Fine-tuning ---")
    model_s2 = TSPFModel(
        input_dim=input_dim, 
        repr_dims=[64, 32], 
        adapter_dim=16, 
        stage=2
    )
    
    # Initialize from Stage 1
    initialize_stage2_from_stage1(model_s1, model_s2)
    
    history_s2 = train_stage2(
        model=model_s2,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        lr=5e-4,
        weight_decay=1e-4,
        lambda_shift=0.1,
        lambda_mi=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # ========== Evaluation ==========
    print("\n--- Evaluation ---")
    results = evaluate_model(model_s2, test_loader)
    
    print(f"\nResults:")
    print(f"  Average Treatment Effect: {results['ate_pred']:.4f}")
    print(f"  ITE std: {np.std(results['ite_pred']):.4f}")
    
    # Save model
    torch.save(model_s2.state_dict(), 'tspf_model.pt')
    print("\nModel saved to tspf_model.pt")
    
    return model_s2, results


# ============================================================================
# Example 2: TSPF with Optimal Transport
# ============================================================================

def example_tspf_ot():
    """Train and evaluate TSPF-OT model"""
    print("\n\n" + "="*80)
    print("EXAMPLE 2: TSPF with Optimal Transport")
    print("="*80)
    
    # Load data (same as Example 1)
    train_data = np.load('jobs_DW_bin_new_10_train.npz')
    test_data = np.load('jobs_DW_bin_new_10_test.npz')
    
    x_train = train_data['x']
    t_train = train_data['t']
    yf_train = train_data['yf']
    n_samples = x_train.shape[0]
    n_obs = int(0.7 * n_samples)
    
    e_train = np.zeros_like(t_train)
    e_train[n_obs:, :] = 1
    
    train_dataset = TSPFDataset(x_train, t_train, yf_train, e_train)
    test_dataset = TSPFDataset(test_data['x'], test_data['t'], test_data['yf'])
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    input_dim = x_train.shape[1]
    
    # ========== Stage 1 with OT ==========
    print("\n--- Stage 1: Pretraining with OT ---")
    model_s1 = TSPFModelOT(
        input_dim=input_dim,
        repr_dims=[64, 32],
        stage=1,
        ot_kappa=0.5,      # Balance global and local
        isp_ratio=0.7      # Keep 70% of dimensions
    )
    
    history_s1 = train_stage1(
        model=model_s1,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        lr=1e-3,
        weight_decay=1e-4,
        lambda_rec=0.1,
        lambda_ot=1.0,     # Weight for OT loss
        use_ot=True,       # Enable OT
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # ========== Stage 2 ==========
    print("\n--- Stage 2: Fine-tuning ---")
    model_s2 = TSPFModelOT(
        input_dim=input_dim,
        repr_dims=[64, 32],
        adapter_dim=16,
        stage=2,
        ot_kappa=0.5,
        isp_ratio=0.7
    )
    
    initialize_stage2_from_stage1(model_s1, model_s2)
    
    history_s2 = train_stage2(
        model=model_s2,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        lr=5e-4,
        weight_decay=1e-4,
        lambda_shift=0.1,
        lambda_mi=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # ========== Evaluation ==========
    print("\n--- Evaluation ---")
    results = evaluate_model(model_s2, test_loader)
    
    print(f"\nResults:")
    print(f"  Average Treatment Effect: {results['ate_pred']:.4f}")
    print(f"  ITE std: {np.std(results['ite_pred']):.4f}")
    
    # Save model
    torch.save(model_s2.state_dict(), 'tspf_ot_model.pt')
    print("\nModel saved to tspf_ot_model.pt")
    
    return model_s2, results


# ============================================================================
# Example 3: Prediction on New Data
# ============================================================================

def example_prediction():
    """Load saved model and make predictions"""
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Making Predictions with Saved Model")
    print("="*80)
    
    # Create a dummy model architecture (must match saved model)
    input_dim = 17
    model = TSPFModelOT(
        input_dim=input_dim,
        repr_dims=[64, 32],
        adapter_dim=16,
        stage=2
    )
    
    # Load weights
    model.load_state_dict(torch.load('tspf_ot_model.pt'))
    model.eval()
    
    # Generate some random test data
    x_new = torch.randn(10, input_dim)
    
    # Predict ITE
    with torch.no_grad():
        ite = model.predict_ite(x_new)
    
    print("\nPredicted Individual Treatment Effects:")
    for i, effect in enumerate(ite):
        print(f"  Unit {i+1}: {effect.item():.4f}")
    
    # Predict for specific treatment
    t_new = torch.ones(10, 1)  # All treated
    with torch.no_grad():
        y_pred, outputs = model(x_new, t_new)
    
    print("\nPredicted Outcomes for Treated Units:")
    for i, outcome in enumerate(y_pred):
        print(f"  Unit {i+1}: {outcome.item():.4f}")
    
    return model, ite


# ============================================================================
# Example 4: Hyperparameter Tuning
# ============================================================================

def example_hyperparameter_tuning():
    """Example of tuning key hyperparameters"""
    print("\n\n" + "="*80)
    print("EXAMPLE 4: Hyperparameter Tuning for TSPF-OT")
    print("="*80)
    
    # Load data
    train_data = np.load('jobs_DW_bin_new_10_train.npz')
    x_train = train_data['x']
    t_train = train_data['t']
    yf_train = train_data['yf']
    
    # Create minimal dataset for quick experiments
    n_obs = int(0.7 * x_train.shape[0])
    e_train = np.zeros_like(t_train)
    e_train[n_obs:, :] = 1
    
    dataset = TSPFDataset(x_train, t_train, yf_train, e_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
    
    input_dim = x_train.shape[1]
    
    # Grid search over key parameters
    kappa_values = [0.3, 0.5, 0.7]
    isp_ratios = [0.5, 0.7, 0.9]
    lambda_ot_values = [0.5, 1.0, 2.0]
    
    best_val_loss = float('inf')
    best_params = {}
    
    print("\nSearching hyperparameters...")
    print(f"  kappa: {kappa_values}")
    print(f"  isp_ratio: {isp_ratios}")
    print(f"  lambda_ot: {lambda_ot_values}")
    
    for kappa in kappa_values:
        for isp_ratio in isp_ratios:
            for lambda_ot in lambda_ot_values:
                print(f"\nTrying: kappa={kappa}, isp_ratio={isp_ratio}, lambda_ot={lambda_ot}")
                
                model = TSPFModelOT(
                    input_dim=input_dim,
                    repr_dims=[64, 32],
                    stage=1,
                    ot_kappa=kappa,
                    isp_ratio=isp_ratio
                )
                
                # Train for fewer epochs during search
                history = train_stage1(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    n_epochs=30,  # Reduced for quick search
                    lr=1e-3,
                    weight_decay=1e-4,
                    lambda_rec=0.1,
                    lambda_ot=lambda_ot,
                    use_ot=True,
                    device='cpu'  # Use CPU for demo
                )
                
                # Get final validation loss
                final_val_loss = history['val_loss'][-1]
                
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    best_params = {
                        'kappa': kappa,
                        'isp_ratio': isp_ratio,
                        'lambda_ot': lambda_ot
                    }
    
    print("\n" + "="*60)
    print("Best Hyperparameters Found:")
    print(f"  kappa: {best_params['kappa']}")
    print(f"  isp_ratio: {best_params['isp_ratio']}")
    print(f"  lambda_ot: {best_params['lambda_ot']}")
    print(f"  Validation Loss: {best_val_loss:.4f}")
    print("="*60)
    
    return best_params


# ============================================================================
# Example 5: Comparing Both Models
# ============================================================================

def example_model_comparison():
    """Compare Standard TSPF vs TSPF-OT"""
    print("\n\n" + "="*80)
    print("EXAMPLE 5: Comparing Standard TSPF vs TSPF-OT")
    print("="*80)
    
    # Train both models (simplified version)
    print("\nThis would train both models and compare:")
    print("  - Factual prediction accuracy")
    print("  - ATE estimation")
    print("  - ITE distribution")
    print("  - Representation balance (e.g., MMD between groups)")
    print("\nSee train_models.py for full implementation")
    
    # Pseudo-code for comparison
    """
    results_standard = {
        'factual_mse': ...,
        'ate': ...,
        'ite_std': ...,
        'mmd_before': ...,
        'mmd_after': ...
    }
    
    results_ot = {
        'factual_mse': ...,
        'ate': ...,
        'ite_std': ...,
        'mmd_before': ...,
        'mmd_after': ...
    }
    
    print("Standard TSPF:")
    print(f"  Factual MSE: {results_standard['factual_mse']:.4f}")
    print(f"  MMD reduction: {results_standard['mmd_before']:.4f} → {results_standard['mmd_after']:.4f}")
    
    print("\nTSPF-OT:")
    print(f"  Factual MSE: {results_ot['factual_mse']:.4f}")
    print(f"  MMD reduction: {results_ot['mmd_before']:.4f} → {results_ot['mmd_after']:.4f}")
    
    print("\nExpected: TSPF-OT should show:")
    print("  - Lower MMD (better balance)")
    print("  - Similar or slightly higher factual MSE (tradeoff)")
    print("  - More reliable ITE estimates (when selection bias is high)")
    """


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TSPF MODELS - EXAMPLE USAGE")
    print("="*80)
    print("\nThis script demonstrates various use cases of TSPF models.")
    print("To run with actual data, make sure you have:")
    print("  - jobs_DW_bin_new_10_train.npz")
    print("  - jobs_DW_bin_new_10_test.npz")
    print("\nAvailable examples:")
    print("  1. example_standard_tspf() - Basic TSPF training")
    print("  2. example_tspf_ot() - TSPF with Optimal Transport")
    print("  3. example_prediction() - Using saved models")
    print("  4. example_hyperparameter_tuning() - HP search")
    print("  5. example_model_comparison() - Compare both models")
    print("\nUncomment the desired example below to run it.")
    print("="*80)
    
    # Uncomment to run examples:
    # model, results = example_standard_tspf()
    # model, results = example_tspf_ot()
    # model, ite = example_prediction()
    # best_params = example_hyperparameter_tuning()
    # example_model_comparison()
