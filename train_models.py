"""
Main training script for TSPF models on Jobs dataset
Trains both standard TSPF and TSPF-OT models
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tspf_implementation import (
    TSPFDataset, TSPFModel, TSPFModelOT,
    train_stage1, train_stage2, initialize_stage2_from_stage1,
    evaluate_model
)


def load_jobs_data(train_path: str, test_path: str):
    """Load and prepare Jobs dataset"""
    print("Loading data...")
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    print("\nDataset information:")
    print(f"Training samples: {train_data['x'].shape[0]}")
    print(f"Test samples: {test_data['x'].shape[0]}")
    print(f"Number of features: {train_data['x'].shape[1]}")
    print(f"Number of realizations: {train_data['x'].shape[2]}")
    
    return train_data, test_data


def prepare_datasets(train_data, test_data, obs_ratio: float = 0.7):
    """
    Prepare datasets for two-stage training
    
    Args:
        train_data: Training data dictionary
        test_data: Test data dictionary
        obs_ratio: Ratio of observational data in training set
        
    Returns:
        Datasets for stage 1 (OBS) and stage 2 (RCT)
    """
    # Extract data
    x_train = train_data['x']
    t_train = train_data['t']
    yf_train = train_data['yf']
    
    x_test = test_data['x']
    t_test = test_data['t']
    yf_test = test_data['yf']
    
    # Create experiment indicators
    # Stage 1 uses observational data (e=0)
    # Stage 2 uses RCT data (e=1)
    n_samples, _, n_realizations = x_train.shape
    n_obs = int(n_samples * obs_ratio)
    
    e_train = np.zeros((n_samples, n_realizations))
    e_train[n_obs:, :] = 1  # Mark last samples as RCT data
    
    # For simplicity, treat test data as RCT
    e_test = np.ones((x_test.shape[0], x_test.shape[2]))
    
    # Create datasets
    train_dataset = TSPFDataset(x_train, t_train, yf_train, e_train)
    test_dataset = TSPFDataset(x_test, t_test, yf_test, e_test)
    
    # Split training into train/val (90/10)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset splits:")
    print(f"Training: {len(train_subset)} samples")
    print(f"Validation: {len(val_subset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    print(f"Observational data ratio: {obs_ratio:.1%}")
    
    return train_subset, val_subset, test_dataset


def plot_training_history(history1, history2, save_path='training_history.png'):
    """Plot training history for both models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Model 1 - Stage 1
    axes[0, 0].plot(history1['stage1']['train_loss'], label='Train Loss', alpha=0.7)
    axes[0, 0].plot(history1['stage1']['val_loss'], label='Val Loss', alpha=0.7)
    axes[0, 0].set_title('Model 1 (Standard TSPF) - Stage 1')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Model 1 - Stage 2
    axes[0, 1].plot(history1['stage2']['train_loss'], label='Train Loss', alpha=0.7)
    axes[0, 1].plot(history1['stage2']['val_loss'], label='Val Loss', alpha=0.7)
    axes[0, 1].set_title('Model 1 (Standard TSPF) - Stage 2')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Model 2 - Stage 1
    axes[1, 0].plot(history2['stage1']['train_loss'], label='Train Loss', alpha=0.7)
    axes[1, 0].plot(history2['stage1']['val_loss'], label='Val Loss', alpha=0.7)
    axes[1, 0].set_title('Model 2 (TSPF-OT) - Stage 1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Model 2 - Stage 2
    axes[1, 1].plot(history2['stage2']['train_loss'], label='Train Loss', alpha=0.7)
    axes[1, 1].plot(history2['stage2']['val_loss'], label='Val Loss', alpha=0.7)
    axes[1, 1].set_title('Model 2 (TSPF-OT) - Stage 2')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining history plot saved to {save_path}")
    plt.close()


def plot_ite_comparison(results1, results2, save_path='ite_comparison.png'):
    """Plot ITE predictions comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ITE distributions
    axes[0].hist(results1['ite_pred'], bins=50, alpha=0.6, label='Standard TSPF', density=True)
    axes[0].hist(results2['ite_pred'], bins=50, alpha=0.6, label='TSPF-OT', density=True)
    axes[0].set_title('ITE Distribution Comparison')
    axes[0].set_xlabel('Predicted ITE')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ITE by treatment
    treated_mask = results1['treatments'].flatten() == 1
    control_mask = results1['treatments'].flatten() == 0
    
    axes[1].scatter(range(np.sum(control_mask)), results1['ite_pred'][control_mask], 
                   alpha=0.3, s=10, label='Control (Model 1)')
    axes[1].scatter(range(np.sum(treated_mask)), results1['ite_pred'][treated_mask], 
                   alpha=0.3, s=10, label='Treated (Model 1)')
    axes[1].set_title('Standard TSPF - ITE by Treatment Group')
    axes[1].set_xlabel('Sample Index (within group)')
    axes[1].set_ylabel('Predicted ITE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(range(np.sum(control_mask)), results2['ite_pred'][control_mask], 
                   alpha=0.3, s=10, label='Control (Model 2)')
    axes[2].scatter(range(np.sum(treated_mask)), results2['ite_pred'][treated_mask], 
                   alpha=0.3, s=10, label='Treated (Model 2)')
    axes[2].set_title('TSPF-OT - ITE by Treatment Group')
    axes[2].set_xlabel('Sample Index (within group)')
    axes[2].set_ylabel('Predicted ITE')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"ITE comparison plot saved to {save_path}")
    plt.close()


def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Paths
    train_path = '/mnt/user-data/uploads/jobs_DW_bin_new_10_train.npz'
    test_path = '/mnt/user-data/uploads/jobs_DW_bin_new_10_test.npz'
    
    # Load data
    train_data, test_data = load_jobs_data(train_path, test_path)
    
    # Prepare datasets
    train_subset, val_subset, test_dataset = prepare_datasets(
        train_data, test_data, obs_ratio=0.7
    )
    
    # Data loaders
    batch_size = 256
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input dimension from data
    input_dim = train_data['x'].shape[1]  # 17 features
    
    print("\n" + "="*80)
    print("TRAINING MODEL 1: Standard TSPF")
    print("="*80)
    
    # Model 1: Standard TSPF
    print("\n--- Stage 1: Pretraining on Observational Data ---")
    model1_stage1 = TSPFModel(
        input_dim=input_dim,
        repr_dims=[64, 32],
        stage=1
    )
    
    history1_stage1 = train_stage1(
        model=model1_stage1,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=150,
        lr=1e-3,
        weight_decay=1e-4,
        lambda_rec=0.1,
        use_ot=False,
        device=device
    )
    
    print("\n--- Stage 2: Fine-tuning on RCT Data ---")
    model1_stage2 = TSPFModel(
        input_dim=input_dim,
        repr_dims=[64, 32],
        adapter_dim=16,
        stage=2
    )
    
    # Initialize from stage 1
    initialize_stage2_from_stage1(model1_stage1, model1_stage2)
    
    history1_stage2 = train_stage2(
        model=model1_stage2,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        lr=5e-4,
        weight_decay=1e-4,
        lambda_shift=0.1,
        lambda_mi=0.01,
        device=device
    )
    
    print("\n" + "="*80)
    print("TRAINING MODEL 2: TSPF with Optimal Transport")
    print("="*80)
    
    # Model 2: TSPF with Optimal Transport
    print("\n--- Stage 1: Pretraining with OT Regularization ---")
    model2_stage1 = TSPFModelOT(
        input_dim=input_dim,
        repr_dims=[64, 32],
        stage=1,
        ot_kappa=0.5,
        isp_ratio=0.7
    )
    
    history2_stage1 = train_stage1(
        model=model2_stage1,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=150,
        lr=1e-3,
        weight_decay=1e-4,
        lambda_rec=0.1,
        lambda_ot=1.0,
        use_ot=True,
        device=device
    )
    
    print("\n--- Stage 2: Fine-tuning on RCT Data ---")
    model2_stage2 = TSPFModelOT(
        input_dim=input_dim,
        repr_dims=[64, 32],
        adapter_dim=16,
        stage=2,
        ot_kappa=0.5,
        isp_ratio=0.7
    )
    
    # Initialize from stage 1
    initialize_stage2_from_stage1(model2_stage1, model2_stage2)
    
    history2_stage2 = train_stage2(
        model=model2_stage2,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        lr=5e-4,
        weight_decay=1e-4,
        lambda_shift=0.1,
        lambda_mi=0.01,
        device=device
    )
    
    print("\n" + "="*80)
    print("EVALUATION ON TEST SET")
    print("="*80)
    
    # Evaluate both models
    print("\nEvaluating Model 1 (Standard TSPF)...")
    results1 = evaluate_model(model1_stage2, test_loader, device=device)
    
    print("\nEvaluating Model 2 (TSPF-OT)...")
    results2 = evaluate_model(model2_stage2, test_loader, device=device)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nModel 1 (Standard TSPF):")
    print(f"  Predicted ATE: {results1['ate_pred']:.4f}")
    print(f"  ITE std: {np.std(results1['ite_pred']):.4f}")
    print(f"  ITE range: [{np.min(results1['ite_pred']):.4f}, {np.max(results1['ite_pred']):.4f}]")
    
    print(f"\nModel 2 (TSPF-OT):")
    print(f"  Predicted ATE: {results2['ate_pred']:.4f}")
    print(f"  ITE std: {np.std(results2['ite_pred']):.4f}")
    print(f"  ITE range: [{np.min(results2['ite_pred']):.4f}, {np.max(results2['ite_pred']):.4f}]")
    
    # Plot results
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    history1 = {'stage1': history1_stage1, 'stage2': history1_stage2}
    history2 = {'stage1': history2_stage1, 'stage2': history2_stage2}
    
    plot_training_history(history1, history2, '/home/claude/training_history.png')
    plot_ite_comparison(results1, results2, '/home/claude/ite_comparison.png')
    
    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    torch.save(model1_stage2.state_dict(), '/home/claude/model1_tspf.pt')
    torch.save(model2_stage2.state_dict(), '/home/claude/model2_tspf_ot.pt')
    print("Models saved successfully!")
    
    # Save results
    np.savez('/home/claude/results.npz',
             model1_ite=results1['ite_pred'],
             model1_ate=results1['ate_pred'],
             model2_ite=results2['ite_pred'],
             model2_ate=results2['ate_pred'],
             treatments=results1['treatments'],
             outcomes=results1['outcomes'])
    print("Results saved successfully!")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - /home/claude/model1_tspf.pt (Standard TSPF model)")
    print("  - /home/claude/model2_tspf_ot.pt (TSPF-OT model)")
    print("  - /home/claude/training_history.png (Training curves)")
    print("  - /home/claude/ite_comparison.png (ITE predictions)")
    print("  - /home/claude/results.npz (Numerical results)")


if __name__ == "__main__":
    main()
