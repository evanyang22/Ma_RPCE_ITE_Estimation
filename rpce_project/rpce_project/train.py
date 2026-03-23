"""
Main training script for RPCE model.

This script demonstrates the complete two-stage training pipeline:
1. Stage 1: Train on observational data
2. Stage 2: Fine-tune on RCT data
3. Evaluate on test set
"""
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from config import set_random_seed, get_device
from data import load_jobs_data, dataset_summary
from models import AutoEncoder
from training import train_stage1, train_stage2, initialize_stage2_from_stage1
from evaluation import evaluate_jobs_policy_risk_and_att


def plot_training_history(history1, history2, save_path=None):
    """Plot training curves from both stages."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Stage 1
    ax = axes[0]
    ax.plot(history1['recon_loss'], label='Reconstruction')
    ax.plot(history1['prop_loss'], label='Propensity')
    ax.plot(history1['pseudo_loss'], label='Pseudo-outcome')
    ax.plot(history1['total_loss'], label='Total', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Stage 1: Observational Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stage 2
    ax = axes[1]
    ax.plot(history2['rct_loss'], label='RCT Loss', linewidth=2)
    if history2.get('transport_distance') is not None:
        ax2 = ax.twinx()
        ax2.plot(history2['transport_distance'], 'r--', label='Transport Distance', alpha=0.7)
        ax2.set_ylabel('Transport Distance', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Stage 2: RCT Fine-tuning')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def main():
    """Main training pipeline."""
    
    # ==================== Configuration ====================
    print("="*80)
    print("RPCE Model Training")
    print("="*80)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    device = get_device()
    
    # Data paths (UPDATE THESE TO YOUR PATHS)
    TRAIN_PATH = "C:/Users/evany/Desktop/MaLabRotation/Data/FredJoData/jobs_DW_bin.new.10.train.npz"
    TEST_PATH = "C:/Users/evany/Desktop/MaLabRotation/Data/FredJoData/jobs_DW_bin.new.10.test.npz"
    
    # Model hyperparameters
    INPUT_DIM = 17  # Jobs dataset has 17 features
    HIDDEN_DIM = 8
    LATENT_DIM = 4
    
    # Training hyperparameters
    STAGE1_EPOCHS = 50
    STAGE1_LR = 1e-3
    STAGE1_BATCH_SIZE = 64
    
    STAGE2_EPOCHS = 50
    STAGE2_LR = 1e-3
    STAGE2_BATCH_SIZE = 64
    
    # ==================== Load Data ====================
    print("\nLoading data...")
    data = load_jobs_data(TRAIN_PATH, TEST_PATH)
    
    print("\nDataset Summary:")
    dataset_summary(data['train_obs'], "Training Observational")
    dataset_summary(data['train_rct'], "Training RCT")
    dataset_summary(data['test_rct'], "Test RCT")
    
    # ==================== Create Model ====================
    print("\nCreating AutoEncoder model...")
    model = AutoEncoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ==================== Stage 1 Training ====================
    print("\n" + "="*80)
    print("STAGE 1: Training on Observational Data")
    print("="*80)
    
    model, history1 = train_stage1(
        model=model,
        obs_dataset=data['train_obs'],
        hidden_dim=HIDDEN_DIM,
        batch_size=STAGE1_BATCH_SIZE,
        lr=STAGE1_LR,
        num_epochs=STAGE1_EPOCHS,
        recon_weight=1.0,
        prop_weight=1.0,
        pseudo_weight=1.0,
        verbose=True,
        device=device
    )
    
    # ==================== Stage 2 Setup ====================
    print("\n" + "="*80)
    print("STAGE 2 SETUP: Initializing RCT Heads")
    print("="*80)
    
    model = initialize_stage2_from_stage1(model)
    print("RCT outcome heads initialized from pseudo-outcome heads")
    
    # ==================== Stage 2 Training ====================
    print("\n" + "="*80)
    print("STAGE 2: Fine-tuning on RCT Data")
    print("="*80)
    
    model, history2 = train_stage2(
        model=model,
        rct_dataset=data['train_rct'],
        obs_dataset=data['train_obs'],
        use_transport=True,
        batch_size=STAGE2_BATCH_SIZE,
        lr=STAGE2_LR,
        num_epochs=STAGE2_EPOCHS,
        freeze_encoder=True,
        verbose=True,
        device=device
    )
    
    # ==================== Evaluation ====================
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    metrics = evaluate_jobs_policy_risk_and_att(
        model=model,
        eval_rct_dataset=data['test_rct'],
        rct_reference_dataset=data['test_rct'],
        batch_size=128,
        threshold=0.0,
        device=device,
        verbose=True
    )
    
    # ==================== Save Results ====================
    print("\nSaving results...")
    
    # Save model
    model_path = Path("checkpoints")
    model_path.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': INPUT_DIM,
            'hidden_dim': HIDDEN_DIM,
            'latent_dim': LATENT_DIM
        },
        'metrics': {k: v for k, v in metrics.items() if not isinstance(v, torch.Tensor)}
    }, model_path / "rpce_model.pt")
    print(f"Model saved to {model_path / 'rpce_model.pt'}")
    
    # Plot training curves
    figures_path = Path("figures")
    figures_path.mkdir(exist_ok=True)
    plot_training_history(history1, history2, save_path=figures_path / "training_curves.png")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
