"""
Example usage of RPCE model with visualization.

This script demonstrates:
1. Loading data
2. Training the model (2 stages)
3. Evaluation
4. Visualization
"""
import torch

from config import set_random_seed, get_device
from data import load_jobs_data, dataset_summary
from models import AutoEncoder
from training import train_stage1, train_stage2, initialize_stage2_from_stage1
from evaluation import evaluate_jobs_policy_risk_and_att
from utils.visualization import (
    plot_training_history,
    plot_cate_distribution,
    plot_policy_analysis,
    plot_latent_space
)


def main():
    """Run complete RPCE pipeline with visualizations."""
    
    # ==================== SETUP ====================
    print("="*80)
    print("RPCE Example: Complete Pipeline")
    print("="*80)
    
    set_random_seed(42)
    device = get_device()
    
    # ==================== DATA ====================
    print("\n[1/6] Loading data...")
    
    # Update these paths to your data files
    TRAIN_PATH = "path/to/jobs_train.npz"
    TEST_PATH = "path/to/jobs_test.npz"
    
    data = load_jobs_data(TRAIN_PATH, TEST_PATH)
    
    # Show data summaries
    for name, dataset in data.items():
        dataset_summary(dataset, name.replace('_', ' ').title())
    
    # ==================== MODEL ====================
    print("\n[2/6] Creating model...")
    
    # Model hyperparameters
    INPUT_DIM = data['train_obs'].tensors[0].shape[1]  # Auto-detect from data
    HIDDEN_DIM = 8
    LATENT_DIM = 4
    
    model = AutoEncoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # ==================== STAGE 1 ====================
    print("\n[3/6] Stage 1: Training on observational data...")
    
    model, history1 = train_stage1(
        model=model,
        obs_dataset=data['train_obs'],
        hidden_dim=HIDDEN_DIM,
        batch_size=64,
        lr=1e-3,
        num_epochs=50,
        recon_weight=1.0,
        prop_weight=1.0,
        pseudo_weight=1.0,
        verbose=True,
        device=device
    )
    
    print("Stage 1 complete!")
    
    # ==================== STAGE 2 ====================
    print("\n[4/6] Stage 2: Fine-tuning on RCT data...")
    
    model = initialize_stage2_from_stage1(model)
    
    model, history2 = train_stage2(
        model=model,
        rct_dataset=data['train_rct'],
        obs_dataset=data['train_obs'],
        use_transport=True,
        batch_size=64,
        lr=1e-3,
        num_epochs=50,
        freeze_encoder=True,
        verbose=True,
        device=device
    )
    
    print("Stage 2 complete!")
    
    # ==================== EVALUATION ====================
    print("\n[5/6] Evaluating model...")
    
    metrics = evaluate_jobs_policy_risk_and_att(
        model=model,
        eval_rct_dataset=data['test_rct'],
        rct_reference_dataset=data['test_rct'],
        batch_size=256,
        threshold=0.0,
        device=device,
        verbose=True
    )
    
    # ==================== VISUALIZATION ====================
    print("\n[6/6] Generating visualizations...")
    
    # Training history
    print("  - Training history plot...")
    plot_training_history(history1, history2, save_path="outputs/training_history.png")
    
    # CATE distribution
    print("  - CATE distribution plot...")
    plot_cate_distribution(
        cate_pred=metrics['cate_pred'],
        confidence=metrics['confidence'],
        treatment=data['test_rct'].tensors[1],
        save_path="outputs/cate_distribution.png"
    )
    
    # Policy analysis
    print("  - Policy analysis plot...")
    plot_policy_analysis(
        cate_pred=metrics['cate_pred'],
        treatment=data['test_rct'].tensors[1],
        outcome=data['test_rct'].tensors[2],
        save_path="outputs/policy_analysis.png"
    )
    
    # Latent space visualization
    print("  - Latent space plot...")
    plot_latent_space(
        model=model,
        dataset=data['test_rct'],
        labels=data['test_rct'].tensors[1],
        save_path="outputs/latent_space.png"
    )
    
    # ==================== SAVE MODEL ====================
    print("\nSaving model...")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': INPUT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'latent_dim': LATENT_DIM,
        'metrics': {k: v for k, v in metrics.items() 
                   if not isinstance(v, torch.Tensor)},
        'stage1_history': history1,
        'stage2_history': history2
    }, 'outputs/rpce_model.pt')
    
    print("Model saved to: outputs/rpce_model.pt")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"Policy Value:      {metrics['policy_value']:.6f}")
    print(f"Policy Risk:       {metrics['policy_risk']:.6f}")
    print(f"ATT (Predicted):   {metrics['att_hat']:.6f}")
    print(f"ATT (Empirical):   {metrics['att_empirical_rct']:.6f}")
    print(f"ATT Error:         {metrics['att_error']:.6f}")
    print(f"ATE (Predicted):   {metrics['predicted_ATE']:.6f}")
    print(f"Mean Confidence:   {metrics['mean_confidence']:.6f}")
    print("="*80)
    
    print("\n✅ Pipeline complete!")
    print("Check the 'outputs/' folder for saved visualizations and model.")
    
    return model, metrics


if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Run pipeline
    model, metrics = main()
