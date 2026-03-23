"""
Visualization utilities for training and evaluation.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_history(history1, history2=None, save_path=None):
    """
    Plot training losses over epochs.
    
    Args:
        history1 (dict): Stage 1 training history
        history2 (dict, optional): Stage 2 training history
        save_path (str, optional): Path to save figure
    """
    if history2 is None:
        # Single stage
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        # Plot each loss component
        losses = ['recon_loss', 'prop_loss', 'pseudo_loss', 'total_loss']
        titles = ['Reconstruction Loss', 'Propensity Loss', 
                 'Pseudo-Outcome Loss', 'Total Loss']
        
        for ax, loss_key, title in zip(axes, losses, titles):
            if loss_key in history1:
                ax.plot(history1[loss_key], label='Stage 1', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend()
    else:
        # Two stages
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Stage 1 losses
        stage1_losses = ['recon_loss', 'prop_loss', 'pseudo_loss', 'total_loss']
        stage1_titles = ['Reconstruction', 'Propensity', 'Pseudo-Outcome', 'Total']
        
        for i, (loss_key, title) in enumerate(zip(stage1_losses, stage1_titles)):
            if loss_key in history1:
                row, col = i // 3, i % 3
                axes[row, col].plot(history1[loss_key], linewidth=2, color='blue')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Loss')
                axes[row, col].set_title(f'Stage 1: {title}')
                axes[row, col].grid(True, alpha=0.3)
        
        # Stage 2 losses
        axes[1, 1].plot(history2['rct_loss'], linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Stage 2: RCT Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Transport distance if available
        if history2.get('transport_distance'):
            axes[1, 2].plot(history2['transport_distance'], linewidth=2, color='green')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Distance')
            axes[1, 2].set_title('Stage 2: Transport Distance')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def plot_cate_distribution(cate_pred, confidence=None, treatment=None, save_path=None):
    """
    Plot CATE prediction distribution.
    
    Args:
        cate_pred (torch.Tensor): CATE predictions
        confidence (torch.Tensor, optional): Confidence scores
        treatment (torch.Tensor, optional): Treatment indicator for stratification
        save_path (str, optional): Path to save figure
    """
    cate_pred = cate_pred.cpu().numpy() if torch.is_tensor(cate_pred) else cate_pred
    
    if confidence is not None:
        confidence = confidence.cpu().numpy() if torch.is_tensor(confidence) else confidence
    
    if treatment is not None:
        treatment = treatment.cpu().numpy() if torch.is_tensor(treatment) else treatment
    
    if confidence is not None and treatment is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    elif confidence is not None or treatment is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = axes.flatten()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
    
    # CATE distribution
    axes[0].hist(cate_pred, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(cate_pred.mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {cate_pred.mean():.3f}')
    axes[0].axvline(0, color='gray', linestyle='-', alpha=0.5)
    axes[0].set_xlabel('Predicted CATE')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('CATE Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if confidence is not None and len(axes) > 1:
        # Confidence distribution
        axes[1].hist(confidence, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1].axvline(confidence.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {confidence.mean():.3f}')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Confidence Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # CATE vs Confidence scatter
        if len(axes) > 2:
            axes[2].scatter(cate_pred, confidence, alpha=0.5, s=10)
            axes[2].set_xlabel('Predicted CATE')
            axes[2].set_ylabel('Confidence')
            axes[2].set_title('CATE vs Confidence')
            axes[2].grid(True, alpha=0.3)
    
    if treatment is not None and len(axes) > 3:
        # CATE by treatment group
        cate_treated = cate_pred[treatment == 1]
        cate_control = cate_pred[treatment == 0]
        
        axes[3].hist(cate_control, bins=30, alpha=0.6, label='Control', edgecolor='black')
        axes[3].hist(cate_treated, bins=30, alpha=0.6, label='Treated', edgecolor='black')
        axes[3].set_xlabel('Predicted CATE')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('CATE by Treatment Group')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CATE distribution plot saved to: {save_path}")
    
    plt.show()


def plot_policy_analysis(cate_pred, treatment, outcome, thresholds=None, save_path=None):
    """
    Analyze policy value across different treatment thresholds.
    
    Args:
        cate_pred (torch.Tensor): CATE predictions
        treatment (torch.Tensor): Observed treatment
        outcome (torch.Tensor): Observed outcome
        thresholds (list, optional): List of thresholds to evaluate
        save_path (str, optional): Path to save figure
    """
    from evaluation import estimate_policy_value_from_rct
    
    cate_pred = cate_pred.cpu() if torch.is_tensor(cate_pred) else cate_pred
    treatment = treatment.cpu() if torch.is_tensor(treatment) else treatment
    outcome = outcome.cpu() if torch.is_tensor(outcome) else outcome
    
    if thresholds is None:
        thresholds = np.linspace(cate_pred.min(), cate_pred.max(), 50)
    
    policy_values = []
    policy_risks = []
    treatment_rates = []
    
    for thresh in thresholds:
        metrics = estimate_policy_value_from_rct(cate_pred, treatment, outcome, threshold=thresh)
        policy_values.append(metrics['policy_value'])
        policy_risks.append(metrics['policy_risk'])
        treatment_rates.append(metrics['p_policy_treat'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Policy value
    axes[0].plot(thresholds, policy_values, linewidth=2)
    best_idx = np.argmax(policy_values)
    axes[0].scatter([thresholds[best_idx]], [policy_values[best_idx]], 
                   color='red', s=100, zorder=5, label='Optimal')
    axes[0].set_xlabel('Treatment Threshold')
    axes[0].set_ylabel('Policy Value')
    axes[0].set_title('Policy Value vs Threshold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Policy risk
    axes[1].plot(thresholds, policy_risks, linewidth=2, color='orange')
    axes[1].scatter([thresholds[best_idx]], [policy_risks[best_idx]], 
                   color='red', s=100, zorder=5)
    axes[1].set_xlabel('Treatment Threshold')
    axes[1].set_ylabel('Policy Risk')
    axes[1].set_title('Policy Risk vs Threshold')
    axes[1].grid(True, alpha=0.3)
    
    # Treatment rate
    axes[2].plot(thresholds, treatment_rates, linewidth=2, color='green')
    axes[2].set_xlabel('Treatment Threshold')
    axes[2].set_ylabel('Treatment Rate')
    axes[2].set_title('Treatment Rate vs Threshold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy analysis plot saved to: {save_path}")
    
    plt.show()
    
    print(f"\nOptimal threshold: {thresholds[best_idx]:.4f}")
    print(f"Optimal policy value: {policy_values[best_idx]:.4f}")
    print(f"Optimal policy risk: {policy_risks[best_idx]:.4f}")
    print(f"Treatment rate at optimal: {treatment_rates[best_idx]:.2%}")


def plot_latent_space(model, dataset, labels=None, save_path=None):
    """
    Visualize latent space using 2D projection.
    
    Args:
        model (nn.Module): Trained model
        dataset (TensorDataset): Dataset to encode
        labels (np.array, optional): Labels for coloring (e.g., treatment)
        save_path (str, optional): Path to save figure
    """
    from sklearn.decomposition import PCA
    
    model.eval()
    X = dataset.tensors[0]
    
    with torch.no_grad():
        z = model.encode(X).cpu().numpy()
    
    # Use PCA if latent dim > 2
    if z.shape[1] > 2:
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z)
        title = f'Latent Space (PCA, {pca.explained_variance_ratio_.sum():.1%} variance)'
    else:
        z_2d = z
        title = 'Latent Space'
    
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, 
                            cmap='RdYlBu', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Treatment')
    else:
        plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6, s=20)
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent space plot saved to: {save_path}")
    
    plt.show()
