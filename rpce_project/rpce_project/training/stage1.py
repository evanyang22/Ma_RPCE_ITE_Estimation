"""
Stage 1 training: AutoEncoder on observational data.

This stage trains:
1. Encoder/Decoder (reconstruction)
2. Propensity head (treatment prediction)
3. Pseudo-outcome heads (biased outcome prediction)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy

from ..models.losses import (
    reconstruction_loss,
    propensity_loss,
    pseudo_outcome_loss
)
from ..utils.data_utils import detect_binary_continuous_columns


def train_stage1(
    model,
    obs_dataset,
    hidden_dim=8,
    batch_size=64,
    lr=1e-3,
    num_epochs=50,
    recon_weight=1.0,
    prop_weight=1.0,
    pseudo_weight=1.0,
    verbose=True,
    device=None
):
    """
    Train Stage 1: AutoEncoder with pseudo-outcome heads on observational data.
    
    Args:
        model (nn.Module): AutoEncoder model
        obs_dataset (TensorDataset): Observational dataset (X, T, Y)
        hidden_dim (int): Hidden dimension (for loss weighting)
        batch_size (int): Training batch size
        lr (float): Learning rate
        num_epochs (int): Number of training epochs
        recon_weight (float): Reconstruction loss weight
        prop_weight (float): Propensity loss weight
        pseudo_weight (float): Pseudo-outcome loss weight
        verbose (bool): Print training progress
        device (str, optional): Device to use
    
    Returns:
        nn.Module: Trained model
        dict: Training history
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    dataloader = DataLoader(obs_dataset, batch_size=batch_size, shuffle=True)
    
    # Get first batch to detect binary/continuous columns
    X_sample = obs_dataset.tensors[0]
    binary_idx, continuous_idx = detect_binary_continuous_columns(X_sample)
    
    if verbose:
        print(f"\nStage 1 Training (Observational Data)")
        print(f"Binary features: {len(binary_idx)}")
        print(f"Continuous features: {len(continuous_idx)}")
        print(f"Device: {device}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        "recon_loss": [],
        "prop_loss": [],
        "pseudo_loss": [],
        "total_loss": []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {"recon": 0, "prop": 0, "pseudo": 0, "total": 0}
        
        for batch_idx, (x_batch, t_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            outputs = model(x_batch)
            
            # Reconstruction loss
            recon_loss_val = reconstruction_loss(
                outputs["x_recon"],
                x_batch,
                binary_idx=binary_idx,
                continuous_idx=continuous_idx
            )
            
            # Propensity loss
            prop_loss_val = propensity_loss(outputs["t_logit"], t_batch)
            
            # Pseudo-outcome loss
            pseudo_loss_val = pseudo_outcome_loss(
                outputs["y0_pseudo"],
                outputs["y1_pseudo"],
                t_batch,
                y_batch,
                outcome_type="continuous"
            )
            
            # Total loss
            total_loss = (
                recon_weight * recon_loss_val +
                prop_weight * prop_loss_val +
                pseudo_weight * pseudo_loss_val
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_losses["recon"] += recon_loss_val.item()
            epoch_losses["prop"] += prop_loss_val.item()
            epoch_losses["pseudo"] += pseudo_loss_val.item()
            epoch_losses["total"] += total_loss.item()
        
        # Average losses
        n_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        # Save history
        history["recon_loss"].append(epoch_losses["recon"])
        history["prop_loss"].append(epoch_losses["prop"])
        history["pseudo_loss"].append(epoch_losses["pseudo"])
        history["total_loss"].append(epoch_losses["total"])
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Recon: {epoch_losses['recon']:.4f} | "
                  f"Prop: {epoch_losses['prop']:.4f} | "
                  f"Pseudo: {epoch_losses['pseudo']:.4f} | "
                  f"Total: {epoch_losses['total']:.4f}")
    
    if verbose:
        print("Stage 1 training complete!")
    
    return model, history
