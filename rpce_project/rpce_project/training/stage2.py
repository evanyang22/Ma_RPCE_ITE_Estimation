"""
Stage 2 training: Fine-tune on RCT data.

This stage:
1. Freezes encoder/decoder (keeps learned representation)
2. Initializes RCT outcome heads from pseudo-outcome heads
3. Trains RCT heads on unconfounded RCT data
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy

from ..models.losses import rct_outcome_loss
from ..transport.sinkhorn import sinkhorn_projection_balanced


def initialize_stage2_from_stage1(model):
    """
    Initialize Stage 2 RCT heads from Stage 1 pseudo-outcome heads.
    
    Args:
        model (nn.Module): Model after Stage 1 training
    
    Returns:
        nn.Module: Model with initialized Stage 2 heads
    """
    model_stage2 = copy.deepcopy(model)
    
    # Copy weights from pseudo-outcome heads to RCT heads
    model_stage2.g0_head.load_state_dict(model.t0_head.state_dict())
    model_stage2.g1_head.load_state_dict(model.t1_head.state_dict())
    
    return model_stage2


def freeze_module(module):
    """Freeze module parameters."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    """Unfreeze module parameters."""
    for param in module.parameters():
        param.requires_grad = True


def train_stage2(
    model,
    rct_dataset,
    obs_dataset=None,
    use_transport=True,
    batch_size=64,
    lr=1e-3,
    num_epochs=50,
    freeze_encoder=True,
    verbose=True,
    device=None
):
    """
    Train Stage 2: RCT outcome heads on randomized data.
    
    Args:
        model (nn.Module): Model after Stage 1 (or initialized for Stage 2)
        rct_dataset (TensorDataset): RCT dataset (X, T, Y)
        obs_dataset (TensorDataset, optional): Observational dataset for transport
        use_transport (bool): Whether to use optimal transport
        batch_size (int): Training batch size
        lr (float): Learning rate
        num_epochs (int): Number of training epochs
        freeze_encoder (bool): Whether to freeze encoder/decoder
        verbose (bool): Print training progress
        device (str, optional): Device to use
    
    Returns:
        nn.Module: Trained model
        dict: Training history
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    
    # Freeze encoder/decoder if specified
    if freeze_encoder:
        freeze_module(model.encoder)
        freeze_module(model.decoder)
        if verbose:
            print("Encoder and Decoder frozen")
    
    # Also freeze pseudo-outcome heads and propensity head (not needed in Stage 2)
    freeze_module(model.t0_head)
    freeze_module(model.t1_head)
    freeze_module(model.propensity_head)
    
    # Ensure RCT heads are unfrozen
    unfreeze_module(model.g0_head)
    unfreeze_module(model.g1_head)
    
    dataloader = DataLoader(rct_dataset, batch_size=batch_size, shuffle=True)
    
    if verbose:
        print(f"\nStage 2 Training (RCT Data)")
        print(f"Use transport: {use_transport}")
        print(f"Device: {device}")
    
    # Optimizer (only for unfrozen parameters)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    
    # Training history
    history = {
        "rct_loss": [],
        "transport_distance": [] if use_transport else None
    }
    
    # Get observational latent representations if using transport
    if use_transport and obs_dataset is not None:
        X_obs = obs_dataset.tensors[0].to(device)
        with torch.no_grad():
            z_obs = model.encode(X_obs)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_transport_dist = 0
        
        for batch_idx, (x_batch, t_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Encode
            z_rct = model.encode(x_batch)
            
            # Apply optimal transport if specified
            if use_transport and obs_dataset is not None:
                # Transport observational to RCT domain
                z_transported = sinkhorn_projection_balanced(z_obs, z_rct)
                
                # Compute transport distance for monitoring
                transport_dist = torch.cdist(z_transported, z_rct, p=2).min(dim=1).values.mean()
                epoch_transport_dist += transport_dist.item()
            
            # Predict outcomes using RCT heads
            y0_pred = model.g0_head(z_rct)
            y1_pred = model.g1_head(z_rct)
            
            # RCT outcome loss
            loss = rct_outcome_loss(
                y0_pred,
                y1_pred,
                t_batch,
                y_batch,
                outcome_type="continuous"
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average losses
        n_batches = len(dataloader)
        epoch_loss /= n_batches
        
        if use_transport and obs_dataset is not None:
            epoch_transport_dist /= n_batches
            history["transport_distance"].append(epoch_transport_dist)
        
        history["rct_loss"].append(epoch_loss)
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch+1}/{num_epochs} | RCT Loss: {epoch_loss:.4f}"
            if use_transport and obs_dataset is not None:
                msg += f" | Transport Dist: {epoch_transport_dist:.4f}"
            print(msg)
    
    if verbose:
        print("Stage 2 training complete!")
    
    return model, history
