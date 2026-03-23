"""
Loss functions for RPCE training.
"""
import torch
import torch.nn as nn


def pseudo_outcome_loss(y0_hat, y1_hat, t_batch, y_batch, weights=None, outcome_type="continuous"):
    """
    Calculate pseudo-outcome loss for observational data.
    
    This loss trains only the factual branch:
    - If t=0, compare y0_hat to y
    - If t=1, compare y1_hat to y
    
    Args:
        y0_hat (torch.Tensor): Predicted outcome for t=0 [batch_size, 1]
        y1_hat (torch.Tensor): Predicted outcome for t=1 [batch_size, 1]
        t_batch (torch.Tensor): Treatment indicator [batch_size]
        y_batch (torch.Tensor): Observed outcome [batch_size]
        weights (torch.Tensor, optional): Sample weights [batch_size, 1]
        outcome_type (str): "continuous" or "binary"
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    t_batch = t_batch.float()
    y_batch = y_batch.float()
    
    # Select appropriate loss function
    if outcome_type == "continuous":
        loss_fn = nn.MSELoss(reduction="none")
    elif outcome_type == "binary":
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    else:
        raise ValueError("outcome_type must be 'continuous' or 'binary'")
    
    # Compute loss for both branches
    loss0 = loss_fn(y0_hat, y_batch)  # [batch_size, 1]
    loss1 = loss_fn(y1_hat, y_batch)  # [batch_size, 1]
    
    # Mask to only use factual outcomes
    masked_loss = (1 - t_batch) * loss0 + t_batch * loss1
    
    # Apply sample weights if provided
    if weights is not None:
        masked_loss = weights * masked_loss
    
    return masked_loss.mean()


def reconstruction_loss(x_recon, x_true, binary_idx=None, continuous_idx=None):
    """
    Calculate reconstruction loss with mixed data types.
    
    Args:
        x_recon (torch.Tensor): Reconstructed features [batch_size, input_dim]
        x_true (torch.Tensor): True features [batch_size, input_dim]
        binary_idx (list, optional): Indices of binary features
        continuous_idx (list, optional): Indices of continuous features
    
    Returns:
        torch.Tensor: Scalar reconstruction loss
    """
    if binary_idx is None and continuous_idx is None:
        # Treat all as continuous
        return nn.MSELoss()(x_recon, x_true)
    
    total_loss = 0.0
    
    # Binary features: use BCE loss
    if binary_idx is not None and len(binary_idx) > 0:
        x_recon_bin = x_recon[:, binary_idx]
        x_true_bin = x_true[:, binary_idx]
        binary_loss = nn.BCEWithLogitsLoss()(x_recon_bin, x_true_bin)
        total_loss += binary_loss
    
    # Continuous features: use MSE loss
    if continuous_idx is not None and len(continuous_idx) > 0:
        x_recon_cont = x_recon[:, continuous_idx]
        x_true_cont = x_true[:, continuous_idx]
        continuous_loss = nn.MSELoss()(x_recon_cont, x_true_cont)
        total_loss += continuous_loss
    
    return total_loss


def propensity_loss(t_logit, t_true):
    """
    Calculate propensity (treatment prediction) loss.
    
    Args:
        t_logit (torch.Tensor): Predicted treatment logits [batch_size, 1]
        t_true (torch.Tensor): True treatment [batch_size]
    
    Returns:
        torch.Tensor: Scalar propensity loss
    """
    t_true = t_true.float().unsqueeze(-1) if t_true.dim() == 1 else t_true.float()
    return nn.BCEWithLogitsLoss()(t_logit, t_true)


def rct_outcome_loss(y0_hat, y1_hat, t_batch, y_batch, outcome_type="continuous"):
    """
    Calculate RCT outcome loss (unconfounded).
    
    Similar to pseudo_outcome_loss but for RCT data where there's no confounding.
    
    Args:
        y0_hat (torch.Tensor): Predicted outcome for t=0 [batch_size, 1]
        y1_hat (torch.Tensor): Predicted outcome for t=1 [batch_size, 1]
        t_batch (torch.Tensor): Treatment indicator [batch_size]
        y_batch (torch.Tensor): Observed outcome [batch_size]
        outcome_type (str): "continuous" or "binary"
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    return pseudo_outcome_loss(y0_hat, y1_hat, t_batch, y_batch, 
                              weights=None, outcome_type=outcome_type)
