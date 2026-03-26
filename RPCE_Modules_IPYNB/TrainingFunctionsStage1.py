import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
import copy
import ot
import sys
from RPCE_Model import AutoEncoder

def pseudo_outcome_loss(y0_hat, y1_hat, t_batch, y_batch, weights=None, outcome_type="continuous"):
    """
    Calculates the outcome loss for both binary and continuous Y variables
    Inputs
    1. y0_hat= predicted outcome for t=0
    2. y1_hat= predicted outcome for t=1 
    3. t_batch: whether t=0 or t=1
    4. y_batch: observed outcome 
    Trains only the factual branch:
      - if t=0, compare y0_hat to y
      - if t=1, compare y1_hat to y
    """

    t_batch = t_batch.float()
    y_batch = y_batch.float()


    if outcome_type == "continuous":
        loss_fn = nn.MSELoss(reduction="none")
    elif outcome_type == "binary":
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    else:
        raise ValueError("outcome_type must be 'continuous' or 'binary'")

    loss0 = loss_fn(y0_hat, y_batch)   # [B, 1]
    loss1 = loss_fn(y1_hat, y_batch)   # [B, 1]

    masked_loss = (1 - t_batch) * loss0 + t_batch * loss1
    #print("weights min:", weights.min().item())
    #print("weights max:", weights.max().item())
    #print("masked_loss min:", masked_loss.min().item())
    #print("masked_loss max:", masked_loss.max().item())
    if weights is not None:  
        masked_loss = weights * masked_loss  
    
    return masked_loss.mean()

#Training Functions
def detect_binary_continuous_columns(X_tensor, tol=1e-6):
    """
    Detect binary vs continuous columns automatically.

    A column is considered binary if its unique values are only {0,1}
    within numerical tolerance

    Returns index of continuous and binary columns
    """
    binary_idx = []
    continuous_idx = []

    n_features = X_tensor.shape[1]

    for i in range(n_features):
        col = X_tensor[:, i]
        unique_vals = torch.unique(col)

        # Round tiny numerical noise
        unique_vals = torch.round(unique_vals / tol) * tol

        # Check whether all unique values are 0 or 1
        is_binary = torch.all((unique_vals == 0) | (unique_vals == 1)).item()

        if is_binary:
            binary_idx.append(i)
        else:
            continuous_idx.append(i)

    return binary_idx, continuous_idx

def train_mixed_autoencoder(
    dataset,
    hidden_dim=8,
    latent_dim=4,
    batch_size=64,
    lr=1e-3,
    num_epochs=50,
    verbose=True,
    device=None,
    outcome_type="binary",
    alpha_recon= 0.1,
    alpha_outcome= 1,
    alpha_prop=0.5
):
    """
    Train an autoencoder on a TensorDataset(X, t, y) or TensorDataset(X).

    Automatically detects binary and continuous columns in X and uses:
      - BCEWithLogitsLoss for binary columns
      - MSELoss for continuous columns

    Returns:
      model, history, binary_idx, continuous_idx
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract X_tensor from TensorDataset
    if not hasattr(dataset, "tensors"):
        raise ValueError("dataset must be a TensorDataset")

    X_tensor = dataset.tensors[0].float()

    # Detect column types
    binary_idx, continuous_idx = detect_binary_continuous_columns(X_tensor)

    if verbose:
        print("Detected binary columns:", binary_idx)
        print("Detected continuous columns:", continuous_idx)

    input_dim = X_tensor.shape[1]
    model = AutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    prop= nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_cont_loss = 0.0
        epoch_bin_loss = 0.0
        epoch_prop_loss =0.0
        epoch_pseudo_loss =0.0

        for X_batch,T_batch,Y_batch in loader:
            X_batch = X_batch.float().to(device)
            T_batch = T_batch.float().to(device)
            Y_batch = Y_batch.float().to(device)
            outputs = model(X_batch)
            #print("X_batch:", X_batch.shape)
            #print("Y_batch:", Y_batch.shape)
            #print("y0_hat:", outputs["y0_pseudo"].shape)
            #print("y1_hat:", outputs["y1_pseudo"].shape)
            #print("binary_idx:", binary_idx)
            #print("continuous_idx:", continuous_idx)
            
            
            loss = 0.0
            cont_loss = torch.tensor(0.0, device=device)
            bin_loss = torch.tensor(0.0, device=device)

            if len(continuous_idx) > 0:
                cont_pred = outputs["x_recon"][:, continuous_idx]
                cont_true = X_batch[:, continuous_idx]
                cont_loss = mse(cont_pred, cont_true)
                loss = loss + alpha_recon * cont_loss

            if len(binary_idx) > 0:
                bin_pred = outputs["x_recon"][:, binary_idx]   # logits
                bin_true = X_batch[:, binary_idx]
                bin_loss = bce(bin_pred, bin_true)
                loss = loss + alpha_recon * bin_loss

            #Propensity training
            T_batch = T_batch.unsqueeze(1).float()
            propensity_loss= prop(outputs["t_logit"],T_batch)
            loss+=alpha_prop * propensity_loss

            '''
            u_hat = T_batch.mean()
            u_hat = torch.clamp(u_hat, min=1e-6, max=1-1e-6)
            w_i = T_batch / (2 * u_hat) + (1 - T_batch) / (2 * (1 - u_hat))
            '''

            e_hat = torch.sigmoid(outputs["t_logit"]).detach()   # per-individual e(x_i)
            e_hat = torch.clamp(e_hat, min=0.05, max=0.95)       # stabilize extremes
            w_i = T_batch / (2*e_hat) + (1-T_batch) / (2*(1-e_hat))
            w_i = w_i / w_i.mean().clamp_min(1e-8)               # normalize within batch
            #pseudooutcome training
            Y_batch = Y_batch.float().view(-1, 1)
            pseudo_loss = pseudo_outcome_loss(
                y0_hat = outputs["y0_pseudo"],
                y1_hat = outputs["y1_pseudo"],
                t_batch=T_batch,
                y_batch=Y_batch,
                weights=w_i,
                outcome_type=outcome_type
            )
            loss +=alpha_outcome * pseudo_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cont_loss += cont_loss.item()
            epoch_bin_loss += bin_loss.item()
            epoch_prop_loss += propensity_loss.item()
            epoch_pseudo_loss += pseudo_loss.item()

        epoch_loss /= len(loader)
        epoch_cont_loss /= len(loader)
        epoch_bin_loss /= len(loader)
        epoch_prop_loss /= len(loader)
        epoch_pseudo_loss /= len(loader)

        history.append({
            "epoch": epoch + 1,
            "total_loss": epoch_loss,
            "continuous_loss": epoch_cont_loss,
            "binary_loss": epoch_bin_loss,
            "propensity_loss":epoch_prop_loss,
            "pseudo_outcome_loss":epoch_pseudo_loss
        })

        if verbose:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Total: {epoch_loss:.4f} | "
                f"Cont: {epoch_cont_loss:.4f} | "
                f"Bin: {epoch_bin_loss:.4f} | "
                f"Propensity: {epoch_prop_loss:.4f}|"
                f"Pseudo-outcome: {epoch_pseudo_loss:.4f}"
            )

    return model, history, binary_idx, continuous_idx