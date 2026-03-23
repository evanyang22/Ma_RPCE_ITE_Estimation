#import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
#import matplotlib.pyplot as plt
#import random
import copy
#import ot
#import sys
from RPCE_Model import AutoEncoder
from TrainingFunctionsStage1 import pseudo_outcome_loss


#Stage 2 training
def initialize_stage2_from_stage1(model):
    #initializes the unconfounded outcome heads g0 and g1 from the confounded outcome heads
    model.g0_head.load_state_dict(copy.deepcopy(model.t0_head.state_dict()))
    model.g1_head.load_state_dict(copy.deepcopy(model.t1_head.state_dict()))

def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True

def clone_params(module):
    return {name: p.detach().clone() for name, p in module.named_parameters()}

def parameter_shift_loss(module, init_params):
    #calculates the L2 distance of the models new parameters from the initial parameters
    loss = 0.0
    for name, p in module.named_parameters():
        loss = loss + torch.sum((p - init_params[name]) ** 2)
    return loss

def train_stage2_rct(
    model,
    dataset,
    batch_size=64,
    lr=1e-3,
    num_epochs=50,
    lambda_shift=1e-4,
    outcome_type="binary",
    freeze_encoder=True,
    verbose=True,
    device=None
):
    """
    Train Stage 2 unconfounded outcome heads on RCT data only.

    Expects dataset = TensorDataset(X, T, Y)

    Trains:
      - y0_rct head on factual samples with T=0
      - y1_rct head on factual samples with T=1

    Optionally applies shift regularization so Stage 2 heads
    do not drift too far from their warm-start initialization.

    Returns:
      model, history
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not hasattr(dataset, "tensors"):
        raise ValueError("dataset must be a TensorDataset")

    if len(dataset.tensors) < 3:
        raise ValueError("dataset must be TensorDataset(X, T, Y)")

    model = model.to(device)

    # ---------------------------------
    # Freeze / unfreeze modules
    # ---------------------------------
    if freeze_encoder:
        freeze_module(model.encoder)
    else:
        unfreeze_module(model.encoder)

    # Freeze Stage 1 pieces
    freeze_module(model.decoder)
    freeze_module(model.propensity_head)
    freeze_module(model.t0_head)
    freeze_module(model.t1_head)

    # Unfreeze Stage 2 heads
    unfreeze_module(model.g0_head)
    unfreeze_module(model.g1_head)

    # ---------------------------------
    # Save initial params for shift loss
    # ---------------------------------
    g0_init = clone_params(model.g0_head)
    g1_init = clone_params(model.g1_head)

    # ---------------------------------
    # Optimizer only over trainable params
    # ---------------------------------
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rct_loss = 0.0
        epoch_shift_loss = 0.0

        for X_batch, T_batch, Y_batch in loader:
            X_batch = X_batch.float().to(device)
            T_batch = T_batch.float().to(device).view(-1, 1)
            Y_batch = Y_batch.float().to(device).view(-1, 1)

            outputs = model(X_batch)

            # Stage 2 factual loss using RCT heads
            rct_loss = pseudo_outcome_loss(
                y0_hat=outputs["y0_rct"],
                y1_hat=outputs["y1_rct"],
                t_batch=T_batch,
                y_batch=Y_batch,
                outcome_type=outcome_type
            )

            # shift penalty
            shift_loss = (
                parameter_shift_loss(model.g0_head, g0_init) +
                parameter_shift_loss(model.g1_head, g1_init)
            )

            loss = rct_loss + lambda_shift * shift_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rct_loss += rct_loss.item()
            epoch_shift_loss += shift_loss.item()

        epoch_loss /= len(loader)
        epoch_rct_loss /= len(loader)
        epoch_shift_loss /= len(loader)

        history.append({
            "epoch": epoch + 1,
            "total_loss": epoch_loss,
            "rct_outcome_loss": epoch_rct_loss,
            "shift_loss": epoch_shift_loss
        })

        if verbose:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Total: {epoch_loss:.4f} | "
                f"RCT outcome: {epoch_rct_loss:.4f} | "
                f"Shift: {epoch_shift_loss:.4f}"
            )

    return model, history

