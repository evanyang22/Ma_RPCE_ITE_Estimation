import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
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
    encoder_lr_ratio=0.05,
    lambda_encoder_shift=1e-3,
    verbose=True,
    device=None
):
    """
    Train Stage 2 unconfounded outcome heads on RCT data only.

    Expects dataset = TensorDataset(X, T, Y)

    Trains:
      - g0_head (unconfounded Y|T=0) on factual control samples
      - g1_head (unconfounded Y|T=1) on factual treated samples
      - Optionally fine-tunes the encoder with a reduced learning rate

    Parameters
    ----------
    model : AutoEncoder
        Model with encoder, g0_head, g1_head already initialized.
    dataset : TensorDataset
        TensorDataset(X, T, Y) from the RCT subset.
    batch_size : int
    lr : float
        Base learning rate for g0/g1 heads.
    num_epochs : int
    lambda_shift : float
        L2 penalty pulling g0/g1 toward their Stage 1 initialization.
    outcome_type : str
        'continuous' or 'binary'.
    freeze_encoder : bool
        If True, fully freeze the encoder (original behavior).
        If False, fine-tune the encoder with a reduced learning rate.
    encoder_lr_ratio : float
        Fraction of `lr` used for encoder fine-tuning.  Only applies when
        freeze_encoder=False.  Default 0.05 means the encoder learns at
        5% of the head learning rate.

        WHY THIS VALUE:
        - The encoder was pre-trained on ~600 OBS samples in Stage 1.
        - Stage 2 only has ~149 RCT samples (20% of ~747).
        - A 20:1 LR ratio (0.05) lets the encoder nudge its representations
          toward what the RCT data needs without catastrophic forgetting
          of the rich structure learned from the larger OBS set.
        - Think of it as: heads make big jumps to fit RCT outcomes,
          encoder makes gentle adjustments to the feature space.

    lambda_encoder_shift : float
        L2 penalty pulling encoder toward its Stage 1 weights.  Only
        applies when freeze_encoder=False.  Acts as a safety net against
        the encoder drifting too far with limited RCT data.

        WHY THIS IS NEEDED:
        - Without it, even a small encoder LR can accumulate large weight
          changes over 50+ epochs on 149 samples.
        - The Stage 1 encoder learned useful structure (reconstruction,
          propensity, pseudo-outcomes) that we want to preserve.
        - This penalty says: "you can move, but the further you go from
          where Stage 1 left you, the more it costs."

    verbose : bool
    device : str or None

    Returns
    -------
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
    # Freeze Stage 1-only pieces always
    # ---------------------------------
    freeze_module(model.decoder)
    freeze_module(model.propensity_head)
    freeze_module(model.t0_head)
    freeze_module(model.t1_head)

    # Unfreeze Stage 2 heads
    unfreeze_module(model.g0_head)
    unfreeze_module(model.g1_head)

    # ---------------------------------
    # Encoder: freeze fully or fine-tune
    # ---------------------------------
    if freeze_encoder:
        freeze_module(model.encoder)
        encoder_init = None
    else:
        unfreeze_module(model.encoder)
        # Snapshot encoder weights before Stage 2 begins
        encoder_init = clone_params(model.encoder)

    # ---------------------------------
    # Save initial params for shift loss
    # ---------------------------------
    g0_init = clone_params(model.g0_head)
    g1_init = clone_params(model.g1_head)

    # ---------------------------------
    # Differential learning rate optimizer
    # ---------------------------------
    # Use parameter groups so the encoder gets a much smaller LR
    # than the g0/g1 heads.  This is the key mechanism:
    #   - Heads at full lr: they need to move substantially from
    #     confounded initialization toward unconfounded predictions
    #   - Encoder at lr * 0.05: gentle adaptation so the representation
    #     captures what matters for RCT outcomes without forgetting
    #     the covariate structure learned from the larger OBS set
    if freeze_encoder:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
    else:
        optimizer = torch.optim.Adam([
            {
                "params": model.encoder.parameters(),
                "lr": lr * encoder_lr_ratio,    # e.g. 1e-3 * 0.05 = 5e-5
            },
            {
                "params": list(model.g0_head.parameters()) +
                          list(model.g1_head.parameters()),
                "lr": lr,                        # full learning rate
            },
        ])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rct_loss = 0.0
        epoch_shift_loss = 0.0
        epoch_enc_shift_loss = 0.0

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

            # Head shift penalty (same as before)
            shift_loss = (
                parameter_shift_loss(model.g0_head, g0_init) +
                parameter_shift_loss(model.g1_head, g1_init)
            )

            loss = rct_loss + lambda_shift * shift_loss

            # Encoder shift penalty (only when fine-tuning)
            enc_shift_loss = torch.tensor(0.0, device=device)
            if not freeze_encoder and encoder_init is not None:
                enc_shift_loss = parameter_shift_loss(
                    model.encoder, encoder_init
                )
                loss = loss + lambda_encoder_shift * enc_shift_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rct_loss += rct_loss.item()
            epoch_shift_loss += shift_loss.item()
            epoch_enc_shift_loss += enc_shift_loss.item()

        epoch_loss /= len(loader)
        epoch_rct_loss /= len(loader)
        epoch_shift_loss /= len(loader)
        epoch_enc_shift_loss /= len(loader)

        history.append({
            "epoch": epoch + 1,
            "total_loss": epoch_loss,
            "rct_outcome_loss": epoch_rct_loss,
            "shift_loss": epoch_shift_loss,
            "encoder_shift_loss": epoch_enc_shift_loss,
        })

        if verbose:
            enc_msg = ""
            if not freeze_encoder:
                enc_msg = f" | Enc shift: {epoch_enc_shift_loss:.4f}"
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Total: {epoch_loss:.4f} | "
                f"RCT outcome: {epoch_rct_loss:.4f} | "
                f"Head shift: {epoch_shift_loss:.4f}"
                f"{enc_msg}"
            )

    return model, history
