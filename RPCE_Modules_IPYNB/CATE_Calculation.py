
import torch
from OptimalTransportFunctions import sinkhorn_projection, sinkhorn_projection_balanced, sinkhorn_projection_balanced_better
from Confidence import compute_confidence
from TrainingFunctionsStage1 import detect_binary_continuous_columns
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
import torch.nn as nn
from typing import Union
import numpy as np
import pandas as pd

@torch.no_grad()
def predict_cate_rpce(model, x_obs, x_rct, outcome_type='continuous', device='cpu'):
    """
    Predicts CATE following the outline's approach.
    
    Args:
        outcome_type: 'continuous' or 'binary'
    """
    model.eval()
    model = model.to(device)

    x_obs = x_obs.to(device).float()
    x_rct = x_rct.to(device).float()

    # Get representations
    z_obs = model.encoder(x_obs)
    z_rct = model.encoder(x_rct)

    # OT projection
    z_tilde, pi_star = sinkhorn_projection_balanced_better(z_obs, z_rct)
    z_tilde = z_tilde.to(device)
    pi_star = pi_star.to(device)

    # Confidence
    c, _, _ = compute_confidence(
        z_obs=z_obs,
        z_rct=z_rct,
        pi_star=pi_star,
        propensity_head=model.propensity_head
    )

    # Get raw predictions from heads
    y1_logit_projected = model.g1_head(z_tilde)
    y0_logit_projected = model.g0_head(z_tilde)
    y1_logit_fallback = model.t1_head(z_obs)
    y0_logit_fallback = model.t0_head(z_obs)
    
    # Apply appropriate transform based on outcome type
    if outcome_type == 'binary':
        # For binary: apply sigmoid then compute differences
        y1_projected = torch.sigmoid(y1_logit_projected)
        y0_projected = torch.sigmoid(y0_logit_projected)
        y1_fallback = torch.sigmoid(y1_logit_fallback)
        y0_fallback = torch.sigmoid(y0_logit_fallback)
    else:  # continuous
        # For continuous: use raw predictions
        y1_projected = y1_logit_projected
        y0_projected = y0_logit_projected
        y1_fallback = y1_logit_fallback
        y0_fallback = y0_logit_fallback
    
    # Compute CATE for each branch (same formula for both cases)
    cate_projected = y1_projected - y0_projected
    cate_fallback = y1_fallback - y0_fallback
    
    # Weighted combination (same formula for both cases)
    cate_final = c * cate_projected + (1.0 - c) * cate_fallback
    
    return (cate_final.view(-1), 
            c.view(-1), 
            cate_projected.view(-1), 
            cate_fallback.view(-1))


'''
def predict_cate_rpce(model, x_obs, x_rct, device='cpu'):
    #Predicts CATE for each sample via a combination of unconfounded and confounded outcomes
    model.eval()
    model = model.to(device)

    x_obs = x_obs.to(device).float()
    x_rct = x_rct.to(device).float()

    # Get representations
    z_obs = model.encoder(x_obs)
    z_rct = model.encoder(x_rct)

    # OT projection
    z_tilde, pi_star = sinkhorn_projection(z_obs, z_rct)

    # move OT outputs to same device if needed
    z_tilde = z_tilde.to(device)
    pi_star = pi_star.to(device)

    # Confidence
    c, _, _ = compute_confidence(
        z_obs=z_obs,
        z_rct=z_rct,
        pi_star=pi_star,
        propensity_head=model.propensity_head
    )

    """
    # CATE estimates - FIXED VERSION
    y1_prob_projected = torch.sigmoid(model.g1_head(z_tilde))
    y0_prob_projected = torch.sigmoid(model.g0_head(z_tilde))
    cate_projected = y1_prob_projected - y0_prob_projected

    y1_prob_fallback = torch.sigmoid(model.t1_head(z_obs))
    y0_prob_fallback = torch.sigmoid(model.t0_head(z_obs))
    cate_fallback = y1_prob_fallback - y0_prob_fallback

    # Weighted combination
    cate_final = c * cate_projected + (1.0 - c) * cate_fallback        # [B,1]
    """
    # CATE estimates - CORRECT VERSION
    # Step 1: Combine outcome predictions in LOGIT space (before sigmoid)
    y1_logit_projected = model.g1_head(z_tilde)  # [B, 1]
    y0_logit_projected = model.g0_head(z_tilde)  # [B, 1]

    y1_logit_fallback = model.t1_head(z_obs)     # [B, 1]
    y0_logit_fallback = model.t0_head(z_obs)     # [B, 1]

    # Step 2: Weighted combination in LOGIT space
    y1_logit_final = c * y1_logit_projected + (1.0 - c) * y1_logit_fallback  # [B, 1]
    y0_logit_final = c * y0_logit_projected + (1.0 - c) * y0_logit_fallback  # [B, 1]

    # Step 3: Convert to probabilities
    y1_prob_final = torch.sigmoid(y1_logit_final)  # [B, 1]
    y0_prob_final = torch.sigmoid(y0_logit_final)  # [B, 1]

    # Step 4: Compute CATE as difference
    cate_final = y1_prob_final - y0_prob_final  # [B, 1]
    unconfoundedCATE= torch.sigmoid(y1_logit_projected-y0_logit_projected)
    confoundedCATE=torch.sigmoid(y1_logit_fallback-y0_logit_fallback)
    return cate_final.view(-1), c.view(-1), unconfoundedCATE.view(-1), confoundedCATE.view(-1) 

'''
@torch.no_grad()
def predict_cate_rpce_in_batches(
    model,
    x_eval,
    x_rct_ref,
    batch_size=256,
    device=None
):
    """
    Batched wrapper around predict_cate_rpce.

    Parameters
    ----------
    model : torch.nn.Module
    x_eval : torch.Tensor
        Covariates to evaluate CATE on.
    x_rct_ref : torch.Tensor
        RCT covariates used as the RPCE reference/projection set.
    batch_size : int
    device : str or None

    Returns
    -------
    cate_pred : torch.Tensor, shape [N]
    conf : torch.Tensor, shape [N]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device).eval()
    x_eval = x_eval.float()
    x_rct_ref = x_rct_ref.float()

    cate_all = []
    conf_all = []

    for start in range(0, x_eval.shape[0], batch_size):
        end = start + batch_size
        x_batch = x_eval[start:end].to(device)
        cate_batch, conf_batch, unconfoundedCATE_batch, confoundedCATE_batch= predict_cate_rpce(
            model=model,
            x_obs=x_batch,
            x_rct=x_rct_ref.to(device),
            device=device
        )
        cate_all.append(cate_batch.detach().cpu().view(-1))
        conf_all.append(conf_batch.detach().cpu().view(-1))

    cate_pred = torch.cat(cate_all, dim=0)
    conf = torch.cat(conf_all, dim=0)
    return cate_pred, conf


def estimate_policy_value_from_rct(
    cate_pred,
    t,
    y,
    threshold=0.0
):
    """
    Estimate policy value on randomized data.

    Policy:
        pi(x) = 1 if cate_pred > threshold else 0

    Value estimator:
        V_hat = P(pi=1) * E[Y | T=1, pi=1]
              + P(pi=0) * E[Y | T=0, pi=0]

    This matches the standard Jobs-style plug-in policy value estimator
    on the randomized subset.

    Parameters
    ----------
    cate_pred : torch.Tensor, shape [N]
    t : torch.Tensor, shape [N]
        Observed treatment, must be 0/1.
    y : torch.Tensor, shape [N]
        Observed outcome, typically binary for Jobs.
    threshold : float
        Treatment threshold.

    Returns
    -------
    metrics : dict
        Contains policy, policy_value, policy_risk, and diagnostic counts.
    """
    cate_pred = cate_pred.view(-1).float().cpu()
    t = t.view(-1).float().cpu()
    y = y.view(-1).float().cpu()

    policy = (cate_pred > threshold).float()

    treat_mask = (policy == 1)
    control_mask = (policy == 0)

    p_treat_policy = treat_mask.float().mean().item()
    p_control_policy = control_mask.float().mean().item()

    # E[Y1 | pi=1] estimated from randomized treated units in policy-treated group
    treated_in_group = ((t == 1) & treat_mask)
    if treated_in_group.any():
        mu1 = y[treated_in_group].mean().item()
    else:
        mu1 = 0.0

    # E[Y0 | pi=0] estimated from randomized control units in policy-control group
    control_in_group = ((t == 0) & control_mask)
    if control_in_group.any():
        mu0 = y[control_in_group].mean().item()
    else:
        mu0 = 0.0

    policy_value = p_treat_policy * mu1 + p_control_policy * mu0
    policy_risk = 1.0 - policy_value

    return {
        "policy": policy,
        "policy_value": policy_value,
        "policy_risk": policy_risk,
        "p_policy_treat": p_treat_policy,
        "p_policy_control": p_control_policy,
        "mu1_policy_treat": mu1,
        "mu0_policy_control": mu0,
        "n_policy_treat": int(treat_mask.sum().item()),
        "n_policy_control": int(control_mask.sum().item()),
        "n_treated_in_policy_treat": int(treated_in_group.sum().item()),
        "n_control_in_policy_control": int(control_in_group.sum().item()),
    }


def estimate_att_from_predictions(cate_pred, t):
    """
    Estimate ATT from predicted individual treatment effects.

    ATT_hat = average predicted tau(x) over treated units only

    Parameters
    ----------
    cate_pred : torch.Tensor, shape [N]
    t : torch.Tensor, shape [N]

    Returns
    -------
    float
    """
    cate_pred = cate_pred.view(-1).float().cpu()
    t = t.view(-1).float().cpu()

    treated_mask = (t == 1)
    if not treated_mask.any():
        raise ValueError("No treated units found, cannot compute ATT_hat.")

    return cate_pred[treated_mask].mean().item()


def empirical_att_from_rct(t, y):
    """
    Empirical ATT target from the randomized subset.

    Since this is randomized data, the treated-control difference in means
    is the standard experimental benchmark used on Jobs-like evaluation.

    Parameters
    ----------
    t : torch.Tensor, shape [N]
    y : torch.Tensor, shape [N]

    Returns
    -------
    float
    """
    t = t.view(-1).float().cpu()
    y = y.view(-1).float().cpu()

    treated = y[t == 1]
    control = y[t == 0]

    if len(treated) == 0 or len(control) == 0:
        raise ValueError("Need both treated and control samples to compute empirical ATT.")

    return (treated.mean() - control.mean()).item()


def evaluate_jobs_policy_risk_and_att(
    model,
    eval_rct_dataset,
    rct_reference_dataset=None,
    batch_size=256,
    threshold=0.0,
    device=None,
    verbose=True
):
    """
    Full Jobs evaluation using the RCT subset.

    Parameters
    ----------
    model : torch.nn.Module
    eval_rct_dataset : TensorDataset
        Should be TensorDataset(X, T, Y), and should be the randomized test subset.
    rct_reference_dataset : TensorDataset or None
        RCT dataset whose X is used as the OT projection reference.
        If None, uses eval_rct_dataset itself.
    batch_size : int
    threshold : float
        Policy threshold; standard choice is 0.0
    device : str or None
    verbose : bool

    Returns
    -------
    metrics : dict
    """
    if not hasattr(eval_rct_dataset, "tensors") or len(eval_rct_dataset.tensors) < 3:
        raise ValueError("eval_rct_dataset must be TensorDataset(X, T, Y)")

    X_eval, T_eval, Y_eval = eval_rct_dataset.tensors[:3]

    if rct_reference_dataset is None:
        X_rct_ref = X_eval
    else:
        if not hasattr(rct_reference_dataset, "tensors"):
            raise ValueError("rct_reference_dataset must be a TensorDataset")
        X_rct_ref = rct_reference_dataset.tensors[0]

    cate_pred, conf = predict_cate_rpce_in_batches(
        model=model,
        x_eval=X_eval,
        x_rct_ref=X_rct_ref,
        batch_size=batch_size,
        device=device
    )

    # Policy risk/value on randomized data
    policy_metrics = estimate_policy_value_from_rct(
        cate_pred=cate_pred,
        t=T_eval,
        y=Y_eval,
        threshold=threshold
    )

    #Predict ATE
    predicted_ATE = cate_pred.mean()

    
    # ATT error
    att_hat = estimate_att_from_predictions(cate_pred, T_eval)
    att_empirical = empirical_att_from_rct(T_eval, Y_eval)
    att_error = abs(att_hat - att_empirical)

    metrics = {
        "policy_value": policy_metrics["policy_value"],
        "policy_risk": policy_metrics["policy_risk"],
        "att_hat": att_hat,
        "att_empirical_rct": att_empirical,
        "att_error": att_error,
        "mean_confidence": conf.mean().item(),
        "std_confidence": conf.std().item(),
        "cate_pred": cate_pred,
        "confidence": conf,
        "policy": policy_metrics["policy"],
        "policy_details": policy_metrics,
        "Predicted_ATE": predicted_ATE
    }
    
    if verbose:
        print(f"Policy value:     {metrics['policy_value']:.6f}")
        print(f"Policy risk:      {metrics['policy_risk']:.6f}")
        print(f"ATT_hat:          {metrics['att_hat']:.6f}")
        print(f"ATT_empirical:    {metrics['att_empirical_rct']:.6f}")
        print(f"ATT error:        {metrics['att_error']:.6f}")
        print(f"Mean confidence:  {metrics['mean_confidence']:.6f}")
        print(f"Std confidence:   {metrics['std_confidence']:.6f}")
        print(f"Predicted ATE:   {metrics['Predicted_ATE']:.6f}")

    return metrics


#Testing Function
def test_mixed_autoencoder(
    model,
    dataset,
    binary_idx=None,
    continuous_idx=None,
    batch_size=64,
    device=None,
    verbose=True
):
    """
    Test an autoencoder on a mixed dataset with binary and continuous features.

    Parameters
    ----------
    model : torch.nn.Module
        Model returning X_recon = model(X_batch)
    dataset : TensorDataset
        Usually TensorDataset(X, t, y) or TensorDataset(X)
    binary_idx : list or None
        Indices of binary columns. If None, detect automatically.
    continuous_idx : list or None
        Indices of continuous columns. If None, detect automatically.
    batch_size : int
        Batch size for evaluation
    device : str or None
        "cuda" or "cpu"
    verbose : bool
        Whether to print metrics

    Returns
    -------
    metrics : dict
        Dictionary with total loss, binary loss, continuous loss,
        continuous RMSE, and binary accuracy
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    if not hasattr(dataset, "tensors"):
        raise ValueError("dataset must be a TensorDataset")

    X_tensor = dataset.tensors[0].float()

    if binary_idx is None or continuous_idx is None:
        binary_idx, continuous_idx = detect_binary_continuous_columns(X_tensor)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mse = nn.MSELoss(reduction="sum")
    bce = nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    total_cont_loss = 0.0
    total_bin_loss = 0.0
    total_prop_loss=0.0
    total_pseudo_loss=0.0

    total_cont_count = 0
    total_bin_count = 0
    correct_bin = 0
    correct_prop=0
    correct_pseudo=0
    total_prop_count=0
    total_pseudo_count=0

    with torch.no_grad():
        for X_batch, T_batch, Y_batch in loader:
            X_batch = X_batch.float().to(device)
            T_batch = T_batch.float().to(device)
            Y_batch = Y_batch.float().to(device)
            
            outputs = model(X_batch)
            X_recon = outputs["x_recon"]
            
            batch_loss = 0.0

            # Continuous part
            if len(continuous_idx) > 0:
                cont_true = X_batch[:, continuous_idx]
                cont_pred = X_recon[:, continuous_idx]

                cont_loss = mse(cont_pred, cont_true)
                total_cont_loss += cont_loss.item()
                total_cont_count += cont_true.numel()
                batch_loss += cont_loss

            # Binary part
            if len(binary_idx) > 0:
                bin_true = X_batch[:, binary_idx]
                bin_logits = X_recon[:, binary_idx]

                bin_loss = bce(bin_logits, bin_true)
                total_bin_loss += bin_loss.item()
                total_bin_count += bin_true.numel()
                batch_loss += bin_loss

                bin_probs = torch.sigmoid(bin_logits)
                bin_pred = (bin_probs > 0.5).float()
                correct_bin += (bin_pred == bin_true).sum().item()
            #Propensity Score Part
            t_true = T_batch.float().view(-1, 1)   # shape [batch, 1]
            t_logits = outputs["t_logit"]          # shape [batch, 1]

            prop_loss = bce(t_logits, t_true)
            total_prop_loss += prop_loss.item()
            total_prop_count += t_true.numel()
            batch_loss += prop_loss

            t_probs = torch.sigmoid(t_logits)
            t_pred = (t_probs > 0.5).float()
            correct_prop += (t_pred == t_true).sum().item()

            #Pseudo-outcome (binary Y)

            y_true = Y_batch.float().view(-1, 1)   # [B,1]

            y0_logits = outputs["y0_pseudo"]       # [B,1]
            y1_logits = outputs["y1_pseudo"]       # [B,1]

            # Compute losses per branch
            bce_none=nn.BCEWithLogitsLoss(reduction="none")
            loss0 = bce_none(y0_logits, y_true)         # [B,1] if reduction="none"
            loss1 = bce_none(y1_logits, y_true)

            # Masked factual loss
            pseudo_loss = (1 - t_true) * loss0 + t_true * loss1

            # Aggregate
            total_pseudo_loss += pseudo_loss.sum().item()
            total_pseudo_count += y_true.numel()
            batch_loss += pseudo_loss.mean()

            # Predictions (only factual branch matters)
            y0_probs = torch.sigmoid(y0_logits)
            y1_probs = torch.sigmoid(y1_logits)

            y_pred = (1 - t_true) * (y0_probs > 0.5).float() + t_true * (y1_probs > 0.5).float()

            correct_pseudo += (y_pred == y_true).sum().item()
            
            
            total_loss += batch_loss.item()

    # Normalize to per-element average
    avg_cont_loss = total_cont_loss / total_cont_count if total_cont_count > 0 else 0.0
    avg_bin_loss = total_bin_loss / total_bin_count if total_bin_count > 0 else 0.0
    avg_prop_loss = total_prop_loss / total_prop_count if total_prop_count > 0 else 0.0
    avg_pseudo_loss = total_pseudo_loss / total_pseudo_count if total_pseudo_count > 0 else 0.0
    avg_total_loss = 0.0

    denom = total_cont_count + total_bin_count + total_prop_count + total_pseudo_count
    if denom > 0:
        avg_total_loss = (total_cont_loss + total_bin_loss + total_prop_loss + total_pseudo_loss) / denom

    cont_rmse = avg_cont_loss ** 0.5 if total_cont_count > 0 else 0.0
    bin_accuracy = correct_bin / total_bin_count if total_bin_count > 0 else 0.0
    prop_accuracy = correct_prop / total_prop_count if total_prop_count > 0 else 0.0
    pseudo_accuracy = correct_pseudo / total_pseudo_count if total_pseudo_count > 0 else 0.0

    metrics = {
        "total_loss": avg_total_loss,
        "continuous_loss_mse": avg_cont_loss,
        "continuous_rmse": cont_rmse,
        "binary_loss_bce": avg_bin_loss,
        "binary_accuracy": bin_accuracy,
        "binary_idx": binary_idx,
        "continuous_idx": continuous_idx,
        "prop_loss_bce": avg_prop_loss,
        "prop_accuracy":prop_accuracy,
        "pseudo_loss":avg_pseudo_loss,
        "pseudo_accuracy":pseudo_accuracy
    }

    if verbose:
        print("Test results")
        print(f"Total loss:          {metrics['total_loss']:.4f}")
        print(f"Continuous MSE:      {metrics['continuous_loss_mse']:.4f}")
        print(f"Continuous RMSE:     {metrics['continuous_rmse']:.4f}")
        print(f"Binary BCE:          {metrics['binary_loss_bce']:.4f}")
        print(f"Binary accuracy:     {metrics['binary_accuracy']:.4f}")
        print(f"Propensity BCE Loss:     {metrics['prop_loss_bce']:.4f}")
        print(f"Propensity accuracy:     {metrics['prop_accuracy']:.4f}")
        print(f"Pseudo-outcome Binary Loss:     {metrics['pseudo_loss']:.4f}")
        print(f"Pseudo-outcome accuracy:     {metrics['pseudo_accuracy']:.4f}")
        print(f"Binary columns:      {metrics['binary_idx']}")
        print(f"Continuous columns:  {metrics['continuous_idx']}")

    return metrics


def calculate_ate_error(mu1: Union[np.ndarray, pd.Series], 
                        mu0: Union[np.ndarray, pd.Series],
                        predicted_ite: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate ATE Error (Average Treatment Effect Error).
    
    The ATE error is the absolute difference between the estimated average
    treatment effect and the true average treatment effect.
    
    Formula: ε_ATE = |ÂTE - ATE|
    where:
        ATE = mean(μ₁ - μ₀)           [true average treatment effect]
        ÂTE = mean(predicted_ite)      [estimated average treatment effect]
    
    Parameters
    ----------
    mu1 : array-like
        True potential outcomes under treatment (T=1)
    mu0 : array-like
        True potential outcomes under control (T=0)
    predicted_ite : array-like
        Model's predicted individual treatment effects
    
    Returns
    -------
    float
        ATE error (lower is better)
    
    Examples
    --------
    >>> mu1 = np.array([10, 12, 15])
    >>> mu0 = np.array([5, 6, 8])
    >>> predicted_ite = np.array([4.5, 5.5, 6.5])
    >>> calculate_ate_error(mu1, mu0, predicted_ite)
    0.5
    
    >>> # With pandas DataFrame
    >>> import pandas as pd
    >>> df = pd.read_excel('data.xlsx')
    >>> ate_error = calculate_ate_error(df['mu1'], df['mu0'], df['RPCE_CATE'])
    >>> print(f"ATE Error: {ate_error:.6f}")
    """
    # Convert to numpy arrays (handle pandas Series properly)
    mu1 = mu1.to_numpy() if isinstance(mu1, pd.Series) else np.asarray(mu1)
    mu0 = mu0.to_numpy() if isinstance(mu0, pd.Series) else np.asarray(mu0)
    predicted_ite = predicted_ite.to_numpy() if isinstance(predicted_ite, pd.Series) else np.asarray(predicted_ite)
    
    # Calculate true ATE
    true_ite = mu1 - mu0
    true_ate = np.mean(true_ite)
    
    # Calculate estimated ATE
    estimated_ate = np.mean(predicted_ite)
    
    # Calculate absolute error
    ate_error = abs(estimated_ate - true_ate)
    
    return float(ate_error)
 
 
def calculate_pehe(mu1: Union[np.ndarray, pd.Series], 
                   mu0: Union[np.ndarray, pd.Series],
                   predicted_ite: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate PEHE (Precision in Estimation of Heterogeneous Effect).
    
    PEHE is the root mean squared error (RMSE) between predicted and true
    individual treatment effects. It measures how well the model captures
    heterogeneous treatment effects across individuals.
    
    Formula: PEHE = √[(1/n) × Σ(predicted_iteᵢ - true_iteᵢ)²]
    where:
        true_iteᵢ = μ₁(xᵢ) - μ₀(xᵢ)   [true individual treatment effect]
    
    Parameters
    ----------
    mu1 : array-like
        True potential outcomes under treatment (T=1)
    mu0 : array-like
        True potential outcomes under control (T=0)
    predicted_ite : array-like
        Model's predicted individual treatment effects
    
    Returns
    -------
    float
        PEHE (lower is better)
    
    Examples
    --------
    >>> mu1 = np.array([10, 12, 15])
    >>> mu0 = np.array([5, 6, 8])
    >>> predicted_ite = np.array([4.5, 5.5, 6.5])
    >>> calculate_pehe(mu1, mu0, predicted_ite)
    0.5
    
    >>> # With pandas DataFrame
    >>> import pandas as pd
    >>> df = pd.read_excel('data.xlsx')
    >>> pehe = calculate_pehe(df['mu1'], df['mu0'], df['RPCE_CATE'])
    >>> print(f"PEHE: {pehe:.6f}")
    """
    # Convert to numpy arrays (handle pandas Series properly)
    mu1 = mu1.to_numpy() if isinstance(mu1, pd.Series) else np.asarray(mu1)
    mu0 = mu0.to_numpy() if isinstance(mu0, pd.Series) else np.asarray(mu0)
    predicted_ite = predicted_ite.to_numpy() if isinstance(predicted_ite, pd.Series) else np.asarray(predicted_ite)
    
    # Calculate true ITE
    true_ite = mu1 - mu0
    
    # Calculate individual errors
    errors = predicted_ite - true_ite
    
    # Calculate RMSE (PEHE)
    pehe = np.sqrt(np.mean(errors ** 2))
    
    return float(pehe)