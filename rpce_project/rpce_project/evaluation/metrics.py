"""
Evaluation metrics for causal inference models.
"""
import torch


def estimate_policy_value_from_rct(cate_pred, t, y, threshold=0.0):
    """
    Estimate policy value on randomized data.
    
    Policy: pi(x) = 1 if cate_pred > threshold else 0
    
    Value estimator:
        V_hat = P(pi=1) * E[Y | T=1, pi=1] + P(pi=0) * E[Y | T=0, pi=0]
    
    Args:
        cate_pred (torch.Tensor): CATE predictions [N]
        t (torch.Tensor): Observed treatment [N]
        y (torch.Tensor): Observed outcome [N]
        threshold (float): Treatment threshold
    
    Returns:
        dict: Policy metrics including value, risk, and diagnostics
    """
    cate_pred = cate_pred.view(-1).float().cpu()
    t = t.view(-1).float().cpu()
    y = y.view(-1).float().cpu()
    
    # Determine policy
    policy = (cate_pred > threshold).float()
    
    treat_mask = (policy == 1)
    control_mask = (policy == 0)
    
    p_treat_policy = treat_mask.float().mean().item()
    p_control_policy = control_mask.float().mean().item()
    
    # E[Y1 | pi=1] from randomized treated units in policy-treated group
    treated_in_group = ((t == 1) & treat_mask)
    mu1 = y[treated_in_group].mean().item() if treated_in_group.any() else 0.0
    
    # E[Y0 | pi=0] from randomized control units in policy-control group
    control_in_group = ((t == 0) & control_mask)
    mu0 = y[control_in_group].mean().item() if control_in_group.any() else 0.0
    
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
    
    Args:
        cate_pred (torch.Tensor): CATE predictions [N]
        t (torch.Tensor): Treatment indicator [N]
    
    Returns:
        float: Estimated ATT
    """
    cate_pred = cate_pred.view(-1).float().cpu()
    t = t.view(-1).float().cpu()
    
    treated_mask = (t == 1)
    if not treated_mask.any():
        raise ValueError("No treated units found, cannot compute ATT_hat.")
    
    return cate_pred[treated_mask].mean().item()


def empirical_att_from_rct(t, y):
    """
    Empirical ATT from randomized subset (ground truth).
    
    Args:
        t (torch.Tensor): Treatment [N]
        y (torch.Tensor): Outcome [N]
    
    Returns:
        float: Empirical ATT
    """
    t = t.view(-1).float().cpu()
    y = y.view(-1).float().cpu()
    
    treated = y[t == 1]
    control = y[t == 0]
    
    if len(treated) == 0 or len(control) == 0:
        raise ValueError("Need both treated and control samples to compute empirical ATT.")
    
    return (treated.mean() - control.mean()).item()


def compute_pehe(cate_pred, cate_true):
    """
    Compute Precision in Estimation of Heterogeneous Effects (PEHE).
    
    PEHE = sqrt(E[(tau_pred - tau_true)^2])
    
    Args:
        cate_pred (torch.Tensor): Predicted CATE [N]
        cate_true (torch.Tensor): True CATE [N]
    
    Returns:
        float: PEHE score (lower is better)
    """
    cate_pred = cate_pred.view(-1).float().cpu()
    cate_true = cate_true.view(-1).float().cpu()
    
    mse = ((cate_pred - cate_true) ** 2).mean()
    pehe = torch.sqrt(mse).item()
    
    return pehe


def compute_ate_error(cate_pred, cate_true):
    """
    Compute ATE estimation error.
    
    Args:
        cate_pred (torch.Tensor): Predicted CATE [N]
        cate_true (torch.Tensor): True CATE [N]
    
    Returns:
        float: Absolute ATE error
    """
    ate_pred = cate_pred.mean().item()
    ate_true = cate_true.mean().item()
    
    return abs(ate_pred - ate_true)
