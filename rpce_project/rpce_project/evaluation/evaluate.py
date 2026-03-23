"""
Main evaluation pipeline for RPCE model.
"""
import torch

from .metrics import (
    estimate_policy_value_from_rct,
    estimate_att_from_predictions,
    empirical_att_from_rct
)
from ..transport.sinkhorn import predict_cate_rpce


def predict_cate_rpce_in_batches(
    model,
    x_eval,
    x_rct_ref,
    batch_size=256,
    device=None
):
    """
    Batched wrapper around predict_cate_rpce.
    
    Args:
        model (nn.Module): Trained model
        x_eval (torch.Tensor): Covariates to evaluate CATE on
        x_rct_ref (torch.Tensor): RCT covariates used as reference
        batch_size (int): Batch size for processing
        device (str, optional): Device to use
    
    Returns:
        tuple: (cate_pred, confidence) tensors of shape [N]
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
        
        cate_batch, conf_batch = predict_cate_rpce(
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
    
    Args:
        model (nn.Module): Trained model
        eval_rct_dataset (TensorDataset): RCT test dataset (X, T, Y)
        rct_reference_dataset (TensorDataset, optional): RCT reference for OT
        batch_size (int): Batch size for evaluation
        threshold (float): Policy threshold (default 0.0)
        device (str, optional): Device to use
        verbose (bool): Print results
    
    Returns:
        dict: Comprehensive evaluation metrics
    """
    if not hasattr(eval_rct_dataset, "tensors") or len(eval_rct_dataset.tensors) < 3:
        raise ValueError("eval_rct_dataset must be TensorDataset(X, T, Y)")
    
    X_eval, T_eval, Y_eval = eval_rct_dataset.tensors[:3]
    
    # Use eval dataset as reference if not provided
    if rct_reference_dataset is None:
        X_rct_ref = X_eval
    else:
        if not hasattr(rct_reference_dataset, "tensors"):
            raise ValueError("rct_reference_dataset must be a TensorDataset")
        X_rct_ref = rct_reference_dataset.tensors[0]
    
    # Predict CATE with confidence
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
    
    # Predicted ATE
    predicted_ATE = cate_pred.mean().item()
    
    # ATT error
    att_hat = estimate_att_from_predictions(cate_pred, T_eval)
    att_empirical = empirical_att_from_rct(T_eval, Y_eval)
    att_error = abs(att_hat - att_empirical)
    
    # Compile all metrics
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
        "predicted_ATE": predicted_ATE
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RPCE Evaluation Results")
        print(f"{'='*60}")
        print(f"Policy value:     {metrics['policy_value']:.6f}")
        print(f"Policy risk:      {metrics['policy_risk']:.6f}")
        print(f"ATT_hat:          {metrics['att_hat']:.6f}")
        print(f"ATT_empirical:    {metrics['att_empirical_rct']:.6f}")
        print(f"ATT error:        {metrics['att_error']:.6f}")
        print(f"Predicted ATE:    {metrics['predicted_ATE']:.6f}")
        print(f"Mean confidence:  {metrics['mean_confidence']:.6f}")
        print(f"Std confidence:   {metrics['std_confidence']:.6f}")
        print(f"{'='*60}\n")
    
    return metrics
