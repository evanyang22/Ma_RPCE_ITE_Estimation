"""
Optimal Transport via Sinkhorn projection for domain adaptation.

This module implements different variants of Sinkhorn algorithm for
transporting latent representations between observational and RCT domains.
"""
import torch
import ot


def sinkhorn_projection(z_obs, z_rct, epsilon=0.5, max_iter=10000):
    """
    Sinkhorn projection using POT library.
    
    Args:
        z_obs (torch.Tensor): Latent representations from observational data [N, latent_dim]
        z_rct (torch.Tensor): Latent representations from RCT data [M, latent_dim]
        epsilon (float): Regularization parameter
        max_iter (int): Maximum iterations
    
    Returns:
        torch.Tensor: Transported observational representations [N, latent_dim]
    """
    # Convert to numpy for POT library
    z_obs_np = z_obs.detach().cpu().numpy()
    z_rct_np = z_rct.detach().cpu().numpy()
    
    # Compute cost matrix (Euclidean distance)
    M = ot.dist(z_obs_np, z_rct_np, metric='euclidean')
    
    # Uniform distributions
    a = ot.unif(z_obs_np.shape[0])
    b = ot.unif(z_rct_np.shape[0])
    
    # Compute optimal transport plan
    T = ot.sinkhorn(a, b, M, reg=epsilon, numItermax=max_iter)
    
    # Transport z_obs to RCT domain
    z_transported = torch.tensor(
        T @ z_rct_np * z_obs_np.shape[0],
        dtype=z_obs.dtype,
        device=z_obs.device
    )
    
    return z_transported


def sinkhorn_projection_unbalanced(z_obs, z_rct, epsilon=0.1, tau=0.5, max_iter=1000):
    """
    Unbalanced Sinkhorn projection.
    
    Allows for mass creation/destruction, useful when sample sizes differ significantly.
    
    Args:
        z_obs (torch.Tensor): Latent representations from observational data
        z_rct (torch.Tensor): Latent representations from RCT data
        epsilon (float): Entropic regularization
        tau (float): Marginal relaxation (smaller = more unbalanced)
        max_iter (int): Maximum iterations
    
    Returns:
        torch.Tensor: Transported observational representations
    """
    z_obs_np = z_obs.detach().cpu().numpy()
    z_rct_np = z_rct.detach().cpu().numpy()
    
    M = ot.dist(z_obs_np, z_rct_np, metric='euclidean')
    
    a = ot.unif(z_obs_np.shape[0])
    b = ot.unif(z_rct_np.shape[0])
    
    # Unbalanced OT with KL relaxation
    T = ot.sinkhorn_unbalanced(
        a, b, M,
        reg=epsilon,
        reg_m=tau,
        numItermax=max_iter
    )
    
    z_transported = torch.tensor(
        T @ z_rct_np * z_obs_np.shape[0],
        dtype=z_obs.dtype,
        device=z_obs.device
    )
    
    return z_transported


def sinkhorn_projection_balanced(z_obs, z_rct, epsilon=0.5, max_iter=2000):
    """
    Balanced Sinkhorn projection with exact marginal constraints.
    
    Args:
        z_obs (torch.Tensor): Latent representations from observational data
        z_rct (torch.Tensor): Latent representations from RCT data
        epsilon (float): Entropic regularization
        max_iter (int): Maximum iterations
    
    Returns:
        torch.Tensor: Transported observational representations
    """
    return sinkhorn_projection(z_obs, z_rct, epsilon=epsilon, max_iter=max_iter)


def compute_wasserstein_distance(z_obs, z_rct, epsilon=0.5):
    """
    Compute Wasserstein distance between two distributions.
    
    Args:
        z_obs (torch.Tensor): First distribution
        z_rct (torch.Tensor): Second distribution
        epsilon (float): Regularization parameter
    
    Returns:
        float: Wasserstein distance
    """
    z_obs_np = z_obs.detach().cpu().numpy()
    z_rct_np = z_rct.detach().cpu().numpy()
    
    M = ot.dist(z_obs_np, z_rct_np, metric='euclidean')
    
    a = ot.unif(z_obs_np.shape[0])
    b = ot.unif(z_rct_np.shape[0])
    
    # Compute Sinkhorn distance
    distance = ot.sinkhorn2(a, b, M, reg=epsilon)
    
    return float(distance)


def predict_cate_rpce(model, x_obs, x_rct, device=None, transport_method="balanced"):
    """
    Predict CATE using RPCE (Robust Proximal Causal Effect) method.
    
    This combines:
    1. Encoding observational covariates to latent space
    2. Optimal transport to RCT domain
    3. Prediction using unconfounded RCT outcome heads
    
    Args:
        model (nn.Module): Trained AutoEncoder model
        x_obs (torch.Tensor): Observational covariates to predict on
        x_rct (torch.Tensor): RCT covariates (reference distribution)
        device (str, optional): Device to use
        transport_method (str): "balanced" or "unbalanced"
    
    Returns:
        tuple: (cate_predictions, confidence_scores)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device).eval()
    x_obs = x_obs.to(device)
    x_rct = x_rct.to(device)
    
    with torch.no_grad():
        # Encode both distributions
        z_obs = model.encode(x_obs)
        z_rct = model.encode(x_rct)
        
        # Transport observational to RCT domain
        if transport_method == "balanced":
            z_transported = sinkhorn_projection_balanced(z_obs, z_rct)
        elif transport_method == "unbalanced":
            z_transported = sinkhorn_projection_unbalanced(z_obs, z_rct)
        else:
            raise ValueError(f"Unknown transport method: {transport_method}")
        
        # Predict outcomes using RCT heads
        y0_pred = model.g0_head(z_transported)
        y1_pred = model.g1_head(z_transported)
        
        # CATE = Y1 - Y0
        cate = (y1_pred - y0_pred).squeeze()
        
        # Confidence: inverse of transport distance
        distances = torch.cdist(z_transported, z_rct, p=2)
        min_distances = distances.min(dim=1).values
        confidence = 1.0 / (1.0 + min_distances)
    
    return cate, confidence
