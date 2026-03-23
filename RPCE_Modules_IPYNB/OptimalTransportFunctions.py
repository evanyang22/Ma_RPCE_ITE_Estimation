#import numpy as np
import torch
#import torch.nn as nn
#from torch.utils.data import DataLoader, TensorDataset
#import matplotlib.pyplot as plt
#import random
#import copy
import ot
#import sys
#from RPCE_Model import AutoEncoder
def sinkhorn_projection(z_obs, z_rct, epsilon=0.5, max_iter=10000):
    """
    1. Computes an optimal transport plan between OBS and RCT representations
    2. Creates a projected version of each OBS point as a weighted average of RCT points
    z_obs: (n, d) OBS representations  
    z_rct: (m, d) RCT representations
    d= dimension of each representation
    n = number of obs samples
    m = number of RCT samples
    """
    n, d = z_obs.shape
    m = z_rct.shape[0]
    
    # Cost matrix
    C = torch.cdist(z_obs, z_rct, p=2) ** 2  # squared Euclidean
    C = C / C.mean().clamp_min(1e-8)

    #print("C min:", C.min().item())
    #print("C max:", C.max().item())
   # print("C mean:", C.mean().item())
    
    # Uniform marginals
    a = torch.ones(n) / n
    b = torch.ones(m) / m
    
    # Sinkhorn
    pi_star = ot.sinkhorn(a.cpu().numpy(), b.cpu().numpy(), 
                          C.cpu().numpy(), epsilon, numItermax=max_iter)
    pi_star = torch.from_numpy(pi_star).float()
    
    # Barycentric projection
    
    #z_tilde normalized
    row_mass = pi_star.sum(dim=1, keepdim=True)
    z_tilde = torch.mm(pi_star, z_rct) / row_mass
    
    
    return z_tilde, pi_star

def sinkhorn_projection_unbalanced(z_obs, z_rct, epsilon=0.1, tau=0.5, max_iter=1000):
    """
    Unbalanced OT - allows marginal violations, more robust
    tau controls how much mass can be destroyed (lower = more flexible)
    """
    n, d = z_obs.shape
    m = z_rct.shape[0]
    
    C = torch.cdist(z_obs, z_rct, p=2) ** 2
    C_median = C.median()
    C = C / C_median.clamp_min(1e-8)
    
    a = torch.ones(n) / n
    b = torch.ones(m) / m
    
    # Use unbalanced Sinkhorn
    pi_star = ot.unbalanced.sinkhorn_knopp_unbalanced(
        a.cpu().numpy(),
        b.cpu().numpy(),
        C.cpu().numpy(),
        reg=epsilon,
        reg_m=tau,  # Mass relaxation parameter
        numItermax=max_iter,
        stopThr=1e-6
    )
    pi_star = torch.from_numpy(pi_star).float()
    
    row_mass = pi_star.sum(dim=1, keepdim=True).clamp_min(1e-8)
    z_tilde = torch.mm(pi_star, z_rct) / row_mass
    
    return z_tilde, pi_star

def sinkhorn_projection_balanced(z_obs, z_rct, epsilon=0.5, max_iter=2000):
    """
    Balanced OT using standard Sinkhorn algorithm.
    More stable than unbalanced for small sample sizes.
    """
    n, d = z_obs.shape
    m = z_rct.shape[0]
    
    # Compute cost matrix
    C = torch.cdist(z_obs, z_rct, p=2) ** 2
    
    # Robust normalization: use mean + std instead of median
    C_scale = C.mean() + C.std()
    C = C / C_scale.clamp_min(1e-8)
    
    # Clip extreme values to prevent numerical issues
    C = torch.clamp(C, min=0.0, max=50.0)
    
    # Uniform marginals
    a = torch.ones(n) / n
    b = torch.ones(m) / m
    
    # Use standard balanced Sinkhorn (more stable)
    try:
        pi_star = ot.sinkhorn(
            a.cpu().numpy(),
            b.cpu().numpy(),
            C.cpu().numpy(),
            reg=epsilon,  # Higher epsilon = more regularization = more stable
            numItermax=max_iter,
            stopThr=1e-8,
            method='sinkhorn',  # Classic Sinkhorn
            verbose=False
        )
        pi_star = torch.from_numpy(pi_star).float()
        
        # Verify convergence
        if torch.isnan(pi_star).any() or torch.isinf(pi_star).any():
            raise ValueError("OT plan contains NaN or Inf")
            
    except Exception as e:
        print(f"Sinkhorn failed: {e}")
        # Fallback: uniform assignment
        pi_star = torch.ones(n, m) / m
    
    # Barycentric projection
    row_mass = pi_star.sum(dim=1, keepdim=True).clamp_min(1e-8)
    z_tilde = torch.mm(pi_star, z_rct) / row_mass
    
    return z_tilde, pi_star
