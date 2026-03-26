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

def sinkhorn_projection_balanced(z_obs, z_rct, epsilon=0.5, max_iter=10000):
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

def sinkhorn_projection_balanced_better(z_obs, z_rct, epsilon=0.5, max_iter=10000):
    """
    Robust balanced OT with adaptive epsilon and better numerical stability.
    """
    n, d = z_obs.shape
    m = z_rct.shape[0]
    
    # Compute cost matrix
    C = torch.cdist(z_obs, z_rct, p=2) ** 2
    '''
    # KEY FIX 1: Use percentile-based normalization (more robust than mean+std)
    C_median = torch.median(C)
    C_75 = torch.quantile(C, 0.75)
    C_scale = (C_75 - C_median).clamp_min(1e-6)
    C = (C - C_median) / C_scale
    
    # KEY FIX 2: Clip to smaller range and ensure non-negative
    C = torch.clamp(C, min=0.0, max=10.0)  # Reduced from 50 to 10
    '''
    # ---- FIXED NORMALIZATION ----
    # Scale costs into a reasonable range for Sinkhorn without introducing
    # negative values.  Use the 75th percentile as a robust scale factor.
    # If all costs are identical (degenerate), fall back to 1.
    C_scale = torch.quantile(C, 0.75).clamp_min(1e-6)
    C = C / C_scale
    # Costs are still >= 0.  Clip extreme outliers only on the high end.
    C = torch.clamp(C, min=0.0, max=20.0)
    
    # Uniform marginals
    a = torch.ones(n) / n
    b = torch.ones(m) / m
    
    # KEY FIX 3: Adaptive epsilon - start high for stability
    # Higher epsilon = more entropy regularization = easier convergence
    epsilon_adaptive = max(epsilon, 0.1)  # Ensure minimum epsilon
    
    # KEY FIX 4: Try with progressively looser tolerances
    for attempt, (tol, method) in enumerate([
        (1e-6, 'sinkhorn_stabilized'),  # Most stable
        (1e-5, 'sinkhorn'),              # Classic
        (1e-4, 'sinkhorn_log'),          # Log-domain
    ]):
        try:
            pi_star = ot.sinkhorn(
                a.cpu().numpy(),
                b.cpu().numpy(),
                C.cpu().numpy(),
                reg=epsilon_adaptive,
                numItermax=max_iter,
                stopThr=tol,
                method=method,
                verbose=False,
                warn=False  # Suppress warnings
            )
            pi_star = torch.from_numpy(pi_star).float()
            
            # Verify convergence
            if not (torch.isnan(pi_star).any() or torch.isinf(pi_star).any()):
                # Additional check: verify marginal constraints
                row_sums = pi_star.sum(dim=1)
                col_sums = pi_star.sum(dim=0)
                if (torch.abs(row_sums - a).max() < 0.01 and 
                    torch.abs(col_sums - b).max() < 0.01):
                    break  # Success!
            else:
                raise ValueError("NaN/Inf detected")
                
        except Exception as e:
            if attempt == 2:  # Last attempt failed
                print(f"All Sinkhorn methods failed. Using fallback.")
                print(f"Cost matrix stats: min={C.min():.3f}, max={C.max():.3f}, "
                      f"mean={C.mean():.3f}, std={C.std():.3f}")
                # Fallback: nearest neighbor assignment
                pi_star = torch.zeros(n, m)
                nearest = torch.argmin(C, dim=1)
                pi_star[torch.arange(n), nearest] = 1.0 / m
            else:
                continue  # Try next method
    
    # Barycentric projection with numerical safety
    row_mass = pi_star.sum(dim=1, keepdim=True).clamp_min(1e-8)
    z_tilde = torch.mm(pi_star, z_rct) / row_mass
    
    return z_tilde, pi_star
