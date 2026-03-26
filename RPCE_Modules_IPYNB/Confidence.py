
import torch



def compute_confidence(z_obs, z_rct, pi_star, propensity_head, gamma=5.0):
    """
    Returns confidence scores with consistent shape [B, 1].
    """
    # Propensity confidence
    e_hat = torch.sigmoid(propensity_head(z_obs)).view(-1)   # [B]
    c_prop = torch.exp(-gamma * (e_hat - 0.5) ** 2)          # [B]

    # Geometric confidence
    C = torch.cdist(z_obs, z_rct, p=2) ** 2                  # [B, M]
    weighted_dist = (pi_star * C).sum(dim=1) / pi_star.sum(dim=1).clamp_min(1e-8)  # [B]
    
    dist_median = weighted_dist.median().clamp_min(1e-8)
    normalized_dist = weighted_dist / dist_median
    
    c_geo = torch.exp(-normalized_dist)                        # [B]

    # Combined confidence
    c = c_prop * c_geo                                       # [B]

    # reshape to [B,1] so it matches outcome heads
    return c.unsqueeze(1), c_prop.unsqueeze(1), c_geo.unsqueeze(1)
