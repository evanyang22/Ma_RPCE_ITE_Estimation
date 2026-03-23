"""
Utility functions for data processing and model training.
"""
import torch


def detect_binary_continuous_columns(X_tensor, tol=1e-6):
    """
    Detect binary vs continuous columns automatically.
    
    A column is considered binary if its unique values are only {0,1}
    within numerical tolerance.
    
    Args:
        X_tensor (torch.Tensor): Input features [N, D]
        tol (float): Numerical tolerance
    
    Returns:
        tuple: (binary_idx, continuous_idx) lists of column indices
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


def compute_ipw_weights(propensity_scores, treatment, clip_min=0.01, clip_max=0.99):
    """
    Compute Inverse Propensity Weighting (IPW) weights.
    
    Args:
        propensity_scores (torch.Tensor): P(T=1|X) [N]
        treatment (torch.Tensor): Treatment indicator [N]
        clip_min (float): Minimum propensity score
        clip_max (float): Maximum propensity score
    
    Returns:
        torch.Tensor: IPW weights [N]
    """
    # Clip propensity scores to avoid extreme weights
    ps_clipped = torch.clamp(propensity_scores, clip_min, clip_max)
    
    # Compute weights: 1/P(T|X) for treated, 1/(1-P(T|X)) for control
    weights = torch.where(
        treatment == 1,
        1.0 / ps_clipped,
        1.0 / (1.0 - ps_clipped)
    )
    
    return weights


def normalize_features(X, feature_means=None, feature_stds=None):
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        X (torch.Tensor): Features [N, D]
        feature_means (torch.Tensor, optional): Pre-computed means
        feature_stds (torch.Tensor, optional): Pre-computed stds
    
    Returns:
        tuple: (X_normalized, means, stds)
    """
    if feature_means is None:
        feature_means = X.mean(dim=0)
    if feature_stds is None:
        feature_stds = X.std(dim=0)
    
    # Avoid division by zero
    feature_stds = torch.where(feature_stds > 0, feature_stds, torch.ones_like(feature_stds))
    
    X_normalized = (X - feature_means) / feature_stds
    
    return X_normalized, feature_means, feature_stds


def get_treatment_stratified_split(dataset, train_ratio=0.8):
    """
    Split dataset while maintaining treatment balance.
    
    Args:
        dataset (TensorDataset): Dataset with (X, T, Y)
        train_ratio (float): Proportion for training
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    from torch.utils.data import Subset
    
    X, t, y = dataset.tensors[:3]
    n = len(dataset)
    
    # Get indices for each treatment group
    t0_idx = (t == 0).nonzero(as_tuple=True)[0]
    t1_idx = (t == 1).nonzero(as_tuple=True)[0]
    
    # Split each group
    n_t0_train = int(len(t0_idx) * train_ratio)
    n_t1_train = int(len(t1_idx) * train_ratio)
    
    # Shuffle
    perm_t0 = torch.randperm(len(t0_idx))
    perm_t1 = torch.randperm(len(t1_idx))
    
    # Create train/val indices
    train_idx = torch.cat([
        t0_idx[perm_t0[:n_t0_train]],
        t1_idx[perm_t1[:n_t1_train]]
    ])
    
    val_idx = torch.cat([
        t0_idx[perm_t0[n_t0_train:]],
        t1_idx[perm_t1[n_t1_train:]]
    ])
    
    # Create subsets
    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())
    
    return train_dataset, val_dataset
