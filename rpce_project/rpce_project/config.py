"""
Configuration and constants for RPCE model.
"""
import random
import numpy as np
import torch


# Random seed for reproducibility
SEED = 42

# Model hyperparameters
DEFAULT_HIDDEN_DIM = 8
DEFAULT_LATENT_DIM = 4
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_NUM_EPOCHS = 50

# Training hyperparameters
RECONSTRUCTION_WEIGHT = 1.0
PROPENSITY_WEIGHT = 1.0
PSEUDO_OUTCOME_WEIGHT = 1.0
RCT_OUTCOME_WEIGHT = 1.0

# Optimal transport parameters
SINKHORN_EPSILON = 0.5
SINKHORN_MAX_ITER = 2000
SINKHORN_TAU = 0.5  # For unbalanced transport

# Evaluation parameters
POLICY_THRESHOLD = 0.0
EVAL_BATCH_SIZE = 256


def set_random_seed(seed=SEED):
    """Set random seed for reproducibility across all libraries."""
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (CPU)
    torch.manual_seed(seed)
    
    # PyTorch (GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device
