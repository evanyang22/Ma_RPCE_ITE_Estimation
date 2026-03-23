"""
Data loading utilities for Jobs dataset.
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset


def createJobsTensorDataset(data_dir, split_by_e=True, return_type="both"):
    """
    Creates TensorDatasets from Jobs dataset using experimental indicator e.

    Args:
        data_dir (str): path to .npz file
        split_by_e (bool): whether to split dataset by experimental indicator
        return_type (str): "both", "rct", or "obs"

    Returns:
        TensorDataset(s): depending on return_type
            - "both": (rct_dataset, obs_dataset)
            - "rct": rct_dataset only
            - "obs": obs_dataset only
    """
    npz = np.load(data_dir)

    # Extract tensors (0th replication)
    X = torch.tensor(npz["x"][:, :, 1], dtype=torch.float32)
    y = torch.tensor(npz["yf"][:, 1], dtype=torch.float32)
    t = torch.tensor(npz["t"][:, 1], dtype=torch.float32)

    # Experimental indicator
    if "e" not in npz:
        raise ValueError("Dataset does not contain experimental indicator 'e'")

    e = torch.tensor(npz["e"][:, 0], dtype=torch.float32)

    if not split_by_e:
        return TensorDataset(X, t, y, e)

    # Create masks for RCT vs observational data
    rct_mask = (e == 1)
    obs_mask = (e == 0)

    # Split datasets
    rct_dataset = TensorDataset(
        X[rct_mask], t[rct_mask], y[rct_mask]
    )

    obs_dataset = TensorDataset(
        X[obs_mask], t[obs_mask], y[obs_mask]
    )

    # Return based on preference
    if return_type == "both":
        return rct_dataset, obs_dataset
    elif return_type == "rct":
        return rct_dataset
    elif return_type == "obs":
        return obs_dataset
    else:
        raise ValueError("return_type must be 'both', 'rct', or 'obs'")


def load_jobs_data(train_path, test_path, split_by_e=True):
    """
    Convenience function to load both train and test datasets.
    
    Args:
        train_path (str): Path to training .npz file
        test_path (str): Path to test .npz file
        split_by_e (bool): Whether to split by experimental indicator
    
    Returns:
        dict: Dictionary containing train/test RCT and observational datasets
    """
    train_rct, train_obs = createJobsTensorDataset(
        train_path, split_by_e=split_by_e, return_type="both"
    )
    
    test_rct, test_obs = createJobsTensorDataset(
        test_path, split_by_e=split_by_e, return_type="both"
    )
    
    return {
        "train_rct": train_rct,
        "train_obs": train_obs,
        "test_rct": test_rct,
        "test_obs": test_obs
    }
