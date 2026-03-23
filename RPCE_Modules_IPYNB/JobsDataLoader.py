import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def createJobsTensorDataset(data_dir, split_by_e=True,slice=0, return_type="both"):
    """
    Creates TensorDatasets from Jobs dataset using experimental indicator e.

    Args:
        data_dir (str): path to .npz file
        split_by_e (bool): whether to split dataset by e
        return_type (str): "both", "rct", or "obs"

    Returns:
        TensorDataset(s)
    """

    npz = np.load(data_dir)

    # Extract tensors (0th replication)
    X = torch.tensor(npz["x"][:, :, slice], dtype=torch.float32)
    y = torch.tensor(npz["yf"][:, slice], dtype=torch.float32)
    t = torch.tensor(npz["t"][:, slice], dtype=torch.float32)

    # Experimental indicator
    if "e" not in npz:
        raise ValueError("Dataset does not contain experimental indicator 'e'")

    e = torch.tensor(npz["e"][:, 0], dtype=torch.float32)

    if not split_by_e:
        return TensorDataset(X, t, y, e)

    # Masks
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
    

def explore_tensor(X):
    print("Shape:", X.shape)
    print("Mean:", X.mean(dim=0))
    print("Std:", X.std(dim=0))
    print("Min:", X.min(dim=0).values)
    print("Max:", X.max(dim=0).values)

    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.DataFrame(X.numpy())
    df.hist(figsize=(10,6), bins=20)
    plt.show()

def getATE(data_dir,slice=0):
    npz = np.load(data_dir)

    ate = torch.tensor(npz["ate"][:, slice], dtype=torch.float32)
    return ate
