"""
Data exploration and visualization utilities.
"""
import torch
import pandas as pd
import matplotlib.pyplot as plt


def explore_tensor(X, name="Tensor"):
    """
    Print statistics and visualize tensor data.
    
    Args:
        X (torch.Tensor): Tensor to explore
        name (str): Name for the tensor (for display purposes)
    """
    print(f"\n{'='*60}")
    print(f"Exploring {name}")
    print(f"{'='*60}")
    print(f"Shape: {X.shape}")
    print(f"Mean: {X.mean(dim=0)}")
    print(f"Std: {X.std(dim=0)}")
    print(f"Min: {X.min(dim=0).values}")
    print(f"Max: {X.max(dim=0).values}")
    
    # Convert to DataFrame for visualization
    df = pd.DataFrame(X.numpy())
    df.hist(figsize=(10, 6), bins=20)
    plt.suptitle(f"{name} Feature Distributions")
    plt.tight_layout()
    plt.show()


def dataset_summary(dataset, name="Dataset"):
    """
    Print summary statistics for a TensorDataset.
    
    Args:
        dataset (TensorDataset): Dataset to summarize
        name (str): Name for the dataset
    """
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Number of samples: {len(dataset)}")
    
    if len(dataset.tensors) >= 3:
        X, t, y = dataset.tensors[:3]
        print(f"Features (X) shape: {X.shape}")
        print(f"Treatment (t) distribution: {t.unique(return_counts=True)}")
        print(f"Outcome (y) mean: {y.mean():.4f}, std: {y.std():.4f}")
