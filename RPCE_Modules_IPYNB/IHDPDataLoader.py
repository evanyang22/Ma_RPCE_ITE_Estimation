import numpy as np
import torch
from torch.utils.data import TensorDataset
def createIHDPTensorDataset(data_dir):
    npz=np.load(data_dir)
    X = torch.tensor(npz["x"][:, :, 0], dtype=torch.float32)
    y = torch.tensor(npz["yf"][:, 0], dtype=torch.float32)
    t = torch.tensor(npz["t"][:, 0], dtype=torch.float32)

    return TensorDataset(X,t,y)

def create_ihdp_rct_obs_datasets(
    npz_path,
    replication=0,
    rct_fraction=0.2,
    seed=42,
    bias_strength=1.0,
    noise_std=0.1,
    standardize_for_propensity=True,
    return_numpy=False
):
    """
    Split one IHDP training replication into an RCT-like subset and an OBS-like subset
    with confounding.

    Parameters
    ----------
    npz_path : str
        Path to IHDP .npz file.
    replication : int
        Which replication/simulation index to use.
    rct_fraction : float
        Fraction of samples assigned to the RCT subset.
    seed : int
        Random seed for reproducibility.
    bias_strength : float
        Strength of confounding in OBS treatment assignment.
        Larger values => treatment depends more strongly on X.
    noise_std : float
        Std of Gaussian noise added to regenerated outcomes.
    standardize_for_propensity : bool
        Whether to z-score X before constructing OBS propensities.
    return_numpy : bool
        If True, also return numpy arrays/dictionaries.

    Returns
    -------
    rct_dataset : TensorDataset
        TensorDataset(X, t, yf, ycf, mu0, mu1)
    obs_dataset : TensorDataset
        TensorDataset(X, t, yf, ycf, mu0, mu1)

    Optionally also returns a dict of numpy arrays if return_numpy=True.

    Notes
    -----
    Expected keys in the IHDP npz:
        ['ate', 'mu1', 'mu0', 'yadd', 'yf', 'ycf', 't', 'x', 'ymul']

    Typical shapes:
        x   : (n, p, R)
        mu0 : (n, R)
        mu1 : (n, R)
        t   : (n, R)
        yf  : (n, R)
        ycf : (n, R)
    """

    rng = np.random.default_rng(seed)
    data = np.load(npz_path)

    required_keys = ["x", "mu0", "mu1"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in npz file.")

    # ---- Extract one replication ----
    X = data["x"][:, :, replication]          # shape: (n, p)
    mu0 = data["mu0"][:, replication]         # shape: (n,)
    mu1 = data["mu1"][:, replication]         # shape: (n,)

    n = X.shape[0]
    if not (0.0 <= rct_fraction <= 1.0):
        raise ValueError("rct_fraction must be between 0 and 1.")

    # ---- Random split into RCT vs OBS groups ----
    indices = rng.permutation(n)
    n_rct = int(round(rct_fraction * n))
    rct_idx = indices[:n_rct]
    obs_idx = indices[n_rct:]

    # ---- Construct propensity model for OBS subset ----
    X_obs = X[obs_idx].copy()

    if standardize_for_propensity:
        mean = X_obs.mean(axis=0, keepdims=True)
        std = X_obs.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        X_for_prop = (X_obs - mean) / std
    else:
        X_for_prop = X_obs

    # Random linear score to induce confounding
    w = rng.normal(size=X_for_prop.shape[1])
    w = w / (np.linalg.norm(w) + 1e-8)

    logits_obs = bias_strength * (X_for_prop @ w)

    # Center logits for numerical stability / balanced-ish treatment rates
    logits_obs = logits_obs - logits_obs.mean()

    p_obs = 1.0 / (1.0 + np.exp(-logits_obs))
    t_obs = rng.binomial(1, p_obs).astype(np.float32)

    # ---- RCT subset gets randomized treatment ----
    t_rct = rng.binomial(1, 0.5, size=len(rct_idx)).astype(np.float32)

    # ---- Regenerate outcomes from mu0 / mu1 ----
    # Add small noise so this behaves like a semi-synthetic observed outcome
    eps_rct = rng.normal(0.0, noise_std, size=len(rct_idx))
    eps_obs = rng.normal(0.0, noise_std, size=len(obs_idx))

    mu0_rct, mu1_rct = mu0[rct_idx], mu1[rct_idx]
    mu0_obs, mu1_obs = mu0[obs_idx], mu1[obs_idx]

    yf_rct = t_rct * mu1_rct + (1.0 - t_rct) * mu0_rct + eps_rct
    ycf_rct = (1.0 - t_rct) * mu1_rct + t_rct * mu0_rct + eps_rct

    yf_obs = t_obs * mu1_obs + (1.0 - t_obs) * mu0_obs + eps_obs
    ycf_obs = (1.0 - t_obs) * mu1_obs + t_obs * mu0_obs + eps_obs

    # ---- Build tensors ----
    def to_tensor_dataset(X_part, t_part, yf_part, ycf_part, mu0_part, mu1_part):
        return TensorDataset(
            torch.tensor(X_part, dtype=torch.float32),
            torch.tensor(t_part, dtype=torch.float32),
            torch.tensor(yf_part, dtype=torch.float32),
            torch.tensor(ycf_part, dtype=torch.float32),
            torch.tensor(mu0_part, dtype=torch.float32),
            torch.tensor(mu1_part, dtype=torch.float32),
        )

    rct_dataset = to_tensor_dataset(
        X[rct_idx], t_rct, yf_rct, ycf_rct, mu0_rct, mu1_rct
    )
    obs_dataset = to_tensor_dataset(
        X[obs_idx], t_obs, yf_obs, ycf_obs, mu0_obs, mu1_obs
    )

    if return_numpy:
        extra = {
            "rct_idx": rct_idx,
            "obs_idx": obs_idx,
            "p_obs": p_obs,
            "t_rct": t_rct,
            "t_obs": t_obs,
            "tau_rct": mu1_rct - mu0_rct,
            "tau_obs": mu1_obs - mu0_obs,
        }
        return rct_dataset, obs_dataset, extra

    return rct_dataset, obs_dataset
