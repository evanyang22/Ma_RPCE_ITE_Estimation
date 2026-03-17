"""
Two-Stage Pretraining-Finetuning (TSPF) Framework Implementation
Based on:
1. "A Two-Stage Pretraining-Finetuning Framework for Treatment Effect Estimation 
   with Unmeasured Confounding" (KDD 2025)
2. "Proximity Matters: Local Proximity Preserved Balancing for Treatment Effect 
   Estimation" (arXiv 2407.01111)

This implementation includes:
- Model 1: Standard TSPF with representation learning
- Model 2: TSPF with Optimal Transport (OT) for treatment selection bias mitigation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from sklearn.decomposition import PCA


class TSPFDataset(Dataset):
    """Dataset wrapper for TSPF training"""
    
    def __init__(self, x: np.ndarray, t: np.ndarray, yf: np.ndarray, 
                 e: Optional[np.ndarray] = None):
        """
        Args:
            x: Covariates [n_samples, n_features] or [n_samples, n_features, n_realizations]
            t: Treatment indicators [n_samples] or [n_samples, n_realizations]
            yf: Factual outcomes [n_samples] or [n_samples, n_realizations]
            e: Experiment indicators (0=observational, 1=RCT) [n_samples, n_realizations]
        """
        # Handle 3D data by flattening realizations
        if len(x.shape) == 3:
            n_samples, n_features, n_realizations = x.shape
            x = x.transpose(0, 2, 1).reshape(-1, n_features)
            t = t.flatten()
            yf = yf.flatten()
            if e is not None:
                e = e.flatten()
        
        self.x = torch.FloatTensor(x)
        self.t = torch.FloatTensor(t).unsqueeze(-1)
        self.yf = torch.FloatTensor(yf).unsqueeze(-1)
        self.e = torch.FloatTensor(e).unsqueeze(-1) if e is not None else None
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.e is not None:
            return self.x[idx], self.t[idx], self.yf[idx], self.e[idx]
        return self.x[idx], self.t[idx], self.yf[idx]


class RepresentationModule(nn.Module):
    """Shared representation network φ"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x):
        return self.network(x)


class RepresentationAdapter(nn.Module):
    """Adapter module φ_U for fine-tuning stage"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ELU(),
            nn.BatchNorm1d(output_dim)
        )
        
    def forward(self, x):
        return self.adapter(x)


class PredictionHead(nn.Module):
    """Outcome prediction head g_t"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.network(x)


class ReconstructionModule(nn.Module):
    """Reconstruction module for regularization"""
    
    def __init__(self, repr_dim: int, output_dim: int):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(repr_dim, repr_dim),
            nn.ELU(),
            nn.Linear(repr_dim, output_dim)
        )
        
    def forward(self, repr):
        return self.decoder(repr)


class OptimalTransportLoss(nn.Module):
    """
    Fused Gromov-Wasserstein Optimal Transport Loss
    Based on the PCR paper (Proximity Matters)
    """
    
    def __init__(self, kappa: float = 0.5, max_iter: int = 50, reg: float = 0.01):
        """
        Args:
            kappa: Balance between Wasserstein (global) and Gromov (local) terms
            max_iter: Maximum iterations for Sinkhorn algorithm
            reg: Entropic regularization strength
        """
        super().__init__()
        self.kappa = kappa
        self.max_iter = max_iter
        self.reg = reg
        
    def compute_cost_matrix(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distance matrix"""
        # x1: [n1, d], x2: [n2, d]
        n1, n2 = x1.size(0), x2.size(0)
        
        # Compute squared Euclidean distances
        x1_norm = (x1 ** 2).sum(1).view(n1, 1)
        x2_norm = (x2 ** 2).sum(1).view(1, n2)
        dist = x1_norm + x2_norm - 2.0 * torch.mm(x1, x2.t())
        
        return torch.clamp(dist, min=0.0)
    
    def compute_gromov_term(self, x1: torch.Tensor, x2: torch.Tensor, 
                           pi: torch.Tensor) -> torch.Tensor:
        """Compute Gromov-Wasserstein term for local proximity preservation"""
        # Compute intra-distribution distance matrices
        D1 = self.compute_cost_matrix(x1, x1)  # [n1, n1]
        D2 = self.compute_cost_matrix(x2, x2)  # [n2, n2]
        
        # Compute the Gromov cost: sum_{i,j,k,l} |D1_{ik} - D2_{jl}|^2 * pi_{ij} * pi_{kl}
        # Simplified computation using matrix operations
        n1, n2 = pi.size()
        
        # Tensor contraction for Gromov-Wasserstein
        # Cost = ||D1||^2 * (pi @ pi.T @ 1) + ||D2||^2 * (pi.T @ pi @ 1)
        #        - 2 * trace(D1 @ pi @ D2 @ pi.T)
        
        term1 = torch.sum((D1 ** 2) * torch.mm(pi, torch.mm(pi.t(), 
                          torch.ones(n1, 1).to(pi.device))))
        term2 = torch.sum((D2 ** 2) * torch.mm(pi.t(), torch.mm(pi, 
                          torch.ones(n2, 1).to(pi.device))))
        term3 = 2 * torch.trace(torch.mm(D1, torch.mm(pi, torch.mm(D2, pi.t()))))
        
        gromov_cost = (term1 + term2 - term3) / (n1 * n2)
        
        return gromov_cost
    
    def sinkhorn(self, cost: torch.Tensor, n_iter: int = 50) -> torch.Tensor:
        """Sinkhorn algorithm for entropic-regularized optimal transport"""
        n, m = cost.size()
        
        # Initialize uniform distributions
        mu = torch.ones(n, 1).to(cost.device) / n
        nu = torch.ones(m, 1).to(cost.device) / m
        
        # Kernel matrix
        K = torch.exp(-cost / self.reg)
        
        # Sinkhorn iterations
        u = torch.ones(n, 1).to(cost.device)
        v = torch.ones(m, 1).to(cost.device)
        
        for _ in range(n_iter):
            u = mu / (K @ v + 1e-8)
            v = nu / (K.t() @ u + 1e-8)
        
        # Transport plan
        pi = u * K * v.t()
        
        return pi
    
    def forward(self, repr_t0: torch.Tensor, repr_t1: torch.Tensor) -> torch.Tensor:
        """
        Compute Fused Gromov-Wasserstein discrepancy
        
        Args:
            repr_t0: Representations for control group [n0, d]
            repr_t1: Representations for treatment group [n1, d]
            
        Returns:
            Fused GW discrepancy
        """
        # Compute Wasserstein cost (global alignment)
        wasserstein_cost = self.compute_cost_matrix(repr_t0, repr_t1)
        
        # Compute transport plan using Sinkhorn
        pi = self.sinkhorn(wasserstein_cost, self.max_iter)
        
        # Wasserstein discrepancy
        w_disc = torch.sum(pi * wasserstein_cost)
        
        # Gromov-Wasserstein discrepancy (local proximity preservation)
        if self.kappa < 1.0:
            gw_disc = self.compute_gromov_term(repr_t0, repr_t1, pi)
        else:
            gw_disc = 0.0
        
        # Fused discrepancy
        fused_disc = self.kappa * w_disc + (1 - self.kappa) * gw_disc
        
        return fused_disc


class InformativeSubspaceProjector:
    """
    Informative Subspace Projector (ISP) using PCA
    Reduces dimensionality to handle curse of dimensionality in OT computation
    """
    
    def __init__(self, ratio: float = 0.5):
        """
        Args:
            ratio: Dimensionality reduction ratio (0 < ratio <= 1)
        """
        self.ratio = ratio
        self.pca = None
        
    def fit(self, representations: torch.Tensor) -> None:
        """Fit PCA on representations"""
        repr_np = representations.detach().cpu().numpy()
        n_components = max(1, int(representations.size(1) * self.ratio))
        
        self.pca = PCA(n_components=n_components)
        self.pca.fit(repr_np)
        
    def transform(self, representations: torch.Tensor) -> torch.Tensor:
        """Project representations to informative subspace"""
        if self.pca is None:
            raise ValueError("Must fit ISP before transform")
        
        repr_np = representations.detach().cpu().numpy()
        projected = self.pca.transform(repr_np)
        
        return torch.FloatTensor(projected).to(representations.device)


class TSPFModel(nn.Module):
    """
    Standard TSPF Model (Model 1)
    Two-stage pretraining-finetuning for treatment effect estimation
    """
    
    def __init__(self, input_dim: int, repr_dims: list = [64, 32], 
                 adapter_dim: int = 16, stage: int = 1):
        """
        Args:
            input_dim: Input feature dimension
            repr_dims: Hidden dimensions for representation network
            adapter_dim: Adapter output dimension (for stage 2)
            stage: Training stage (1 or 2)
        """
        super().__init__()
        
        self.stage = stage
        self.input_dim = input_dim
        
        # Representation module (shared)
        self.phi = RepresentationModule(input_dim, repr_dims)
        repr_dim = self.phi.output_dim
        
        if stage == 1:
            # Stage 1: Pretrain on observational data
            self.g0 = PredictionHead(repr_dim)
            self.g1 = PredictionHead(repr_dim)
            self.reconstruction = ReconstructionModule(repr_dim, input_dim)
            
        else:
            # Stage 2: Fine-tune on RCT data
            self.phi_u = RepresentationAdapter(repr_dim, adapter_dim)
            self.g0 = PredictionHead(repr_dim + adapter_dim)
            self.g1 = PredictionHead(repr_dim + adapter_dim)
            
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass
        
        Args:
            x: Input covariates [batch_size, input_dim]
            t: Treatment indicators [batch_size, 1]
            
        Returns:
            predictions: Predicted outcomes [batch_size, 1]
            outputs: Dictionary of intermediate outputs
        """
        # Shared representation
        phi_x = self.phi(x)
        
        outputs = {'phi': phi_x}
        
        if self.stage == 1:
            # Stage 1: Separate heads for each treatment
            y0_pred = self.g0(phi_x)
            y1_pred = self.g1(phi_x)
            
            # Select prediction based on treatment
            predictions = t * y1_pred + (1 - t) * y0_pred
            
            # Reconstruction for regularization
            x_recon = self.reconstruction(phi_x)
            
            outputs.update({
                'y0': y0_pred,
                'y1': y1_pred,
                'x_recon': x_recon
            })
            
        else:
            # Stage 2: Adapter + augmented representation
            phi_u_x = self.phi_u(phi_x)
            phi_aug = torch.cat([phi_x, phi_u_x], dim=1)
            
            y0_pred = self.g0(phi_aug)
            y1_pred = self.g1(phi_aug)
            
            predictions = t * y1_pred + (1 - t) * y0_pred
            
            outputs.update({
                'phi_u': phi_u_x,
                'phi_aug': phi_aug,
                'y0': y0_pred,
                'y1': y1_pred
            })
        
        return predictions, outputs
    
    def predict_ite(self, x: torch.Tensor) -> torch.Tensor:
        """Predict Individual Treatment Effect"""
        with torch.no_grad():
            phi_x = self.phi(x)
            
            if self.stage == 1:
                y0_pred = self.g0(phi_x)
                y1_pred = self.g1(phi_x)
            else:
                phi_u_x = self.phi_u(phi_x)
                phi_aug = torch.cat([phi_x, phi_u_x], dim=1)
                y0_pred = self.g0(phi_aug)
                y1_pred = self.g1(phi_aug)
            
            ite = y1_pred - y0_pred
            
        return ite


class TSPFModelOT(nn.Module):
    """
    TSPF Model with Optimal Transport (Model 2)
    Incorporates optimal transport for treatment selection bias mitigation
    Based on the PCR (Proximity Matters) paper
    """
    
    def __init__(self, input_dim: int, repr_dims: list = [64, 32],
                 adapter_dim: int = 16, stage: int = 1, 
                 ot_kappa: float = 0.5, isp_ratio: float = 0.7):
        """
        Args:
            input_dim: Input feature dimension
            repr_dims: Hidden dimensions for representation network
            adapter_dim: Adapter output dimension (for stage 2)
            stage: Training stage (1 or 2)
            ot_kappa: Balance parameter for OT (0=pure Gromov, 1=pure Wasserstein)
            isp_ratio: Dimensionality reduction ratio for ISP
        """
        super().__init__()
        
        self.stage = stage
        self.input_dim = input_dim
        self.ot_kappa = ot_kappa
        
        # Representation module
        self.phi = RepresentationModule(input_dim, repr_dims)
        repr_dim = self.phi.output_dim
        
        # Optimal transport loss
        self.ot_loss = OptimalTransportLoss(kappa=ot_kappa)
        
        # Informative Subspace Projector
        self.isp = InformativeSubspaceProjector(ratio=isp_ratio)
        self.isp_fitted = False
        
        if stage == 1:
            # Stage 1: Pretrain on observational data
            self.g0 = PredictionHead(repr_dim)
            self.g1 = PredictionHead(repr_dim)
            self.reconstruction = ReconstructionModule(repr_dim, input_dim)
            
        else:
            # Stage 2: Fine-tune on RCT data
            self.phi_u = RepresentationAdapter(repr_dim, adapter_dim)
            self.g0 = PredictionHead(repr_dim + adapter_dim)
            self.g1 = PredictionHead(repr_dim + adapter_dim)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with OT-based alignment"""
        # Shared representation
        phi_x = self.phi(x)
        
        outputs = {'phi': phi_x}
        
        if self.stage == 1:
            y0_pred = self.g0(phi_x)
            y1_pred = self.g1(phi_x)
            predictions = t * y1_pred + (1 - t) * y0_pred
            x_recon = self.reconstruction(phi_x)
            
            outputs.update({
                'y0': y0_pred,
                'y1': y1_pred,
                'x_recon': x_recon
            })
            
        else:
            phi_u_x = self.phi_u(phi_x)
            phi_aug = torch.cat([phi_x, phi_u_x], dim=1)
            
            y0_pred = self.g0(phi_aug)
            y1_pred = self.g1(phi_aug)
            predictions = t * y1_pred + (1 - t) * y0_pred
            
            outputs.update({
                'phi_u': phi_u_x,
                'phi_aug': phi_aug,
                'y0': y0_pred,
                'y1': y1_pred
            })
        
        return predictions, outputs
    
    def compute_ot_discrepancy(self, representations: torch.Tensor, 
                               treatments: torch.Tensor) -> torch.Tensor:
        """
        Compute optimal transport discrepancy between treatment groups
        Uses ISP for dimensionality reduction
        """
        # Split by treatment
        t0_mask = treatments.squeeze() == 0
        t1_mask = treatments.squeeze() == 1
        
        repr_t0 = representations[t0_mask]
        repr_t1 = representations[t1_mask]
        
        # Need sufficient samples in both groups
        if len(repr_t0) < 2 or len(repr_t1) < 2:
            return torch.tensor(0.0).to(representations.device)
        
        # Apply ISP for dimensionality reduction
        if not self.isp_fitted:
            self.isp.fit(representations)
            self.isp_fitted = True
        
        repr_t0_proj = self.isp.transform(repr_t0)
        repr_t1_proj = self.isp.transform(repr_t1)
        
        # Compute OT discrepancy
        ot_disc = self.ot_loss(repr_t0_proj, repr_t1_proj)
        
        return ot_disc
    
    def predict_ite(self, x: torch.Tensor) -> torch.Tensor:
        """Predict Individual Treatment Effect"""
        with torch.no_grad():
            phi_x = self.phi(x)
            
            if self.stage == 1:
                y0_pred = self.g0(phi_x)
                y1_pred = self.g1(phi_x)
            else:
                phi_u_x = self.phi_u(phi_x)
                phi_aug = torch.cat([phi_x, phi_u_x], dim=1)
                y0_pred = self.g0(phi_aug)
                y1_pred = self.g1(phi_aug)
            
            ite = y1_pred - y0_pred
            
        return ite


def train_stage1(model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, n_epochs: int = 100,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 lambda_rec: float = 0.1, lambda_ot: float = 1.0,
                 use_ot: bool = False, device: str = 'cuda') -> Dict:
    """
    Train Stage 1: Pretraining on observational data
    
    Args:
        model: TSPF or TSPF-OT model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        lambda_rec: Weight for reconstruction loss
        lambda_ot: Weight for optimal transport loss
        use_ot: Whether to use optimal transport
        device: Device for training
        
    Returns:
        Dictionary of training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           patience=10, factor=0.5)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_factual': [],
        'val_factual': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        train_factual_losses = []
        
        for batch in train_loader:
            if len(batch) == 4:
                x, t, yf, e = batch
                e = e.to(device)
                # Filter for observational data only in stage 1
                obs_mask = e.squeeze() == 0
                if obs_mask.sum() == 0:
                    continue
                x = x[obs_mask]
                t = t[obs_mask]
                yf = yf[obs_mask]
            else:
                x, t, yf = batch
            
            x, t, yf = x.to(device), t.to(device), yf.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred, outputs = model(x, t)
            
            # Factual outcome loss
            factual_loss = F.mse_loss(y_pred, yf)
            
            # Reconstruction loss
            if 'x_recon' in outputs:
                recon_loss = F.mse_loss(outputs['x_recon'], x)
            else:
                recon_loss = 0.0
            
            # Total loss for stage 1
            loss = factual_loss + lambda_rec * recon_loss
            
            # Add OT loss if using TSPF-OT
            if use_ot and hasattr(model, 'compute_ot_discrepancy'):
                ot_disc = model.compute_ot_discrepancy(outputs['phi'], t)
                loss = loss + lambda_ot * ot_disc
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_factual_losses.append(factual_loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_factual_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    x, t, yf, e = batch
                    e = e.to(device)
                    obs_mask = e.squeeze() == 0
                    if obs_mask.sum() == 0:
                        continue
                    x = x[obs_mask]
                    t = t[obs_mask]
                    yf = yf[obs_mask]
                else:
                    x, t, yf = batch
                
                x, t, yf = x.to(device), t.to(device), yf.to(device)
                
                y_pred, outputs = model(x, t)
                factual_loss = F.mse_loss(y_pred, yf)
                
                if 'x_recon' in outputs:
                    recon_loss = F.mse_loss(outputs['x_recon'], x)
                else:
                    recon_loss = 0.0
                
                loss = factual_loss + lambda_rec * recon_loss
                
                if use_ot and hasattr(model, 'compute_ot_discrepancy'):
                    ot_disc = model.compute_ot_discrepancy(outputs['phi'], t)
                    loss = loss + lambda_ot * ot_disc
                
                val_losses.append(loss.item())
                val_factual_losses.append(factual_loss.item())
        
        # Record history
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        avg_train_factual = np.mean(train_factual_losses)
        avg_val_factual = np.mean(val_factual_losses) if val_factual_losses else float('inf')
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_factual'].append(avg_train_factual)
        history['val_factual'].append(avg_val_factual)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Train Factual: {avg_train_factual:.4f}, Val Factual: {avg_val_factual:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history


def initialize_stage2_from_stage1(stage1_model: nn.Module, stage2_model: nn.Module):
    """
    Initialize stage 2 model parameters from stage 1 model
    Uses partial initialization strategy from TSPF paper
    """
    # Copy representation module weights
    stage2_model.phi.load_state_dict(stage1_model.phi.state_dict())
    
    # Freeze the pretrained representation
    for param in stage2_model.phi.parameters():
        param.requires_grad = False
    
    # Initialize prediction heads to match stage 1 output initially
    # This ensures continuity between stages
    with torch.no_grad():
        # Copy stage 1 prediction head weights to stage 2
        # Stage 2 has larger input due to adapter, so we only copy the first part
        repr_dim = stage1_model.phi.output_dim
        
        # For g0
        stage2_g0_weight = stage2_model.g0.network[0].weight.data
        stage2_g0_bias = stage2_model.g0.network[0].bias.data
        stage1_g0_weight = stage1_model.g0.network[0].weight.data
        stage1_g0_bias = stage1_model.g0.network[0].bias.data
        
        stage2_g0_weight[:, :repr_dim] = stage1_g0_weight
        stage2_g0_bias[:] = stage1_g0_bias
        
        # For g1
        stage2_g1_weight = stage2_model.g1.network[0].weight.data
        stage2_g1_bias = stage2_model.g1.network[0].bias.data
        stage1_g1_weight = stage1_model.g1.network[0].weight.data
        stage1_g1_bias = stage1_model.g1.network[0].bias.data
        
        stage2_g1_weight[:, :repr_dim] = stage1_g1_weight
        stage2_g1_bias[:] = stage1_g1_bias
    
    print("Stage 2 model initialized from Stage 1")


def train_stage2(model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, n_epochs: int = 100,
                 lr: float = 5e-4, weight_decay: float = 1e-4,
                 lambda_shift: float = 0.1, lambda_mi: float = 0.01,
                 device: str = 'cuda') -> Dict:
    """
    Train Stage 2: Fine-tuning on RCT data
    
    Args:
        model: Stage 2 TSPF model
        train_loader: Training data loader (RCT data)
        val_loader: Validation data loader
        n_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        lambda_shift: Weight for distribution shift regularization
        lambda_mi: Weight for mutual information regularization
        device: Device for training
        
    Returns:
        Dictionary of training history
    """
    model = model.to(device)
    
    # Only train adapter and prediction heads (phi is frozen)
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=10, factor=0.5)
    
    # Store initial adapter parameters for shift regularization
    initial_params = {}
    for name, param in model.named_parameters():
        if 'phi_u' in name or 'g0' in name or 'g1' in name:
            if param.requires_grad:
                initial_params[name] = param.clone().detach()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_factual': [],
        'val_factual': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        train_factual_losses = []
        
        for batch in train_loader:
            if len(batch) == 4:
                x, t, yf, e = batch
                e = e.to(device)
                # Filter for RCT data only in stage 2
                rct_mask = e.squeeze() == 1
                if rct_mask.sum() == 0:
                    continue
                x = x[rct_mask]
                t = t[rct_mask]
                yf = yf[rct_mask]
            else:
                x, t, yf = batch
            
            x, t, yf = x.to(device), t.to(device), yf.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred, outputs = model(x, t)
            
            # Factual outcome loss
            factual_loss = F.mse_loss(y_pred, yf)
            
            # Distribution shift regularization
            # Penalize deviation from initial parameters
            shift_loss = 0.0
            for name, param in model.named_parameters():
                if name in initial_params and param.requires_grad:
                    shift_loss += torch.sum((param - initial_params[name]) ** 2)
            
            # Mutual information regularization (encourage independence)
            # Use covariance as proxy for MI
            if 'phi' in outputs and 'phi_u' in outputs:
                phi = outputs['phi']
                phi_u = outputs['phi_u']
                
                # Center the representations
                phi_centered = phi - phi.mean(dim=0, keepdim=True)
                phi_u_centered = phi_u - phi_u.mean(dim=0, keepdim=True)
                
                # Compute covariance
                cov = torch.mm(phi_centered.t(), phi_u_centered) / (phi.size(0) - 1)
                mi_loss = torch.sum(cov ** 2)
            else:
                mi_loss = 0.0
            
            # Total loss
            loss = factual_loss + lambda_shift * shift_loss + lambda_mi * mi_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_factual_losses.append(factual_loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_factual_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    x, t, yf, e = batch
                    e = e.to(device)
                    rct_mask = e.squeeze() == 1
                    if rct_mask.sum() == 0:
                        continue
                    x = x[rct_mask]
                    t = t[rct_mask]
                    yf = yf[rct_mask]
                else:
                    x, t, yf = batch
                
                x, t, yf = x.to(device), t.to(device), yf.to(device)
                
                y_pred, outputs = model(x, t)
                factual_loss = F.mse_loss(y_pred, yf)
                
                val_losses.append(factual_loss.item())
                val_factual_losses.append(factual_loss.item())
        
        # Record history
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        avg_train_factual = np.mean(train_factual_losses)
        avg_val_factual = np.mean(val_factual_losses) if val_factual_losses else float('inf')
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_factual'].append(avg_train_factual)
        history['val_factual'].append(avg_val_factual)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history


def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                   device: str = 'cuda') -> Dict:
    """
    Evaluate model on test set
    Compute PEHE and ATE metrics
    """
    model = model.to(device)
    model.eval()
    
    all_ite_pred = []
    all_ite_true = []
    all_y0_true = []
    all_y1_true = []
    all_t = []
    all_yf = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                x, t, yf, e = batch
            else:
                x, t, yf = batch
            
            x = x.to(device)
            
            # Predict ITE
            ite_pred = model.predict_ite(x)
            
            all_ite_pred.append(ite_pred.cpu().numpy())
            all_t.append(t.numpy())
            all_yf.append(yf.numpy())
    
    all_ite_pred = np.concatenate(all_ite_pred)
    all_t = np.concatenate(all_t)
    all_yf = np.concatenate(all_yf)
    
    # Note: For real evaluation, we would need ground truth ITE
    # Here we return the predictions for demonstration
    results = {
        'ite_pred': all_ite_pred,
        'treatments': all_t,
        'outcomes': all_yf,
        'ate_pred': np.mean(all_ite_pred)
    }
    
    return results


if __name__ == "__main__":
    print("TSPF Implementation loaded successfully!")
    print("\nAvailable models:")
    print("1. TSPFModel - Standard TSPF framework")
    print("2. TSPFModelOT - TSPF with Optimal Transport for bias mitigation")
