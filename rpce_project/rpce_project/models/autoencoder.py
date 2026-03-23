"""
AutoEncoder model architecture for RPCE.

The AutoEncoder consists of:
1. Encoder: Maps covariates X to latent representation Z
2. Decoder: Reconstructs X from Z
3. Multiple prediction heads:
   - Propensity head (treatment prediction)
   - Pseudo-outcome heads (T0, T1) for observational data
   - Unconfounded outcome heads (G0, G1) for RCT data
"""
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    AutoEncoder with multiple prediction heads for causal inference.
    
    Architecture:
        - Encoder: X -> hidden -> hidden/2 -> latent_dim
        - Decoder: latent_dim -> hidden/2 -> hidden -> X
        - Propensity head: latent_dim -> 1 (treatment probability)
        - Pseudo-outcome heads: latent_dim -> hidden -> 1 (biased outcomes)
        - RCT outcome heads: latent_dim -> hidden -> 1 (unconfounded outcomes)
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Initialize AutoEncoder.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Hidden layer dimension
            latent_dim (int): Latent representation dimension
        """
        super().__init__()
        
        # Encoder: X -> Z
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Decoder: Z -> X_reconstructed
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Propensity head: predicts treatment probability
        self.propensity_head = nn.Linear(latent_dim, 1)
        
        # Stage 1: Pseudo-outcome heads (trained on observational data)
        # T=0 head: predicts biased outcome under control
        self.t0_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # T=1 head: predicts biased outcome under treatment
        self.t1_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Stage 2: Unconfounded outcome heads (trained on RCT data)
        # G0 head: predicts unconfounded outcome under control
        self.g0_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # G1 head: predicts unconfounded outcome under treatment
        self.g1_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
        
        Returns:
            dict: Dictionary containing all model outputs:
                - x_recon: Reconstructed input
                - t_logit: Treatment propensity logits
                - y0_pseudo: Pseudo-outcome for T=0 (observational)
                - y1_pseudo: Pseudo-outcome for T=1 (observational)
                - y0_rct: Unconfounded outcome for T=0 (RCT)
                - y1_rct: Unconfounded outcome for T=1 (RCT)
        """
        # Encode to latent representation
        z = self.encoder(x)
        
        # Reconstruction
        x_recon = self.decoder(z)
        
        # Treatment propensity
        t_logit = self.propensity_head(z)
        
        # Stage 1: Pseudo-outcomes (observational)
        y0_pseudo = self.t0_head(z)
        y1_pseudo = self.t1_head(z)
        
        # Stage 2: Unconfounded outcomes (RCT)
        y0_rct = self.g0_head(z)
        y1_rct = self.g1_head(z)
        
        return {
            "x_recon": x_recon,
            "t_logit": t_logit,
            "y0_pseudo": y0_pseudo,
            "y1_pseudo": y1_pseudo,
            "y0_rct": y0_rct,
            "y1_rct": y1_rct,
            "z": z  # Include latent representation for transport
        }
    
    def encode(self, x):
        """Get latent representation."""
        return self.encoder(x)
    
    def decode(self, z):
        """Reconstruct from latent representation."""
        return self.decoder(z)
