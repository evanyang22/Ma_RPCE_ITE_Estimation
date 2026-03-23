#AutoEncoder: an Encoder paired with a Decoder to ensure x => z => x_recon is close
#Encoder: multi-layer feedforward network 
#Turns covariates X => representation Z
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,latent_dim):#,binary_index,cont_index
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Hidden layer 1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), #Hidden Layer 2
            nn.ReLU(),
            nn.Linear(hidden_dim//2,latent_dim)
        )
        '''
        Stage 1
        Contains 3 output heads
        1. Reconstruction head using a decoder
        2. 2 Pseudooutcome heads trained on OBS data
        3. Propenssity head
        '''
        #Output heads
        #Reconstruction head via decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2), # Hidden layer 1
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim), #Hidden Layer 2
            nn.ReLU(),
            nn.Linear(hidden_dim,input_dim)
        )


        #prospensity_head
        self.propensity_head= nn.Linear(latent_dim,1)

        #pseudo-outcome head, T=0, predicts a biased outcome estimation Y0
        self.t0_head =nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
        #pseudo-outcome head, T=1, predicts a biased outcome estimation Y1
        self.t1_head=nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
        '''
        Stage 2
        1. Unconfounded outcome heads trained on RCT data
        '''
        self.g0_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.g1_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


        
    def forward(self,x):
        z = self.encoder(x) #representation Z

        
        t_logit = self.propensity_head(z)
        x_recon = self.decoder(z)

        #pseudooutcomes
        y0_pseudo=self.t0_head(z)
        y1_pseudo=self.t1_head(z)

        # Stage 2 unconfounded outcomes
        y0_rct = self.g0_head(z)
        y1_rct = self.g1_head(z)
        return {
            "x_recon" :x_recon,
            "t_logit" :t_logit,
            "y0_pseudo":y0_pseudo,
            "y1_pseudo":y1_pseudo,
            "y0_rct": y0_rct,
            "y1_rct":y1_rct
        }
