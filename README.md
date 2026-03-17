# TSPF Framework Implementation for Treatment Effect Estimation

This implementation provides two models based on the Two-Stage Pretraining-Finetuning (TSPF) framework for estimating individual treatment effects from observational and RCT data.

## Overview

### Papers Implemented

1. **"A Two-Stage Pretraining-Finetuning Framework for Treatment Effect Estimation with Unmeasured Confounding"** (KDD 2025)
   - Proposes a two-stage approach: pretrain on large observational data, fine-tune on small RCT data
   - Uses partial parameter initialization to prevent overfitting
   - Includes representation adapter module for calibrating biases

2. **"Proximity Matters: Local Proximity Preserved Balancing for Treatment Effect Estimation"** (arXiv 2407.01111)
   - Introduces optimal transport with local proximity preservation
   - Uses Fused Gromov-Wasserstein distance to balance treatment groups
   - Includes Informative Subspace Projector (ISP) to handle curse of dimensionality

## Models

### Model 1: Standard TSPF
A baseline implementation of the TSPF framework with:
- Shared representation network (φ)
- Separate prediction heads for control (g₀) and treatment (g₁) groups
- Reconstruction module for regularization
- Two-stage training: Stage 1 (observational) → Stage 2 (RCT)

### Model 2: TSPF with Optimal Transport (TSPF-OT)
Enhanced TSPF with treatment selection bias mitigation:
- All features of Model 1
- **Fused Gromov-Wasserstein optimal transport** for representation balancing
  - κ parameter balances global (Wasserstein) vs local (Gromov) alignment
  - Sinkhorn algorithm for efficient computation
- **Informative Subspace Projector (ISP)** using PCA
  - Reduces dimensionality to handle curse of dimensionality
  - Preserves most informative directions

## Architecture

### Stage 1: Pretraining on Observational Data
```
Input (X) → Representation Network (φ) → Representation (R)
                                            ↓
                                    ┌───────┴────────┐
                                    ↓                ↓
                            Control Head (g₀)  Treatment Head (g₁)
                                    ↓                ↓
                                   Y₀               Y₁
```

**Loss Components:**
- Factual outcome loss: MSE between predicted and actual outcomes
- Reconstruction loss: MSE for input reconstruction from representation
- **OT loss (Model 2 only)**: Fused GW discrepancy between treatment groups

### Stage 2: Fine-tuning on RCT Data
```
Input (X) → Frozen φ → Base Representation (R_base)
                              ↓
                      Adapter (φ_U) → Augmented Rep (R_aug)
                              ↓
                    Concatenate [R_base, R_aug]
                              ↓
                      ┌───────┴────────┐
                      ↓                ↓
              Control Head (g₀)  Treatment Head (g₁)
                      ↓                ↓
                     Y₀               Y₁
```

**Key Features:**
- Representation network (φ) is frozen
- Adapter module (φ_U) learns to correct biases
- Partial initialization strategy prevents overfitting
- Regularization to prevent deviation from Stage 1

## Implementation Details

### Key Components

1. **RepresentationModule**: Multi-layer perceptron with ELU activation and batch normalization
2. **RepresentationAdapter**: Learns bias correction in Stage 2
3. **PredictionHead**: Maps representations to outcome predictions
4. **OptimalTransportLoss**: Computes Fused Gromov-Wasserstein discrepancy
5. **InformativeSubspaceProjector**: PCA-based dimensionality reduction

### Optimal Transport Details

The Fused Gromov-Wasserstein loss combines:

**Wasserstein Term (Global Alignment):**
```
W(R₀, R₁) = min_π Σᵢⱼ ||R₀ᵢ - R₁ⱼ||² πᵢⱼ
```

**Gromov-Wasserstein Term (Local Proximity):**
```
G(R₀, R₁) = min_π Σᵢⱼₖₗ ||D₀ᵢₖ - D₁ⱼₗ||² πᵢⱼπₖₗ
```

**Fused Loss:**
```
F(R₀, R₁) = κ·W(R₀, R₁) + (1-κ)·G(R₀, R₁)
```

Where:
- R₀, R₁ are representations for control and treatment groups
- π is the transport plan (optimized via Sinkhorn algorithm)
- D₀, D₁ are intra-group distance matrices
- κ balances global vs local alignment

## Usage

### Basic Training

```python
from tspf_implementation import TSPFModel, TSPFModelOT, train_stage1, train_stage2

# Model 1: Standard TSPF
model1_stage1 = TSPFModel(input_dim=17, repr_dims=[64, 32], stage=1)
history1 = train_stage1(model1_stage1, train_loader, val_loader, use_ot=False)

model1_stage2 = TSPFModel(input_dim=17, repr_dims=[64, 32], adapter_dim=16, stage=2)
initialize_stage2_from_stage1(model1_stage1, model1_stage2)
history2 = train_stage2(model1_stage2, train_loader, val_loader)

# Model 2: TSPF-OT
model2_stage1 = TSPFModelOT(input_dim=17, repr_dims=[64, 32], stage=1, 
                            ot_kappa=0.5, isp_ratio=0.7)
history1 = train_stage1(model2_stage1, train_loader, val_loader, use_ot=True, lambda_ot=1.0)

model2_stage2 = TSPFModelOT(input_dim=17, repr_dims=[64, 32], adapter_dim=16, stage=2)
initialize_stage2_from_stage1(model2_stage1, model2_stage2)
history2 = train_stage2(model2_stage2, train_loader, val_loader)
```

### Prediction

```python
# Predict Individual Treatment Effects (ITE)
ite = model.predict_ite(x_test)

# Predict outcomes for specific treatment
y_pred, outputs = model(x_test, t_test)

# Average Treatment Effect (ATE)
ate = ite.mean()
```

## Hyperparameters

### Stage 1 (Pretraining)
- `repr_dims`: [64, 32] - Hidden dimensions for representation network
- `lr`: 1e-3 - Learning rate
- `weight_decay`: 1e-4 - L2 regularization
- `lambda_rec`: 0.1 - Weight for reconstruction loss
- `lambda_ot`: 1.0 - Weight for OT loss (Model 2 only)
- `n_epochs`: 150 - Maximum epochs
- `batch_size`: 256

### Stage 2 (Fine-tuning)
- `adapter_dim`: 16 - Dimension of adapter module
- `lr`: 5e-4 - Learning rate (lower than Stage 1)
- `lambda_shift`: 0.1 - Weight for parameter shift regularization
- `lambda_mi`: 0.01 - Weight for mutual information regularization
- `n_epochs`: 100

### Optimal Transport (Model 2)
- `ot_kappa`: 0.5 - Balance between Wasserstein and Gromov terms
  - 0.0 = pure Gromov (local proximity only)
  - 1.0 = pure Wasserstein (global alignment only)
  - 0.3-0.7 = good balance (recommended)
- `isp_ratio`: 0.7 - Dimensionality reduction ratio
  - Higher = more dimensions preserved, but slower and more prone to curse of dimensionality
  - Lower = faster computation, but may lose information
- `reg`: 0.01 - Entropic regularization for Sinkhorn
- `max_iter`: 50 - Maximum Sinkhorn iterations

## Data Format

The Jobs dataset should be in `.npz` format with the following arrays:

- `x`: Covariates [n_samples, n_features, n_realizations]
- `t`: Treatment indicators [n_samples, n_realizations] (0=control, 1=treatment)
- `yf`: Factual outcomes [n_samples, n_realizations]
- `e`: Experiment indicators [n_samples, n_realizations] (0=observational, 1=RCT)

For the provided Jobs dataset:
- n_samples: 2570 (train), 642 (test)
- n_features: 17
- n_realizations: 10

## Training Procedure

1. **Data Preparation**
   - Split observational and RCT data based on experiment indicator
   - Create train/validation split (90/10)
   
2. **Stage 1: Pretraining**
   - Train only on observational data (e=0)
   - Learn shared representation and outcome predictors
   - Model 2: Apply OT regularization to balance treatment groups
   
3. **Stage 2: Fine-tuning**
   - Initialize from Stage 1 with partial initialization
   - Freeze representation network (φ)
   - Train adapter and prediction heads on RCT data (e=1)
   - Regularize to prevent deviation from Stage 1

## Key Differences Between Models

| Feature | Model 1 (Standard TSPF) | Model 2 (TSPF-OT) |
|---------|------------------------|-------------------|
| Treatment Balance | None | Fused Gromov-Wasserstein OT |
| Local Proximity | Not preserved | Preserved via Gromov term |
| Dimensionality | Full | Reduced via ISP |
| Stage 1 Loss | Factual + Reconstruction | Factual + Reconstruction + OT |
| Best For | Small selection bias | Large selection bias |

## Expected Results

Model 2 (TSPF-OT) should generally outperform Model 1 when:
- Treatment selection bias is significant
- Treatment and control groups have different covariate distributions
- Local proximity matters (similar units should have similar outcomes)

Both models should significantly outperform single-stage methods by:
- Leveraging large observational data for representation learning
- Using RCT data to correct for unmeasured confounding
- Preventing overfitting on small RCT samples

## Files

- `tspf_implementation.py`: Core model implementations
- `train_models.py`: Training script for Jobs dataset
- `README.md`: This file

## Requirements

```
torch>=1.9.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{zhou2025tspf,
  title={A Two-Stage Pretraining-Finetuning Framework for Treatment Effect Estimation with Unmeasured Confounding},
  author={Zhou, Chuan and others},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025}
}

@article{wang2024proximity,
  title={Proximity Matters: Local Proximity Preserved Balancing for Treatment Effect Estimation},
  author={Wang, Hao and others},
  journal={arXiv preprint arXiv:2407.01111},
  year={2024}
}
```

## Notes

### Why Two Stages?
- **Stage 1**: Learn robust representations from abundant observational data
- **Stage 2**: Correct for unmeasured confounding using unbiased RCT data
- This leverages the strengths of both data types

### Why Optimal Transport?
- Traditional methods use global metrics (MMD, Wasserstein) that ignore local structure
- OT with Gromov term preserves local proximity: similar units are matched together
- This leads to better representation alignment and more accurate treatment effect estimates

### Why ISP?
- High-dimensional representations suffer from curse of dimensionality
- OT computation becomes unreliable with limited samples in high dimensions
- ISP reduces dimensions while preserving most informative directions via PCA
- Balances computational efficiency with information preservation

## Troubleshooting

**Q: Training is slow**
- Reduce batch size
- Reduce ISP ratio (for Model 2)
- Use GPU if available

**Q: Model overfits in Stage 2**
- Increase lambda_shift (prevents deviation from Stage 1)
- Reduce learning rate
- Add more dropout

**Q: OT loss is NaN**
- Reduce learning rate
- Increase entropic regularization (reg parameter)
- Check for numerical instabilities in representations

**Q: Poor performance on test set**
- Ensure sufficient RCT data in Stage 2
- Try different obs_ratio values
- Tune hyperparameters (especially ot_kappa and isp_ratio)
