# RPCE Architecture & Data Flow

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RPCE SYSTEM                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │   CONFIG     │──────│    DATA      │──────│   MODELS     │ │
│  │              │      │              │      │              │ │
│  │ • Seeds      │      │ • Loaders    │      │ • AutoEncoder│ │
│  │ • Settings   │      │ • Exploration│      │ • Losses     │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│         │                      │                      │         │
│         └──────────────────────┼──────────────────────┘         │
│                                │                                │
│                   ┌────────────▼────────────┐                   │
│                   │      TRAINING           │                   │
│                   │                         │                   │
│                   │  ┌────────────────┐    │                   │
│                   │  │   Stage 1      │    │                   │
│                   │  │ Observational  │    │                   │
│                   │  └────────┬───────┘    │                   │
│                   │           │             │                   │
│                   │  ┌────────▼───────┐    │                   │
│                   │  │   Stage 2      │    │                   │
│                   │  │     RCT        │    │                   │
│                   │  └────────────────┘    │                   │
│                   └─────────┬───────────────┘                   │
│                             │                                   │
│              ┌──────────────┴──────────────┐                    │
│              │                             │                    │
│    ┌─────────▼─────────┐      ┌───────────▼──────────┐        │
│    │    TRANSPORT      │      │    EVALUATION        │        │
│    │                   │      │                      │        │
│    │ • Sinkhorn       │      │ • Metrics           │        │
│    │ • Optimal        │      │ • Policy Value      │        │
│    │   Transport      │      │ • ATT/ATE           │        │
│    └───────────────────┘      └──────────────────────┘        │
│              │                             │                    │
│              └──────────────┬──────────────┘                    │
│                             │                                   │
│                    ┌────────▼────────┐                          │
│                    │     UTILS       │                          │
│                    │                 │                          │
│                    │ • Visualization │                          │
│                    │ • Data Utils    │                          │
│                    └─────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow: Two-Stage Training

```
STAGE 1: OBSERVATIONAL DATA
═══════════════════════════

Input: X_obs, T_obs, Y_obs (biased, confounded)

    X_obs ──┬──▶ Encoder ──▶ Z (latent representation)
            │                │
            │                ├──▶ Decoder ──▶ X_recon
            │                │    └─ Loss: Reconstruction
            │                │
            │                ├──▶ Propensity Head ──▶ T_pred
            │                │    └─ Loss: Propensity
            │                │
            │                └──▶ Pseudo-Outcome Heads
            │                     ├─▶ T0 Head ──▶ Y0_pseudo
            │                     └─▶ T1 Head ──▶ Y1_pseudo
            │                          └─ Loss: Pseudo-Outcome
            │
            └──▶ Total Loss = λ₁·L_recon + λ₂·L_prop + λ₃·L_pseudo


STAGE 2: RCT DATA
═════════════════

Input: X_rct, T_rct, Y_rct (unbiased, randomized)

    X_rct ──▶ Encoder (FROZEN) ──▶ Z_rct
                                    │
                                    ├──▶ G0 Head ──▶ Y0_rct
                                    │    (initialized from T0)
                                    │
                                    └──▶ G1 Head ──▶ Y1_rct
                                         (initialized from T1)
                                         
                                    └─ Loss: RCT Outcome

    Optional: Optimal Transport
    Z_obs ──▶ Sinkhorn ──▶ Z_transported ──▶ RCT Domain
              (use Z_rct as reference)
```

## 🔄 Prediction Pipeline

```
NEW DATA PREDICTION
═══════════════════

Input: X_new (observational data to predict on)
Reference: X_rct (RCT data for transport)

Step 1: Encode
    X_new ──▶ Encoder ──▶ Z_new
    X_rct ──▶ Encoder ──▶ Z_rct

Step 2: Transport (RPCE)
    Z_new ──▶ Optimal Transport ──▶ Z_transported
              (project to RCT domain using Z_rct)

Step 3: Predict Outcomes
    Z_transported ──┬──▶ G0 Head ──▶ Y0_pred
                    └──▶ G1 Head ──▶ Y1_pred

Step 4: Compute CATE
    CATE = Y1_pred - Y0_pred

Step 5: Compute Confidence
    Confidence = 1 / (1 + distance(Z_transported, Z_rct))

Output: (CATE, Confidence)
```

## 🎯 Model Components

```
AUTOENCODER ARCHITECTURE
════════════════════════

┌─────────────────────────────────────────────────────────┐
│                     INPUT: X (17 features)               │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │          ENCODER              │
         │                               │
         │  Linear(17, 8) + ReLU        │
         │  Linear(8, 4) + ReLU         │
         │  Linear(4, latent_dim)       │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │      LATENT: Z (4-dim)        │
         └───────────────┬───────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼─────┐                    ┌───▼──────┐
    │ DECODER  │                    │  HEADS   │
    │          │                    │          │
    │ Linear(4,│                    │ ▸ Prop   │
    │   4)+ReLU│                    │ ▸ T0     │
    │ Linear(4,│                    │ ▸ T1     │
    │   8)+ReLU│                    │ ▸ G0     │
    │ Linear(8,│                    │ ▸ G1     │
    │   17)    │                    │          │
    └────┬─────┘                    └────┬─────┘
         │                               │
    ┌────▼─────┐                    ┌────▼─────┐
    │ X_recon  │                    │Predictions│
    └──────────┘                    └──────────┘
```

## 🚀 Execution Flows

### Training Flow
```
1. Initialize
   └─▶ set_random_seed(42)
   └─▶ model = AutoEncoder(17, 8, 4)

2. Stage 1
   └─▶ train_stage1(model, obs_data)
       ├─▶ For each epoch:
       │   ├─▶ Forward pass
       │   ├─▶ Compute losses
       │   ├─▶ Backward pass
       │   └─▶ Update weights
       └─▶ Return trained model, history

3. Stage 2 Init
   └─▶ initialize_stage2_from_stage1(model)
       └─▶ Copy T0/T1 weights to G0/G1

4. Stage 2
   └─▶ train_stage2(model, rct_data)
       ├─▶ Freeze encoder/decoder
       ├─▶ For each epoch:
       │   ├─▶ Encode X_rct → Z_rct
       │   ├─▶ Predict Y0, Y1
       │   ├─▶ Compute RCT loss
       │   └─▶ Update G0/G1 only
       └─▶ Return trained model, history
```

### Evaluation Flow
```
1. Predict CATE
   └─▶ predict_cate_rpce_in_batches(model, X_eval, X_rct_ref)
       ├─▶ Encode X_eval → Z_eval
       ├─▶ Encode X_rct_ref → Z_rct
       ├─▶ Transport: Z_eval → Z_transported
       ├─▶ Predict: Y0, Y1 from Z_transported
       └─▶ Return CATE = Y1 - Y0, Confidence

2. Evaluate Policy
   └─▶ estimate_policy_value_from_rct(cate, T, Y)
       ├─▶ Define policy: π(x) = 1 if CATE > threshold
       ├─▶ Compute: V = P(π=1)·E[Y|T=1,π=1] + P(π=0)·E[Y|T=0,π=0]
       └─▶ Return policy_value, policy_risk

3. Evaluate ATT
   └─▶ estimate_att_from_predictions(cate, T)
       └─▶ Return mean(CATE[T==1])
```

## 🔧 Key Design Patterns

### 1. **Factory Pattern**
```python
# config.py
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
```

### 2. **Strategy Pattern**
```python
# transport/sinkhorn.py
def predict_cate_rpce(model, x_obs, x_rct, transport_method="balanced"):
    if transport_method == "balanced":
        z_transported = sinkhorn_projection_balanced(...)
    elif transport_method == "unbalanced":
        z_transported = sinkhorn_projection_unbalanced(...)
```

### 3. **Template Method Pattern**
```python
# training/stage1.py & stage2.py
def train_stageX(model, dataset, ...):
    # Template: setup → loop → cleanup
    optimizer = setup_optimizer(model)
    for epoch in range(num_epochs):
        loss = training_step(model, batch)
        optimizer.step()
    return model, history
```

### 4. **Dependency Injection**
```python
# All training/evaluation functions accept dependencies
def train_stage1(model, dataset, device=None):
    # Can inject custom device, dataset, etc.
```

## 📦 Import Graph

```
train.py
  ├─▶ config (set_random_seed, get_device)
  ├─▶ data (load_jobs_data)
  ├─▶ models (AutoEncoder)
  ├─▶ training (train_stage1, train_stage2)
  └─▶ evaluation (evaluate_jobs_policy_risk_and_att)

inference.py
  ├─▶ models (AutoEncoder)
  ├─▶ transport (predict_cate_rpce)
  └─▶ data (createJobsTensorDataset)

training/stage1.py
  ├─▶ models.losses (reconstruction_loss, propensity_loss, ...)
  └─▶ utils.data_utils (detect_binary_continuous_columns)

training/stage2.py
  ├─▶ models.losses (rct_outcome_loss)
  └─▶ transport.sinkhorn (sinkhorn_projection_balanced)

evaluation/evaluate.py
  ├─▶ evaluation.metrics (estimate_policy_value_from_rct, ...)
  └─▶ transport.sinkhorn (predict_cate_rpce)
```

---

This architecture ensures:
✅ Clear separation of concerns
✅ Easy to test individual components
✅ Simple to extend/modify
✅ Production-ready structure
