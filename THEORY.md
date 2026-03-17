# Theoretical Background: TSPF and Optimal Transport for Treatment Effect Estimation

## Problem Setting

### Causal Inference Framework

We work within the **potential outcomes framework** (Rubin, 1974):

- Each unit i has two potential outcomes:
  - Y₁ᵢ: outcome if treated (T=1)
  - Y₀ᵢ: outcome if not treated (T=0)
  
- Only one outcome is observed (fundamental problem of causal inference):
  - Yᵢ = TᵢY₁ᵢ + (1-Tᵢ)Y₀ᵢ

### Causal Estimands

**Individual Treatment Effect (ITE):**
```
τ(xᵢ) = Y₁ᵢ - Y₀ᵢ
```

**Conditional Average Treatment Effect (CATE):**
```
τ(x) = E[Y₁ - Y₀ | X = x]
```

**Average Treatment Effect (ATE):**
```
τ = E[Y₁ - Y₀]
```

## Challenges

### 1. Missing Counterfactuals
We never observe both Y₁ᵢ and Y₀ᵢ for the same unit. This makes direct ITE estimation impossible.

### 2. Treatment Selection Bias
In observational data, treatment assignment is not random:
- P(X|T=1) ≠ P(X|T=0)
- Confounders affect both treatment and outcome
- Models trained on one group may not generalize to the other

### 3. Unmeasured Confounding
Even with all observed covariates X, there may exist unobserved confounders U:
- Ignorability assumption may be violated: (Y₀, Y₁) ⊥̸ T | X
- This leads to biased estimates

## TSPF Framework Solution

### Key Idea
Combine two complementary data sources:
1. **Observational (OBS) data**: Large sample size, but has selection bias
2. **Randomized Controlled Trial (RCT) data**: Unbiased, but small sample size

### Two-Stage Approach

**Stage 1: Pretraining on OBS Data**
Learn a good representation ϕ(X) that captures relevant information for outcome prediction:
```
min L_factual(ϕ, g₀, g₁) + λ_rec L_rec(ϕ)
```

Where:
- L_factual: Mean squared error on factual outcomes
- L_rec: Reconstruction loss (ensures representation preserves information)
- ϕ: Representation network
- g₀, g₁: Outcome prediction heads

**Stage 2: Fine-tuning on RCT Data**
Adjust the representation to account for unmeasured confounding:
```
min L_factual(ϕ, ϕ_U, g₀, g₁) + λ_shift L_shift + λ_MI L_MI
```

Where:
- ϕ: Frozen representation from Stage 1
- ϕ_U: Learnable adapter module
- L_shift: Regularization to prevent excessive deviation
- L_MI: Mutual information regularization

### Why This Works

**Assumption**: Unmeasured confounding bias is approximately additive or can be captured by a low-dimensional adjustment.

The adapter ϕ_U learns to capture the bias:
```
E[Y₁ | X] = f₁(ϕ(X)) + η(X)
E[Y₀ | X] = f₀(ϕ(X)) + η(X)
```

Where η(X) represents the unmeasured confounding bias.

In RCT data (unconfounded):
```
E[Y₁ | X] = f₁(ϕ(X)) + ϕ_U(ϕ(X))
E[Y₀ | X] = f₀(ϕ(X)) + ϕ_U(ϕ(X))
```

The adapter learns the correction without requiring explicit bias modeling.

## Optimal Transport Enhancement

### Motivation

Standard representation learning methods minimize global discrepancy metrics:
- Maximum Mean Discrepancy (MMD)
- Wasserstein distance
- KL divergence

**Problem**: These ignore local structure. Similar units should be matched together.

### Local Proximity Principle

**Key insight**: For accurate ITE estimation, similar units should have similar outcomes.

In the representation space:
- If ||x₁ - x₂|| is small, then |τ(x₁) - τ(x₂)| should also be small
- This requires preserving local neighborhoods during representation learning

### Fused Gromov-Wasserstein Distance

#### Standard Wasserstein Distance (Global)
Optimal transport cost between distributions P₀ and P₁:
```
W(P₀, P₁) = min_π Σᵢⱼ c(x₀ᵢ, x₁ⱼ) π(i,j)
```

Where:
- c(x₀ᵢ, x₁ⱼ) = ||x₀ᵢ - x₁ⱼ||²: cost of matching unit i to unit j
- π: transport plan (how much mass moves from i to j)
- Constraints: π preserves mass (row sums = P₀, column sums = P₁)

**Limitation**: Only considers pairwise distances, ignores neighborhood structure.

#### Gromov-Wasserstein Distance (Local)
Compares the internal geometry of distributions:
```
GW(P₀, P₁) = min_π Σᵢⱼₖₗ L(d₀(i,k), d₁(j,l)) π(i,j) π(k,l)
```

Where:
- d₀(i,k) = ||x₀ᵢ - x₀ₖ||²: distance within control group
- d₁(j,l) = ||x₁ⱼ - x₁ₗ||²: distance within treatment group
- L(a,b) = |a-b|²: measures how well internal structures match

**Key property**: Matches units with similar neighborhoods.

If x₀ᵢ is close to x₀ₖ and x₁ⱼ is close to x₁ₗ with similar distances, then:
- High probability of matching (i→j) AND (k→l)
- Preserves local proximity

#### Fused Gromov-Wasserstein (FGW)
Combines both global alignment and local proximity:
```
FGW(P₀, P₁) = min_π [κ·Σᵢⱼ c(x₀ᵢ, x₁ⱼ) π(i,j) 
                   + (1-κ)·Σᵢⱼₖₗ L(d₀(i,k), d₁(j,l)) π(i,j) π(k,l)]
```

Where κ ∈ [0,1] balances:
- κ=1: Pure Wasserstein (global only)
- κ=0: Pure Gromov (local only)
- κ∈(0.3, 0.7): Good balance (recommended)

### Sinkhorn Algorithm

Direct solution of OT problem is computationally expensive: O(n³log n).

**Entropic Regularization** makes it easier:
```
W_ε(P₀, P₁) = min_π [⟨C, π⟩ - ε H(π)]
```

Where:
- H(π) = -Σᵢⱼ πᵢⱼ(log πᵢⱼ - 1): entropy of transport plan
- ε > 0: regularization strength

**Sinkhorn Algorithm**: Iterative matrix scaling
```
Initialize: u = 1, v = 1, K = exp(-C/ε)

Repeat until convergence:
  u ← a / (Kv)
  v ← b / (Kᵀu)

Transport plan: π = diag(u) K diag(v)
```

Complexity: O(n²/ε²) - much faster for small ε.

### Informative Subspace Projector (ISP)

#### Curse of Dimensionality

In high dimensions d:
- Distances become uninformative: all pairs have similar distances
- Sample complexity of OT grows exponentially: O(n^(-2/d))
- Need huge samples to reliably estimate transport plan

**Problem for TSPF**: RCT data is limited, high-dimensional representations make OT unreliable.

#### Solution: ISP via PCA

Project representations to informative k-dimensional subspace:
```
U* = arg min_U ||R - RUUᵀ||²  s.t. UᵀU = I
```

Where:
- R: n×d representation matrix
- U: d×k projection matrix
- k = P·d (P is reduction ratio, e.g., 0.7)

**PCA solution**: U* contains top k eigenvectors of RᵀR.

Then compute FGW in reduced space:
```
FGW_ISP(P₀, P₁) = FGW(P₀U*, P₁U*)
```

**Benefits**:
- Reduces dimension from d to k = Pd
- Sample complexity improves: O(n^(-2/k)) vs O(n^(-2/d))
- Preserves most variance (via PCA)
- Faster OT computation

**Tradeoff**: P controls information vs efficiency
- P=1.0: Full dimension (max info, slow, high sample needs)
- P=0.5: Half dimension (less info, faster, lower sample needs)
- P∈[0.6, 0.8]: Good balance (recommended)

## Theoretical Guarantees

### Generalization Bound (Shalit et al., 2017)

For representations R = ϕ(X), the PEHE can be bounded:
```
ε_PEHE ≤ 2[ε_factual + B·IPM(P₀, P₁) - 2σ²_Y]
```

Where:
- ε_factual: Expected factual prediction error
- IPM: Integral probability metric (e.g., Wasserstein distance)
- B: Lipschitz constant of outcome functions
- σ²_Y: Variance of outcomes

**Interpretation**: 
- Minimize factual error → learn good predictors
- Minimize IPM → balance treatment groups
- Both lead to lower PEHE

### Sample Complexity with ISP

With dimension reduction ratio P:
```
E[|FGW(P₀, P₁) - FGW(P̂₀, P̂₁)|] ≲ n^(-2/(Pd))
```

Where P̂₀, P̂₁ are empirical distributions with n samples.

**Benefit of ISP**: Effective dimension Pd < d improves sample complexity.

## Integration in TSPF-OT

### Stage 1 with OT Regularization

**Objective**:
```
min L_factual + λ_rec·L_rec + λ_OT·FGW_ISP(R₀, R₁)
```

Where:
- R₀ = ϕ(X₀): Representations for control group
- R₁ = ϕ(X₁): Representations for treatment group

**Effect**:
- Learns representations that balance treatment groups
- Preserves local proximity between similar units
- Handles high dimensions via ISP

### Stage 2: Same as Standard TSPF

The improved representations from Stage 1 carry over to Stage 2.

## Comparison: Standard TSPF vs TSPF-OT

| Aspect | Standard TSPF | TSPF-OT |
|--------|--------------|---------|
| Treatment balance | None (relies on factual loss) | Explicit via FGW |
| Local structure | Not preserved | Preserved via Gromov term |
| High dimensions | May suffer from curse | Mitigated via ISP |
| Stage 1 loss | Factual + Reconstruction | Factual + Reconstruction + FGW |
| Computational cost | Lower | Higher (due to OT) |
| Performance | Good for small bias | Better for large bias |

## When to Use Each Model

### Use Standard TSPF when:
- Treatment selection bias is small
- Observational data is high quality
- Computational resources are limited
- Sample size is very small (OT may be unreliable)

### Use TSPF-OT when:
- Treatment selection bias is significant
- Treatment and control distributions differ substantially
- Local proximity matters (similar units should match)
- Sufficient sample size for reliable OT computation (n > 100 per group)

## Practical Considerations

### Hyperparameter Selection

**ot_kappa (κ)**:
- Controls balance between global and local alignment
- Start with κ=0.5 (equal weight)
- Increase if need more global alignment (groups far apart)
- Decrease if need more local preservation (groups overlap)

**isp_ratio (P)**:
- Controls dimensionality reduction
- Higher P (0.8-0.9): More information, slower, needs more samples
- Lower P (0.5-0.7): Faster, works with fewer samples, may lose info
- Recommended: Start with 0.7, adjust based on performance

**lambda_ot (λ_OT)**:
- Weight for OT loss in Stage 1
- Too low: Insufficient balancing
- Too high: May harm factual prediction
- Recommended: λ_OT ∈ [0.5, 2.0]

### Diagnostics

**Check if OT is helping**:
1. Visualize t-SNE of representations by treatment group
   - Good: Groups overlap well
   - Bad: Clear separation between groups

2. Compare covariate distributions after transformation
   - Compute MMD or Wasserstein between groups
   - Should decrease with TSPF-OT vs Standard TSPF

3. Check factual prediction performance
   - If OT hurts factual loss significantly, reduce λ_OT

**Check if ISP is appropriate**:
1. Plot explained variance vs number of components
   - Should retain >90% with P=0.7
   - If not, may need higher P or more data preprocessing

2. Compare OT loss with and without ISP
   - Should be similar (ISP preserves main structure)

## References

1. Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies.

2. Shalit, U., Johansson, F. D., & Sontag, D. (2017). Estimating individual treatment effect: generalization bounds and algorithms. ICML.

3. Zhou, C., et al. (2025). A Two-Stage Pretraining-Finetuning Framework for Treatment Effect Estimation with Unmeasured Confounding. KDD.

4. Wang, H., et al. (2024). Proximity Matters: Local Proximity Preserved Balancing for Treatment Effect Estimation. arXiv:2407.01111.

5. Peyré, G., & Cuturi, M. (2019). Computational optimal transport. Foundations and Trends in Machine Learning.

6. Mémoli, F. (2011). Gromov–Wasserstein distances and the metric approach to object matching. Foundations of Computational Mathematics.
