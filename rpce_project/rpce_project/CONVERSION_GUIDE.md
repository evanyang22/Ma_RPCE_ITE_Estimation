# Notebook to Modular Code: Conversion Guide

## 🔄 Before & After Comparison

### Original Notebook Structure (RPCE.ipynb)

```
Cell 1:  Imports + random seed setup (26 lines)
Cell 2:  createJobsTensorDataset function (51 lines)
Cell 3:  Loading train/test data (12 lines)
Cell 4:  explore_tensor function (17 lines)
Cell 5:  Commented exploration code (2 lines)
Cell 6:  AutoEncoder class definition (87 lines)
Cell 7:  pseudo_outcome_loss function (36 lines)
Cell 8:  Training functions (174 lines)
Cell 9:  Model initialization (2 lines)
Cell 10: Stage 2 training functions (158 lines)
Cell 11: Transfer learning code (14 lines)
Cell 12: Optimal transport functions (204 lines)
Cell 13: Import comment (2 lines)
Cell 14: Test function (193 lines)
Cell 15: Test execution (1 line)
Cell 16: Evaluation functions (280 lines)
Cell 17: Evaluation execution (9 lines)
Cell 18: Empty cell (1 line)

TOTAL: ~1,269 lines in 18 cells
```

### New Modular Structure

```
rpce_project/
├── config.py                 (65 lines)   ← Cell 1 (setup)
├── data/
│   ├── loader.py            (84 lines)   ← Cell 2, 3 (data loading)
│   └── exploration.py       (44 lines)   ← Cell 4, 5 (visualization)
├── models/
│   ├── autoencoder.py       (143 lines)  ← Cell 6 (model architecture)
│   └── losses.py            (126 lines)  ← Cell 7 (loss functions)
├── training/
│   ├── stage1.py            (145 lines)  ← Cell 8 (stage 1 training)
│   └── stage2.py            (193 lines)  ← Cell 10, 11 (stage 2 training)
├── transport/
│   └── sinkhorn.py          (177 lines)  ← Cell 12 (optimal transport)
├── evaluation/
│   ├── metrics.py           (117 lines)  ← Cell 16 (metrics)
│   └── evaluate.py          (150 lines)  ← Cell 14, 16, 17 (evaluation)
├── utils/
│   ├── data_utils.py        (134 lines)  ← Helper functions
│   └── visualization.py     (285 lines)  ← New: plotting utilities
├── train.py                 (206 lines)  ← New: main training script
├── inference.py             (230 lines)  ← New: inference script
├── example.py               (145 lines)  ← New: example workflow
├── README.md                (400+ lines) ← New: documentation
└── requirements.txt         (14 lines)   ← New: dependencies

TOTAL: ~2,658 lines in organized modules
```

## 📊 Key Improvements

### 1. **Separation of Concerns**

**Before (Notebook):**
```python
# Everything in one cell
def train_mixed_autoencoder(...):
    # 174 lines mixing:
    # - Data preprocessing
    # - Loss computation
    # - Training loop
    # - Visualization
    # - Model saving
```

**After (Modular):**
```python
# training/stage1.py - Training logic only
def train_stage1(model, dataset, ...):
    # Training loop
    
# models/losses.py - Loss functions only
def reconstruction_loss(...):
def propensity_loss(...):
def pseudo_outcome_loss(...):

# utils/visualization.py - Plotting only
def plot_training_history(...):
```

### 2. **Reusability**

**Before:**
```python
# Cell 8: All training code in one function
# Can't reuse parts independently
# Hard to modify specific components
```

**After:**
```python
# Import only what you need
from training import train_stage1
from models import AutoEncoder, pseudo_outcome_loss
from transport import predict_cate_rpce

# Mix and match components
# Easy to swap implementations
# Clear dependencies
```

### 3. **Testability**

**Before:**
```python
# Cells executed in sequence
# No easy way to test individual functions
# Must re-run entire notebook
```

**After:**
```python
# Each module can be tested independently
import unittest
from models.losses import pseudo_outcome_loss

class TestLosses(unittest.TestCase):
    def test_pseudo_outcome_loss(self):
        # Test specific function
        ...
```

### 4. **Documentation**

**Before:**
```python
# Minimal docstrings
# Functionality spread across cells
# Hard to understand flow
```

**After:**
```python
# Comprehensive docstrings
"""
Calculate pseudo-outcome loss for observational data.

This loss trains only the factual branch:
- If t=0, compare y0_hat to y
- If t=1, compare y1_hat to y

Args:
    y0_hat (torch.Tensor): Predicted outcome for t=0
    ...
Returns:
    torch.Tensor: Scalar loss value
"""
```

## 🎯 Usage Comparison

### Training a Model

**Before (Notebook):**
```python
# Run cells 1-9 in sequence
# Cell 1: Imports
# Cell 2: Define createJobsTensorDataset
# Cell 3: Load data
# ...
# Cell 8: Define and run train_mixed_autoencoder
# Cell 9: Initialize model
# Can't skip or reorder
```

**After (Modular):**
```python
# Clean, explicit workflow
from config import set_random_seed
from data import load_jobs_data
from models import AutoEncoder
from training import train_stage1, train_stage2

set_random_seed(42)
data = load_jobs_data(train_path, test_path)
model = AutoEncoder(17, 8, 4)
model, hist = train_stage1(model, data['train_obs'])
```

### Making Predictions

**Before (Notebook):**
```python
# Copy-paste code from cells
# Modify inline
# Hope dependencies are loaded
```

**After (Modular):**
```python
# Use dedicated inference script
python inference.py \
    --model_path model.pt \
    --data_path new_data.npz \
    --rct_reference_path rct_data.npz
```

## 🔧 Modification Examples

### Adding a New Loss Function

**Before:**
```python
# Find Cell 7 or 8
# Add function inline
# Modify train_mixed_autoencoder
# Re-run entire notebook
```

**After:**
```python
# 1. Add to models/losses.py
def my_new_loss(pred, target):
    ...

# 2. Update training/stage1.py
from ..models.losses import my_new_loss

# 3. Use it
total_loss = recon_loss + my_new_loss(...)
```

### Changing the Architecture

**Before:**
```python
# Modify Cell 6
# Re-run all dependent cells
# Manual tracking of changes
```

**After:**
```python
# 1. Modify models/autoencoder.py
class AutoEncoder(nn.Module):
    def __init__(self, ...):
        # Add new layers
        
# 2. Everything else works automatically
# No need to modify training code
```

### Adding Visualization

**Before:**
```python
# Add plotting code to Cell 8 or 17
# Mixes visualization with logic
# Hard to reuse plots
```

**After:**
```python
# 1. Add to utils/visualization.py
def plot_my_analysis(...):
    ...

# 2. Use anywhere
from utils.visualization import plot_my_analysis
plot_my_analysis(results)
```

## 📈 Benefits Summary

| Aspect | Notebook | Modular |
|--------|----------|---------|
| **Organization** | Linear cells | Logical modules |
| **Reusability** | Copy-paste | Import |
| **Testing** | Manual | Automated |
| **Collaboration** | Merge conflicts | Clean git diffs |
| **Documentation** | Scattered | Centralized |
| **Deployment** | Difficult | Easy (train.py) |
| **Debugging** | Cell by cell | Function by function |
| **Scaling** | Doesn't scale | Scales well |

## 🚀 Migration Path

If you want to migrate your own notebook:

1. **Identify logical groups** (data, model, training, evaluation)
2. **Extract functions** into appropriate modules
3. **Add type hints and docstrings**
4. **Create __init__.py** files for clean imports
5. **Write main scripts** (train.py, inference.py)
6. **Add visualization utilities**
7. **Document in README.md**
8. **Add requirements.txt**

## 💡 Best Practices Applied

✅ **Single Responsibility**: Each module does one thing
✅ **DRY (Don't Repeat Yourself)**: Shared code in utils/
✅ **Clear Interfaces**: Well-defined function signatures
✅ **Type Hints**: Better IDE support and documentation
✅ **Comprehensive Docstrings**: Self-documenting code
✅ **Separation of Concerns**: Logic vs visualization vs config
✅ **Version Control Friendly**: Small, focused files
✅ **Production Ready**: Easy to deploy and maintain
