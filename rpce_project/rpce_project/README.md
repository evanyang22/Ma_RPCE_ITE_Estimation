# RPCE Model - Robust Proximal Causal Effect Estimation

This project implements a two-stage deep learning model for causal inference using observational and randomized control trial (RCT) data.

## 📋 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Running in VS Code](#running-in-vs-code)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The RPCE model combines:
1. **AutoEncoder**: Learns latent representations of covariates
2. **Two-Stage Training**:
   - Stage 1: Train on observational data with pseudo-outcomes
   - Stage 2: Fine-tune on RCT data with unconfounded outcomes
3. **Optimal Transport**: Uses Sinkhorn projection to align observational and RCT distributions
4. **CATE Estimation**: Predicts Conditional Average Treatment Effects

## 💻 Installation

### Step 1: Install Python
- Download Python 3.9+ from [python.org](https://www.python.org/downloads/)
- During installation, **check "Add Python to PATH"**

### Step 2: Install VS Code
- Download from [code.visualstudio.com](https://code.visualstudio.com/)
- Install the **Python extension** (by Microsoft)

### Step 3: Install Dependencies
Open a terminal in VS Code (Terminal → New Terminal) and run:

```bash
# Navigate to project directory
cd path/to/rpce_project

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Running in VS Code

### Method 1: Run Entire Script (Recommended)

1. **Open the project folder** in VS Code:
   - File → Open Folder → Select `rpce_project`

2. **Update data paths** in `train.py`:
   ```python
   # Line 50-51 in train.py
   TRAIN_PATH = "your/path/to/jobs_DW_bin.new.10.train.npz"
   TEST_PATH = "your/path/to/jobs_DW_bin.new.10.test.npz"
   ```

3. **Select Python interpreter**:
   - Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from your virtual environment

4. **Run the training script**:
   - Open `train.py`
   - Press `F5` or click the ▷ button in top-right
   - OR: Right-click in editor → "Run Python File in Terminal"

### Method 2: Interactive Python (REPL)

1. Open VS Code terminal
2. Activate virtual environment
3. Start Python:
   ```bash
   python
   ```
4. Run interactively:
   ```python
   from config import set_random_seed
   from data import load_jobs_data
   from models import AutoEncoder
   
   set_random_seed(42)
   data = load_jobs_data("train.npz", "test.npz")
   model = AutoEncoder(input_dim=17, hidden_dim=8, latent_dim=4)
   ```

### Method 3: Jupyter Notebook in VS Code

1. Install Jupyter extension in VS Code
2. Create a new notebook: `analysis.ipynb`
3. Import and use modules:
   ```python
   # Cell 1
   from train import main
   
   # Cell 2
   model, metrics = main()
   
   # Cell 3
   print(f"Policy Risk: {metrics['policy_risk']:.4f}")
   ```

## 📁 Project Structure

```
rpce_project/
├── train.py                     # Main training script - START HERE
├── config.py                    # Configuration & seeds
├── requirements.txt             # Python dependencies
├── data/
│   ├── loader.py               # Data loading functions
│   └── exploration.py          # Visualization utilities
├── models/
│   ├── autoencoder.py          # Model architecture
│   └── losses.py               # Loss functions
├── training/
│   ├── stage1.py               # Observational training
│   └── stage2.py               # RCT fine-tuning
├── transport/
│   └── sinkhorn.py             # Optimal transport
├── evaluation/
│   ├── metrics.py              # Evaluation metrics
│   └── evaluate.py             # Evaluation pipeline
└── utils/
    └── data_utils.py           # Helper functions
```

## 📖 Usage

### Basic Training

```python
from train import main

# Run complete training pipeline
model, metrics = main()

# View results
print(f"Policy Risk: {metrics['policy_risk']:.4f}")
print(f"ATT Error: {metrics['att_error']:.4f}")
```

### Custom Training

```python
from config import set_random_seed, get_device
from data import load_jobs_data
from models import AutoEncoder
from training import train_stage1, train_stage2, initialize_stage2_from_stage1

# Setup
set_random_seed(42)
device = get_device()

# Load data
data = load_jobs_data("train.npz", "test.npz")

# Create model
model = AutoEncoder(input_dim=17, hidden_dim=8, latent_dim=4)

# Stage 1: Train on observational data
model, history1 = train_stage1(
    model=model,
    obs_dataset=data['train_obs'],
    num_epochs=50,
    device=device
)

# Stage 2: Fine-tune on RCT data
model = initialize_stage2_from_stage1(model)
model, history2 = train_stage2(
    model=model,
    rct_dataset=data['train_rct'],
    obs_dataset=data['train_obs'],
    num_epochs=50,
    device=device
)
```

### Evaluation Only

```python
import torch
from evaluation import evaluate_jobs_policy_risk_and_att

# Load saved model
checkpoint = torch.load("checkpoints/rpce_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
metrics = evaluate_jobs_policy_risk_and_att(
    model=model,
    eval_rct_dataset=data['test_rct'],
    verbose=True
)
```

## 🔧 Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'config'`

**Solution**:
- Make sure you're in the `rpce_project` directory
- Run Python from the project root:
  ```bash
  cd rpce_project
  python train.py
  ```

### CUDA/GPU Issues

**Problem**: `CUDA out of memory` or GPU not detected

**Solution**:
- The code automatically falls back to CPU
- To force CPU: Set `device='cpu'` in function calls
- Reduce batch size if using GPU

### File Not Found

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**:
- Check data paths in `train.py` (lines 50-51)
- Use absolute paths or paths relative to project root
- Example: `"C:/Users/YourName/Data/train.npz"`

### POT Library Issues

**Problem**: `ImportError: cannot import name 'ot'`

**Solution**:
```bash
pip install pot --upgrade
```

## 🎓 Model Details

### Stage 1: Observational Training
- **Input**: Observational data (X, T, Y)
- **Trains**:
  - Encoder/Decoder (reconstruction)
  - Propensity head (treatment prediction)
  - Pseudo-outcome heads (biased outcome estimation)
- **Loss**: Reconstruction + Propensity + Pseudo-outcome

### Stage 2: RCT Fine-tuning
- **Input**: RCT data (X, T, Y) + Observational latents
- **Freezes**: Encoder, Decoder, Stage 1 heads
- **Trains**: Unconfounded outcome heads (g0, g1)
- **Uses**: Optimal transport for domain adaptation

## 📊 Outputs

After training, you'll find:
- `checkpoints/rpce_model.pt` - Saved model
- `figures/training_curves.png` - Training plots
- Console output with evaluation metrics
