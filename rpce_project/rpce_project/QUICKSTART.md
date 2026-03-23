# QUICK START GUIDE

## ⚡ 5-Minute Setup

### 1. Open in VS Code
```bash
# Open VS Code
# File → Open Folder → Select rpce_project
```

### 2. Install Dependencies
Open Terminal in VS Code (`Ctrl+``) and run:
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux

# Install packages
pip install -r requirements.txt
```

### 3. Test Installation
```bash
python test_installation.py
```
You should see all ✓ checkmarks.

### 4. Update Data Paths
Edit `train.py` line 50-51:
```python
TRAIN_PATH = "C:/your/path/to/train.npz"
TEST_PATH = "C:/your/path/to/test.npz"
```

### 5. Run Training
**Option A: Using Run Button**
- Open `train.py`
- Click the ▷ button in top-right
- OR press `F5`

**Option B: Using Terminal**
```bash
python train.py
```

## 📊 What to Expect

The training will:
1. Load your data
2. Train Stage 1 (50 epochs) - ~2-5 minutes
3. Train Stage 2 (50 epochs) - ~2-5 minutes
4. Evaluate and save results
5. Create plots in `figures/`
6. Save model to `checkpoints/`

## 🎯 Output Files

After training completes:
```
rpce_project/
├── checkpoints/
│   └── rpce_model.pt          # Trained model
├── figures/
│   └── training_curves.png    # Training plots
```

## 🔍 Debugging in VS Code

### Set Breakpoints
1. Click left of line number to add red dot
2. Press `F5` to run in debug mode
3. Use debug controls to step through code

### View Variables
- When paused, hover over variables to see values
- Check "Variables" panel in left sidebar

### Debug Console
- Type variable names to inspect them
- Run Python code while paused

## 📝 Common Commands

```bash
# Activate environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Run training
python train.py

# Run tests
python test_installation.py

# Start Python interactive
python

# Deactivate environment
deactivate
```

## ⚠️ Common Issues

### "No module named 'config'"
**Solution**: Make sure you're in `rpce_project` folder
```bash
cd rpce_project
python train.py
```

### "Cannot find data files"
**Solution**: Use absolute paths in `train.py`
```python
TRAIN_PATH = "C:/Users/YourName/Desktop/Data/train.npz"
```

### VS Code doesn't find Python
**Solution**: Select interpreter
- `Ctrl+Shift+P` → "Python: Select Interpreter"
- Choose the one with `venv` in the path

## 🚀 Next Steps

1. **Experiment with hyperparameters** in `train.py`:
   - `HIDDEN_DIM`: Model capacity
   - `LATENT_DIM`: Representation size
   - `STAGE1_EPOCHS` / `STAGE2_EPOCHS`: Training length

2. **Custom analysis** - Create new file:
   ```python
   from evaluation import evaluate_jobs_policy_risk_and_att
   from data import load_jobs_data
   
   # Your analysis code here
   ```

3. **Visualize results**:
   ```python
   import matplotlib.pyplot as plt
   plt.scatter(metrics['cate_pred'], metrics['confidence'])
   plt.xlabel('Predicted CATE')
   plt.ylabel('Confidence')
   plt.show()
   ```

## 📚 Learn More

- See `README.md` for detailed documentation
- Check individual module files for function docs
- Example: `models/autoencoder.py` has detailed architecture comments
