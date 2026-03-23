# VS Code Setup Guide for RPCE Project

## 🎯 Step-by-Step Instructions

### Step 1: Install VS Code (if not already installed)
1. Download from https://code.visualstudio.com/
2. Run installer
3. During installation, check:
   - ✓ Add "Open with Code" to context menu
   - ✓ Add to PATH

### Step 2: Install Python Extension
1. Open VS Code
2. Click Extensions icon (□ icon on left sidebar) OR press `Ctrl+Shift+X`
3. Search for "Python"
4. Install "Python" by Microsoft (the one with 80M+ downloads)
5. Also install "Pylance" (usually comes with Python extension)

### Step 3: Open the Project
1. File → Open Folder
2. Navigate to `rpce_project` folder
3. Click "Select Folder"

### Step 4: Set Up Python Environment

**In VS Code Terminal** (View → Terminal OR `Ctrl+``):

```bash
# Navigate to project directory (if not there)
cd path/to/rpce_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# You should see (venv) in your terminal prompt

# Install dependencies
pip install -r requirements.txt
```

### Step 5: Select Python Interpreter
1. Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac)
2. Type: "Python: Select Interpreter"
3. Choose the interpreter that shows `./venv/Scripts/python.exe` (Windows) or `./venv/bin/python` (Mac/Linux)

### Step 6: Verify Installation
1. Open `test_installation.py`
2. Right-click in editor → "Run Python File in Terminal"
3. You should see all ✓ checkmarks

## ▶️ Running the Model

### Method 1: Using the Run Button (Easiest)
1. Open `train.py`
2. Update data paths on lines 50-51:
   ```python
   TRAIN_PATH = "C:/your/path/to/train.npz"
   TEST_PATH = "C:/your/path/to/test.npz"
   ```
3. Click the ▷ (Run) button in the top-right corner
4. OR press `F5`

### Method 2: Using Terminal
```bash
# Make sure venv is activated (you see (venv) in prompt)
python train.py
```

### Method 3: Interactive Debugging
1. Click to the left of a line number to set a breakpoint (red dot appears)
2. Press `F5` to start debugging
3. Code will pause at breakpoint
4. Use debug toolbar:
   - Continue (F5)
   - Step Over (F10)
   - Step Into (F11)
   - Step Out (Shift+F11)

## 🎨 VS Code Features for This Project

### Code Navigation
- `Ctrl+Click` on a function name → Jump to definition
- `Alt+←` → Go back
- `Ctrl+P` → Quick file search
- `Ctrl+Shift+F` → Search across all files

### Code Completion
- Start typing, VS Code suggests completions
- `Ctrl+Space` → Force show completions
- Tab to accept suggestion

### Viewing Documentation
- Hover over any function to see its docstring
- Example: Hover over `AutoEncoder` to see what it does

### Split View
- Right-click a file tab → "Split Right"
- View multiple files side-by-side
- Useful for comparing train.py with model files

### Terminal Management
- `Ctrl+`` → Toggle terminal
- Click + to open new terminal
- Click trash icon to close terminal

## 🔧 Useful Keyboard Shortcuts

| Action | Windows/Linux | Mac |
|--------|--------------|-----|
| Run File | F5 | F5 |
| Toggle Terminal | Ctrl+` | Cmd+` |
| Command Palette | Ctrl+Shift+P | Cmd+Shift+P |
| Quick Open File | Ctrl+P | Cmd+P |
| Find in Files | Ctrl+Shift+F | Cmd+Shift+F |
| Go to Line | Ctrl+G | Cmd+G |
| Comment Line | Ctrl+/ | Cmd+/ |
| Save All | Ctrl+K S | Cmd+K S |

## 📊 Viewing Results

### Training Output
- Watch the terminal for progress
- Training losses print every 10 epochs
- Final evaluation metrics print at the end

### Generated Files
After training completes:
1. Click Explorer icon (📁) in left sidebar
2. Navigate to:
   - `checkpoints/rpce_model.pt` - Saved model
   - `figures/training_curves.png` - Training plots
3. Click on `.png` files to view them in VS Code

### Plots
- `.png` files open in VS Code preview
- Right-click image → "Reveal in File Explorer" to open folder

## 🐛 Debugging Tips

### If imports fail:
1. Check that venv is activated (see `(venv)` in terminal)
2. Select correct Python interpreter (Step 5 above)
3. Make sure you're in `rpce_project` folder

### If "module not found":
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### If code changes don't take effect:
```bash
# Restart Python
# In terminal: Ctrl+C to stop
# Run again: python train.py
```

### View Python output:
- Check the TERMINAL tab (not Problems or Debug Console)
- Make sure correct terminal is selected if you have multiple

## 💡 Pro Tips

1. **Use Integrated Terminal**: Stay in VS Code instead of switching windows

2. **Format Code**: Install Black formatter
   ```bash
   pip install black
   ```
   Then: Right-click → "Format Document"

3. **Type Hints**: VS Code shows parameter types when you hover

4. **Snippets**: Type `def` and press Tab for function template

5. **Multi-Cursor**: Alt+Click to place multiple cursors

6. **Zen Mode**: View → Appearance → Zen Mode for distraction-free coding

## 🎓 Learning Resources

- VS Code Python Tutorial: https://code.visualstudio.com/docs/python/python-tutorial
- Debugging: https://code.visualstudio.com/docs/python/debugging
- Keyboard Shortcuts: Help → Keyboard Shortcuts Reference

## ❓ Common Questions

**Q: Do I need to activate venv every time?**
A: Yes, each time you open a new terminal. VS Code should auto-activate if interpreter is set correctly.

**Q: Can I use Jupyter notebooks?**
A: Yes! Install Jupyter extension, then create `.ipynb` files.

**Q: How do I change model parameters?**
A: Edit `train.py` lines 53-62 (hyperparameters section).

**Q: Where are print() statements shown?**
A: In the TERMINAL tab at the bottom of VS Code.

**Q: Can I pause execution?**
A: Yes, click left of line number to set breakpoint, then press F5 to debug.
