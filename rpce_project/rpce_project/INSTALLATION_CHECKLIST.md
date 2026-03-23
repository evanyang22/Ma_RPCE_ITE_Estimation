# RPCE Installation Checklist

Print this and check off each step as you complete it!

## ☐ 1. Prerequisites
- [ ] Python 3.9+ installed
  - Test: Open terminal/command prompt, type `python --version`
  - Should show: Python 3.9.x or higher
  - If not: Download from https://python.org

- [ ] VS Code installed
  - Download from https://code.visualstudio.com/

## ☐ 2. Open Project in VS Code
- [ ] Extract `rpce_project.zip` (or `.tar.gz`)
- [ ] Open VS Code
- [ ] File → Open Folder → Select `rpce_project`

## ☐ 3. Install Python Extension
- [ ] Click Extensions icon (left sidebar)
- [ ] Search "Python"
- [ ] Install "Python" by Microsoft

## ☐ 4. Create Virtual Environment
Open terminal in VS Code (`Ctrl+``) and run:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

- [ ] Terminal shows `(venv)` in prompt

## ☐ 5. Install Dependencies
```bash
pip install -r requirements.txt
```

- [ ] Installation completes without errors
- [ ] See "Successfully installed..." messages

## ☐ 6. Select Python Interpreter
- [ ] Press `Ctrl+Shift+P`
- [ ] Type "Python: Select Interpreter"
- [ ] Choose the one with `venv` in the path

## ☐ 7. Test Installation
```bash
python test_installation.py
```

- [ ] All tests show ✓ (green checkmarks)
- [ ] No ✗ (red X marks)

## ☐ 8. Update Data Paths
- [ ] Open `train.py`
- [ ] Find lines 50-51
- [ ] Update paths to your data files:
  ```python
  TRAIN_PATH = "your/actual/path/here.npz"
  TEST_PATH = "your/actual/path/here.npz"
  ```

## ☐ 9. Run Training
**Option A: Use Run Button**
- [ ] Keep `train.py` open
- [ ] Click ▷ button (top-right)

**Option B: Use Terminal**
```bash
python train.py
```

## ☐ 10. Verify Results
After training completes:
- [ ] `checkpoints/rpce_model.pt` exists
- [ ] `figures/training_curves.png` exists
- [ ] Terminal shows evaluation metrics

---

## 🎉 Success Criteria

You're done when you see:
```
================================================================================
Training Complete!
================================================================================
```

And find these files:
- ✓ `checkpoints/rpce_model.pt`
- ✓ `figures/training_curves.png`

---

## ❌ Troubleshooting

### If Step 4 fails (venv creation):
```bash
# Try with python3
python3 -m venv venv
```

### If Step 5 fails (pip install):
```bash
# Upgrade pip first
pip install --upgrade pip
# Then retry
pip install -r requirements.txt
```

### If Step 7 fails (imports):
- Make sure venv is activated (see `(venv)` in prompt)
- Try: `pip install -r requirements.txt --upgrade`

### If Step 9 fails (file not found):
- Check data paths are correct
- Use absolute paths: `C:/Users/Name/Data/file.npz`
- Make sure files exist at those locations

### If nothing works:
1. Close VS Code completely
2. Open fresh terminal/command prompt
3. Navigate to `rpce_project`
4. Start from Step 4 again

---

## 📞 Quick Reference

**Activate venv:**
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

**Run training:**
- `python train.py`

**Test installation:**
- `python test_installation.py`

**Deactivate venv:**
- `deactivate`

---

## ⏱️ Expected Timeline

- Steps 1-7: 10-15 minutes
- Step 8: 1 minute
- Step 9 (training): 5-10 minutes
- Total: ~20-30 minutes

---

**Date Completed:** _______________

**Notes:**
_____________________________________________________
_____________________________________________________
_____________________________________________________
