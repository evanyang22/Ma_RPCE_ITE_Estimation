"""
Quick test script to verify RPCE installation.

Run this to check if all modules are installed correctly.
"""

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...\n")
    
    tests = [
        ("PyTorch", lambda: __import__('torch')),
        ("NumPy", lambda: __import__('numpy')),
        ("Matplotlib", lambda: __import__('matplotlib')),
        ("Pandas", lambda: __import__('pandas')),
        ("POT (Optimal Transport)", lambda: __import__('ot')),
        ("Config module", lambda: __import__('config')),
        ("Data module", lambda: __import__('data')),
        ("Models module", lambda: __import__('models')),
        ("Training module", lambda: __import__('training')),
        ("Transport module", lambda: __import__('transport')),
        ("Evaluation module", lambda: __import__('evaluation')),
        ("Utils module", lambda: __import__('utils')),
    ]
    
    failed = []
    
    for name, import_func in tests:
        try:
            import_func()
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            failed.append(name)
    
    print("\n" + "="*60)
    if not failed:
        print("✓ All imports successful!")
        print("Your installation is ready to use.")
    else:
        print(f"✗ {len(failed)} import(s) failed:")
        for name in failed:
            print(f"  - {name}")
        print("\nRun: pip install -r requirements.txt")
    print("="*60)
    
    return len(failed) == 0


def test_model_creation():
    """Test if model can be created."""
    print("\nTesting model creation...")
    
    try:
        from models import AutoEncoder
        model = AutoEncoder(input_dim=17, hidden_dim=8, latent_dim=4)
        print("✓ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_device():
    """Test device availability."""
    print("\nTesting device availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("! CUDA not available (will use CPU)")
        
        # Test MPS (Mac)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✓ MPS (Mac GPU) available")
        
        return True
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("RPCE Installation Test")
    print("="*60)
    
    imports_ok = test_imports()
    model_ok = test_model_creation() if imports_ok else False
    device_ok = test_device() if imports_ok else False
    
    print("\n" + "="*60)
    if imports_ok and model_ok:
        print("✓ All tests passed!")
        print("\nYou're ready to run:")
        print("  python train.py")
    else:
        print("✗ Some tests failed")
        print("\nPlease fix the issues above before running train.py")
    print("="*60)


if __name__ == "__main__":
    main()
