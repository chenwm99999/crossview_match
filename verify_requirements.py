# verify_requirements.py - Verify all packages work with our architecture

import sys
import subprocess

def check_import(package_name, import_name=None):
    """Try importing a package"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name:30s} OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name:30s} FAILED: {e}")
        return False

def check_model_availability():
    """Check if our specific models are available"""
    print("\n" + "="*60)
    print("MODEL AVAILABILITY CHECK")
    print("="*60)
    
    # Check Swin-Tiny
    try:
        import timm
        models = timm.list_models('swin_tiny*')
        if 'swin_tiny_patch4_window7_224' in models:
            print("✓ Swin-Tiny model available")
        else:
            print("✗ Swin-Tiny model NOT FOUND")
            print(f"  Available: {models}")
    except Exception as e:
        print(f"✗ timm check failed: {e}")
    
    # Check CLIP
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        print("✓ CLIP ViT-B/32 loadable")
    except Exception as e:
        print(f"✗ CLIP load failed: {e}")
    
    # Check FAISS
    try:
        import faiss
        # Test creating an index
        index = faiss.IndexFlatIP(512)
        print(f"✓ FAISS working (version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'})")
        
        # Check GPU support
        if hasattr(faiss, 'get_num_gpus'):
            num_gpus = faiss.get_num_gpus()
            print(f"  FAISS GPU count: {num_gpus}")
    except Exception as e:
        print(f"✗ FAISS check failed: {e}")

def check_pytorch_setup():
    """Verify PyTorch configuration"""
    print("\n" + "="*60)
    print("PYTORCH CONFIGURATION")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
            
            # Test tensor creation on GPU
            x = torch.randn(2, 3).cuda()
            print(f"  GPU tensor creation: OK")
        else:
            print("  ⚠ No GPU detected")
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")

def main():
    print("="*60)
    print("REQUIREMENTS VERIFICATION")
    print("="*60)
    print(f"Python: {sys.version.split()[0]}\n")
    
    # Core packages
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("timm", "timm"),
        ("CLIP", "clip"),
        ("FAISS", "faiss"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Pillow", "PIL"),
        ("OpenCV", "cv2"),
        ("SciPy", "scipy"),
        ("Albumentations", "albumentations"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("TensorBoard", "tensorboard"),
        ("Weights & Biases", "wandb"),
        ("scikit-learn", "sklearn"),
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("PyYAML", "yaml"),
        ("tqdm", "tqdm"),
        ("h5py", "h5py"),
        ("gdown", "gdown"),
        ("einops", "einops"),
        ("ftfy", "ftfy"),
        ("regex", "regex"),
    ]
    
    success = 0
    failed = 0
    
    for package_name, import_name in packages:
        if check_import(package_name, import_name):
            success += 1
        else:
            failed += 1
    
    # Check model availability
    check_model_availability()
    
    # Check PyTorch setup
    check_pytorch_setup()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Successful: {success}/{len(packages)}")
    print(f"✗ Failed: {failed}/{len(packages)}")
    
    if failed == 0:
        print("\n🎉 All requirements verified successfully!")
        print("Ready to start Phase 0 implementation.")
    else:
        print(f"\n⚠ {failed} package(s) failed. Please reinstall:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()