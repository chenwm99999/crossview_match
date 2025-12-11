"""Test loading pre-trained models"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
from pathlib import Path

def test_l2ltr():
    """Test L2LTR checkpoint"""
    print("\n" + "=" * 60)
    print("TESTING L2LTR CHECKPOINT")
    print("=" * 60)
    
    checkpoint_path = Path("pretrained/l2ltr/l2ltr_cvusa.pth")
    
    if not checkpoint_path.exists():
        print(f" Not found: {checkpoint_path}")
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f" Loaded: {checkpoint_path.name}")
        print(f"  Size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  Keys: {list(checkpoint.keys())}")
        
        # L2LTR has separate models for ground and satellite
        if 'model_grd' in checkpoint and 'model_sat' in checkpoint:
            print(f"\n  Structure: Separate ground and satellite encoders")
            
            # Count parameters for each
            grd_params = sum(p.numel() for p in checkpoint['model_grd'].values() if isinstance(p, torch.Tensor))
            sat_params = sum(p.numel() for p in checkpoint['model_sat'].values() if isinstance(p, torch.Tensor))
            
            print(f"  Ground encoder: {grd_params / 1e6:.1f}M params")
            print(f"  Satellite encoder: {sat_params / 1e6:.1f}M params")
            print(f"  Total: {(grd_params + sat_params) / 1e6:.1f}M params")
            
            # Look for L2L layers in ground model
            grd_keys = list(checkpoint['model_grd'].keys())
            l2l_layers = [k for k in grd_keys if 'cross' in k.lower() or 'attn' in k.lower()]
            
            print(f"\n  Sample layer names from ground encoder:")
            for key in grd_keys[:5]:
                print(f"    - {key}")
            
            if l2l_layers:
                print(f"\n  L2L/Attention layers found: {len(l2l_layers)}")
                print(f"  Example: {l2l_layers[0]}")
        
        print(f"\n L2LTR checkpoint is valid")
        print(f"\n  Use case:")
        print(f"    - Reference for implementing L2L cross-attention")
        print(f"    - Study layer structure and naming")
        print(f"    - Cannot directly use (ViT architecture, not Swin)")
        
        return True
        
    except Exception as e:
        print(f"Error loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_baseline():
    """Test baseline checkpoint"""
    print("\n" + "=" * 60)
    print("TESTING UNIVERSITY-1652 BASELINE")
    print("=" * 60)
    
    checkpoint_path = Path("pretrained/university1652_baseline/resnet50_uni1652.pth")
    
    if not checkpoint_path.exists():
        print(f" Not found: {checkpoint_path}")
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f" Loaded: {checkpoint_path.name}")
        print(f"  Size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  Architecture: ResNet-50")
        print(f"  Performance: 86.7% R@1")
        print(f"\n Baseline checkpoint is valid")
        return True
        
    except Exception as e:
        print(f" Error loading: {e}")
        return False

def main():
    print("=" * 60)
    print("PRE-TRAINED MODEL VERIFICATION")
    print("=" * 60)
    
    l2ltr_ok = test_l2ltr()
    baseline_ok = test_baseline()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"L2LTR checkpoint: {' OK' if l2ltr_ok else ' Missing'}")
    print(f"Baseline checkpoint: {' OK' if baseline_ok else ' Missing'}")
    
    print("\n" + "=" * 60)
    print("TRAINING STRATEGY")
    print("=" * 60)
    print("Mode: Train from scratch")
    print("Initialization: Swin-Tiny ImageNet-22k")
    print("Teachers: CLIP + DINOv2 (frozen)")
    print("Reference: L2LTR for L2L implementation")
    print("Baseline: ResNet-50")
    print("=" * 60)

if __name__ == "__main__":
    main()