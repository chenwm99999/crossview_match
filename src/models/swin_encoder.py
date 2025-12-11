"""Swin-Tiny backbone encoder for cross-view matching"""

import torch
import torch.nn as nn
import timm

class SwinEncoder(nn.Module):
    """
    Swin-Tiny encoder extracting multi-scale features
    Modified to accept 256x256 input (instead of default 224x224)
    
    Output stages:
        Stage 2: [B, 96, 64, 64]
        Stage 3: [B, 192, 32, 32]
        Stage 4: [B, 384, 16, 16]
    """
    
    def __init__(self, pretrained=True, img_size=256):
        super().__init__()
        
        # Load Swin-Tiny with custom image size
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3],  # Stages 2, 3, 4
            img_size=img_size  #  Set to 256 
        )
        
        print(f" Swin-Tiny loaded")
        print(f"  Pretrained: {pretrained}")
        print(f"  Input size: {img_size}x{img_size}")
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, 256, 256]
        Returns:
            List of features from stages 2, 3, 4
        """
        features = self.backbone(x)
        return features


# Test, comments out when release
# if __name__ == "__main__":
#     print("Testing SwinEncoder")
    
#     model = SwinEncoder(pretrained=True, img_size=256)
    
#     # Test forward pass
#     x = torch.randn(2, 3, 256, 256)
#     features = model(x)
    
#     print(f"\nInput shape: {x.shape}")
#     print(f"Number of output stages: {len(features)}")
    
#     for i, feat in enumerate(features):
#         print(f"  Stage {i+2}: {feat.shape}")
    
#     print("\n SwinEncoder test passed!")