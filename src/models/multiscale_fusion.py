"""Multi-scale feature fusion for cross-view matching"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFusion(nn.Module):
    """
    Fuse features from multiple Swin stages
    
    Motivation: Different scales capture different information
    - Stage 2 (32 32): Global layout, building arrangement
    - Stage 3 (16 16): Mid-level features, architectural style
    - Stage 4 (8 8): Fine details, textures
    
    Strategy: Project all to common dimension, weight and combine
    """
    
    def __init__(self, dims=[192, 384, 768], output_dim=768):
        super().__init__()
        
        # Project each stage to output_dim
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, output_dim, kernel_size=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU()
            ) for dim in dims
        ])
        
        # Learnable fusion weights (initialized to uniform)
        self.fusion_weights = nn.Parameter(torch.ones(len(dims)) / len(dims))
        
        print(f"  MultiScaleFusion created")
        print(f"  Input dims: {dims}")
        print(f"  Output dim: {output_dim}")
    
    def forward(self, features):
        """
        Args:
            features: List of [stage2, stage3, stage4]
                stage2: [B, H2, W2, C2]
                stage3: [B, H3, W3, C3]
                stage4: [B, H4, W4, C4]
        
        Returns:
            fused: [B, H4, W4, output_dim]
        """
        
        # Get target spatial size from stage 4
        target_size = features[-1].shape[1:3]  # (H4, W4)
        
        # Project and upsample all stages
        projected = []
        for feat, proj in zip(features, self.projs):
            # [B, H, W, C] → [B, C, H, W] for Conv2d
            feat = feat.permute(0, 3, 1, 2)
            
            # Project to common dimension
            feat_proj = proj(feat)
            
            # Upsample to target size
            if feat_proj.shape[-2:] != target_size:
                feat_proj = F.interpolate(
                    feat_proj,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            
            projected.append(feat_proj)
        
        # Stack and weight
        stacked = torch.stack(projected, dim=0)  # [3, B, C, H, W]
        
        # Apply learnable weights
        weights = F.softmax(self.fusion_weights, dim=0).view(-1, 1, 1, 1, 1)
        fused = (stacked * weights).sum(dim=0)  # [B, C, H, W]
        
        # Back to [B, H, W, C]
        fused = fused.permute(0, 2, 3, 1)
        
        return fused


# Test, comment out when release
# if __name__ == "__main__":
#     print("Testing MultiScaleFusion..")
    
#     fusion = MultiScaleFusion(dims=[192, 384, 768], output_dim=768)
    
#     # Create dummy multi-scale features
#     B = 2
#     stage2 = torch.randn(B, 32, 32, 192)
#     stage3 = torch.randn(B, 16, 16, 384)
#     stage4 = torch.randn(B, 8, 8, 768)
    
#     features = [stage2, stage3, stage4]
    
#     fused = fusion(features)
    
#     print(f"\nInput shapes:")
#     for i, f in enumerate(features):
#         print(f"  Stage {i+2}: {f.shape}")
    
#     print(f"\nOutput shape: {fused.shape}")
#     print(f"Fusion weights: {F.softmax(fusion.fusion_weights, dim=0)}")
    
#     print("\n MultiScaleFusion test passed")