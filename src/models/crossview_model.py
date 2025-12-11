"""Complete cross-view matching model with Swin + L2L + Dual heads"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_encoder import SwinEncoder
from .l2l_attention import L2LCrossAttention
from .projection_heads import DualProjectionHead

class CrossViewModel(nn.Module):
    """
    Complete cross-view matching model
    
    Architecture:
        Input: Street (256x256) + Drone (256x256)
                        ↓
        Swin-Tiny Encoders (separate for street/drone)
                        ↓
        L2L Cross-Attention at stages 2, 3, 4
                        ↓
        Global Average Pooling
                            ↓
        Dual Projection Heads (CLIP 512-dim + DINOv2 384-dim)
                        ↓
        Output: Normalized embeddings
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Separate encoders for street and drone
        self.street_encoder = SwinEncoder(pretrained=pretrained, img_size=256)
        self.drone_encoder = SwinEncoder(pretrained=pretrained, img_size=256)
        
        # L2L cross-attention at each stage
        # Stage 2: 192 channels, Stage 3: 384 channels, Stage 4: 768 channels
        self.l2l_stage2 = L2LCrossAttention(dim=192, num_heads=8)
        self.l2l_stage3 = L2LCrossAttention(dim=384, num_heads=8)
        self.l2l_stage4 = L2LCrossAttention(dim=768, num_heads=8)
        
        # Projection heads (input from stage 4 after pooling)
        self.projection = DualProjectionHead(input_dim=768)
        
        print("CrossViewModel initialized")
    
    def forward(self, street_img, drone_img, return_both_embeddings=True):
        """
        Args:
            street_img: [B, 3, 256, 256]
            drone_img: [B, 3, 256, 256]
            return_both_embeddings: Return both CLIP and DINOv2 embeddings
        
        Returns:
            If return_both_embeddings=True:
                street_clip, street_dino, drone_clip, drone_dino
            Else:
                street_clip, drone_clip (512-dim only, for inference)
        """
        
        # Extract multi-scale features
        street_feats = self.street_encoder(street_img)  # [stage2, stage3, stage4]
        drone_feats = self.drone_encoder(drone_img)
        
        # Apply L2L cross-attention at each stage
        street_feats[0], drone_feats[0] = self.l2l_stage2(street_feats[0], drone_feats[0])
        street_feats[1], drone_feats[1] = self.l2l_stage3(street_feats[1], drone_feats[1])
        street_feats[2], drone_feats[2] = self.l2l_stage4(street_feats[2], drone_feats[2])
        
        # Global average pooling on final stage (stage 4)
        # Input: [B, H, W, C] → Output: [B, C]
        street_pooled = street_feats[2].mean(dim=[1, 2])  # [B, 768]
        drone_pooled = drone_feats[2].mean(dim=[1, 2])    # [B, 768]
        
        # Project to CLIP and DINOv2 spaces
        if return_both_embeddings:
            street_clip, street_dino = self.projection(street_pooled, return_both=True)
            drone_clip, drone_dino = self.projection(drone_pooled, return_both=True)
            return street_clip, street_dino, drone_clip, drone_dino
        else:
            street_clip = self.projection(street_pooled, return_both=False)
            drone_clip = self.projection(drone_pooled, return_both=False)
            return street_clip, drone_clip


# Test, comment out when release
# if __name__ == "__main__":
#     print("Testing CrossViewModel")
    
#     model = CrossViewModel(pretrained=False)  # Use False for quick test
    
#     # Test forward pass
#     B = 2
#     street_img = torch.randn(B, 3, 256, 256)
#     drone_img = torch.randn(B, 3, 256, 256)
    
#     print(f"\nInput shapes:")
#     print(f"  Street: {street_img.shape}")
#     print(f"  Drone: {drone_img.shape}")
    
#     # Test with both embeddings
#     print(f"\nTesting with both embeddings (training mode)")
#     street_clip, street_dino, drone_clip, drone_dino = model(
#         street_img, drone_img, return_both_embeddings=True
#     )
    
#     print(f"  Street CLIP: {street_clip.shape}")
#     print(f"  Street DINOv2: {street_dino.shape}")
#     print(f"  Drone CLIP: {drone_clip.shape}")
#     print(f"  Drone DINOv2: {drone_dino.shape}")
    
#     # Test CLIP-only (inference mode)
#     print(f"\nTesting CLIP-only (inference mode)")
#     street_clip, drone_clip = model(
#         street_img, drone_img, return_both_embeddings=False
#     )
    
#     print(f"  Street CLIP: {street_clip.shape}")
#     print(f"  Drone CLIP: {drone_clip.shape}")
    
#     # Verify normalization
#     norms = torch.norm(street_clip, p=2, dim=1)
#     print(f"\nL2 norms: {norms}")
#     assert torch.allclose(norms, torch.ones(B), atol=1e-5), "Not normalized"
    
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    
#     print("\nCrossViewModel test passed")