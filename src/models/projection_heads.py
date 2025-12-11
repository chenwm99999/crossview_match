"""Projection heads for multi-teacher distillation"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DualProjectionHead(nn.Module):
    """
    Dual projection heads for CLIP and DINOv2 distillation
    
    Takes Swin final features (384-dim) and projects to:
    - 512-dim for CLIP alignment
    - 384-dim for DINOv2 alignment
    """
    
    def __init__(self, input_dim=384, dropout=0.1):
        super().__init__()
        
        # CLIP projection head (384 → 512)
        self.clip_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512)
        )
        
        # DINOv2 projection head (384 → 384)
        self.dino_proj = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(384, 384)
        )
        
        print(f"DualProjectionHead created")
        print(f"Input: {input_dim}-dim")
        print(f" CLIP output: 512-dim")
        print(f" DINOv2 output: 384-dim")
    
    def forward(self, features, return_both=True):
        """
        Args:
            features: [B, 384] (pooled Swin features)
            return_both: If True, return both projections; if False, only CLIP
        
        Returns:
            If return_both=True: (clip_emb, dino_emb)
            If return_both=False: clip_emb only
        """
        
        # Project to CLIP space
        clip_emb = self.clip_proj(features)
        clip_emb = F.normalize(clip_emb, p=2, dim=1)  # L2 normalize
        
        if return_both:
            # Project to DINOv2 space
            dino_emb = self.dino_proj(features)
            dino_emb = F.normalize(dino_emb, p=2, dim=1)  # L2 normalize
            
            return clip_emb, dino_emb
        else:
            return clip_emb


# Test, comment out when release
# if __name__ == "__main__":
#     print("Testing DualProjectionHead")
    
#     proj_head = DualProjectionHead(input_dim=384)
    
#     # Test input
#     B = 4
#     features = torch.randn(B, 384)
    
#     print(f"\nInput shape: {features.shape}")
    
#     # Test both outputs
#     clip_emb, dino_emb = proj_head(features, return_both=True)
    
#     print(f"CLIP embedding: {clip_emb.shape}")
#     print(f"DINOv2 embedding: {dino_emb.shape}")
    
#     # Verify normalization
#     clip_norms = torch.norm(clip_emb, p=2, dim=1)
#     dino_norms = torch.norm(dino_emb, p=2, dim=1)
    
#     print(f"\nCLIP L2 norms: {clip_norms}")
#     print(f"DINOv2 L2 norms: {dino_norms}")
    
#     assert torch.allclose(clip_norms, torch.ones(B), atol=1e-5), "CLIP not normalized"
#     assert torch.allclose(dino_norms, torch.ones(B), atol=1e-5), "DINOv2 not normalized"
    
#     print("\n Embeddings are properly L2-normalized")
    
#     # Test CLIP-only mode
#     clip_only = proj_head(features, return_both=False)
#     print(f"\nCLIP-only mode: {clip_only.shape}")
    
#     print("\nDualProjectionHead test passed!")