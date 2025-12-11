"""Attention-based pooling for global feature aggregation"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    """
    Learnable attention pooling - model learns which spatial regions are important
    
    Better than global average pooling because:
    - Focuses on discriminative regions (building centers, not sky)
    - Provides interpretable attention maps
    - Adaptive to each image
    """
    
    def __init__(self, dim):
        super().__init__()
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.ReLU(),
            nn.Conv2d(dim // 8, 1, kernel_size=1),
            nn.Sigmoid()  # Output: [0, 1] attention weights
        )
        
        print(f" AttentionPooling created (dim={dim})")
    
    def forward(self, x):
        """
        Args:
            x: [B, H, W, C]
        
        Returns:
            pooled: [B, C]
            attn_map: [B, 1, H, W] (for visualization)
        """
        
        # [B, H, W, C] → [B, C, H, W] for conv
        x_conv = x.permute(0, 3, 1, 2)
        
        # Compute attention map
        attn_map = self.attention(x_conv)  # [B, 1, H, W]
        
        # Apply attention weights
        weighted = x_conv * attn_map  # Broadcast multiplication
        
        # Normalize by sum of attention weighted average
        pooled = weighted.sum(dim=[2, 3]) / (attn_map.sum(dim=[2, 3]) + 1e-6)
        
        return pooled, attn_map


# Test, comment out when release
# if __name__ == "__main__":
#     print("Testing AttentionPooling..")
    
#     pool = AttentionPooling(dim=768)
    
#     # Test input
#     B = 2
#     x = torch.randn(B, 8, 8, 768)
    
#     pooled, attn_map = pool(x)
    
#     print(f"\nInput: {x.shape}")
#     print(f"Output: {pooled.shape}")
#     print(f"Attention map: {attn_map.shape}")
    
#     # Check attention values
#     print(f"\nAttention statistics:")
#     print(f"  Min: {attn_map.min().item():.4f}")
#     print(f"  Max: {attn_map.max().item():.4f}")
#     print(f"  Mean: {attn_map.mean().item():.4f}")
    
#     print("\n AttentionPooling test passed")