"""Layer-to-Layer Cross-Attention module"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class L2LCrossAttention(nn.Module):
    """
    Layer-to-Layer Cross-Attention between street and drone features
    
    Allows street branch to attend to drone branch and vice versa.
    Uses residual connections for stable training.
    
    Args:
        dim: Feature dimension (96, 192, or 384 for Swin stages)
        num_heads: Number of attention heads
    """
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Cross-attention: street queries drone
        self.cross_attn_s2d = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Cross-attention: drone queries street
        self.cross_attn_d2s = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization for stability
        self.norm_street = nn.LayerNorm(dim)
        self.norm_drone = nn.LayerNorm(dim)
        
        print(f" L2LCrossAttention created (dim={dim}, heads={num_heads})")
    
    def forward(self, street_feat, drone_feat):
        """
        Args:
            street_feat: [B, H, W, C] or [B, N, C]
            drone_feat: [B, H, W, C] or [B, N, C]
        
        Returns:
            street_out: [B, H, W, C] or [B, N, C]
            drone_out: [B, H, W, C] or [B, N, C]
        """
        
        # Handle different input formats
        original_shape = street_feat.shape
        
        if len(street_feat.shape) == 4:
            # Input is [B, H, W, C], need to flatten to [B, H*W, C]
            B, H, W, C = street_feat.shape
            street_flat = street_feat.reshape(B, H * W, C)
            drone_flat = drone_feat.reshape(B, H * W, C)
            need_reshape = True
        else:
            # Already [B, N, C]
            street_flat = street_feat
            drone_flat = drone_feat
            need_reshape = False
        
        # Cross-attention: street queries drone
        street_cross, _ = self.cross_attn_s2d(
            query=street_flat,
            key=drone_flat,
            value=drone_flat
        )
        
        # Residual connection + layer norm
        street_out = self.norm_street(street_flat + street_cross)
        
        # Cross-attention by drone queries street
        drone_cross, _ = self.cross_attn_d2s(
            query=drone_flat,
            key=street_flat,
            value=street_flat
        )
        
        # Residual connection + layer norm
        drone_out = self.norm_drone(drone_flat + drone_cross)
        
        # Reshape back to original format if needed
        if need_reshape:
            street_out = street_out.reshape(B, H, W, C)
            drone_out = drone_out.reshape(B, H, W, C)
        
        return street_out, drone_out


# Test, comment out when release
# if __name__ == "__main__":
#     print("Testing L2LCrossAttention...")
    
#     # Test with different dimensions
#     test_dims = [
#         (96, 32, 32),   # Stage 2
#         (192, 16, 16),  # Stage 3
#         (384, 8, 8)     # Stage 4
#     ]
    
#     for dim, H, W in test_dims:
#         print(f"\n--- Testing dim={dim}, spatial={H}x{W} ---")
        
#         l2l = L2LCrossAttention(dim=dim, num_heads=8)
        
#         # Create dummy features [B, H, W, C]
#         B = 2
#         street = torch.randn(B, H, W, dim)
#         drone = torch.randn(B, H, W, dim)
        
#         # Forward pass
#         street_out, drone_out = l2l(street, drone)
        
#         print(f"  Input:  {street.shape}")
#         print(f"  Output: {street_out.shape}")
#         assert street_out.shape == street.shape, "Shape mismatch!"
#         assert drone_out.shape == drone.shape, "Shape mismatch!"
        
#         print(f"  L2L test passed for dim={dim}")
    
#     print("\nL2LCrossAttention tests passed")