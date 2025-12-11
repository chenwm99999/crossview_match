"""Enhanced cross-view model with multi-scale fusion and attention pooling"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_encoder import SwinEncoder
from .l2l_attention import L2LCrossAttention
from .projection_heads import DualProjectionHead
from .multiscale_fusion import MultiScaleFusion
from .attention_pooling import AttentionPooling

class CrossViewModelV2(nn.Module):
    """
    Enhanced cross-view matching model
    
    Improvements over baseline:
    1. Multi-scale fusion (combines features from 3 Swin stages)
    2. Attention pooling (learns important spatial regions)
    3. Enhanced L2L with FFN
    """
    
    def __init__(self, pretrained=True, use_multiscale=True, use_attn_pool=True):
        super().__init__()
        
        self.use_multiscale = use_multiscale
        self.use_attn_pool = use_attn_pool
        
        # Separate encoders
        self.street_encoder = SwinEncoder(pretrained=pretrained, img_size=256)
        self.drone_encoder = SwinEncoder(pretrained=pretrained, img_size=256)
        
        # L2L at each stage
        self.l2l_stage2 = L2LCrossAttention(dim=192, num_heads=8)
        self.l2l_stage3 = L2LCrossAttention(dim=384, num_heads=8)
        self.l2l_stage4 = L2LCrossAttention(dim=768, num_heads=8)
        
        # Multi-scale fusion
        if use_multiscale:
            self.multiscale_fusion_street = MultiScaleFusion(
                dims=[192, 384, 768],
                output_dim=768
            )
            self.multiscale_fusion_drone = MultiScaleFusion(
                dims=[192, 384, 768],
                output_dim=768
            )
        
        # Attention pooling
        if use_attn_pool:
            self.attn_pool_street = AttentionPooling(dim=768)
            self.attn_pool_drone = AttentionPooling(dim=768)
        
        # Projection heads
        self.projection = DualProjectionHead(input_dim=768)
        
        print("  Enhanced CrossViewModel initialized")
        print(f"  Multi-scale fusion: {use_multiscale}")
        print(f"  Attention pooling: {use_attn_pool}")
    
    def forward(self, street_img, drone_img, return_both_embeddings=True, return_attn_maps=False):
        """
        Args:
            return_attn_maps: If True, also return attention maps for visualization
        """
        
        # Extract multi-scale features
        street_feats = self.street_encoder(street_img)
        drone_feats = self.drone_encoder(drone_img)
        
        # Apply L2L cross-attention at each stage
        street_feats[0], drone_feats[0] = self.l2l_stage2(street_feats[0], drone_feats[0])
        street_feats[1], drone_feats[1] = self.l2l_stage3(street_feats[1], drone_feats[1])
        street_feats[2], drone_feats[2] = self.l2l_stage4(street_feats[2], drone_feats[2])
        
        # Multi-scale fusion
        if self.use_multiscale:
            street_fused = self.multiscale_fusion_street(street_feats)  # [B, 8, 8, 768]
            drone_fused = self.multiscale_fusion_drone(drone_feats)
        else:
            # Just use stage 4
            street_fused = street_feats[2]
            drone_fused = drone_feats[2]
        
        # Attention pooling
        if self.use_attn_pool:
            street_pooled, street_attn = self.attn_pool_street(street_fused)
            drone_pooled, drone_attn = self.attn_pool_drone(drone_fused)
        else:
            # Global average pooling
            street_pooled = street_fused.mean(dim=[1, 2])
            drone_pooled = drone_fused.mean(dim=[1, 2])
            street_attn, drone_attn = None, None
        
        # Project to CLIP and DINOv2 spaces
        if return_both_embeddings:
            street_clip, street_dino = self.projection(street_pooled, return_both=True)
            drone_clip, drone_dino = self.projection(drone_pooled, return_both=True)
            
            if return_attn_maps:
                return (street_clip, street_dino, drone_clip, drone_dino, 
                       street_attn, drone_attn)
            return street_clip, street_dino, drone_clip, drone_dino
        else:
            street_clip = self.projection(street_pooled, return_both=False)
            drone_clip = self.projection(drone_pooled, return_both=False)
            
            if return_attn_maps:
                return street_clip, drone_clip, street_attn, drone_attn
            return street_clip, drone_clip