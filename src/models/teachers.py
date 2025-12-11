"""Frozen teacher models for knowledge distillation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class FrozenTeachers(nn.Module):
    """
    Manages frozen CLIP and DINOv2 teacher models
    
    Both teachers provide semantic/visual guidance but are NOT trained.
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        
        self.device = device
        
        # Load CLIP ViT-B/32
        print("Loading CLIP ViT-B/32...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        print("CLIP loaded and frozen")
        
        # Load DINOv2 ViT-S/14
        print("Loading DINOv2 ViT-S/14...")
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dinov2_model = self.dinov2_model.to(device)
        self.dinov2_model.eval()
        
        # Freeze DINOv2
        for param in self.dinov2_model.parameters():
            param.requires_grad = False
        
        print("DINOv2 loaded and frozen")
        
        print(f"\n FrozenTeachers initialized")
        print(f"  CLIP output: 512-dim")
        print(f"  DINOv2 output: 384-dim")
        print(f"  Device: {device}")
    
    @torch.no_grad()
    def encode_clip(self, images):
        """
        Encode images with CLIP
        
        Args:
            images: [B, 3, 256, 256]
        Returns:
            clip_features: [B, 512] (L2-normalized)
        """
        # CLIP expects 224x224
        images_224 = F.interpolate(images, size=224, mode='bilinear', align_corners=False)
        
        # Encode
        clip_features = self.clip_model.encode_image(images_224)
        
        # Normalize
        clip_features = F.normalize(clip_features.float(), p=2, dim=1)
        
        return clip_features
    
    @torch.no_grad()
    def encode_dinov2(self, images):
        """
        Encode images with DINOv2
        
        Args:
            images: [B, 3, 256, 256] (will be resized to 224x224)
        Returns:
            dino_features: [B, 384] (L2-normalized)
        """
        # DINOv2 expects 224x224
        images_224 = F.interpolate(images, size=224, mode='bilinear', align_corners=False)
        
        # Encode
        dino_features = self.dinov2_model(images_224)
        
        # Normalize
        dino_features = F.normalize(dino_features, p=2, dim=1)
        
        return dino_features
    
    @torch.no_grad()
    def forward(self, street_img, drone_img):
        """
        Encode both street and drone images with both teachers
        
        Args:
            street_img: [B, 3, 256, 256]
            drone_img: [B, 3, 256, 256]
        
        Returns:
            Dictionary with all teacher features
        """
        # CLIP features
        clip_street = self.encode_clip(street_img)
        clip_drone = self.encode_clip(drone_img)
        
        # DINOv2 features
        dino_street = self.encode_dinov2(street_img)
        dino_drone = self.encode_dinov2(drone_img)
        
        return {
            'clip_street': clip_street,
            'clip_drone': clip_drone,
            'dino_street': dino_street,
            'dino_drone': dino_drone
        }


# Test, comment out when release
# if __name__ == "__main__":
#     print("Testing FrozenTeachers")
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
    
#     # Load teachers
#     teachers = FrozenTeachers(device=device)
    
#     # Test images
#     B = 2
#     street_img = torch.randn(B, 3, 256, 256).to(device)
#     drone_img = torch.randn(B, 3, 256, 256).to(device)
    
#     print(f"\nInput shapes:")
#     print(f"  Street: {street_img.shape}")
#     print(f"  Drone: {drone_img.shape}")
    
#     # Encode
#     teacher_features = teachers(street_img, drone_img)
    
#     print(f"\nTeacher outputs:")
#     for key, value in teacher_features.items():
#         print(f"  {key}: {value.shape}")
        
#         # Verify normalization
#         norms = torch.norm(value, p=2, dim=1)
#         print(f"    L2 norms: {norms.cpu()}")
    
#     print("\n FrozenTeachers test passed!")