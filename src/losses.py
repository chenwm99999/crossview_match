"""Loss functions for cross-view matching"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """Triplet loss using cosine similarity (for normalized embeddings)"""
    
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Use cosine similarity instead of L2 distance
        Since embeddings are L2-normalized, cosine is more appropriate
        """
        # Cosine similarity (for normalized embeddings)
        # sim = a · b / (||a|| ||b||) = a · b (when normalized)
        sim_pos = (anchor * positive).sum(dim=1)  # [B]
        sim_neg = (anchor * negative).sum(dim=1)  # [B]
        
        # Triplet loss: maximize (sim_pos - sim_neg)
        # loss = max(0, margin - (sim_pos - sim_neg))
        loss = F.relu(self.margin - (sim_pos - sim_neg))
        
        return loss.mean()


class DistillationLoss(nn.Module):
    """
    MSE distillation loss between student and teacher embeddings
    Both must be L2-normalized before computing MSE
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, student_emb, teacher_emb):
        """
        Args:
            student_emb: [B, D] - L2-normalized student embeddings
            teacher_emb: [B, D] - L2-normalized teacher embeddings
        
        Returns:
            loss: scalar
        """
        return F.mse_loss(student_emb, teacher_emb)


class CombinedLoss(nn.Module):
    """Combined loss with margin-based triplet"""
    
    def __init__(self, lambda_clip=0.8, lambda_dino=0.6, triplet_margin=0.3):
        super().__init__()
        
        self.lambda_clip = lambda_clip
        self.lambda_dino = lambda_dino
        
        self.triplet_loss = TripletLoss(margin=triplet_margin)  # Use margin
        self.distill_loss = DistillationLoss()
        
        print(f"  CombinedLoss initialized")
        print(f"  λ_clip: {lambda_clip}")
        print(f"  λ_dino: {lambda_dino}")
        print(f"  Triplet margin: {triplet_margin}")
    
    def forward(self, student_outputs, teacher_outputs):
        """
        Args:
            student_outputs: dict with keys:
                - 'street_clip': [B, 512]
                - 'street_dino': [B, 384]
                - 'drone_clip': [B, 512]
                - 'drone_dino': [B, 384]
                - 'drone_neg_clip': [B, 512] (negative sample)
            
            teacher_outputs: dict with keys:
                - 'clip_street': [B, 512]
                - 'clip_drone': [B, 512]
                - 'dino_street': [B, 384]
                - 'dino_drone': [B, 384]
        
        Returns:
            loss_dict: Dictionary with all loss components
        """
        
        # Triplet loss use CLIP embeddings for matching
        loss_triplet = self.triplet_loss(
            anchor=student_outputs['street_clip'],
            positive=student_outputs['drone_clip'],
            negative=student_outputs['drone_neg_clip']
        )
        
        # CLIP distillation loss
        loss_clip_street = self.distill_loss(
            student_outputs['street_clip'],
            teacher_outputs['clip_street']
        )
        loss_clip_drone = self.distill_loss(
            student_outputs['drone_clip'],
            teacher_outputs['clip_drone']
        )
        loss_clip = loss_clip_street + loss_clip_drone
        
        # DINOv2 distillation loss
        loss_dino_street = self.distill_loss(
            student_outputs['street_dino'],
            teacher_outputs['dino_street']
        )
        loss_dino_drone = self.distill_loss(
            student_outputs['drone_dino'],
            teacher_outputs['dino_drone']
        )
        loss_dino = loss_dino_street + loss_dino_drone
        
        # Combined loss
        loss_total = loss_triplet + self.lambda_clip * loss_clip + self.lambda_dino * loss_dino
        
        return {
            'total': loss_total,
            'triplet': loss_triplet,
            'clip': loss_clip,
            'dino': loss_dino
        }


# Test, comment out when release
# if __name__ == "__main__":
#     print("Testing Loss Functions...")
    
#     # Create loss function
#     criterion = CombinedLoss(lambda_clip=0.5, lambda_dino=0.3, alpha=10.0)
    
#     # Create dummy embeddings WITH gradients enabled
#     B = 4
    
#     student_outputs = {
#         'street_clip': F.normalize(torch.randn(B, 512, requires_grad=True), p=2, dim=1),
#         'street_dino': F.normalize(torch.randn(B, 384, requires_grad=True), p=2, dim=1),
#         'drone_clip': F.normalize(torch.randn(B, 512, requires_grad=True), p=2, dim=1),
#         'drone_dino': F.normalize(torch.randn(B, 384, requires_grad=True), p=2, dim=1),
#         'drone_neg_clip': F.normalize(torch.randn(B, 512, requires_grad=True), p=2, dim=1),
#     }
    
#     teacher_outputs = {
#         'clip_street': F.normalize(torch.randn(B, 512), p=2, dim=1),  # No grad needed for teacher
#         'clip_drone': F.normalize(torch.randn(B, 512), p=2, dim=1),
#         'dino_street': F.normalize(torch.randn(B, 384), p=2, dim=1),
#         'dino_drone': F.normalize(torch.randn(B, 384), p=2, dim=1),
#     }
    
#     # Compute losses
#     losses = criterion(student_outputs, teacher_outputs)
    
#     print(f"\nLoss values:")
#     for key, value in losses.items():
#         print(f"  {key:10s}: {value.item():.4f}")
    
#     # Verify backprop works
#     losses['total'].backward()
#     print(f"\nBackward pass successful")
    
#     print("\nLoss functions test passed")