import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""Debug model to find the issue"""

import torch
from src.models.crossview_model import CrossViewModel

model = CrossViewModel(pretrained=False)
model.eval()

# Test input
street = torch.randn(2, 3, 256, 256)
drone = torch.randn(2, 3, 256, 256)

print("Testing forward pass...")

# Get intermediate features
street_feats = model.street_encoder(street)
print(f"\nSwin outputs:")
for i, f in enumerate(street_feats):
    print(f"  Stage {i+2}: {f.shape}")

# After L2L
street_feats[0], drone_feats_0 = model.l2l_stage2(street_feats[0], model.drone_encoder(drone)[0])
print(f"\nAfter L2L stage 2: {street_feats[0].shape}")

# After pooling
pooled = street_feats[2].mean(dim=[1, 2])
print(f"\nAfter pooling: {pooled.shape}")

# After projection
street_clip, street_dino = model.projection(pooled, return_both=True)
print(f"\nFinal embeddings:")
print(f"  CLIP: {street_clip.shape}")
print(f"  DINOv2: {street_dino.shape}")

# Check normalization
clip_norms = torch.norm(street_clip, p=2, dim=1)
print(f"\nCLIP norms: {clip_norms}")

# Full forward
out = model(street, drone, return_both_embeddings=False)
print(f"\nFull model output: {out[0].shape}, {out[1].shape}")

# Test similarity computation
similarity = out[0] @ out[1].t()
print(f"\nSimilarity matrix: {similarity.shape}")
print(f"Similarity values: min={similarity.min():.3f}, max={similarity.max():.3f}")