import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import torch.nn.functional as F

# Simulate matched pair (should have high similarity)
anchor = F.normalize(torch.randn(4, 512), p=2, dim=1)
positive = anchor + torch.randn(4, 512) * 0.1  # Similar to anchor
positive = F.normalize(positive, p=2, dim=1)

# Negative (random)
negative = F.normalize(torch.randn(4, 512), p=2, dim=1)

# Check similarities
sim_pos = (anchor * positive).sum(dim=1)
sim_neg = (anchor * negative).sum(dim=1)

print(f"Positive similarity: {sim_pos}")  # Should be ~0.9
print(f"Negative similarity: {sim_neg}")  # Should be ~0.2

# Compute loss
loss = F.relu(0.3 - (sim_pos - sim_neg))
print(f"Loss: {loss}")  # Should be small if working correctly