import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import faiss
import pickle
from src.models.crossview_model_v2 import CrossViewModelV2
from src.dataset import CrossViewDataset

# Load model
model = CrossViewModelV2(pretrained=False).cuda()
ckpt = torch.load('checkpoints/phase1_v2_epoch_100.pth')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load FAISS
index = faiss.read_index('models/drone_index.faiss')
with open('models/drone_paths.pkl', 'rb') as f:
    paths = pickle.load(f)

# Load dataset
dataset = CrossViewDataset('data/University-Release', 'test', 'drone', False, True)

# Test first sample
street_img, gt_drone, bid, uid, bnum = dataset[0]
street_img = street_img.unsqueeze(0).cuda()

# Extract embedding
with torch.no_grad():
    street_clip, _ = model(street_img, street_img, return_both_embeddings=False)

# Search FAISS
emb = street_clip.cpu().numpy()
sims, indices = index.search(emb, k=5)

print(f"Query building: {bid}")
print(f"Top-5 similarities: {sims[0]}")
print(f"Top-5 indices: {indices[0]}")
print(f"\nTop-5 retrieved paths:")
for i, idx in enumerate(indices[0][:5]):
    print(f"  {i+1}. {paths[idx]} (sim: {sims[0][i]:.3f})")

# Check if top-1 is correct
print(f"\nTop-1 correct? {bid in paths[indices[0][0]]}")