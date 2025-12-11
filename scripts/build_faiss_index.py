"""Build FAISS index from Phase 1 trained model"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset import CrossViewDataset
from src.models.crossview_model_v2 import CrossViewModelV2

def build_index(checkpoint_path, data_root, output_dir='models'):
    """
    Build FAISS index for drone images
    
    Args:
        checkpoint_path: Path to Phase 1 best model
        data_root: Dataset root directory
        output_dir: Where to save index and metadata
    """
    
    print("=" * 60)
    print("BUILDING FAISS INDEX")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load trained model
    print(f"\n1. Loading model from: {checkpoint_path}")
    model = CrossViewModelV2(pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f" Model loaded (epoch {checkpoint['epoch']}, R@1: {checkpoint['best_recall']*100:.2f}%)")
    
    # Load dataset (use both train and test drone images)
    print("\n2. Loading drone images..")
    
    # Train drones
    train_dataset = CrossViewDataset(
        root=data_root,
        split='train',
        mode='drone',
        use_augmentation=False,
        use_first_image=True
    )
    
    # Test drones (gallery)
    test_dataset = CrossViewDataset(
        root=data_root,
        split='test',
        mode='drone',
        use_augmentation=False,
        use_first_image=True
    )
    
    print(f" Train drone images: {len(train_dataset)}")
    print(f" Test drone images: {len(test_dataset)}")
    
    # Extract embeddings
    print("\n3. Extracting drone embeddings..")
    
    all_embeddings = []
    all_metadata = []
    
    for dataset, split_name in [(train_dataset, 'train'), (test_dataset, 'test')]:
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        
        for _, drone_img, building_ids, univ_ids, build_nums in tqdm(loader, desc=f'{split_name} drones'):
            drone_img = drone_img.to(device)
            
            with torch.no_grad():
                # Get CLIP embeddings (512-dim for retrieval)
                _, drone_clip = model(drone_img, drone_img, return_both_embeddings=False)
            
            # Store embeddings and metadata
            all_embeddings.append(drone_clip.cpu().numpy())
            
            for bid, uid, bnum in zip(building_ids, univ_ids, build_nums):
                all_metadata.append({
                    'building_id': bid,
                    'university_id': uid.item(),
                    'building_num': bnum.item(),
                    'split': split_name
                })
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)  # [N, 512]
    print(f" Total embeddings: {all_embeddings.shape}")
    
    # Build FAISS index
    print("\n4. Building FAISS index..")
    
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    index.add(all_embeddings)
    
    print(f"  FAISS index created")
    print(f"  Type: IndexFlatIP (exact search)")
    print(f"  Dimension: {dim}")
    print(f"  Total vectors: {index.ntotal}")
    
    # Save index
    index_path = output_dir / 'drone_index.faiss'
    faiss.write_index(index, str(index_path))
    print(f" Saved index: {index_path}")
    
    # Save metadata
    metadata_path = output_dir / 'drone_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(all_metadata, f)
    print(f" Saved metadata: {metadata_path}")
    
    # Verify index works
    print("\n5. Verifying index...")
    test_query = all_embeddings[:1]  # First embedding
    similarities, indices = index.search(test_query, k=5)
    
    print(f" Search test:")
    print(f"  Top-1 similarity: {similarities[0][0]:.4f} (~1.0)")
    print(f"  Top-1 index: {indices[0][0]} (0)")
    
    if indices[0][0] == 0 and similarities[0][0] > 0.99:
        print("Index verification PASSED")
    else:
        print("Index might have issues")
    
    print("\n" + "=" * 60)
    print("INDEX BUILDING COMPLETE")
    print("=" * 60)
    print(f"Saved files:")
    print(f"  - {index_path}")
    print(f"  - {metadata_path}")
    print(f"Total database size: {len(all_metadata)} drone images")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/phase1_v2_best.pth')
    parser.add_argument('--data_root', type=str, default='data/University-Release')
    parser.add_argument('--output', type=str, default='models')
    
    args = parser.parse_args()
    
    build_index(args.checkpoint, args.data_root, args.output)