"""Complete dataset class with proper augmentation for cross-view matching"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CrossViewDataset(Dataset):
    """
    Cross-view dataset for street-drone matching with proper augmentation
    
    Args:
        root: Dataset root directory
        split: 'train' or 'test'
        mode: 'drone' or 'satellite' for aerial view
        use_augmentation: Apply data augmentation (True for training)
        use_first_image: Always use first image per building
    """
    
    def __init__(self, root, split='train', mode='drone', 
                 use_augmentation=True, use_first_image=True):
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.use_augmentation = use_augmentation and split == 'train'
        self.use_first_image = use_first_image
        
        print(f"\nInitializing CrossViewDataset:")
        print(f"  Split: {split}")
        print(f"  Mode: street ↔ {mode}")
        print(f"  Augmentation: {self.use_augmentation}")
        
        # Set up directories
        if split == 'train':
            self.street_dir = self.root / 'train' / 'street'
            self.aerial_dir = self.root / 'train' / mode
        else:  # test
            self.street_dir = self.root / 'test' / 'query_street'
            self.aerial_dir = self.root / 'test' / f'query_{mode}'
        
        # Collect image paths
        self.building_ids = []
        self.street_imgs = {}
        self.aerial_imgs = {}
        
        self._load_image_paths()
        
        # Setup transforms
        self._setup_transforms()
        
        print(f"  Buildings loaded: {len(self.building_ids)}")
        print(f"  Total pairs: {len(self)}")
    
    def _load_image_paths(self):
        """Load all image paths organized by building"""
        
        # Scan street directory
        if self.street_dir.exists():
            for building_folder in self.street_dir.iterdir():
                if building_folder.is_dir():
                    building_id = building_folder.name
                    images = (list(building_folder.glob('*.jpg')) + 
                            list(building_folder.glob('*.jpeg')) + 
                            list(building_folder.glob('*.png')))
                    if images:
                        self.street_imgs[building_id] = sorted(images)
        
        # Scan aerial directory
        if self.aerial_dir.exists():
            for building_folder in self.aerial_dir.iterdir():
                if building_folder.is_dir():
                    building_id = building_folder.name
                    images = (list(building_folder.glob('*.jpg')) + 
                            list(building_folder.glob('*.jpeg')) + 
                            list(building_folder.glob('*.png')))
                    if images:
                        self.aerial_imgs[building_id] = sorted(images)
        
        # Get buildings with both views
        self.building_ids = sorted(
            set(self.street_imgs.keys()) & set(self.aerial_imgs.keys())
        )
    
    def _setup_transforms(self):
        """Setup augmentation and normalization transforms"""
        
        # ImageNet normalization stats
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if self.use_augmentation:
            # Street augmentation (more aggressive)
            self.street_transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.8
                ),
                A.RandomResizedCrop(
                    256, 256,
                    scale=(0.9, 1.0),
                    ratio=(0.95, 1.05),
                    p=0.5
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
            
            # Drone augmentation (preserve spatial structure)
            self.aerial_transform = A.Compose([
                A.Resize(256, 256),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.8
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            # Test time: no augmentation
            transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
            self.street_transform = transform
            self.aerial_transform = transform
    
    def __len__(self):
        return len(self.building_ids)
    
    def __getitem__(self, idx):
        building_id = self.building_ids[idx]
        
        # Select images
        if self.use_first_image:
            street_path = self.street_imgs[building_id][0]
            aerial_path = self.aerial_imgs[building_id][0]
        else:
            street_path = random.choice(self.street_imgs[building_id])
            aerial_path = random.choice(self.aerial_imgs[building_id])
        
        # Load images
        try:
            street_img = Image.open(street_path).convert('RGB')
            aerial_img = Image.open(aerial_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {building_id}: {e}")
            return (torch.zeros(3, 256, 256), 
                torch.zeros(3, 256, 256),
                building_id, 0, 0)
        
        # Convert PIL to numpy for albumentations
        import numpy as np
        street_np = np.array(street_img)
        aerial_np = np.array(aerial_img)
        
        # Apply transforms
        street_img = self.street_transform(image=street_np)['image']
        aerial_img = self.aerial_transform(image=aerial_np)['image']
        
        # Get labels
        building_num = idx  # Use index (0 to 700)
        university_id = min(idx // 16, 42)  # Cap at 42 (43 classes: 0-42)

        return street_img, aerial_img, building_id, university_id, building_num


def test_augmentation():
    """Test augmentation pipeline"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("=" * 60)
    print("TESTING AUGMENTATION PIPELINE")
    print("=" * 60)
    
    # Create datasets
    train_dataset = CrossViewDataset(
        root="data/University-Release",
        split='train',
        mode='drone',
        use_augmentation=True
    )
    
    test_dataset = CrossViewDataset(
        root="data/University-Release",
        split='train',
        mode='drone',
        use_augmentation=False
    )
    
    # Get same image with and without augmentation
    idx = 0
    
    # Without augmentation
    street_orig, aerial_orig, bid, uid, bnum = test_dataset[idx]
    
    # With augmentation (get 3 different versions)
    augmented_samples = []
    for _ in range(3):
        street_aug, aerial_aug, _, _, _ = train_dataset[idx]
        augmented_samples.append((street_aug, aerial_aug))
    
    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def denormalize(img):
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        return img.permute(1, 2, 0).numpy()
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Augmentation Test - Building {bid}', fontsize=16)
    
    # Row 1: Street views
    axes[0, 0].imshow(denormalize(street_orig))
    axes[0, 0].set_title('Street (Original)')
    axes[0, 0].axis('off')
    
    for i, (street_aug, _) in enumerate(augmented_samples):
        axes[0, i+1].imshow(denormalize(street_aug))
        axes[0, i+1].set_title(f'Street (Aug {i+1})')
        axes[0, i+1].axis('off')
    
    # Row 2: Drone views
    axes[1, 0].imshow(denormalize(aerial_orig))
    axes[1, 0].set_title('Drone (Original)')
    axes[1, 0].axis('off')
    
    for i, (_, aerial_aug) in enumerate(augmented_samples):
        axes[1, i+1].imshow(denormalize(aerial_aug))
        axes[1, i+1].set_title(f'Drone (Aug {i+1})')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('logs/augmentation_test.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to: logs/augmentation_test.png")
    print("\nAugmentation test complete!")


if __name__ == "__main__":
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Test augmentation
    test_augmentation()