"""Test dataset loading with proper drone image handling"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml
import random

class University1652Dataset(Dataset):
    """Dataset with proper handling of multiple drone images per building"""
    
    def __init__(self, root, split='train', mode='drone', use_first_image=True):
        """
        Args:
            root: Dataset root directory
            split: 'train' or 'test'
            mode: 'drone' or 'satellite' for aerial view
            use_first_image: If True, use first image per building; 
                           If False, randomly sample from available images
        """
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.use_first_image = use_first_image
        
        print(f"\nInitializing {split} dataset...")
        print(f"Mode: street ↔ {mode}")
        print(f"Image selection: {'First image' if use_first_image else 'Random sample'}")
        
        # Set up directories
        if split == 'train':
            self.street_dir = self.root / 'train' / 'street'
            self.aerial_dir = self.root / 'train' / mode  # 'drone' or 'satellite'
        else:  # test
            self.street_dir = self.root / 'test' / 'query_street'
            self.aerial_dir = self.root / 'test' / f'query_{mode}'
        
        # Get building folders
        self.building_ids = []
        self.street_imgs = {}
        self.aerial_imgs = {}
        
        # Scan street directory
        if self.street_dir.exists():
            for building_folder in sorted(self.street_dir.iterdir()):
                if building_folder.is_dir():
                    building_id = building_folder.name
                    
                    # Get street images for this building
                    street_images = list(building_folder.glob('*.jpg')) + \
                                  list(building_folder.glob('*.jpeg')) + \
                                  list(building_folder.glob('*.png'))
                    
                    if street_images:
                        self.street_imgs[building_id] = street_images
        
        # Scan aerial directory (drone or satellite)
        if self.aerial_dir.exists():
            for building_folder in sorted(self.aerial_dir.iterdir()):
                if building_folder.is_dir():
                    building_id = building_folder.name
                    
                    # Get aerial images for this building
                    aerial_images = list(building_folder.glob('*.jpg')) + \
                                  list(building_folder.glob('*.jpeg')) + \
                                  list(building_folder.glob('*.png'))
                    
                    if aerial_images:
                        self.aerial_imgs[building_id] = aerial_images
        
        # Get buildings that have both street and aerial images
        self.building_ids = sorted(
            set(self.street_imgs.keys()) & set(self.aerial_imgs.keys())
        )
        
        print(f"Found {len(self.street_imgs)} buildings with street images")
        print(f"Found {len(self.aerial_imgs)} buildings with {mode} images")
        print(f"Buildings with both views: {len(self.building_ids)}")
        
        if len(self.building_ids) > 0:
            # Show sample statistics
            sample_id = self.building_ids[0]
            print(f"\nSample building {sample_id}:")
            print(f"  Street images: {len(self.street_imgs[sample_id])}")
            print(f"  {mode.capitalize()} images: {len(self.aerial_imgs[sample_id])}")
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.building_ids)
    
    def __getitem__(self, idx):
        building_id = self.building_ids[idx]
        
        # Get street image (pick first or random)
        street_images = self.street_imgs[building_id]
        if self.use_first_image:
            street_path = street_images[0]
        else:
            street_path = random.choice(street_images)
        
        # Get aerial image (pick first or random)
        aerial_images = self.aerial_imgs[building_id]
        if self.use_first_image:
            aerial_path = aerial_images[0]
        else:
            aerial_path = random.choice(aerial_images)
        
        # Load and transform
        try:
            street_img = Image.open(street_path).convert('RGB')
            aerial_img = Image.open(aerial_path).convert('RGB')
            
            street = self.transform(street_img)
            aerial = self.transform(aerial_img)
        except Exception as e:
            print(f"Error loading {building_id}: {e}")
            return torch.zeros(3, 256, 256), torch.zeros(3, 256, 256), building_id, 0, 0
        
        # Get labels
        try:
            building_num = int(building_id)
            university_id = (building_num - 1) // 23
        except:
            building_num = 0
            university_id = 0
        
        return street, aerial, building_id, university_id, building_num


def test_dataset():
    """Test dataset with both drone and satellite"""
    
    print("=" * 60)
    print("TESTING DATASET LOADING")
    print("=" * 60)
    
    data_root = "data/University-Release"
    
    # Test both modes
    for mode in ['drone', 'satellite']:
        print(f"\n{'=' * 60}")
        print(f"Testing: STREET ↔ {mode.upper()}")
        print(f"{'=' * 60}")
        
        try:
            # Create dataset
            dataset = University1652Dataset(
                root=data_root,
                split='train',
                mode=mode,
                use_first_image=True  # Use first image for consistency
            )
            
            if len(dataset) == 0:
                print(f"No valid pairs found for {mode}")
                continue
            
            # Create dataloader
            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0
            )
            
            # Test batch
            print(f"\nLoading one batch...")
            street, aerial, building_ids, univ_ids, build_nums = next(iter(loader))
            
            print(f"Street shape: {street.shape}")
            print(f"{mode.capitalize()} shape: {aerial.shape}")
            print(f"Building IDs: {building_ids}")
            print(f"Dataset length: {len(dataset)} pairs")
            
            # Show some statistics
            print(f"\n  Sample buildings:")
            for i in range(min(3, len(dataset))):
                bid = dataset.building_ids[i]
                n_street = len(dataset.street_imgs[bid])
                n_aerial = len(dataset.aerial_imgs[bid])
                print(f"    {bid}: {n_street} street, {n_aerial} {mode}")
            
            print(f"\n{mode.upper()} dataset works")
            
        except Exception as e:
            print(f"Error with {mode}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_dataset()