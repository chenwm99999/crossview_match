"""Inspect University-1652 dataset and create label mapping"""

import os
from pathlib import Path
from collections import defaultdict
import json

def inspect_university1652(data_root):
    """Inspect University-1652 dataset structure and create label mapping"""
    
    data_root = Path(data_root)
    print(f"Inspecting dataset at: {data_root}")
    print("=" * 60)
    
    # Check main directories
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        print("ERROR: train or test directory not found")
        return
    
    # Count images by type (check for both .jpg and .jpeg extensions)
    splits = {
        "train_drone": train_dir / "drone",
        "train_street": train_dir / "street",
        "train_satellite": train_dir / "satellite",
        "train_google": train_dir / "google",
        "test_query_drone": test_dir / "query_drone",
        "test_query_street": test_dir / "query_street",
        "test_query_satellite": test_dir / "query_satellite",
        "test_gallery_drone": test_dir / "gallery_drone",
        "test_gallery_street": test_dir / "gallery_street",
        "test_gallery_satellite": test_dir / "gallery_satellite",
    }
    
    print("\nImage counts:")
    total_train = 0
    total_test = 0
    
    # Track buildings and their image counts
    building_stats = {}
    
    for split_name, split_path in splits.items():
        if split_path.exists():
            # Count all image files (jpg, jpeg, png)
            all_images = (list(split_path.glob("**/*.jpg")) + 
                         list(split_path.glob("**/*.jpeg")) + 
                         list(split_path.glob("**/*.png")))
            count = len(all_images)
            print(f"  {split_name:30s}: {count:6d} images")
            
            if 'train' in split_name:
                total_train += count
            else:
                total_test += count
            
            # For drone/street, count unique buildings and images per building
            if split_name in ['train_drone', 'train_street']:
                buildings = {}
                for img_path in all_images:
                    building_id = img_path.parent.name
                    if building_id not in buildings:
                        buildings[building_id] = []
                    buildings[building_id].append(img_path.name)
                
                building_stats[split_name] = buildings
                print(f"    → Unique buildings: {len(buildings)}")
                
                # Show sample statistics for multiple images per building
                if buildings:
                    image_counts = [len(imgs) for imgs in buildings.values()]
                    avg_images = sum(image_counts) / len(image_counts)
                    max_images = max(image_counts)
                    min_images = min(image_counts)
                    print(f"    → Images per building: avg={avg_images:.1f}, min={min_images}, max={max_images}")
                    
                    # Show a sample building
                    sample_building = list(buildings.keys())[0]
                    sample_images = buildings[sample_building]
                    print(f"    → Sample building {sample_building}: {len(sample_images)} images")
                    print(f"       First 3: {sample_images[:3]}")
        else:
            print(f"  {split_name:30s}: NOT FOUND")
    
    print(f"\n  Total training images: {total_train}")
    print(f"  Total test images: {total_test}")
    
    # Extract university and building information
    print("\n" + "=" * 60)
    print("Extracting label information...")
    
    university_map = defaultdict(set)
    building_to_university = {}
    building_names = {}
    
    # Parse from train/drone directory
    train_drone = train_dir / "drone"
    if train_drone.exists():
        for building_folder in train_drone.iterdir():
            if building_folder.is_dir():
                building_id = building_folder.name
                
                # Check if this folder has images
                images = (list(building_folder.glob("*.jpg")) + 
                         list(building_folder.glob("*.jpeg")) + 
                         list(building_folder.glob("*.png")))
                
                if images and building_id.isdigit() and len(building_id) == 4:
                    # University-1652 has 72 universities, ~23 buildings each
                    # Buildings 0001-0023 → University 0
                    # Buildings 0024-0046 → University 1, etc.
                    building_num = int(building_id)
                    university_id = (building_num - 1) // 23  # Approximate grouping
                    
                    building_to_university[building_id] = university_id
                    university_map[university_id].add(building_id)
                    building_names[building_id] = f"Building_{building_id}"
    
    num_universities = len(university_map)
    num_buildings = len(building_to_university)
    
    print(f"\nDataset statistics:")
    print(f"  Total universities: {num_universities}")
    print(f"  Total buildings: {num_buildings}")
    
    if num_universities > 0:
        print(f"  Avg buildings per university: {num_buildings/num_universities:.1f}")
    
    # Show sample universities
    if num_universities > 0:
        print("\nSample university IDs and their building counts:")
        for univ_id in sorted(university_map.keys())[:5]:
            buildings = university_map[univ_id]
            sample_buildings = sorted(list(buildings))[:3]
            print(f"  University {univ_id:02d}: {len(buildings)} buildings (e.g., {', '.join(sample_buildings)})")
    
    # Important note about multiple images
    print("\n" + "=" * 60)
    print("Multiple Images Per Building")
    print("=" * 60)
    if 'train_drone' in building_stats:
        drone_buildings = building_stats['train_drone']
        if drone_buildings:
            sample_id = list(drone_buildings.keys())[0]
            sample_count = len(drone_buildings[sample_id])
    
    # Create label mapping JSON
    if num_buildings > 0:
        label_file = data_root.parent / "university_labels.json"
        labels = {}
        
        # Load actual university names if available
        university_names = {i: f"University_{i:02d}" for i in range(72)}
        
        for building_id, univ_id in building_to_university.items():
            labels[building_id] = {
                "university_id": int(univ_id),
                "building_id": int(building_id),
                "university_name": university_names.get(univ_id, f"University_{univ_id:02d}"),
                "building_name": building_names.get(building_id, f"Building_{building_id}")
            }
        
        with open(label_file, 'w') as f:
            json.dump(labels, f, indent=2)
        
        print(f"\n Label mapping saved to: {label_file}")
        print(f"  Total entries: {len(labels)}")
    else:
        print("\n No buildings found - check dataset structure")
    
    # Check image sizes
    print("\n" + "=" * 60)
    print("Checking sample image dimensions...")
    
    # Check drone images
    sample_images = []
    if train_drone.exists():
        for building_folder in list(train_drone.iterdir())[:2]:
            if building_folder.is_dir():
                imgs = (list(building_folder.glob("*.jpg")) + 
                       list(building_folder.glob("*.jpeg")) + 
                       list(building_folder.glob("*.png")))
                if imgs:
                    sample_images.append(imgs[0])  # First image from building
    
    if sample_images:
        from PIL import Image
        for img_path in sample_images[:3]:
            try:
                img = Image.open(img_path)
                building_id = img_path.parent.name
                print(f"  {building_id}/{img_path.name}: {img.size} ({img.mode})")
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
    
    print("\n" + "=" * 60)
    print("Dataset inspection complete")

if __name__ == "__main__":
    # Allow command line argument or use default
    import sys
    if len(sys.argv) > 1:
        data_root = Path(sys.argv[1])
    else:
        data_root = Path(__file__).parent.parent / "data" / "University-Release"
    
    inspect_university1652(data_root)