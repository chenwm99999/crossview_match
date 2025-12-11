import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.dataset import CrossViewDataset

dataset = CrossViewDataset('data/University-Release', 'train', 'drone')

# Check actual building IDs
building_ids = sorted(dataset.building_ids)
print(f"Building IDs: {building_ids[:10]} ... {building_ids[-10:]}")
print(f"Total: {len(building_ids)}")

# Check one sample
_, _, bid, uid, bnum = dataset[0]
print(f"\nSample: building_id={bid}, university_id={uid}, building_num={bnum}")

# Extract all unique university IDs from building numbers
univ_ids_from_nums = set()
for i in range(len(dataset)):
    _, _, _, uid, bnum = dataset[i]
    univ_ids_from_nums.add(uid)

print(f"\nUnique university IDs: {sorted(univ_ids_from_nums)}")
print(f"Count: {len(univ_ids_from_nums)}")