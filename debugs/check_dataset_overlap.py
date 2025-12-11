
import pickle

with open('models/drone_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Show first 20 building IDs in FAISS
print("First 20 buildings in FAISS index:")
for i in range(20):
    print(f"  Index {i}: Building {metadata[i]['building_id']} ({metadata[i]['split']})")

# Check if our query buildings are in FAISS
query_buildings = ['1030', '0900', '1099', '0846']
print("\nSearching for query buildings in FAISS:")
for qb in query_buildings:
    found = [i for i, m in enumerate(metadata) if m['building_id'] == qb]
    if found:
        print(f"  {qb}: Found at index {found[0]}")
    else:
        print(f"  {qb}: NOT FOUND!")