This folder contains generated FAISS indices and metadata for fast retrieval.

## Files Generated During Training

After Phase 1 training completes, the following files will be generated here:

| File | Size | Description |
|------|------|-------------|
| drone_index.faiss | ~100-500MB | FAISS index for drone image embeddings |
| drone_metadata.pkl | ~10-50MB | Metadata for drone images (IDs, labels) |
| drone_paths.pkl | ~5-20MB | File paths for drone images |

## Link

Download from https://drive.google.com/file/d/14ksep7WE2z57EsHUKdvOZJpgnPyFSmr8/view?usp=sharing
*OR GENERATE*

## How to Generate

These files are automatically created when you run:
``````bash
# After Phase 1 training completes
python src/training/build_faiss_index.py --checkpoint checkpoints/phase1_v2_best.pth
``````

Or they will be generated automatically during Phase 2 training setup.

## Do Not Track in Git

These files are **not tracked in Git** because:
- They are large (100MB+)
- They can be regenerated from your trained models
- They are environment-specific

## Regenerating Files

If you need to regenerate these files:

1. Make sure you have the Phase 1 checkpoint: `checkpoints/phase1_v2_best.pth`
2. Run the index building script (see above)
3. Wait 5-10 minutes for processing