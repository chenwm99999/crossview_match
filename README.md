# Cross-View Geo-Localization with Hierarchical Classification

## Introduction

This project implements a two-phase deep learning system for cross-view geo-localization on the University-1652 dataset. **Phase 1** uses a Swin Transformer with Layer-to-Layer cross-attention and dual-teacher knowledge distillation (CLIP + DINOv2) to match street-view images to aerial drone views, achieving 76.89% Recall@1. **Phase 2** builds a hierarchical classifier on top of Phase 1 embeddings to predict university locations from street images. The system demonstrates modern techniques including multi-scale feature fusion, attention-based pooling, and mixed-precision training.

---

## Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda create -n dl_crossview python=3.10 -y
conda activate dl_crossview

# Install PyTorch and FAISS
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install faiss-cpu -c pytorch -y

# Install other dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

**Download University-1652 dataset:**

1. Visit: https://github.com/layumi/University1652-Baseline?tab=readme-ov-file#about-dataset
2. Download from Google Drive link (select "University-Release" version)
3. Extract the zip file
4. Place extracted folder in `data/` directory

**Correct structure:**
```
data/University-Release/
├── train/
│   ├── drone/
│   └── street/
└── test/
    ├── query_drone/
    └── query_street/
```

**Verify dataset:**
```bash
python scripts/inspect_dataset.py
```

### 3. Pre-trained Models

**For university1652-model:**
1. Download from: https://github.com/layumi/University1652-Baseline?tab=readme-ov-file#trained-model
2. Extract resnet50_uni1652.pth into ./pretrained/university1652_baseline
3. Refer to ./pretrained/university1652_baseline/README for more details

**For L2LTR pretrained model:**
1. Download from: https://github.com/yanghongji2007/cross_view_localization_L2LTR/tree/main
2. Extract cvusa model into ./pretrained/l2ltr
3. Rename it as l2ltr_cvusa.pth
3. Refer to ./pretrained/l2ltr/README for more details

**For our best pretrained checkpoint:**
1. Download from: https://drive.google.com/file/d/14ksep7WE2z57EsHUKdvOZJpgnPyFSmr8/view?usp=sharing
2. Extract phase1_v2_best.pth and phase2_best.pth into ./checkpoints
3. Extract drone_index.faiss, drone_metadata.pkl and drone_paths.pkl into ./models
4. Refer to ./models/README for more details

### 4. Training

**Phase 1 (Cross-View Matching):**
```bash
python train_phase1_v2.py
```
- Expected time: ~2 hours (100 epochs on RTX 3090)
- Expected result: ~77% Recall@1
- Saves: `checkpoints/phase1_v2_best.pth`

**Phase 2 (University Classification):**
```bash
python train_phase2.py
```
- Expected time: ~40 minutes (40 epochs)
- Expected result: ~10% University Top-1 accuracy
- Saves: `checkpoints/phase2_university_best.pth`

### 5. Evaluation
```bash
python evaluate.py
```

Generates results in `results/final_evaluation/`:
- Metrics summary (JSON)
- Recall curve plot
- Confusion matrix
- Retrieval examples

### 6. App Usage
```bash
# start backend server
cd backend
python app.py

# In a new terminal, start frontend web endpoint
cd frontend
python -m http.server 3000
```
Open http://localhost:3000

---

## Project Structure
```
crossview_match/
├── src/
│   ├── models/
│   │   ├── crossview_model_v2.py         # Phase 1 model (Swin + L2L + dual teachers)
│   │   ├── hierarchical_classifier.py    # Phase 2 classifier
│   │   ├── swin_encoder.py               # Swin-Tiny backbone
│   │   ├── l2l_attention.py              # Layer-to-layer cross-attention
│   │   ├── multiscale_fusion.py          # Multi-scale feature fusion
│   │   ├── attention_pooling.py          # Learnable attention pooling
│   │   └── teachers.py                   # CLIP and DINOv2 teachers
│   ├── dataset.py                        # University-1652 dataset loader
│   └── losses.py                         # Triplet + distillation losses
│
├── train_phase1_v2.py                   # Phase 1 training script
├── train_phase2.py                      # Phase 2 training script
├── evaluate.py                          # Final evaluation
│
├── configs/
│   ├── phase1_v2.yaml                  # Phase 1 hyperparameters
│   └── phase2.yaml                     # Phase 2 hyperparameters
│
├── checkpoints/                        # Saved model weights
├── results/                            # Training curves and visualizations
├── data/                              # Dataset location
└── README.md                          # This file
```

---

## Key Files

| File | Purpose |
|------|---------|
| `train_phase1_v2.py` | Trains cross-view matching model with dual teachers |
| `train_phase2.py` | Trains university classifier on Phase 1 embeddings |
| `evaluate.py` | Generates metrics and visualizations for paper |
| `src/models/crossview_model_v2.py` | Main model architecture |
| `src/losses.py` | Triplet loss + CLIP/DINOv2 distillation |
| `configs/phase1_v2.yaml` | Training hyperparameters |

---

## Results

**Phase 1 (Cross-View Matching):**
- Recall@1: 76.89%
- Recall@5: 98.00%
- Recall@10: 99.14%

**Phase 2 (University Classification):**
- University Top-1: ~10%
- University Top-3: ~25%

**Dataset:** 701 buildings from 44 universities

---

## Requirements

- Python 3.10+
- CUDA 11.8+
- 16GB+ GPU memory
- ~10GB disk space (dataset + models)

---

## Citation

If you use this code, please cite:
- University-1652 dataset: Zheng et al., 2020
- Swin Transformer: Liu et al., 2021
- L2LTR: Yang et al., 2021
- CLIP: Radford et al., 2021
- DINOv2: Oquab et al., 2023