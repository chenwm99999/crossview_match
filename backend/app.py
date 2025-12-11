"""
Phase 3: Web Application Backend
FastAPI server for cross-view geo-localization inference
"""

import os
import sys
import torch
import faiss
import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import Dict

BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import models
try:
    from models.crossview_model_v2 import CrossViewModelV2
    print(" Models imported successfully")
except ImportError:
    try:
        from src.models.crossview_model_v2 import CrossViewModelV2
        print(" Models imported via src.models")
    except ImportError as e:
        print(f" Failed to import models: {e}")
        raise

# Configuration

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"
PHASE1_CHECKPOINT = CHECKPOINT_DIR / "phase1_v2_best.pth"
PHASE2_CHECKPOINT = CHECKPOINT_DIR / "phase2_best.pth"
FAISS_INDEX_PATH = MODELS_DIR / "drone_index.faiss"
DRONE_METADATA_PATH = MODELS_DIR / "drone_metadata.pkl"
DRONE_PATHS_PATH = MODELS_DIR / "drone_paths.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256

# University names mapping
UNIVERSITY_NAMES = {}

# Custom Model Loader

class UniversityOnlyClassifier(torch.nn.Module):
    """Matches your exact 3-layer architecture: 1024→512→256→44"""
    
    def __init__(self, encoder, num_universities=44, dropout=0.5):
        super().__init__()
        
        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.num_universities = num_universities
        self.num_buildings = None
        
        # Feature refinement
        self.refine = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(1024, 1024),
            torch.nn.LayerNorm(1024)
        )
        
        # University head - 3 layers: 1024 → 512 → 256 → num_universities
        # Layer 0-3: 1024 → 512
        # Layer 4-7: 512 → 256  
        # Layer 8: 256 → num_universities (final layer)
        self.university_head = torch.nn.Sequential(
            # Layer 0
            torch.nn.Linear(1024, 512),
            # Layer 1
            torch.nn.LayerNorm(512),
            # Layer 2
            torch.nn.GELU(),
            # Layer 3
            torch.nn.Dropout(dropout),
            
            # Layer 4
            torch.nn.Linear(512, 256),
            # Layer 5
            torch.nn.LayerNorm(256),
            # Layer 6
            torch.nn.GELU(),
            # Layer 7
            torch.nn.Dropout(dropout),
            
            # Layer 8 - Final output layer
            torch.nn.Linear(256, num_universities)
        )
        
        print(f" UniversityOnlyClassifier created")
        print(f"  Architecture: 1024 → 512 → 256 → {num_universities}")
        print(f"  Final layer index: 8")
    
    def forward(self, street_img, drone_img):
        with torch.no_grad():
            street_clip, drone_clip = self.encoder(
                street_img, drone_img, return_both_embeddings=False
            )
        
        combined = torch.cat([street_clip, drone_clip], dim=1)
        refined = self.refine(combined)
        refined = refined + combined
        
        univ_logits = self.university_head(refined)
        
        return univ_logits, None  # No building prediction

# Model Loading

class InferenceEngine:
    """Handles model loading and inference"""
    
    def __init__(self):
        self.device = DEVICE
        self.phase1_model = None
        self.phase2_model = None
        self.faiss_index = None
        self.drone_metadata = None
        self.drone_paths = None
        self.num_universities = None
        self.num_buildings = None
        self.has_building_prediction = False
        
        print(f"Using device: {self.device}")
        print(f"Project root: {PROJECT_ROOT}")
        self.load_models()
        self.load_faiss_index()
        self.setup_university_names()
    
    def detect_num_universities(self, state_dict):
        """Detect number of universities from the LAST layer of university_head"""
        # Find the highest numbered university_head layer
        univ_layers = {}
        for key in state_dict.keys():
            if 'university_head' in key and '.weight' in key:
                # Extract layer number (e.g., "university_head.8.weight" -> 8)
                try:
                    parts = key.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        layer_num = int(parts[1])
                        shape = state_dict[key].shape
                        if len(shape) == 2:  # Linear layer
                            univ_layers[layer_num] = shape[0]  # output dimension
                except:
                    continue
        
        if univ_layers:
            # Get the LAST layer (highest number)
            max_layer = max(univ_layers.keys())
            num_univ = univ_layers[max_layer]
            print(f" Found university_head layer {max_layer}: {num_univ} classes")
            print(f" Architecture from checkpoint: 3-layer (1024→512→256→{num_univ})")
            return num_univ
        
        return None
    
    def load_models(self):
        """Load Phase 1 and Phase 2 models"""
        print("Loading Phase 1 model...")
        
        # Load Phase 1
        self.phase1_model = CrossViewModelV2().to(self.device)
        
        if PHASE1_CHECKPOINT.exists():
            checkpoint = torch.load(PHASE1_CHECKPOINT, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.phase1_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.phase1_model.load_state_dict(checkpoint)
            print(f" Loaded Phase 1 from {PHASE1_CHECKPOINT}")
        else:
            raise FileNotFoundError(f"Phase 1 checkpoint not found")
        
        self.phase1_model.eval()
        
        print("\nLoading Phase 2 model...")
        
        # Load checkpoint and detect architecture
        checkpoint = torch.load(PHASE2_CHECKPOINT, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Properly detect num universities
        num_univ = self.detect_num_universities(state_dict)
        
        # Check if building head exists
        has_building = any('building_head' in key for key in state_dict.keys())
        
        self.num_universities = num_univ or 44
        self.has_building_prediction = has_building
        self.num_buildings = None  # Your model doesn't predict buildings
        
        print(f"\n Creating model:")
        print(f"  Universities: {self.num_universities}")
        print(f"  Building prediction: {'Yes' if has_building else 'No'}")
        
        # Create model with matching architecture
        self.phase2_model = UniversityOnlyClassifier(
            encoder=self.phase1_model,
            num_universities=self.num_universities,
            dropout=0.5
        ).to(self.device)
        
        # Load weights with strict=False to handle any mismatches
        missing_keys = self.phase2_model.load_state_dict(state_dict, strict=False)
        
        if missing_keys.missing_keys:
            print(f" Missing keys: {len(missing_keys.missing_keys)}")
        if missing_keys.unexpected_keys:
            print(f" Unexpected keys (ignored): {len(missing_keys.unexpected_keys)}")
        
        print(f" Loaded Phase 2 from {PHASE2_CHECKPOINT}")
        
        self.phase2_model.eval()
        print(" Models loaded successfully\n")
    
    def setup_university_names(self):
        """Setup university name mapping"""
        global UNIVERSITY_NAMES
        UNIVERSITY_NAMES = {
            i: f"University_{i}" for i in range(self.num_universities)
        }
    
    def load_faiss_index(self):
        """Load FAISS index and metadata"""
        print("Loading FAISS index...")
        
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found")
        
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        print(f" FAISS index loaded: {self.faiss_index.ntotal} drone images")
        
        # Load metadata
        if DRONE_METADATA_PATH.exists():
            with open(DRONE_METADATA_PATH, 'rb') as f:
                self.drone_metadata = pickle.load(f)
            print(f"✓ Metadata loaded: {len(self.drone_metadata)} entries")
        else:
            self.drone_metadata = {}
        
        # Load drone paths
        if DRONE_PATHS_PATH.exists():
            with open(DRONE_PATHS_PATH, 'rb') as f:
                self.drone_paths = pickle.load(f)
            print(f"✓ Drone paths loaded: {len(self.drone_paths)} paths\n")
        else:
            self.drone_paths = []
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        image_array = np.array(image).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.float().to(self.device)
    
    @torch.no_grad()
    def predict(self, street_image: Image.Image, top_k: int = 5) -> Dict:
        """Full inference pipeline"""
        
        # Step 1: Encode street image
        street_tensor = self.preprocess_image(street_image)
        street_emb = self.phase1_model.encode_street(street_tensor, output_dim=512)
        street_emb_np = street_emb.cpu().numpy()
        
        # Step 2: FAISS search
        similarities, indices = self.faiss_index.search(street_emb_np, k=top_k)
        
        top_idx = int(indices[0][0])
        top_similarity = float(similarities[0][0])
        
        matched_metadata = self.drone_metadata.get(top_idx, {})
        matched_drone_path = self.drone_paths[top_idx] if top_idx < len(self.drone_paths) else None
        
        # Load matched drone image
        if matched_drone_path and Path(matched_drone_path).exists():
            try:
                matched_drone_img = Image.open(matched_drone_path).convert('RGB')
                matched_drone_tensor = self.preprocess_image(matched_drone_img)
            except:
                matched_drone_tensor = torch.zeros_like(street_tensor)
        else:
            matched_drone_tensor = torch.zeros_like(street_tensor)
        
        # Step 3: Classification
        univ_logits, building_logits = self.phase2_model(street_tensor, matched_drone_tensor)
        
        # University predictions
        univ_probs = torch.softmax(univ_logits, dim=1)
        univ_top_probs, univ_top_indices = torch.topk(univ_probs, k=min(3, self.num_universities), dim=1)
        
        # Building predictions (placeholder since not available)
        building_predictions = [
            {"id": 0, "confidence": 0.0, "note": "Building prediction not available in this model"}
        ]
        
        # Format results
        results = {
            "university_predictions": [
                {
                    "id": int(univ_top_indices[0][i]),
                    "name": UNIVERSITY_NAMES.get(int(univ_top_indices[0][i]), f"University_{int(univ_top_indices[0][i])}"),
                    "confidence": float(univ_top_probs[0][i])
                }
                for i in range(len(univ_top_indices[0]))
            ],
            "building_predictions": building_predictions,
            "matched_drone": {
                "index": top_idx,
                "similarity": top_similarity,
                "metadata": matched_metadata
            },
            "retrieval_results": [
                {"index": int(indices[0][i]), "similarity": float(similarities[0][i])}
                for i in range(min(top_k, len(indices[0])))
            ],
            "model_info": {
                "num_universities": self.num_universities,
                "has_building_prediction": False
            }
        }
        
        return results

# FastAPI Application

app = FastAPI(
    title="Cross-View Geo-localization API",
    description="Predict university from street images",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_engine = None

@app.on_event("startup")
async def startup_event():
    global inference_engine
    try:
        inference_engine = InferenceEngine()
        print("=" * 70)
        print(" Server ready!")
        print(f"  Universities: {inference_engine.num_universities}")
        print(f"  Building prediction: No")
        print(f"  Drone images: {inference_engine.faiss_index.ntotal}")
        print("=" * 70)
    except Exception as e:
        print(f" Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Cross-View Geo-localization API",
        "device": str(DEVICE),
        "models_loaded": inference_engine is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "num_universities": inference_engine.num_universities if inference_engine else None,
        "has_building_prediction": False,
        "num_drone_images": inference_engine.faiss_index.ntotal if inference_engine else 0
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        results = inference_engine.predict(image, top_k=5)
        return JSONResponse(content=results)
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/universities")
async def get_universities():
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "universities": [
            {"id": i, "name": UNIVERSITY_NAMES.get(i, f"University_{i}")}
            for i in range(inference_engine.num_universities)
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("Starting Cross-View Geo-localization Server...")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 70)
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")