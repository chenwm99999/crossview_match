"""
Custom checkpoint loader that recreates the exact architecture from checkpoint
"""

import torch
import torch.nn as nn
from pathlib import Path

def load_phase2_from_checkpoint(checkpoint_path, encoder, device='cpu'):
    """
    Load Phase 2 model by recreating the exact architecture from checkpoint
    
    Args:
        checkpoint_path: Path to phase2 checkpoint
        encoder: Phase 1 encoder (will be frozen)
        device: Device to load model on
    
    Returns:
        Loaded Phase 2 model
    """
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    print("Analyzing checkpoint architecture...")
    
    # Detect architecture from state_dict
    num_universities = None
    num_buildings = None
    has_refine_layer = False
    has_building_head = False
    
    for key in state_dict.keys():
        # Check for refine layer
        if 'refine' in key:
            has_refine_layer = True
        
        # Check for building head
        if 'building_head' in key:
            has_building_head = True
        
        # Detect number of universities (last layer of university_head)
        if 'university_head' in key and 'weight' in key:
            shape = state_dict[key].shape
            if len(shape) == 2:
                num_universities = max(num_universities or 0, shape[0])
        
        # Detect number of buildings (last layer of building_head)
        if 'building_head' in key and 'weight' in key:
            shape = state_dict[key].shape
            if len(shape) == 2:
                num_buildings = max(num_buildings or 0, shape[0])
    
    print(f" Detected architecture:")
    print(f"  - Universities: {num_universities}")
    print(f"  - Buildings: {num_buildings if has_building_head else 'None (university only)'}")
    print(f"  - Has refine layer: {has_refine_layer}")
    
    # Create model with matching architecture
    if has_building_head and num_buildings:
        # Full hierarchical model
        from models.hierarchical_classifier import HierarchicalClassifier
        model = HierarchicalClassifier(
            encoder=encoder,
            num_universities=num_universities or 44,
            num_buildings=num_buildings or 701
        ).to(device)
    else:
        # University-only model
        model = UniversityOnlyClassifier(
            encoder=encoder,
            num_universities=num_universities or 44
        ).to(device)
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Successfully loaded Phase 2 checkpoint")
        
        # Check what was loaded
        loaded_keys = set(state_dict.keys())
        model_keys = set(model.state_dict().keys())
        missing = model_keys - loaded_keys
        unexpected = loaded_keys - model_keys
        
        if missing:
            print(f"⚠ Missing keys (will use random init): {len(missing)}")
        if unexpected:
            print(f"⚠ Unexpected keys (ignored): {len(unexpected)}")
            
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        raise
    
    model.eval()
    return model, num_universities, num_buildings if has_building_head else None


class UniversityOnlyClassifier(nn.Module):
    """Simplified classifier for university-only predictions"""
    
    def __init__(self, encoder, num_universities=44, dropout=0.5):
        super().__init__()
        
        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.num_universities = num_universities
        self.num_buildings = None  # No building prediction
        
        # Feature refinement (match your architecture)
        self.refine = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024)
        )
        
        # University head (match your architecture)
        self.university_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(128, num_universities)
        )
        
        print(f" UniversityOnlyClassifier initialized")
        print(f"  Universities: {num_universities}")
    
    def forward(self, street_img, drone_img):
        with torch.no_grad():
            street_clip, drone_clip = self.encoder(
                street_img, drone_img, return_both_embeddings=False
            )
        
        combined = torch.cat([street_clip, drone_clip], dim=1)
        refined = self.refine(combined)
        refined = refined + combined
        
        univ_logits = self.university_head(refined)
        
        # Return None for building logits (no building prediction)
        return univ_logits, None