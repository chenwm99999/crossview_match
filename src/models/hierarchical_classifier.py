"""HierarchicalClassifier with feature refinement for similar embeddings"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalClassifier(nn.Module):
    """
    Enhanced classifier optimized for similar embeddings
    
    Key improvements:
    1. Feature refinement layer to increase discriminability
    2. Deeper networks with residual connections
    3. Strong regularization to prevent overfitting
    """
    
    def __init__(self, encoder, num_universities=44, num_buildings=701, dropout=0.5):
        super().__init__()
        
        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024)
        )
        
        # University head (4-layer deep)
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
        
        # Building head (4-layer deep with larger hidden dims)
        self.building_head = nn.Sequential(
            nn.Linear(1024, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(1536, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(1536, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(1024, num_buildings)
        )
        
        print(" HierarchicalClassifier")
        print(f"  Feature refinement: 1024→1024 (2-layer)")
        print(f"  University: 1024→512→256→128→{num_universities}")
        print(f"  Building: 1024→1536→1536→1024→{num_buildings}")
        print(f"  Dropout: {dropout}")
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Trainable: {trainable/1e6:.2f}M")
    
    def forward(self, street_img, drone_img):
        with torch.no_grad():
            street_clip, drone_clip = self.encoder(
                street_img, drone_img, return_both_embeddings=False
            )
        
        combined = torch.cat([street_clip, drone_clip], dim=1)
        refined = self.refine(combined)
        refined = refined + combined
        
        # Only university classification
        univ_logits = self.university_head(refined)
        
        # Return None for building logits
        return univ_logits, None