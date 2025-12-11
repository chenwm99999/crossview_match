"""Phase 1: Cross-View Matching Training Script"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

from src.dataset import CrossViewDataset
from src.models.crossview_model import CrossViewModel
from src.models.teachers import FrozenTeachers
from src.losses import CombinedLoss


class Phase1Trainer:
    """Complete trainer for Phase 1 cross-view matching"""
    
    def __init__(self, config_path='configs/default.yaml'):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['training']['device']
        self.seed = self.config['training']['seed']
        
        # Set random seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Setup paths
        self.checkpoint_dir = Path(self.config['paths']['checkpoints'])
        self.log_dir = Path(self.config['paths']['logs'])
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir / 'phase1' / datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        print("=" * 60)
        print("CROSS-VIEW MATCHING TRAINING")
        print("=" * 60)
        print(f"Config: {config_path}")
        print(f"Device: {self.device}")
        print(f"Seed: {self.seed}")
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()
        
    def _setup_data(self):
        """Setup datasets and dataloaders"""
        print("\nSetting up datasets...")
        
        # Training dataset
        self.train_dataset = CrossViewDataset(
            root=self.config['data']['root'],
            split='train',
            mode='drone',
            use_augmentation=True,
            use_first_image=True
        )
        
        # Validation dataset (use test set for validation)
        self.val_dataset = CrossViewDataset(
            root=self.config['data']['root'],
            split='test',
            mode='drone',
            use_augmentation=False,
            use_first_image=True
        )
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['phase1']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['phase1']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        print(f" Train samples: {len(self.train_dataset)}")
        print(f" Val samples: {len(self.val_dataset)}")
        print(f" Batch size: {self.config['phase1']['batch_size']}")
        print(f" Train batches: {len(self.train_loader)}")
        
    def _setup_model(self):
        """Setup model and teachers"""
        print("\nSetting up model...")
        
        # Student model
        self.model = CrossViewModel(
            pretrained=self.config['model']['use_imagenet_pretrained']
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"  Model created")
        print(f"  Total params: {total_params/1e6:.2f}M")
        print(f"  Trainable params: {trainable_params/1e6:.2f}M")
        
        # Teacher models
        self.teachers = FrozenTeachers(device=self.device)
        
    def _setup_training(self):
        """Setup loss, optimizer, scheduler"""
        print("\nSetting up training...")
        
        # Loss function
        self.criterion = CombinedLoss(
            lambda_clip=self.config['phase1']['lambda_clip'],
            lambda_dino=self.config['phase1']['lambda_dino'],
            alpha=self.config['phase1']['alpha_triplet']
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['phase1']['lr'],
            weight_decay=self.config['phase1']['weight_decay'],
            betas=self.config['phase1']['betas']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['phase1']['epochs'],
            eta_min=self.config['phase1']['min_lr']
        )
        
        # Warmup scheduler
        self.warmup_epochs = self.config['phase1']['warmup_epochs']
        self.warmup_lr_schedule = np.linspace(
            0, self.config['phase1']['lr'], self.warmup_epochs
        )
        
        # Training state
        self.start_epoch = 0
        self.best_recall = 0.0
        
        print(f" Optimizer: AdamW (lr={self.config['phase1']['lr']})")
        print(f" Scheduler: CosineAnnealing")
        print(f" Warmup: {self.warmup_epochs} epochs")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        # Warmup learning rate
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr_schedule[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        epoch_losses = {'total': 0, 'triplet': 0, 'clip': 0, 'dino': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["phase1"]["epochs"]}')
        
        for batch_idx, (street_img, drone_img, building_ids, univ_ids, build_nums) in enumerate(pbar):
            # Move to device
            street_img = street_img.to(self.device)
            drone_img = drone_img.to(self.device)
            
            # Sample negative drones (ensure different from positive)
            B = len(drone_img)
            neg_shift = torch.randint(1, B, (B,))
            neg_indices = (torch.arange(B) + neg_shift) % B
            drone_neg = drone_img[neg_indices]
            
            # Forward: Student model
            street_clip, street_dino, drone_clip, drone_dino = self.model(
                street_img, drone_img, return_both_embeddings=True
            )
            
            # Get negative embeddings
            _, _, drone_neg_clip, _ = self.model(
                street_img, drone_neg, return_both_embeddings=True
            )
            
            # Forward: Teachers (frozen)
            teacher_features = self.teachers(street_img, drone_img)
            
            # Prepare outputs
            student_outputs = {
                'street_clip': street_clip,
                'street_dino': street_dino,
                'drone_clip': drone_clip,
                'drone_dino': drone_dino,
                'drone_neg_clip': drone_neg_clip
            }
            
            # Compute losses
            losses = self.criterion(student_outputs, teacher_features)
            
            # Backward
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config['training']['grad_clip']
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'tri': f"{losses['triplet'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to TensorBoard
            if batch_idx % self.config['phase1']['log_interval'] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                for key, value in losses.items():
                    self.writer.add_scalar(f'train/{key}_loss', value.item(), global_step)
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate and compute Recall@K"""
        self.model.eval()
        
        print("\nValidating..")
        
        # Extract all embeddings
        street_embs = []
        drone_embs = []
        
        for street_img, drone_img, _, _, _ in tqdm(self.val_loader, desc='Extracting embeddings'):
            street_img = street_img.to(self.device)
            drone_img = drone_img.to(self.device)
            
            # Get CLIP embeddings (512-dim for retrieval)
            street_clip, drone_clip = self.model(
                street_img, drone_img, return_both_embeddings=False
            )
            
            street_embs.append(street_clip.cpu())
            drone_embs.append(drone_clip.cpu())
        
        # Concatenate
        street_embs = torch.cat(street_embs, dim=0)  # [N, 512]
        drone_embs = torch.cat(drone_embs, dim=0)    # [N, 512]
        
        # Compute similarity matrix
        similarity = street_embs @ drone_embs.t()  # [N, N]
        
        # Compute Recall@K
        recalls = {}
        for k in [1, 5, 10]:
            # Get top-K indices
            _, topk_indices = torch.topk(similarity, k, dim=1)
            
            # Check if correct match is in top-K
            correct_indices = torch.arange(len(street_embs)).unsqueeze(1)
            correct_in_topk = (topk_indices == correct_indices).any(dim=1)
            
            recall = correct_in_topk.float().mean().item()
            recalls[f'R@{k}'] = recall
            
            print(f"  Recall@{k}: {recall*100:.2f}%")
        
        # Log to TensorBoard
        for key, value in recalls.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
        
        return recalls
    
    def save_checkpoint(self, epoch, recalls, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_recall': self.best_recall,
            'recalls': recalls,
            'config': self.config
        }
        
        # Save regular checkpoint
        if (epoch + 1) % self.config['phase1']['save_interval'] == 0:
            path = self.checkpoint_dir / f'phase1_epoch_{epoch+1}.pth'
            torch.save(checkpoint, path)
            print(f" Saved checkpoint: {path}")
        
        # Save best model
        if is_best:
            path = self.checkpoint_dir / 'phase1_best.pth'
            torch.save(checkpoint, path)
            print(f" Saved BEST model: {path} (R@1: {recalls['R@1']*100:.2f}%)")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Total epochs: {self.config['phase1']['epochs']}")
        print(f"Validation every: {self.config['phase1']['val_interval']} epochs")
        print(f"Checkpoint every: {self.config['phase1']['save_interval']} epochs")
        print("=" * 60 + "\n")
        
        for epoch in range(self.start_epoch, self.config['phase1']['epochs']):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config['phase1']['epochs']}")
            print(f"{'='*60}")
            
            # Train one epoch
            train_losses = self.train_epoch(epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"    - Triplet: {train_losses['triplet']:.4f}")
            print(f"    - CLIP: {train_losses['clip']:.4f}")
            print(f"    - DINOv2: {train_losses['dino']:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Validation
            if (epoch + 1) % self.config['phase1']['val_interval'] == 0:
                recalls = self.validate(epoch)
                
                # Check if best
                is_best = recalls['R@1'] > self.best_recall
                if is_best:
                    self.best_recall = recalls['R@1']
                
                # Save checkpoint
                self.save_checkpoint(epoch, recalls, is_best=is_best)
            else:
                recalls = None
            
            # Step scheduler (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            # Log epoch metrics
            self.writer.add_scalar('train/epoch_loss', train_losses['total'], epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best Recall@1: {self.best_recall*100:.2f}%")
        print(f"Best model saved at: {self.checkpoint_dir / 'phase1_best.pth'}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Phase 1 Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Phase1Trainer(config_path=args.config)
    
    # Resume if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.start_epoch = checkpoint['epoch'] + 1
        trainer.best_recall = checkpoint['best_recall']
        print(f" Resumed from epoch {trainer.start_epoch}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()