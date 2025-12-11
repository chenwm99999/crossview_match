"""Phase 2: University Classification Training"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

from src.dataset import CrossViewDataset
from src.models.crossview_model_v2 import CrossViewModelV2
from src.models.hierarchical_classifier import HierarchicalClassifier


class Phase2Trainer:
    """University classification trainer"""
    
    def __init__(self, config_path='configs/phase2.yaml', phase1_checkpoint='checkpoints/phase1_v2_best.pth'):
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['training']['device']
        self.phase1_checkpoint = phase1_checkpoint
        
        self.checkpoint_dir = Path(self.config['paths']['checkpoints'])
        self.log_dir = Path(self.config['paths']['logs'])
        self.models_dir = Path(self.config['paths']['models'])
        
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        run_name = f"phase2_university_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=self.log_dir / run_name)
        
        print("=" * 70)
        print("PHASE 2: UNIVERSITY CLASSIFICATION")
        print("=" * 70)
        print(f"Config: {config_path}")
        print(f"Phase 1: {phase1_checkpoint}")
        
        self._setup_data()
        self._setup_model()
        self._setup_training()
        
    def _setup_data(self):
        """Setup datasets"""
        print("\nLoading datasets...")
        
        self.train_dataset = CrossViewDataset(
            root=self.config['data']['root'],
            split='train',
            mode='drone',
            use_augmentation=False,
            use_first_image=True
        )
        
        self.val_dataset = CrossViewDataset(
            root=self.config['data']['root'],
            split='test',
            mode='drone',
            use_augmentation=False,
            use_first_image=True
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['phase2']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['phase2']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        print(f" Train: {len(self.train_dataset)}")
        print(f" Val: {len(self.val_dataset)}")
        
    def _setup_model(self):
        """Load encoder and create classifier"""
        print("\nLoading model...")
        
        encoder = CrossViewModelV2(pretrained=False).to(self.device)
        checkpoint = torch.load(self.phase1_checkpoint, map_location=self.device)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        
        print(f" Phase 1 loaded (R@1: {checkpoint['best_recall']*100:.2f}%)")
        
        self.model = HierarchicalClassifier(
            encoder,
            num_universities=self.config['model']['num_universities'],
            num_buildings=self.config['model']['num_buildings'],
            dropout=self.config['phase2']['dropout']
        ).to(self.device)
        
    def _setup_training(self):
        """Setup optimizer and scheduler"""
        print("\nSetting up training...")
        
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['phase2']['lr'],
            weight_decay=self.config['phase2']['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['phase2']['epochs'],
            eta_min=self.config['phase2']['min_lr']
        )
        
        self.warmup_epochs = 3
        self.warmup_lr = np.linspace(0, self.config['phase2']['lr'], self.warmup_epochs)
        
        self.start_epoch = 0
        self.best_univ_acc = 0.0
        
        print(f" AdamW (lr={self.config['phase2']['lr']})")
        
    def train_epoch(self, epoch):
        """Train one epoch - university only"""
        self.model.train()
        self.model.encoder.eval()
        
        # Warmup
        if epoch < self.warmup_epochs:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.warmup_lr[epoch]
        
        epoch_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for street_img, drone_img, _, univ_ids, _ in pbar:
            street_img = street_img.to(self.device)
            drone_img = drone_img.to(self.device)
            univ_ids = univ_ids.to(self.device)
            
            # Forward
            univ_logits, _ = self.model(street_img, drone_img)
            
            # Loss - university only
            label_smoothing = self.config['phase2']['label_smoothing']
            loss = F.cross_entropy(univ_logits, univ_ids, label_smoothing=label_smoothing)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})
        
        epoch_loss /= len(self.train_loader)
        
        return epoch_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate university classification"""
        self.model.eval()
        
        correct_top1 = 0
        correct_top3 = 0
        total = 0
        
        for street_img, drone_img, _, univ_ids, _ in tqdm(self.val_loader, desc='Validation'):
            street_img = street_img.to(self.device)
            drone_img = drone_img.to(self.device)
            univ_ids = univ_ids.to(self.device)
            
            # Forward
            univ_logits, _ = self.model(street_img, drone_img)
            
            # Predictions
            pred_top1 = univ_logits.argmax(dim=1)
            pred_top3 = univ_logits.topk(3, dim=1).indices
            
            correct_top1 += (pred_top1 == univ_ids).sum().item()
            correct_top3 += (univ_ids.unsqueeze(1) == pred_top3).any(dim=1).sum().item()
            
            total += len(univ_ids)
        
        metrics = {
            'univ_top1': correct_top1 / total,
            'univ_top3': correct_top3 / total
        }
        
        print(f"\n  University Top-1: {metrics['univ_top1']*100:.2f}%")
        print(f"  University Top-3: {metrics['univ_top3']*100:.2f}%")
        
        self.writer.add_scalar('val/univ_top1', metrics['univ_top1'], epoch)
        self.writer.add_scalar('val/univ_top3', metrics['univ_top3'], epoch)
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if (epoch + 1) % self.config['phase2']['save_interval'] == 0:
            path = self.checkpoint_dir / f'phase2_univ_epoch_{epoch+1}.pth'
            torch.save(checkpoint, path)
        
        if is_best:
            path = self.checkpoint_dir / 'phase2_university_best.pth'
            torch.save(checkpoint, path)
            print(f"  → Best: {metrics['univ_top1']*100:.2f}%")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 70)
        print("TRAINING START (University Classification Only)")
        print("=" * 70)
        print(f"Epochs: {self.config['phase2']['epochs']}")
        print(f"Task: 44-way university classification")
        print("=" * 70)
        
        for epoch in range(self.start_epoch, self.config['phase2']['epochs']):
            
            # Train
            loss = self.train_epoch(epoch)
            
            # Validate
            if (epoch + 1) % self.config['phase2']['val_interval'] == 0:
                print(f"\nEpoch {epoch+1}")
                print(f"  Loss: {loss:.4f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                metrics = self.validate(epoch)
                
                is_best = metrics['univ_top1'] > self.best_univ_acc
                if is_best:
                    self.best_univ_acc = metrics['univ_top1']
                    self.save_checkpoint(epoch, metrics, is_best=True)
            
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
        
        print("\n" + "=" * 70)
        print("PHASE 2 COMPLETE")
        print("=" * 70)
        print(f"Best University Top-1: {self.best_univ_acc*100:.2f}%")
        print("=" * 70)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/phase2.yaml')
    parser.add_argument('--phase1_ckpt', type=str, default='checkpoints/phase1_v2_best.pth')
    
    args = parser.parse_args()
    
    trainer = Phase2Trainer(args.config, args.phase1_ckpt)
    trainer.train()


if __name__ == "__main__":
    main()