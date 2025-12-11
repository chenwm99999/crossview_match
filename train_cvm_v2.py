"""
Phase 1 Enhanced Training Script with Publication-Quality Logging

Enhancements over baseline:
1. Multi-scale fusion + Attention pooling
2. Mixed precision training (FP16)
3. Comprehensive metrics logging for paper
4. Attention map visualization
5. Embedding visualization (t-SNE)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.dataset import CrossViewDataset
from src.models.crossview_model_v2 import CrossViewModelV2 as CrossViewModel # V2 model
from src.models.teachers import FrozenTeachers
from src.losses import CombinedLoss


class Phase1TrainerV2:
    """Enhanced trainer with publication-quality features"""
    
    def __init__(self, config_path='configs/default.yaml'):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['training']['device']
        self.seed = self.config['training']['seed']
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        
        # Setup paths
        self.checkpoint_dir = Path(self.config['paths']['checkpoints'])
        self.log_dir = Path(self.config['paths']['logs'])
        self.results_dir = Path(self.config['paths'].get('results', 'results'))
        
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # TensorBoard with timestamped directory
        run_name = f"phase1_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=self.log_dir / run_name)
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        print("=" * 70)
        print("PHASE 1 ENHANCED TRAINING (V2)")
        print("=" * 70)
        print(f"Config: {config_path}")
        print(f"Device: {self.device}")
        print(f"Seed: {self.seed}")
        print(f"Mixed Precision: Enabled (FP16)")
        print(f"Run name: {run_name}")
        
        # Initialize
        self._setup_data()
        self._setup_model()
        self._setup_training()
        
        # Tracking for paper plots
        self.train_losses_history = []
        self.val_recalls_history = []
        self.lr_history = []
        
    def _setup_data(self):
        """Setup datasets and dataloaders"""
        print("\n" + "=" * 70)
        print("DATA SETUP")
        print("=" * 70)
        
        # Training dataset
        self.train_dataset = CrossViewDataset(
            root=self.config['data']['root'],
            split='train',
            mode='drone',
            use_augmentation=True,
            use_first_image=True
        )
        
        # Validation dataset
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
        print(f" Batches/epoch: {len(self.train_loader)}")
        
    def _setup_model(self):
        """Setup enhanced model with multi-scale fusion"""
        print("\n" + "=" * 70)
        print("MODEL SETUP")
        print("=" * 70)
        
        # Enhanced student model (V2)
        self.model = CrossViewModel(
            pretrained=self.config['model']['use_imagenet_pretrained'],
            use_multiscale=True,   # Enable multi-scale fusion
            use_attn_pool=True     # Enable attention pooling
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"  Model (V2) created")
        print(f"  Total params: {total_params/1e6:.2f}M")
        print(f"  Trainable: {trainable_params/1e6:.2f}M")
        
        # Teacher models
        self.teachers = FrozenTeachers(device=self.device)
        
    def _setup_training(self):
        """Setup loss, optimizer, scheduler"""
        print("\n" + "=" * 70)
        print("TRAINING SETUP")
        print("=" * 70)
        
        # Loss function
        self.criterion = CombinedLoss(
            lambda_clip=self.config['phase1']['lambda_clip'],
            lambda_dino=self.config['phase1']['lambda_dino'],
            triplet_margin=self.config['phase1']['triplet_margin']
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['phase1']['lr'],
            weight_decay=self.config['phase1']['weight_decay'],
            betas=self.config['phase1']['betas']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['phase1']['epochs'],
            eta_min=self.config['phase1']['min_lr']
        )
        
        # Warmup
        self.warmup_epochs = self.config['phase1']['warmup_epochs']
        self.warmup_lr_schedule = np.linspace(
            0, self.config['phase1']['lr'], self.warmup_epochs
        )
        
        # Training state
        self.start_epoch = 0
        self.best_recall = 0.0
        self.global_step = 0
        
        print(f" Loss: Triplet + CLIP(λ={self.config['phase1']['lambda_clip']}) + DINOv2(λ={self.config['phase1']['lambda_dino']})")
        print(f" Optimizer: AdamW (lr={self.config['phase1']['lr']}, wd={self.config['phase1']['weight_decay']})")
        print(f" Scheduler: CosineAnnealing")
        print(f" Warmup: {self.warmup_epochs} epochs")
        print(f" Mixed Precision: FP16 enabled")
        
    def train_epoch(self, epoch):
        """Train one epoch with mixed precision"""
        self.model.train()
        
        # Warmup learning rate
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr_schedule[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        epoch_losses = {'total': 0, 'triplet': 0, 'clip': 0, 'dino': 0}
        epoch_grad_norms = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["phase1"]["epochs"]}')
        
        for batch_idx, (street_img, drone_img, building_ids, univ_ids, build_nums) in enumerate(pbar):
            street_img = street_img.to(self.device)
            drone_img = drone_img.to(self.device)
            
            # Sample negative drones (ensure different from positive)
            B = len(drone_img)
            neg_shift = torch.randint(1, B, (B,))
            neg_indices = (torch.arange(B) + neg_shift) % B
            drone_neg = drone_img[neg_indices]
            
            # mixed precision forward
            with autocast():
                # Student forward
                street_clip, street_dino, drone_clip, drone_dino = self.model(
                    street_img, drone_img, return_both_embeddings=True
                )
                
                _, _, drone_neg_clip, _ = self.model(
                    street_img, drone_neg, return_both_embeddings=True
                )
                
                # Teacher forward (frozen)
                teacher_features = self.teachers(street_img, drone_img)
                
                # Compute losses
                student_outputs = {
                    'street_clip': street_clip,
                    'street_dino': street_dino,
                    'drone_clip': drone_clip,
                    'drone_dino': drone_dino,
                    'drone_neg_clip': drone_neg_clip
                }
                
                losses = self.criterion(student_outputs, teacher_features)
            
            # Backward with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(losses['total']).backward()
            
            # Gradient clipping (unscale first)
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config['training']['grad_clip']
            )
            epoch_grad_norms.append(grad_norm.item())
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'tri': f"{losses['triplet'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Detailed logging every N batches
            if batch_idx % self.config['phase1']['log_interval'] == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(f'train_batch/{key}_loss', value.item(), self.global_step)
                self.writer.add_scalar('train_batch/grad_norm', grad_norm.item(), self.global_step)
            
            self.global_step += 1
        
        # Compute epoch averages
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        avg_grad_norm = np.mean(epoch_grad_norms)
        
        # Log epoch metrics
        for key, value in epoch_losses.items():
            self.writer.add_scalar(f'train_epoch/{key}_loss', value, epoch)
        self.writer.add_scalar('train_epoch/avg_grad_norm', avg_grad_norm, epoch)
        
        # Store for plotting
        self.train_losses_history.append(epoch_losses)
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        
        return epoch_losses, avg_grad_norm
    
    @torch.no_grad()
    def validate(self, epoch, save_visualizations=False):
        """
        Comprehensive validation with metrics for paper
        
        Computes:
        1. Recall@1/5/10/20
        2. Mean Average Precision (mAP)
        3. Per-rank accuracy distribution
        """
        self.model.eval()
        
        print("\n" + "=" * 70)
        print(f"VALIDATION - Epoch {epoch+1}")
        print("=" * 70)
        
        # Extract all embeddings
        street_embs = []
        drone_embs = []
        building_ids_all = []
        attention_maps_street = []  # For visualization
        attention_maps_drone = []
        
        for street_img, drone_img, building_ids, _, _ in tqdm(self.val_loader, desc='Extracting embeddings'):
            street_img = street_img.to(self.device)
            drone_img = drone_img.to(self.device)
            
            # Get embeddings + attention maps
            if save_visualizations and len(attention_maps_street) < 5:
                # Save first 5 attention maps for visualization
                street_clip, drone_clip, s_attn, d_attn = self.model(
                    street_img, drone_img,
                    return_both_embeddings=False,
                    return_attn_maps=True
                )
                attention_maps_street.append(s_attn.cpu())
                attention_maps_drone.append(d_attn.cpu())
            else:
                street_clip, drone_clip = self.model(
                    street_img, drone_img,
                    return_both_embeddings=False
                )
            
            street_embs.append(street_clip.cpu())
            drone_embs.append(drone_clip.cpu())
            building_ids_all.extend(building_ids)
        
        # Concatenate
        street_embs = torch.cat(street_embs, dim=0)  # [N, 512]
        drone_embs = torch.cat(drone_embs, dim=0)    # [N, 512]
        
        N = len(street_embs)
        
        # Compute similarity matrix
        similarity = street_embs @ drone_embs.t()  # [N, N]
        
        # Recall@K
        recalls = {}
        for k in [1, 5, 10, 20]:
            _, topk_indices = torch.topk(similarity, k, dim=1)
            correct_indices = torch.arange(N).unsqueeze(1)
            correct_in_topk = (topk_indices == correct_indices).any(dim=1)
            recall = correct_in_topk.float().mean().item()
            recalls[f'R@{k}'] = recall
            
            print(f"  Recall@{k:2d}: {recall*100:6.2f}%")
            self.writer.add_scalar(f'val/Recall@{k}', recall, epoch)
        
        # Mean Average Precision (for paper)
        # For each query, compute AP
        aps = []
        for i in range(min(N, 100)):  # Compute for first 100 (faster)
            sorted_indices = similarity[i].argsort(descending=True)
            # Find where correct match appears
            correct_rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            # AP = 1 / (rank + 1)
            ap = 1.0 / (correct_rank + 1)
            aps.append(ap)
        
        mAP = np.mean(aps)
        print(f"  mAP@100: {mAP*100:6.2f}%")
        self.writer.add_scalar('val/mAP', mAP, epoch)
        
        # Rank Distribution
        correct_ranks = []
        for i in range(N):
            sorted_indices = similarity[i].argsort(descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            correct_ranks.append(rank)
        
        # Log rank statistics
        correct_ranks = np.array(correct_ranks)
        print(f"  Median rank: {np.median(correct_ranks):.0f}")
        print(f"  Mean rank: {np.mean(correct_ranks):.1f}")
        print(f"  Rank ≤ 5: {(correct_ranks < 5).sum() / N * 100:.1f}%")
        
        # Store for history
        self.val_recalls_history.append(recalls)
        
        # Attention Maps
        if save_visualizations and attention_maps_street:
            self._save_attention_maps(
                attention_maps_street[0][:4],  # First 4 samples
                attention_maps_drone[0][:4],
                epoch
            )
        
        #  Embedding t-SNE when final epoch
        if save_visualizations:
            self._save_tsne_plot(street_embs[:500], drone_embs[:500], epoch)
        
        # save metrics
        if epoch % 5 == 0:
            self._save_training_curves(epoch)
        
        return recalls
    
    def _save_attention_maps(self, street_attn, drone_attn, epoch):
        """Save attention map visualizations for paper"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Attention Maps - Epoch {epoch+1}', fontsize=16)
        
        for i in range(4):
            # Street attention
            attn_s = street_attn[i, 0].numpy()  # [H, W]
            axes[0, i].imshow(attn_s, cmap='hot', interpolation='nearest')
            axes[0, i].set_title(f'Street {i+1}')
            axes[0, i].axis('off')
            
            # Drone attention
            attn_d = drone_attn[i, 0].numpy()
            axes[1, i].imshow(attn_d, cmap='hot', interpolation='nearest')
            axes[1, i].set_title(f'Drone {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        save_path = self.results_dir / f'attention_maps_epoch_{epoch+1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved attention maps: {save_path}")
    
    def _save_tsne_plot(self, street_embs, drone_embs, epoch):
        """Save t-SNE embedding visualization for paper"""
        print(f"  Computing t-SNE ...")
        
        # Combine embeddings
        combined = torch.cat([street_embs, drone_embs], dim=0).numpy()
        
        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(combined)
        
        # Split back
        N = len(street_embs)
        street_2d = embedded[:N]
        drone_2d = embedded[N:]
        
        # Plot
        plt.figure(figsize=(10, 10))
        plt.scatter(street_2d[:, 0], street_2d[:, 1], c='blue', alpha=0.5, s=20, label='Street')
        plt.scatter(drone_2d[:, 0], drone_2d[:, 1], c='red', alpha=0.5, s=20, label='Drone')
        
        # Draw lines connecting matched pairs
        for i in range(min(50, N)):
            plt.plot([street_2d[i, 0], drone_2d[i, 0]],
                    [street_2d[i, 1], drone_2d[i, 1]],
                    'gray', alpha=0.2, linewidth=0.5)
        
        plt.legend()
        plt.title(f'Embedding Space (t-SNE) - Epoch {epoch+1}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(alpha=0.3)
        
        save_path = self.results_dir / f'tsne_epoch_{epoch+1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Saved t-SNE plot: {save_path}")
    
    def _save_training_curves(self, epoch):
        """Save training curves for paper (loss, recall, LR)"""
        if len(self.train_losses_history) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Training losses
        epochs_so_far = range(1, len(self.train_losses_history) + 1)
        
        total_losses = [h['total'] for h in self.train_losses_history]
        triplet_losses = [h['triplet'] for h in self.train_losses_history]
        clip_losses = [h['clip'] for h in self.train_losses_history]
        dino_losses = [h['dino'] for h in self.train_losses_history]
        
        axes[0, 0].plot(epochs_so_far, total_losses, label='Total', linewidth=2)
        axes[0, 0].plot(epochs_so_far, triplet_losses, label='Triplet', alpha=0.7)
        axes[0, 0].plot(epochs_so_far, clip_losses, label='CLIP', alpha=0.7)
        axes[0, 0].plot(epochs_so_far, dino_losses, label='DINOv2', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: Validation Recall@K
        if self.val_recalls_history:
            val_epochs = [i * self.config['phase1']['val_interval'] + 1 
                         for i in range(len(self.val_recalls_history))]
            
            for k in [1, 5, 10]:
                recalls_k = [h[f'R@{k}'] * 100 for h in self.val_recalls_history]
                axes[0, 1].plot(val_epochs, recalls_k, label=f'R@{k}', marker='o')
            
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Recall (%)')
            axes[0, 1].set_title('Validation Recall@K')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Learning Rate Schedule
        axes[1, 0].plot(range(1, len(self.lr_history) + 1), self.lr_history, color='orange', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Plot 4: Loss Components Breakdown
        if len(self.train_losses_history) > 0:
            latest = self.train_losses_history[-1]
            components = ['triplet', 'clip', 'dino']
            values = [latest[c] for c in components]
            
            axes[1, 1].bar(components, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[1, 1].set_ylabel('Loss Value')
            axes[1, 1].set_title(f'Loss Components (Epoch {epoch+1})')
            axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.results_dir / f'training_curves_epoch_{epoch+1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Saved training curves: {save_path}")
    
    def save_checkpoint(self, epoch, recalls, is_best=False):
        """Save checkpoint with comprehensive metadata"""
        checkpoint = {
            # Model state
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            
            # Metrics
            'best_recall': self.best_recall,
            'current_recalls': recalls,
            'train_losses_history': self.train_losses_history,
            'val_recalls_history': self.val_recalls_history,
            
            # Config and metadata
            'config': self.config,
            'model_version': 'v2_enhanced',
            'enhancements': ['multiscale_fusion', 'attention_pooling', 'mixed_precision']
        }
        
        # Save periodic checkpoint
        if (epoch + 1) % self.config['phase1']['save_interval'] == 0:
            path = self.checkpoint_dir / f'phase1_v2_epoch_{epoch+1}.pth'
            torch.save(checkpoint, path)
            print(f" Checkpoint: {path}")
        
        # Save best model
        if is_best:
            path = self.checkpoint_dir / 'phase1_v2_best.pth'
            torch.save(checkpoint, path)
            print(f"BEST model: {path} (R@1: {recalls['R@1']*100:.2f}%)")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Total epochs: {self.config['phase1']['epochs']}")
        print(f"Validation: Every {self.config['phase1']['val_interval']} epochs")
        print(f"Checkpoints: Every {self.config['phase1']['save_interval']} epochs")
        print("=" * 70 + "\n")
        
        for epoch in range(self.start_epoch, self.config['phase1']['epochs']):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{self.config['phase1']['epochs']}")
            print(f"{'='*70}")
            
            # Train
            train_losses, avg_grad_norm = self.train_epoch(epoch)
            
            # Print summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"    - Triplet: {train_losses['triplet']:.4f}")
            print(f"    - CLIP: {train_losses['clip']:.4f}")
            print(f"    - DINOv2: {train_losses['dino']:.4f}")
            print(f"  Avg Grad Norm: {avg_grad_norm:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Validate
            if (epoch + 1) % self.config['phase1']['val_interval'] == 0:
                # final epoch
                save_viz = (epoch + 1) == self.config['phase1']['epochs']
                
                recalls = self.validate(epoch, save_visualizations=save_viz)
                
                # Check if best
                is_best = recalls['R@1'] > self.best_recall
                if is_best:
                    print(f"\n  New best R@1: {self.best_recall*100:.2f}% → {recalls['R@1']*100:.2f}%")
                    self.best_recall = recalls['R@1']
                
                # Save checkpoint
                self.save_checkpoint(epoch, recalls, is_best=is_best)
            
            # Step scheduler
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
        
        # Final summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best Recall@1: {self.best_recall*100:.2f}%")
        print(f"Checkpoints saved in: {self.checkpoint_dir}")
        print(f"Logs saved in: {self.log_dir}")
        print(f"Results/plots saved in: {self.results_dir}")
        print("=" * 70)
        
        # Save final training curves
        self._save_training_curves(epoch)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Phase 1 Enhanced Training (V2)')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Phase1TrainerV2(config_path=args.config)
    
    # Resume if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        trainer.start_epoch = checkpoint['epoch'] + 1
        trainer.best_recall = checkpoint['best_recall']
        trainer.train_losses_history = checkpoint.get('train_losses_history', [])
        trainer.val_recalls_history = checkpoint.get('val_recalls_history', [])
        print(f"Resumed from epoch {trainer.start_epoch}, best R@1: {trainer.best_recall*100:.2f}%")
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()