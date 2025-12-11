"""
Comprehensive evaluation for final report
Focuses on Phase 1 results with essential Phase 2 metrics
"""

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from src.dataset import CrossViewDataset
from src.models.crossview_model_v2 import CrossViewModelV2
from src.models.hierarchical_classifier import HierarchicalClassifier


class FinalEvaluator:
    """Generate results for paper"""
    
    def __init__(self, phase1_ckpt='checkpoints/phase1_v2_best.pth',
                 phase2_ckpt='checkpoints/phase2_univ_epoch_20.pth',
                 data_root='data/University-Release'):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results_dir = Path('results/final_evaluation')
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Load Phase 1 model
        print("\nLoading Phase 1 model...")
        self.phase1_model = CrossViewModelV2(pretrained=False).to(self.device)
        ckpt = torch.load(phase1_ckpt, map_location=self.device)
        self.phase1_model.load_state_dict(ckpt['model_state_dict'])
        self.phase1_model.eval()
        self.phase1_recall = ckpt['best_recall']
        print(f"Phase 1 loaded (R@1: {self.phase1_recall*100:.2f}%)")
        
        # Load Phase 2 model
        print("\nLoading Phase 2 model...")
        encoder = CrossViewModelV2(pretrained=False).to(self.device)
        encoder.load_state_dict(ckpt['model_state_dict'])
        
        self.phase2_model = HierarchicalClassifier(encoder, 44, 701).to(self.device)
        ckpt2 = torch.load(phase2_ckpt, map_location=self.device)
        self.phase2_model.load_state_dict(ckpt2['model_state_dict'])
        self.phase2_model.eval()
        self.phase2_acc = ckpt2['metrics']['univ_top1']
        print(f"Phase 2 loaded (Univ Top-1: {self.phase2_acc*100:.2f}%)")
        
        # Load test set
        print("\nLoading test set...")
        self.test_dataset = CrossViewDataset(
            root=data_root,
            split='test',
            mode='drone',
            use_augmentation=False,
            use_first_image=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Test samples: {len(self.test_dataset)}")
    
    @torch.no_grad()
    def evaluate_phase1(self):
        """Phase 1: Cross-view matching evaluation"""
        print("\n" + "=" * 70)
        print("PHASE 1 EVALUATION")
        print("=" * 70)
        
        # Extract embeddings
        street_embs = []
        drone_embs = []
        
        for street, drone, _, _, _ in tqdm(self.test_loader, desc='Extracting embeddings'):
            street = street.to(self.device)
            drone = drone.to(self.device)
            
            s_clip, d_clip = self.phase1_model(street, drone, return_both_embeddings=False)
            street_embs.append(s_clip.cpu())
            drone_embs.append(d_clip.cpu())
        
        street_embs = torch.cat(street_embs)
        drone_embs = torch.cat(drone_embs)
        
        # Similarity matrix
        similarity = street_embs @ drone_embs.t()
        N = len(street_embs)
        
        # Recall@K
        recalls = {}
        for k in [1, 5, 10, 20, 50]:
            _, topk = torch.topk(similarity, k, dim=1)
            correct = (topk == torch.arange(N).unsqueeze(1)).any(dim=1)
            recalls[k] = correct.float().mean().item()
        
        print("\nRecall@K Results:")
        for k, r in recalls.items():
            print(f"  R@{k:2d}: {r*100:6.2f}%")
        
        # Mean rank
        ranks = []
        for i in range(N):
            sorted_idx = similarity[i].argsort(descending=True)
            rank = (sorted_idx == i).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)
        
        ranks = np.array(ranks)
        print(f"\nRank Statistics:")
        print(f"  Mean: {ranks.mean():.1f}")
        print(f"  Median: {np.median(ranks):.0f}")
        print(f"  95th percentile: {np.percentile(ranks, 95):.0f}")
        
        # Create recall curve plot
        self._plot_recall_curve(recalls)
        
        # Save results
        results = {
            'phase1_crossview_matching': {
                'recall_at_k': {f'R@{k}': f'{v*100:.2f}%' for k, v in recalls.items()},
                'mean_rank': float(ranks.mean()),
                'median_rank': float(np.median(ranks))
            }
        }
        
        return results
    
    @torch.no_grad()
    def evaluate_phase2(self):
        """Phase 2: University classification evaluation"""
        print("\n" + "=" * 70)
        print("PHASE 2 EVALUATION")
        print("=" * 70)
        
        correct_top1 = 0
        correct_top3 = 0
        total = 0
        
        # Per-university accuracy
        per_univ_correct = {}
        per_univ_total = {}
        
        all_true = []
        all_pred = []
        
        for street, drone, _, univ_ids, _ in tqdm(self.test_loader, desc='Evaluating'):
            street = street.to(self.device)
            drone = drone.to(self.device)
            
            univ_logits, _ = self.phase2_model(street, drone)
            
            pred_top1 = univ_logits.argmax(dim=1)
            pred_top3 = univ_logits.topk(3, dim=1).indices
            
            correct_top1 += (pred_top1.cpu() == univ_ids).sum().item()
            correct_top3 += (univ_ids.unsqueeze(1) == pred_top3.cpu()).any(dim=1).sum().item()
            
            # Store for confusion matrix
            all_true.extend(univ_ids.cpu().numpy())
            all_pred.extend(pred_top1.cpu().numpy())
            
            # Per-university
            for uid, pred in zip(univ_ids, pred_top1.cpu()):
                uid = uid.item()
                if uid not in per_univ_correct:
                    per_univ_correct[uid] = 0
                    per_univ_total[uid] = 0
                
                per_univ_correct[uid] += (pred == uid)
                per_univ_total[uid] += 1
            
            total += len(univ_ids)
        
        top1_acc = correct_top1 / total
        top3_acc = correct_top3 / total
        
        print(f"\nUniversity Classification:")
        print(f"  Top-1: {top1_acc*100:.2f}%")
        print(f"  Top-3: {top3_acc*100:.2f}%")
        
        # Find best/worst universities
        univ_accs = {uid: per_univ_correct[uid]/per_univ_total[uid] 
                     for uid in per_univ_correct}
        
        best_univs = sorted(univ_accs.items(), key=lambda x: x[1], reverse=True)[:3]
        worst_univs = sorted(univ_accs.items(), key=lambda x: x[1])[:3]
        
        print(f"\nBest 3 Universities:")
        for uid, acc in best_univs:
            print(f"  Univ {uid}: {acc*100:.1f}%")
        
        print(f"\nWorst 3 Universities:")
        for uid, acc in worst_univs:
            print(f"  Univ {uid}: {acc*100:.1f}%")
        
        results = {
            'phase2_university_classification': {
                'top1_accuracy': f'{top1_acc*100:.2f}%',
                'top3_accuracy': f'{top3_acc*100:.2f}%',
                'num_classes': 44,
                'num_test_samples': total
            }
        }
        
        # Generate confusion matrix
        self._plot_confusion_matrix(np.array(all_true), np.array(all_pred), num_classes=44)
        
        return results
    
    def _plot_confusion_matrix(self, y_true, y_pred, num_classes=44):
        """Plot confusion matrix for university classification"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        print("\nGenerating confusion matrix...")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        
        # Normalize by row (true labels)
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                    xticklabels=False, yticklabels=False,
                    cbar_kws={'label': 'Normalized Frequency'},
                    ax=ax)
        
        ax.set_xlabel('Predicted University', fontsize=12)
        ax.set_ylabel('True University', fontsize=12)
        ax.set_title(f'Confusion Matrix - University Classification\n(44 classes, Top-1 Acc: {(y_true == y_pred).mean()*100:.1f}%)',
                    fontsize=14)
        
        plt.tight_layout()
        save_path = self.results_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: {save_path}")
        
        return cm

    def _plot_retrieval_examples(self, num_examples=6):
        """Plot retrieval examples (query street → top-5 drone matches)"""
        print("\nGenerating retrieval examples...")
        
        # Select diverse examples
        indices = np.linspace(0, len(self.test_dataset)-1, num_examples, dtype=int)
        
        fig, axes = plt.subplots(num_examples, 6, figsize=(18, 3*num_examples))
        
        for row, idx in enumerate(indices):
            street, drone_gt, bid, uid, _ = self.test_dataset[idx]
            street = street.unsqueeze(0).to(self.device)
            drone_gt_img = drone_gt
            
            # Extract embedding
            with torch.no_grad():
                street_clip, _ = self.phase1_model(street, street, return_both_embeddings=False)
            
            # Get all drone embeddings for this example
            all_drone_embs = []
            for _, d, _, _, _ in self.test_loader:
                d = d.to(self.device)
                with torch.no_grad():
                    _, d_clip = self.phase1_model(d, d, return_both_embeddings=False)
                all_drone_embs.append(d_clip.cpu())
            
            all_drone_embs = torch.cat(all_drone_embs)
            
            # Compute similarities
            sims = (street_clip.cpu() @ all_drone_embs.t()).squeeze()
            top5_indices = sims.topk(5).indices
            top5_sims = sims.topk(5).values
            
            # Denormalize images for display
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            def denorm(img):
                img = img * std + mean
                img = torch.clamp(img, 0, 1)
                return img.permute(1, 2, 0).numpy()
            
            # Plot query (column 0)
            axes[row, 0].imshow(denorm(street.cpu().squeeze()))
            axes[row, 0].set_title(f'Query\nBuilding {bid}', fontsize=9)
            axes[row, 0].axis('off')
            
            # Plot top-5 retrievals (columns 1-5)
            for col, (ret_idx, sim) in enumerate(zip(top5_indices, top5_sims), 1):
                _, drone_ret, ret_bid, _, _ = self.test_dataset[ret_idx]
                
                axes[row, col].imshow(denorm(drone_ret))
                
                # Mark if correct
                is_correct = (ret_bid == bid)
                color = 'green' if is_correct else 'red'
                border = '✓' if is_correct else '✗'
                
                axes[row, col].set_title(f'#{col} {border}\nBuild {ret_bid}\nSim: {sim:.2f}',
                                        fontsize=9, color=color)
                axes[row, col].axis('off')
                
                # Add border
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
        
        plt.suptitle('Phase 1: Retrieval Examples (Street Query → Top-5 Drone Matches)',
                    fontsize=16, y=0.995)
        plt.tight_layout()
        
        save_path = self.results_dir / 'retrieval_examples.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: {save_path}")
    
    def _plot_recall_curve(self, recalls):
        """Plot Recall@K curve for paper"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ks = sorted(recalls.keys())
        rs = [recalls[k] * 100 for k in ks]
        
        ax.plot(ks, rs, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('K (Number of Retrieved Results)', fontsize=12)
        ax.set_ylabel('Recall@K (%)', fontsize=12)
        ax.set_title('Phase 1: Cross-View Retrieval Performance', fontsize=14)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Add value labels
        for k, r in zip(ks, rs):
            ax.text(k, r + 2, f'{r:.1f}%', ha='center', fontsize=10)
        
        plt.tight_layout()
        save_path = self.results_dir / 'phase1_recall_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
    
    def generate_summary_table(self, phase1_results, phase2_results):
        """Generate results table for paper"""
        
        summary = {
            **phase1_results,
            **phase2_results,
            'model_info': {
                'architecture': 'Swin-Tiny + L2L + CLIP + DINOv2',
                'parameters': '65.2M (Phase 1) + 3.5M (Phase 2)',
                'enhancements': ['Multi-scale fusion', 'Attention pooling', 'Dual-teacher distillation']
            }
        }
        
        # Save JSON
        with open(self.results_dir / 'final_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print table
        print("\n" + "=" * 70)
        print("FINAL RESULTS SUMMARY")
        print("=" * 70)
        print("\nPhase 1 (Cross-View Matching):")
        print(f"  Recall@1:  {phase1_results['phase1_crossview_matching']['recall_at_k']['R@1']}")
        print(f"  Recall@5:  {phase1_results['phase1_crossview_matching']['recall_at_k']['R@5']}")
        print(f"  Recall@10: {phase1_results['phase1_crossview_matching']['recall_at_k']['R@10']}")
        
        print("\nPhase 2 (University Classification):")
        print(f"  Top-1: {phase2_results['phase2_university_classification']['top1_accuracy']}")
        print(f"  Top-3: {phase2_results['phase2_university_classification']['top3_accuracy']}")
        
        print("\n" + "=" * 70)
        print(f"Results saved: {self.results_dir}")
        print("=" * 70)
    
    def run(self):
        """Run complete evaluation"""
        
        # Evaluate Phase 1
        phase1_results = self.evaluate_phase1()
        
        # Evaluate Phase 2
        phase2_results = self.evaluate_phase2()
        
        # Retrieval examples
        self._plot_retrieval_examples(num_examples=6)
        
        # Generate summary
        self.generate_summary_table(phase1_results, phase2_results)
        
        print("\n Evaluation complete!")
        print(f" Results saved in: {self.results_dir}")
        print("\nGenerated files:")
        print(f"\nGenerated files:")
        print("  1. final_results.json")
        print("  2. phase1_recall_curve.png")
        print("  3. confusion_matrix.png")
        print("  4. retrieval_examples.png")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase1', type=str, default='checkpoints/phase1_v2_best.pth')
    parser.add_argument('--phase2', type=str, default='checkpoints/phase2_university_best.pth')
    parser.add_argument('--data', type=str, default='data/University-Release')
    
    args = parser.parse_args()
    
    evaluator = FinalEvaluator(args.phase1, args.phase2, args.data)
    evaluator.run()


if __name__ == "__main__":
    main()