import torch

# Load both
best = torch.load('checkpoints/phase1_v2_best.pth')
epoch100 = torch.load('checkpoints/phase1_v2_epoch_100.pth')

print("phase1_v2_best.pth:")
print(f"  Epoch: {best['epoch']}")
print(f"  R@1: {best['best_recall']*100:.2f}%")

print("\nphase1_v2_epoch_100.pth:")
print(f"  Epoch: {epoch100['epoch']}")
print(f"  R@1: {epoch100.get('best_recall', epoch100.get('current_recalls', {}).get('R@1', 0))*100:.2f}%")

print("\nUse:", "epoch_100" if epoch100['epoch'] == 99 else "best")