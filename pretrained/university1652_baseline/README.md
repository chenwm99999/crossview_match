# University-1652 Baseline Models

## Source
- Paper: University-1652 dataset paper
- Repository: https://github.com/layumi/University1652-Baseline
- Downloaded: 2025-12-01
- Choose three_view_long_share_d0.75_256_s1_google

## Files
- `resnet50_uni1652.pth` - ResNet-50 trained on University-1652
  - Architecture: ResNet-50
  - Performance: ~86.7% R@1
  - Use: Baseline comparison for ablation study

## Notes
- Different architecture (ResNet vs Swin)
- Cannot directly use weights for initialization
- Use for performance comparison only