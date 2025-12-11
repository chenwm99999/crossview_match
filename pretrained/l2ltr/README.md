# L2LTR Pre-trained Models

## Source
- Paper: Cross-view Geo-localization with Layer-to-Layer Transformer (NeurIPS 2021)
- Repository: https://github.com/yanghongji2007/cross_view_localization_L2LTR
- Downloaded: 2025-12-01
- Choose CVUSA

## Files
- `l2ltr_cvusa.pth` - L2LTR trained on CVUSA dataset
  - Architecture: ViT-Base + L2L cross-attention
  - Performance: ~93.9% R@1 on CVUSA
  - Use: Reference for L2L implementation

## Notes
- Architecture is ViT (not Swin), so cannot directly use weights
- Useful for understanding L2L cross-attention mechanism
- Can compare training strategies and hyperparameters