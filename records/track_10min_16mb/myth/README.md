# RecurrentGPT — Depth Recurrence Submission

## Architecture
- **RecurrentGPT**: 6 physical transformer layers × 4 recurrence passes = 24 effective depth
- **Model dim**: 576, GQA 9 query heads / 3 KV heads (head_dim=64)
- **SwiGLU FFN**: 3x expansion (hidden=1728)
- **BigramHash Embedding**: hash_size=12288, proj_dim=128→576
- **Tied embeddings** with RoPE (base=10000)
- **Learnable residual mixing** (mix_alpha per recurrence pass)
- **Logit softcap**: 30.0

## Quantization
Mixed-precision post-training quantization:
- **Embedding**: FP16 (preserves token representation fidelity)
- **Attention Q/K**: Int6 (higher precision for attention patterns)
- **FFN/MLP weights**: Int5 (compression priority)
- **Other weights**: Int8
- **Scaling factors**: Int8
- **Compression**: zlib-9

## Results (1×H100 SXM, 10,000 steps)

| Metric | Value |
|--------|-------|
| val_bpb (pre-quantization) | **1.1934** |
| val_bpb (post-quantization) | **1.3246** |
| Quantization penalty | +0.1312 |
| Compressed artifact size | 15.9 MB |
| Parameters | 25.5M |
| Step speed | 1,267 ms/step |

### Validation BPB over training
| Step | val_bpb |
|------|---------|
| 1,000 | 1.3790 |
| 2,000 | 1.3153 |
| 3,000 | 1.2861 |
| 4,000 | 1.2675 |
| 5,000 | 1.2542 |
| 6,000 | 1.2450 |
| 7,000 | 1.2362 |
| 8,000 | 1.2284 |
| 9,000 | 1.2201 |
| 10,000 | 1.1934 |

## Known Issues & Next Steps
- **Quantization penalty is too large (+0.13)**. Pre-quantization BPB (1.1934) is competitive, but post-quantization degrades significantly.
- Planned improvements:
  - Stochastic Weight Averaging (SWA) for smoother weight distributions
  - Quantization-Aware Training (QAT) with Straight-Through Estimator
  - Weight decay (0.04) to constrain weight magnitudes
  - Gradient clipping (0.3) to reduce outlier weights
  - Higher Muon momentum (0.95 → 0.99)

## Reproduction
```bash
# 1×H100:
torchrun --standalone --nproc_per_node=1 records/track_10min_16mb/myth/train_gpt.py

# 8×H100 (official):
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/myth/train_gpt.py
```

## Key Design Choice: Depth Recurrence
Instead of using 9-11 independent layers like other submissions, RecurrentGPT reuses 6 physical layers 4 times. This trades unique per-layer parameters for increased effective depth within the same parameter budget. Each recurrence pass has a learnable mixing coefficient to control how much of the original embedding versus the refined representation is used.
