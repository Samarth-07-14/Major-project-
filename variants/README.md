# Model variants (code only — not trained)

Four small deviations from the main 4-stream model in `src/`. The main codebase is untouched;
variants import its data pipeline, branches and metrics, and share a training harness in
`common.py`. Outputs would be written to `variants/<name>/outputs/`, never to `checkpoints/` or `results/`.

| Folder | What differs from the main model |
|---|---|
| `v1_transformer_fusion` | Fusion head: streams become tokens, a small transformer encoder mixes them (replaces softmax gating). Focal loss, Adam + cosine. |
| `v2_lightweight_streams` | Veg/texture branches use depthwise-separable convs + GELU; ViT branch uses 1 block; concat fusion; weighted CE with label smoothing. |
| `v3_stream_dropout` | Whole streams are randomly zeroed during training (modality dropout) before attention fusion; AdamW + OneCycle; stronger RGB augmentation. |
| `v4_deep_supervision` | 3-block ViT; each stream gets an auxiliary classifier whose loss is added to the fused loss; concat fusion with LayerNorm; SGD + momentum. |

Run (needs `src.data.precompute_features` already run):

    python -m variants.v1_transformer_fusion.train
