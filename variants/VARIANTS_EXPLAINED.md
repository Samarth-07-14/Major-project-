# The Original Model and the Four Variants — Explained

This document describes five things: the original (trained, working) model in `src/`, followed by four
code-only variants in `variants/`. The variants have **not** been trained or run; they exist to show
slightly different design approaches. The original codebase is unchanged.

The task throughout is 4-class aerial/field image classification: **Broadleaf, Grass, Soil, Soybean**
(3832 train / 959 test images, 224×224 RGB).

---

## 0. Original model — 4-stream attention-fusion network (`src/`)

**Result:** 99.48% test accuracy, 0.9998 AUC (5 errors on 959 test images). Trained with seed 42.

### Idea
A single RGB network (the paper's baseline) reaches ~92%. The key insight of this project is that the
classes are distinguished by cues a plain RGB network doesn't learn well from ~4,800 images: vegetation
colour indices, fine texture, and general "world knowledge" of what plants look like. So the model runs
**four parallel streams**, each producing a 64-dimensional feature vector, and fuses them.

### The streams

| Stream | Input | Network | Output |
|---|---|---|---|
| **1. RGB** (two sub-branches) | Raw 224×224 RGB | **ViT branch:** 16×16 patch conv embedding (64-d) + positional embedding, 2 transformer encoder blocks (4 heads, key_dim 128, FFN 128→64, residual + LayerNorm), mean-pool over patches. **CAB branch:** 2× (3×3 Conv → BatchNorm → ReLU), then squeeze-and-excitation channel attention, global-average-pool. | 2 × 64-d |
| **2. Vegetation / colour** | 9-channel stack: ExG, ExG−ExR, VARI, HSV (3), Lab (3) | 3× (Conv3×3 → BN → ReLU) with two 2×2 max-pools, global-average-pool | 64-d |
| **3. Texture** | 12-channel stack: 4 GLCM maps (contrast, homogeneity, energy, correlation) + 8 Gabor responses (4 orientations × 2 frequencies) | Same 3-conv CNN as Stream 2 | 64-d |
| **4. CLIP semantic** | Frozen CLIP ViT-B/32 image embedding (512-d, precomputed and cached) | Small MLP 512 → 128 → 64 | 64-d |

Streams 2–4 inputs are computed once ahead of time (`src/data/precompute_features.py`) and cached to
disk, so training only runs the small learned branches on top of them.

### Fusion (the final model uses **attention fusion**)
The five 64-d vectors (ViT, CAB, veg, texture, CLIP) are concatenated (320-d). A small gating MLP
(320 → 128 → 5, softmax) produces **one weight per stream, per sample**. Each stream vector is scaled by its
weight, the scaled vectors are re-concatenated, and passed through Dense(128) → ReLU → Dropout(0.4) →
Dense(4). This lets the model down-weight a stream that is uninformative for a particular image.
(A plain concat-fusion head is also implemented; it scored 99.06%.)

### Training recipe
- Loss: focal loss (γ=2) with inverse-frequency class weights (targets the weaker Grass class).
- Optimizer: Adam, LR 1e-3 with cosine decay to 1e-5.
- Up to 25 epochs, early stopping on validation loss with patience 5, but not before epoch 10.
- 10% of train held out for validation; random horizontal flip on the RGB input only.
- Best-validation-loss checkpoint is evaluated on the test set.

### Ablation history (why each piece exists)

| Stage | Accuracy |
|---|---|
| RGB only (Stream 1) | 92.2% |
| + vegetation/colour | 93.0% |
| + texture | 97.6% (biggest single jump) |
| + CLIP, concat fusion | 99.1% |
| + CLIP, attention fusion (**final**) | **99.5%** |

Key files: `src/models/fusion.py` (fusion heads + `MultiStreamModel`), `src/models/streams/*`,
`src/train/train_multistream.py`, `src/train/losses.py`.

---

## How the variants are organised

Each variant keeps the **same four input streams and the same data pipeline**, and changes one or two
design choices (fusion mechanism, branch architecture, regularisation, loss/optimiser). They all reuse
code from `src/` and share a training loop in `variants/common.py`, which differs from the original loop
only in being configurable (model, loss and optimiser/scheduler are supplied by each variant). Results
would be written to `variants/<name>/outputs/`, never over the original checkpoints or results.

Each variant folder contains `model.py` (architecture) and `train.py` (loss, optimiser, schedule, entry
point). Run with `python -m variants.<folder>.train` — *not run yet*.

---

## 1. `v1_transformer_fusion` — transformer instead of gating

**What changes:** the fusion head. The original computes a softmax weight per stream and takes a weighted
concatenation. Here each stream's 64-d vector is treated as a **token**: a learned stream-ID embedding is
added, and a one-layer transformer encoder (4 heads, FFN 128, pre-LayerNorm, dropout 0.3) lets the
streams **attend to each other** before the tokens are mean-pooled and classified by a small MLP
(64 → 128 → 4).

**Why it's different:** gating scores each stream on its own from a single MLP over the whole concatenation;
self-attention lets, say, the texture token modulate how the CLIP token is read. It is a richer
interaction model, at the cost of more parameters in the fusion stage and somewhat more risk of overfitting
on a small dataset.

**Training:** unchanged from the original — focal loss (γ=2, class-weighted), Adam 1e-3, cosine decay,
25 epochs. This isolates the fusion change.

---

## 2. `v2_lightweight_streams` — smaller, cheaper branches

**What changes:** the branches get lighter.
- Veg and texture branches use **depthwise-separable convolutions** (depthwise 3×3 + pointwise 1×1) with
  **GELU** instead of standard 3×3 convs with ReLU. This cuts their parameter and compute cost
  substantially while keeping the same 64-d output.
- The ViT branch has **1 encoder block** instead of 2.
- Fusion is the plain **concat** head (no gating).

**Why it's different:** it tests how much of the original's accuracy comes from branch capacity versus
from the streams' information content. It would train faster and be easier to deploy, and may lose a
little accuracy.

**Training:** class-weighted cross-entropy with **label smoothing 0.1** (instead of focal loss), Adam with
a higher LR of 2e-3, cosine decay, 20 epochs.

---

## 3. `v3_stream_dropout` — forcing the model not to over-rely on one stream

**What changes:** regularisation. Keeps the original's attention fusion, but during training each stream's
vector is **randomly zeroed per sample with probability 0.2** (modality dropout). At least one stream is
always kept, and surviving streams are rescaled so the expected magnitude matches inference time. At eval
time nothing is dropped.

**Why it's different:** the original may lean heavily on the strongest stream (likely CLIP or texture).
Stream dropout makes every stream learn to be useful on its own and makes the model robust if a stream is
noisy or missing — at the possible cost of slightly lower peak accuracy, since the model is sometimes
trained with less information.

**Training:** plain cross-entropy (no class weighting), **AdamW** (weight decay 1e-2) with a **OneCycle**
learning-rate schedule (peak 2e-3, stepped every batch, 20% warm-up), and stronger RGB augmentation:
horizontal + vertical flips and brightness/contrast jitter (0.2).

---

## 4. `v4_deep_supervision` — every stream must classify on its own

**What changes:** the supervision signal and fusion normalisation.
- Each of the five stream vectors gets its **own auxiliary linear classifier** (64 → 4). The training loss is
  `CE(fused) + 0.3 × mean(CE(aux_i))`, so every branch is pushed to be discriminative by itself rather than
  relying on the fusion head to sort things out.
- The ViT branch is deeper (**3 encoder blocks**).
- Fusion is concat with a **LayerNorm** on the concatenated features (no gating) → Dense(128) → ReLU →
  Dropout(0.4) → Dense(4).
- The model returns `(logits, [aux_logits…])`; metrics use the fused logits only.

**Why it's different:** deep supervision is a classic way to improve gradient flow into each branch and
stop a dominant stream from starving the others of learning signal. Normalising before fusion also
removes scale differences between streams.

**Training:** **SGD with Nesterov momentum** (LR 0.05, momentum 0.9, weight decay 5e-4) and cosine decay,
instead of Adam — a deliberately different optimiser family.

---

## Side-by-side summary

| | Original | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| Fusion | Attention gating | Transformer over stream tokens | Concat | Attention gating | Concat + LayerNorm |
| ViT blocks | 2 | 2 | 1 | 2 | 3 |
| Veg/texture conv | Standard, ReLU | Standard, ReLU | Depthwise-separable, GELU | Standard, ReLU | Standard, ReLU |
| Extra regularisation | Dropout 0.4 | Dropout 0.3 | Label smoothing 0.1 | Stream dropout p=0.2 | Auxiliary per-stream losses |
| Loss | Focal (weighted) | Focal (weighted) | Weighted CE + smoothing | CE | CE + 0.3·aux CE |
| Optimiser | Adam 1e-3 | Adam 1e-3 | Adam 2e-3 | AdamW + OneCycle | SGD-Nesterov 0.05 |
| Epochs (max) | 25 | 25 | 20 | 25 | 25 |
| Trained? | Yes — 99.48% | No | No | No | No |
