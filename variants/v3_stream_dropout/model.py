"""Variant 3: stream (modality) dropout before attention fusion. During training each stream's
feature vector is zeroed with probability p per sample (never all streams at once), so the
fusion head can't over-rely on any single stream (e.g. CLIP)."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.fusion import AttentionFusion
from src.train.train_multistream import build_branches


class StreamDropoutModel(nn.Module):
    def __init__(self, streams: list[str], num_classes: int = 4, p_drop: float = 0.2):
        super().__init__()
        branches = build_branches(streams)
        self.stream_names = list(branches.keys())
        self.branches = nn.ModuleDict(branches)
        self.fusion = AttentionFusion(len(branches), num_classes)
        self.p_drop = p_drop

    def _mask(self, batch: int, n: int, device):
        keep = (torch.rand(batch, n, device=device) >= self.p_drop).float()
        dead = keep.sum(dim=1) == 0  # guarantee at least one stream survives
        if dead.any():
            keep[dead, torch.randint(0, n, (int(dead.sum()),), device=device)] = 1.0
        return keep

    def forward(self, inputs):
        feats = [self.branches[n](inputs[n]) for n in self.stream_names]
        if self.training and self.p_drop > 0:
            keep = self._mask(feats[0].size(0), len(feats), feats[0].device)
            # rescale so the expected magnitude matches eval time
            scale = len(feats) / keep.sum(dim=1, keepdim=True)
            feats = [f * keep[:, i : i + 1] * scale for i, f in enumerate(feats)]
        return self.fusion(feats)
