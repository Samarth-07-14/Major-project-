"""Variant 4: deep supervision. Each stream vector gets its own linear auxiliary classifier; the
fused head sees LayerNorm'd concatenated features. The model always returns
(logits, [aux_logits...]); the harness uses the first element for metrics."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.fusion import STREAM_DIM
from src.models.streams.clip_semantic import ClipBranch
from src.models.streams.texture import TextureBranch
from src.models.streams.veg_color import VegColorBranch
from src.models.streams.vit_cab import CABBranch, ViTBranch


class DeepSupervisionModel(nn.Module):
    def __init__(self, streams: list[str], num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
        b = {}
        if "rgb" in streams:
            b["rgb_vit"] = ViTBranch(num_blocks=3)
            b["rgb_cab"] = CABBranch()
        if "veg" in streams:
            b["veg"] = VegColorBranch()
        if "texture" in streams:
            b["texture"] = TextureBranch()
        if "clip" in streams:
            b["clip"] = ClipBranch()
        self.stream_names = list(b.keys())
        self.branches = nn.ModuleDict(b)
        self.aux_heads = nn.ModuleDict({n: nn.Linear(STREAM_DIM, num_classes) for n in b})
        in_dim = len(b) * STREAM_DIM
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs):
        feats = [self.branches[n](inputs[n]) for n in self.stream_names]
        h = self.dropout(torch.relu(self.fc(self.norm(torch.cat(feats, dim=1)))))
        logits = self.classifier(h)
        aux = [self.aux_heads[n](f) for n, f in zip(self.stream_names, feats)]
        return logits, aux
