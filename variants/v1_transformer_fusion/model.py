"""Variant 1: replace the softmax-gating fusion with a small transformer encoder over stream tokens."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.fusion import STREAM_DIM
from src.train.train_multistream import build_branches


class TransformerFusion(nn.Module):
    """Each 64-d stream vector is a token (+ learned stream-id embedding); one encoder layer lets
    streams attend to each other; mean-pooled tokens -> MLP classifier."""

    def __init__(self, num_streams: int, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.stream_embed = nn.Parameter(torch.zeros(1, num_streams, STREAM_DIM))
        nn.init.trunc_normal_(self.stream_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=STREAM_DIM, nhead=4, dim_feedforward=128, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(STREAM_DIM)
        self.classifier = nn.Sequential(nn.Linear(STREAM_DIM, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, feats: list[torch.Tensor]):
        tokens = torch.stack(feats, dim=1) + self.stream_embed  # (B, S, 64)
        tokens = self.encoder(tokens)
        return self.classifier(self.norm(tokens.mean(dim=1)))


class TransformerFusionModel(nn.Module):
    def __init__(self, streams: list[str], num_classes: int = 4):
        super().__init__()
        branches = build_branches(streams)
        self.stream_names = list(branches.keys())
        self.branches = nn.ModuleDict(branches)
        self.fusion = TransformerFusion(len(branches), num_classes)

    def forward(self, inputs):
        return self.fusion([self.branches[n](inputs[n]) for n in self.stream_names])
