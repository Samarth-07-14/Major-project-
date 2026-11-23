"""Variant 1 training: transformer fusion, focal loss, Adam + cosine (same schedule as main model)."""
import torch

from src.data.dataset import CLASSES
from src.train.losses import FocalLoss, class_weights_from_counts
from variants.common import run_variant
from variants.v1_transformer_fusion.model import TransformerFusionModel


def build_loss(counts, device):
    alpha = class_weights_from_counts(dict(counts), CLASSES).to(device)
    focal = FocalLoss(gamma=2.0, alpha=alpha)
    return lambda out, y: focal(out, y)


def build_optimizer(model, steps_per_epoch, max_epochs):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    return opt, sched, False


if __name__ == "__main__":
    run_variant(
        tag="v1_transformer_fusion",
        out_dir="variants/v1_transformer_fusion/outputs",
        build_model=lambda s: TransformerFusionModel(s, num_classes=len(CLASSES)),
        build_loss=build_loss,
        build_optimizer=build_optimizer,
    )
