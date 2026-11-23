"""Variant 4 training: fused cross-entropy + 0.3 * mean auxiliary per-stream cross-entropy,
SGD with Nesterov momentum and cosine decay."""
import torch

from src.data.dataset import CLASSES
from variants.common import run_variant
from variants.v4_deep_supervision.model import DeepSupervisionModel

AUX_WEIGHT = 0.3


def build_loss(counts, device):
    ce = torch.nn.CrossEntropyLoss()

    def loss_fn(out, y):
        logits, aux = out
        loss = ce(logits, y)
        if aux:
            loss = loss + AUX_WEIGHT * sum(ce(a, y) for a in aux) / len(aux)
        return loss

    return loss_fn


def build_optimizer(model, steps_per_epoch, max_epochs):
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-4)
    return opt, sched, False


if __name__ == "__main__":
    run_variant(
        tag="v4_deep_supervision",
        out_dir="variants/v4_deep_supervision/outputs",
        build_model=lambda s: DeepSupervisionModel(s, num_classes=len(CLASSES)),
        build_loss=build_loss,
        build_optimizer=build_optimizer,
    )
