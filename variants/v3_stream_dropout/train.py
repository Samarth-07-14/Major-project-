"""Variant 3 training: AdamW + OneCycle (stepped per batch), cross-entropy, stronger RGB augmentation."""
import torch
from torchvision import transforms

from src.data.dataset import CLASSES
from variants.common import run_variant
from variants.v3_stream_dropout.model import StreamDropoutModel


def build_loss(counts, device):
    ce = torch.nn.CrossEntropyLoss()
    return lambda out, y: ce(out, y)


def build_optimizer(model, steps_per_epoch, max_epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, epochs=max_epochs, steps_per_epoch=steps_per_epoch, pct_start=0.2
    )
    return opt, sched, True


if __name__ == "__main__":
    rgb_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]
    )
    run_variant(
        tag="v3_stream_dropout",
        out_dir="variants/v3_stream_dropout/outputs",
        build_model=lambda s: StreamDropoutModel(s, num_classes=len(CLASSES), p_drop=0.2),
        build_loss=build_loss,
        build_optimizer=build_optimizer,
        rgb_train_tf=rgb_tf,
    )
