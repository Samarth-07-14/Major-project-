"""Shared training harness for the variant experiments. Reuses the data pipeline, metrics and
input mapping from the main codebase (src/) unchanged; each variant supplies its own model,
loss, optimizer and schedule through `run_variant`.

Variant outputs go to variants/<name>/outputs/ so the main model's checkpoints/ and
results/ are never overwritten.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.data.dataset import CLASSES, read_manifest
from src.data.multistream_dataset import MultiStreamDataset, collate_multistream
from src.eval.metrics import compute_metrics, format_report
from src.train.train_multistream import get_device, inputs_for_branches

SEED = 42
BATCH_SIZE = 32
VAL_FRAC = 0.10
ALL_STREAMS = ["rgb", "veg", "texture", "clip"]


def run_epoch(model, loader, streams, device, loss_fn, optimizer=None, scheduler=None):
    """loss_fn(model_output, labels) -> scalar. The model output may be a tensor of logits or a
    tuple whose first element is the logits (e.g. (logits, aux_logits))."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0
    all_true, all_pred, all_proba = [], [], []
    for batch_inputs, labels in loader:
        labels = labels.to(device)
        model_inputs = inputs_for_branches(batch_inputs, streams, device)
        with torch.set_grad_enabled(is_train):
            out = model(model_inputs)
            loss = loss_fn(out, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:  # per-step schedulers (e.g. OneCycle)
                    scheduler.step()
        logits = out[0] if isinstance(out, tuple) else out
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
        proba = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_proba.append(proba)
        all_pred.append(proba.argmax(axis=1))
        all_true.append(labels.cpu().numpy())
    metrics = compute_metrics(np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_proba))
    metrics["loss"] = total_loss / n
    return metrics


def make_loaders(streams, rgb_train_tf=None):
    train_items = read_manifest("train")
    labels_str = [c for _, c in train_items]
    train_idx, val_idx = train_test_split(
        list(range(len(train_items))), test_size=VAL_FRAC, random_state=SEED, stratify=labels_str
    )
    rgb_train_tf = rgb_train_tf or transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    rgb_eval_tf = transforms.Compose([transforms.ToTensor()])
    st = tuple(streams)
    train_ds = Subset(MultiStreamDataset("train", streams=st, rgb_transform=rgb_train_tf), train_idx)
    val_ds = Subset(MultiStreamDataset("train", streams=st, rgb_transform=rgb_eval_tf), val_idx)
    test_ds = MultiStreamDataset("test", streams=st, rgb_transform=rgb_eval_tf)
    kw = dict(batch_size=BATCH_SIZE, collate_fn=collate_multistream)
    return (
        DataLoader(train_ds, shuffle=True, num_workers=4, **kw),
        DataLoader(val_ds, shuffle=False, num_workers=2, **kw),
        DataLoader(test_ds, shuffle=False, num_workers=2, **kw),
        Counter(labels_str),
    )


def run_variant(
    tag,
    out_dir,
    build_model,
    build_loss,
    build_optimizer,
    streams=ALL_STREAMS,
    max_epochs=25,
    patience=5,
    min_epochs=10,
    rgb_train_tf=None,
):
    """build_model(streams) -> nn.Module
    build_loss(class_counts, device) -> loss_fn(output, labels)
    build_optimizer(model, steps_per_epoch, max_epochs) -> (optimizer, scheduler, step_per_batch: bool)
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    print(f"[{tag}] device={device} streams={streams}")

    train_loader, val_loader, test_loader, counts = make_loaders(streams, rgb_train_tf)
    model = build_model(streams).to(device)
    print(f"[{tag}] params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    loss_fn = build_loss(counts, device)
    optimizer, scheduler, per_batch = build_optimizer(model, len(train_loader), max_epochs)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{tag}_best.pt"
    best_val, no_improve, history = float("inf"), 0, []

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        tr = run_epoch(model, train_loader, streams, device, loss_fn, optimizer, scheduler if per_batch else None)
        va = run_epoch(model, val_loader, streams, device, loss_fn)
        if not per_batch:
            scheduler.step()
        print(
            f"[{tag}] epoch {epoch:2d}/{max_epochs}  train_loss={tr['loss']:.4f} "
            f"train_acc={tr['accuracy']*100:.2f}%  val_loss={va['loss']:.4f} val_acc={va['accuracy']*100:.2f}%"
        )
        history.append(
            {"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["accuracy"],
             "val_loss": va["loss"], "val_acc": va["accuracy"]}
        )
        if va["loss"] < best_val - 1e-5:
            best_val, no_improve = va["loss"], 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience and epoch >= min_epochs:
                print(f"[{tag}] Early stopping at epoch {epoch}.")
                break
    print(f"[{tag}] Training done in {(time.time()-t0)/60:.1f} min. Best val_loss={best_val:.4f}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test = run_epoch(model, test_loader, streams, device, loss_fn)
    report = format_report(test, title=f"{tag} - TEST SET")
    print("\n" + report)
    (out_dir / f"{tag}_test_metrics.json").write_text(json.dumps(test, indent=2))
    (out_dir / f"{tag}_history.json").write_text(json.dumps(history, indent=2))
    (out_dir / f"{tag}_report.txt").write_text(report + "\n")
