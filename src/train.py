"""
Training module for the powder benchmark.

One function `train_model` handles training a single (model, seed) configuration.
The same function will be used for ResNet-18, ConvNeXt, ViT, etc. — just pass a different model_name.

Designed to run on both CPU (small debugging runs) and GPU (real training).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from dataset import PowderDataset, get_class_names


# ----------------------------------------------------------------------
# Config dataclass — everything that defines a single run lives here
# ----------------------------------------------------------------------
@dataclass
class TrainConfig:
    model_name: str = "resnet18"
    pretrained: bool = True
    image_size: int = 224
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    debug_subset: Optional[int] = None
    output_dir: str = "../results/runs"
    data_root: Optional[str] = None   # ← ADD THIS LINE


# ----------------------------------------------------------------------
# Reproducibility helpers
# ----------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Transforms — grayscale images → 3-channel for pretrained models
# ----------------------------------------------------------------------
def build_transforms(image_size: int, train: bool):
    """
    Build image transforms.

    Pretrained models on ImageNet expect 3-channel RGB input normalized
    with ImageNet statistics. Our images are grayscale, so we replicate
    the single channel three times.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),                              # → [1, H, W] in [0, 1]
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),     # → [3, H, W]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


# ----------------------------------------------------------------------
# One epoch of training or evaluation
# ----------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="train" if train else "eval", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, preds)

    return avg_loss, acc, preds, labels


# ----------------------------------------------------------------------
# Main training function — call this for each (model, seed) combo
# ----------------------------------------------------------------------
def train_model(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Run: {cfg.model_name} | seed={cfg.seed} | device={device}")
    if cfg.debug_subset:
        print(f"DEBUG MODE: using {cfg.debug_subset} training images only")
    print(f"{'='*60}\n")

    # Paths
    # Use config-specified path if given, otherwise default to repo-relative
    if cfg.data_root is not None:
        data_root = Path(cfg.data_root)
    else:
        data_root = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent / cfg.output_dir
    run_name = f"{cfg.model_name}_seed{cfg.seed}"
    if cfg.debug_subset:
        run_name += f"_debug{cfg.debug_subset}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_tf = build_transforms(cfg.image_size, train=True)
    test_tf = build_transforms(cfg.image_size, train=False)

    train_ds = PowderDataset(data_root, split="train", transform=train_tf)
    test_ds = PowderDataset(data_root, split="test", transform=test_tf)

    # Optionally shrink the training set for fast debugging
    if cfg.debug_subset is not None:
        indices = list(range(cfg.debug_subset))
        train_ds = Subset(train_ds, indices)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )

    # Model — timm gives us pretrained weights + correct final layer
    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=8,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Training loop
    history = []
    best_test_acc = 0.0
    start_time = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        test_loss, test_acc, test_preds, test_labels = run_epoch(
            model, test_loader, criterion, optimizer, device, train=False
        )
        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time_sec": epoch_time,
        })

        print(
            f"Epoch {epoch:2d}/{cfg.num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"{epoch_time:.1f}s"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Save best confusion matrix
            best_cm = confusion_matrix(test_labels, test_preds)
            best_preds = test_preds
            best_labels = test_labels

    total_time = time.time() - start_time

    # Save everything for this run
    results = {
        "config": asdict(cfg),
        "history": history,
        "best_test_acc": float(best_test_acc),
        "final_test_acc": float(history[-1]["test_acc"]),
        "total_time_sec": total_time,
        "n_params": int(n_params),
        "device": str(device),
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez(
        run_dir / "predictions.npz",
        confusion_matrix=best_cm,
        y_true=best_labels,
        y_pred=best_preds,
    )

    print(f"\nFinished {run_name}")
    print(f"  Best test accuracy: {best_test_acc:.4f}")
    print(f"  Final test accuracy: {history[-1]['test_acc']:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Saved to: {run_dir}\n")

    return results


# ----------------------------------------------------------------------
# Entry point for command-line use
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Default: quick debug run on CPU to verify the pipeline works
    cfg = TrainConfig(
        model_name="resnet18",
        num_epochs=3,
        debug_subset=64,    # only 64 training images
        seed=42,
    )
    train_model(cfg)