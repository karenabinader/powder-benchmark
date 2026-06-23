"""
Hyperparameter search for the powder-classification thesis.

Stage 1 of the two-stage tuning plan: for one architecture, sweep
optimizer x learning-rate x batch-size and pick the best combination by
*validation* accuracy. The validation set is carved out of the TRAINING
images only, so the 1024-image test set is never used for selection.

Reuses the project's own transforms / training loop so the search is
consistent with the final training runs.

Stage 2 (run the winning config across 5 seeds on the full training set
and report on the test set) is done separately with train_model().
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Subset

from train import set_seed, build_transforms, run_epoch
from dataset import PowderDataset


def make_stratified_train_val(data_root, val_frac=0.2, split_seed=123):
    """Split the TRAIN images into train'/val index lists, balanced per class."""
    base = PowderDataset(data_root, split="train", transform=None)
    by_class = defaultdict(list)
    for idx, (_path, label) in enumerate(base.samples):
        by_class[label].append(idx)

    rng = np.random.RandomState(split_seed)
    train_idx, val_idx = [], []
    for label, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_train = int(round(len(idxs) * (1.0 - val_frac)))
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:])
    return train_idx, val_idx


def build_optimizer(name, params, lr, weight_decay=1e-4):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def _make_model(model_name, image_size, device):
    kwargs = dict(pretrained=True, num_classes=8)
    if any(t in model_name for t in ("vit", "swin")):
        kwargs["img_size"] = image_size
    return timm.create_model(model_name, **kwargs).to(device)


def search_one(model_name, optimizer, lr, batch_size, image_size, epochs,
               data_root, out_dir, split_seed=123, train_seed=42,
               pretrained=True, num_workers=2):
    """Train one config on train', return + save its best validation accuracy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_name}__{optimizer}__lr{lr:g}__bs{batch_size}__size{image_size}"
    out_path = out_dir / f"{tag}.json"
    if out_path.exists():
        with open(out_path) as f:
            res = json.load(f)
        print(f"SKIP (done): {tag}  best_val={res['best_val_acc']:.4f}")
        return res

    set_seed(train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx, val_idx = make_stratified_train_val(data_root, split_seed=split_seed)
    train_ds = Subset(PowderDataset(data_root, "train", build_transforms(image_size, True)), train_idx)
    val_ds = Subset(PowderDataset(data_root, "train", build_transforms(image_size, False)), val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    kwargs = dict(pretrained=pretrained, num_classes=8)
    if any(t in model_name for t in ("vit", "swin")):
        kwargs["img_size"] = image_size
    model = timm.create_model(model_name, **kwargs).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = build_optimizer(optimizer, model.parameters(), lr)

    print(f"\n--- {tag} ---")
    best_val = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        _, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optim, device, True)
        _, va_acc, _, _ = run_epoch(model, val_loader, criterion, optim, device, False)
        best_val = max(best_val, va_acc)
        print(f"  epoch {epoch:2d}/{epochs}  train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}")

    res = {
        "tag": tag, "model": model_name, "optimizer": optimizer, "lr": lr,
        "batch_size": batch_size, "image_size": image_size, "epochs": epochs,
        "best_val_acc": float(best_val), "minutes": round((time.time() - t0) / 60, 1),
    }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  -> best_val_acc = {best_val:.4f}  ({res['minutes']} min)")
    return res


def run_search(model_name, data_root, out_dir, image_size=512, epochs=10,
               batch_options=(32, 16)):
    """Sweep optimizer x lr x batch for one model. Returns ranked results."""
    # optimizer-appropriate learning rates (AdamW small, SGD larger)
    grid = [("adamw", lr) for lr in (1e-3, 1e-4, 1e-5)] + \
           [("sgd", lr) for lr in (1e-1, 1e-2, 1e-3)]

    print("=" * 60)
    print(f"HP search: {model_name} @ {image_size}, {len(grid) * len(batch_options)} configs, "
          f"{epochs} epochs each (1 seed, validation split)")
    print("=" * 60)

    results = []
    for opt_name, lr in grid:
        for bs in batch_options:
            batch = bs
            while batch >= 4:
                try:
                    results.append(search_one(model_name, opt_name, lr, batch,
                                               image_size, epochs, data_root, out_dir))
                    break
                except RuntimeError as e:
                    torch.cuda.empty_cache()
                    if "out of memory" in str(e).lower() and batch > 4:
                        print(f"  OOM at batch {batch} -> retry at {batch // 2}")
                        batch //= 2
                    else:
                        print(f"  !! {opt_name} lr={lr:g} bs={bs} failed: {e}")
                        break

    results.sort(key=lambda r: r["best_val_acc"], reverse=True)
    print("\n" + "=" * 60)
    print(f"RANKED RESULTS for {model_name} (by validation accuracy)")
    print("=" * 60)
    for r in results:
        print(f"  val={r['best_val_acc']:.4f}  {r['optimizer']:<5} "
              f"lr={r['lr']:<7g} bs={r['batch_size']:<3}  ({r['minutes']} min)")
    if results:
        best = results[0]
        print("\nBEST CONFIG:")
        print(f"  optimizer = {best['optimizer']}")
        print(f"  lr        = {best['lr']:g}")
        print(f"  batch     = {best['batch_size']}")
        print(f"  val_acc   = {best['best_val_acc']:.4f}")
        print("\nUse these in the Stage-2 (5-seed) run.")
    return results
