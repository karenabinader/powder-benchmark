"""
PyTorch dataset for the DeCost-Holm 2016 synthetic powder dataset.

The dataset has 8 classes (a-h), each with 256 PNG images in a 'renders/' subfolder.
Following DeCost & Holm 2016, we split deterministically:
  - Training:  first 128 images per class (1, 2, ..., 128)
  - Testing:   last  128 images per class (129, 130, ..., 256)
"""

from pathlib import Path
from typing import Literal
import re

import torch
from torch.utils.data import Dataset
from PIL import Image


# Map folder prefix letter -> integer class label
CLASS_LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h"]
CLASS_TO_IDX = {letter: idx for idx, letter in enumerate(CLASS_LETTERS)}


class PowderDataset(Dataset):
    """8-class synthetic powder dataset."""

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "test"] = "train",
        transform=None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data root not found: {self.data_root}\n"
                f"Make sure the dataset is extracted to this location."
            )

        # Discover the 8 class folders by their starting letter
        self.samples = self._discover_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {self.data_root}. "
                f"Check that class folders contain a 'renders/' subdirectory with .png files."
            )

    def _discover_samples(self) -> list[tuple[Path, int]]:
        """Walk the data folder and collect (image_path, class_label) pairs."""
        samples = []

        for class_folder in sorted(self.data_root.iterdir()):
            if not class_folder.is_dir():
                continue

            # Folder name starts with the class letter, e.g. 'a-lognormal-loc0.1-shape0.5'
            first_char = class_folder.name[0].lower()
            if first_char not in CLASS_TO_IDX:
                continue
            class_idx = CLASS_TO_IDX[first_char]

            renders_dir = class_folder / "renders"
            if not renders_dir.is_dir():
                continue

            # Sort images by their numeric suffix: particles1.png, particles2.png, ...
            png_files = sorted(
                renders_dir.glob("*.png"),
                key=lambda p: int(re.search(r"\d+", p.stem).group()),
            )

            # Deterministic split: first 128 -> train, last 128 -> test
            if self.split == "train":
                selected = png_files[:128]
            elif self.split == "test":
                selected = png_files[128:256]
            else:
                raise ValueError(f"Unknown split: {self.split}")

            for img_path in selected:
                samples.append((img_path, class_idx))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, int]:
        img_path, label = self.samples[idx]
        # Open as grayscale ('L') — SEM images are single-channel
        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_class_names() -> list[str]:
    """Return the 8 class names in label order."""
    return CLASS_LETTERS.copy()