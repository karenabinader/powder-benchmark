"""Quick sanity check for PowderDataset."""

from pathlib import Path
from collections import Counter

from dataset import PowderDataset, get_class_names


def main():
    data_root = Path(__file__).parent.parent / "data"

    print(f"Looking for data in: {data_root}\n")

    for split in ["train", "test"]:
        ds = PowderDataset(data_root, split=split)
        print(f"=== {split.upper()} ===")
        print(f"  Total images: {len(ds)}")

        # Count per class
        labels = [label for _, label in ds.samples]
        counts = Counter(labels)
        class_names = get_class_names()
        for idx in range(8):
            print(f"  Class {class_names[idx]} (label={idx}): {counts[idx]} images")

        # Inspect one sample
        img, label = ds[0]
        print(f"  Sample 0: label={label} ({class_names[label]}), "
              f"image size={img.size}, mode={img.mode}")
        print()


if __name__ == "__main__":
    main()