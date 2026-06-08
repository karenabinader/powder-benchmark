"""
Diagnostic checks on the watershed baseline result.

Investigates:
1. Train/test split correctness (no overlap, correct labels)
2. The actual particle radii being detected (vs the 2016 paper's expected ranges)
3. Per-class accuracy (does it pass the smell test?)
4. Whether the chi-squared SVM is overconfident on certain classes
5. Sanity check: does the histogram look like the 2016 paper's expected output?
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
from pathlib import Path
from collections import Counter
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from sklearn.metrics import confusion_matrix, accuracy_score
from dataset import PowderDataset

OUTPUT_DIR = Path('results/baselines')
PIXELS_PER_BLENDER_UNIT = 512.0 / 11.0

print("=" * 70)
print("WATERSHED RESULT DIAGNOSTIC")
print("=" * 70)
print()

# ----------------------------------------------------------------------
# Check 1 — Data integrity
# ----------------------------------------------------------------------
print("CHECK 1: Train/test split integrity")
print("-" * 70)

ds_train = PowderDataset('data', split='train', transform=None)
ds_test = PowderDataset('data', split='test', transform=None)

train_labels = []
for i in range(len(ds_train)):
    _, l = ds_train[i]
    train_labels.append(l)
test_labels = []
for i in range(len(ds_test)):
    _, l = ds_test[i]
    test_labels.append(l)

train_label_counts = Counter(train_labels)
test_label_counts = Counter(test_labels)

print(f"  Train: {len(ds_train)} images, label distribution:")
for cls in sorted(train_label_counts):
    print(f"    Class {cls}: {train_label_counts[cls]}")
print(f"  Test:  {len(ds_test)} images, label distribution:")
for cls in sorted(test_label_counts):
    print(f"    Class {cls}: {test_label_counts[cls]}")

# A subtle but important check: do the same file paths appear in train and test?
# This would indicate data leakage.
print(f"\n  Checking for path overlap between train and test...")
if hasattr(ds_train, 'image_paths') and hasattr(ds_test, 'image_paths'):
    train_paths = set(str(p) for p in ds_train.image_paths)
    test_paths = set(str(p) for p in ds_test.image_paths)
    overlap = train_paths & test_paths
    if overlap:
        print(f"  ⚠ WARNING: {len(overlap)} files appear in BOTH train and test")
        for p in list(overlap)[:5]:
            print(f"      {p}")
    else:
        print(f"  ✓ No file overlap. Train and test use disjoint files.")
else:
    print(f"  (Cannot directly verify — PowderDataset doesn't expose paths)")

# Sample a few images from each split and check the indices used
print(f"\n  Sample train indices and their labels (first 5):")
for i in range(5):
    print(f"    Train[{i}] -> label {ds_train[i][1]}")
print(f"  Sample test indices and their labels (first 5):")
for i in range(5):
    print(f"    Test[{i}] -> label {ds_test[i][1]}")

# ----------------------------------------------------------------------
# Check 2 — Inspect the actual segmentation on a few images
# ----------------------------------------------------------------------
print("\n\nCHECK 2: What the watershed segmentation actually detects")
print("-" * 70)

def segment_and_measure(img_np):
    thresh = threshold_otsu(img_np)
    binary = img_np > thresh
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=5, labels=binary)
    if len(coords) == 0:
        return np.array([]), binary, np.zeros_like(img_np, dtype=int)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary)
    props = regionprops(labels)
    if len(props) == 0:
        return np.array([]), binary, labels
    areas_px = np.array([p.area for p in props])
    eq_radii_px = np.sqrt(areas_px / np.pi)
    eq_radii_bu = eq_radii_px / PIXELS_PER_BLENDER_UNIT
    return eq_radii_bu, binary, labels

# Sample one image from each class in the train set
print(f"  Sampling one image per class and inspecting segmentation...\n")
sample_indices_by_class = {}
for i in range(len(ds_train)):
    _, lbl = ds_train[i]
    if lbl not in sample_indices_by_class:
        sample_indices_by_class[lbl] = i
    if len(sample_indices_by_class) == 8:
        break

print(f"  {'Class':<6} {'#segs':<7} {'min_r':<7} {'med_r':<7} {'max_r':<7} {'mean_r':<7}")
print(f"  {'(label)':<6} {'(Blender units)':<35}")
for cls in sorted(sample_indices_by_class):
    idx = sample_indices_by_class[cls]
    img, _ = ds_train[idx]
    img_np = np.array(img)
    radii, _, _ = segment_and_measure(img_np)
    if len(radii) == 0:
        print(f"  {cls:<6} {0:<7} (no segments)")
    else:
        print(f"  {cls:<6} {len(radii):<7} {radii.min():<7.3f} {np.median(radii):<7.3f} {radii.max():<7.3f} {radii.mean():<7.3f}")

print(f"\n  Reference: 2016 paper's class mean particle size = 0.10 or 0.12 Blender units")
print(f"  Expected min radii: ~0.02 (small particles)")
print(f"  Expected max radii: ~0.3 (large particles)")
print(f"  If our radii are way outside these ranges, our scale conversion is wrong.")

# ----------------------------------------------------------------------
# Check 3 — Per-class accuracy from saved predictions
# ----------------------------------------------------------------------
print("\n\nCHECK 3: Per-class accuracy (sanity check)")
print("-" * 70)

# Load saved predictions
preds_path = OUTPUT_DIR / "watershed_predictions.npz"
if preds_path.exists():
    data = np.load(preds_path)
    y_true = data['y_true']
    y_pred = data['y_pred']
    overall = accuracy_score(y_true, y_pred)
    print(f"  Overall accuracy:                       {overall:.4f}")
    print(f"  Should match what watershed script reported (0.9492)")
    print()
    print(f"  Per-class recall:")
    for cls in range(8):
        mask = y_true == cls
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == cls).mean()
            n_correct = (y_pred[mask] == cls).sum()
            n_total = mask.sum()
            print(f"    Class {cls}: {cls_acc:.4f} ({n_correct}/{n_total})")
    print()
    print(f"  Confusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  True\\Pred | 0    1    2    3    4    5    6    7")
    print(f"  ----------|" + "-" * 49)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:4d}" for v in row)
        print(f"  Class {i}   | {row_str}")
else:
    print(f"  ⚠ Predictions file not found at {preds_path}")

# ----------------------------------------------------------------------
# Check 4 — Histogram bin distribution sanity check
# ----------------------------------------------------------------------
print("\n\nCHECK 4: Are histogram bins meaningful or empty?")
print("-" * 70)

# Sample 50 train images and check histogram occupancy
print(f"  Sampling 50 random train images...")
np.random.seed(0)
indices = np.random.choice(len(ds_train), size=50, replace=False)
total_hist = np.zeros(25)
for idx in indices:
    img, _ = ds_train[idx]
    img_np = np.array(img)
    radii, _, _ = segment_and_measure(img_np)
    if len(radii) > 0:
        hist, _ = np.histogram(radii, bins=25, range=(0.0, 0.6))
        total_hist += hist

print(f"  Aggregate histogram bin counts (across 50 sample images):")
bin_edges = np.linspace(0.0, 0.6, 26)
for i in range(25):
    bin_low = bin_edges[i]
    bin_high = bin_edges[i+1]
    count = int(total_hist[i])
    bar_len = int(count / 50) if total_hist.max() > 50 else int(count)
    bar = "█" * bar_len
    print(f"    [{bin_low:.3f} - {bin_high:.3f}]  {count:6d}  {bar}")

n_empty_bins = (total_hist == 0).sum()
print(f"\n  Empty bins: {n_empty_bins}/25")
if n_empty_bins > 15:
    print(f"  ⚠ MANY EMPTY BINS — most particle radii fall outside the (0.0, 0.6) range!")
    print(f"     Either particles are bigger/smaller than expected, or scale is wrong.")
elif n_empty_bins > 8:
    print(f"  ⚠ Some empty bins — investigate if this is expected.")
else:
    print(f"  ✓ Histogram occupancy looks reasonable.")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)