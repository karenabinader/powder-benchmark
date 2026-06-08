"""
Watershed segmentation + SVM baseline implementation.

Matches the methodology of DeCost & Holm (2016):
- Otsu thresholding to a binary mask
- Distance transform on the foreground
- Watershed segmentation to separate touching particles
- Compute equivalent radius (sqrt(area / pi)) for each segmented region
- Build 25-bin histogram of equivalent radii per image (0.0 to 0.6 range)
- Chi-squared kernel SVM classifier

Saves results to results/baselines/watershed_results.json
"""
import sys
sys.path.insert(0, 'src')

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import accuracy_score, confusion_matrix
from dataset import PowderDataset

# ----------------------------------------------------------------------
# Configuration — matches 2016 paper
# ----------------------------------------------------------------------
N_HIST_BINS = 25
HIST_RANGE = (0.0, 0.6)   # As per 2016 paper, units of "arbitrary Blender units"
                          # We'll express equivalent radii in same units assuming
                          # the image spans 11 Blender units wide (per 2016 paper)
SEED = 42

# The 2016 paper uses Blender units for particle radii. Our images are 512x512
# pixels representing an 11-unit-wide region (per Section 2.1 of the 2016 paper),
# so 1 Blender unit = 512/11 ≈ 46.5 pixels.
PIXELS_PER_BLENDER_UNIT = 512.0 / 11.0

OUTPUT_DIR = Path('results/baselines')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== WATERSHED + chi-squared SVM PIPELINE ===")
print(f"Histogram bins: {N_HIST_BINS} (range {HIST_RANGE} Blender units)")
print(f"Random seed: {SEED}")
print()

np.random.seed(SEED)


def segment_and_measure(img_np):
    """
    Run the watershed pipeline on one grayscale image and return per-particle
    equivalent radii in Blender units.

    Returns: 1D array of equivalent radii (may be empty).
    """
    # 1. Otsu thresholding
    try:
        thresh = threshold_otsu(img_np)
    except ValueError:
        # All pixels the same value
        return np.array([])

    binary = img_np > thresh

    # 2. Distance transform on the foreground
    distance = ndi.distance_transform_edt(binary)

    # 3. Find local maxima as watershed seeds
    # min_distance=5 prevents tiny adjacent maxima creating over-segmentation
    coords = peak_local_max(distance, min_distance=5, labels=binary)
    if len(coords) == 0:
        return np.array([])

    # 4. Mark seeds and run watershed
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary)

    # 5. For each labeled region, compute equivalent radius
    props = regionprops(labels)
    if len(props) == 0:
        return np.array([])

    areas_px = np.array([p.area for p in props])
    # Equivalent radius in pixels = sqrt(area / pi)
    eq_radii_px = np.sqrt(areas_px / np.pi)
    # Convert to Blender units
    eq_radii_bu = eq_radii_px / PIXELS_PER_BLENDER_UNIT

    return eq_radii_bu


def radii_to_histogram(radii, n_bins, hist_range):
    """L1-normalized histogram of particle radii, with fixed binning."""
    if len(radii) == 0:
        return np.zeros(n_bins)
    hist, _ = np.histogram(radii, bins=n_bins, range=hist_range)
    if hist.sum() == 0:
        return np.zeros(n_bins)
    return hist.astype(float) / hist.sum()


# ----------------------------------------------------------------------
# Step 1 — Load datasets
# ----------------------------------------------------------------------
print("Step 1: Loading datasets...")
ds_train = PowderDataset('data', split='train', transform=None)
ds_test = PowderDataset('data', split='test', transform=None)
print(f"  Train: {len(ds_train)} images")
print(f"  Test:  {len(ds_test)} images")

# ----------------------------------------------------------------------
# Step 2 — Process training images
# ----------------------------------------------------------------------
print("\nStep 2: Processing training images...")
t0 = time.time()

train_histograms = []
train_labels = []
total_particles_train = 0

for i in range(len(ds_train)):
    img, label = ds_train[i]
    img_np = np.array(img)
    radii = segment_and_measure(img_np)
    total_particles_train += len(radii)
    hist = radii_to_histogram(radii, N_HIST_BINS, HIST_RANGE)
    train_histograms.append(hist)
    train_labels.append(label)
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(ds_train)} train images")

train_histograms = np.array(train_histograms)
train_labels = np.array(train_labels)
print(f"  Train histograms shape: {train_histograms.shape}")
print(f"  Avg particles segmented per image: {total_particles_train/len(ds_train):.0f}")
print(f"  Time: {time.time() - t0:.1f}s")

# ----------------------------------------------------------------------
# Step 3 — Process test images
# ----------------------------------------------------------------------
print("\nStep 3: Processing test images...")
t0 = time.time()

test_histograms = []
test_labels = []
total_particles_test = 0

for i in range(len(ds_test)):
    img, label = ds_test[i]
    img_np = np.array(img)
    radii = segment_and_measure(img_np)
    total_particles_test += len(radii)
    hist = radii_to_histogram(radii, N_HIST_BINS, HIST_RANGE)
    test_histograms.append(hist)
    test_labels.append(label)
    if (i + 1) % 200 == 0:
        print(f"  Processed {i+1}/{len(ds_test)} test images")

test_histograms = np.array(test_histograms)
test_labels = np.array(test_labels)
print(f"  Test histograms shape: {test_histograms.shape}")
print(f"  Avg particles segmented per image: {total_particles_test/len(ds_test):.0f}")
print(f"  Time: {time.time() - t0:.1f}s")

# ----------------------------------------------------------------------
# Step 4 — Train and evaluate chi-squared SVM
# ----------------------------------------------------------------------
print("\nStep 4: Training chi-squared SVM classifier...")
t0 = time.time()

K_train = chi2_kernel(train_histograms, train_histograms)
K_test = chi2_kernel(test_histograms, train_histograms)
print(f"  Kernel matrices computed in {time.time() - t0:.1f}s")

svm = SVC(kernel='precomputed', C=1.0, random_state=SEED)
svm.fit(K_train, train_labels)
predictions = svm.predict(K_test)

accuracy = accuracy_score(test_labels, predictions)
cm = confusion_matrix(test_labels, predictions)

print(f"  SVM trained and evaluated. Time: {time.time() - t0:.1f}s")
print()

# ----------------------------------------------------------------------
# Step 5 — Save results
# ----------------------------------------------------------------------
print("=" * 50)
print("=== RESULT ===")
print(f"Watershed + chi-squared SVM test accuracy: {accuracy:.4f}")
print(f"2016 paper reported:                       0.9020")
print(f"Difference:                                {accuracy - 0.902:+.4f}")
print("=" * 50)

results = {
    "method": "Watershed segmentation + chi-squared SVM (this implementation)",
    "config": {
        "thresholding": "Otsu",
        "segmentation": "Watershed (skimage)",
        "min_peak_distance": 5,
        "n_hist_bins": N_HIST_BINS,
        "hist_range_blender_units": list(HIST_RANGE),
        "pixels_per_blender_unit": PIXELS_PER_BLENDER_UNIT,
        "kernel": "chi-squared",
        "C": 1.0,
        "seed": SEED,
    },
    "test_accuracy": float(accuracy),
    "reference_2016_accuracy": 0.902,
    "confusion_matrix": cm.tolist(),
    "n_train": len(train_labels),
    "n_test": len(test_labels),
    "avg_particles_per_train_image": total_particles_train / len(ds_train),
    "avg_particles_per_test_image": total_particles_test / len(ds_test),
}

out_path = OUTPUT_DIR / "watershed_results.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {out_path}")

np.savez(
    OUTPUT_DIR / "watershed_predictions.npz",
    confusion_matrix=cm,
    y_true=test_labels,
    y_pred=predictions,
)
print(f"Confusion matrix saved to: {OUTPUT_DIR / 'watershed_predictions.npz'}")