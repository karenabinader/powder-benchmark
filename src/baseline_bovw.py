"""
Full BOVW + SVM baseline implementation.

Matches the methodology of DeCost & Holm (2016) as closely as possible:
- BOTH Difference-of-Gaussians (DoG) and Harris-Laplace (HL) keypoint detectors
- SIFT descriptors (128-D) computed at all keypoints from both detectors
- k-means clustering into 100 visual words (descriptors subsampled to 100k)
- L1-normalized frequency histograms per image
- Chi-squared kernel SVM classifier

Saves results to results/baselines/bovw_results.json
"""
import sys
sys.path.insert(0, 'src')

import json
import time
from pathlib import Path

import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import accuracy_score, confusion_matrix
from dataset import PowderDataset

# ----------------------------------------------------------------------
# Configuration — matches 2016 paper
# ----------------------------------------------------------------------
N_VISUAL_WORDS = 100
N_DESCRIPTORS_FOR_KMEANS = 100_000
SEED = 42

OUTPUT_DIR = Path('results/baselines')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== FULL BOVW + chi-squared SVM PIPELINE ===")
print(f"Keypoint detectors: DoG (via SIFT) + Harris-Laplace")
print(f"Vocabulary size: {N_VISUAL_WORDS}")
print(f"Descriptors subsampled for k-means: {N_DESCRIPTORS_FOR_KMEANS}")
print(f"Random seed: {SEED}")
print()

np.random.seed(SEED)

# Create both detectors
sift = cv2.SIFT_create()  # DoG keypoints + SIFT descriptors
hl_detector = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()


def extract_descriptors(img_np):
    """
    Detect keypoints with both DoG and Harris-Laplace, compute SIFT descriptors
    at all of them, and concatenate. Returns shape (N, 128).
    """
    # DoG keypoints + SIFT descriptors (in one call)
    dog_kp, dog_desc = sift.detectAndCompute(img_np, None)
    if dog_desc is None:
        dog_desc = np.zeros((0, 128), dtype=np.float32)

    # Harris-Laplace keypoints, then SIFT descriptors at those points
    hl_kp = hl_detector.detect(img_np)
    if len(hl_kp) > 0:
        _, hl_desc = sift.compute(img_np, hl_kp)
        if hl_desc is None:
            hl_desc = np.zeros((0, 128), dtype=np.float32)
    else:
        hl_desc = np.zeros((0, 128), dtype=np.float32)

    # Concatenate descriptors from both detectors
    if dog_desc.shape[0] == 0 and hl_desc.shape[0] == 0:
        return np.zeros((0, 128), dtype=np.float32)
    return np.vstack([dog_desc, hl_desc])


# ----------------------------------------------------------------------
# Step 1 — Load datasets
# ----------------------------------------------------------------------
print("Step 1: Loading datasets...")
ds_train = PowderDataset('data', split='train', transform=None)
ds_test = PowderDataset('data', split='test', transform=None)
print(f"  Train: {len(ds_train)} images")
print(f"  Test:  {len(ds_test)} images")

# ----------------------------------------------------------------------
# Step 2 — Extract SIFT descriptors from ALL training images
# ----------------------------------------------------------------------
print("\nStep 2: Extracting DoG + Harris-Laplace SIFT from training images...")
t0 = time.time()

all_train_descriptors_list = []
train_image_descriptors = []
train_labels = []

for i in range(len(ds_train)):
    img, label = ds_train[i]
    img_np = np.array(img)
    descriptors = extract_descriptors(img_np)
    all_train_descriptors_list.append(descriptors)
    train_image_descriptors.append(descriptors)
    train_labels.append(label)
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(ds_train)} train images")

all_train_descriptors = np.vstack(all_train_descriptors_list)
train_labels = np.array(train_labels)
print(f"  Total descriptors collected: {all_train_descriptors.shape}")
print(f"  Avg descriptors per image: {all_train_descriptors.shape[0] / len(ds_train):.0f}")
print(f"  Time: {time.time() - t0:.1f}s")

# ----------------------------------------------------------------------
# Step 3 — Subsample descriptors and run k-means
# ----------------------------------------------------------------------
print(f"\nStep 3: K-means clustering into {N_VISUAL_WORDS} visual words...")
t0 = time.time()

n_total = all_train_descriptors.shape[0]
if n_total > N_DESCRIPTORS_FOR_KMEANS:
    sample_indices = np.random.choice(n_total, size=N_DESCRIPTORS_FOR_KMEANS, replace=False)
    descriptors_for_kmeans = all_train_descriptors[sample_indices]
    print(f"  Subsampled {N_DESCRIPTORS_FOR_KMEANS} of {n_total} descriptors")
else:
    descriptors_for_kmeans = all_train_descriptors
    print(f"  Using all {n_total} descriptors (no subsampling needed)")

kmeans = KMeans(n_clusters=N_VISUAL_WORDS, random_state=SEED, n_init=10, verbose=0)
kmeans.fit(descriptors_for_kmeans)
print(f"  Vocabulary built. Time: {time.time() - t0:.1f}s")

# Free memory
del all_train_descriptors_list
del descriptors_for_kmeans

# ----------------------------------------------------------------------
# Step 4 — Build histograms for training images
# ----------------------------------------------------------------------
print("\nStep 4: Building histograms for training images...")
t0 = time.time()

def descriptors_to_histogram(descriptors, kmeans_model, n_words):
    """Map descriptors to nearest visual word, count occurrences, L1-normalize."""
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(n_words)
    word_assignments = kmeans_model.predict(descriptors)
    hist, _ = np.histogram(word_assignments, bins=range(n_words + 1))
    if hist.sum() == 0:
        return np.zeros(n_words)
    return hist.astype(float) / hist.sum()

train_histograms = np.array([
    descriptors_to_histogram(d, kmeans, N_VISUAL_WORDS)
    for d in train_image_descriptors
])
del train_image_descriptors
print(f"  Train histograms shape: {train_histograms.shape}")
print(f"  Time: {time.time() - t0:.1f}s")

# ----------------------------------------------------------------------
# Step 5 — Extract SIFT and build histograms for test images
# ----------------------------------------------------------------------
print("\nStep 5: Processing test images...")
t0 = time.time()

test_histograms = []
test_labels = []
for i in range(len(ds_test)):
    img, label = ds_test[i]
    img_np = np.array(img)
    descriptors = extract_descriptors(img_np)
    hist = descriptors_to_histogram(descriptors, kmeans, N_VISUAL_WORDS)
    test_histograms.append(hist)
    test_labels.append(label)
    if (i + 1) % 200 == 0:
        print(f"  Processed {i+1}/{len(ds_test)} test images")

test_histograms = np.array(test_histograms)
test_labels = np.array(test_labels)
print(f"  Test histograms shape: {test_histograms.shape}")
print(f"  Time: {time.time() - t0:.1f}s")

# ----------------------------------------------------------------------
# Step 6 — Chi-squared kernel SVM
# ----------------------------------------------------------------------
print("\nStep 6: Training chi-squared SVM classifier...")
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
# Step 7 — Save results
# ----------------------------------------------------------------------
print("=" * 50)
print("=== RESULT ===")
print(f"BOVW + chi-squared SVM test accuracy: {accuracy:.4f}")
print(f"2016 paper reported:                  0.8900")
print(f"Difference:                           {accuracy - 0.89:+.4f}")
print("=" * 50)

results = {
    "method": "BOVW + chi-squared SVM (this implementation, DoG + Harris-Laplace)",
    "config": {
        "n_visual_words": N_VISUAL_WORDS,
        "n_descriptors_for_kmeans": N_DESCRIPTORS_FOR_KMEANS,
        "kernel": "chi-squared",
        "C": 1.0,
        "seed": SEED,
        "keypoint_detectors": "Difference-of-Gaussians (via OpenCV SIFT) + Harris-Laplace (via xfeatures2d)",
        "note": "Matches the keypoint detection of DeCost and Holm (2016). SIFT descriptor library differs (OpenCV vs VLFeat).",
    },
    "test_accuracy": float(accuracy),
    "reference_2016_accuracy": 0.890,
    "confusion_matrix": cm.tolist(),
    "n_train": len(train_labels),
    "n_test": len(test_labels),
}

out_path = OUTPUT_DIR / "bovw_results.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {out_path}")

np.savez(
    OUTPUT_DIR / "bovw_predictions.npz",
    confusion_matrix=cm,
    y_true=test_labels,
    y_pred=predictions,
)
print(f"Confusion matrix saved to: {OUTPUT_DIR / 'bovw_predictions.npz'}")