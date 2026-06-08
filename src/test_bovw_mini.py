"""
Mini BOVW pipeline — proof-of-concept before running the full version.

Uses 10 training images per class (80 total), vocabulary size 20.
Should run in ~30 seconds and give accuracy > 50%.

If this works, the full pipeline (Step 7.5) will also work.
"""
import sys
sys.path.insert(0, 'src')

import time
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from dataset import PowderDataset

# ----------------------------------------------------------------------
# Configuration (mini version)
# ----------------------------------------------------------------------
N_PER_CLASS = 10      # Use 10 train images per class (80 total)
N_VISUAL_WORDS = 20   # Tiny vocabulary
N_CLASSES = 8

print(f"=== MINI BOVW PIPELINE ===")
print(f"Train images per class: {N_PER_CLASS} (total: {N_PER_CLASS * N_CLASSES})")
print(f"Vocabulary size: {N_VISUAL_WORDS}")
print()

# ----------------------------------------------------------------------
# Step 1 — Load datasets and pick subset
# ----------------------------------------------------------------------
print("Step 1: Loading datasets...")
ds_train = PowderDataset('data', split='train', transform=None)
ds_test = PowderDataset('data', split='test', transform=None)

# Pick a balanced subset of training images
train_indices = []
counts = [0] * N_CLASSES
for idx in range(len(ds_train)):
    _, lbl = ds_train[idx]
    if counts[lbl] < N_PER_CLASS:
        train_indices.append(idx)
        counts[lbl] += 1
    if all(c >= N_PER_CLASS for c in counts):
        break

print(f"  Selected {len(train_indices)} training images")

# For testing, use all test images (1024)
test_indices = list(range(len(ds_test)))
print(f"  Using all {len(test_indices)} test images")

# ----------------------------------------------------------------------
# Step 2 — Extract SIFT descriptors from training images
# ----------------------------------------------------------------------
print("\nStep 2: Extracting SIFT descriptors from training images...")
t0 = time.time()

sift = cv2.SIFT_create()
all_train_descriptors = []   # For building the vocabulary
train_image_descriptors = [] # Per-image, for later histogram construction
train_labels = []

for i, idx in enumerate(train_indices):
    img, label = ds_train[idx]
    img_np = np.array(img)
    _, descriptors = sift.detectAndCompute(img_np, None)
    if descriptors is not None:
        all_train_descriptors.append(descriptors)
        train_image_descriptors.append(descriptors)
        train_labels.append(label)
    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1}/{len(train_indices)} train images")

all_train_descriptors = np.vstack(all_train_descriptors)
print(f"  Total descriptors: {all_train_descriptors.shape}")
print(f"  Time: {time.time() - t0:.1f}s")

# ----------------------------------------------------------------------
# Step 3 — K-means to build vocabulary
# ----------------------------------------------------------------------
print(f"\nStep 3: Clustering descriptors into {N_VISUAL_WORDS} visual words...")
t0 = time.time()

kmeans = KMeans(n_clusters=N_VISUAL_WORDS, random_state=42, n_init=10)
kmeans.fit(all_train_descriptors)
print(f"  Vocabulary built. Time: {time.time() - t0:.1f}s")

# ----------------------------------------------------------------------
# Step 4 — Build histograms for training images
# ----------------------------------------------------------------------
print("\nStep 4: Building histograms for training images...")

def descriptors_to_histogram(descriptors, kmeans, n_words):
    """Map descriptors to nearest visual word, count occurrences, normalize."""
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(n_words)
    word_assignments = kmeans.predict(descriptors)
    hist, _ = np.histogram(word_assignments, bins=range(n_words + 1))
    return hist.astype(float) / hist.sum()  # L1-normalize

train_histograms = np.array([
    descriptors_to_histogram(d, kmeans, N_VISUAL_WORDS)
    for d in train_image_descriptors
])
print(f"  Train histograms shape: {train_histograms.shape}")

# ----------------------------------------------------------------------
# Step 5 — Extract SIFT and build histograms for test images
# ----------------------------------------------------------------------
print("\nStep 5: Processing test images...")
t0 = time.time()

test_histograms = []
test_labels = []
for i, idx in enumerate(test_indices):
    img, label = ds_test[idx]
    img_np = np.array(img)
    _, descriptors = sift.detectAndCompute(img_np, None)
    hist = descriptors_to_histogram(descriptors, kmeans, N_VISUAL_WORDS)
    test_histograms.append(hist)
    test_labels.append(label)
    if (i + 1) % 200 == 0:
        print(f"  Processed {i+1}/{len(test_indices)} test images")

test_histograms = np.array(test_histograms)
test_labels = np.array(test_labels)
print(f"  Test histograms shape: {test_histograms.shape}")
print(f"  Time: {time.time() - t0:.1f}s")

# ----------------------------------------------------------------------
# Step 6 — Train SVM and evaluate
# ----------------------------------------------------------------------
print("\nStep 6: Training SVM classifier...")
t0 = time.time()

svm = SVC(kernel='rbf', C=1.0, random_state=42)
svm.fit(train_histograms, train_labels)

predictions = svm.predict(test_histograms)
accuracy = accuracy_score(test_labels, predictions)

print(f"  SVM trained. Time: {time.time() - t0:.1f}s")
print()
print(f"=== RESULT ===")
print(f"Mini BOVW test accuracy: {accuracy:.4f}")
print(f"(Random guessing would be: {1.0/N_CLASSES:.4f})")
print(f"Trivial baseline got: 0.8040")
print(f"2016 BOVW (full version) got: 0.890")
print()
print("Mini version uses 80 train images and 20 visual words.")
print("Full version (Step 7.5) will use 1024 train images and 100 visual words.")