"""
Trivial baseline: hand-crafted pixel-statistic features + logistic regression.

This is NOT meant to be competitive — it's the floor. If a CNN can't beat this,
the CNN is broken.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from dataset import PowderDataset, get_class_names


def extract_features(img: Image.Image) -> np.ndarray:
    """Compute a small set of hand-crafted features from a grayscale image."""
    arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize to 0-1

    # Basic statistics
    mean = arr.mean()
    std = arr.std()

    # Percentiles capture the distribution shape
    p10, p25, p50, p75, p90, p95, p99 = np.percentile(
        arr, [10, 25, 50, 75, 90, 95, 99]
    )

    # Simple "particle area" via thresholding at different levels
    # Larger particles tend to have more bright pixels
    frac_above_0_3 = (arr > 0.3).mean()
    frac_above_0_5 = (arr > 0.5).mean()
    frac_above_0_7 = (arr > 0.7).mean()

    # Edge density via simple gradient magnitude
    gy, gx = np.gradient(arr)
    edge_mag = np.sqrt(gx**2 + gy**2)
    edge_mean = edge_mag.mean()
    edge_std = edge_mag.std()

    return np.array([
        mean, std,
        p10, p25, p50, p75, p90, p95, p99,
        frac_above_0_3, frac_above_0_5, frac_above_0_7,
        edge_mean, edge_std,
    ], dtype=np.float32)


def build_feature_matrix(dataset: PowderDataset, desc: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels for every sample in the dataset."""
    X = []
    y = []
    for i in tqdm(range(len(dataset)), desc=desc):
        img, label = dataset[i]
        X.append(extract_features(img))
        y.append(label)
    return np.stack(X), np.array(y)


def main():
    data_root = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Load both splits
    train_ds = PowderDataset(data_root, split="train")
    test_ds = PowderDataset(data_root, split="test")

    # Extract features
    print("Extracting training features...")
    X_train, y_train = build_feature_matrix(train_ds, desc="train")
    print("Extracting test features...")
    X_test, y_test = build_feature_matrix(test_ds, desc="test")

    print(f"\nFeature matrix shapes: train={X_train.shape}, test={X_test.shape}")

    # Standardize features (mean 0, std 1) — important for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    print("\nTraining logistic regression...")
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        random_state=42,
        multi_class="multinomial",
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_train = clf.predict(X_train_scaled)
    y_pred_test = clf.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\n{'='*50}")
    print(f"TRIVIAL BASELINE — hand-crafted features + logistic regression")
    print(f"{'='*50}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy:     {test_acc:.4f}")
    print(f"Chance accuracy:   {1/8:.4f} (1/8 for 8 classes)")
    print(f"DeCost & Holm BOVW: 0.889")
    print(f"{'='*50}\n")

    # Detailed per-class report
    class_names = get_class_names()
    print("Per-class report (test set):")
    print(classification_report(y_test, y_pred_test, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print("Confusion matrix (test set):")
    print("Rows = true class, columns = predicted class")
    print("     " + "  ".join([f"{n:>4}" for n in class_names]))
    for i, row in enumerate(cm):
        print(f"  {class_names[i]} " + "  ".join([f"{v:>4d}" for v in row]))

    # Save results for later comparison
    np.savez(
        results_dir / "baseline_simple_results.npz",
        train_acc=train_acc,
        test_acc=test_acc,
        confusion_matrix=cm,
        y_test=y_test,
        y_pred_test=y_pred_test,
    )
    print(f"\nResults saved to {results_dir / 'baseline_simple_results.npz'}")


if __name__ == "__main__":
    main()