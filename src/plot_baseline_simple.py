"""Plot the confusion matrix from the trivial baseline for the thesis."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_class_names


def main():
    results_dir = Path(__file__).parent.parent / "results"

    # Load saved results
    data = np.load(results_dir / "baseline_simple_results.npz")
    cm = data["confusion_matrix"]
    test_acc = float(data["test_acc"])

    class_names = get_class_names()

    # Normalize rows so each row sums to 1 (per-class recall view)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Two-panel figure: raw counts and normalized
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], cbar=False,
    )
    axes[0].set_title("Confusion matrix (raw counts)", fontsize=13)
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], cbar=True, vmin=0, vmax=1,
    )
    axes[1].set_title("Confusion matrix (row-normalized)", fontsize=13)
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("True class")

    fig.suptitle(
        f"Trivial baseline (hand-crafted features + logistic regression) — Test accuracy: {test_acc:.1%}",
        fontsize=14,
    )

    plt.tight_layout()
    out_path = results_dir / "03_baseline_simple_confusion.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()