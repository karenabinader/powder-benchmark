"""
Analyze experimental results from Phase 1 and Phase 2.

Reads results.json files from each run folder and produces summary figures
for the thesis.

Usage:
    cd src
    python analyze_results.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ----------------------------------------------------------------------
# Where to read from and write to
# ----------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent / "results" / "runs"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Mapping from folder name → display label and metadata
# ----------------------------------------------------------------------
RUN_REGISTRY = {
    "resnet18_seed42_size224": {
        "label": "ResNet-18 @ 224",
        "family": "CNN (classic)",
        "params_M": 11.2,
        "resolution": 224,
    },
    "resnet18_seed42": {
        "label": "ResNet-18 @ 384",
        "family": "CNN (classic)",
        "params_M": 11.2,
        "resolution": 384,
    },
    "resnet50_seed42_size384": {
        "label": "ResNet-50 @ 384",
        "family": "CNN (classic)",
        "params_M": 25.5,
        "resolution": 384,
    },
    "efficientnet_b0_seed42_size384": {
        "label": "EfficientNet-B0 @ 384",
        "family": "CNN (efficient)",
        "params_M": 4.0,
        "resolution": 384,
    },
    "convnext_tiny_seed42_size384": {
        "label": "ConvNeXt-Tiny @ 384",
        "family": "CNN (modern)",
        "params_M": 27.8,
        "resolution": 384,
    },
    "vit_small_patch16_384_seed42_size384": {
        "label": "ViT-Small @ 384",
        "family": "Transformer",
        "params_M": 22.0,
        "resolution": 384,
    },
    "swin_tiny_patch4_window7_224_seed42_size224": {
        "label": "Swin-Tiny @ 224",
        "family": "Transformer",
        "params_M": 27.5,
        "resolution": 224,
    },
}

# Reference points from external sources (not our experiments)
EXTERNAL_REFERENCES = [
    {"label": "Trivial features\n+ logistic reg.", "test_acc": 0.804, "family": "Baseline"},
    {"label": "BOVW + SVM\n(2016)", "test_acc": 0.890, "family": "Reference"},
    {"label": "Watershed + SVM\n(2016)", "test_acc": 0.902, "family": "Reference"},
]


# ----------------------------------------------------------------------
# Load all runs into a single DataFrame
# ----------------------------------------------------------------------
def load_all_results() -> pd.DataFrame:
    rows = []
    for folder_name, meta in RUN_REGISTRY.items():
        run_dir = RESULTS_DIR / folder_name
        results_json = run_dir / "results.json"
        if not results_json.exists():
            print(f"⚠ Missing: {results_json}")
            continue

        with open(results_json) as f:
            data = json.load(f)

        rows.append({
            "folder": folder_name,
            "label": meta["label"],
            "family": meta["family"],
            "params_M": meta["params_M"],
            "resolution": meta["resolution"],
            "best_test_acc": data["best_test_acc"],
            "final_test_acc": data["final_test_acc"],
            "train_time_min": data["total_time_sec"] / 60,
            "history": data["history"],
        })

    df = pd.DataFrame(rows).sort_values("best_test_acc", ascending=False).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# Figure 1 — Main comparison bar chart (polished)
# ----------------------------------------------------------------------
def figure_main_comparison(df: pd.DataFrame):
    # Taller figure to fit the BOVW label above the bars
    fig, ax = plt.subplots(figsize=(13, 8))

    # Prepare combined data: our runs + external references
    bars = []
    for _, row in df.iterrows():
        bars.append({
            "label": row["label"],
            "acc": row["best_test_acc"],
            "family": row["family"],
        })
    for ref in EXTERNAL_REFERENCES:
        bars.append({
            "label": ref["label"],
            "acc": ref["test_acc"],
            "family": ref["family"],
        })

    # Sort by accuracy
    bars = sorted(bars, key=lambda b: b["acc"])

    labels = [b["label"] for b in bars]
    accs = [b["acc"] for b in bars]
    families = [b["family"] for b in bars]

    # Color per family
    family_colors = {
        "CNN (classic)": "#1f77b4",
        "CNN (modern)": "#2ca02c",
        "CNN (efficient)": "#17becf",
        "Transformer": "#ff7f0e",
        "Reference": "#9467bd",
        "Baseline": "#7f7f7f",
    }
    colors = [family_colors[f] for f in families]

    # Draw bars
    y_positions = range(len(labels))
    ax.barh(y_positions, accs, color=colors, edgecolor="black", linewidth=0.6)

    # Value labels on bars (larger font)
    for i, acc in enumerate(accs):
        ax.text(acc + 0.004, i, f"{acc:.3f}", va="center", fontsize=11, fontweight="bold")

    # Y-axis: labels with larger font
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11)

    # X-axis
    ax.set_xlim(0.75, 1.0)
    ax.set_xlabel("Test accuracy", fontsize=12, fontweight="bold")

    # Title
    ax.set_title(
        "Architecture comparison on synthetic powder classification\n"
        "(single seed = 42, training set = 1024 images, test set = 1024 images)",
        fontsize=13, pad=15,
    )

    # 2016 BOVW reference line — clean, with annotation BELOW the top edge
    bovw_acc = 0.890
    ax.axvline(bovw_acc, color="purple", linestyle="--", linewidth=1.5, alpha=0.6, zorder=0)
    # Annotation positioned to the side, not vertical
    ax.annotate(
        "2016 BOVW\nbaseline (0.890)",
        xy=(bovw_acc, len(labels) - 0.5),
        xytext=(bovw_acc - 0.035, len(labels) - 0.2),
        fontsize=9, color="purple",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="purple", alpha=0.5, lw=1),
    )

    # Grid for readability
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    # Custom legend — positioned outside the plot area to not cover bars
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, edgecolor="black", label=family)
        for family, color in family_colors.items()
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=10,
        framealpha=0.95,
        bbox_to_anchor=(0.98, 0.02),
    )

    # Remove spines on the top and right for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig1_main_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")  # higher DPI for thesis
    print(f"✓ Saved: {out_path}")
    plt.show()

# ----------------------------------------------------------------------
# Figure 2 — Training curves (polished)
# ----------------------------------------------------------------------
def figure_training_curves(df: pd.DataFrame):
    """Two side-by-side panels: test loss and test accuracy over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Custom color scheme that emphasizes the 224 vs 384 ResNet-18 contrast
    color_map = {
        "ResNet-18 @ 384":      "#1f77b4",  # blue - hero
        "ResNet-18 @ 224":      "#aec7e8",  # light blue - hero's foil
        "ResNet-50 @ 384":      "#ff7f0e",  # orange
        "EfficientNet-B0 @ 384": "#17becf", # cyan
        "ConvNeXt-Tiny @ 384":  "#2ca02c",  # green
        "ViT-Small @ 384":      "#d62728",  # red
        "Swin-Tiny @ 224":      "#9467bd",  # purple
    }

    # Linestyles: dashed for 224-resolution runs, solid for 384
    def linestyle_for(label):
        return "--" if "@ 224" in label else "-"

    # Linewidth: extra emphasis on the two ResNet-18 runs (our hero comparison)
    def linewidth_for(label):
        return 2.5 if "ResNet-18" in label else 1.5

    # ===== LEFT PANEL: Test loss =====
    ax_loss = axes[0]
    for _, row in df.iterrows():
        epochs = [h["epoch"] for h in row["history"]]
        test_loss = [h["test_loss"] for h in row["history"]]
        ax_loss.plot(
            epochs, test_loss,
            label=row["label"],
            color=color_map.get(row["label"], "#888888"),
            linewidth=linewidth_for(row["label"]),
            linestyle=linestyle_for(row["label"]),
            marker="o", markersize=3, alpha=0.9,
        )
    ax_loss.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax_loss.set_ylabel("Test loss", fontsize=12, fontweight="bold")
    ax_loss.set_title("Test loss over training", fontsize=12)
    ax_loss.set_xlim(0.5, 20.5)
    ax_loss.set_ylim(0.1, 2.1)
    ax_loss.grid(True, linestyle=":", alpha=0.4)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)

    # ===== RIGHT PANEL: Test accuracy (zoomed) =====
    ax_acc = axes[1]
    for _, row in df.iterrows():
        epochs = [h["epoch"] for h in row["history"]]
        test_acc = [h["test_acc"] for h in row["history"]]
        ax_acc.plot(
            epochs, test_acc,
            label=row["label"],
            color=color_map.get(row["label"], "#888888"),
            linewidth=linewidth_for(row["label"]),
            linestyle=linestyle_for(row["label"]),
            marker="o", markersize=3, alpha=0.9,
        )
    ax_acc.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax_acc.set_ylabel("Test accuracy", fontsize=12, fontweight="bold")
    ax_acc.set_title("Test accuracy over training (y-axis zoomed)", fontsize=12)
    ax_acc.set_xlim(0.5, 20.5)
    ax_acc.set_ylim(0.30, 1.00)  # Zoomed to interesting range

    # 2016 BOVW reference line
    ax_acc.axhline(0.890, color="purple", linestyle=":", linewidth=2, alpha=0.7, zorder=0)
    ax_acc.text(
        20.5, 0.890, " 2016 BOVW (0.89)",
        fontsize=10, color="purple", va="center", ha="left", fontweight="bold",
    )

    ax_acc.grid(True, linestyle=":", alpha=0.4)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)

    # Single legend below — explicitly labeled with style key
    handles, labels = ax_acc.get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        title="(solid = 384, dashed = 224; thick ResNet-18 = hero comparison)",
        title_fontsize=9,
    )

    fig.suptitle(
        "Training dynamics across architectures (20 epochs, seed=42)",
        fontsize=14, fontweight="bold", y=0.99,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_path = OUTPUT_DIR / "fig2_training_curves.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.show()

# ----------------------------------------------------------------------
# Figure 3 — Confusion matrices grid
# ----------------------------------------------------------------------
def figure_confusion_matrices(df: pd.DataFrame):
    """Grid of confusion matrices, one per model."""
    n_models = len(df)
    # Layout: 2 rows × 4 columns (7 models + 1 empty slot)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_flat = axes.flatten()

    class_names = ["a", "b", "c", "d", "e", "f", "g", "h"]

    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes_flat[idx]

        # Load predictions.npz for this run
        npz_path = RESULTS_DIR / row["folder"] / "predictions.npz"
        data = np.load(npz_path)
        cm = data["confusion_matrix"]

        # Normalize rows (each row sums to 1) so heatmap is comparable
        # across rows of different sizes
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Plot heatmap
        im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1, aspect="equal")

        # Annotate each cell with the count
        for i in range(8):
            for j in range(8):
                count = int(cm[i, j])
                if count == 0:
                    continue  # skip zeros for cleaner look
                # Text color: white on dark cells, black on light cells
                color = "white" if cm_normalized[i, j] > 0.5 else "black"
                ax.text(j, i, str(count), ha="center", va="center",
                        color=color, fontsize=8)

        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted class", fontsize=9)
        ax.set_ylabel("True class", fontsize=9)
        ax.set_title(
            f"{row['label']}\nbest acc = {row['best_test_acc']:.3f}",
            fontsize=10,
        )

        # Highlight the "hard pair" cells the 2016 paper predicted
        # e=4, g=6  →  cells (4,6) and (6,4)
        # f=5, h=7  →  cells (5,7) and (7,5)
        from matplotlib.patches import Rectangle
        hard_pairs = [(4, 6), (6, 4), (5, 7), (7, 5)]
        for i, j in hard_pairs:
            ax.add_patch(Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=1.5,
            ))

    # Hide the unused 8th subplot
    axes_flat[-1].axis("off")
    # Use the empty space for a colorbar
    cbar_ax = fig.add_axes([0.78, 0.10, 0.18, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Row-normalized count (per-class recall)", fontsize=9)

    # Legend explaining red boxes
    axes_flat[-1].text(
        0.5, 0.6,
        "Red boxes mark the four cells\ncorresponding to the two\nhardest pairs:\n\n"
        "(e ↔ g) and (f ↔ h)\n\n"
        "These lognormal-vs-Weibull-fit\npairs are statistically the\n"
        "closest distributions in the\nDeCost & Holm 2016 dataset.",
        ha="center", va="center", fontsize=9,
        transform=axes_flat[-1].transAxes,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="red", linewidth=1),
    )

    fig.suptitle(
        "Confusion matrices across architectures (seed=42)\n"
        "Rows = true class, columns = predicted class. Values are test image counts.",
        fontsize=13, fontweight="bold", y=1.00,
    )

    plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.06,
                        wspace=0.3, hspace=0.4)
    out_path = OUTPUT_DIR / "fig3_confusion_matrices.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.show()

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    df = load_all_results()
    print(f"\nLoaded {len(df)} runs:")
    print(df[["label", "best_test_acc", "params_M", "resolution", "train_time_min"]].to_string(index=False))

    print("\nGenerating figures...")
    figure_main_comparison(df)
    figure_training_curves(df)
    figure_confusion_matrices(df)


if __name__ == "__main__":
    main()