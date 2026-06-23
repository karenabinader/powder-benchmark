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
    {"label": "BOVW + SVM\n(this work)", "test_acc": 0.8613, "family": "Reproduced"},
    {"label": "Watershed + SVM\n(this work)", "test_acc": 0.9492, "family": "Reproduced"},
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
        "Reproduced": "#A569BD",
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
# Figure 4 — Per-class accuracy heatmap
# ----------------------------------------------------------------------
def figure_per_class_heatmap(df: pd.DataFrame):
    """Heatmap of per-class recall (rows = models, columns = classes a-h)."""
    class_names = ["a", "b", "c", "d", "e", "f", "g", "h"]
    hard_classes = ["e", "f", "g", "h"]  # involved in the hardest pairs

    # Sort models by overall accuracy descending (best at top)
    df_sorted = df.sort_values("best_test_acc", ascending=False).reset_index(drop=True)

    # Build the recall matrix: rows = models, columns = classes
    recall_matrix = np.zeros((len(df_sorted), 8))
    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        npz_path = RESULTS_DIR / row["folder"] / "predictions.npz"
        data = np.load(npz_path)
        cm = data["confusion_matrix"]
        # Per-class recall = diagonal / row sum
        recall_matrix[idx] = cm.diagonal() / cm.sum(axis=1)

    # Add a final "Overall" column = each model's mean accuracy
    overall_col = df_sorted["best_test_acc"].values.reshape(-1, 1)
    full_matrix = np.hstack([recall_matrix, overall_col])
    full_class_names = class_names + ["Overall"]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))

    # Adaptive color scale — most variation is in 0.70–1.00 range
    im = ax.imshow(full_matrix, cmap="Blues", vmin=0.70, vmax=1.00, aspect="auto")

    # Thin vertical separator before the "Overall" column
    ax.axvline(x=7.5, color="black", linewidth=1.2, zorder=3)

    # Annotate every cell with its value
    for i in range(len(df_sorted)):
        for j in range(len(full_class_names)):
            value = full_matrix[i, j]
            # Text color: white on dark cells, black on light cells
            color = "white" if value > 0.88 else "black"
            ax.text(j, i, f"{value:.2f}",
                    ha="center", va="center",
                    color=color, fontsize=10, zorder=4)

    # Ticks and labels
    ax.set_xticks(range(len(full_class_names)))
    ax.set_xticklabels(full_class_names, fontsize=11)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["label"].values, fontsize=10)
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Model (sorted by overall accuracy)", fontsize=11)

    # Hard-class footnote
    ax.text(
        0.5, -0.18,
        "Classes e, f, g, h were designed to be the hardest in the DeCost & Holm 2016 dataset (closest distribution pairs).",
        ha="center", va="top", fontsize=9, style="italic",
        transform=ax.transAxes,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Per-class recall", fontsize=10)

    # Title
    fig.suptitle(
        "Per-class accuracy heatmap (seed=42)\n"
        "Rows = model (sorted by overall accuracy). "
        "Columns = class. Values are per-class recall (diagonal / 128).",
        fontsize=12, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig4_per_class_heatmap.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.show()
    
    # ----------------------------------------------------------------------
# Figure 5 — Parameters vs Accuracy scatter
# ----------------------------------------------------------------------
def figure_params_vs_accuracy(df: pd.DataFrame):
    """Scatter plot of model parameters vs test accuracy, colored by resolution."""

    # Hard-coded parameter counts (millions), from timm model definitions
    # Keys must match the 'label' values used in the rest of the analysis
    params_lookup = {
        "ResNet-18 @ 384":     11.2,
        "ResNet-18 @ 224":     11.2,
        "ResNet-50 @ 384":     23.5,
        "ViT-Small @ 384":     21.7,
        "ConvNeXt-Tiny @ 384": 27.8,
        "Swin-Tiny @ 224":     27.5,
        "EfficientNet-B0 @ 384": 4.0,
    }

    # Build arrays for plotting
    labels = df["label"].values
    accuracies = df["best_test_acc"].values
    params = np.array([params_lookup[label] for label in labels])

    # Determine resolution per model (224 vs 384) from the label
    resolutions = np.array([224 if "@ 224" in label else 384 for label in labels])

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color coding: 224 = orange, 384 = blue
    colors_by_res = {224: "#E69F00", 384: "#0072B2"}
    point_colors = [colors_by_res[r] for r in resolutions]

    # Scatter
    ax.scatter(params, accuracies, c=point_colors, s=180, edgecolor="black",
               linewidth=1.2, zorder=3)

    # Label each point with the model name
    # Use small offsets to avoid label collisions
    label_offsets = {
        "ResNet-18 @ 384":       (0.6, 0.003),
        "ResNet-18 @ 224":       (0.6, -0.005),
        "ResNet-50 @ 384":       (0.6, 0.003),
        "ViT-Small @ 384":       (0.6, -0.012),
        "ConvNeXt-Tiny @ 384":   (-0.6, 0.005),
        "Swin-Tiny @ 224":       (0.6, 0.003),
        "EfficientNet-B0 @ 384": (0.6, 0.003),
    }
    for i, label in enumerate(labels):
        dx, dy = label_offsets.get(label, (0.6, 0.003))
        ha = "right" if dx < 0 else "left"
        ax.annotate(label,
                    xy=(params[i], accuracies[i]),
                    xytext=(params[i] + dx, accuracies[i] + dy),
                    fontsize=10, ha=ha, va="center", zorder=4)

    # Reference line: 2016 BOVW baseline
    ax.axhline(y=0.890, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    ax.text(28.5, 0.890, "2016 BOVW baseline (0.890)",
            fontsize=9, va="center", ha="right", color="gray", style="italic")

    # Reference line: 2016 watershed baseline
    ax.axhline(y=0.902, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, zorder=2)
    ax.text(28.5, 0.902, "2016 watershed baseline (0.902)",
            fontsize=9, va="center", ha="right", color="gray", style="italic")

    # Axis labels and limits
    ax.set_xlabel("Trainable parameters (millions)", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_xlim(0, 32)
    ax.set_ylim(0.85, 1.00)
    ax.grid(True, alpha=0.3, zorder=1)

    # Legend for the resolution color coding
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2",
               markeredgecolor="black", markersize=12, label="Input resolution: 384"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E69F00",
               markeredgecolor="black", markersize=12, label="Input resolution: 224"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.95)

    # Title and subtitle
    fig.suptitle(
        "Model size vs test accuracy (seed=42)\n"
        "Larger models do not yield higher accuracy on this dataset.",
        fontsize=13, fontweight="bold", y=0.98,
    )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig5_params_vs_accuracy.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.show()

# ======================================================================
# PHASE 5 — Multi-seed analysis
# Loads all 35 runs (7 architectures × 5 seeds), computes mean ± std,
# and produces multiseed versions of figures 1, 2, 4, 5.
# ======================================================================

# Extended registry covering all 35 folders
# Maps folder name → (canonical_label, params_M, resolution)
# canonical_label is what we group by; all 5 seeds of one model share it
MULTISEED_REGISTRY = {}

# Build by combining each base label with each seed's folder name pattern
_MODEL_DEFS = [
    # (canonical_label, params_M, resolution, base_folder_pattern)
    # base_folder_pattern uses {seed} as the placeholder
    ("ResNet-18 @ 224",      11.2, 224, "resnet18_seed{seed}_size224"),
    ("ResNet-50 @ 384",      25.5, 384, "resnet50_seed{seed}_size384"),
    ("EfficientNet-B0 @ 384", 4.0, 384, "efficientnet_b0_seed{seed}_size384"),
    ("ConvNeXt-Tiny @ 384",  27.8, 384, "convnext_tiny_seed{seed}_size384"),
    ("ViT-Small @ 384",      22.0, 384, "vit_small_patch16_384_seed{seed}_size384"),
    ("Swin-Tiny @ 224",      27.5, 224, "swin_tiny_patch4_window7_224_seed{seed}_size224"),
]

# ResNet-18 @ 384 is special — seed=42 uses legacy folder name without size suffix
_RESNET18_384_FOLDERS = {
    42: "resnet18_seed42",
    0:  "resnet18_seed0_size384",
    1:  "resnet18_seed1_size384",
    2:  "resnet18_seed2_size384",
    3:  "resnet18_seed3_size384",
}

for label, params, res, pattern in _MODEL_DEFS:
    for seed in [42, 0, 1, 2, 3]:
        folder = pattern.format(seed=seed)
        MULTISEED_REGISTRY[folder] = {
            "label": label, "params_M": params, "resolution": res, "seed": seed,
        }
# Add ResNet-18 @ 384 manually (legacy seed=42 folder)
for seed, folder in _RESNET18_384_FOLDERS.items():
    MULTISEED_REGISTRY[folder] = {
        "label": "ResNet-18 @ 384", "params_M": 11.2, "resolution": 384, "seed": seed,
    }


def load_all_seeds() -> pd.DataFrame:
    """Load all 35 runs (7 models × 5 seeds) into one DataFrame."""
    rows = []
    for folder_name, meta in MULTISEED_REGISTRY.items():
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
            "params_M": meta["params_M"],
            "resolution": meta["resolution"],
            "seed": meta["seed"],
            "best_test_acc": data["best_test_acc"],
            "history": data["history"],
        })
    df = pd.DataFrame(rows)
    return df


def aggregate_by_model(df_all: pd.DataFrame) -> pd.DataFrame:
    """Group by label, compute mean and std of best_test_acc across seeds."""
    agg = df_all.groupby("label").agg(
        mean_acc=("best_test_acc", "mean"),
        std_acc=("best_test_acc", "std"),
        n_seeds=("best_test_acc", "count"),
        params_M=("params_M", "first"),
        resolution=("resolution", "first"),
    ).reset_index()
    agg = agg.sort_values("mean_acc", ascending=False).reset_index(drop=True)
    return agg


def get_mean_per_class_recall(df_all: pd.DataFrame, label: str) -> np.ndarray:
    """For one model (across all its seeds), compute mean per-class recall."""
    subset = df_all[df_all["label"] == label]
    recalls = []
    for _, row in subset.iterrows():
        npz_path = RESULTS_DIR / row["folder"] / "predictions.npz"
        cm = np.load(npz_path)["confusion_matrix"]
        recalls.append(cm.diagonal() / cm.sum(axis=1))
    return np.array(recalls).mean(axis=0)

# ----------------------------------------------------------------------
# Figure 1 multiseed — Bar chart with error bars
# ----------------------------------------------------------------------
def figure_main_comparison_multiseed(df_all: pd.DataFrame):
    """Bar chart of mean test accuracy with std error bars."""
    agg = aggregate_by_model(df_all)

    fig, ax = plt.subplots(figsize=(13, 8))

    # Build the combined list: our models (with error bars) + external refs (no error)
    bars = []
    for _, row in agg.iterrows():
        # Determine family from label for consistent coloring
        label = row["label"]
        if "ResNet" in label or "EfficientNet" in label:
            family = "CNN (classic)" if "ResNet" in label else "CNN (efficient)"
        elif "ConvNeXt" in label:
            family = "CNN (modern)"
        else:
            family = "Transformer"
        bars.append({
            "label": label, "acc": row["mean_acc"], "std": row["std_acc"],
            "family": family,
        })
    for ref in EXTERNAL_REFERENCES:
        bars.append({
            "label": ref["label"], "acc": ref["test_acc"], "std": 0.0,
            "family": ref["family"],
        })

    bars = sorted(bars, key=lambda b: b["acc"])

    labels = [b["label"] for b in bars]
    accs = [b["acc"] for b in bars]
    stds = [b["std"] for b in bars]
    families = [b["family"] for b in bars]

    family_colors = {
        "CNN (classic)": "#1f77b4", "CNN (modern)": "#2ca02c",
        "CNN (efficient)": "#17becf", "Transformer": "#ff7f0e",
        "Reference": "#9467bd", "Reproduced": "#A569BD", "Baseline": "#7f7f7f",
    }
    colors = [family_colors[f] for f in families]

    y_positions = range(len(labels))
    # Horizontal error bars (xerr instead of yerr because barh is horizontal)
    ax.barh(y_positions, accs, color=colors, edgecolor="black", linewidth=0.6,
            xerr=stds, error_kw={"ecolor": "black", "capsize": 4, "elinewidth": 1.2})

    # Value labels — show mean ± std for our models, just value for refs
    for i, (acc, std) in enumerate(zip(accs, stds)):
        if std > 0:
            text = f"{acc:.3f} ± {std:.3f}"
        else:
            text = f"{acc:.3f}"
        ax.text(acc + std + 0.005, i, text, va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0.75, 1.02)
    ax.set_xlabel("Test accuracy (mean ± std across 5 seeds)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Architecture comparison on synthetic powder classification\n"
        "(mean ± std across seeds 42, 0, 1, 2, 3; training set = 1024 images, test set = 1024 images)",
        fontsize=13, pad=15,
    )

    # 2016 BOVW reference line
    ax.axvline(0.890, color="purple", linestyle="--", linewidth=1.5, alpha=0.6, zorder=0)
    ax.annotate("2016 BOVW\nbaseline (0.890)",
                xy=(0.890, len(labels) - 0.5),
                xytext=(0.890 - 0.035, len(labels) - 0.2),
                fontsize=9, color="purple", ha="center",
                arrowprops=dict(arrowstyle="->", color="purple", alpha=0.5, lw=1))

    ax.set_axisbelow(True)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="black", label=f)
                       for f, c in family_colors.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
              framealpha=0.95, bbox_to_anchor=(0.98, 0.02))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig1_main_comparison_multiseed.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.show()


# ----------------------------------------------------------------------
# Figure 2 multiseed — Training curves with 5 seeds per model
# ----------------------------------------------------------------------
def figure_training_curves_multiseed(df_all: pd.DataFrame):
    """Training curves with semi-transparent individual seeds + bold mean line."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    color_map = {
        "ResNet-18 @ 384":       "#1f77b4",
        "ResNet-18 @ 224":       "#aec7e8",
        "ResNet-50 @ 384":       "#ff7f0e",
        "EfficientNet-B0 @ 384": "#17becf",
        "ConvNeXt-Tiny @ 384":   "#2ca02c",
        "ViT-Small @ 384":       "#d62728",
        "Swin-Tiny @ 224":       "#9467bd",
    }

    def linestyle_for(label):
        return "--" if "@ 224" in label else "-"

    def linewidth_for(label):
        return 2.5 if "ResNet-18" in label else 1.5

    # Group by label so we can plot 5 seeds + 1 mean per model
    unique_labels = df_all["label"].unique()

    for ax, metric, ylabel, title, ylim in [
        (axes[0], "test_loss", "Test loss", "Test loss over training", (0.1, 2.1)),
        (axes[1], "test_acc", "Test accuracy", "Test accuracy over training (y-axis zoomed)", (0.30, 1.00)),
    ]:
        for label in unique_labels:
            subset = df_all[df_all["label"] == label]
            color = color_map.get(label, "#888888")
            ls = linestyle_for(label)
            lw = linewidth_for(label)

            # Each seed's individual curve, semi-transparent
            all_values = []
            for _, row in subset.iterrows():
                epochs = [h["epoch"] for h in row["history"]]
                values = [h[metric] for h in row["history"]]
                ax.plot(epochs, values, color=color, linewidth=0.8,
                        linestyle=ls, alpha=0.35, zorder=2)
                all_values.append(values)

            # Bold mean line on top
            mean_values = np.mean(all_values, axis=0)
            epochs = list(range(1, len(mean_values) + 1))
            ax.plot(epochs, mean_values, label=label, color=color,
                    linewidth=lw, linestyle=ls, marker="o", markersize=3,
                    alpha=0.95, zorder=3)

        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(*ylim)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 2016 BOVW line on accuracy panel
    axes[1].axhline(0.890, color="purple", linestyle=":", linewidth=2, alpha=0.7, zorder=0)
    axes[1].text(20.5, 0.890, " 2016 BOVW (0.89)", fontsize=10, color="purple",
                 va="center", ha="left", fontweight="bold")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               ncol=4, fontsize=10, frameon=True, framealpha=0.95,
               title="(solid = 384, dashed = 224; bold = mean of 5 seeds, faint = individual seeds)",
               title_fontsize=9)

    fig.suptitle(
        "Training dynamics across architectures (5 seeds per model: 42, 0, 1, 2, 3)",
        fontsize=14, fontweight="bold", y=0.99,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_path = OUTPUT_DIR / "fig2_training_curves_multiseed.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.show()


# ----------------------------------------------------------------------
# Figure 4 multiseed — Per-class heatmap of mean recall across seeds
# ----------------------------------------------------------------------
def figure_per_class_heatmap_multiseed(df_all: pd.DataFrame):
    """Heatmap of mean per-class recall across 5 seeds."""
    class_names = ["a", "b", "c", "d", "e", "f", "g", "h"]
    agg = aggregate_by_model(df_all)  # already sorted by mean_acc descending

    # Build the recall matrix: rows = models (sorted), columns = classes
    recall_matrix = np.zeros((len(agg), 8))
    for idx, (_, row) in enumerate(agg.iterrows()):
        recall_matrix[idx] = get_mean_per_class_recall(df_all, row["label"])

    # Add "Overall" column = mean accuracy across seeds
    overall_col = agg["mean_acc"].values.reshape(-1, 1)
    full_matrix = np.hstack([recall_matrix, overall_col])
    full_class_names = class_names + ["Overall"]

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(full_matrix, cmap="Blues", vmin=0.70, vmax=1.00, aspect="auto")
    ax.axvline(x=7.5, color="black", linewidth=1.2, zorder=3)

    for i in range(len(agg)):
        for j in range(len(full_class_names)):
            value = full_matrix[i, j]
            color = "white" if value > 0.88 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color=color, fontsize=10, zorder=4)

    ax.set_xticks(range(len(full_class_names)))
    ax.set_xticklabels(full_class_names, fontsize=11)
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(agg["label"].values, fontsize=10)
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Model (sorted by mean overall accuracy)", fontsize=11)

    ax.text(0.5, -0.18,
            "Classes e, f, g, h were designed to be the hardest in the DeCost & Holm 2016 dataset (closest distribution pairs).",
            ha="center", va="top", fontsize=9, style="italic",
            transform=ax.transAxes)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean per-class recall (across 5 seeds)", fontsize=10)

    fig.suptitle(
        "Per-class accuracy heatmap (mean across 5 seeds: 42, 0, 1, 2, 3)\n"
        "Rows = model (sorted by mean overall accuracy). "
        "Columns = class. Values are mean per-class recall.",
        fontsize=12, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig4_per_class_heatmap_multiseed.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.show()


# ----------------------------------------------------------------------
# Figure 5 multiseed — Params vs accuracy scatter with error bars
# ----------------------------------------------------------------------
def figure_params_vs_accuracy_multiseed(df_all: pd.DataFrame):
    """Scatter plot of params vs accuracy with vertical error bars."""
    agg = aggregate_by_model(df_all)

    labels = agg["label"].values
    mean_accs = agg["mean_acc"].values
    std_accs = agg["std_acc"].values
    params = agg["params_M"].values
    resolutions = agg["resolution"].values

    fig, ax = plt.subplots(figsize=(10, 7))

    colors_by_res = {224: "#E69F00", 384: "#0072B2"}
    point_colors = [colors_by_res[r] for r in resolutions]

    # Scatter with vertical error bars
    ax.errorbar(params, mean_accs, yerr=std_accs, fmt="none",
                ecolor="black", elinewidth=1.2, capsize=5, zorder=2)
    ax.scatter(params, mean_accs, c=point_colors, s=180, edgecolor="black",
               linewidth=1.2, zorder=3)

    label_offsets = {
        "ResNet-18 @ 384":       (0.6, 0.003),
        "ResNet-18 @ 224":       (0.6, -0.005),
        "ResNet-50 @ 384":       (0.6, 0.003),
        "ViT-Small @ 384":       (0.6, -0.012),
        "ConvNeXt-Tiny @ 384":   (-0.6, 0.005),
        "Swin-Tiny @ 224":       (0.6, 0.003),
        "EfficientNet-B0 @ 384": (0.6, 0.003),
    }
    for i, label in enumerate(labels):
        dx, dy = label_offsets.get(label, (0.6, 0.003))
        ha = "right" if dx < 0 else "left"
        ax.annotate(label,
                    xy=(params[i], mean_accs[i]),
                    xytext=(params[i] + dx, mean_accs[i] + dy),
                    fontsize=10, ha=ha, va="center", zorder=4)

    ax.axhline(y=0.890, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    ax.text(28.5, 0.890, "2016 BOVW baseline (0.890)",
            fontsize=9, va="center", ha="right", color="gray", style="italic")

    ax.axhline(y=0.902, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, zorder=2)
    ax.text(28.5, 0.902, "2016 watershed baseline (0.902)",
            fontsize=9, va="center", ha="right", color="gray", style="italic")

    ax.set_xlabel("Trainable parameters (millions)", fontsize=12)
    ax.set_ylabel("Test accuracy (mean ± std)", fontsize=12)
    ax.set_xlim(0, 32)
    ax.set_ylim(0.85, 1.00)
    ax.grid(True, alpha=0.3, zorder=1)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2",
               markeredgecolor="black", markersize=12, label="Input resolution: 384"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E69F00",
               markeredgecolor="black", markersize=12, label="Input resolution: 224"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.95)

    fig.suptitle(
        "Model size vs test accuracy (mean ± std across 5 seeds: 42, 0, 1, 2, 3)\n"
        "Larger models do not yield higher accuracy on this dataset.",
        fontsize=13, fontweight="bold", y=0.98,
    )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig5_params_vs_accuracy_multiseed.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.show()
    
# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # ===== Single-seed analysis (original 5 figures) =====
    df = load_all_results()
    print(f"\nLoaded {len(df)} single-seed runs:")
    print(df[["label", "best_test_acc", "params_M", "resolution", "train_time_min"]].to_string(index=False))

    print("\nGenerating original (single-seed) figures...")
    figure_main_comparison(df)
    figure_training_curves(df)
    figure_confusion_matrices(df)
    figure_per_class_heatmap(df)
    figure_params_vs_accuracy(df)

    # ===== Multi-seed analysis (Phase 5, four new figures) =====
    df_all = load_all_seeds()
    print(f"\nLoaded {len(df_all)} multi-seed runs:")
    print(aggregate_by_model(df_all).to_string(index=False))

    print("\nGenerating multi-seed figures...")
    figure_main_comparison_multiseed(df_all)
    figure_training_curves_multiseed(df_all)
    figure_per_class_heatmap_multiseed(df_all)
    figure_params_vs_accuracy_multiseed(df_all)


# ----------------------------------------------------------------------
# Figure 6 — Deep models at 512 vs. 2016-paper methods (resolution matched)
# All deep models and both classical baselines run at 512x512, so this is
# a like-for-like comparison with no downsampling on either side.
# Loads the *_size512 runs directly (independent of MULTISEED_REGISTRY).
# ----------------------------------------------------------------------
_ARCH_512 = {
    "resnet18":                     ("ResNet-18 @ 512",       "CNN (classic)"),
    "resnet50":                     ("ResNet-50 @ 512",       "CNN (classic)"),
    "convnext_tiny":                ("ConvNeXt-Tiny @ 512",   "CNN (modern)"),
    "efficientnet_b0":              ("EfficientNet-B0 @ 512", "CNN (efficient)"),
    "vit_small_patch16_384":        ("ViT-Small @ 512*",      "Transformer"),
    "swin_tiny_patch4_window7_224": ("Swin-Tiny @ 512*",      "Transformer"),
}


def load_512_runs() -> pd.DataFrame:
    """Load every *_size512 run by scanning the runs folder directly."""
    rows = []
    for run_dir in sorted(RESULTS_DIR.glob("*_size512")):
        results_json = run_dir / "results.json"
        if not results_json.exists():
            continue
        core = run_dir.name[:-len("_size512")]      # e.g. resnet50_seed3
        model_name, seed = core.rsplit("_seed", 1)
        if model_name not in _ARCH_512:
            continue
        with open(results_json) as f:
            data = json.load(f)
        rows.append({
            "model": model_name,
            "seed": int(seed),
            "best_test_acc": data["best_test_acc"],
        })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} runs at 512x512 "
          f"({df['model'].nunique() if not df.empty else 0} architectures)")
    return df


def figure_512_vs_baselines():
    """Bar chart: deep models @512 vs classical 2016-paper methods @512."""
    df = load_512_runs()
    if df.empty:
        print(f"No *_size512 runs found in {RESULTS_DIR}. Copy them from Drive first.")
        return

    family_colors = {
        "CNN (classic)":   "#1f77b4",
        "CNN (modern)":    "#2ca02c",
        "CNN (efficient)": "#17becf",
        "Transformer":     "#ff7f0e",
        "Reproduced":      "#A569BD",
    }

    bars = []
    for model_name, (label, family) in _ARCH_512.items():
        sub = df[df["model"] == model_name]
        if sub.empty:
            print(f"  (no 512 runs yet for {model_name})")
            continue
        bars.append({
            "label": label,
            "family": family,
            "mean": sub["best_test_acc"].mean(),
            "std": sub["best_test_acc"].std(ddof=1) if len(sub) > 1 else 0.0,
            "n": len(sub),
        })

    # Reproduced 2016-paper methods — these also ran at 512x512
    bars.append({"label": "Watershed + SVM\n(this work, 512)", "family": "Reproduced",
                 "mean": 0.9492, "std": 0.0, "n": None})
    bars.append({"label": "BOVW + SVM\n(this work, 512)", "family": "Reproduced",
                 "mean": 0.8613, "std": 0.0, "n": None})

    bars.sort(key=lambda b: b["mean"])          # highest ends up on top in barh

    labels = [b["label"] for b in bars]
    means = [b["mean"] for b in bars]
    stds = [b["std"] for b in bars]
    colors = [family_colors[b["family"]] for b in bars]
    y = np.arange(len(bars))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(y, means, xerr=stds, color=colors, edgecolor="black",
            linewidth=0.6, error_kw=dict(ecolor="black", capsize=4, lw=1.2),
            zorder=3)

    for yi, b in zip(y, bars):
        txt = f"{b['mean']:.3f}" + (f" ± {b['std']:.3f}" if b["n"] else "")
        ax.text(b["mean"] + b["std"] + 0.002, yi, txt,
                va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Test accuracy", fontsize=12, fontweight="bold")
    ax.set_xlim(0.80, 0.98)
    ax.set_title(
        "Deep models vs. 2016-paper methods — all at 512×512 resolution\n"
        "(deep models: mean ± std across 5 seeds; classical methods reproduced at 512)",
        fontsize=13, fontweight="bold")

    # 2016 published reference lines (the paper's own reported numbers)
    ax.axvline(0.890, color="purple", linestyle="--", linewidth=1.4, alpha=0.6, zorder=1)
    ax.text(0.890, len(bars) - 0.4, " 2016 BOVW (0.890)", color="purple",
            fontsize=9, va="top", ha="left", rotation=90)
    ax.axvline(0.902, color="gray", linestyle=":", linewidth=1.4, alpha=0.7, zorder=1)
    ax.text(0.902, len(bars) - 0.4, " 2016 Watershed (0.902)", color="gray",
            fontsize=9, va="top", ha="left", rotation=90)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, edgecolor="black", label=f)
                      for f, c in family_colors.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.95)

    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    fig.text(0.01, 0.01,
             "*ViT and Swin adapted to 512 (interpolated / padded position embeddings); "
             "the four CNNs run at 512 natively.",
             fontsize=8, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = OUTPUT_DIR / "fig6_deep512_vs_baselines.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\u2713 Saved: {out_path}")


# ======================================================================
# 512x512 versions of the four multi-seed figures.
# These load the *_size512 runs (all 6 architectures, 5 seeds each) and
# reproduce fig1/fig2/fig4/fig5 at full native resolution. They save to
# *_512.png filenames so the existing 224/384 figures are untouched.
# *ViT and Swin were adapted to 512 (interpolated / padded embeddings).
# ======================================================================
_MODEL_DEFS_512 = [
    # (model_name, label, family, params_M)
    ("resnet18",                     "ResNet-18 @ 512",       "CNN (classic)",   11.2),
    ("resnet50",                     "ResNet-50 @ 512",       "CNN (classic)",   25.5),
    ("convnext_tiny",                "ConvNeXt-Tiny @ 512",   "CNN (modern)",    27.8),
    ("efficientnet_b0",              "EfficientNet-B0 @ 512", "CNN (efficient)", 4.0),
    ("vit_small_patch16_384",        "ViT-Small @ 512*",      "Transformer",     22.0),
    ("swin_tiny_patch4_window7_224", "Swin-Tiny @ 512*",      "Transformer",     27.5),
]

_FAMILY_COLORS_512 = {
    "CNN (classic)": "#1f77b4", "CNN (modern)": "#2ca02c",
    "CNN (efficient)": "#17becf", "Transformer": "#ff7f0e",
    "Reference": "#9467bd", "Reproduced": "#A569BD", "Baseline": "#7f7f7f",
}


def load_all_seeds_512() -> pd.DataFrame:
    """Load all 30 runs (6 architectures x 5 seeds) at 512x512."""
    rows = []
    for model_name, label, family, params in _MODEL_DEFS_512:
        for seed in [42, 0, 1, 2, 3]:
            folder = f"{model_name}_seed{seed}_size512"
            results_json = RESULTS_DIR / folder / "results.json"
            if not results_json.exists():
                print(f"\u26a0 Missing: {results_json}")
                continue
            with open(results_json) as f:
                data = json.load(f)
            rows.append({
                "folder": folder, "label": label, "family": family,
                "params_M": params, "resolution": 512, "seed": seed,
                "best_test_acc": data["best_test_acc"], "history": data["history"],
            })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} runs at 512x512")
    return df


def _family_from_label_512(label):
    if "ResNet" in label:
        return "CNN (classic)"
    if "EfficientNet" in label:
        return "CNN (efficient)"
    if "ConvNeXt" in label:
        return "CNN (modern)"
    return "Transformer"


def figure_main_comparison_512(df_all: pd.DataFrame):
    """Fig 1 at 512: bar chart of mean test accuracy with std error bars."""
    if df_all.empty:
        print("No 512 runs loaded; nothing to plot."); return
    agg = aggregate_by_model(df_all)
    fig, ax = plt.subplots(figsize=(13, 8))

    bars = []
    for _, row in agg.iterrows():
        bars.append({"label": row["label"], "acc": row["mean_acc"],
                     "std": row["std_acc"], "family": _family_from_label_512(row["label"])})
    for ref in EXTERNAL_REFERENCES:
        bars.append({"label": ref["label"], "acc": ref["test_acc"],
                     "std": 0.0, "family": ref["family"]})
    bars = sorted(bars, key=lambda b: b["acc"])

    labels = [b["label"] for b in bars]
    accs = [b["acc"] for b in bars]
    stds = [b["std"] for b in bars]
    colors = [_FAMILY_COLORS_512[b["family"]] for b in bars]
    y = range(len(labels))

    ax.barh(y, accs, color=colors, edgecolor="black", linewidth=0.6,
            xerr=stds, error_kw={"ecolor": "black", "capsize": 4, "elinewidth": 1.2})
    for i, (acc, std) in enumerate(zip(accs, stds)):
        text = f"{acc:.3f} \u00b1 {std:.3f}" if std > 0 else f"{acc:.3f}"
        ax.text(acc + std + 0.005, i, text, va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0.75, 1.02)
    ax.set_xlabel("Test accuracy (mean \u00b1 std across 5 seeds)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Architecture comparison at 512\u00d7512 (full native resolution)\n"
        "(mean \u00b1 std across seeds 42, 0, 1, 2, 3; train = 1024 images, test = 1024 images)",
        fontsize=13, pad=15)
    ax.axvline(0.890, color="purple", linestyle="--", linewidth=1.5, alpha=0.6, zorder=0)
    ax.annotate("2016 BOVW\nbaseline (0.890)", xy=(0.890, len(labels) - 0.5),
                xytext=(0.890 - 0.035, len(labels) - 0.2), fontsize=9, color="purple",
                ha="center", arrowprops=dict(arrowstyle="->", color="purple", alpha=0.5, lw=1))
    ax.set_axisbelow(True); ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="black", label=f)
                       for f, c in _FAMILY_COLORS_512.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
              framealpha=0.95, bbox_to_anchor=(0.98, 0.02))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.text(0.01, 0.005, "*ViT and Swin adapted to 512 (interpolated / padded embeddings).",
             fontsize=8, style="italic", color="#555555")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig1_main_comparison_512.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\u2713 Saved: {out_path}")
    plt.show()


def figure_training_curves_512(df_all: pd.DataFrame):
    """Fig 2 at 512: training curves, 5 faint seeds + bold mean per model."""
    if df_all.empty:
        print("No 512 runs loaded; nothing to plot."); return
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    color_map = {
        "ResNet-18 @ 512":       "#1f77b4",
        "ResNet-50 @ 512":       "#ff7f0e",
        "EfficientNet-B0 @ 512": "#17becf",
        "ConvNeXt-Tiny @ 512":   "#2ca02c",
        "ViT-Small @ 512*":      "#d62728",
        "Swin-Tiny @ 512*":      "#9467bd",
    }
    unique_labels = df_all["label"].unique()
    for ax, metric, ylabel, title, ylim in [
        (axes[0], "test_loss", "Test loss", "Test loss over training", (0.1, 2.1)),
        (axes[1], "test_acc", "Test accuracy", "Test accuracy over training (y-axis zoomed)", (0.30, 1.00)),
    ]:
        for label in unique_labels:
            subset = df_all[df_all["label"] == label]
            color = color_map.get(label, "#888888")
            lw = 2.5 if "ResNet-18" in label else 1.5
            all_values = []
            for _, row in subset.iterrows():
                epochs = [h["epoch"] for h in row["history"]]
                values = [h[metric] for h in row["history"]]
                ax.plot(epochs, values, color=color, linewidth=0.8, alpha=0.35, zorder=2)
                all_values.append(values)
            mean_values = np.mean(all_values, axis=0)
            epochs = list(range(1, len(mean_values) + 1))
            ax.plot(epochs, mean_values, label=label, color=color, linewidth=lw,
                    marker="o", markersize=3, alpha=0.95, zorder=3)
        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0.5, 20.5); ax.set_ylim(*ylim)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[1].axhline(0.890, color="purple", linestyle=":", linewidth=2, alpha=0.7, zorder=0)
    axes[1].text(20.5, 0.890, " 2016 BOVW (0.89)", fontsize=10, color="purple",
                 va="center", ha="left", fontweight="bold")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               ncol=3, fontsize=10, frameon=True, framealpha=0.95,
               title="(all models at 512; bold = mean of 5 seeds, faint = individual seeds)",
               title_fontsize=9)
    fig.suptitle("Training dynamics at 512\u00d7512 (5 seeds per model: 42, 0, 1, 2, 3)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_path = OUTPUT_DIR / "fig2_training_curves_512.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\u2713 Saved: {out_path}")
    plt.show()


def _mean_recall_512(df_all, label):
    """Mean per-class recall across seeds for one label; robust to missing npz."""
    subset = df_all[df_all["label"] == label]
    recalls = []
    for _, row in subset.iterrows():
        npz_path = RESULTS_DIR / row["folder"] / "predictions.npz"
        if not npz_path.exists():
            continue
        cm = np.load(npz_path)["confusion_matrix"]
        recalls.append(cm.diagonal() / cm.sum(axis=1))
    if not recalls:
        return np.full(8, np.nan)
    return np.array(recalls).mean(axis=0)


def figure_per_class_heatmap_512(df_all: pd.DataFrame):
    """Fig 4 at 512: per-class mean recall heatmap."""
    if df_all.empty:
        print("No 512 runs loaded; nothing to plot."); return
    class_names = ["a", "b", "c", "d", "e", "f", "g", "h"]
    agg = aggregate_by_model(df_all)
    recall_matrix = np.zeros((len(agg), 8))
    for idx, (_, row) in enumerate(agg.iterrows()):
        recall_matrix[idx] = _mean_recall_512(df_all, row["label"])
    overall_col = agg["mean_acc"].values.reshape(-1, 1)
    full_matrix = np.hstack([recall_matrix, overall_col])
    full_class_names = class_names + ["Overall"]

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(full_matrix, cmap="Blues", vmin=0.70, vmax=1.00, aspect="auto")
    ax.axvline(x=7.5, color="black", linewidth=1.2, zorder=3)
    for i in range(len(agg)):
        for j in range(len(full_class_names)):
            value = full_matrix[i, j]
            if np.isnan(value):
                ax.text(j, i, "n/a", ha="center", va="center", color="black", fontsize=9, zorder=4)
                continue
            color = "white" if value > 0.88 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10, zorder=4)
    ax.set_xticks(range(len(full_class_names))); ax.set_xticklabels(full_class_names, fontsize=11)
    ax.set_yticks(range(len(agg))); ax.set_yticklabels(agg["label"].values, fontsize=10)
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Model (sorted by mean overall accuracy)", fontsize=11)
    ax.text(0.5, -0.18,
            "Classes e, f, g, h were designed to be the hardest in the DeCost & Holm 2016 dataset (closest distribution pairs).",
            ha="center", va="top", fontsize=9, style="italic", transform=ax.transAxes)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean per-class recall (across 5 seeds)", fontsize=10)
    fig.suptitle(
        "Per-class accuracy heatmap at 512\u00d7512 (mean across 5 seeds: 42, 0, 1, 2, 3)\n"
        "Rows = model (sorted by mean overall accuracy). Columns = class. Values are mean per-class recall.",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig4_per_class_heatmap_512.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\u2713 Saved: {out_path}")
    plt.show()


def figure_params_vs_accuracy_512(df_all: pd.DataFrame):
    """Fig 5 at 512: params vs accuracy, colored by family (all one resolution)."""
    if df_all.empty:
        print("No 512 runs loaded; nothing to plot."); return
    agg = aggregate_by_model(df_all)
    labels = agg["label"].values
    mean_accs = agg["mean_acc"].values
    std_accs = agg["std_acc"].values
    params = agg["params_M"].values
    families = [_family_from_label_512(l) for l in labels]
    point_colors = [_FAMILY_COLORS_512[f] for f in families]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(params, mean_accs, yerr=std_accs, fmt="none",
                ecolor="black", elinewidth=1.2, capsize=5, zorder=2)
    ax.scatter(params, mean_accs, c=point_colors, s=180, edgecolor="black",
               linewidth=1.2, zorder=3)
    for i, label in enumerate(labels):
        ax.annotate(label, xy=(params[i], mean_accs[i]),
                    xytext=(params[i] + 0.6, mean_accs[i] + 0.003),
                    fontsize=10, ha="left", va="center", zorder=4)
    ax.axhline(y=0.890, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    ax.text(31.5, 0.890, "2016 BOVW baseline (0.890)", fontsize=9, va="center",
            ha="right", color="gray", style="italic")
    ax.axhline(y=0.902, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, zorder=2)
    ax.text(31.5, 0.902, "2016 watershed baseline (0.902)", fontsize=9, va="center",
            ha="right", color="gray", style="italic")
    ax.set_xlabel("Trainable parameters (millions)", fontsize=12)
    ax.set_ylabel("Test accuracy (mean \u00b1 std)", fontsize=12)
    ax.set_xlim(0, 32); ax.set_ylim(0.85, 1.00)
    ax.grid(True, alpha=0.3, zorder=1)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=_FAMILY_COLORS_512[f], edgecolor="black", label=f)
                       for f in ["CNN (classic)", "CNN (modern)", "CNN (efficient)", "Transformer"]]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.95)
    fig.suptitle(
        "Model size vs test accuracy at 512\u00d7512 (mean \u00b1 std across 5 seeds: 42, 0, 1, 2, 3)\n"
        "Larger models do not yield higher accuracy on this dataset.",
        fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig5_params_vs_accuracy_512.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\u2713 Saved: {out_path}")
    plt.show()
