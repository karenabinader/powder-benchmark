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
# Main
# ----------------------------------------------------------------------
def main():
    df = load_all_results()
    print(f"\nLoaded {len(df)} runs:")
    print(df[["label", "best_test_acc", "params_M", "resolution", "train_time_min"]].to_string(index=False))

    print("\nGenerating figures...")
    figure_main_comparison(df)


if __name__ == "__main__":
    main()