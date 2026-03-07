#!/usr/bin/env python3
"""
Plotting utilities for subset diversity analysis.

Uses matplotlib for histograms, boxplots, and PCA scatter plots.
All functions are self-contained — they create, save, and close figures.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger("plotting_utils")

# Defer matplotlib import to avoid issues in headless environments
_MPL_AVAILABLE = None


def _check_matplotlib() -> bool:
    """Lazy-check matplotlib availability."""
    global _MPL_AVAILABLE
    if _MPL_AVAILABLE is None:
        try:
            import matplotlib

            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt  # noqa: F401

            _MPL_AVAILABLE = True
        except ImportError:
            _MPL_AVAILABLE = False
            logger.warning("matplotlib not available — plots will be skipped")
    return _MPL_AVAILABLE


def plot_metric_histograms(rows: list[dict], out_dir: Path) -> None:
    """Plot histogram for each metric across subsets.

    Args:
        rows: List of dicts, each with keys: subset_id, n_samples,
              cosine_distance, l2_distance, variance_score.
        out_dir: Directory to save plot images.
    """
    if not _check_matplotlib() or not rows:
        return
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["cosine_distance", "l2_distance", "variance_score"]
    labels = ["Cosine Distance", "L2 Distance", "Variance Score"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric, label in zip(axes, metrics, labels):
        values = [r[metric] for r in rows if metric in r]
        if not values:
            continue
        ax.hist(values, bins=max(5, len(values) // 3), edgecolor="black", alpha=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label}")
        ax.axvline(
            sum(values) / len(values),
            color="red",
            linestyle="--",
            label=f"Mean={sum(values) / len(values):.4f}",
        )
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "metric_histograms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved histogram plot: {path}")


def plot_metric_boxplots(rows: list[dict], out_dir: Path) -> None:
    """Plot boxplots for each metric across subsets.

    Args:
        rows: List of dicts with metric values.
        out_dir: Directory to save plot images.
    """
    if not _check_matplotlib() or not rows:
        return
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["cosine_distance", "l2_distance", "variance_score"]
    labels = ["Cosine Distance", "L2 Distance", "Variance Score"]

    data = []
    tick_labels = []
    for metric, label in zip(metrics, labels):
        values = [r[metric] for r in rows if metric in r]
        if values:
            data.append(values)
            tick_labels.append(label)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(data) * 2), 5))
    bp = ax.boxplot(data, tick_labels=tick_labels, patch_artist=True)
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Diversity Metrics Across Subsets")
    ax.set_ylabel("Value")

    fig.tight_layout()
    path = out_dir / "metric_boxplots.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved boxplot: {path}")


def plot_pca_scatter(
    emb: torch.Tensor,
    labels: list[str],
    out_path: Path,
    title: str = "PCA Scatter",
) -> None:
    """2D PCA scatter plot of embeddings using torch.pca_lowrank.

    Args:
        emb: Embedding matrix of shape [N, D].
        labels: Label for each point (length N).
        out_path: File path to save the plot.
        title: Plot title.
    """
    if not _check_matplotlib() or emb.shape[0] < 2:
        return
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # PCA via torch (no sklearn dependency)
    emb_centered = emb - emb.mean(dim=0, keepdim=True)
    _, _, V = torch.pca_lowrank(emb_centered.float(), q=2)
    coords = (emb_centered.float() @ V[:, :2]).numpy()  # [N, 2]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        coords[:, 0], coords[:, 1], alpha=0.7, edgecolors="black", linewidths=0.5
    )

    # Annotate points (truncate long labels)
    for i, label in enumerate(labels):
        short = label[:15] + "…" if len(label) > 15 else label
        ax.annotate(short, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved PCA scatter: {out_path}")


def plot_similarity_heatmap(
    sim_matrix: torch.Tensor,
    labels: list[str],
    out_path: Path,
    title: str = "Pairwise Cosine Similarity",
) -> None:
    """Plot a heatmap of a similarity/distance matrix.

    Args:
        sim_matrix: Square matrix [N, N].
        labels: Label for each row/col.
        out_path: File path to save the plot.
        title: Plot title.
    """
    if not _check_matplotlib():
        return
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) * 0.8)))
    im = ax.imshow(sim_matrix.numpy(), cmap="RdYlGn", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short_labels = [l[:12] + "…" if len(l) > 12 else l for l in labels]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = sim_matrix[i, j].item()
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved heatmap: {out_path}")
