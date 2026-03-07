#!/usr/bin/env python3
"""
Analyze image subset diversity using DINOv2 features.

Workflow:
  1. Discover all images in a directory
  2. Extract CLS features for ALL images (once, batched)
  3. Generate multiple subsets via different sampling strategies
  4. Compute diversity metrics per subset
  5. Output metrics.csv + plots

Usage:
    # Using local dataset (after manage_local_datasets.py --download images)
    python extract_feature/analyze_image_subset_diversity.py \\
        --image-dir extract_feature/local_datasets/images \\
        --subset-size 3 --n-subsets 10 \\
        --output-dir extract_feature/outputs/image_diversity

    # Demo mode: download sample images automatically
    python extract_feature/analyze_image_subset_diversity.py --demo \\
        --subset-size 3 --n-subsets 5 \\
        --output-dir extract_feature/outputs/image_diversity_demo
"""

import argparse
import csv
import logging
import sys
import tempfile
from pathlib import Path

import torch

# Re-use model/transform/image loading from existing script
from extract_image_features import (
    AVAILABLE_MODELS,
    DEMO_URLS,
    download_demo_images,
    extract_cls_features,
    load_images,
    load_model,
    make_transform,
)

from diversity_metrics import compute_all_metrics, pairwise_cosine_matrix
from subset_sampling import generate_subset_indices
from plotting_utils import (
    plot_metric_boxplots,
    plot_metric_histograms,
    plot_pca_scatter,
    plot_similarity_heatmap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze_image_subset_diversity")


def discover_images(image_dir: Path) -> list[Path]:
    """Find all image files in a directory."""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in suffixes)
    logger.info(f"Found {len(paths)} images in {image_dir}")
    return paths


def extract_all_cls_features(
    image_paths: list[Path],
    model_name: str,
    device: torch.device,
    batch_size: int = 8,
    resize: int = 256,
    crop: int = 224,
) -> tuple[torch.Tensor, list[str]]:
    """Extract CLS features for all images, processing in batches for memory.

    Returns:
        all_features: [N, D] tensor of CLS features (on CPU).
        all_names: List of image filenames.
    """
    transform = make_transform(resize, crop)
    model = load_model(model_name, device)

    # Load all images
    batch, names = load_images(image_paths, transform)
    n_images = batch.shape[0]
    logger.info(f"Extracting CLS features for {n_images} images...")

    # Process in batches
    all_cls = []
    n_batches = (n_images + batch_size - 1) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_images)
        sub_batch = batch[start:end]
        cls = extract_cls_features(model, sub_batch, device)
        all_cls.append(cls.cpu())
        if n_batches > 1:
            logger.info(f"  Batch {i + 1}/{n_batches} done ({end}/{n_images} images)")

    all_features = torch.cat(all_cls, dim=0)  # [N, D]
    logger.info(f"All features extracted: shape={list(all_features.shape)}")
    return all_features, names


def analyze_subsets(
    all_features: torch.Tensor,
    all_names: list[str],
    subset_size: int,
    n_subsets: int,
    strategy: str = "random",
    seed: int = 42,
) -> list[dict]:
    """Generate subsets and compute diversity metrics for each.

    Returns list of metric dicts, one per subset.
    """
    n_images = all_features.shape[0]
    indices_list = generate_subset_indices(
        n_images, subset_size, n_subsets, seed=seed, strategy=strategy
    )

    rows = []
    for i, indices in enumerate(indices_list):
        subset_features = all_features[indices]  # [subset_size, D]
        metrics = compute_all_metrics(subset_features)
        metrics["subset_id"] = i
        metrics["strategy"] = strategy
        metrics["image_names"] = ",".join(all_names[idx] for idx in indices)
        rows.append(metrics)

        logger.info(
            f"  Subset {i}: size={metrics['n_samples']}, "
            f"cos_dist={metrics['cosine_distance']:.4f}, "
            f"l2_dist={metrics['l2_distance']:.4f}, "
            f"var={metrics['variance_score']:.6f}"
        )

    return rows


def save_metrics_csv(rows: list[dict], output_path: Path) -> None:
    """Save subset metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subset_id",
        "strategy",
        "n_samples",
        "cosine_distance",
        "l2_distance",
        "variance_score",
        "image_names",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Metrics saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze image subset diversity using DINOv2"
    )
    parser.add_argument(
        "--model",
        default="dinov2_vits14",
        choices=AVAILABLE_MODELS,
        help="Model variant",
    )
    parser.add_argument("--image-dir", type=Path, help="Directory of images")
    parser.add_argument(
        "--demo", action="store_true", help="Download and use demo images from COCO"
    )
    parser.add_argument(
        "--subset-size", type=int, required=True, help="Number of images per subset"
    )
    parser.add_argument(
        "--n-subsets", type=int, default=10, help="Number of subsets to generate"
    )
    parser.add_argument(
        "--strategy",
        default="random",
        choices=["random", "sliding"],
        help="Subset sampling strategy",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for GPU inference"
    )
    parser.add_argument("--resize", type=int, default=256, help="Resize shorter side")
    parser.add_argument("--crop", type=int, default=224, help="Center crop size")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── 1. Collect image paths ──
    if args.demo:
        demo_dir = Path(tempfile.mkdtemp(prefix="dinov2_diversity_demo_"))
        logger.info(f"Demo mode: downloading {len(DEMO_URLS)} sample images")
        image_paths = download_demo_images(DEMO_URLS, demo_dir)
    elif args.image_dir:
        image_paths = discover_images(args.image_dir)
    else:
        parser.error("Provide --image-dir or --demo")

    if len(image_paths) < 2:
        logger.error(f"Need at least 2 images, got {len(image_paths)}")
        sys.exit(1)

    if args.subset_size > len(image_paths):
        logger.error(
            f"subset_size ({args.subset_size}) > number of images ({len(image_paths)})"
        )
        sys.exit(1)

    # ── 2. Extract ALL features once ──
    logger.info("=" * 60)
    logger.info("Step 1: Extracting CLS features for all images")
    logger.info("=" * 60)
    all_features, all_names = extract_all_cls_features(
        image_paths,
        args.model,
        device,
        batch_size=args.batch_size,
        resize=args.resize,
        crop=args.crop,
    )

    # ── 3. Generate subsets and compute metrics ──
    logger.info("\n" + "=" * 60)
    logger.info(
        f"Step 2: Analyzing {args.n_subsets} subsets (size={args.subset_size}, strategy={args.strategy})"
    )
    logger.info("=" * 60)
    rows = analyze_subsets(
        all_features,
        all_names,
        args.subset_size,
        args.n_subsets,
        strategy=args.strategy,
        seed=args.seed,
    )

    # ── 4. Full-dataset metrics ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Full-dataset diversity metrics")
    logger.info("=" * 60)
    full_metrics = compute_all_metrics(all_features)
    logger.info(f"  Full dataset ({len(all_names)} images):")
    logger.info(f"    cosine_distance:  {full_metrics['cosine_distance']:.4f}")
    logger.info(f"    l2_distance:      {full_metrics['l2_distance']:.4f}")
    logger.info(f"    variance_score:   {full_metrics['variance_score']:.6f}")

    # ── 5. Summary statistics ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Subset summary statistics")
    logger.info("=" * 60)
    for metric_key in ["cosine_distance", "l2_distance", "variance_score"]:
        values = [r[metric_key] for r in rows]
        mean_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        logger.info(
            f"  {metric_key}: mean={mean_val:.4f}, min={min_val:.4f}, max={max_val:.4f}"
        )

    # ── 6. Save results ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Saving results")
    logger.info("=" * 60)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    save_metrics_csv(rows, output_dir / "metrics.csv")

    # Plots
    plot_metric_histograms(rows, output_dir)
    plot_metric_boxplots(rows, output_dir)

    # PCA scatter of all images
    plot_pca_scatter(
        all_features,
        all_names,
        output_dir / "pca_scatter.png",
        title="Image Embeddings (PCA)",
    )

    # Pairwise similarity heatmap of all images
    sim_matrix = pairwise_cosine_matrix(all_features)
    plot_similarity_heatmap(
        sim_matrix,
        all_names,
        output_dir / "similarity_heatmap.png",
        title="Image Pairwise Cosine Similarity",
    )

    # Save features
    torch.save(
        {
            "model": args.model,
            "image_names": all_names,
            "cls_features": all_features,
            "full_metrics": full_metrics,
            "subset_metrics": rows,
        },
        output_dir / "features_and_metrics.pt",
    )
    logger.info(f"Features + metrics saved to {output_dir / 'features_and_metrics.pt'}")

    logger.info(f"\nAll results saved to {output_dir}/")
    logger.info("Done.")


if __name__ == "__main__":
    main()
