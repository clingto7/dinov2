#!/usr/bin/env python3
"""
Analyze video subset diversity using DINOv2 features.

Workflow:
  1. Discover all video files in a directory
  2. For each video: extract CLS features → compute temporal mean → get one embedding [D]
  3. Generate multiple subsets of videos
  4. Compute diversity metrics per subset
  5. Output metrics.csv + plots

Usage:
    # Using local dataset (after manage_local_datasets.py --download videos)
    python extract_feature/analyze_video_subset_diversity.py \\
        --video-dir extract_feature/local_datasets/videos \\
        --subset-size 2 --n-subsets 5 --fps 1 --max-frames 12 \\
        --output-dir extract_feature/outputs/video_diversity

    # Demo mode: download sample videos automatically
    python extract_feature/analyze_video_subset_diversity.py --demo \\
        --subset-size 2 --n-subsets 3 --max-frames 10 \\
        --output-dir extract_feature/outputs/video_diversity_demo
"""

import argparse
import csv
import logging
import os
import sys
import tempfile
from pathlib import Path

import torch

# Re-use functions from existing video script
from extract_video_features import (
    AVAILABLE_MODELS,
    DEMO_VIDEOS,
    extract_features_batched,
    frames_to_batch,
    load_model,
    make_transform,
    sample_frames,
)

from diversity_metrics import compute_all_metrics, pairwise_cosine_matrix
from subset_sampling import generate_subset_indices
from plotting_utils import (
    plot_metric_boxplots,
    plot_metric_histograms,
    plot_pca_scatter,
    plot_similarity_heatmap,
)

# Bypass SOCKS proxy for HuggingFace downloads
os.environ.setdefault("ALL_PROXY", "")
os.environ.setdefault("all_proxy", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze_video_subset_diversity")

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def discover_videos(video_dir: Path) -> list[Path]:
    """Find all video files in a directory."""
    paths = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    logger.info(f"Found {len(paths)} videos in {video_dir}")
    return paths


def download_demo_videos_simple(output_dir: Path, max_count: int = 4) -> list[Path]:
    """Download demo videos for testing. Returns list of paths."""
    import requests

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    items = list(DEMO_VIDEOS.items())[:max_count]
    for i, (name, info) in enumerate(items):
        filepath = output_dir / info["filename"]
        if filepath.exists():
            logger.info(f"  [{i + 1}/{len(items)}] Already exists: {filepath.name}")
        else:
            logger.info(
                f"  [{i + 1}/{len(items)}] Downloading: {name} ({info['source']})"
            )
            try:
                resp = requests.get(info["url"], timeout=120, stream=True)
                resp.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                logger.info(f"    Downloaded to {filepath}")
            except Exception as e:
                logger.warning(f"    Failed to download {name}: {e}")
                continue
        paths.append(filepath)

    return paths


def extract_video_embedding(
    video_path: Path,
    model: torch.nn.Module,
    transform: torch.nn.Module,
    device: torch.device,
    fps: float = 1.0,
    max_frames: int = 20,
    batch_size: int = 16,
) -> tuple[torch.Tensor, dict]:
    """Extract a single embedding for a video (temporal mean of CLS features).

    Returns:
        embedding: [D] tensor (CPU).
        info: dict with video metadata and per-frame stats.
    """
    frames, actual_fps, video_info, is_rgb = sample_frames(
        video_path, target_fps=fps, max_frames=max_frames
    )
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    batch = frames_to_batch(frames, transform, is_rgb=is_rgb)
    features = extract_features_batched(model, batch, device, batch_size=batch_size)

    cls_features = features["cls_features"]  # [N_frames, D]
    temporal_mean = cls_features.mean(dim=0)  # [D] — video-level embedding
    temporal_var = cls_features.var(dim=0).mean().item()

    info = {
        "video_name": video_path.name,
        "n_frames": len(frames),
        "duration_sec": video_info["duration_sec"],
        "temporal_variance": temporal_var,
    }
    return temporal_mean, info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze video subset diversity using DINOv2"
    )
    parser.add_argument(
        "--model",
        default="dinov2_vits14",
        choices=AVAILABLE_MODELS,
        help="Model variant",
    )
    parser.add_argument("--video-dir", type=Path, help="Directory of videos")
    parser.add_argument(
        "--demo", action="store_true", help="Download and use demo videos"
    )
    parser.add_argument(
        "--demo-count", type=int, default=4, help="Number of demo videos to use"
    )
    parser.add_argument(
        "--subset-size", type=int, required=True, help="Number of videos per subset"
    )
    parser.add_argument(
        "--n-subsets", type=int, default=5, help="Number of subsets to generate"
    )
    parser.add_argument(
        "--strategy",
        default="random",
        choices=["random", "sliding"],
        help="Subset sampling strategy",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--fps", type=float, default=1.0, help="Target sampling FPS per video"
    )
    parser.add_argument(
        "--max-frames", type=int, default=20, help="Max frames per video"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for GPU inference"
    )
    parser.add_argument("--resize", type=int, default=256, help="Resize shorter side")
    parser.add_argument("--crop", type=int, default=224, help="Center crop size")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── 1. Collect video paths ──
    if args.demo:
        demo_dir = Path(tempfile.mkdtemp(prefix="dinov2_video_diversity_demo_"))
        logger.info(f"Demo mode: downloading up to {args.demo_count} demo videos")
        video_paths = download_demo_videos_simple(demo_dir, max_count=args.demo_count)
    elif args.video_dir:
        video_paths = discover_videos(args.video_dir)
    else:
        parser.error("Provide --video-dir or --demo")

    if len(video_paths) < 2:
        logger.error(f"Need at least 2 videos, got {len(video_paths)}")
        sys.exit(1)

    if args.subset_size > len(video_paths):
        logger.error(
            f"subset_size ({args.subset_size}) > number of videos ({len(video_paths)})"
        )
        sys.exit(1)

    # ── 2. Extract per-video embeddings ──
    logger.info("=" * 60)
    logger.info(f"Step 1: Extracting embeddings for {len(video_paths)} videos")
    logger.info("=" * 60)

    model = load_model(args.model, device)
    transform = make_transform(args.resize, args.crop)

    embeddings = []
    video_names = []
    video_infos = []
    for i, vpath in enumerate(video_paths):
        logger.info(f"\n--- Video {i + 1}/{len(video_paths)}: {vpath.name} ---")
        try:
            emb, info = extract_video_embedding(
                vpath,
                model,
                transform,
                device,
                fps=args.fps,
                max_frames=args.max_frames,
                batch_size=args.batch_size,
            )
            embeddings.append(emb)
            video_names.append(vpath.stem)
            video_infos.append(info)
            logger.info(
                f"  Embedding dim={emb.shape[0]}, temporal_var={info['temporal_variance']:.6f}, "
                f"frames={info['n_frames']}, duration={info['duration_sec']:.1f}s"
            )
        except Exception as e:
            logger.warning(f"  Failed to process {vpath.name}: {e}")

    if len(embeddings) < 2:
        logger.error("Need at least 2 successful video embeddings")
        sys.exit(1)

    all_features = torch.stack(embeddings)  # [V, D]
    logger.info(f"\nAll embeddings: shape={list(all_features.shape)}")

    # ── 3. Generate subsets and compute metrics ──
    logger.info("\n" + "=" * 60)
    logger.info(
        f"Step 2: Analyzing {args.n_subsets} subsets (size={args.subset_size}, strategy={args.strategy})"
    )
    logger.info("=" * 60)

    n_videos = all_features.shape[0]
    actual_subset_size = min(args.subset_size, n_videos)
    indices_list = generate_subset_indices(
        n_videos,
        actual_subset_size,
        args.n_subsets,
        seed=args.seed,
        strategy=args.strategy,
    )

    rows = []
    for i, indices in enumerate(indices_list):
        subset_features = all_features[indices]
        metrics = compute_all_metrics(subset_features)
        metrics["subset_id"] = i
        metrics["strategy"] = args.strategy
        metrics["video_names"] = ",".join(video_names[idx] for idx in indices)
        rows.append(metrics)

        logger.info(
            f"  Subset {i}: size={metrics['n_samples']}, "
            f"cos_dist={metrics['cosine_distance']:.4f}, "
            f"l2_dist={metrics['l2_distance']:.4f}, "
            f"var={metrics['variance_score']:.6f}"
        )

    # ── 4. Full-dataset metrics ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Full-dataset diversity metrics")
    logger.info("=" * 60)
    full_metrics = compute_all_metrics(all_features)
    logger.info(f"  Full dataset ({len(video_names)} videos):")
    logger.info(f"    cosine_distance:  {full_metrics['cosine_distance']:.4f}")
    logger.info(f"    l2_distance:      {full_metrics['l2_distance']:.4f}")
    logger.info(f"    variance_score:   {full_metrics['variance_score']:.6f}")

    # ── 5. Per-video temporal variance summary ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Per-video temporal variance (content dynamics)")
    logger.info("=" * 60)
    for info in video_infos:
        logger.info(
            f"  {info['video_name']}: temporal_var={info['temporal_variance']:.6f}, frames={info['n_frames']}"
        )

    # ── 6. Save results ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Saving results")
    logger.info("=" * 60)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    fieldnames = [
        "subset_id",
        "strategy",
        "n_samples",
        "cosine_distance",
        "l2_distance",
        "variance_score",
        "video_names",
    ]
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Metrics CSV saved to {csv_path}")

    # Plots
    plot_metric_histograms(rows, output_dir)
    plot_metric_boxplots(rows, output_dir)
    plot_pca_scatter(
        all_features,
        video_names,
        output_dir / "pca_scatter.png",
        title="Video Embeddings (PCA)",
    )

    sim_matrix = pairwise_cosine_matrix(all_features)
    plot_similarity_heatmap(
        sim_matrix,
        video_names,
        output_dir / "similarity_heatmap.png",
        title="Video Pairwise Cosine Similarity",
    )

    # Save features
    torch.save(
        {
            "model": args.model,
            "video_names": video_names,
            "video_infos": video_infos,
            "embeddings": all_features,
            "full_metrics": full_metrics,
            "subset_metrics": rows,
            "similarity_matrix": sim_matrix,
        },
        output_dir / "features_and_metrics.pt",
    )
    logger.info(f"Features + metrics saved to {output_dir / 'features_and_metrics.pt'}")

    logger.info(f"\nAll results saved to {output_dir}/")
    logger.info("Done.")


if __name__ == "__main__":
    main()
