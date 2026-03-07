#!/usr/bin/env python3
"""
Analyze RoboTwin episode subset diversity using DINOv2 features.

Workflow:
  1. Load episode metadata for a RoboTwin task
  2. For each episode: extract frames -> CLS features -> temporal mean embedding [D]
  3. Generate multiple subsets of episodes
  4. Compute diversity metrics per subset
  5. Output metrics.csv + plots

Usage:
    # Demo mode: beat_block_hammer/clean first 10 episodes
    python extract_feature/analyze_robotwin_diversity.py --demo \
        --subset-size 3 --n-subsets 5 \
        --output-dir extract_feature/outputs/robotwin_diversity_demo

    # Task mode
    python extract_feature/analyze_robotwin_diversity.py \
        --task beat_block_hammer --variant clean \
        --subset-size 5 --n-subsets 10 \
        --output-dir extract_feature/outputs/robotwin_diversity
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image

from extract_video_features import (
    AVAILABLE_MODELS,
    load_model,
    make_transform,
    extract_features_batched,
)
from diversity_metrics import compute_all_metrics, pairwise_cosine_matrix
from subset_sampling import generate_subset_indices
from plotting_utils import (
    plot_metric_boxplots,
    plot_metric_histograms,
    plot_pca_scatter,
    plot_similarity_heatmap,
)
from robotwin_loader import (
    ALL_TASKS,
    CAMERA_KEYS,
    load_task_metadata,
    download_video,
    extract_episode_frames,
)

os.environ.setdefault("ALL_PROXY", "")
os.environ.setdefault("all_proxy", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze_robotwin_diversity")


def pil_frames_to_batch(frames: list[Image.Image], transform) -> torch.Tensor:
    return torch.stack([transform(img) for img in frames])


def parse_episode_indices(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("--episode-indices is empty after parsing")

    indices = []
    for part in parts:
        try:
            idx = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid episode index: {part!r}") from exc
        if idx < 0:
            raise ValueError(f"Episode index must be >= 0, got {idx}")
        indices.append(idx)

    return sorted(set(indices))


def select_episode_indices(
    all_episode_indices: list[int],
    n_episodes: int | None,
    episode_indices_arg: str | None,
    demo: bool,
) -> list[int]:
    if demo:
        return all_episode_indices[:10]

    if episode_indices_arg:
        requested = parse_episode_indices(episode_indices_arg)
        available = set(all_episode_indices)
        missing = [idx for idx in requested if idx not in available]
        if missing:
            raise ValueError(f"Requested episodes not found in dataset: {missing}")
        return requested

    if n_episodes is None:
        return all_episode_indices

    if n_episodes < 1:
        raise ValueError("--n-episodes must be >= 1")
    return all_episode_indices[:n_episodes]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze RoboTwin episode subset diversity using DINOv2"
    )
    parser.add_argument(
        "--task", type=str, choices=ALL_TASKS, help="RoboTwin task name"
    )
    parser.add_argument(
        "--variant",
        default="clean",
        choices=["clean", "randomized"],
        help="Dataset variant",
    )
    parser.add_argument(
        "--camera",
        default="observation.images.cam_high",
        choices=CAMERA_KEYS,
        help="Camera key",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Number of episodes to analyze (default: all)",
    )
    parser.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help="Comma-separated specific episode indices",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use beat_block_hammer/clean with first 10 episodes",
    )
    parser.add_argument(
        "--model",
        default="dinov2_vits14",
        choices=AVAILABLE_MODELS,
        help="DINOv2 model variant",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=2.0,
        help="Target sampling FPS per episode",
    )
    parser.add_argument(
        "--max-frames", type=int, default=10, help="Max frames per episode"
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        required=True,
        help="Number of episodes per subset",
    )
    parser.add_argument(
        "--n-subsets",
        type=int,
        default=5,
        help="Number of subsets to generate",
    )
    parser.add_argument(
        "--strategy",
        default="random",
        choices=["random", "sliding"],
        help="Subset sampling strategy",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for GPU inference"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    if args.demo:
        task_name = "beat_block_hammer"
        variant = "clean"
        logger.info(
            "Demo mode enabled: task=beat_block_hammer, variant=clean, first 10 episodes"
        )
    else:
        if args.task is None:
            parser.error("Provide --task or use --demo")
        task_name = args.task
        variant = args.variant

    logger.info("=" * 60)
    logger.info(f"Step 1: Loading metadata for {task_name}/{variant}")
    logger.info("=" * 60)
    meta = load_task_metadata(task_name, variant=variant)

    if args.camera not in meta.camera_keys:
        logger.error(
            f"Camera {args.camera!r} not found for task. Available: {meta.camera_keys}"
        )
        sys.exit(1)

    all_episode_indices = [ep.episode_index for ep in meta.episodes]
    try:
        selected_episode_indices = select_episode_indices(
            all_episode_indices,
            n_episodes=args.n_episodes,
            episode_indices_arg=args.episode_indices,
            demo=args.demo,
        )
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if len(selected_episode_indices) < 2:
        logger.error(f"Need at least 2 episodes, got {len(selected_episode_indices)}")
        sys.exit(1)

    if args.subset_size > len(selected_episode_indices):
        logger.error(
            f"subset_size ({args.subset_size}) > number of selected episodes ({len(selected_episode_indices)})"
        )
        sys.exit(1)

    logger.info(
        f"Selected {len(selected_episode_indices)} episodes, camera={args.camera}, "
        f"target_fps={args.target_fps}, max_frames={args.max_frames}"
    )

    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Loading DINOv2 model and RoboTwin video")
    logger.info("=" * 60)
    model = load_model(args.model, device)
    transform = make_transform()
    video_path = download_video(task_name, args.camera, variant=variant)
    logger.info(f"Video cached at: {video_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Extracting episode embeddings")
    logger.info("=" * 60)

    episode_map = {ep.episode_index: ep for ep in meta.episodes}
    embeddings = []
    successful_episode_indices: list[int] = []

    for i, episode_index in enumerate(selected_episode_indices):
        ep = episode_map[episode_index]
        logger.info(
            f"\n--- Episode {episode_index} ({i + 1}/{len(selected_episode_indices)}) --- "
            f"len={ep.length}"
        )
        try:
            frames = extract_episode_frames(
                video_path,
                ep,
                args.camera,
                target_fps=args.target_fps,
                max_frames=args.max_frames,
            )
            if not frames:
                logger.warning("  No frames extracted, skipping")
                continue

            batch = pil_frames_to_batch(frames, transform)
            features = extract_features_batched(
                model, batch, device, batch_size=args.batch_size
            )
            cls_features = features["cls_features"]
            episode_embedding = cls_features.mean(dim=0)

            embeddings.append(episode_embedding)
            successful_episode_indices.append(episode_index)

            logger.info(
                f"  Frames={len(frames)}, cls_shape={list(cls_features.shape)}, "
                f"embedding_dim={episode_embedding.shape[0]}"
            )
        except Exception as exc:
            logger.warning(f"  Failed to process episode {episode_index}: {exc}")

    if len(embeddings) < 2:
        logger.error("Need at least 2 successful episode embeddings")
        sys.exit(1)

    all_features = torch.stack(embeddings)
    logger.info(f"\nAll embeddings: shape={list(all_features.shape)}")

    logger.info("\n" + "=" * 60)
    logger.info(
        f"Step 4: Analyzing {args.n_subsets} subsets "
        f"(size={args.subset_size}, strategy={args.strategy})"
    )
    logger.info("=" * 60)

    n_episodes = all_features.shape[0]
    actual_subset_size = min(args.subset_size, n_episodes)
    indices_list = generate_subset_indices(
        n_episodes,
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
        metrics["episode_indices"] = ",".join(
            str(successful_episode_indices[idx]) for idx in indices
        )
        rows.append(metrics)

        logger.info(
            f"  Subset {i}: size={metrics['n_samples']}, "
            f"cos_dist={metrics['cosine_distance']:.4f}, "
            f"l2_dist={metrics['l2_distance']:.4f}, "
            f"var={metrics['variance_score']:.6f}, "
            f"episodes=[{metrics['episode_indices']}]"
        )

    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Full-dataset diversity metrics")
    logger.info("=" * 60)
    full_metrics = compute_all_metrics(all_features)
    logger.info(f"  Full dataset ({len(successful_episode_indices)} episodes):")
    logger.info(f"    cosine_distance:  {full_metrics['cosine_distance']:.4f}")
    logger.info(f"    l2_distance:      {full_metrics['l2_distance']:.4f}")
    logger.info(f"    variance_score:   {full_metrics['variance_score']:.6f}")

    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Saving results")
    logger.info("=" * 60)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics.csv"
    fieldnames = [
        "subset_id",
        "strategy",
        "n_samples",
        "cosine_distance",
        "l2_distance",
        "variance_score",
        "episode_indices",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Metrics CSV saved to {csv_path}")

    episode_labels = [f"ep_{idx:03d}" for idx in successful_episode_indices]
    plot_metric_histograms(rows, output_dir)
    plot_metric_boxplots(rows, output_dir)
    plot_pca_scatter(
        all_features,
        episode_labels,
        output_dir / "pca_scatter.png",
        title="RoboTwin Episode Embeddings (PCA)",
    )

    sim_matrix = pairwise_cosine_matrix(all_features)
    plot_similarity_heatmap(
        sim_matrix,
        episode_labels,
        output_dir / "similarity_heatmap.png",
        title="RoboTwin Episode Pairwise Cosine Similarity",
    )

    torch.save(
        {
            "model": args.model,
            "task_name": task_name,
            "variant": variant,
            "camera": args.camera,
            "episode_indices": successful_episode_indices,
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
