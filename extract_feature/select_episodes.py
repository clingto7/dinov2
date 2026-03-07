#!/usr/bin/env python3
"""
Diversity-based episode selection for RoboTwin datasets.

Given precomputed episode embeddings (from analyze_robotwin_diversity.py),
selects a subset of episodes that maximizes diversity using various strategies.

Can also run standalone: load a RoboTwin task, extract embeddings, then select.

Strategies:
    - greedy_maxdist: Greedily pick the episode most distant from the current set.
    - greedy_maxvar: Greedily pick the episode that maximizes feature variance.
    - kmeans: Use K-Means clustering, pick one episode per cluster (closest to centroid).
    - random: Random baseline for comparison.

Usage:
    # From precomputed features (output of analyze_robotwin_diversity.py)
    python extract_feature/select_episodes.py \
        --features extract_feature/outputs/robotwin_diversity_demo/features_and_metrics.pt \
        --n-select 5 --strategy greedy_maxdist \
        --output-dir extract_feature/outputs/robotwin_selection

    # Standalone: load task, extract, then select
    python extract_feature/select_episodes.py \
        --task beat_block_hammer --variant clean --n-episodes 20 \
        --n-select 5 --strategy greedy_maxdist \
        --output-dir extract_feature/outputs/robotwin_selection

    # Demo mode
    python extract_feature/select_episodes.py --demo \
        --n-select 5 --strategy greedy_maxdist \
        --output-dir extract_feature/outputs/robotwin_selection_demo
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from extract_video_features import (
    AVAILABLE_MODELS,
    load_model,
    make_transform,
    extract_features_batched,
)
from diversity_metrics import compute_all_metrics, pairwise_cosine_matrix
from plotting_utils import (
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
logger = logging.getLogger("select_episodes")


# ============================================================
# Selection strategies
# ============================================================


def select_greedy_maxdist(
    embeddings: torch.Tensor, n_select: int, seed: int = 42
) -> list[int]:
    """Greedily select episodes that maximize minimum pairwise cosine distance.

    Starts from a random seed episode, then iteratively picks the episode
    whose minimum cosine distance to all already-selected episodes is largest.

    Args:
        embeddings: Episode embeddings [N, D].
        n_select: Number of episodes to select.
        seed: Random seed for initial episode choice.

    Returns:
        List of selected indices (into embeddings tensor).
    """
    n = embeddings.shape[0]
    if n_select >= n:
        return list(range(n))

    # Precompute pairwise cosine distance matrix (1 - similarity)
    normed = F.normalize(embeddings, p=2, dim=-1)
    sim = normed @ normed.T  # [N, N]
    dist = 1.0 - sim  # [N, N], cosine distance

    # Start with a deterministic seed episode
    rng = torch.Generator()
    rng.manual_seed(seed)
    first = torch.randint(n, (1,), generator=rng).item()

    selected = [first]
    remaining = set(range(n)) - {first}

    for _ in range(n_select - 1):
        # For each remaining candidate, compute min distance to selected set
        best_idx = -1
        best_min_dist = -1.0
        for cand in remaining:
            min_dist = min(dist[cand, s].item() for s in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = cand
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def select_greedy_maxvar(
    embeddings: torch.Tensor, n_select: int, seed: int = 42
) -> list[int]:
    """Greedily select episodes that maximize feature variance of the selected set.

    Starts from the two most distant episodes, then iteratively adds the episode
    that maximizes the variance of the selected subset.

    Args:
        embeddings: Episode embeddings [N, D].
        n_select: Number of episodes to select.
        seed: Random seed (unused here, kept for interface consistency).

    Returns:
        List of selected indices.
    """
    n = embeddings.shape[0]
    if n_select >= n:
        return list(range(n))

    # Start with the two most distant episodes
    normed = F.normalize(embeddings, p=2, dim=-1)
    sim = normed @ normed.T
    # Mask diagonal so self-similarity doesn't interfere
    sim.fill_diagonal_(2.0)
    dist = 1.0 - sim
    # Find the pair with maximum distance
    flat_idx = dist.argmax().item()
    i, j = flat_idx // n, flat_idx % n

    selected = [i, j]
    remaining = set(range(n)) - {i, j}

    while len(selected) < n_select and remaining:
        best_idx = -1
        best_var = -1.0
        for cand in remaining:
            trial = selected + [cand]
            trial_emb = embeddings[trial]  # [k+1, D]
            var_score = trial_emb.var(dim=0).mean().item()
            if var_score > best_var:
                best_var = var_score
                best_idx = cand
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def select_kmeans(
    embeddings: torch.Tensor, n_select: int, seed: int = 42, n_iter: int = 100
) -> list[int]:
    """Select episodes using K-Means clustering.

    Runs K-Means with k=n_select clusters on L2-normalized embeddings,
    then picks the episode closest to each cluster centroid.

    Args:
        embeddings: Episode embeddings [N, D].
        n_select: Number of episodes to select (= number of clusters).
        seed: Random seed for centroid initialization.
        n_iter: Number of K-Means iterations.

    Returns:
        List of selected indices.
    """
    n, d = embeddings.shape
    if n_select >= n:
        return list(range(n))

    normed = F.normalize(embeddings, p=2, dim=-1)  # [N, D]

    # Initialize centroids randomly
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    centroids = normed[perm[:n_select]].clone()  # [K, D]

    for _ in range(n_iter):
        # Assign each point to nearest centroid (cosine similarity)
        sim = normed @ centroids.T  # [N, K]
        assignments = sim.argmax(dim=1)  # [N]

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_select):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = normed[mask].mean(dim=0)
            else:
                # Empty cluster: reinitialize to random point
                rand_idx = torch.randint(n, (1,), generator=rng).item()
                new_centroids[k] = normed[rand_idx]
        centroids = F.normalize(new_centroids, p=2, dim=-1)

    # Pick the point closest to each centroid
    sim = normed @ centroids.T  # [N, K]
    selected = []
    used = set()
    for k in range(n_select):
        scores = sim[:, k].clone()
        # Mask out already-selected
        for idx in used:
            scores[idx] = -2.0
        best = scores.argmax().item()
        selected.append(best)
        used.add(best)

    return selected


def select_random(embeddings: torch.Tensor, n_select: int, seed: int = 42) -> list[int]:
    """Random episode selection (baseline).

    Args:
        embeddings: Episode embeddings [N, D].
        n_select: Number of episodes to select.
        seed: Random seed.

    Returns:
        List of selected indices.
    """
    n = embeddings.shape[0]
    if n_select >= n:
        return list(range(n))
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    return sorted(perm[:n_select].tolist())


STRATEGIES = {
    "greedy_maxdist": select_greedy_maxdist,
    "greedy_maxvar": select_greedy_maxvar,
    "kmeans": select_kmeans,
    "random": select_random,
}


# ============================================================
# Embedding extraction (standalone mode)
# ============================================================


def pil_frames_to_batch(frames: list[Image.Image], transform) -> torch.Tensor:
    """Convert PIL frames to a transformed batch tensor."""
    return torch.stack([transform(img) for img in frames])


def extract_episode_embeddings(
    task_name: str,
    variant: str,
    camera_key: str,
    episode_indices: list[int],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    target_fps: float = 2.0,
    max_frames: int = 10,
    batch_size: int = 16,
) -> tuple[torch.Tensor, list[int]]:
    """Extract DINOv2 embeddings for specified episodes.

    Returns:
        embeddings: [N_success, D] tensor.
        successful_indices: list of episode indices that were successfully processed.
    """
    meta = load_task_metadata(task_name, variant=variant)
    video_path = download_video(task_name, camera_key, variant=variant)
    episode_map = {ep.episode_index: ep for ep in meta.episodes}

    embeddings = []
    successful_indices = []

    for i, ep_idx in enumerate(episode_indices):
        ep = episode_map.get(ep_idx)
        if ep is None:
            logger.warning(f"Episode {ep_idx} not found, skipping")
            continue

        logger.info(
            f"  Episode {ep_idx} ({i + 1}/{len(episode_indices)}): len={ep.length}"
        )
        try:
            frames = extract_episode_frames(
                video_path, ep, camera_key, target_fps=target_fps, max_frames=max_frames
            )
            if not frames:
                logger.warning(f"  No frames extracted for episode {ep_idx}")
                continue

            batch = pil_frames_to_batch(frames, transform)
            features = extract_features_batched(
                model, batch, device, batch_size=batch_size
            )
            cls_features = features["cls_features"]  # [N_frames, D]
            episode_embedding = cls_features.mean(dim=0)  # [D]

            embeddings.append(episode_embedding)
            successful_indices.append(ep_idx)
            logger.info(
                f"    {len(frames)} frames → embedding dim={episode_embedding.shape[0]}"
            )
        except Exception as exc:
            logger.warning(f"  Failed: {exc}")

    if not embeddings:
        raise RuntimeError("No episodes were successfully processed")

    return torch.stack(embeddings), successful_indices


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diversity-based episode selection for RoboTwin"
    )

    # Input source (mutually exclusive)
    input_group = parser.add_argument_group("input source")
    input_group.add_argument(
        "--features",
        type=Path,
        help="Precomputed features .pt file (from analyze_robotwin_diversity.py)",
    )
    input_group.add_argument(
        "--task",
        type=str,
        choices=ALL_TASKS,
        help="RoboTwin task name (standalone mode)",
    )
    input_group.add_argument(
        "--demo",
        action="store_true",
        help="Demo mode: beat_block_hammer/clean first 20 eps",
    )

    # Task-specific args (standalone mode)
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
        help="Number of episodes to load (standalone)",
    )
    parser.add_argument(
        "--model",
        default="dinov2_vits14",
        choices=AVAILABLE_MODELS,
        help="DINOv2 model",
    )
    parser.add_argument(
        "--target-fps", type=float, default=2.0, help="Target FPS for frame sampling"
    )
    parser.add_argument(
        "--max-frames", type=int, default=10, help="Max frames per episode"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="GPU batch size")

    # Selection args
    parser.add_argument(
        "--n-select", type=int, required=True, help="Number of episodes to select"
    )
    parser.add_argument(
        "--strategy",
        default="greedy_maxdist",
        choices=list(STRATEGIES.keys()),
        help="Selection strategy",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run all strategies and compare results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── 1. Load or compute embeddings ──
    if args.features:
        logger.info(f"Loading precomputed features from {args.features}")
        data = torch.load(args.features, map_location="cpu", weights_only=False)
        embeddings = data["embeddings"]
        episode_indices = data["episode_indices"]
        task_name = data.get("task_name", "unknown")
        variant = data.get("variant", "unknown")
        logger.info(
            f"  Task: {task_name}/{variant}, {len(episode_indices)} episodes, dim={embeddings.shape[1]}"
        )
    elif args.demo or args.task:
        if args.demo:
            task_name = "beat_block_hammer"
            variant = "clean"
            logger.info("Demo mode: beat_block_hammer/clean, first 20 episodes")
        else:
            task_name = args.task
            variant = args.variant

        logger.info(f"Standalone mode: loading {task_name}/{variant}")
        model = load_model(args.model, device)
        transform = make_transform()

        # Determine episode indices
        meta = load_task_metadata(task_name, variant=variant)
        all_ep_indices = [ep.episode_index for ep in meta.episodes]
        if args.demo:
            all_ep_indices = all_ep_indices[:20]
        elif args.n_episodes is not None:
            all_ep_indices = all_ep_indices[: args.n_episodes]

        logger.info(f"Extracting embeddings for {len(all_ep_indices)} episodes...")
        embeddings, episode_indices = extract_episode_embeddings(
            task_name,
            variant,
            args.camera,
            all_ep_indices,
            model,
            transform,
            device,
            target_fps=args.target_fps,
            max_frames=args.max_frames,
            batch_size=args.batch_size,
        )
        logger.info(f"Embeddings: shape={list(embeddings.shape)}")
    else:
        parser.error("Provide --features, --task, or --demo")
        return  # unreachable but helps type checker

    n_episodes = embeddings.shape[0]
    if args.n_select > n_episodes:
        logger.error(f"n_select ({args.n_select}) > available episodes ({n_episodes})")
        sys.exit(1)
    if args.n_select < 1:
        logger.error("n_select must be >= 1")
        sys.exit(1)

    # ── 2. Run selection ──
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies_to_run = list(STRATEGIES.keys()) if args.compare_all else [args.strategy]

    all_results = []
    for strategy_name in strategies_to_run:
        logger.info("\n" + "=" * 60)
        logger.info(
            f"Strategy: {strategy_name} (selecting {args.n_select} from {n_episodes})"
        )
        logger.info("=" * 60)

        strategy_fn = STRATEGIES[strategy_name]
        selected_local = strategy_fn(embeddings, args.n_select, seed=args.seed)
        selected_episode_ids = [episode_indices[i] for i in selected_local]

        # Compute metrics for selected subset
        selected_emb = embeddings[selected_local]
        selected_metrics = compute_all_metrics(selected_emb)

        # Compute full-dataset metrics for comparison
        full_metrics = compute_all_metrics(embeddings)

        result = {
            "strategy": strategy_name,
            "n_select": args.n_select,
            "n_total": n_episodes,
            "selected_local_indices": selected_local,
            "selected_episode_ids": selected_episode_ids,
            "selected_metrics": selected_metrics,
            "full_metrics": full_metrics,
        }
        all_results.append(result)

        logger.info(f"  Selected episodes: {selected_episode_ids}")
        logger.info(f"  Selected subset metrics:")
        logger.info(f"    cosine_distance:  {selected_metrics['cosine_distance']:.4f}")
        logger.info(f"    l2_distance:      {selected_metrics['l2_distance']:.4f}")
        logger.info(f"    variance_score:   {selected_metrics['variance_score']:.6f}")
        logger.info(f"  Full dataset metrics (for comparison):")
        logger.info(f"    cosine_distance:  {full_metrics['cosine_distance']:.4f}")
        logger.info(f"    l2_distance:      {full_metrics['l2_distance']:.4f}")
        logger.info(f"    variance_score:   {full_metrics['variance_score']:.6f}")

        # Diversity ratio (selected / full)
        if full_metrics["cosine_distance"] > 0:
            ratio = (
                selected_metrics["cosine_distance"] / full_metrics["cosine_distance"]
            )
            logger.info(f"  Diversity ratio (cos_dist selected/full): {ratio:.4f}")

    # ── 3. Save results ──
    logger.info("\n" + "=" * 60)
    logger.info("Saving results")
    logger.info("=" * 60)

    # CSV summary
    csv_path = output_dir / "selection_results.csv"
    fieldnames = [
        "strategy",
        "n_select",
        "n_total",
        "cosine_distance",
        "l2_distance",
        "variance_score",
        "full_cosine_distance",
        "full_l2_distance",
        "full_variance_score",
        "diversity_ratio_cos",
        "selected_episodes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            sm = result["selected_metrics"]
            fm = result["full_metrics"]
            ratio_cos = (
                sm["cosine_distance"] / fm["cosine_distance"]
                if fm["cosine_distance"] > 0
                else 0.0
            )
            writer.writerow(
                {
                    "strategy": result["strategy"],
                    "n_select": result["n_select"],
                    "n_total": result["n_total"],
                    "cosine_distance": f"{sm['cosine_distance']:.6f}",
                    "l2_distance": f"{sm['l2_distance']:.6f}",
                    "variance_score": f"{sm['variance_score']:.8f}",
                    "full_cosine_distance": f"{fm['cosine_distance']:.6f}",
                    "full_l2_distance": f"{fm['l2_distance']:.6f}",
                    "full_variance_score": f"{fm['variance_score']:.8f}",
                    "diversity_ratio_cos": f"{ratio_cos:.4f}",
                    "selected_episodes": ",".join(
                        str(x) for x in result["selected_episode_ids"]
                    ),
                }
            )
    logger.info(f"CSV saved to {csv_path}")

    # JSON with selected episode lists (machine-readable)
    json_path = output_dir / "selected_episodes.json"
    json_data = {}
    for result in all_results:
        json_data[result["strategy"]] = {
            "selected_episode_ids": result["selected_episode_ids"],
            "n_select": result["n_select"],
            "n_total": result["n_total"],
            "metrics": {
                "cosine_distance": result["selected_metrics"]["cosine_distance"],
                "l2_distance": result["selected_metrics"]["l2_distance"],
                "variance_score": result["selected_metrics"]["variance_score"],
            },
        }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"JSON saved to {json_path}")

    # Plots: PCA scatter with selected episodes highlighted, and similarity heatmap
    # Use the primary strategy result for plotting
    primary = all_results[0]
    selected_set = set(primary["selected_local_indices"])
    episode_labels = [
        f"ep_{episode_indices[i]:03d}{'*' if i in selected_set else ''}"
        for i in range(n_episodes)
    ]

    plot_pca_scatter(
        embeddings,
        episode_labels,
        output_dir / "pca_scatter.png",
        title=f"Episode Embeddings — {primary['strategy']} selected (*)",
    )

    sim_matrix = pairwise_cosine_matrix(embeddings)
    plot_similarity_heatmap(
        sim_matrix,
        episode_labels,
        output_dir / "similarity_heatmap.png",
        title="Episode Pairwise Cosine Similarity",
    )

    # Save full data
    torch.save(
        {
            "task_name": task_name,
            "variant": variant,
            "embeddings": embeddings,
            "episode_indices": episode_indices,
            "selection_results": all_results,
            "similarity_matrix": sim_matrix,
        },
        output_dir / "selection_data.pt",
    )
    logger.info(f"Full data saved to {output_dir / 'selection_data.pt'}")

    # ── 4. Comparison summary ──
    if len(all_results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("Strategy Comparison Summary")
        logger.info("=" * 60)
        logger.info(
            f"  {'Strategy':<18} {'Cos Dist':>10} {'L2 Dist':>10} {'Variance':>12} {'Ratio':>8}"
        )
        logger.info(f"  {'-' * 18} {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 8}")
        for result in all_results:
            sm = result["selected_metrics"]
            fm = result["full_metrics"]
            ratio = (
                sm["cosine_distance"] / fm["cosine_distance"]
                if fm["cosine_distance"] > 0
                else 0.0
            )
            logger.info(
                f"  {result['strategy']:<18} "
                f"{sm['cosine_distance']:>10.4f} "
                f"{sm['l2_distance']:>10.4f} "
                f"{sm['variance_score']:>12.6f} "
                f"{ratio:>8.4f}"
            )
        fm = all_results[0]["full_metrics"]
        logger.info(
            f"  {'(full dataset)':<18} "
            f"{fm['cosine_distance']:>10.4f} "
            f"{fm['l2_distance']:>10.4f} "
            f"{fm['variance_score']:>12.6f} "
            f"{'1.0000':>8}"
        )

    logger.info(f"\nAll results saved to {output_dir}/")
    logger.info("Done.")


if __name__ == "__main__":
    main()
