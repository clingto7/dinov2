#!/usr/bin/env python3
"""
RoboTwin dataset loader for DINOv2 feature extraction.

Loads episodes from the RoboTwin-LeRobot-v3.0 dataset on HuggingFace,
decodes AV1 video frames using PyAV, and provides PIL Images ready for
DINOv2 preprocessing.

Architecture:
    - Uses huggingface_hub for downloading metadata + video files (cached)
    - Uses PyAV (libdav1d) for AV1 video decoding (OpenCV can't handle AV1)
    - Reads episode boundaries from LeRobot v3.0 parquet metadata
    - Returns frames as PIL.Image.Image (RGB) for DINOv2 transform pipeline

Dataset structure on HuggingFace (hxma/RoboTwin-LeRobot-v3.0):
    {task_name}/
        aloha-agilex_clean_50/        # 50 clean demos (fixed environment)
            meta/info.json             # dataset metadata (fps, features, shapes)
            meta/episodes/chunk-000/file-000.parquet  # episode boundaries
            meta/tasks.parquet         # task descriptions
            data/chunk-000/file-000.parquet            # actions/states per timestep
            videos/observation.images.cam_high/chunk-000/file-000.mp4
            videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4
            videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4
        aloha-agilex_randomized_500/  # 500 domain-randomized demos

Camera keys:
    - observation.images.cam_high       (overhead camera, 640x480)
    - observation.images.cam_left_wrist (left wrist camera, 640x480)
    - observation.images.cam_right_wrist (right wrist camera, 640x480)

All videos are AV1-encoded MP4 at 30 FPS.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import av
import pyarrow.parquet as pq
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("robotwin_loader")

# Bypass SOCKS proxy for HuggingFace downloads
os.environ.setdefault("ALL_PROXY", "")
os.environ.setdefault("all_proxy", "")

# ============================================================
# Constants
# ============================================================

REPO_ID = "hxma/RoboTwin-LeRobot-v3.0"

CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

DATA_VARIANTS = {
    "clean": "aloha-agilex_clean_50",
    "randomized": "aloha-agilex_randomized_500",
}

ALL_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


# ============================================================
# Data structures
# ============================================================


@dataclass
class EpisodeInfo:
    """Metadata for a single episode within a RoboTwin task."""

    episode_index: int
    dataset_from_index: int
    dataset_to_index: int
    length: int  # number of frames
    # Per-camera video timestamps (seconds)
    video_timestamps: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class TaskDatasetInfo:
    """Metadata for a RoboTwin task dataset (one task + one variant)."""

    task_name: str
    variant: str  # "clean" or "randomized"
    repo_id: str
    total_episodes: int
    total_frames: int
    fps: float
    video_resolution: tuple[int, int]  # (height, width)
    camera_keys: list[str]
    episodes: list[EpisodeInfo]


# ============================================================
# Core loader functions
# ============================================================


def _hf_download(repo_id: str, filename: str) -> str:
    """Download a file from HuggingFace Hub (cached).

    Returns:
        Local path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, repo_type="dataset")


def load_task_metadata(
    task_name: str,
    variant: str = "clean",
    repo_id: str = REPO_ID,
) -> TaskDatasetInfo:
    """Load metadata for a RoboTwin task (info.json + episode parquet).

    Args:
        task_name: Task name (e.g., "beat_block_hammer").
        variant: "clean" for clean_50, "randomized" for randomized_500.
        repo_id: HuggingFace dataset repo ID.

    Returns:
        TaskDatasetInfo with episode boundaries and video metadata.

    Raises:
        ValueError: If task_name or variant is invalid.
    """
    if task_name not in ALL_TASKS:
        raise ValueError(f"Unknown task: {task_name!r}. Available: {ALL_TASKS}")
    if variant not in DATA_VARIANTS:
        raise ValueError(f"Unknown variant: {variant!r}. Use 'clean' or 'randomized'.")

    variant_dir = DATA_VARIANTS[variant]
    base_path = f"{task_name}/{variant_dir}"

    # Download and parse info.json
    info_path = _hf_download(repo_id, f"{base_path}/meta/info.json")
    with open(info_path) as f:
        info = json.load(f)

    # Extract camera keys from features
    camera_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    # Get video resolution from first camera
    first_cam = info["features"][camera_keys[0]]
    video_h = first_cam["shape"][0]
    video_w = first_cam["shape"][1]

    # Download and parse episode metadata
    episodes_path = _hf_download(
        repo_id, f"{base_path}/meta/episodes/chunk-000/file-000.parquet"
    )
    table = pq.read_table(episodes_path)

    episodes = []
    for i in range(table.num_rows):
        video_timestamps = {}
        for cam_key in camera_keys:
            # Column names use / separator in parquet
            ts_from_col = f"videos/{cam_key}/from_timestamp"
            ts_to_col = f"videos/{cam_key}/to_timestamp"
            ts_from = table.column(ts_from_col)[i].as_py()
            ts_to = table.column(ts_to_col)[i].as_py()
            video_timestamps[cam_key] = (ts_from, ts_to)

        ep = EpisodeInfo(
            episode_index=table.column("episode_index")[i].as_py(),
            dataset_from_index=table.column("dataset_from_index")[i].as_py(),
            dataset_to_index=table.column("dataset_to_index")[i].as_py(),
            length=table.column("length")[i].as_py(),
            video_timestamps=video_timestamps,
        )
        episodes.append(ep)

    return TaskDatasetInfo(
        task_name=task_name,
        variant=variant,
        repo_id=repo_id,
        total_episodes=info["total_episodes"],
        total_frames=info["total_frames"],
        fps=info["fps"],
        video_resolution=(video_h, video_w),
        camera_keys=camera_keys,
        episodes=episodes,
    )


def download_video(
    task_name: str,
    camera_key: str,
    variant: str = "clean",
    repo_id: str = REPO_ID,
    chunk_index: int = 0,
    file_index: int = 0,
) -> str:
    """Download a video file from HuggingFace Hub (cached).

    Args:
        task_name: Task name (e.g., "beat_block_hammer").
        camera_key: Camera key (e.g., "observation.images.cam_high").
        variant: "clean" or "randomized".
        repo_id: HuggingFace dataset repo ID.
        chunk_index: Video chunk index (usually 0).
        file_index: Video file index within chunk (usually 0).

    Returns:
        Local path to the downloaded video file.
    """
    variant_dir = DATA_VARIANTS[variant]
    video_path = (
        f"{task_name}/{variant_dir}/videos/{camera_key}/"
        f"chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    logger.info(f"Downloading video: {video_path}")
    return _hf_download(repo_id, video_path)


def extract_episode_frames(
    video_path: str,
    episode: EpisodeInfo,
    camera_key: str,
    target_fps: float | None = None,
    max_frames: int | None = None,
) -> list[Image.Image]:
    """Extract frames for a specific episode from a multi-episode video file.

    Uses PyAV (libdav1d) for AV1 decoding with timestamp-based seeking.

    Args:
        video_path: Local path to the video file (all episodes concatenated).
        episode: EpisodeInfo with video timestamps for the target episode.
        camera_key: Camera key to look up timestamps.
        target_fps: If set, subsample frames to this FPS (default: use all frames).
        max_frames: Maximum number of frames to extract per episode.

    Returns:
        List of PIL.Image.Image in RGB format.
    """
    ts_start, ts_end = episode.video_timestamps[camera_key]

    container = av.open(video_path)
    stream = container.streams.video[0]
    avg_rate = stream.average_rate
    time_base = stream.time_base
    video_fps = float(avg_rate) if avg_rate is not None else 30.0

    # Seek to episode start
    # PyAV seek uses stream time_base units
    if time_base is not None:
        target_pts = int(ts_start / time_base)
    else:
        target_pts = int(ts_start * video_fps)
    container.seek(target_pts, stream=stream)

    frames: list[Image.Image] = []

    # Determine frame sampling interval
    if target_fps is not None and target_fps < video_fps:
        frame_interval = video_fps / target_fps
    else:
        frame_interval = 1.0

    frame_count = 0
    next_sample_at = 0.0

    for frame in container.decode(video=0):
        if frame.pts is None or time_base is None:
            continue
        frame_time = float(frame.pts * time_base)

        # Skip frames before episode start (seek may land before target)
        if frame_time < ts_start - 0.001:
            continue

        # Stop at episode end
        if frame_time >= ts_end + 0.001:
            break

        # Subsample based on target FPS
        if frame_count >= next_sample_at:
            arr = frame.to_ndarray(format="rgb24")  # [H, W, 3] uint8
            pil_img = Image.fromarray(arr)
            frames.append(pil_img)
            next_sample_at += frame_interval

            if max_frames is not None and len(frames) >= max_frames:
                break

        frame_count += 1

    container.close()
    return frames


def extract_all_episodes_frames(
    task_name: str,
    camera_key: str = "observation.images.cam_high",
    variant: str = "clean",
    target_fps: float | None = 2.0,
    max_frames_per_episode: int | None = 10,
    episode_indices: list[int] | None = None,
    repo_id: str = REPO_ID,
) -> tuple[TaskDatasetInfo, dict[int, list[Image.Image]]]:
    """Extract frames from multiple episodes of a RoboTwin task.

    This is the main entry point for batch episode processing.

    Args:
        task_name: Task name (e.g., "beat_block_hammer").
        camera_key: Camera to extract from.
        variant: "clean" or "randomized".
        target_fps: Subsample frames to this FPS (None = all frames).
        max_frames_per_episode: Max frames per episode (None = unlimited).
        episode_indices: Specific episodes to extract (None = all).
        repo_id: HuggingFace dataset repo ID.

    Returns:
        Tuple of (TaskDatasetInfo, dict mapping episode_index -> list of PIL Images).
    """
    # Load metadata
    logger.info(f"Loading metadata for {task_name}/{variant}...")
    meta = load_task_metadata(task_name, variant, repo_id)
    logger.info(
        f"  {meta.total_episodes} episodes, {meta.total_frames} total frames, "
        f"{meta.fps} FPS, resolution={meta.video_resolution}"
    )

    # Download video file
    video_path = download_video(task_name, camera_key, variant, repo_id)
    logger.info(f"  Video cached at: {video_path}")

    # Determine which episodes to process
    if episode_indices is not None:
        episodes_to_process = [
            ep for ep in meta.episodes if ep.episode_index in episode_indices
        ]
        if len(episodes_to_process) != len(episode_indices):
            found = {ep.episode_index for ep in episodes_to_process}
            missing = set(episode_indices) - found
            logger.warning(f"  Episodes not found: {missing}")
    else:
        episodes_to_process = meta.episodes

    # Extract frames for each episode
    results: dict[int, list[Image.Image]] = {}
    for i, ep in enumerate(episodes_to_process):
        logger.info(
            f"  Episode {ep.episode_index} ({i + 1}/{len(episodes_to_process)}): "
            f"length={ep.length} frames, "
            f"ts=[{ep.video_timestamps[camera_key][0]:.3f}s - "
            f"{ep.video_timestamps[camera_key][1]:.3f}s]"
        )
        frames = extract_episode_frames(
            video_path,
            ep,
            camera_key,
            target_fps=target_fps,
            max_frames=max_frames_per_episode,
        )
        results[ep.episode_index] = frames
        logger.info(f"    Extracted {len(frames)} frames")

    return meta, results


def list_available_tasks(repo_id: str = REPO_ID) -> list[str]:
    """List all available task names in the RoboTwin dataset.

    Returns:
        Sorted list of task names.
    """
    return list(ALL_TASKS)


# ============================================================
# CLI for quick inspection
# ============================================================


def main() -> None:
    """CLI for inspecting RoboTwin dataset structure."""
    import argparse

    parser = argparse.ArgumentParser(description="RoboTwin dataset loader / inspector")
    parser.add_argument("--list-tasks", action="store_true", help="List all tasks")
    parser.add_argument("--task", type=str, help="Task name to inspect")
    parser.add_argument(
        "--variant",
        default="clean",
        choices=["clean", "randomized"],
        help="Data variant",
    )
    parser.add_argument(
        "--camera",
        default="observation.images.cam_high",
        choices=CAMERA_KEYS,
        help="Camera to use",
    )
    parser.add_argument(
        "--extract-episode",
        type=int,
        default=None,
        help="Extract frames from a specific episode (saves to --output-dir)",
    )
    parser.add_argument("--target-fps", type=float, default=2.0, help="Target FPS")
    parser.add_argument(
        "--max-frames", type=int, default=10, help="Max frames per episode"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for extracted frames",
    )
    args = parser.parse_args()

    if args.list_tasks:
        print(f"RoboTwin tasks ({len(ALL_TASKS)}):")
        for i, task in enumerate(ALL_TASKS):
            print(f"  {i + 1:2d}. {task}")
        return

    if args.task:
        meta = load_task_metadata(args.task, args.variant)
        print(f"\nTask: {meta.task_name} ({meta.variant})")
        print(f"  Episodes: {meta.total_episodes}")
        print(f"  Total frames: {meta.total_frames}")
        print(f"  FPS: {meta.fps}")
        print(f"  Resolution: {meta.video_resolution[1]}x{meta.video_resolution[0]}")
        print(f"  Cameras: {meta.camera_keys}")
        print(f"\n  Episode summary:")
        for ep in meta.episodes[:5]:
            ts = ep.video_timestamps[args.camera]
            print(
                f"    Episode {ep.episode_index:3d}: "
                f"length={ep.length:4d}, "
                f"ts=[{ts[0]:7.3f}s - {ts[1]:7.3f}s]"
            )
        if len(meta.episodes) > 5:
            print(f"    ... ({len(meta.episodes) - 5} more)")

        if args.extract_episode is not None:
            ep_idx = args.extract_episode
            output_dir = args.output_dir or Path(
                f"extract_feature/outputs/robotwin_{args.task}_ep{ep_idx}"
            )

            _, frames_dict = extract_all_episodes_frames(
                args.task,
                camera_key=args.camera,
                variant=args.variant,
                target_fps=args.target_fps,
                max_frames_per_episode=args.max_frames,
                episode_indices=[ep_idx],
            )

            if ep_idx in frames_dict:
                output_dir.mkdir(parents=True, exist_ok=True)
                frames = frames_dict[ep_idx]
                for j, img in enumerate(frames):
                    img_path = output_dir / f"ep{ep_idx:03d}_frame{j:04d}.png"
                    img.save(img_path)
                print(f"\n  Saved {len(frames)} frames to {output_dir}/")
            else:
                print(f"\n  Episode {ep_idx} not found.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
