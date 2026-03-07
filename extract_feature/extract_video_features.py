#!/usr/bin/env python3
"""
Extract DINOv2 features from video frames.

Usage:
    # Extract from a video file
    python scripts/extract_video_features.py --video /path/to/video.mp4

    # Demo mode: download a sample video
    python scripts/extract_video_features.py --demo

    # Control frame sampling
    python scripts/extract_video_features.py --video vid.mp4 --fps 2  # sample at 2 FPS
    python scripts/extract_video_features.py --video vid.mp4 --max-frames 50

    # Save features
    python scripts/extract_video_features.py --demo --output video_features.pt

    # Extract intermediate layers
    python scripts/extract_video_features.py --demo --intermediate-layers 4
"""

import argparse
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

try:
    import av

    _HAS_PYAV = True
except ImportError:
    _HAS_PYAV = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("extract_video_features")

os.environ["ALL_PROXY"] = ""
os.environ["all_proxy"] = ""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINOV2_REPO = str(Path(__file__).resolve().parent.parent)

AVAILABLE_MODELS = [
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
    "dinov2_vits14_reg",
    "dinov2_vitb14_reg",
    "dinov2_vitl14_reg",
    "dinov2_vitg14_reg",
]

DEMO_VIDEOS = {
    "kinetics_archery": {
        "url": "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/9f4ed38128a355c352527209101be3e326471816/train/archery/-1q7jA3DXQM_000005_000015.mp4",
        "filename": "kinetics_archery.mp4",
        "source": "Kinetics-mini (HuggingFace)",
    },
    "ucf101_baby_crawling": {
        "url": "https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/b9984b8d2a95e4a1879e1b071e9433858d0bc24a/v_BabyCrawling_g19_c02.avi",
        "filename": "ucf101_baby_crawling.avi",
        "source": "UCF101-subset (HuggingFace)",
    },
    "ucf101_basketball_dunk": {
        "url": "https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/b9984b8d2a95e4a1879e1b071e9433858d0bc24a/v_BasketballDunk_g14_c06.avi",
        "filename": "ucf101_basketball_dunk.avi",
        "source": "UCF101-subset (HuggingFace)",
    },
    "lerobot_xarm_bimanual": {
        "url": "https://huggingface.co/datasets/lerobot/utokyo_xarm_bimanual/resolve/13533b961150faf398ae9b82cf6d6323b56f9070/videos/observation.images.image/chunk-000/file-000.mp4",
        "filename": "lerobot_xarm_bimanual.mp4",
        "source": "LeRobot utokyo_xarm_bimanual (HuggingFace)",
    },
    "opencv_vtest": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/fe160f3eed3ec0344baff4bfb6a0771d01b5882d/samples/data/vtest.avi",
        "filename": "opencv_vtest.avi",
        "source": "OpenCV samples",
    },
    "bigbuckbunny": {
        "url": "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
        "filename": "BigBuckBunny_320x180.mp4",
        "source": "Blender Foundation (CC-BY)",
    },
}


def make_transform(resize_size: int = 256, crop_size: int = 224) -> transforms.Compose:
    """Create the canonical DINOv2 evaluation transform."""
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_model(model_name: str, device: torch.device) -> nn.Module:
    """Load a DINOv2 model from the local repo via torch.hub."""
    logger.info(f"Loading model: {model_name} from {DINOV2_REPO}")
    model = torch.hub.load(DINOV2_REPO, model_name, source="local")
    model = model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(
        f"Model loaded: {n_params:.1f}M parameters, embed_dim={model.embed_dim}"
    )
    return model


def download_demo_videos(names: list[str], output_dir: Path) -> list[tuple[str, Path]]:
    """Download demo videos for testing, skipping failed downloads."""
    import requests

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_videos: list[tuple[str, Path]] = []

    for idx, name in enumerate(names, start=1):
        demo = DEMO_VIDEOS[name]
        filepath = output_dir / demo["filename"]

        if filepath.exists():
            logger.info(f"[{idx}/{len(names)}] Demo video already exists: {filepath}")
            downloaded_videos.append((name, filepath))
            continue

        logger.info(
            f"[{idx}/{len(names)}] Downloading demo '{name}' from {demo['source']}: {demo['url']}"
        )
        try:
            resp = requests.get(demo["url"], timeout=120, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(
                            f"\r  [{idx}/{len(names)}] Downloading {name}: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r  [{idx}/{len(names)}] Downloading {name}: {downloaded / 1e6:.1f} MB",
                            end="",
                            flush=True,
                        )
            print()
            logger.info(f"Downloaded to {filepath}")
            downloaded_videos.append((name, filepath))
        except Exception as e:
            print()
            logger.warning(f"Failed to download demo '{name}': {e}. Skipping.")
            if filepath.exists():
                filepath.unlink(missing_ok=True)

    return downloaded_videos


def _sample_frames_opencv(
    video_path: Path,
    target_fps: float | None = None,
    max_frames: int | None = None,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> tuple[list[np.ndarray], float, dict, bool]:
    """Sample frames using OpenCV (cv2.VideoCapture).

    Returns BGR numpy arrays.  Returns (frames, actual_fps, info, is_rgb).
    is_rgb is always False for OpenCV (frames are BGR).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    info = {
        "video_fps": video_fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }
    logger.info(
        f"Video: {video_path.name} | {width}x{height} | {video_fps:.1f} FPS | {total_frames} frames | {duration:.1f}s"
    )

    # Determine frame indices to sample
    if end_sec is None:
        end_sec = duration
    start_frame = int(start_sec * video_fps)
    end_frame = min(int(end_sec * video_fps), total_frames)

    if target_fps is not None and target_fps < video_fps:
        step = video_fps / target_fps
        frame_indices = [
            int(start_frame + i * step)
            for i in range(int((end_frame - start_frame) / step))
        ]
    else:
        frame_indices = list(range(start_frame, end_frame))
        target_fps = video_fps

    if max_frames is not None and len(frame_indices) > max_frames:
        # Uniformly subsample
        step = len(frame_indices) / max_frames
        frame_indices = [frame_indices[int(i * step)] for i in range(max_frames)]

    logger.info(
        f"Sampling {len(frame_indices)} frames (effective FPS: {len(frame_indices) / (end_sec - start_sec):.1f})"
    )

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    actual_fps = len(frames) / (end_sec - start_sec) if (end_sec - start_sec) > 0 else 0
    logger.info(f"Extracted {len(frames)} frames (OpenCV)")
    return frames, actual_fps, info, False


def _sample_frames_pyav(
    video_path: Path,
    target_fps: float | None = None,
    max_frames: int | None = None,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> tuple[list[np.ndarray], float, dict, bool]:
    """Sample frames using PyAV (supports AV1 via libdav1d).

    Returns RGB numpy arrays.  Returns (frames, actual_fps, info, is_rgb).
    is_rgb is always True for PyAV.
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    avg_rate = stream.average_rate
    time_base = stream.time_base
    video_fps = float(avg_rate) if avg_rate is not None else 30.0
    total_frames = stream.frames if stream.frames > 0 else 0
    codec_name = stream.codec_context.name if stream.codec_context else "unknown"

    # Get dimensions
    width = stream.codec_context.width if stream.codec_context else 0
    height = stream.codec_context.height if stream.codec_context else 0

    # Calculate duration
    if stream.duration is not None and time_base is not None:
        duration = float(stream.duration * time_base)
    elif total_frames > 0 and video_fps > 0:
        duration = total_frames / video_fps
    else:
        duration = 0.0

    # If total_frames is unknown, estimate from duration
    if total_frames == 0 and duration > 0 and video_fps > 0:
        total_frames = int(duration * video_fps)

    info = {
        "video_fps": video_fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": duration,
        "codec": codec_name,
    }
    logger.info(
        f"Video: {video_path.name} | {width}x{height} | {video_fps:.1f} FPS | "
        f"{total_frames} frames | {duration:.1f}s | codec={codec_name} (PyAV)"
    )

    if end_sec is None:
        end_sec = duration

    # Calculate frame sampling interval
    if target_fps is not None and target_fps < video_fps:
        frame_interval = video_fps / target_fps
    else:
        frame_interval = 1.0

    # Calculate expected frame count for logging
    available_duration = end_sec - start_sec
    if available_duration <= 0:
        container.close()
        return [], 0.0, info, True

    expected_frames = int(available_duration * video_fps / frame_interval)
    if max_frames is not None:
        expected_frames = min(expected_frames, max_frames)
    logger.info(
        f"Sampling up to {expected_frames} frames (effective FPS: "
        f"{expected_frames / available_duration:.1f})"
    )

    # Seek to start_sec if needed
    if start_sec > 0 and time_base is not None:
        target_pts = int(start_sec / time_base)
        container.seek(target_pts, stream=stream)

    frames: list[np.ndarray] = []
    frame_count = 0
    next_sample_at = 0.0

    for frame in container.decode(video=0):
        if frame.pts is None or time_base is None:
            # Fallback: count frames sequentially
            frame_time = frame_count / video_fps
        else:
            frame_time = float(frame.pts * time_base)

        # Skip frames before start
        if frame_time < start_sec - 0.001:
            continue

        # Stop at end
        if frame_time >= end_sec + 0.001:
            break

        # Subsample based on target FPS
        if frame_count >= next_sample_at:
            arr = frame.to_ndarray(format="rgb24")  # [H, W, 3] uint8 RGB
            frames.append(arr)
            next_sample_at += frame_interval

            if max_frames is not None and len(frames) >= max_frames:
                break

        frame_count += 1

    container.close()

    actual_fps = len(frames) / available_duration if available_duration > 0 else 0.0
    logger.info(f"Extracted {len(frames)} frames (PyAV)")
    return frames, actual_fps, info, True


def sample_frames(
    video_path: Path,
    target_fps: float | None = None,
    max_frames: int | None = None,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> tuple[list[np.ndarray], float, dict, bool]:
    """Sample frames from a video file.

    Tries OpenCV first for speed. If OpenCV extracts 0 frames (e.g. AV1 codec
    not supported), automatically falls back to PyAV which supports AV1 via
    libdav1d.

    Args:
        video_path: Path to video file
        target_fps: Target sampling FPS (None = use all frames)
        max_frames: Maximum number of frames to extract
        start_sec: Start time in seconds
        end_sec: End time in seconds (None = end of video)

    Returns:
        frames: List of numpy arrays (BGR if OpenCV, RGB if PyAV)
        actual_fps: The effective sampling rate
        info: Video metadata dict
        is_rgb: True if frames are RGB (PyAV), False if BGR (OpenCV)
    """
    # Try OpenCV first
    try:
        frames, actual_fps, info, is_rgb = _sample_frames_opencv(
            video_path, target_fps, max_frames, start_sec, end_sec
        )
        if frames:
            return frames, actual_fps, info, is_rgb
        # OpenCV opened the file but extracted 0 frames — likely codec issue
        logger.warning(
            f"OpenCV extracted 0 frames from {video_path.name}. "
            "Falling back to PyAV (likely unsupported codec like AV1)."
        )
    except RuntimeError:
        logger.warning(f"OpenCV cannot open {video_path.name}. Falling back to PyAV.")

    # Fallback to PyAV
    if not _HAS_PYAV:
        raise RuntimeError(
            f"Cannot decode {video_path.name}: OpenCV failed and PyAV is not installed. "
            "Install with: uv add av"
        )

    return _sample_frames_pyav(video_path, target_fps, max_frames, start_sec, end_sec)


def frames_to_batch(
    frames: list[np.ndarray], transform: transforms.Compose, is_rgb: bool = False
) -> torch.Tensor:
    """Convert numpy frames to a transformed batch tensor.

    Args:
        frames: List of numpy arrays (BGR from OpenCV, or RGB from PyAV).
        transform: Torchvision transform pipeline.
        is_rgb: If True, frames are already RGB (PyAV). If False, BGR (OpenCV).
    """
    tensors = []
    for frame in frames:
        if is_rgb:
            pil_img = Image.fromarray(frame)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
        tensors.append(transform(pil_img))
    return torch.stack(tensors)


def extract_features_batched(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
    intermediate_layers: int = 0,
) -> dict[str, torch.Tensor]:
    """Extract features from frames in batches to manage GPU memory.

    Returns dict with:
        cls_features: [N, D]
        patch_tokens: [N, P, D]
        (optional) intermediate_cls: list of [N, D] per layer
        (optional) intermediate_patches: list of [N, P, D] per layer
    """
    n_frames = batch.shape[0]
    n_batches = math.ceil(n_frames / batch_size)

    all_cls = []
    all_patches = []
    all_inter_cls = (
        [[] for _ in range(intermediate_layers)] if intermediate_layers > 0 else []
    )
    all_inter_patches = (
        [[] for _ in range(intermediate_layers)] if intermediate_layers > 0 else []
    )

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_frames)
        sub_batch = batch[start:end].to(device)

        with torch.inference_mode():
            # Full features
            out = model.forward_features(sub_batch)
            all_cls.append(out["x_norm_clstoken"].cpu())
            all_patches.append(out["x_norm_patchtokens"].cpu())

            # Intermediate layers
            if intermediate_layers > 0:
                inter = model.get_intermediate_layers(
                    sub_batch,
                    n=intermediate_layers,
                    return_class_token=True,
                    reshape=False,
                    norm=True,
                )
                for layer_idx, (patches, cls) in enumerate(inter):
                    all_inter_cls[layer_idx].append(cls.cpu())
                    all_inter_patches[layer_idx].append(patches.cpu())

        if n_batches > 1:
            logger.info(f"  Batch {i + 1}/{n_batches} done ({end}/{n_frames} frames)")

    result: dict[str, torch.Tensor | list] = {
        "cls_features": torch.cat(all_cls, dim=0),
        "patch_tokens": torch.cat(all_patches, dim=0),
    }
    if intermediate_layers > 0:
        result["intermediate_cls"] = [torch.cat(x, dim=0) for x in all_inter_cls]
        result["intermediate_patches"] = [
            torch.cat(x, dim=0) for x in all_inter_patches
        ]

    return result


def print_feature_stats(name: str, tensor: torch.Tensor) -> None:
    """Print shape and basic statistics."""
    logger.info(
        f"  {name}: shape={list(tensor.shape)}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 features from video frames"
    )
    parser.add_argument("--model", default="dinov2_vits14", choices=AVAILABLE_MODELS)
    parser.add_argument("--video", type=Path, help="Path to video file")
    parser.add_argument(
        "--demo", action="store_true", help="Download and use a demo video"
    )
    parser.add_argument(
        "--demo-names",
        type=str,
        default=None,
        help="Comma-separated demo video names from DEMO_VIDEOS (default: all)",
    )
    parser.add_argument(
        "--demo-count",
        type=int,
        default=None,
        help="Max number of demo videos to process (default: all)",
    )
    parser.add_argument("--output", type=Path, help="Save features to .pt file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-video outputs in multi-demo mode",
    )
    parser.add_argument(
        "--fps", type=float, default=None, help="Target sampling FPS (default: 1)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=100, help="Max frames to extract"
    )
    parser.add_argument(
        "--start", type=float, default=0.0, help="Start time in seconds"
    )
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for GPU inference"
    )
    parser.add_argument(
        "--intermediate-layers",
        type=int,
        default=0,
        help="Number of intermediate layers",
    )
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--crop", type=int, default=224)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.fps is None:
        args.fps = 1.0  # default: sample 1 frame per second

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Get video path
    if args.demo:
        selected_demo_names = list(DEMO_VIDEOS.keys())
        if args.demo_names:
            requested = [x.strip() for x in args.demo_names.split(",") if x.strip()]
            if not requested:
                parser.error("--demo-names is empty after parsing")
            invalid = [name for name in requested if name not in DEMO_VIDEOS]
            if invalid:
                parser.error(
                    f"Unknown demo names: {invalid}. Available demos: {list(DEMO_VIDEOS.keys())}"
                )
            selected_demo_names = requested

        if args.demo_count is not None:
            if args.demo_count <= 0:
                parser.error("--demo-count must be > 0")
            selected_demo_names = selected_demo_names[: args.demo_count]

        if not selected_demo_names:
            parser.error("No demo videos selected")

        demo_dir = args.output_dir or Path(
            tempfile.mkdtemp(prefix="dinov2_video_demo_")
        )
        demo_videos = download_demo_videos(selected_demo_names, demo_dir)
        if not demo_videos:
            logger.error("All demo video downloads failed")
            sys.exit(1)

        if len(demo_videos) > 1:
            if args.output:
                logger.warning(
                    "--output is ignored in multi-demo mode; use --output-dir for per-video outputs"
                )

            model = load_model(args.model, device)
            transform = make_transform(args.resize, args.crop)

            demo_results: list[dict] = []
            for demo_name, video_path in demo_videos:
                logger.info("\n" + "#" * 80)
                logger.info(f"Processing demo video: {demo_name} ({video_path.name})")
                logger.info("#" * 80)

                frames, actual_fps, video_info, is_rgb = sample_frames(
                    video_path,
                    target_fps=args.fps,
                    max_frames=args.max_frames,
                    start_sec=args.start,
                    end_sec=args.end,
                )
                if not frames:
                    logger.warning(f"No frames extracted for '{demo_name}', skipping")
                    continue

                batch = frames_to_batch(frames, transform, is_rgb=is_rgb)
                logger.info(f"Frame batch shape: {batch.shape}")

                logger.info("=" * 60)
                logger.info("Extracting features...")
                logger.info("=" * 60)
                features = extract_features_batched(
                    model,
                    batch,
                    device,
                    batch_size=args.batch_size,
                    intermediate_layers=args.intermediate_layers,
                )

                cls_features = features["cls_features"]
                patch_tokens = features["patch_tokens"]

                logger.info("\n" + "=" * 60)
                logger.info("1. Per-frame Feature Summary")
                logger.info("=" * 60)
                print_feature_stats("CLS features (all frames)", cls_features)
                print_feature_stats("Patch tokens (all frames)", patch_tokens)

                logger.info("\n" + "=" * 60)
                logger.info("2. Temporal Statistics (across frames)")
                logger.info("=" * 60)

                temporal_mean = cls_features.mean(dim=0)
                temporal_var = cls_features.var(dim=0)
                logger.info(f"  CLS temporal mean (across {len(frames)} frames):")
                logger.info(f"    mean of means: {temporal_mean.mean().item():.6f}")
                logger.info(f"    std of means:  {temporal_mean.std().item():.6f}")
                logger.info("  CLS temporal variance:")
                logger.info(f"    mean variance: {temporal_var.mean().item():.6f}")
                logger.info(
                    "    This measures how much the video content changes over time."
                )

                consecutive_sim = None
                if len(frames) > 1:
                    consecutive_sim = torch.nn.functional.cosine_similarity(
                        cls_features[:-1], cls_features[1:], dim=-1
                    )
                    logger.info(
                        "\n  Frame-to-frame cosine similarity (temporal smoothness):"
                    )
                    logger.info(f"    mean: {consecutive_sim.mean().item():.4f}")
                    logger.info(f"    min:  {consecutive_sim.min().item():.4f}")
                    logger.info(f"    max:  {consecutive_sim.max().item():.4f}")
                    logger.info(f"    std:  {consecutive_sim.std().item():.4f}")

                    threshold = consecutive_sim.mean() - 2 * consecutive_sim.std()
                    scene_changes = (consecutive_sim < threshold).nonzero(
                        as_tuple=True
                    )[0]
                    if len(scene_changes) > 0:
                        logger.info(
                            f"\n  Potential scene changes (sim < {threshold.item():.4f}):"
                        )
                        for idx in scene_changes:
                            logger.info(
                                f"    Frame {idx.item()} -> {idx.item() + 1}: sim={consecutive_sim[idx].item():.4f}"
                            )
                    else:
                        logger.info(
                            f"\n  No scene changes detected (all consecutive sims > {threshold.item():.4f})"
                        )

                logger.info("\n" + "=" * 60)
                logger.info("3. Spatial Statistics (per frame)")
                logger.info("=" * 60)

                patch_var_per_frame = patch_tokens.var(dim=1).mean(dim=-1)
                logger.info("  Patch token variance per frame (spatial diversity):")
                logger.info(f"    mean: {patch_var_per_frame.mean().item():.6f}")
                logger.info(f"    std:  {patch_var_per_frame.std().item():.6f}")
                logger.info(
                    f"    min:  {patch_var_per_frame.min().item():.6f} (frame {patch_var_per_frame.argmin().item()})"
                )
                logger.info(
                    f"    max:  {patch_var_per_frame.max().item():.6f} (frame {patch_var_per_frame.argmax().item()})"
                )

                if args.intermediate_layers > 0:
                    logger.info("\n" + "=" * 60)
                    logger.info(
                        f"4. Intermediate Layer Features (last {args.intermediate_layers} layers)"
                    )
                    logger.info("=" * 60)
                    for layer_idx in range(args.intermediate_layers):
                        inter_cls = features["intermediate_cls"][layer_idx]
                        inter_patches = features["intermediate_patches"][layer_idx]
                        logger.info(f"  Layer {layer_idx}:")
                        print_feature_stats("    CLS", inter_cls)
                        print_feature_stats("    Patches", inter_patches)
                        inter_var = inter_cls.var(dim=0).mean()
                        logger.info(
                            f"    Temporal CLS variance: {inter_var.item():.6f}"
                        )

                logger.info("\n" + "=" * 60)
                logger.info("5. Video-level Summary")
                logger.info("=" * 60)
                logger.info(f"  Video: {video_path.name}")
                logger.info(f"  Duration: {video_info['duration_sec']:.1f}s")
                logger.info(f"  Frames extracted: {len(frames)}")
                logger.info(f"  Effective FPS: {actual_fps:.1f}")
                logger.info(f"  Feature dim: {cls_features.shape[-1]}")
                logger.info(f"  Patches per frame: {patch_tokens.shape[1]}")
                logger.info(
                    f"  Total feature variance: {temporal_var.mean().item():.6f}"
                )
                if consecutive_sim is not None:
                    logger.info(
                        f"  Temporal smoothness (mean cos sim): {consecutive_sim.mean().item():.4f}"
                    )
                logger.info("\n  These metrics can be used for data selection:")
                logger.info("    - Feature variance -> content diversity of this video")
                logger.info("    - Temporal smoothness -> motion/change rate")
                logger.info("    - Spatial diversity -> scene complexity per frame")

                demo_results.append(
                    {
                        "name": demo_name,
                        "video_path": str(video_path),
                        "video_info": video_info,
                        "n_frames": len(frames),
                        "effective_fps": actual_fps,
                        "cls_features": cls_features,
                        "patch_tokens": patch_tokens,
                        "temporal_mean": temporal_mean,
                        "temporal_var": temporal_var,
                        "intermediate_cls": features.get("intermediate_cls", None),
                        "intermediate_patches": features.get(
                            "intermediate_patches", None
                        ),
                    }
                )

                if args.output_dir:
                    save_dict = {
                        "model": args.model,
                        "demo_name": demo_name,
                        "video_path": str(video_path),
                        "video_info": video_info,
                        "n_frames": len(frames),
                        "effective_fps": actual_fps,
                        "cls_features": cls_features,
                        "patch_tokens": patch_tokens,
                        "temporal_mean": temporal_mean,
                        "temporal_var": temporal_var,
                    }
                    if args.intermediate_layers > 0:
                        save_dict["intermediate_cls"] = features["intermediate_cls"]
                        save_dict["intermediate_patches"] = features[
                            "intermediate_patches"
                        ]
                    per_video_output = args.output_dir / f"{demo_name}_features.pt"
                    torch.save(save_dict, per_video_output)
                    logger.info(f"Saved features to {per_video_output}")

            if not demo_results:
                logger.error("No demo videos were successfully processed")
                sys.exit(1)

            if len(demo_results) > 1:
                video_names = [result["name"] for result in demo_results]
                embeddings = torch.stack(
                    [result["temporal_mean"] for result in demo_results], dim=0
                )
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                similarity = embeddings @ embeddings.T

                logger.info("\n" + "=" * 80)
                logger.info(
                    "Inter-video Pairwise Cosine Similarity (temporal_mean embeddings)"
                )
                logger.info("=" * 80)

                max_name_len = max(len(name) for name in video_names)
                col_width = 13
                header = " " * (max_name_len + 2) + "".join(
                    [f"{name[: col_width - 1]:>{col_width}}" for name in video_names]
                )
                logger.info(header)
                for i, name in enumerate(video_names):
                    row_values = "".join(
                        [
                            f"{similarity[i, j].item():>{col_width}.4f}"
                            for j in range(len(video_names))
                        ]
                    )
                    logger.info(f"{name:<{max_name_len}}  {row_values}")

                if args.output_dir:
                    summary_path = args.output_dir / "summary.pt"
                    torch.save(
                        {
                            "model": args.model,
                            "video_names": video_names,
                            "similarity_matrix": similarity,
                            "temporal_mean_embeddings": embeddings,
                            "fps": args.fps,
                            "max_frames": args.max_frames,
                            "start": args.start,
                            "end": args.end,
                        },
                        summary_path,
                    )
                    logger.info(f"Saved multi-video summary to {summary_path}")

            logger.info("\nDone.")
            return

        video_path = demo_videos[0][1]
    elif args.video:
        video_path = args.video
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            sys.exit(1)
    else:
        parser.error("Provide --video or --demo")

    # Sample frames
    frames, actual_fps, video_info, is_rgb = sample_frames(
        video_path,
        target_fps=args.fps,
        max_frames=args.max_frames,
        start_sec=args.start,
        end_sec=args.end,
    )
    if not frames:
        logger.error("No frames extracted")
        sys.exit(1)

    # Load model
    model = load_model(args.model, device)
    transform = make_transform(args.resize, args.crop)

    # Transform frames
    batch = frames_to_batch(frames, transform, is_rgb=is_rgb)
    logger.info(f"Frame batch shape: {batch.shape}")

    # Extract features
    logger.info("=" * 60)
    logger.info("Extracting features...")
    logger.info("=" * 60)
    features = extract_features_batched(
        model,
        batch,
        device,
        batch_size=args.batch_size,
        intermediate_layers=args.intermediate_layers,
    )

    cls_features = features["cls_features"]  # [N, D]
    patch_tokens = features["patch_tokens"]  # [N, P, D]

    # ── 1. Per-frame features ──
    logger.info("\n" + "=" * 60)
    logger.info("1. Per-frame Feature Summary")
    logger.info("=" * 60)
    print_feature_stats("CLS features (all frames)", cls_features)
    print_feature_stats("Patch tokens (all frames)", patch_tokens)

    # ── 2. Temporal statistics ──
    logger.info("\n" + "=" * 60)
    logger.info("2. Temporal Statistics (across frames)")
    logger.info("=" * 60)

    # CLS feature statistics across time
    temporal_mean = cls_features.mean(dim=0)  # [D]
    temporal_var = cls_features.var(dim=0)  # [D]
    logger.info(f"  CLS temporal mean (across {len(frames)} frames):")
    logger.info(f"    mean of means: {temporal_mean.mean().item():.6f}")
    logger.info(f"    std of means:  {temporal_mean.std().item():.6f}")
    logger.info("  CLS temporal variance:")
    logger.info(f"    mean variance: {temporal_var.mean().item():.6f}")
    logger.info("    This measures how much the video content changes over time.")

    # Frame-to-frame cosine similarity (temporal smoothness)
    consecutive_sim = None
    if len(frames) > 1:
        consecutive_sim = torch.nn.functional.cosine_similarity(
            cls_features[:-1], cls_features[1:], dim=-1
        )
        logger.info("\n  Frame-to-frame cosine similarity (temporal smoothness):")
        logger.info(f"    mean: {consecutive_sim.mean().item():.4f}")
        logger.info(f"    min:  {consecutive_sim.min().item():.4f}")
        logger.info(f"    max:  {consecutive_sim.max().item():.4f}")
        logger.info(f"    std:  {consecutive_sim.std().item():.4f}")

        # Detect scene changes (low similarity = potential scene change)
        threshold = consecutive_sim.mean() - 2 * consecutive_sim.std()
        scene_changes = (consecutive_sim < threshold).nonzero(as_tuple=True)[0]
        if len(scene_changes) > 0:
            logger.info(f"\n  Potential scene changes (sim < {threshold.item():.4f}):")
            for idx in scene_changes:
                logger.info(
                    f"    Frame {idx.item()} -> {idx.item() + 1}: sim={consecutive_sim[idx].item():.4f}"
                )
        else:
            logger.info(
                f"\n  No scene changes detected (all consecutive sims > {threshold.item():.4f})"
            )

    # ── 3. Spatial statistics per frame ──
    logger.info("\n" + "=" * 60)
    logger.info("3. Spatial Statistics (per frame)")
    logger.info("=" * 60)

    # Patch variance per frame
    patch_var_per_frame = patch_tokens.var(dim=1).mean(dim=-1)  # [N]
    logger.info("  Patch token variance per frame (spatial diversity):")
    logger.info(f"    mean: {patch_var_per_frame.mean().item():.6f}")
    logger.info(f"    std:  {patch_var_per_frame.std().item():.6f}")
    logger.info(
        f"    min:  {patch_var_per_frame.min().item():.6f} (frame {patch_var_per_frame.argmin().item()})"
    )
    logger.info(
        f"    max:  {patch_var_per_frame.max().item():.6f} (frame {patch_var_per_frame.argmax().item()})"
    )

    # ── 4. Intermediate layers ──
    if args.intermediate_layers > 0:
        logger.info("\n" + "=" * 60)
        logger.info(
            f"4. Intermediate Layer Features (last {args.intermediate_layers} layers)"
        )
        logger.info("=" * 60)
        for layer_idx in range(args.intermediate_layers):
            inter_cls = features["intermediate_cls"][layer_idx]
            inter_patches = features["intermediate_patches"][layer_idx]
            logger.info(f"  Layer {layer_idx}:")
            print_feature_stats("    CLS", inter_cls)
            print_feature_stats("    Patches", inter_patches)
            inter_var = inter_cls.var(dim=0).mean()
            logger.info(f"    Temporal CLS variance: {inter_var.item():.6f}")

    # ── 5. Video-level summary ──
    logger.info("\n" + "=" * 60)
    logger.info("5. Video-level Summary")
    logger.info("=" * 60)
    logger.info(f"  Video: {video_path.name}")
    logger.info(f"  Duration: {video_info['duration_sec']:.1f}s")
    logger.info(f"  Frames extracted: {len(frames)}")
    logger.info(f"  Effective FPS: {actual_fps:.1f}")
    logger.info(f"  Feature dim: {cls_features.shape[-1]}")
    logger.info(f"  Patches per frame: {patch_tokens.shape[1]}")
    logger.info(f"  Total feature variance: {temporal_var.mean().item():.6f}")
    if consecutive_sim is not None:
        logger.info(
            f"  Temporal smoothness (mean cos sim): {consecutive_sim.mean().item():.4f}"
        )
    logger.info("\n  These metrics can be used for data selection:")
    logger.info("    - Feature variance -> content diversity of this video")
    logger.info("    - Temporal smoothness -> motion/change rate")
    logger.info("    - Spatial diversity -> scene complexity per frame")

    # ── Save ──
    if args.output:
        save_dict = {
            "model": args.model,
            "video_path": str(video_path),
            "video_info": video_info,
            "n_frames": len(frames),
            "effective_fps": actual_fps,
            "cls_features": cls_features,
            "patch_tokens": patch_tokens,
            "temporal_mean": temporal_mean,
            "temporal_var": temporal_var,
        }
        if args.intermediate_layers > 0:
            save_dict["intermediate_cls"] = features["intermediate_cls"]
            save_dict["intermediate_patches"] = features["intermediate_patches"]
        torch.save(save_dict, args.output)
        logger.info(f"\nFeatures saved to {args.output}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
