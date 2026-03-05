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
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("extract_video_features")

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

# Short, royalty-free sample video
DEMO_VIDEO_URL = (
    "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4"
)


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


def download_demo_video(output_dir: Path) -> Path:
    """Download a demo video for testing."""
    import requests

    filepath = output_dir / "BigBuckBunny_320x180.mp4"
    if filepath.exists():
        logger.info(f"Demo video already exists: {filepath}")
        return filepath
    logger.info(f"Downloading demo video from {DEMO_VIDEO_URL}")
    resp = requests.get(DEMO_VIDEO_URL, timeout=120, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(
                    f"\r  Downloading: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)",
                    end="",
                    flush=True,
                )
    print()
    logger.info(f"Downloaded to {filepath}")
    return filepath


def sample_frames(
    video_path: Path,
    target_fps: float | None = None,
    max_frames: int | None = None,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> tuple[list[np.ndarray], float, dict]:
    """Sample frames from a video file.

    Args:
        video_path: Path to video file
        target_fps: Target sampling FPS (None = use all frames)
        max_frames: Maximum number of frames to extract
        start_sec: Start time in seconds
        end_sec: End time in seconds (None = end of video)

    Returns:
        frames: List of BGR numpy arrays
        actual_fps: The effective sampling rate
        info: Video metadata dict
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
    logger.info(f"Extracted {len(frames)} frames")
    return frames, actual_fps, info


def frames_to_batch(
    frames: list[np.ndarray], transform: transforms.Compose
) -> torch.Tensor:
    """Convert BGR numpy frames to a transformed batch tensor."""
    tensors = []
    for frame in frames:
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
    parser.add_argument("--output", type=Path, help="Save features to .pt file")
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
        demo_dir = Path(tempfile.mkdtemp(prefix="dinov2_video_demo_"))
        video_path = download_demo_video(demo_dir)
    elif args.video:
        video_path = args.video
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            sys.exit(1)
    else:
        parser.error("Provide --video or --demo")

    # Sample frames
    frames, actual_fps, video_info = sample_frames(
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
    batch = frames_to_batch(frames, transform)
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
    logger.info(f"  CLS temporal variance:")
    logger.info(f"    mean variance: {temporal_var.mean().item():.6f}")
    logger.info(f"    This measures how much the video content changes over time.")

    # Frame-to-frame cosine similarity (temporal smoothness)
    if len(frames) > 1:
        consecutive_sim = torch.nn.functional.cosine_similarity(
            cls_features[:-1], cls_features[1:], dim=-1
        )
        logger.info(f"\n  Frame-to-frame cosine similarity (temporal smoothness):")
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
    logger.info(f"  Patch token variance per frame (spatial diversity):")
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
            print_feature_stats(f"    CLS", inter_cls)
            print_feature_stats(f"    Patches", inter_patches)
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
    if len(frames) > 1:
        logger.info(
            f"  Temporal smoothness (mean cos sim): {consecutive_sim.mean().item():.4f}"
        )
    logger.info(f"\n  These metrics can be used for data selection:")
    logger.info(f"    - Feature variance -> content diversity of this video")
    logger.info(f"    - Temporal smoothness -> motion/change rate")
    logger.info(f"    - Spatial diversity -> scene complexity per frame")

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
