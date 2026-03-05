#!/usr/bin/env python3
"""
Extract DINOv2 features from a batch of images.

Usage:
    # Extract from a directory of images
    python scripts/extract_image_features.py --image-dir /path/to/images

    # Extract from specific image files
    python scripts/extract_image_features.py --images img1.jpg img2.png

    # Download sample images and extract (demo mode)
    python scripts/extract_image_features.py --demo

    # Choose model variant
    python scripts/extract_image_features.py --demo --model dinov2_vitb14

    # Save features to file
    python scripts/extract_image_features.py --demo --output features.pt

    # Extract intermediate layer features
    python scripts/extract_image_features.py --demo --intermediate-layers 4
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("extract_image_features")

# DINOv2 canonical eval transform
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

DEMO_URLS = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",  # cats on couch
    "http://images.cocodataset.org/val2017/000000397133.jpg",  # kitchen
    "http://images.cocodataset.org/val2017/000000252219.jpg",  # hydrant
    "http://images.cocodataset.org/val2017/000000087038.jpg",  # elephant
    "http://images.cocodataset.org/val2017/000000174482.jpg",  # airplane
]


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


def download_demo_images(urls: list[str], output_dir: Path) -> list[Path]:
    """Download demo images from URLs."""
    import requests

    paths = []
    for i, url in enumerate(urls):
        filename = url.split("/")[-1]
        filepath = output_dir / filename
        if filepath.exists():
            logger.info(f"  [{i + 1}/{len(urls)}] Already exists: {filepath.name}")
        else:
            logger.info(f"  [{i + 1}/{len(urls)}] Downloading: {url}")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            filepath.write_bytes(resp.content)
        paths.append(filepath)
    return paths


def load_images(
    image_paths: list[Path], transform: transforms.Compose
) -> tuple[torch.Tensor, list[str]]:
    """Load and transform a list of images into a batch tensor."""
    tensors = []
    names = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            tensors.append(transform(img))
            names.append(path.name)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    if not tensors:
        raise RuntimeError("No images loaded successfully")
    batch = torch.stack(tensors)
    logger.info(f"Loaded {len(tensors)} images -> batch shape: {batch.shape}")
    return batch, names


def extract_cls_features(
    model: nn.Module, batch: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Extract CLS token features (global image representation)."""
    batch = batch.to(device)
    with torch.inference_mode():
        cls_features = model(batch)  # [B, D]
    return cls_features


def extract_full_features(
    model: nn.Module, batch: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    """Extract full feature dict: CLS token, patch tokens, (optional) register tokens."""
    batch = batch.to(device)
    with torch.inference_mode():
        out = model.forward_features(batch)
    result = {
        "cls_token": out["x_norm_clstoken"],  # [B, D]
        "patch_tokens": out["x_norm_patchtokens"],  # [B, N, D]
    }
    if "x_norm_regtokens" in out and out["x_norm_regtokens"].numel() > 0:
        result["reg_tokens"] = out["x_norm_regtokens"]  # [B, R, D]
    return result


def extract_intermediate_features(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    n_layers: int = 4,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract features from the last N intermediate layers.

    Returns list of (patch_tokens, cls_token) tuples per layer.
    """
    batch = batch.to(device)
    with torch.inference_mode():
        feats = model.get_intermediate_layers(
            batch, n=n_layers, return_class_token=True, reshape=False, norm=True
        )
    # feats is a list of (patch_tokens [B,N,D], cls_token [B,D])
    return feats


def print_feature_stats(name: str, tensor: torch.Tensor) -> None:
    """Print shape and basic statistics for a feature tensor."""
    logger.info(
        f"  {name}: shape={list(tensor.shape)}, "
        f"mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}, "
        f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DINOv2 features from images")
    parser.add_argument(
        "--model",
        default="dinov2_vits14",
        choices=AVAILABLE_MODELS,
        help="Model variant",
    )
    parser.add_argument("--images", nargs="+", type=Path, help="Image file paths")
    parser.add_argument("--image-dir", type=Path, help="Directory of images")
    parser.add_argument(
        "--demo", action="store_true", help="Download and use demo images"
    )
    parser.add_argument("--output", type=Path, help="Save features to .pt file")
    parser.add_argument(
        "--intermediate-layers",
        type=int,
        default=0,
        help="Number of intermediate layers to extract (0=skip)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--resize", type=int, default=256, help="Resize shorter side to this"
    )
    parser.add_argument("--crop", type=int, default=224, help="Center crop size")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Collect image paths
    image_paths: list[Path] = []
    if args.demo:
        demo_dir = Path(tempfile.mkdtemp(prefix="dinov2_demo_"))
        logger.info(
            f"Demo mode: downloading {len(DEMO_URLS)} sample images to {demo_dir}"
        )
        image_paths = download_demo_images(DEMO_URLS, demo_dir)
    elif args.images:
        image_paths = args.images
    elif args.image_dir:
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        image_paths = sorted(
            p for p in args.image_dir.iterdir() if p.suffix.lower() in suffixes
        )
        logger.info(f"Found {len(image_paths)} images in {args.image_dir}")
    else:
        parser.error("Provide --images, --image-dir, or --demo")

    if not image_paths:
        logger.error("No images found")
        sys.exit(1)

    # Load model
    model = load_model(args.model, device)
    transform = make_transform(args.resize, args.crop)

    # Load images
    batch, names = load_images(image_paths, transform)

    # ── 1. CLS token features ──
    logger.info("=" * 60)
    logger.info("1. CLS Token Features (global image representation)")
    logger.info("=" * 60)
    cls_features = extract_cls_features(model, batch, device)
    print_feature_stats("cls_features", cls_features)

    # Per-image CLS token norms
    norms = cls_features.norm(dim=-1)
    for name, norm in zip(names, norms):
        logger.info(f"  {name}: L2 norm = {norm.item():.4f}")

    # Pairwise cosine similarity
    if len(names) > 1:
        cos_sim = torch.nn.functional.cosine_similarity(
            cls_features.unsqueeze(0), cls_features.unsqueeze(1), dim=-1
        )
        logger.info(f"\n  Pairwise cosine similarity matrix:")
        header = "  " + " " * 20 + "  ".join(f"{n[:8]:>8}" for n in names)
        logger.info(header)
        for i, name in enumerate(names):
            row = f"  {name[:20]:<20}" + "  ".join(
                f"{cos_sim[i, j].item():>8.4f}" for j in range(len(names))
            )
            logger.info(row)

    # ── 2. Full features (CLS + patches) ──
    logger.info("\n" + "=" * 60)
    logger.info("2. Full Features (CLS + patch tokens)")
    logger.info("=" * 60)
    full_features = extract_full_features(model, batch, device)
    for key, val in full_features.items():
        print_feature_stats(key, val)

    # Patch token variance per image (measure of spatial diversity)
    patch_var = full_features["patch_tokens"].var(dim=1).mean(dim=-1)  # [B]
    logger.info(f"\n  Per-image patch token variance (spatial diversity):")
    for name, var in zip(names, patch_var):
        logger.info(f"    {name}: {var.item():.6f}")

    # ── 3. Intermediate layer features (optional) ──
    inter_feats = None
    if args.intermediate_layers > 0:
        logger.info("\n" + "=" * 60)
        logger.info(
            f"3. Intermediate Layer Features (last {args.intermediate_layers} layers)"
        )
        logger.info("=" * 60)
        inter_feats = extract_intermediate_features(
            model, batch, device, args.intermediate_layers
        )
        for layer_idx, (patch_tokens, cls_token) in enumerate(inter_feats):
            logger.info(f"  Layer {layer_idx}:")
            print_feature_stats(f"    patch_tokens", patch_tokens)
            print_feature_stats(f"    cls_token", cls_token)

    # ── 4. Dataset-level statistics (relevant to data selection) ──
    logger.info("\n" + "=" * 60)
    logger.info("4. Dataset-level Statistics")
    logger.info("=" * 60)

    # Feature variance across dataset (how diverse are these images?)
    dataset_feature_var = cls_features.var(dim=0).mean()
    logger.info(
        f"  CLS feature variance (across {len(names)} images): {dataset_feature_var.item():.6f}"
    )

    # Mean pairwise distance
    if len(names) > 1:
        pairwise_dist = torch.cdist(
            cls_features.unsqueeze(0), cls_features.unsqueeze(0)
        ).squeeze(0)
        mask = torch.triu(torch.ones_like(pairwise_dist, dtype=torch.bool), diagonal=1)
        mean_dist = pairwise_dist[mask].mean()
        logger.info(f"  Mean pairwise L2 distance: {mean_dist.item():.4f}")
        logger.info(
            f"  This variance/distance can be used as a data diversity metric for subset selection."
        )

    # ── Save features ──
    if args.output:
        save_dict = {
            "model": args.model,
            "image_names": names,
            "cls_features": cls_features.cpu(),
            "patch_tokens": full_features["patch_tokens"].cpu(),
        }
        if "reg_tokens" in full_features:
            save_dict["reg_tokens"] = full_features["reg_tokens"].cpu()
        if args.intermediate_layers > 0 and inter_feats is not None:
            save_dict["intermediate_layers"] = [
                {"patch_tokens": p.cpu(), "cls_token": c.cpu()} for p, c in inter_feats
            ]
        torch.save(save_dict, args.output)
        logger.info(f"\nFeatures saved to {args.output}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
