#!/usr/bin/env python3
"""
Manage local demo datasets for DINOv2 feature extraction experiments.

Downloads demo images (COCO val2017 samples) and demo videos (Kinetics-mini,
UCF101-subset, LeRobot, OpenCV, BigBuckBunny) into a local directory structure
that can be used by the analysis scripts.

Usage:
    # List available datasets
    python extract_feature/manage_local_datasets.py --list

    # Download all demo data
    python extract_feature/manage_local_datasets.py --download all

    # Download only images
    python extract_feature/manage_local_datasets.py --download images

    # Download only videos
    python extract_feature/manage_local_datasets.py --download videos

    # Custom output directory
    python extract_feature/manage_local_datasets.py --download all --data-dir /path/to/data

    # Check what's already downloaded
    python extract_feature/manage_local_datasets.py --status
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("manage_local_datasets")

# Bypass SOCKS proxy for downloads
os.environ.setdefault("ALL_PROXY", "")
os.environ.setdefault("all_proxy", "")

# Import demo registries from existing scripts
from extract_image_features import DEMO_URLS
from extract_video_features import DEMO_VIDEOS

# Default data directory (gitignored)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "local_datasets"

# Additional COCO images for richer demo set
EXTRA_IMAGE_URLS = [
    "http://images.cocodataset.org/val2017/000000001675.jpg",  # bike
    "http://images.cocodataset.org/val2017/000000109916.jpg",  # dog
    "http://images.cocodataset.org/val2017/000000289343.jpg",  # kite
    "http://images.cocodataset.org/val2017/000000360661.jpg",  # ski
    "http://images.cocodataset.org/val2017/000000458790.jpg",  # baseball
]

ALL_IMAGE_URLS = DEMO_URLS + EXTRA_IMAGE_URLS


def list_datasets() -> None:
    """Print available demo datasets."""
    print("\n" + "=" * 70)
    print("Available Demo Datasets")
    print("=" * 70)

    print("\n── Images ──")
    print(f"  Source: COCO val2017")
    print(f"  Count:  {len(ALL_IMAGE_URLS)} images")
    for i, url in enumerate(ALL_IMAGE_URLS):
        filename = url.split("/")[-1]
        tag = " (core demo)" if i < len(DEMO_URLS) else " (extra)"
        print(f"    {i + 1:2d}. {filename}{tag}")

    print("\n── Videos ──")
    print(f"  Count:  {len(DEMO_VIDEOS)} videos")
    for name, info in DEMO_VIDEOS.items():
        print(f"    • {name}")
        print(f"      Source:   {info['source']}")
        print(f"      Filename: {info['filename']}")

    print("\n" + "=" * 70)
    print(f"Total: {len(ALL_IMAGE_URLS)} images + {len(DEMO_VIDEOS)} videos")
    print("=" * 70 + "\n")


def download_images(data_dir: Path) -> int:
    """Download demo images. Returns count of successfully downloaded images."""
    import requests

    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    for i, url in enumerate(ALL_IMAGE_URLS):
        filename = url.split("/")[-1]
        filepath = image_dir / filename

        if filepath.exists():
            logger.info(f"  [{i + 1}/{len(ALL_IMAGE_URLS)}] Already exists: {filename}")
            success += 1
            continue

        logger.info(f"  [{i + 1}/{len(ALL_IMAGE_URLS)}] Downloading: {filename}")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            filepath.write_bytes(resp.content)
            size_kb = len(resp.content) / 1024
            logger.info(f"    Saved ({size_kb:.0f} KB)")
            success += 1
        except Exception as e:
            logger.warning(f"    Failed: {e}")

    logger.info(f"Images: {success}/{len(ALL_IMAGE_URLS)} available in {image_dir}")
    return success


def download_videos(data_dir: Path) -> int:
    """Download demo videos. Returns count of successfully downloaded videos."""
    import requests

    video_dir = data_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    items = list(DEMO_VIDEOS.items())
    for i, (name, info) in enumerate(items):
        filepath = video_dir / info["filename"]

        if filepath.exists():
            logger.info(f"  [{i + 1}/{len(items)}] Already exists: {info['filename']}")
            success += 1
            continue

        logger.info(f"  [{i + 1}/{len(items)}] Downloading: {name} ({info['source']})")
        try:
            resp = requests.get(info["url"], timeout=120, stream=True)
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
                            f"\r    Downloading: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r    Downloading: {downloaded / 1e6:.1f} MB",
                            end="",
                            flush=True,
                        )
            print()
            logger.info(f"    Saved to {filepath}")
            success += 1
        except Exception as e:
            print()
            logger.warning(f"    Failed to download {name}: {e}")
            if filepath.exists():
                filepath.unlink(missing_ok=True)

    logger.info(f"Videos: {success}/{len(items)} available in {video_dir}")
    return success


def show_status(data_dir: Path) -> None:
    """Show download status of local datasets."""
    print(f"\nDataset directory: {data_dir}")
    print("=" * 60)

    # Images
    image_dir = data_dir / "images"
    if image_dir.exists():
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        total_size = sum(f.stat().st_size for f in image_files)
        print(f"\n  Images: {len(image_files)}/{len(ALL_IMAGE_URLS)} downloaded")
        print(f"  Size:   {total_size / 1024 / 1024:.1f} MB")
        print(f"  Path:   {image_dir}")
        for f in sorted(image_files):
            size_kb = f.stat().st_size / 1024
            print(f"    ✓ {f.name} ({size_kb:.0f} KB)")

        # Check missing
        existing_names = {f.name for f in image_files}
        missing = [
            url.split("/")[-1]
            for url in ALL_IMAGE_URLS
            if url.split("/")[-1] not in existing_names
        ]
        if missing:
            for name in missing:
                print(f"    ✗ {name} (not downloaded)")
    else:
        print(f"\n  Images: not downloaded (0/{len(ALL_IMAGE_URLS)})")

    # Videos
    video_dir = data_dir / "videos"
    if video_dir.exists():
        video_files = [
            f
            for f in video_dir.iterdir()
            if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        ]
        total_size = sum(f.stat().st_size for f in video_files)
        print(f"\n  Videos: {len(video_files)}/{len(DEMO_VIDEOS)} downloaded")
        print(f"  Size:   {total_size / 1024 / 1024:.1f} MB")
        print(f"  Path:   {video_dir}")
        for f in sorted(video_files):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"    ✓ {f.name} ({size_mb:.1f} MB)")

        # Check missing
        existing_names = {f.name for f in video_files}
        missing = [
            info["filename"]
            for info in DEMO_VIDEOS.values()
            if info["filename"] not in existing_names
        ]
        if missing:
            for name in missing:
                print(f"    ✗ {name} (not downloaded)")
    else:
        print(f"\n  Videos: not downloaded (0/{len(DEMO_VIDEOS)})")

    print("\n" + "=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage local demo datasets for DINOv2 experiments"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available demo datasets"
    )
    parser.add_argument(
        "--download",
        choices=["images", "videos", "all"],
        help="Download demo datasets",
    )
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Dataset directory (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()

    if not any([args.list, args.download, args.status]):
        parser.error("Provide --list, --download, or --status")

    if args.list:
        list_datasets()

    if args.download:
        logger.info(f"Dataset directory: {args.data_dir}")
        if args.download in ("images", "all"):
            logger.info("\n" + "=" * 60)
            logger.info("Downloading images...")
            logger.info("=" * 60)
            download_images(args.data_dir)

        if args.download in ("videos", "all"):
            logger.info("\n" + "=" * 60)
            logger.info("Downloading videos...")
            logger.info("=" * 60)
            download_videos(args.data_dir)

        logger.info("\nDownload complete.")

    if args.status:
        show_status(args.data_dir)


if __name__ == "__main__":
    main()
