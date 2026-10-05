"""CLAHE + bilateral filter preprocessing for IAMA-Net.

Applies contrast-limited adaptive histogram equalization on the L-channel
of LAB color space, followed by an edge-preserving bilateral filter.
Goal: boost local contrast on faint/low-contrast defects (crazing, silk spots)
without amplifying noise or blurring crack/pit edges.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def illumination_normalize(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
    d: int = 5,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> np.ndarray:
    """Apply CLAHE on L-channel of LAB space, then bilateral filter.

    Args:
        img: Input BGR image (uint8).
        clip_limit: CLAHE clip limit for contrast limiting.
        tile_grid: CLAHE tile grid size (rows, cols).
        d: Bilateral filter diameter.
        sigma_color: Bilateral filter sigma in color space.
        sigma_space: Bilateral filter sigma in coordinate space.

    Returns:
        Preprocessed BGR image (uint8).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_ch = clahe.apply(l_ch)
    img_clahe = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(img_clahe, d, sigma_color, sigma_space)


def preprocess_dataset(
    src_dir: Path,
    dst_dir: Path,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
    d: int = 5,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> int:
    """Apply illumination normalization to all images in src_dir, save to dst_dir.

    Preserves subdirectory structure. Copies label files (.txt) unchanged.

    Args:
        src_dir: Source directory containing images.
        dst_dir: Destination directory for preprocessed images.
        clip_limit: CLAHE clip limit.
        tile_grid: CLAHE tile grid size.
        d: Bilateral filter diameter.
        sigma_color: Bilateral filter sigma in color space.
        sigma_space: Bilateral filter sigma in coordinate space.

    Returns:
        Number of images processed.
    """
    import shutil

    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for src_path in sorted(src_dir.rglob("*")):
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        if src_path.suffix.lower() in image_extensions:
            img = cv2.imread(str(src_path))
            if img is None:
                print(f"Warning: Could not read {src_path}, skipping.")
                continue
            processed = illumination_normalize(
                img, clip_limit, tile_grid, d, sigma_color, sigma_space
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_path), processed)
            count += 1
        else:
            # Copy non-image files (labels, etc.) as-is
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_path), str(dst_path))

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess dataset with CLAHE + bilateral filter")
    parser.add_argument("--src", type=str, required=True, help="Source dataset directory")
    parser.add_argument("--dst", type=str, required=True, help="Destination directory")
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--d", type=int, default=9)
    parser.add_argument("--sigma-color", type=float, default=75.0)
    parser.add_argument("--sigma-space", type=float, default=75.0)
    args = parser.parse_args()

    n = preprocess_dataset(
        Path(args.src), Path(args.dst),
        clip_limit=args.clip_limit, d=args.d,
        sigma_color=args.sigma_color, sigma_space=args.sigma_space,
    )
    print(f"Preprocessed {n} images: {args.src} -> {args.dst}")
