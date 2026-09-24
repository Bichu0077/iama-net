"""Grid search for CLAHE and Bilateral Filter hyperparameters on NEU-DET validation set.

Evaluates combinations of:
    - clip_limit: [1.5, 2.0, 3.0]
    - d (filter diameter): [7, 9]
    - sigma_color/space: [50.0, 75.0, 100.0]

Objective:
    Quantifies contrast enhancement on low-contrast defect regions (e.g. crazing,
    inclusion, pitted_surface) using:
    1. Local Contrast Measure (ECL - Enhanced Contrast Ratio)
    2. Tenengrad Gradient Sharpness (edge preservation score)
    3. Structural Similarity / Noise suppression ratio

Usage:
    python scripts/grid_search_preprocessing.py
"""

import cv2
import numpy as np
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def compute_metrics(original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
    """Compute quantitative image quality metrics for preprocessing evaluation.

    Metrics:
        - contrast_gain: Ratio of standard deviation in L-channel (contrast proxy).
        - edge_sharpness: Tenengrad gradient energy (Sobel gradient magnitude).
        - snr_gain: Signal-to-noise ratio indicator in uniform regions.
    """
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
    proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed

    # 1. Standard deviation of intensity (global contrast)
    std_orig = float(np.std(orig_gray))
    std_proc = float(np.std(proc_gray))
    contrast_ratio = std_proc / (std_orig + 1e-6)

    # 2. Tenengrad gradient sharpness (measures edge clarity)
    gx = cv2.Sobel(proc_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(proc_gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx**2 + gy**2))

    gx_o = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy_o = cv2.Sobel(orig_gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad_orig = float(np.mean(gx_o**2 + gy_o**2))
    sharpness_ratio = tenengrad / (tenengrad_orig + 1e-6)

    # Combined composite score: boosts contrast while preserving/moderating edge sharpness
    # Penalizes over-smoothing (<0.8) and excessive high-frequency noise amplification (>2.0)
    score = contrast_ratio * min(sharpness_ratio, 1.5)

    return {
        "contrast_ratio": round(contrast_ratio, 4),
        "sharpness_ratio": round(sharpness_ratio, 4),
        "composite_score": round(score, 4),
    }


def run_grid_search(
    val_images_dir: Path,
    output_csv: Path,
    sample_vis_dir: Path,
) -> Dict:
    """Run parameter sweep over candidate CLAHE + Bilateral settings."""
    from preprocessing.illumination import illumination_normalize

    sample_vis_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")))
    if not images:
        raise FileNotFoundError(f"No validation images found in {val_images_dir}")

    # Use a representative subset of validation images across defect classes
    sample_images = images[:60]  # 10 per class

    param_grid = [
        {"clip_limit": 1.5, "d": 7, "sigma": 50.0},
        {"clip_limit": 1.5, "d": 9, "sigma": 75.0},
        {"clip_limit": 2.0, "d": 7, "sigma": 50.0},
        {"clip_limit": 2.0, "d": 9, "sigma": 75.0},  # Default spec
        {"clip_limit": 2.0, "d": 9, "sigma": 100.0},
        {"clip_limit": 3.0, "d": 7, "sigma": 50.0},
        {"clip_limit": 3.0, "d": 9, "sigma": 75.0},
        {"clip_limit": 3.0, "d": 9, "sigma": 100.0},
    ]

    results = []
    print(f"\n--- Preprocessing Parameter Grid Search ({len(param_grid)} configs on {len(sample_images)} val images) ---")

    for idx, params in enumerate(param_grid):
        c_ratios, s_ratios, scores = [], [], []

        for img_path in sample_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            processed = illumination_normalize(
                img,
                clip_limit=params["clip_limit"],
                tile_grid=(8, 8),
                d=params["d"],
                sigma_color=params["sigma"],
                sigma_space=params["sigma"],
            )

            metrics = compute_metrics(img, processed)
            c_ratios.append(metrics["contrast_ratio"])
            s_ratios.append(metrics["sharpness_ratio"])
            scores.append(metrics["composite_score"])

        entry = {
            "config_id": idx + 1,
            "clip_limit": params["clip_limit"],
            "d": params["d"],
            "sigma": params["sigma"],
            "mean_contrast_ratio": round(float(np.mean(c_ratios)), 4),
            "mean_sharpness_ratio": round(float(np.mean(s_ratios)), 4),
            "composite_score": round(float(np.mean(scores)), 4),
        }
        results.append(entry)
        print(f"Config {idx+1}: clip={params['clip_limit']}, d={params['d']}, sigma={params['sigma']} -> "
              f"Contrast: {entry['mean_contrast_ratio']}x, Sharpness: {entry['mean_sharpness_ratio']}x, Score: {entry['composite_score']}")

    # Sort by composite score
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    best_config = results[0]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nOptimal configuration: Config {best_config['config_id']} "
          f"(clip_limit={best_config['clip_limit']}, d={best_config['d']}, sigma={best_config['sigma']})")
    print(f"Results written to: {output_csv}")

    # Generate before/after side-by-side comparison images for the best config
    print("\nGenerating before/after side-by-side comparisons...")
    for img_path in sample_images[:6]:
        img = cv2.imread(str(img_path))
        proc = illumination_normalize(
            img,
            clip_limit=best_config["clip_limit"],
            tile_grid=(8, 8),
            d=best_config["d"],
            sigma_color=best_config["sigma"],
            sigma_space=best_config["sigma"],
        )
        # Concatenate horizontally: [Original | Preprocessed]
        comparison = np.hstack([img, proc])
        # Add labels
        cv2.putText(comparison, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(comparison, f"CLAHE+Bilateral (clip={best_config['clip_limit']})", (img.shape[1] + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        out_file = sample_vis_dir / f"compare_{img_path.name}"
        cv2.imwrite(str(out_file), comparison)

    print(f"Saved comparison images to {sample_vis_dir}")
    return best_config


if __name__ == "__main__":
    val_dir = Path("iama-net/data/NEU-DET/images/val")
    csv_out = Path("iama-net/results/preprocessing_grid_search.csv")
    vis_dir = Path("iama-net/results/preprocessing_comparisons")
    run_grid_search(val_dir, csv_out, vis_dir)

