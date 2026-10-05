"""Grad-CAM / Spatial Attention visualization for IAMA-Net.

Hooks the feature attention layers in M1 (baseline) and M4 (full IAMA-Net),
computes spatial activation heatmaps (EigenCAM/Grad-CAM), and saves
side-by-side comparative visualizations across defect classes.

Usage:
    python scripts/gradcam.py --m1-weights runs/M1_baseline_seed42_20260924_195541/weights/best.pt \
                              --m4-weights runs/M4_full_seed42_20261004_174346/weights/best.pt \
                              --images-dir data/NEU-DET/images/test \
                              --output results/gradcam_examples
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_cam(feature_map: torch.Tensor) -> np.ndarray:
    """Compute spatial activation heatmap from feature map using first principal component.

    Args:
        feature_map: Tensor of shape (1, C, H, W).

    Returns:
        Heatmap array of shape (H, W) normalized to [0, 1].
    """
    f = feature_map[0].detach().cpu().float()
    c, h, w = f.shape
    f_flat = f.view(c, -1).numpy()  # (C, H*W)

    # Center channels
    f_centered = f_flat - f_flat.mean(axis=1, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(f_centered, full_matrices=False)
        cam = vt[0].reshape(h, w)
    except Exception:
        # Fallback to channel-wise mean if SVD fails
        cam = f.mean(dim=0).numpy()

    # Flip sign if needed so peak activation corresponds to high feature variance
    if np.abs(cam.max()) < np.abs(cam.min()):
        cam = -cam

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.astype(np.float32)


def overlay_heatmap(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a normalized heatmap onto an image.

    Args:
        img: BGR image (H, W, 3).
        heatmap: 2D array in [0, 1].
        alpha: Blend weight for the heatmap.
        colormap: OpenCV colormap enum.

    Returns:
        Blended BGR image.
    """
    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), colormap
    )
    return cv2.addWeighted(img, 1.0 - alpha, heatmap_colored, alpha, 0)


def generate_gradcam_comparisons(
    m1_weights: str,
    m4_weights: str,
    images_dir: str,
    output_dir: str,
    device: str = "0",
    samples_per_class: int = 2,
) -> List[str]:
    """Generate side-by-side comparison heatmaps for M1 vs M4.

    Args:
        m1_weights: Path to M1 baseline checkpoint.
        m4_weights: Path to M4 full model checkpoint.
        images_dir: Path to directory containing test images.
        output_dir: Directory to save generated images.
        device: Device string ('0' or 'cpu').
        samples_per_class: Number of representative images per class.

    Returns:
        List of generated file paths.
    """
    from ultralytics import YOLO

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading M1 model from: {m1_weights}")
    m1 = YOLO(m1_weights)
    print(f"Loading M4 model from: {m4_weights}")
    m4 = YOLO(m4_weights)

    # Register hooks on feature attention layers
    # M1 layer 10 is C2PSA (P5 attention)
    # M4 layer 13 is ECA_SpatialAttn (P5 attention)
    m1_acts = []
    m4_acts = []

    m1.model.model[10].register_forward_hook(lambda m, i, o: m1_acts.append(o))
    m4.model.model[13].register_forward_hook(lambda m, i, o: m4_acts.append(o))

    img_dir = Path(images_dir)
    test_files = sorted(img_dir.glob("*.jpg"))

    # Group by defect class
    class_groups = {}
    for f in test_files:
        cls_name = f.stem.rsplit("_", 1)[0]
        class_groups.setdefault(cls_name, []).append(f)

    print(f"Found {len(class_groups)} defect classes: {list(class_groups.keys())}")

    saved_images = []
    for cls_name, files in sorted(class_groups.items()):
        selected_files = files[:samples_per_class]
        for img_file in selected_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Run M1 inference
            m1_acts.clear()
            res_m1 = m1.predict(img, imgsz=640, device=device, verbose=False)
            if m1_acts:
                cam_m1 = compute_cam(m1_acts[-1])
            else:
                cam_m1 = np.zeros((20, 20), dtype=np.float32)

            # Run M4 inference
            m4_acts.clear()
            res_m4 = m4.predict(img, imgsz=640, device=device, verbose=False)
            if m4_acts:
                cam_m4 = compute_cam(m4_acts[-1])
            else:
                cam_m4 = np.zeros((20, 20), dtype=np.float32)

            # Generate overlay
            overlay_m1 = overlay_heatmap(img, cam_m1, alpha=0.45)
            overlay_m4 = overlay_heatmap(img, cam_m4, alpha=0.45)

            # Draw detection bounding boxes
            annotated_m1 = overlay_m1.copy()
            for box in res_m1[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{m1.names[cls_id]} {conf:.2f}"
                cv2.rectangle(annotated_m1, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(
                    annotated_m1, label, (xyxy[0], max(15, xyxy[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA
                )

            annotated_m4 = overlay_m4.copy()
            for box in res_m4[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{m4.names[cls_id]} {conf:.2f}"
                cv2.rectangle(annotated_m4, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 2)
                cv2.putText(
                    annotated_m4, label, (xyxy[0], max(15, xyxy[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
                )

            # Add title headers to each panel
            header_h = 35
            def make_panel(sub_img, title):
                panel = np.zeros((h + header_h, w, 3), dtype=np.uint8)
                panel[header_h:, :] = sub_img
                # Gray header
                panel[:header_h, :] = (40, 40, 40)
                cv2.putText(
                    panel, title, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                )
                return panel

            p1 = make_panel(img, f"Input: {img_file.name}")
            p2 = make_panel(annotated_m1, f"M1 Baseline (C2PSA Attention)")
            p3 = make_panel(annotated_m4, f"M4 IAMA-Net (ECA+Spatial)")

            comparison = np.hstack([p1, p2, p3])

            out_file = out_path / f"comparison_{img_file.stem}.jpg"
            cv2.imwrite(str(out_file), comparison)
            saved_images.append(str(out_file))
            print(f"Generated comparison: {out_file.name}")

    print(f"\nSaved {len(saved_images)} comparison heatmaps to: {out_path}")
    return saved_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM / Spatial Attention visualization for IAMA-Net")
    parser.add_argument("--m1-weights", type=str,
                        default="runs/M1_baseline_seed42_20260924_195541/weights/best.pt",
                        help="M1 baseline weights")
    parser.add_argument("--m4-weights", type=str,
                        default="runs/M4_full_seed42_20261004_174346/weights/best.pt",
                        help="M4 full model weights")
    parser.add_argument("--images-dir", type=str,
                        default="data/NEU-DET/images/test",
                        help="Directory containing test images")
    parser.add_argument("--output", type=str,
                        default="results/gradcam_examples",
                        help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default="0",
                        help="Device to use ('0' or 'cpu')")
    parser.add_argument("--samples-per-class", type=int, default=2,
                        help="Number of representative samples per defect class")
    args = parser.parse_args()

    generate_gradcam_comparisons(
        m1_weights=str(PROJECT_ROOT / args.m1_weights),
        m4_weights=str(PROJECT_ROOT / args.m4_weights),
        images_dir=str(PROJECT_ROOT / args.images_dir),
        output_dir=str(PROJECT_ROOT / args.output),
        device=args.device,
        samples_per_class=args.samples_per_class,
    )
