"""Small Detection Pipeline Script for testing trained weights.

Loads the trained weights (default: M1 baseline best.pt), runs defect
inference on test images, displays bounding boxes with confidence scores,
and saves annotated visualizations.

Usage:
    # Test on default test samples
    python scripts/detect.py

    # Test on a specific image or folder
    python scripts/detect.py --source data/NEU-DET/images/test/patches_1.jpg --conf 0.25

    # Use custom weights and device
    python scripts/detect.py --weights runs/M1_baseline_seed42_20260924_195541/weights/best.pt --device 0
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_default_weights() -> Path:
    """Find the best.pt weights from runs/ directory."""
    runs_dir = PROJECT_ROOT / "runs"
    best_weights = list(runs_dir.rglob("best.pt"))
    if not best_weights:
        raise FileNotFoundError(f"No 'best.pt' weights found in {runs_dir}")
    # Return the most recently modified best.pt
    best_weights.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return best_weights[0]


def run_pipeline(
    weights_path: Path,
    source: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    device: str = "0",
    output_dir: Optional[Path] = None,
) -> List[dict]:
    """Run detection pipeline on image(s) using trained weights.

    Args:
        weights_path: Path to .pt weights file.
        source: Path to an image file or directory of images.
        conf_threshold: Minimum confidence threshold.
        iou_threshold: NMS IoU threshold.
        imgsz: Inference image size.
        device: '0' for GPU or 'cpu'.
        output_dir: Directory to save annotated visualization images.

    Returns:
        List of detection summary dictionaries per image.
    """
    from ultralytics import YOLO
    import torch

    # Check device availability
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    print("=" * 70)
    print("             IAMA-Net Defect Detection Pipeline")
    print("=" * 70)
    print(f"Loading weights: {weights_path}")
    print(f"Device:          {device.upper() if device == 'cpu' else f'CUDA (GPU {device})'}")
    print(f"Confidence:      {conf_threshold}")
    print(f"Image Size:      {imgsz}x{imgsz}")

    # 1. Initialize model
    model = YOLO(str(weights_path))
    class_names = model.names
    print(f"Model Classes:   {list(class_names.values())}")
    print("-" * 70)

    # 2. Collect input images
    if source.is_dir():
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = sorted([p for p in source.iterdir() if p.suffix.lower() in image_exts])
    elif source.is_file():
        image_paths = [source]
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    if not image_paths:
        print(f"No images found at: {source}")
        return []

    print(f"Processing {len(image_paths)} image(s)...")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_detections = []

    # 3. Run inference loop
    for i, img_path in enumerate(image_paths, 1):
        t0 = time.perf_counter()

        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        dt = (time.perf_counter() - t0) * 1000  # ms
        result = results[0]
        boxes = result.boxes

        num_boxes = len(boxes) if boxes is not None else 0
        print(f"\n[{i}/{len(image_paths)}] {img_path.name} | {dt:.1f}ms | {num_boxes} defect(s) detected:")

        img_detections = {
            "image": img_path.name,
            "inference_ms": round(dt, 2),
            "num_defects": num_boxes,
            "defects": [],
        }

        if num_boxes > 0:
            for b in boxes:
                cls_id = int(b.cls[0].item())
                cls_name = class_names[cls_id]
                conf = float(b.conf[0].item())
                xyxy = [round(coord, 1) for coord in b.xyxy[0].tolist()]

                print(f"   -> Class: {cls_name:<16} Confidence: {conf:.2%}  BBox: {xyxy}")
                img_detections["defects"].append({
                    "class": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": xyxy,
                })
        else:
            print("   -> No defects detected above confidence threshold.")

        # 4. Save visualization
        if output_dir:
            annotated_img = result.plot()
            out_file = output_dir / f"pred_{img_path.name}"
            import cv2
            cv2.imwrite(str(out_file), annotated_img)
            img_detections["saved_to"] = str(out_file)

        all_detections.append(img_detections)

    print("\n" + "=" * 70)
    print(f"Pipeline finished! Processed {len(image_paths)} image(s).")
    if output_dir:
        print(f"Annotated images saved to: {output_dir}")
    print("=" * 70)

    return all_detections


def get_default_test_samples() -> List[Path]:
    """Pick 1 representative image per defect class from the test split."""
    test_dir = PROJECT_ROOT / "data" / "NEU-DET" / "images" / "test"
    if not test_dir.exists():
        return []

    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    samples = []
    for cls_prefix in classes:
        matches = list(test_dir.glob(f"{cls_prefix}_*.jpg"))
        if matches:
            samples.append(sorted(matches)[0])
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test defect detection pipeline using trained weights")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to trained weights .pt file (default: finds latest best.pt)")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to image or folder of images (default: 1 test sample per class)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (default: 640)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU (default: '0')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save annotated images (default: results/detection_tests)")
    args = parser.parse_args()

    # Resolve weights
    if args.weights:
        weights_file = Path(args.weights).resolve()
    else:
        weights_file = find_default_weights()

    # Resolve output directory
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (PROJECT_ROOT / "results" / "detection_tests")

    # Resolve source images
    if args.source:
        src_path = Path(args.source).resolve()
        run_pipeline(
            weights_path=weights_file,
            source=src_path,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            output_dir=out_dir,
        )
    else:
        # Default mode: select representative test samples across classes
        test_samples = get_default_test_samples()
        if not test_samples:
            print("Default test directory not found. Please specify --source <image_path>.")
            sys.exit(1)

        print(f"No source specified. Selected {len(test_samples)} representative test samples (1 per defect class).")
        temp_dir = out_dir / "_input_samples"
        temp_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for s in test_samples:
            shutil.copy2(str(s), str(temp_dir / s.name))

        run_pipeline(
            weights_path=weights_file,
            source=temp_dir,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            output_dir=out_dir,
        )

        # Clean up temporary staging
        shutil.rmtree(str(temp_dir), ignore_errors=True)
