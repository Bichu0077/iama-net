"""Evaluation script for IAMA-Net.

Computes Precision, Recall, F1, mAP@0.5, and FPS on a given checkpoint.

Usage:
    python scripts/evaluate.py --weights runs/M1_baseline_seed42/weights/best.pt --data data/NEU-DET/data.yaml
"""

import argparse
import sys
import time
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate(
    weights: str,
    data_yaml: str,
    imgsz: int = 640,
    batch_size: int = 8,
    device: str = "0",
    split: str = "test",
    save_csv: bool = True,
    output_dir: str = None,
) -> dict:
    """Evaluate a trained model checkpoint.

    Args:
        weights: Path to model weights (.pt file).
        data_yaml: Path to dataset YAML.
        imgsz: Image size for evaluation.
        batch_size: Batch size.
        device: Device string.
        split: Dataset split to evaluate on ('val' or 'test').
        save_csv: Whether to save results to CSV.
        output_dir: Directory to save results.

    Returns:
        Dictionary with Precision, Recall, F1, mAP50, FPS.
    """
    from ultralytics import YOLO
    import torch

    print(f"Evaluating: {weights}")
    print(f"Data: {data_yaml} (split: {split})")
    print(f"Device: {device}")

    model = YOLO(weights)

    # Run validation
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        split=split,
        verbose=True,
    )

    # Extract metrics
    precision = float(results.box.mp)  # Mean precision
    recall = float(results.box.mr)     # Mean recall
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    map50 = float(results.box.map50)   # mAP@0.5

    # Measure FPS
    fps = _measure_fps(model, imgsz, device)

    metrics = {
        "weights": weights,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "mAP@0.5": round(map50, 4),
        "FPS": round(fps, 1),
    }

    print(f"\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if save_csv and output_dir:
        out_path = Path(output_dir) / "eval_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
        print(f"Results saved to: {out_path}")

    return metrics


def _measure_fps(model, imgsz: int, device: str, n_warmup: int = 10,
                 n_runs: int = 50) -> float:
    """Measure inference FPS.

    Args:
        model: YOLO model.
        imgsz: Image size.
        device: Device string.
        n_warmup: Number of warmup iterations.
        n_runs: Number of timed iterations.

    Returns:
        Frames per second.
    """
    import torch
    import numpy as np

    # Create a dummy input
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(n_warmup):
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)

    # Timed runs
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return n_runs / elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IAMA-Net checkpoint")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset YAML")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test",
                        help="Split to evaluate on ('val' or 'test', default: test)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    evaluate(
        weights=args.weights,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        split=args.split,
        output_dir=args.output_dir or str(Path(args.weights).parent.parent),
    )
