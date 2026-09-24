"""Cross-dataset evaluation: NEU-DET trained models on GC10-DET.

Uses class-agnostic localization transfer protocol:
- Collapses all predictions and ground truth to a single 'defect' class
- Reports detection recall and mAP as a measure of domain transfer
- No fine-tuning is performed

Rationale: NEU-DET (6 classes) and GC10-DET (10 classes) have minimal class
overlap (only 'inclusion' appears in both, with different visual patterns).
Per-class mAP comparison is not meaningful across these datasets.

Usage:
    python scripts/cross_dataset_eval.py --weights runs/M1/best.pt --gc10-data data/GC10-DET/data.yaml
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def cross_dataset_eval(
    weights: str,
    gc10_data_yaml: str,
    imgsz: int = 640,
    batch_size: int = 8,
    device: str = "0",
    output_dir: str = None,
) -> dict:
    """Evaluate NEU-DET-trained model on GC10-DET (class-agnostic).

    Protocol:
        1. Load model trained on NEU-DET
        2. Run inference on GC10-DET test set
        3. Collapse all predictions to single 'defect' class
        4. Compute class-agnostic mAP, Precision, Recall

    Args:
        weights: Path to model weights trained on NEU-DET.
        gc10_data_yaml: Path to GC10-DET dataset YAML.
        imgsz: Image size.
        batch_size: Batch size.
        device: Device string.
        output_dir: Directory to save results.

    Returns:
        Dictionary with class-agnostic metrics.
    """
    from ultralytics import YOLO

    print(f"\nCross-dataset evaluation (class-agnostic)")
    print(f"Model: {weights}")
    print(f"Eval data: {gc10_data_yaml}")
    print(f"Protocol: All classes collapsed to 'defect'")
    print(f"No fine-tuning performed.\n")

    model = YOLO(weights)

    # Run validation on GC10-DET
    # Note: class-agnostic eval - we collapse to single class
    results = model.val(
        data=gc10_data_yaml,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        verbose=True,
        # single_cls=True will treat all classes as one 'defect' class
        single_cls=True,
    )

    precision = float(results.box.mp)
    recall = float(results.box.mr)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    map50 = float(results.box.map50)

    metrics = {
        "weights": weights,
        "Protocol": "class-agnostic (single_cls)",
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "mAP@0.5": round(map50, 4),
    }

    print(f"\nCross-dataset results (class-agnostic):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if output_dir:
        out_path = Path(output_dir) / "cross_dataset_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if out_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(metrics)
        print(f"Results saved to: {out_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-dataset eval on GC10-DET")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--gc10-data", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    cross_dataset_eval(
        weights=args.weights,
        gc10_data_yaml=args.gc10_data,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )
