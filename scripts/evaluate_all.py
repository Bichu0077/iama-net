"""Run comprehensive evaluation on NEU-DET test set and GC10-DET transfer set."""

import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import evaluate
from scripts.cross_dataset_eval import cross_dataset_eval


def main():
    models = [
        ("M1", "Baseline YOLOv11s", "runs/M1_baseline_seed42_20260924_195541/weights/best.pt", "data/NEU-DET/data.yaml"),
        ("M2", "YOLOv11s + CLAHE/Bilateral", "runs/M2_aligned_seed42/weights/best.pt", "data/NEU-DET_preprocessed/data.yaml"),
        ("M3", "YOLOv11s + ECA/SpatialAttn", "runs/M3_aligned_seed42/weights/best.pt", "data/NEU-DET/data.yaml"),
        ("M4", "Full IAMA-Net", "runs/M4_aligned_seed42/weights/best.pt", "data/NEU-DET_preprocessed/data.yaml"),
    ]

    print("\n" + "=" * 70)
    print("STEP 1: EVALUATION ON NEU-DET TEST SET (270 images)")
    print("=" * 70 + "\n")

    ablation_rows = []
    for cfg, desc, w_path, data_path in models:
        full_w = str(PROJECT_ROOT / w_path)
        full_d = str(PROJECT_ROOT / data_path)
        print(f"\n--- Evaluating {cfg}: {desc} ---")
        metrics = evaluate(weights=full_w, data_yaml=full_d, split="test", device="0", save_csv=False)
        row = {
            "Config": cfg,
            "Description": desc,
            "mAP@0.5": metrics["mAP@0.5"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "FPS": metrics["FPS"],
        }
        ablation_rows.append(row)
        print(f"-> {cfg} Test mAP@0.5: {metrics['mAP@0.5']} | Precision: {metrics['Precision']} | Recall: {metrics['Recall']} | FPS: {metrics['FPS']}")

    # Save ablation table
    ablation_csv = PROJECT_ROOT / "results" / "ablation_table.csv"
    with open(ablation_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ablation_rows[0].keys())
        writer.writeheader()
        writer.writerows(ablation_rows)
    print(f"\nSaved master ablation table to {ablation_csv}")

    print("\n" + "=" * 70)
    print("STEP 2: ZERO-SHOT CROSS-DATASET BENCHMARK (GC10-DET)")
    print("=" * 70 + "\n")

    gc10_d = str(PROJECT_ROOT / "data" / "GC10-DET" / "data.yaml")
    gc10_rows = []
    for cfg, desc, w_path, _ in models:
        full_w = str(PROJECT_ROOT / w_path)
        print(f"\n--- GC10-DET Transfer for {cfg} ---")
        res_gc = cross_dataset_eval(weights=full_w, gc10_data_yaml=gc10_d, split="test", device="0", output_dir=None)
        row = {
            "Config": cfg,
            "Weights": full_w,
            "Protocol": res_gc["Protocol"],
            "mAP@0.5": res_gc["mAP@0.5"],
            "Precision": res_gc["Precision"],
            "Recall": res_gc["Recall"],
            "F1": res_gc["F1"],
        }
        gc10_rows.append(row)
        print(f"-> {cfg} GC10 Transfer mAP@0.5: {res_gc['mAP@0.5']} | Precision: {res_gc['Precision']} | Recall: {res_gc['Recall']}")

    gc10_csv = PROJECT_ROOT / "results" / "cross_dataset_results.csv"
    with open(gc10_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gc10_rows[0].keys())
        writer.writeheader()
        writer.writerows(gc10_rows)
    print(f"\nSaved cross-dataset table to {gc10_csv}")


if __name__ == "__main__":
    main()

