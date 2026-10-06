"""Audit side-check: sensitivity of val() metrics to batch size.

The OLD scripts/evaluate.py used batch=8; the v3 unified eval uses batch=4
(4 GB VRAM safety). Ultralytics val uses rect batching, so batch composition
changes letterbox padding and can shift metrics slightly. This script runs
val() at batch 4 and 8 for all NEU-DET primary combos and records the deltas,
so every old-vs-new mismatch can be attributed (checkpoint vs protocol vs
no known cause).

Output: results_v3/audit/val_batch_sensitivity.csv
"""

import csv
import sys

from iama_env import register_iama_modules, CHECKPOINTS, AUDIT_DIR, NEU_RAW, NEU_PP

register_iama_modules()
from ultralytics import YOLO  # noqa: E402

COMBOS = [
    ("M1", NEU_RAW), ("M2", NEU_PP), ("M3", NEU_RAW), ("M4", NEU_PP), ("ANCHOR", NEU_RAW),
]


def main():
    rows = []
    for model_id, root in COMBOS:
        model = YOLO(str(CHECKPOINTS[model_id]))
        for batch in (4, 8):
            try:
                res = model.val(data=str(root / "data.yaml"), split="test", imgsz=640,
                                batch=batch, device="0", plots=False, verbose=False, workers=2,
                                project=str(AUDIT_DIR / "valruns"),
                                name=f"batchsens_{model_id}_b{batch}", exist_ok=True)
                rows.append({"model": model_id, "batch": batch,
                             "mAP@0.5": round(float(res.box.map50), 6),
                             "mAP@0.5:0.95": round(float(res.box.map), 6),
                             "precision": round(float(res.box.mp), 6),
                             "recall": round(float(res.box.mr), 6)})
                print(model_id, batch, rows[-1])
            except Exception as e:  # noqa: BLE001
                rows.append({"model": model_id, "batch": batch, "mAP@0.5": f"ERROR: {e}"})
                print(model_id, batch, "ERROR", e)
        del model
    with open(AUDIT_DIR / "val_batch_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "batch", "mAP@0.5", "mAP@0.5:0.95", "precision", "recall"])
        w.writeheader()
        w.writerows(rows)
    print("saved val_batch_sensitivity.csv")


if __name__ == "__main__":
    main()
