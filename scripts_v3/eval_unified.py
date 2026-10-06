"""Phase 1.1 + 1.3: unified evaluation of all five original checkpoints.

For every (model, dataset, input-variant, scope) combination this script runs
ultralytics val() with save_json=True and captures validator.jdict, i.e. the
per-image detections produced by the EXACT val pipeline (rect letterbox,
conf=0.001, NMS IoU=0.7, imgsz=640, max_det=300) in original-image pixel
coordinates. These caches feed:
  - our own COCO-style metric implementation (metrics_lib) as an independent
    cross-check of the val() numbers,
  - the bootstrap CI script (1.4),
  - scopes ultralytics val() cannot express directly (GC10 ALL 2300 images,
    via a combined-split data yaml).

Fixed protocol (identical everywhere): imgsz=640, batch=4 (4 GB VRAM),
device=0, plots off, workers=2. val runs are written under
results_v3/audit/valruns (never under runs/).

Outputs (results_v3/audit/):
  val_point_estimates.csv   - ultralytics val() metrics per combo
  own_point_estimates.csv   - metrics_lib metrics per combo
  per_class_ap.csv          - per-class AP50/AP50-95 per combo
  predictions/*.jsonl       - per-image detection caches
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from iama_env import (register_iama_modules, CHECKPOINTS, AUDIT_DIR, PRED_DIR,
                      NEU_RAW, NEU_PP, NEU_GRAY, NEU_NOBIL, GC10_RAW, GC10_PP,
                      PROJECT_ROOT, NEU_NAMES, GC10_NAMES)

register_iama_modules()

from ultralytics import YOLO  # noqa: E402
import metrics_lib as ML  # noqa: E402

IMGSZ = 640
BATCH = 4
DEVICE = "0"

# (model, dataset, variant, scope, primary)
NEU_COMBOS = [
    ("M1", "neu", "raw", "test", True),
    ("M2", "neu", "pp", "test", True),
    ("M3", "neu", "raw", "test", True),
    ("M4", "neu", "pp", "test", True),
    ("ANCHOR", "neu", "raw", "test", True),
    ("M2", "neu", "raw", "test", False),   # input-mismatch diagnostic
    ("M4", "neu", "raw", "test", False),   # input-mismatch diagnostic
]
GC10_COMBOS = [
    # OLD protocol replication: test split only, single_cls
    ("M1", "gc10", "raw", "test", True),
    ("M2", "gc10", "raw", "test", True),
    ("M3", "gc10", "raw", "test", True),
    ("M4", "gc10", "raw", "test", True),
    ("ANCHOR", "gc10", "raw", "test", True),
    # NEW primary: ALL 2300 images, single_cls
    ("M1", "gc10", "raw", "all", True),
    ("M2", "gc10", "raw", "all", True),
    ("M3", "gc10", "raw", "all", True),
    ("M4", "gc10", "raw", "all", True),
    ("ANCHOR", "gc10", "raw", "all", True),
    # ALL images, per-class predictions (for restricted-overlap protocol 1.3b
    # and per-GC10-class analyses)
    ("M1", "gc10", "raw", "all_perclass", True),
    ("M2", "gc10", "raw", "all_perclass", True),
    ("M3", "gc10", "raw", "all_perclass", True),
    ("M4", "gc10", "raw", "all_perclass", True),
    ("ANCHOR", "gc10", "raw", "all_perclass", True),
    # train-distribution diagnostic for preprocessed models
    ("M2", "gc10", "pp", "all", False),
    ("M4", "gc10", "pp", "all", False),
    ("M2", "gc10", "pp", "test", False),
    ("M4", "gc10", "pp", "test", False),
    ("M2", "gc10", "pp", "all_perclass", False),
    ("M4", "gc10", "pp", "all_perclass", False),
]

DS_ROOTS = {("neu", "raw"): NEU_RAW, ("neu", "pp"): NEU_PP,
            ("neu", "gray"): NEU_GRAY, ("neu", "nobil"): NEU_NOBIL,
            ("gc10", "raw"): GC10_RAW, ("gc10", "pp"): GC10_PP}
SPLITS_BY_SCOPE = {"test": ["test"], "all": ["train", "val", "test"],
                   "all_perclass": ["train", "val", "test"]}

GT_CACHE = {}
SPLIT_LOOKUP = {}


def ensure_all_yaml(dataset, variant):
    """Write a data.yaml whose 'test' key covers train+val+test images."""
    root = DS_ROOTS[(dataset, variant)]
    y = AUDIT_DIR / f"{dataset}_{variant}_all.yaml"
    if not y.exists():
        names = GC10_NAMES if dataset == "gc10" else NEU_NAMES
        txt = [f'path: "{root.as_posix()}"',
               "train: images/train",
               "val: images/val",
               "test:", "  - images/train", "  - images/val", "  - images/test",
               f"nc: {len(names)}", "names:"]
        for i, n in enumerate(names):
            txt.append(f"  {i}: {n}")
        y.write_text("\n".join(txt) + "\n")
    return str(y)


def val_yaml(dataset, variant, scope):
    if scope in ("all", "all_perclass"):
        return ensure_all_yaml(dataset, variant)
    return str(DS_ROOTS[(dataset, variant)] / "data.yaml")


def image_list(dataset, variant, scope):
    root = DS_ROOTS[(dataset, variant)]
    out = []
    for split in SPLITS_BY_SCOPE[scope]:
        d = root / "images" / split
        files = sorted(p for p in d.iterdir()
                       if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        out.extend([(split, p) for p in files])
    return out


def load_gt(dataset, variant, scope):
    """GT boxes in original pixel coords: stem -> (Ng,5) [cls,x1,y1,x2,y2]; sizes."""
    root = DS_ROOTS[(dataset, variant)]
    gt = {}
    sizes = {}
    for split in SPLITS_BY_SCOPE[scope]:
        lab_dir = root / "labels" / split
        img_dir = root / "images" / split
        for lp in sorted(lab_dir.glob("*.txt")):
            rows = []
            for line in lp.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                c = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:5])
                rows.append([c, cx, cy, w, h])
            gt[lp.stem] = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
        for ip in img_dir.iterdir():
            if ip.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                img = cv2.imread(str(ip))
                if img is not None:
                    sizes[ip.stem] = (img.shape[1], img.shape[0])
    gt_px = {}
    for stem, rows in gt.items():
        if stem not in sizes:
            continue
        W, H = sizes[stem]
        out = np.zeros((len(rows), 5))
        if len(rows):
            out[:, 0] = rows[:, 0]
            cx, cy, w, h = rows[:, 1] * W, rows[:, 2] * H, rows[:, 3] * W, rows[:, 4] * H
            out[:, 1] = cx - w / 2
            out[:, 2] = cy - h / 2
            out[:, 3] = cx + w / 2
            out[:, 4] = cy + h / 2
        gt_px[stem] = out
    return gt_px, sizes


def build_gt_cache(dataset, variant, scope):
    key = (dataset, variant, scope)
    if key in GT_CACHE:
        return
    GT_CACHE[key] = load_gt(dataset, variant, scope)
    lookup = {}
    for split in SPLITS_BY_SCOPE[scope]:
        d = DS_ROOTS[(dataset, variant)] / "images" / split
        for p in d.iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                if p.stem in lookup:
                    raise RuntimeError(f"stem collision across splits: {p.stem} "
                                       f"({lookup[p.stem]} vs {split}) — cache keyed by stem would be wrong")
                lookup[p.stem] = split
    SPLIT_LOOKUP[key] = lookup


def run_val_cached(model, model_id, dataset, variant, scope):
    """Run val(save_json=True) and write the per-image detection cache."""
    build_gt_cache(dataset, variant, scope)
    gt_px, sizes = GT_CACHE[(dataset, variant, scope)]
    split_of = SPLIT_LOOKUP[(dataset, variant, scope)]

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PRED_DIR / f"{model_id}__{dataset}__{variant}__{scope}.jsonl"

    name = f"{model_id}_{dataset}_{variant}_{scope}"
    t0 = time.time()
    res = model.val(
        data=val_yaml(dataset, variant, scope),
        imgsz=IMGSZ, batch=BATCH, device=DEVICE, split="test",
        single_cls=(dataset == "gc10" and scope != "all_perclass"),
        save_json=True, plots=False, verbose=False, workers=2,
        project=str(AUDIT_DIR / "valruns"), name=name, exist_ok=True,
    )
    # ultralytics 8.3.40 writes jdict to save_dir/predictions.json when save_json=True
    pred_json = AUDIT_DIR / "valruns" / name / "predictions.json"
    with open(pred_json, "r", encoding="utf-8") as f:
        jdict = json.load(f)

    by_img = defaultdict(list)
    for e in jdict:
        stem = str(e["image_id"])
        x, y, w, h = e["bbox"]
        by_img[stem].append([int(e["category_id"]), float(e["score"]),
                             x, y, x + w, y + h])

    paths = image_list(dataset, variant, scope)
    with open(cache_path, "w", encoding="utf-8", newline="\n") as f:
        for split, p in paths:
            stem = p.stem
            W, H = sizes[stem]
            g = gt_px.get(stem, np.zeros((0, 5)))
            rec = {
                "file": p.name, "split": split_of[stem],
                "width": W, "height": H,
                "dets": by_img.get(stem, []),
                "gts": [[int(r[0])] + [round(float(v), 3) for v in r[1:]] for r in g],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"  val+cache {dataset}/{variant}/{scope}: {len(paths)} imgs, {len(jdict)} dets "
          f"in {time.time()-t0:.0f}s", flush=True)
    return res, cache_path


def val_rows_from_res(res, model_id, dataset, variant, scope, primary):
    box = res.box
    rows = []
    common = dict(model=model_id, dataset=dataset, variant=variant, scope=scope,
                  source="ultralytics_val", primary=primary)
    for metric, value in [("mAP@0.5", float(box.map50)), ("mAP@0.5:0.95", float(box.map)),
                          ("precision_mean", float(box.mp)), ("recall_mean", float(box.mr))]:
        rows.append({**common, "metric": metric, "value": round(value, 6)})
    names = res.names
    agnostic = (dataset == "gc10" and scope != "all_perclass")
    for idx, c in enumerate(box.ap_class_index):
        nm = "defect(agnostic)" if agnostic else names[int(c)]
        rows.append({**common, "metric": f"AP50_class_{nm}", "value": round(float(box.ap50[idx]), 6)})
        rows.append({**common, "metric": f"AP5095_class_{nm}", "value": round(float(box.ap[idx]), 6)})
    return rows


def own_metrics(cache_path, dataset, scope):
    records = ML.load_cache(cache_path)
    collapse = (dataset == "gc10" and scope != "all_perclass")
    arrs = ML.records_to_arrays(records, collapse=collapse)
    thrs = ML.IOU_THRESHOLDS_5095
    ap, n_gt = ML.compute_ap_table(arrs, thrs, agnostic=collapse)
    ap50 = {c: ap[c][0.5] for c in ap}
    ap5095 = {c: float(np.nanmean([ap[c][float(t)] for t in thrs])) for c in ap}
    valid = [c for c in ap50 if not np.isnan(ap50[c])]
    rows = {}
    rows["mAP@0.5"] = float(np.mean([ap50[c] for c in valid])) if valid else float("nan")
    rows["mAP@0.5:0.95"] = float(np.mean([ap5095[c] for c in valid])) if valid else float("nan")
    prf = ML.pooled_prf(arrs, agnostic=collapse)
    rows["precision_mean"] = prf["mp"]
    rows["recall_mean"] = prf["mr"]
    rows["f1_mean"] = prf["f1"]
    rows["best_conf"] = prf["best_conf"]
    return rows, ap50, ap5095, n_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["neu", "gc10", "all"], default="all")
    ap.add_argument("--models", default=None)
    ap.add_argument("--registry", default=None,
                    help="JSON file {model_id: weights_path} overriding the original CHECKPOINTS")
    ap.add_argument("--combos", default=None,
                    help="JSON file: list of [model, dataset, variant, scope, primary] overriding built-in combos")
    ap.add_argument("--out-prefix", default="",
                    help="prefix for output CSV names (e.g. matched_) to keep result sets separate")
    args = ap.parse_args()

    registry = {k: (Path(v) if Path(v).is_absolute() else PROJECT_ROOT / v)
                for k, v in json.loads(Path(args.registry).read_text()).items()} \
        if args.registry else dict(CHECKPOINTS)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    combos = []
    if args.combos:
        combos = [tuple(c) for c in json.loads(Path(args.combos).read_text())]
    else:
        if args.stage in ("neu", "all"):
            combos += NEU_COMBOS
        if args.stage in ("gc10", "all"):
            combos += GC10_COMBOS
    if args.models:
        keep = set(args.models.split(","))
        combos = [c for c in combos if c[0] in keep]
    combos = [c for c in combos
              if not (c[1] == "gc10" and c[2] == "pp" and not (GC10_PP / "data.yaml").exists())]

    val_rows, own_rows, percls_rows = [], [], []
    # merge existing CSVs so stages can run incrementally
    out_p = args.out_prefix
    for path, acc in [(AUDIT_DIR / f"{out_p}val_point_estimates.csv", val_rows),
                      (AUDIT_DIR / f"{out_p}own_point_estimates.csv", own_rows),
                      (AUDIT_DIR / f"{out_p}per_class_ap.csv", percls_rows)]:
        if path.exists():
            with open(path, newline="") as f:
                acc.extend(csv.DictReader(f))

    def drop_existing(rows, combo):
        model_id, dataset, variant, scope, _ = combo
        return [r for r in rows
                if not (r.get("model") == model_id and r.get("dataset") == dataset
                        and r.get("variant") == variant and r.get("scope") == scope)]

    models_needed = sorted({c[0] for c in combos}, key=list(registry).index)
    for model_id in models_needed:
        print(f"\n=== {model_id} ({registry[model_id].name}) ===", flush=True)
        model = YOLO(str(registry[model_id]))
        for combo in [c for c in combos if c[0] == model_id]:
            model_id_, dataset, variant, scope, primary = combo
            val_rows = drop_existing(val_rows, combo)
            own_rows = drop_existing(own_rows, combo)
            percls_rows = drop_existing(percls_rows, combo)

            res, cache_path = run_val_cached(model, model_id, dataset, variant, scope)
            if scope == "all_perclass":
                # Cache only: NEU pred class indices do not correspond to GC10
                # GT classes, so val()/own per-class metrics here are not
                # meaningful. Consumed by gc10_restricted_overlap.py which
                # applies an explicit class mapping.
                print("  (all_perclass: cache only, metrics computed by "
                      "gc10_restricted_overlap.py)", flush=True)
                continue
            val_rows += val_rows_from_res(res, model_id, dataset, variant, scope, primary)
            box = res.box
            print(f"  val : mAP50={box.map50:.4f} mAP5095={box.map:.4f} P={box.mp:.4f} R={box.mr:.4f}",
                  flush=True)

            rows, ap50, ap5095, n_gt = own_metrics(cache_path, dataset, scope)
            common = dict(model=model_id, dataset=dataset, variant=variant, scope=scope,
                          source="metrics_lib", primary=primary, cache=cache_path.name)
            for k, v in rows.items():
                own_rows.append({**common, "metric": k, "value": round(float(v), 6)})
            cls_names = GC10_NAMES if dataset == "gc10" else NEU_NAMES
            collapse = (dataset == "gc10" and scope != "all_perclass")
            for cidx in sorted(ap50):
                nm = "defect(agnostic)" if collapse else cls_names[cidx]
                percls_rows.append({**{k: common[k] for k in ("model", "dataset", "variant", "scope", "primary")},
                                    "class": nm, "n_gt": n_gt[cidx],
                                    "AP50": round(ap50[cidx], 6), "AP5095": round(ap5095[cidx], 6)})
            print(f"  own : mAP50={rows['mAP@0.5']:.4f} mAP5095={rows['mAP@0.5:0.95']:.4f} "
                  f"P={rows['precision_mean']:.4f} R={rows['recall_mean']:.4f}", flush=True)

            def dump(path, rws):
                if not rws:
                    return
                keys = list(rws[0].keys())
                for r in rws:
                    for k in r:
                        if k not in keys:
                            keys.append(k)
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(rws)
            dump(AUDIT_DIR / f"{out_p}val_point_estimates.csv", val_rows)
            dump(AUDIT_DIR / f"{out_p}own_point_estimates.csv", own_rows)
            dump(AUDIT_DIR / f"{out_p}per_class_ap.csv", percls_rows)
        del model

    print("\nDONE. Saved val_point_estimates.csv, own_point_estimates.csv, per_class_ap.csv")


if __name__ == "__main__":
    main()


