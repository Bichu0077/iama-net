"""Phase 3.4: transfer-confound controls on GC10-DET (evaluation only).

(a) multi-scale: class-agnostic val() at imgsz 640/1024/1280 on ALL 2300
    images for M1m and M4m (resolution-shift control: 2048x1000 downscaled
    vs NEU 200x200 upscaled).
(b) defect-scale-matched tiling: 2048x1000 images cut into 640x640 tiles at
    stride 640 (x in {0,640,1280,1408}, y in {0,360}; right/bottom tiles
    shifted to stay in-bounds), predicted at native scale, boxes mapped back
    and merged with class-agnostic NMS (IoU 0.7), then evaluated with the
    same COCO-style AP + 1000-resample image bootstrap as Phase 1.
    Models: M1m..M4m (matched), raw inputs.
(c) per-GC10-class zero-shot results from the all_perclass caches: class-
    agnostic detections matched against each GC10 class separately
    (mAP@0.5 and recall@conf 0.25).

Outputs (results_v3/phase3/):
  transfer_multiscale.csv, transfer_tiling.csv (+ predictions/*.jsonl caches),
  transfer_perclass.csv
"""

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from iama_env import register_iama_modules, AUDIT_DIR, PRED_DIR, GC10_RAW, PROJECT_ROOT

register_iama_modules()

from ultralytics import YOLO  # noqa: E402
import metrics_lib as ML  # noqa: E402
from bootstrap_ci import PreparedData, summarize  # noqa: E402

OUT = PROJECT_ROOT / "results_v3" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)
TILE_DIRS = OUT / "predictions"
TILE_DIRS.mkdir(exist_ok=True)

REGISTRY = {
    "M1m": PROJECT_ROOT / "runs_v3" / "M1_matched_seed0" / "weights" / "best.pt",
    "M2m": PROJECT_ROOT / "runs_v3" / "M2_matched_seed0" / "weights" / "best.pt",
    "M3m": PROJECT_ROOT / "runs_v3" / "M3_matched_seed0" / "weights" / "best.pt",
    "M4m": PROJECT_ROOT / "runs_v3" / "M4_matched_seed0" / "weights" / "best.pt",
}
TILE = 640
STRIDE = 640
B_BOOT = 1000
SEED = 20261005
CONF = 0.001
IOU_NMS = 0.7


def tile_positions(length, tile=TILE, stride=STRIDE):
    pos = list(range(0, max(length - tile, 0) + 1, stride))
    if not pos or pos[-1] + tile < length:
        pos.append(max(length - tile, 0))
    return sorted(set(pos))


def gt_lookup():
    """stem -> (split, gts (N,5) px, W, H)."""
    out = {}
    for split in ("train", "val", "test"):
        lab = GC10_RAW / "labels" / split
        imgd = GC10_RAW / "images" / split
        for lp in sorted(lab.glob("*.txt")):
            rows = []
            for line in lp.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    c = int(parts[0])
                    cx, cy, w, h = (float(v) for v in parts[1:5])
                    rows.append([c, cx, cy, w, h])
            rows = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
            ip = imgd / f"{lp.stem}.jpg"
            img = cv2.imread(str(ip))
            H, W = img.shape[:2]
            gts = np.zeros((len(rows), 5))
            if len(rows):
                gts[:, 0] = rows[:, 0]
                gts[:, 1] = (rows[:, 1] - rows[:, 3] / 2) * W
                gts[:, 2] = (rows[:, 2] - rows[:, 4] / 2) * H
                gts[:, 3] = (rows[:, 1] + rows[:, 3] / 2) * W
                gts[:, 4] = (rows[:, 2] + rows[:, 4] / 2) * H
            out[lp.stem] = (split, gts, W, H)
    return out


def multiscale(models, scales=(640, 1024, 1280)):
    rows = []
    all_yaml = str(AUDIT_DIR / "gc10_raw_all.yaml")
    for name in models:
        model = YOLO(str(REGISTRY[name]))
        for imgsz in scales:
            batch = 4 if imgsz <= 1024 else 2
            t0 = time.time()
            res = model.val(data=all_yaml, split="test", imgsz=imgsz, batch=batch,
                            device="0", single_cls=True, plots=False, verbose=False,
                            workers=2, project=str(OUT / "valruns"),
                            name=f"ms_{name}_{imgsz}", exist_ok=True)
            rows.append({"model": name, "imgsz": imgsz, "batch": batch,
                         "mAP@0.5": round(float(res.box.map50), 6),
                         "mAP@0.5:0.95": round(float(res.box.map), 6),
                         "precision": round(float(res.box.mp), 6),
                         "recall": round(float(res.box.mr), 6),
                         "minutes": round((time.time() - t0) / 60, 1)})
            print(f"multiscale {name}@{imgsz}: mAP50={res.box.map50:.4f} "
                  f"({rows[-1]['minutes']} min)", flush=True)
            with open(OUT / "transfer_multiscale.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        del model
        torch.cuda.empty_cache()


def tiling(models):
    gt = gt_lookup()
    stems = sorted(gt.keys())
    xs = tile_positions(2048)
    ys = tile_positions(1000)
    print(f"tiles per image: {len(xs)}x{len(ys)} = {len(xs)*len(ys)}", flush=True)
    for name in models:
        cache = TILE_DIRS / f"{name}__gc10_tiled.jsonl"
        if cache.exists() and cache.stat().st_size > 0:
            print(f"tiling cache exists for {name}", flush=True)
            continue
        model = YOLO(str(REGISTRY[name]))
        t0 = time.time()
        with open(cache, "w", encoding="utf-8", newline="\n") as f:
            for n, stem in enumerate(stems):
                split, gts, W, H = gt[stem]
                img = cv2.imread(str(GC10_RAW / "images" / split / f"{stem}.jpg"))
                tiles, offsets = [], []
                for y0 in ys:
                    for x0 in xs:
                        tiles.append(img[y0:y0 + TILE, x0:x0 + TILE])
                        offsets.append((x0, y0))
                dets_all = []
                for s in range(0, len(tiles), 8):
                    chunk = tiles[s:s + 8]
                    res = model.predict(source=chunk, imgsz=TILE, conf=CONF, iou=IOU_NMS,
                                        device="0", max_det=300, verbose=False)
                    for r, (ox, oy) in zip(res, offsets[s:s + 8]):
                        if r.boxes is not None and len(r.boxes):
                            b = r.boxes.xyxy.cpu().numpy().copy()
                            b[:, [0, 2]] += ox
                            b[:, [1, 3]] += oy
                            cf = r.boxes.conf.cpu().numpy()
                            dets_all.append(np.column_stack([cf, b]))  # (score,x1,y1,x2,y2)
                if dets_all:
                    dets = np.vstack(dets_all)
                    keep = ML.nms_xyxy(dets[:, 1:5], dets[:, 0], IOU_NMS)
                    dets = dets[keep]
                    det_rows = [[0, float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])]
                                for d in dets]
                else:
                    det_rows = []
                rec = {"file": f"{stem}.jpg", "split": split, "width": W, "height": H,
                       "dets": det_rows,
                       "gts": [[int(g[0])] + [round(float(v), 2) for v in g[1:]] for g in gts]}
                f.write(json.dumps(rec) + "\n")
                if (n + 1) % 200 == 0:
                    print(f"  {name}: tiled {n+1}/{len(stems)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"tiling {name}: done in {(time.time()-t0)/60:.1f} min", flush=True)
        del model
        torch.cuda.empty_cache()


def tiling_metrics(models):
    rows = []
    for name in models:
        cache = TILE_DIRS / f"{name}__gc10_tiled.jsonl"
        if not cache.exists():
            continue
        recs = ML.load_cache(cache)
        prep = PreparedData(recs, collapse=True, eval_scope="all")
        rng = np.random.default_rng(SEED)
        point, _ = prep.metrics_at(prep.full_index())
        boot = {"mAP@0.5": [], "mAP@0.5:0.95": []}
        for b in range(B_BOOT):
            idx = rng.integers(0, prep.n_images, prep.n_images)
            m, _ = prep.metrics_at(idx)
            for k in boot:
                boot[k].append(m[k])
        for k in boot:
            s = summarize(boot[k])
            rows.append({"model": name, "protocol": "tiled_640_stride640_agnostic_nms0.7",
                         "metric": k, "point": round(float(point[k]), 6),
                         "ci95_low": round(s["lo"], 6), "ci95_high": round(s["hi"], 6),
                         "n_images": prep.n_images, "B": B_BOOT, "seed": SEED})
        print(f"tiling metrics {name}: mAP50={point['mAP@0.5']:.4f}", flush=True)
    with open(OUT / "transfer_tiling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def perclass(models):
    rows = []
    gc10_names = ["crease", "crescent_gap", "inclusion", "oil_spot", "punching_hole",
                  "rolled_pit", "silk_spot", "waist_folding", "water_spot", "welding_line"]
    for name in models:
        cache = PRED_DIR / f"{name}__gc10__raw__all_perclass.jsonl"
        if not cache.exists():
            print(f"perclass: missing cache for {name}", flush=True)
            continue
        recs = ML.load_cache(cache)
        # class-agnostic collapsed dets, GT classes kept, split retained
        base = []
        for r in recs:
            d = np.asarray(r["dets"], dtype=np.float64).reshape(-1, 6)
            g = np.asarray(r["gts"], dtype=np.float64).reshape(-1, 5)
            d = ML.collapse_single_class(d, IOU_NMS)
            base.append((d, g, r["split"]))
        for ci, cname in enumerate(gc10_names):
            def filt(g):
                return g[g[:, 0].astype(int) == ci] if len(g) else np.zeros((0, 5))
            arrs_all = [(d, filt(g)) for d, g, sp in base]
            ap, n_gt = ML.compute_ap_table(arrs_all, [0.5], agnostic=True, classes=[0])
            tp = tot_det = ngt = 0
            for d, g in arrs_all:
                ngt += len(g)
                dd = d[d[:, 1] >= 0.25] if len(d) else d
                tot_det += len(dd)
                if len(dd) and len(g):
                    s, t, _ = ML.match_image(dd, g, 0.5, agnostic=True)
                    tp += int(t.sum())
            rows.append({"model": name, "gc10_class": cname, "scope": "all2300",
                         "n_gt": ngt, "AP50_agnostic": round(float(ap[0][0.5]), 6),
                         "recall_at_0.25": round(tp / ngt, 6) if ngt else "",
                         "tp_at_0.25": tp, "n_det_at_0.25": tot_det})
            arrs_t = [(d, filt(g)) for d, g, sp in base if sp == "test"]
            ap_t, n_gt_t = ML.compute_ap_table(arrs_t, [0.5], agnostic=True, classes=[0])
            rows.append({"model": name, "gc10_class": cname, "scope": "test230",
                         "n_gt": n_gt_t[0], "AP50_agnostic": round(float(ap_t[0][0.5]), 6),
                         "recall_at_0.25": "", "tp_at_0.25": "", "n_det_at_0.25": ""})
        print(f"perclass {name}: done", flush=True)
    with open(OUT / "transfer_perclass.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="M1m,M2m,M3m,M4m")
    ap.add_argument("--skip-multiscale", action="store_true")
    ap.add_argument("--only", default=None, choices=["multiscale", "tiling", "perclass"],
                    help="run a single stage (for reruns after partial failures)")
    args = ap.parse_args()
    models = args.models.split(",")

    for m in models:
        if not REGISTRY[m].exists():
            raise SystemExit(f"{REGISTRY[m]} missing — run Phase 2/3 training first")

    if args.only == "perclass":
        perclass(models)
        return
    if args.only == "tiling":
        tiling(models)
        tiling_metrics(models)
        return
    if args.only == "multiscale":
        multiscale(["M1m", "M4m"])
        return

    if not args.skip_multiscale:
        multiscale(["M1m", "M4m"])
    tiling(models)
    tiling_metrics(models)
    perclass(models)
    print("transfer controls done")


if __name__ == "__main__":
    main()
