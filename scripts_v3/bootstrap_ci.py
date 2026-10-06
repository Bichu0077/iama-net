"""Phase 1.4: bootstrap 95% CIs for every audited metric + paired bootstrap
tests for M4 vs M1.

Method
------
- Unit of resampling: images (with replacement), B=1000, seed=20261005.
- Metrics recomputed per resample from cached per-image predictions
  (conf=0.001, NMS IoU=0.7 caches written by eval_unified.py):
    * mAP@0.5, mAP@0.5:0.95 (macro mean over classes present; COCO 101-pt AP)
    * per-class AP@0.5
    * mean Precision / Recall / F1 at the FULL-SET max-mean-F1 confidence
      (conf fixed at the point estimate to keep the statistic well-defined)
- Class-agnostic (GC10) protocol: cached per-class detections are merged and
  re-NMS'd across classes (IoU 0.7), GT collapsed to one class -- matches
  ultralytics single_cls semantics.
- Paired bootstrap M4 vs M1: identical resampled image indices for both
  models; two-sided p = 2*min(P(delta<=0), P(delta>=0)), capped at 1.0.

Efficiency: detections are globally sorted by score once; per resample only
a boolean selection mask is applied, and AP is computed from TP positions
(recall increments) rather than the full PR curve.

Outputs (results_v3/audit/):
  bootstrap_ci.csv
  paired_bootstrap_m4_vs_m1.csv
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from iama_env import AUDIT_DIR, PRED_DIR, NEU_NAMES, GC10_NAMES
import metrics_lib as ML

B = 1000
SEED = 20261005
THRS = ML.IOU_THRESHOLDS_5095

# (model, dataset, variant, scope_in_cache, eval_scope, collapse)
# eval_scope: which images from the cache enter the metric ("all" or "test")
COMBOS = [
    ("M1", "neu", "raw", "test", "all", False),
    ("M2", "neu", "pp", "test", "all", False),
    ("M3", "neu", "raw", "test", "all", False),
    ("M4", "neu", "pp", "test", "all", False),
    ("ANCHOR", "neu", "raw", "test", "all", False),
    ("M2", "neu", "raw", "test", "all", False),
    ("M4", "neu", "raw", "test", "all", False),
    ("M1", "gc10", "raw", "all", "all", True),
    ("M2", "gc10", "raw", "all", "all", True),
    ("M3", "gc10", "raw", "all", "all", True),
    ("M4", "gc10", "raw", "all", "all", True),
    ("ANCHOR", "gc10", "raw", "all", "all", True),
    ("M2", "gc10", "pp", "all", "all", True),
    ("M4", "gc10", "pp", "all", "all", True),
    ("M1", "gc10", "raw", "all", "test", True),
    ("M2", "gc10", "raw", "all", "test", True),
    ("M3", "gc10", "raw", "all", "test", True),
    ("M4", "gc10", "raw", "all", "test", True),
    ("ANCHOR", "gc10", "raw", "all", "test", True),
]

PAIRED = [
    # (dataset, variantM1, variantM4, cache_scope, eval_scope, collapse, label)
    ("neu", "raw", "pp", "test", "all", False, "NEU-DET test (M1 raw vs M4 preprocessed inputs)"),
    ("gc10", "raw", "raw", "all", "all", True, "GC10 all images, raw inputs"),
    ("gc10", "raw", "raw", "all", "test", True, "GC10 test split, raw inputs"),
]


class PreparedData:
    """Pre-sorted detection arrays + per-image GT counts for fast bootstrap."""

    def __init__(self, records, collapse, eval_scope):
        if eval_scope != "all":
            records = [r for r in records if r["split"] == eval_scope]
        arrs = ML.records_to_arrays(records, collapse=collapse)
        self.files = [r["file"] for r in records]
        self.n_images = len(arrs)
        self.classes = sorted({int(c) for _, g in arrs if len(g) for c in np.asarray(g)[:, 0]})
        if collapse:
            self.classes = [0]
        # per-class global det table sorted by score desc
        self.score = {}
        self.img_of_det = {}
        self.tp_sorted = {}   # cls -> (Nd, n_thr) bool in sorted order
        for c in self.classes:
            scores, imgs, boxes = [], [], []
            for ii, (d, g) in enumerate(arrs):
                if len(d):
                    m = np.ones(len(d), dtype=bool) if collapse else (d[:, 0].astype(int) == c)
                    if m.any():
                        scores.append(d[m, 1])
                        imgs.append(np.full(int(m.sum()), ii))
                        boxes.append(d[m, 2:6])
            if not scores:
                self.score[c] = np.zeros(0)
                self.img_of_det[c] = np.zeros(0, dtype=int)
                self.tp_sorted[c] = np.zeros((0, len(THRS)), dtype=bool)
                continue
            scores = np.concatenate(scores)
            imgs = np.concatenate(imgs).astype(int)
            boxes = np.concatenate(boxes)
            order = np.argsort(-scores, kind="stable")
            self.score[c] = scores[order]
            self.img_of_det[c] = imgs[order]
            boxes_sorted = boxes[order]
            nd = len(order)
            tp = np.zeros((nd, len(THRS)), dtype=bool)
            # group det rows by image (stable sort keeps score-desc order within image)
            sidx = np.argsort(imgs[order], kind="stable")
            sorted_img = imgs[order][sidx]
            starts = np.searchsorted(sorted_img, np.arange(self.n_images), side="left")
            ends = np.searchsorted(sorted_img, np.arange(self.n_images), side="right")
            for ii in range(self.n_images):
                if starts[ii] == ends[ii]:
                    continue
                rows = sidx[starts[ii]:ends[ii]]
                g = arrs[ii][1]
                if len(g):
                    gboxes = g[:, 1:5] if collapse else g[g[:, 0].astype(int) == c][:, 1:5]
                else:
                    gboxes = np.zeros((0, 4))
                ious = ML.iou_xyxy(boxes_sorted[rows], gboxes) if len(gboxes) else np.zeros((len(rows), 0))
                for ti, t in enumerate(THRS):
                    matched = np.zeros(len(gboxes), dtype=bool)
                    for k in range(len(rows)):
                        if not len(gboxes):
                            break
                        j = int(np.argmax(ious[k]))
                        if not matched[j] and ious[k, j] >= t:
                            matched[j] = True
                            tp[rows[k], ti] = True
            self.tp_sorted[c] = tp
        # per-image per-class GT counts
        self.gt_count = np.zeros((self.n_images, max(self.classes) + 1 if self.classes else 1), dtype=int)
        for ii, (d, g) in enumerate(arrs):
            if len(g):
                for c, cnt in zip(*np.unique(g[:, 0].astype(int), return_counts=True)):
                    if collapse:
                        self.gt_count[ii, 0] += cnt
                    elif c in self.classes:
                        self.gt_count[ii, c] += cnt

    # ---------------------------------------------------------------- metrics

    def metrics_at(self, idx, prf_conf=None):
        """Compute metrics for image index array idx (may contain repeats)."""
        sel_counts = np.bincount(idx, minlength=self.n_images)
        out = {}
        ap50 = {}
        ap5095 = {}
        for c in self.classes:
            nd = len(self.score[c])
            n_gt = int(self.gt_count[idx, c].sum())
            if nd == 0 or n_gt == 0:
                ap50[c] = np.nan
                ap5095[c] = np.nan
                continue
            keep = sel_counts[self.img_of_det[c]]
            rep = np.repeat(np.arange(nd), keep)
            aps = []
            for ti in range(len(THRS)):
                tps = self.tp_sorted[c][rep, ti]
                if not tps.any():
                    aps.append(0.0)
                    continue
                # full-curve AP identical to ultralytics compute_ap
                tpc = np.cumsum(tps)
                prec = tpc / np.arange(1, len(tps) + 1)
                rec = tpc / n_gt
                aps.append(ML.average_precision(rec, prec, method="coco"))
            ap50[c] = aps[0]
            ap5095[c] = float(np.mean(aps))
        valid = [c for c in self.classes if not np.isnan(ap50[c])]
        out["mAP@0.5"] = float(np.mean([ap50[c] for c in valid])) if valid else np.nan
        out["mAP@0.5:0.95"] = float(np.mean([ap5095[c] for c in valid])) if valid else np.nan
        # P/R/F1 at fixed per-class conf
        if prf_conf is not None:
            ps, rs = [], []
            for c in self.classes:
                n_gt = int(self.gt_count[idx, c].sum())
                if n_gt == 0:
                    continue  # class absent from this resample
                nd = len(self.score[c])
                if nd == 0:
                    ps.append(0.0)
                    rs.append(0.0)
                    continue
                above = self.score[c] >= prf_conf[c]
                if not above.any():
                    ps.append(0.0)
                    rs.append(0.0)
                    continue
                mult = sel_counts[self.img_of_det[c][above]]
                tpf = self.tp_sorted[c][above, 0]
                tp = float(mult[tpf].sum())
                fp = float(mult[~tpf].sum())
                p = tp / (tp + fp) if (tp + fp) else 0.0
                ps.append(p)
                rs.append(tp / n_gt)
            mp = float(np.mean(ps)) if ps else np.nan
            mr = float(np.mean(rs)) if rs else np.nan
            out["precision_mean"] = mp
            out["recall_mean"] = mr
            out["f1_mean"] = 2 * mp * mr / (mp + mr + 1e-8) if ps and (mp + mr) > 0 else 0.0
        return out, ap50

    def full_index(self):
        return np.arange(self.n_images)


def prepare(model, dataset, variant, cache_scope, eval_scope, collapse):
    path = PRED_DIR / f"{model}__{dataset}__{variant}__{cache_scope}.jsonl"
    if not path.exists():
        return None
    records = ML.load_cache(path)
    return PreparedData(records, collapse, eval_scope)


def bootstrap_one(prep, rng):
    t0 = time.time()
    full = prep.full_index()
    # single shared conf grid (ultralytics-style: one conf maximising MEAN F1
    # across classes), computed on the full set and held fixed for resamples
    grid = np.linspace(0, 1, 400)
    f1_sum = np.zeros(len(grid))
    n_cls_valid = 0
    per_class_curve = {}
    for c in prep.classes:
        nd = len(prep.score[c])
        n_gt = int(prep.gt_count[:, c].sum())
        if nd == 0 or n_gt == 0:
            continue
        n_cls_valid += 1
        tp_all = prep.tp_sorted[c][:, 0]
        tpc_all = np.cumsum(tp_all)
        f1c = np.zeros(len(grid))
        for gi, tau in enumerate(grid):
            m = int((prep.score[c] >= tau).sum())
            if m == 0:
                f1c[gi] = 0.0
                continue
            tp = int(tpc_all[m - 1])
            p = tp / m
            r = tp / n_gt
            f1c[gi] = 2 * p * r / (p + r + 1e-8) if (p + r) > 0 else 0.0
        per_class_curve[c] = f1c
        f1_sum += f1c
    prf_conf = {}
    if n_cls_valid:
        best_i = int(f1_sum.argmax())
        for c in per_class_curve:
            prf_conf[c] = float(grid[best_i])
    point, point_ap50 = prep.metrics_at(full, prf_conf=prf_conf if prf_conf else None)

    boot = {k: [] for k in ["mAP@0.5", "mAP@0.5:0.95", "precision_mean", "recall_mean", "f1_mean"]}
    boot_ap50 = {c: [] for c in prep.classes}
    for b in range(B):
        idx = rng.integers(0, prep.n_images, prep.n_images)
        m, ap50 = prep.metrics_at(idx, prf_conf=prf_conf)
        for k in boot:
            boot[k].append(m[k])
        for c in prep.classes:
            boot_ap50[c].append(ap50[c])
        if (b + 1) % 100 == 0:
            print(f"    resample {b+1}/{B} ({time.time()-t0:.0f}s)", flush=True)
    return point, point_ap50, boot, boot_ap50, prf_conf


def summarize(values):
    a = np.asarray([v for v in values if v == v], dtype=float)  # drop nan
    if len(a) == 0:
        return dict(mean=np.nan, std=np.nan, lo=np.nan, hi=np.nan)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=1)),
                lo=float(np.percentile(a, 2.5)), hi=float(np.percentile(a, 97.5)))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", default=None,
                    help="JSON: list of [model, dataset, variant, cache_scope, eval_scope, collapse]")
    ap.add_argument("--paired", default=None,
                    help="JSON: list of [dataset, variantM1, variantM4, cache_scope, eval_scope, collapse, label]")
    ap.add_argument("--out-prefix", default="", help="prefix for output CSV names")
    ap.add_argument("--models-a", default="M1", help="baseline model id for paired tests")
    ap.add_argument("--models-b", default="M4", help="challenger model id for paired tests")
    args = ap.parse_args()

    global COMBOS, PAIRED
    if args.combos:
        COMBOS = [tuple(c) for c in json.loads(Path(args.combos).read_text())]
    if args.paired:
        PAIRED = [tuple(p) for p in json.loads(Path(args.paired).read_text())]
    A_ID, B_ID = args.models_a, args.models_b

    rng = np.random.default_rng(SEED)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    preps = {}
    out_ci = AUDIT_DIR / f"{args.out_prefix}bootstrap_ci.csv"
    for combo in COMBOS:
        model, dataset, variant, cache_scope, eval_scope, collapse = combo
        print(f"== {model} {dataset}/{variant} scope={eval_scope} ==", flush=True)
        prep = prepare(model, dataset, variant, cache_scope, eval_scope, collapse)
        if prep is None:
            print("   cache missing, skipped")
            continue
        preps[combo] = prep
        point, point_ap50, boot, boot_ap50, prf_conf = bootstrap_one(prep, rng)
        cls_names = GC10_NAMES if dataset == "gc10" else NEU_NAMES
        for k, v in point.items():
            s = summarize(boot[k])
            rows.append(dict(model=model, dataset=dataset, variant=variant, scope=eval_scope,
                             metric=k, cls="", point=round(float(v), 6),
                             boot_mean=round(s["mean"], 6), boot_std=round(s["std"], 6),
                             ci95_low=round(s["lo"], 6), ci95_high=round(s["hi"], 6),
                             B=B, seed=SEED))
        for c in prep.classes:
            s = summarize(boot_ap50[c])
            nm = "defect(agnostic)" if collapse else cls_names[c]
            rows.append(dict(model=model, dataset=dataset, variant=variant, scope=eval_scope,
                             metric="AP50_class", cls=nm, point=round(float(point_ap50[c]), 6),
                             boot_mean=round(s["mean"], 6), boot_std=round(s["std"], 6),
                             ci95_low=round(s["lo"], 6), ci95_high=round(s["hi"], 6),
                             B=B, seed=SEED))
        # dump incrementally
        with open(out_ci, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"   point mAP50={point['mAP@0.5']:.4f} CI=({summarize(boot['mAP@0.5'])['lo']:.4f},"
              f"{summarize(boot['mAP@0.5'])['hi']:.4f})", flush=True)

    # ------------------------------------------------------- paired bootstrap
    paired_rows = []
    prng = np.random.default_rng(SEED + 1)
    for dataset, v1, v4, cache_scope, eval_scope, collapse, label in PAIRED:
        c1 = (A_ID, dataset, v1, cache_scope, eval_scope, collapse)
        c4 = (B_ID, dataset, v4, cache_scope, eval_scope, collapse)
        # lazily prepare caches when running paired-only (empty --combos)
        for cc in (c1, c4):
            if cc not in preps:
                preps[cc] = prepare(*cc)
        p1, p4 = preps.get(c1), preps.get(c4)
        if p1 is None or p4 is None:
            print(f"paired {label}: missing caches, skipped")
            continue
        assert p1.n_images == p4.n_images, f"image count mismatch {p1.n_images} vs {p4.n_images}"
        assert p1.files == p4.files, "image ordering/filenames differ between paired caches"
        print(f"== paired M4 vs M1: {label} ==", flush=True)
        # point deltas
        pt1, _ = p1.metrics_at(p1.full_index())  # A=baseline
        pt4, _ = p4.metrics_at(p4.full_index())
        deltas = {k: [] for k in ["mAP@0.5", "mAP@0.5:0.95"]}
        for b in range(B):
            idx = prng.integers(0, p1.n_images, p1.n_images)
            m1, _ = p1.metrics_at(idx)
            m4, _ = p4.metrics_at(idx)
            for k in deltas:
                deltas[k].append(m4[k] - m1[k])
            if (b + 1) % 100 == 0:
                print(f"    paired resample {b+1}/{B}", flush=True)
        for k in deltas:
            a = np.asarray([d for d in deltas[k] if d == d])
            d_point = pt4[k] - pt1[k]
            frac_le0 = float((a <= 0).mean())
            frac_ge0 = float((a >= 0).mean())
            p_two = min(1.0, 2 * min(frac_le0, frac_ge0))
            paired_rows.append(dict(dataset=dataset, scope=eval_scope, label=label, metric=k,
                                    A_model=A_ID, B_model=B_ID,
                                    A_point=round(float(pt1[k]), 6), B_point=round(float(pt4[k]), 6),
                                    delta_point_pp=round(float(d_point) * 100, 4),
                                    delta_boot_mean_pp=round(float(a.mean()) * 100, 4),
                                    delta_ci95_low_pp=round(float(np.percentile(a, 2.5)) * 100, 4),
                                    delta_ci95_high_pp=round(float(np.percentile(a, 97.5)) * 100, 4),
                                    frac_delta_le_0=round(frac_le0, 4),
                                    p_two_sided=round(p_two, 4), B=B, seed=SEED + 1))
    with open(AUDIT_DIR / f"{args.out_prefix}paired_bootstrap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()) if paired_rows else ["none"])
        w.writeheader()
        w.writerows(paired_rows)
    print("saved paired_bootstrap csv")


if __name__ == "__main__":
    main()

