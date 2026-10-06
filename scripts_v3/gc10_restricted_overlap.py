"""Phase 1.3b: restricted-overlap zero-shot protocol on GC10-DET.

Class mapping (documented, fixed BEFORE looking at results):
  NEU-DET inclusion (cls 1)        <-> GC10-DET inclusion (cls 2)
      both are non-metallic foreign-material inclusions in steel surface.
  NEU-DET pitted_surface (cls 3)   <-> GC10-DET rolled_pit (cls 5)
  NEU-DET rolled-in_scale (cls 4)  <-> GC10-DET rolled_pit (cls 5)
      rolled_pit = pits left by oxide scale rolled into the surface; visually
      localised dark depressions/indentations, the closest GC10 analogue to
      NEU pitted_surface and rolled-in_scale. (GC10 test split contains only
      1 rolled_pit instance; the ALL-images scope has 85, hence the all-image
      requirement of task 1.3.)
  Excluded: NEU crazing/patches/scratches have no GC10 counterpart; GC10
      crease/crescent_gap/oil_spot/punching_hole/silk_spot/waist_folding/
      water_spot/welding_line are process- or stain-specific classes absent
      from NEU-DET.

Protocol: predictions restricted to mapped NEU classes {1,3,4}, GT restricted
to mapped GC10 classes {2,5}, both collapsed to a single 'defect' class
(cross-class NMS IoU 0.7 then merged matching), evaluated on ALL 2300 images
and on the 230-image test split. Bootstrap 95% CIs (1000 resamples, seed as
in bootstrap_ci.py).

Input: prediction caches from eval_unified.py (raw inputs for all models;
preprocessed-input caches for M2/M4 as a secondary diagnostic).

Output: results_v3/audit/gc10_restricted_overlap.csv
"""

import argparse
import csv
import json
import sys

import numpy as np

from pathlib import Path

from iama_env import AUDIT_DIR, PRED_DIR
import metrics_lib as ML

B = 1000
SEED = 20261005

PRED_KEEP = [1, 3, 4]   # NEU inclusion, pitted_surface, rolled-in_scale
GT_KEEP = [2, 5]        # GC10 inclusion, rolled_pit

COMBOS = [
    ("M1", "raw", "all"), ("M2", "raw", "all"), ("M3", "raw", "all"),
    ("M4", "raw", "all"), ("ANCHOR", "raw", "all"),
    ("M2", "pp", "all"), ("M4", "pp", "all"),
    ("M1", "raw", "test"), ("M2", "raw", "test"), ("M3", "raw", "test"),
    ("M4", "raw", "test"), ("ANCHOR", "raw", "test"),
]


def ap_single(scores, tp_sorted, n_gt):
    """Full-curve AP identical to ultralytics compute_ap."""
    if n_gt == 0:
        return np.nan
    if len(tp_sorted) == 0 or not tp_sorted.any():
        return 0.0
    tpc = np.cumsum(tp_sorted)
    prec = tpc / np.arange(1, len(tp_sorted) + 1)
    rec = tpc / n_gt
    return ML.average_precision(rec, prec, method="coco")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", default=None, help="JSON: list of [model, variant, scope]")
    ap.add_argument("--out", default="gc10_restricted_overlap.csv")
    args = ap.parse_args()
    combos = [tuple(c) for c in json.loads(Path(args.combos).read_text())] if args.combos else COMBOS

    rows = []
    for model, variant, scope in combos:
        path = PRED_DIR / f"{model}__gc10__{variant}__all_perclass.jsonl"
        if not path.exists():
            print(f"missing cache {path.name}, skip")
            continue
        recs = ML.load_cache(path)
        if scope == "test":
            recs = [r for r in recs if r["split"] == "test"]
        n_img = len(recs)

        # per-image restricted dets/gts, collapsed & re-NMS'd within the kept set
        dets_list, gts_list = [], []
        for r in recs:
            d = np.asarray(r["dets"], dtype=np.float64).reshape(-1, 6)
            g = np.asarray(r["gts"], dtype=np.float64).reshape(-1, 5)
            d = d[np.isin(d[:, 0].astype(int), PRED_KEEP)] if len(d) else d
            g = g[np.isin(g[:, 0].astype(int), GT_KEEP)] if len(g) else g
            if len(d):
                keep = ML.nms_xyxy(d[:, 2:6], d[:, 1], 0.7)
                d = d[keep]
                d[:, 0] = 0
            if len(g):
                g = g.copy()
                g[:, 0] = 0
            dets_list.append(d)
            gts_list.append(g)

        total_gt = int(sum(len(g) for g in gts_list))
        total_det = int(sum(len(d) for d in dets_list))

        # global sort + per-thr TP flags (reuse bootstrap machinery at thr 0.5 only + 5095)
        thrs = ML.IOU_THRESHOLDS_5095
        scores_all, img_all = [], []
        for ii, d in enumerate(dets_list):
            if len(d):
                scores_all.append(d[:, 1])
                img_all.append(np.full(len(d), ii))
        if scores_all:
            scores_all = np.concatenate(scores_all)
            img_all = np.concatenate(img_all)
            order = np.argsort(-scores_all, kind="stable")
            # rebuild per-image ordered det boxes
            boxes_sorted = np.concatenate([d[:, 2:6] for d in dets_list])[order] if total_det else np.zeros((0, 4))
            img_sorted = img_all[order]
            scores_sorted = scores_all[order]
            tp = np.zeros((len(order), len(thrs)), dtype=bool)
            # group by image: stable argsort keeps score-desc order within image
            sidx = np.argsort(img_sorted, kind="stable")
            sorted_img = img_sorted[sidx]
            starts = np.searchsorted(sorted_img, np.arange(n_img), side="left")
            ends = np.searchsorted(sorted_img, np.arange(n_img), side="right")
            for ii in range(n_img):
                if starts[ii] == ends[ii]:
                    continue
                rows_idx = sidx[starts[ii]:ends[ii]]
                gb = gts_list[ii][:, 1:5] if len(gts_list[ii]) else np.zeros((0, 4))
                if not len(gb):
                    continue
                ious = ML.iou_xyxy(boxes_sorted[rows_idx], gb)
                for ti, t in enumerate(thrs):
                    matched = np.zeros(len(gb), dtype=bool)
                    for k in range(len(rows_idx)):
                        j = int(np.argmax(ious[k]))
                        if not matched[j] and ious[k, j] >= t:
                            matched[j] = True
                            tp[rows_idx[k], ti] = True
        else:
            scores_sorted = np.zeros(0)
            tp = np.zeros((0, len(thrs)), dtype=bool)
            img_sorted = np.zeros(0, dtype=int)

        def metrics_for_counts(sel_counts):
            n_gt = int(sum(sel_counts[ii] * len(gts_list[ii]) for ii in range(n_img)))
            if len(scores_sorted) == 0:
                return dict(map50=np.nan, map5095=np.nan, p=np.nan, r=np.nan, f1=np.nan)
            keep = sel_counts[img_sorted]
            rep = np.repeat(np.arange(len(keep)), keep)
            aps = [ap_single(None, tp[rep, ti], n_gt) for ti in range(len(thrs))]
            map50 = aps[0]
            map5095 = float(np.mean([a for a in aps if a == a]))
            # P/R at fixed conf 0.25 (industrial alarm threshold, documented)
            above = scores_sorted >= 0.25
            mult = keep[above]
            tpf = tp[above, 0]
            tpn = float(mult[tpf].sum())
            fpn = float(mult[~tpf].sum())
            p = tpn / (tpn + fpn) if (tpn + fpn) else np.nan
            r = tpn / n_gt if n_gt else np.nan
            f1 = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else np.nan
            return dict(map50=map50, map5095=map5095, p=p, r=r, f1=f1)

        full = np.ones(n_img, dtype=int)
        point = metrics_for_counts(full)

        rng = np.random.default_rng(SEED)
        boot = {k: [] for k in point}
        for b in range(B):
            idx = rng.integers(0, n_img, n_img)
            cnt = np.bincount(idx, minlength=n_img)
            m = metrics_for_counts(cnt)
            for k in boot:
                boot[k].append(m[k])

        def ci(vals):
            a = np.asarray([v for v in vals if v == v], dtype=float)
            return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)

        for k in ["map50", "map5095", "p", "r", "f1"]:
            lo, hi = ci(boot[k])
            a = np.asarray([v for v in boot[k] if v == v])
            rows.append({
                "model": model, "input_variant": variant, "scope": scope,
                "protocol": "restricted_overlap(pred={NEU inclusion,pitted,rolled-in_scale} vs gt={GC10 inclusion,rolled_pit}, collapsed)",
                "n_images": n_img, "n_gt_restricted": total_gt, "n_dets_restricted": total_det,
                "pr_conf": 0.25,
                "metric": k, "point": round(float(point[k]), 6) if point[k] == point[k] else "",
                "boot_mean": round(float(a.mean()), 6) if len(a) else "",
                "ci95_low": round(lo, 6), "ci95_high": round(hi, 6), "B": B, "seed": SEED,
            })
        print(f"{model}/{variant}/{scope}: n_gt={total_gt} mAP50={point['map50']:.4f} "
              f"P@0.25={point['p']} R@0.25={point['r']}", flush=True)

    with open(AUDIT_DIR / args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()


