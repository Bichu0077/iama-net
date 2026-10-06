"""Shared detection-metrics library for the v3 audit (numpy, no ultralytics).

Implements COCO-style matching and AP so that bootstrap CIs can be computed
from cached per-image predictions without re-running inference.

Conventions
-----------
- Boxes are xyxy in ORIGINAL image pixel coordinates.
- Predictions cache format (JSONL, one line per image):
  {"file": name, "split": split, "width": w, "height": h,
   "dets": [[cls, score, x1, y1, x2, y2], ...],
   "gts":  [[cls, x1, y1, x2, y2], ...]}
- AP: 101-point interpolation ("coco") and step-interpolation ("area"),
  matching ultralytics.metrics.average_precision definitions.
"""

import json

import numpy as np

IOU_THRESHOLDS_5095 = np.round(np.arange(0.5, 0.96, 0.05), 2)


# ---------------------------------------------------------------- IoU / match

def iou_xyxy(a, b):
    """IoU between sets. a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_image(dets, gts, thr, agnostic=False, pred_cls_keep=None, gt_cls_keep=None):
    """Greedy per-image matching, highest score first.

    Args:
        dets: (Nd, 6) array [cls, score, x1, y1, x2, y2]
        gts:  (Ng, 5) array [cls, x1, y1, x2, y2]
        thr:  IoU threshold
        agnostic: if True, class ids are ignored in matching
        pred_cls_keep / gt_cls_keep: optional sets of class ids to restrict to
            (for restricted-overlap protocols). Dets outside pred_cls_keep are
            dropped; GTs outside gt_cls_keep are dropped entirely.

    Returns:
        scores: (Nd',) confidence of kept dets (sorted desc)
        tp:     (Nd',) bool true-positive flags
        n_gt:   number of GT boxes counted for this image
    """
    if dets is None or len(dets) == 0:
        scores = np.zeros(0)
        tp = np.zeros(0, dtype=bool)
        n_gt = 0 if gts is None else _count_gt(gts, gt_cls_keep)
        return scores, tp, n_gt

    d = np.asarray(dets, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64) if gts is not None and len(gts) else np.zeros((0, 5))

    if pred_cls_keep is not None:
        d = d[np.isin(d[:, 0].astype(int), list(pred_cls_keep))]
    if gt_cls_keep is not None and len(g):
        g = g[np.isin(g[:, 0].astype(int), list(gt_cls_keep))]

    order = np.argsort(-d[:, 1], kind="stable")
    d = d[order]
    scores = d[:, 1].copy()
    tp = np.zeros(len(d), dtype=bool)

    if len(g) == 0:
        return scores, tp, 0

    if agnostic:
        gcls = np.zeros(len(g), dtype=int)
        dcls = np.zeros(len(d), dtype=int)
    else:
        gcls = g[:, 0].astype(int)
        dcls = d[:, 0].astype(int)

    matched = np.zeros(len(g), dtype=bool)
    ious = iou_xyxy(d[:, 2:6], g[:, 1:5])
    for i in range(len(d)):
        cand = np.where((gcls == dcls[i]) & (~matched))[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmax(ious[i, cand])]
        if ious[i, j] >= thr:
            matched[j] = True
            tp[i] = True
    return scores, tp, int(len(g))


def _count_gt(gts, gt_cls_keep):
    if gts is None or len(gts) == 0:
        return 0
    g = np.asarray(gts)
    if gt_cls_keep is not None:
        g = g[np.isin(g[:, 0].astype(int), list(gt_cls_keep))]
    return int(len(g))


# ------------------------------------------------------------------- NMS

def nms_xyxy(boxes, scores, iou_thr=0.7):
    """Greedy NMS. boxes (N,4) xyxy, scores (N,). Returns kept indices."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(-scores, kind="stable")
    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    b = np.asarray(boxes, dtype=np.float64)
    for pos in range(len(order)):
        if suppressed[pos]:
            continue
        i = order[pos]
        keep.append(i)
        rest = order[pos + 1:]
        rest = rest[~suppressed[pos + 1:]]
        if len(rest) == 0:
            break
        ious = iou_xyxy(b[i:i + 1], b[rest])[0]
        drop = rest[ious >= iou_thr]
        idx_map = {v: k for k, v in enumerate(order)}
        for j in drop:
            suppressed[idx_map[j]] = True
    return np.array(keep, dtype=int)


def collapse_single_class(dets, iou_thr=0.7):
    """Merge all det classes to one and re-NMS across classes.

    Mimics ultralytics val(single_cls=True) post-processing on cached dets.
    dets: (N,6) [cls, score, x1,y1,x2,y2] -> returns (M,6) with cls=0.
    """
    if dets is None or len(dets) == 0:
        return np.zeros((0, 6))
    d = np.asarray(dets, dtype=np.float64).copy()
    keep = nms_xyxy(d[:, 2:6], d[:, 1], iou_thr)
    d = d[keep]
    d[:, 0] = 0
    return d


# ------------------------------------------------------------------- AP

def average_precision(rec, prec, method="coco"):
    """AP from a PR curve (recall, precision arrays in detection order).

    'coco' matches ultralytics.utils.metrics.compute_ap exactly:
    sentinel-padded, vectorised monotone-decreasing envelope, 101-point
    trapezoid interpolation.
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))  # precision envelope
    if method == "area":
        i = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    elif method == "coco":
        x = np.linspace(0, 1, 101)
        return float(np.trapezoid(np.interp(x, mrec, mpre), x))
    raise ValueError(method)


def ap_from_dets(scores, tp, n_gt, method="coco"):
    """AP given pooled detections (already sorted desc by score) and tp flags."""
    if n_gt == 0:
        return float("nan")  # class absent
    if len(scores) == 0:
        return 0.0
    tpc = np.cumsum(tp)
    fpc = np.cumsum(~tp)
    rec = tpc / n_gt
    prec = tpc / (tpc + fpc)
    return average_precision(rec, prec, method=method)


def compute_ap_table(records, thresholds, agnostic=False, classes=None,
                     pred_cls_keep=None, gt_cls_keep=None, method="coco"):
    """Compute per-class AP at each IoU threshold from image records.

    records: list of (dets_array, gts_array) per image.
    classes: iterable of class ids to compute (default: those present in GT).
    Returns: ap[class][thr], n_gt[class]
    """
    if classes is None:
        present = set()
        for _, g in records:
            if g is not None and len(g):
                present.update(int(c) for c in np.asarray(g)[:, 0])
        if gt_cls_keep is not None:
            present &= set(gt_cls_keep)
        classes = sorted(present)

    # pool per class
    pooled = {c: {"scores": [], "tp": {t: [] for t in thresholds}, "n_gt": 0} for c in classes}
    for d, g in records:
        for c in classes:
            dc = None
            if agnostic:
                # all dets count for the single collapsed class (c==0 expected)
                dc = np.asarray(d) if d is not None and len(d) else np.zeros((0, 6))
                if pred_cls_keep is not None and len(dc):
                    dc = dc[np.isin(dc[:, 0].astype(int), list(pred_cls_keep))]
            else:
                if d is not None and len(d):
                    dd = np.asarray(d)
                    dc = dd[dd[:, 0].astype(int) == c]
                else:
                    dc = np.zeros((0, 6))
            gc = None
            if g is not None and len(g):
                gg = np.asarray(g)
                if gt_cls_keep is not None:
                    gg = gg[np.isin(gg[:, 0].astype(int), list(gt_cls_keep))]
                if agnostic:
                    gc = gg
                else:
                    gc = gg[gg[:, 0].astype(int) == c]
            else:
                gc = np.zeros((0, 5))
            pooled[c]["n_gt"] += len(gc)
            if len(dc) == 0:
                continue
            order = np.argsort(-dc[:, 1], kind="stable")
            scores = dc[order, 1]
            pooled[c]["scores"].append(scores)
            boxes = dc[order, 2:6]
            gboxes = gc[:, 1:5] if len(gc) else np.zeros((0, 4))
            for t in thresholds:
                tp = np.zeros(len(dc), dtype=bool)
                if len(gboxes):
                    ious = iou_xyxy(boxes, gboxes)
                    matched = np.zeros(len(gboxes), dtype=bool)
                    for i in range(len(boxes)):
                        if ious.shape[1] == 0:
                            break
                        j = int(np.argmax(ious[i]))
                        if not matched[j] and ious[i, j] >= t:
                            matched[j] = True
                            tp[i] = True
                pooled[c]["tp"][t].append(tp)

    ap = {}
    n_gt = {}
    for c in classes:
        n_gt[c] = pooled[c]["n_gt"]
        ap[c] = {}
        if not pooled[c]["scores"]:
            for t in thresholds:
                ap[c][float(t)] = 0.0 if pooled[c]["n_gt"] > 0 else float("nan")
            continue
        for t in thresholds:
            scores = np.concatenate(pooled[c]["scores"])
            tp = np.concatenate(pooled[c]["tp"][t])
            order = np.argsort(-scores, kind="stable")
            ap[c][float(t)] = ap_from_dets(scores[order], tp[order], pooled[c]["n_gt"], method=method)
    return ap, n_gt


# ------------------------------------------------------- P/R/F1 (max-F1 conf)

def pooled_prf(records, n_conf=1000, agnostic=False, classes=None,
               pred_cls_keep=None, gt_cls_keep=None):
    """Ultralytics-style mean P/R/F1 at the conf maximising mean F1.

    Returns dict with mp, mr, f1, best_conf, and per-class P/R/F1/AP arrays
    at that conf.
    """
    if classes is None:
        present = set()
        for _, g in records:
            if g is not None and len(g):
                gg = np.asarray(g)
                if gt_cls_keep is not None:
                    gg = gg[np.isin(gg[:, 0].astype(int), list(gt_cls_keep))]
                if agnostic:
                    present.add(0)
                else:
                    present.update(int(c) for c in gg[:, 0])
        classes = sorted(present) if present else [0]

    px = np.linspace(0, 1, n_conf)
    p_curve = np.zeros((len(classes), n_conf))
    r_curve = np.zeros((len(classes), n_conf))
    for ci, c in enumerate(classes):
        scores_all, tp_all, n_gt = _pool_class(records, c, thr=0.5, agnostic=agnostic,
                                               pred_cls_keep=pred_cls_keep, gt_cls_keep=gt_cls_keep)
        if len(scores_all) == 0:
            continue
        order = np.argsort(-scores_all, kind="stable")
        tp_sorted = tp_all[order]
        scores_sorted = scores_all[order]
        tpc = np.cumsum(tp_sorted)
        fpc = np.cumsum(~tp_sorted)
        rec = tpc / max(n_gt, 1)
        prec = tpc / (tpc + fpc)
        # interpolate onto conf grid, ultralytics-style:
        # at conf x keep dets with score >= x; precision left-bound is 1.0
        idx = np.searchsorted(-scores_sorted, -px, side="right")  # #dets with score >= x
        idx = np.clip(idx, 0, len(scores_sorted))
        r_curve[ci] = np.where(idx > 0, rec[np.clip(idx - 1, 0, len(rec) - 1)], 0.0)
        p_curve[ci] = np.where(idx > 0, prec[np.clip(idx - 1, 0, len(prec) - 1)], 1.0)

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-8)
    i = int(f1_curve.mean(0).argmax())
    mp = float(p_curve[:, i].mean())
    mr = float(r_curve[:, i].mean())
    f1 = 2 * mp * mr / (mp + mr + 1e-8)
    return {
        "mp": mp, "mr": mr, "f1": f1, "best_conf": float(px[i]),
        "p_per_class": p_curve[:, i].tolist(),
        "r_per_class": r_curve[:, i].tolist(),
        "classes": list(classes),
    }


def _pool_class(records, cls, thr, agnostic=False, pred_cls_keep=None, gt_cls_keep=None):
    """Pool detections for one class (or collapsed class) at a single IoU thr."""
    scores, tps = [], []
    n_gt = 0
    for d, g in records:
        d = np.asarray(d) if d is not None and len(d) else np.zeros((0, 6))
        g = np.asarray(g) if g is not None and len(g) else np.zeros((0, 5))
        if pred_cls_keep is not None and len(d):
            d = d[np.isin(d[:, 0].astype(int), list(pred_cls_keep))]
        if gt_cls_keep is not None and len(g):
            g = g[np.isin(g[:, 0].astype(int), list(gt_cls_keep))]
        if agnostic:
            dc = d
            gc = g
        else:
            dc = d[d[:, 0].astype(int) == cls] if len(d) else d
            gc = g[g[:, 0].astype(int) == cls] if len(g) else g
        n_gt += len(gc)
        if len(dc) == 0:
            continue
        s, tp, _ = match_image(dc, gc, thr, agnostic=False)
        scores.append(s)
        tps.append(tp)
    if not scores:
        return np.zeros(0), np.zeros(0, dtype=bool), n_gt
    return np.concatenate(scores), np.concatenate(tps), n_gt


# ------------------------------------------------------------------ cache IO

def load_cache(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def records_to_arrays(recs, collapse=False, nms_iou=0.7):
    """Convert cache records to list of (dets (N,6), gts (M,5)) arrays.

    If collapse=True, detections are re-NMS'd across classes and both det and
    GT class ids are set to 0 (single_cls semantics).
    """
    out = []
    for r in recs:
        d = np.asarray(r["dets"], dtype=np.float64).reshape(-1, 6)
        g = np.asarray(r["gts"], dtype=np.float64).reshape(-1, 5)
        if collapse:
            d = collapse_single_class(d, nms_iou)
            if len(g):
                g = g.copy()
                g[:, 0] = 0
        out.append((d, g))
    return out
