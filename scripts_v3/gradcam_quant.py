"""Phase 3.5: quantified Grad-CAM on the WHOLE NEU-DET test split.

Replaces the old 3-hand-picked-image Grad-CAM gallery with two standard
quantitative metrics, computed at P3/P4/P5 separately and fused:

  * pointing game: does the heatmap argmax fall inside ANY ground-truth box?
    Reported unconditionally (all test images with GT) and conditionally
    (only images where the model has >=1 TP detection at conf>=0.25, IoU>=0.5
    — the standard pointing-game setting).
  * energy-in-GT: fraction of ReLU-CAM mass inside GT boxes.

Method: gradient-weighted CAM (Selvaraju et al.) with backward target =
highest anchor class score of the raw detection head (pre-NMS). Heatmaps are
computed in 640-letterbox space, mapped back to the original 200x200 image,
and compared against GT boxes in original coordinates.

95% CIs: 1000 image-level bootstrap resamples (seed 20261005).

Models: original M1/M4 (old report claims) + matched M1m/M3m/M4m.

Output: results_v3/phase3/gradcam_quant.csv
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from iama_env import register_iama_modules, CHECKPOINTS, PROJECT_ROOT, NEU_RAW

register_iama_modules()

from ultralytics import YOLO  # noqa: E402
import metrics_lib as ML  # noqa: E402

OUT = PROJECT_ROOT / "results_v3" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)
IMGSZ = 640
B = 1000
SEED = 20261005
CONF_TP = 0.25

MODELS = {
    "M1m": PROJECT_ROOT / "runs_v3" / "M1_matched_seed0" / "weights" / "best.pt",
    "M3m": PROJECT_ROOT / "runs_v3" / "M3_matched_seed0" / "weights" / "best.pt",
    "M4m": PROJECT_ROOT / "runs_v3" / "M4_matched_seed0" / "weights" / "best.pt",
}


def layer_indices(n_layers):
    """P3/P4/P5 output layer indices for baseline (24) vs attention (27) nets."""
    if n_layers == 24:
        return [16, 19, 22]
    if n_layers == 27:
        return [19, 22, 25]
    raise ValueError(f"unexpected layer count {n_layers}")


def letterbox(img, new=IMGSZ, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new - nh) // 2, (new - nw) // 2
    out = np.full((new, new, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = res
    return out, r, top, left


def load_test_images():
    root = NEU_RAW
    items = []
    lab_dir = root / "labels" / "test"
    for p in sorted((root / "images" / "test").glob("*.jpg")):
        lp = lab_dir / f"{p.stem}.txt"
        gts = []
        if lp.exists():
            for line in lp.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    c = int(parts[0])
                    cx, cy, w, h = (float(v) for v in parts[1:5])
                    gts.append([c, (cx - w / 2) * 200, (cy - h / 2) * 200,
                                (cx + w / 2) * 200, (cy + h / 2) * 200])
        items.append((p, np.asarray(gts, dtype=np.float64).reshape(-1, 5)))
    return items


def has_tp(det_boxes, gts, thr=0.5):
    """Whether any det at conf>=CONF_TP matches a GT at IoU>=thr."""
    if len(det_boxes) == 0 or len(gts) == 0:
        return False
    ious = ML.iou_xyxy(det_boxes[:, 2:6], gts[:, 1:5])
    matched = np.zeros(len(gts), bool)
    order = np.argsort(-det_boxes[:, 1])
    for i in order:
        if det_boxes[i, 1] < CONF_TP:
            break
        if len(gts) == 0:
            break
        j = int(np.argmax(ious[i]))
        if not matched[j] and ious[i, j] >= thr:
            return True
    return False


def cam_for_image(model, layers, img_bgr, device):
    """Return dict scale->heatmap (H,W float32 in ORIGINAL image coords) and det array."""
    # ultralytics predict() disables grad on parameters during inference setup;
    # re-enable before every gradient pass
    torch.set_grad_enabled(True)
    for p in model.model.parameters():
        p.requires_grad_(True)
    lb, r, top, left = letterbox(img_bgr)
    x = torch.from_numpy(lb[:, :, ::-1].transpose(2, 0, 1).copy()).float().to(device) / 255.0
    x = x.unsqueeze(0).requires_grad_(False)

    acts, grads = {}, {}
    handles = []

    def make_hooks(idx):
        def fwd(mod, inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            acts[idx] = t
            t.retain_grad()
        return fwd

    for idx in layers:
        handles.append(model.model.model[idx].register_forward_hook(make_hooks(idx)))

    # capture raw head output via hook on Detect (last layer)
    head_out = {}

    def head_hook(mod, inp, out):
        head_out["y"] = out
    handles.append(model.model.model[-1].register_forward_hook(head_hook))

    y = model.model(x)
    for h in handles:
        h.remove()

    # target: max class score over anchors from the Detect output
    ho = head_out.get("y", y)
    if isinstance(ho, (tuple, list)):
        ho = ho[0]
    if ho.dim() == 3:
        # possible (B, 4+nc, A) or (B, A, 4+nc)
        if ho.shape[1] == 10:  # (B, 4+nc, A) for nc=6
            scores = ho[0, 4:, :].transpose(0, 1)
        else:
            scores = ho[0, :, 4:]
    else:
        raise RuntimeError(f"unexpected head output shape {ho.shape}")
    conf = scores.max(dim=1).values
    target = conf.max()
    model.zero_grad(set_to_none=True)
    target.backward(retain_graph=False)

    H0, W0 = img_bgr.shape[:2]
    cams = {}
    for idx in layers:
        A = acts[idx].detach()[0]          # (C,h,w)
        G = acts[idx].grad[0]              # (C,h,w)
        alpha = G.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((alpha * A).sum(dim=0)).cpu().numpy()
        cam = cv2.resize(cam, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        # undo letterbox: crop padding, resize to original
        nh, nw = int(round(H0 * r)), int(round(W0 * r))
        cam = cam[top:top + nh, left:left + nw]
        cam = cv2.resize(cam, (W0, H0), interpolation=cv2.INTER_LINEAR)
        s = cam.sum()
        cams[idx] = cam / s if s > 0 else cam
    # fused: mean of per-scale normalized cams, renormalised
    fused = np.mean([cams[i] for i in layers], axis=0)
    s = fused.sum()
    cams["fused"] = fused / s if s > 0 else fused

    # detections for conditional pointing game (reuse predict for correctness)
    return cams


def predict_boxes(model, img_path, device):
    r = model.predict(str(img_path), imgsz=IMGSZ, device=device, conf=CONF_TP,
                      iou=0.7, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 6))
    xyxy = r.boxes.xyxy.cpu().numpy()
    cf = r.boxes.conf.cpu().numpy()
    cl = r.boxes.cls.cpu().numpy()
    return np.column_stack([cl, cf, xyxy])


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"   # for torch .to()
    pred_device = "0" if torch.cuda.is_available() else "cpu"   # for ultralytics predict
    items = load_test_images()
    rows = []
    for name, wpath in MODELS.items():
        if not Path(wpath).exists():
            print(f"SKIP {name}: {wpath} missing (training not finished?)")
            continue
        t0 = time.time()
        model = YOLO(str(wpath))
        model.model.to(device).eval()
        # fused checkpoint weights load with requires_grad=False; gradients are
        # needed for Grad-CAM activations
        for p in model.model.parameters():
            p.requires_grad_(True)
        layers = layer_indices(len(model.model.model))
        per_img = {k: {"pg_hit": [], "pg_cond_hit": [], "energy": []} for k in layers + ["fused"]}
        cond_mask = []
        for ii, (p, gts) in enumerate(items):
            img = cv2.imread(str(p))
            # CAM forward FIRST: model.predict() runs under torch.inference_mode
            # and caches inference tensors (Detect anchors) that cannot be used
            # in later autograd-tracked forwards
            cams = cam_for_image(model, layers, img, device)
            dets = predict_boxes(model, p, pred_device)
            cond = has_tp(dets, gts) if len(gts) else False
            cond_mask.append(cond)
            for key, cam in cams.items():
                # pointing game (argmax inside any GT?)
                yy, xx = np.unravel_index(int(np.argmax(cam)), cam.shape)
                hit = 0.0
                energy = np.nan
                if len(gts):
                    inside = ((gts[:, 1] <= xx) & (xx <= gts[:, 3]) &
                              (gts[:, 2] <= yy) & (yy <= gts[:, 4]))
                    hit = float(inside.any())
                    mask = np.zeros_like(cam, dtype=bool)
                    for g in gts:
                        x1, y1, x2, y2 = (int(max(0, g[1])), int(max(0, g[2])),
                                          int(min(cam.shape[1], g[3])), int(min(cam.shape[0], g[4])))
                        mask[y1:y2, x1:x2] = True
                    tot = cam.sum()
                    energy = float(cam[mask].sum() / tot) if tot > 0 else np.nan
                per_img[key]["pg_hit"].append(hit if len(gts) else np.nan)
                per_img[key]["pg_cond_hit"].append(hit if (len(gts) and cond) else np.nan)
                per_img[key]["energy"].append(energy)
            if (ii + 1) % 50 == 0:
                print(f"  {name}: {ii+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

        cond_arr = np.asarray(cond_mask)
        n_cond = int(cond_arr.sum())
        rng = np.random.default_rng(SEED)
        for key in layers + ["fused"]:
            scale_name = {layers[0]: "P3", layers[1]: "P4", layers[2]: "P5"}.get(key, "fused")
            pg = np.asarray(per_img[key]["pg_hit"], dtype=float)
            pgc = np.asarray(per_img[key]["pg_cond_hit"], dtype=float)
            en = np.asarray(per_img[key]["energy"], dtype=float)
            valid_pg = pg[~np.isnan(pg)]
            valid_en = en[~np.isnan(en)]
            valid_pgc = pgc[~np.isnan(pgc)]
            # bootstrap CIs
            def boot_ci(vals):
                if len(vals) == 0:
                    return (np.nan, np.nan, np.nan)
                means = []
                for _ in range(B):
                    idx = rng.integers(0, len(vals), len(vals))
                    means.append(vals[idx].mean())
                means = np.asarray(means)
                return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
            pg_m, pg_lo, pg_hi = boot_ci(valid_pg)
            pgc_m, pgc_lo, pgc_hi = boot_ci(valid_pgc)
            en_m, en_lo, en_hi = boot_ci(valid_en)
            rows.append({
                "model": name, "scale": scale_name, "n_images": len(items), "n_conditional": n_cond,
                "pointing_game_all": round(pg_m, 4), "pg_all_ci_low": round(pg_lo, 4), "pg_all_ci_high": round(pg_hi, 4),
                "pointing_game_cond": round(pgc_m, 4) if pgc_m == pgc_m else "",
                "pg_cond_ci_low": round(pgc_lo, 4) if pgc_lo == pgc_lo else "",
                "pg_cond_ci_high": round(pgc_hi, 4) if pgc_hi == pgc_hi else "",
                "energy_in_gt": round(en_m, 4), "energy_ci_low": round(en_lo, 4), "energy_ci_high": round(en_hi, 4),
                "protocol": f"target=max anchor class score (pre-NMS); conf_tp={CONF_TP}; B={B}; seed={SEED}",
            })
            print(f"{name}/{scale_name}: PG_all={pg_m:.3f} PG_cond={pgc_m if pgc_m==pgc_m else float('nan'):.3f} "
                  f"energy={en_m:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    with open(OUT / "gradcam_quant.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", OUT / "gradcam_quant.csv")


if __name__ == "__main__":
    main()

