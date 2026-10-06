"""Phase 3 aggregation -> results_v3/{attention_alternatives,preprocessing_controls,
external_reference,transfer_controls}.csv. All numbers read from evaluation CSVs.

Sources (results_v3/audit/): matched_* (Phase 2 battery), p3_* (Phase 3 battery),
gc10_restricted_*.csv; (results_v3/phase3/): gradcam_quant.csv,
transfer_multiscale.csv, transfer_tiling.csv, transfer_perclass.csv.
"""

import csv
import json
from pathlib import Path

from iama_env import AUDIT_DIR, PROJECT_ROOT

OUT = PROJECT_ROOT / "results_v3"
P3 = OUT / "phase3"


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dump(path, rows):
    if not rows:
        print(f"WARN no rows for {path}")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("saved", path)


def main():
    val2 = read_csv(AUDIT_DIR / "matched_val_point_estimates.csv")
    val3 = read_csv(AUDIT_DIR / "p3_val_point_estimates.csv")
    boot2 = read_csv(AUDIT_DIR / "matched_bootstrap_ci.csv")
    boot3 = read_csv(AUDIT_DIR / "p3_bootstrap_ci.csv")
    pc2 = read_csv(AUDIT_DIR / "matched_per_class_ap.csv")
    pc3 = read_csv(AUDIT_DIR / "p3_per_class_ap.csv")
    fps2 = read_csv(AUDIT_DIR / "matched_fps_benchmark_summary.csv")
    fps3 = read_csv(AUDIT_DIR / "p3_fps_benchmark_summary.csv")
    paired = (read_csv(AUDIT_DIR / "matched_paired_bootstrap.csv")
              + read_csv(AUDIT_DIR / "matched_pM2_paired_bootstrap.csv")
              + read_csv(AUDIT_DIR / "matched_pM3_paired_bootstrap.csv"))
    for tag in ("se", "cbam", "ca", "v8", "gray", "nobil"):
        paired += read_csv(AUDIT_DIR / f"p3_{tag}_paired_bootstrap.csv")

    val = val2 + val3
    boot = boot2 + boot3
    percls = pc2 + pc3
    fps = fps2 + fps3

    def v(model, ds, variant, scope, metric):
        for r in val:
            if (r["model"], r["dataset"], r["variant"], r["scope"], r["metric"]) == \
               (model, ds, variant, scope, metric):
                return r["value"]
        return ""

    def b(model, ds, variant, scope, metric):
        for r in boot:
            if (r["model"], r["dataset"], r["variant"], r["scope"], r["metric"], r.get("cls", "")) == \
               (model, ds, variant, scope, metric, ""):
                return r
        return {}

    def bc(model, ds, variant, scope, cls):
        for r in boot:
            if (r["model"], r["dataset"], r["variant"], r["scope"], r["metric"], r.get("cls", "")) == \
               (model, ds, variant, scope, "AP50_class", cls):
                return r
        return {}

    def pcl(model, ds, variant, scope, cls):
        for r in percls:
            if (r["model"], r["dataset"], r["variant"], r["scope"], r["class"]) == \
               (model, ds, variant, scope, cls):
                return r["AP50"]
        return ""

    def fp(model, prec="fp32", src="real_neu_test_image"):
        for r in fps:
            if (r["model"], r["precision"], r["source"]) == (model, prec, src):
                return r["fps_mean"], r["fps_std"], r["inf_ms_mean"]
        return "", "", ""

    def pair(bmodel, ds, metric, label_contains):
        for r in paired:
            if r["B_model"] == bmodel and r["dataset"] == ds and r["metric"] == metric \
               and label_contains.lower() in r["label"].lower():
                return r
        return {}

    # param counts
    def params(ckpt):
        try:
            import torch
            import sys
            sys.path.insert(0, str(PROJECT_ROOT / "scripts_v3"))
            from iama_env import register_iama_modules
            register_iama_modules()
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            m = ck["model"].float() if hasattr(ck.get("model"), "parameters") else None
            if m is None:
                from ultralytics import YOLO
                m = YOLO(str(ckpt)).model
            return sum(p.numel() for p in m.parameters())
        except Exception as e:  # noqa: BLE001
            return f"ERR:{e}"

    neu_classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

    # --------------------------------------- attention_alternatives.csv
    rows = []
    attn_models = [("M1m", "none (baseline)", "raw", "runs_v3/M1_matched_seed0/weights/best.pt"),
                   ("M3m", "ECA+spatial (IAMA)", "raw", "runs_v3/M3_matched_seed0/weights/best.pt"),
                   ("M3se", "SE", "raw", "runs_v3/M3se_matched_seed0/weights/best.pt"),
                   ("M3cbam", "CBAM", "raw", "runs_v3/M3cbam_matched_seed0/weights/best.pt")]
    for m, label, variant, ck in attn_models:
        b50 = b(m, "neu", variant, "all", "mAP@0.5")
        b95 = b(m, "neu", variant, "all", "mAP@0.5:0.95")
        g50 = b(m, "gc10", "raw", "all", "mAP@0.5")
        f32 = fp(m)
        row = {"model_id": m, "attention": label, "seed": 0, "epochs": 100,
               "params_total": params(PROJECT_ROOT / ck),
               "neu_mAP50": v(m, "neu", variant, "test", "mAP@0.5"),
               "neu_mAP50_ci": f"[{b50.get('ci95_low','')}, {b50.get('ci95_high','')}]",
               "neu_mAP5095": v(m, "neu", variant, "test", "mAP@0.5:0.95"),
               "neu_mAP5095_ci": f"[{b95.get('ci95_low','')}, {b95.get('ci95_high','')}]",
               "neu_P": v(m, "neu", variant, "test", "precision_mean"),
               "neu_R": v(m, "neu", variant, "test", "recall_mean"),
               "gc10_all_mAP50": v(m, "gc10", "raw", "all", "mAP@0.5"),
               "gc10_all_mAP50_ci": f"[{g50.get('ci95_low','')}, {g50.get('ci95_high','')}]",
               "fps_fp32": f32[0], "fps_fp32_std": f32[1], "inf_ms_fp32": f32[2]}
        for cname in neu_classes:
            row[f"AP50_{cname}"] = pcl(m, "neu", variant, "test", cname)
        if m != "M3m" and m != "M1m":
            pr = pair(m, "neu", "mAP@0.5", "NEU")
            pg = pair(m, "gc10", "mAP@0.5", "all2300")
            row["delta_neu_mAP50_vs_M3m_pp"] = pr.get("delta_point_pp", "")
            row["delta_neu_ci_pp"] = f"[{pr.get('delta_ci95_low_pp','')}, {pr.get('delta_ci95_high_pp','')}]"
            row["delta_neu_p"] = pr.get("p_two_sided", "")
            row["delta_gc10_mAP50_vs_M3m_pp"] = pg.get("delta_point_pp", "")
            row["delta_gc10_p"] = pg.get("p_two_sided", "")
        rows.append(row)
    dump(OUT / "attention_alternatives.csv", rows)

    # --------------------------------------- preprocessing_controls.csv
    rows = []
    pp_models = [("M1m", "none (raw)", "raw", "runs_v3/M1_matched_seed0/weights/best.pt"),
                 ("M2m", "LAB CLAHE + bilateral d5 s50 (original)", "pp", "runs_v3/M2_matched_seed0/weights/best.pt"),
                 ("M2gray", "grayscale CLAHE, no bilateral", "gray", "runs_v3/M2gray_matched_seed0/weights/best.pt")]
    for m, label, variant, ck in pp_models:
        b50 = b(m, "neu", variant, "all", "mAP@0.5")
        b95 = b(m, "neu", variant, "all", "mAP@0.5:0.95")
        craz = bc(m, "neu", variant, "all", "crazing")
        row = {"model_id": m, "preprocessing": label, "seed": 0, "epochs": 100,
               "params_total": params(PROJECT_ROOT / ck) if "M2" in m else params(PROJECT_ROOT / ck),
               "neu_mAP50": v(m, "neu", variant, "test", "mAP@0.5"),
               "neu_mAP50_ci": f"[{b50.get('ci95_low','')}, {b50.get('ci95_high','')}]",
               "neu_mAP5095": v(m, "neu", variant, "test", "mAP@0.5:0.95"),
               "neu_mAP5095_ci": f"[{b95.get('ci95_low','')}, {b95.get('ci95_high','')}]",
               "neu_P": v(m, "neu", variant, "test", "precision_mean"),
               "neu_R": v(m, "neu", variant, "test", "recall_mean"),
               "AP50_crazing": pcl(m, "neu", variant, "test", "crazing"),
               "AP50_crazing_ci": f"[{craz.get('ci95_low','')}, {craz.get('ci95_high','')}]",
               "gc10_all_mAP50_raw_inputs": v(m, "gc10", "raw", "all", "mAP@0.5")}
        for cname in neu_classes:
            row[f"AP50_{cname}"] = pcl(m, "neu", variant, "test", cname)
        if m in ("M2gray", "M2nobil"):
            pr = pair(m, "neu", "mAP@0.5", "NEU")
            row["delta_neu_mAP50_vs_M2m_pp"] = pr.get("delta_point_pp", "")
            row["delta_neu_ci_pp"] = f"[{pr.get('delta_ci95_low_pp','')}, {pr.get('delta_ci95_high_pp','')}]"
            row["delta_neu_p"] = pr.get("p_two_sided", "")
        rows.append(row)
    dump(OUT / "preprocessing_controls.csv", rows)

    # --------------------------------------- external_reference.csv
    # TRIMMED 2026-10-05 per user (YOLOv8s reference dropped for time).
    # Kept as no-op so downstream tooling does not break.

    # --------------------------------------- transfer_controls.csv
    rows = []
    for r in read_csv(P3 / "transfer_multiscale.csv"):
        rows.append({"control": "multiscale_agnostic_all2300", "model": r["model"],
                     "setting": f"imgsz={r['imgsz']}", "metric": "mAP@0.5",
                     "value": r["mAP@0.5"], "mAP5095": r["mAP@0.5:0.95"],
                     "P": r["precision"], "R": r["recall"]})
    for r in read_csv(P3 / "transfer_tiling.csv"):
        if r["metric"] == "mAP@0.5":
            rows.append({"control": "tiling_640_defect_scale_matched", "model": r["model"],
                         "setting": r["protocol"], "metric": "mAP@0.5", "value": r["point"],
                         "ci95": f"[{r['ci95_low']}, {r['ci95_high']}]"})
    for r in read_csv(P3 / "transfer_perclass.csv"):
        if r["scope"] == "all2300":
            rows.append({"control": "per_gc10_class_agnostic", "model": r["model"],
                         "setting": r["gc10_class"], "metric": "AP50 + recall@0.25",
                         "value": r["AP50_agnostic"], "n_gt": r["n_gt"],
                         "recall_at_0.25": r["recall_at_0.25"]})
    # preprocessing at test time (per model): raw vs pp GC10 + NEU
    for m in ("M1m", "M2m", "M3m", "M4m"):
        for ds, variants in [("gc10", ("raw", "pp")), ("neu", ("raw", "pp"))]:
            vals = {}
            for var in variants:
                vv = v(m, ds, var, "all" if ds == "gc10" else "test", "mAP@0.5")
                vals[var] = vv
            rows.append({"control": "preprocessing_at_test_time", "model": m,
                         "setting": f"{ds}: raw vs preprocessed", "metric": "mAP@0.5",
                         "value": f"raw={vals['raw']} pp={vals['pp']}"})
    dump(OUT / "transfer_controls.csv", rows)

    # gradcam is already a clean CSV; copy reference
    print("gradcam_quant.csv rows:", len(read_csv(P3 / "gradcam_quant.csv")))


if __name__ == "__main__":
    main()
