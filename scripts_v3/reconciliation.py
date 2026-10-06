"""Phase 1.1: reconciliation of OLD reported numbers vs NEW recomputed ones.

Sources of OLD numbers (read-only):
  results/ablation_table.csv        (master NEU-DET test table)
  results/eval_results.csv          (includes the identity-mapped ANCHOR row)
  results/cross_dataset_results.csv (GC10 test, class-agnostic)
  results/IAMA_Net_Academic_Research_Report.md section 4.2 (class-wise table,
    quoted verbatim below as REPORT_CLASSWISE)

Sources of NEW numbers:
  results_v3/audit/val_point_estimates.csv   (ultralytics val, batch 4)
  results_v3/audit/own_point_estimates.csv   (metrics_lib implementation)
  results_v3/audit/per_class_ap.csv
  results_v3/audit/param_counts.csv

Outputs:
  results_v3/audit/reconciliation_neu.csv
  results_v3/audit/reconciliation_gc10.csv
  results_v3/audit/reconciliation_classwise.csv
  results_v3/audit/reconciliation_summary.md
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

from iama_env import AUDIT_DIR, PROJECT_ROOT

OLD_RESULTS = PROJECT_ROOT / "results"

# Quoted verbatim from results/IAMA_Net_Academic_Research_Report.md, section
# 4.2 "Class-Wise Performance Breakdown" (lines 274-280), mAP@0.5 per class.
REPORT_CLASSWISE = {
    # class:            M1     M2     M3     M4
    "crazing":        [0.490, 0.412, 0.434, 0.386],
    "inclusion":      [0.781, 0.866, 0.878, 0.865],
    "patches":        [0.944, 0.791, 0.813, 0.812],
    "pitted_surface": [0.803, 0.814, 0.831, 0.813],
    "rolled-in_scale":[0.658, 0.602, 0.583, 0.569],
    "scratches":      [0.889, 0.918, 0.926, 0.931],
}
REPORT_CLASS_ORDER = ["M1", "M2", "M3", "M4"]
REPORT_MASTER_MAP50 = {"M1": 0.7527, "M2": 0.7444, "M3": 0.7484, "M4": 0.7333,
                       "ANCHOR": 0.7608}  # ANCHOR row from eval_results.csv / report 4.1


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    lines = []

    # ------------------------------------------------------------ load new
    val_rows = read_csv(AUDIT_DIR / "val_point_estimates.csv")
    own_rows = read_csv(AUDIT_DIR / "own_point_estimates.csv")
    percls_rows = read_csv(AUDIT_DIR / "per_class_ap.csv")

    val = defaultdict(dict)   # (model,dataset,variant,scope) -> metric -> value
    for r in val_rows:
        val[(r["model"], r["dataset"], r["variant"], r["scope"])][r["metric"]] = f(r["value"])
    own = defaultdict(dict)   # (model,dataset,variant,scope) -> metric -> value
    for r in own_rows:
        own[(r["model"], r["dataset"], r["variant"], r["scope"])][r["metric"]] = f(r["value"])
    percls = {}               # (model,dataset,variant,scope,class) -> AP50
    for r in percls_rows:
        percls[(r["model"], r["dataset"], r["variant"], r["scope"], r["class"])] = f(r["AP50"])

    # ------------------------------------------------------------ load old
    old_abl = {r["Config"]: r for r in read_csv(OLD_RESULTS / "ablation_table.csv")}
    old_eval = {Path(r["weights"]).stem: r for r in read_csv(OLD_RESULTS / "eval_results.csv")}
    old_gc10 = {r["Config"]: r for r in read_csv(OLD_RESULTS / "cross_dataset_results.csv")}

    # ------------------------------------------------- NEU reconciliation
    neu_rows = []
    primary_variant = {"M1": "raw", "M2": "pp", "M3": "raw", "M4": "pp", "ANCHOR": "raw"}
    for model, variant in primary_variant.items():
        key_v = (model, "neu", variant, "test")
        key_o = (model, "neu", variant, "test")
        old_map50 = old_eval.get("best" if model != "ANCHOR" else "m3_aligned_init_seed42", {}).get("mAP@0.5")
        if model == "ANCHOR":
            for stem, r in old_eval.items():
                if "m3_aligned_init" in stem:
                    old_map50 = r["mAP@0.5"]
        if model in old_abl:
            old_map50 = old_abl[model]["mAP@0.5"]
            old_p, old_r, old_f1, old_fps = (old_abl[model]["Precision"], old_abl[model]["Recall"],
                                             old_abl[model]["F1"], old_abl[model]["FPS"])
        else:
            row = old_eval.get("m3_aligned_init_seed42", {})
            old_p, old_r, old_f1, old_fps = row.get("Precision"), row.get("Recall"), row.get("F1"), row.get("FPS")
        new_v_map50 = val[key_v].get("mAP@0.5")
        new_v_map = val[key_v].get("mAP@0.5:0.95")
        new_v_p = val[key_v].get("precision_mean")
        new_v_r = val[key_v].get("recall_mean")
        new_o = own.get(key_o, {})
        for metric, old, newv, newo in [
            ("mAP@0.5", f(old_map50), new_v_map50, new_o.get("mAP@0.5")),
            ("mAP@0.5:0.95", None, new_v_map, new_o.get("mAP@0.5:0.95")),
            ("precision", f(old_p), new_v_p, new_o.get("precision_mean")),
            ("recall", f(old_r), new_v_r, new_o.get("recall_mean")),
            ("F1(derived from mean P,R)", f(old_f1), None, new_o.get("f1_mean")),
        ]:
            delta = None
            if old is not None and newv is not None:
                delta = round(newv - old, 4)
            neu_rows.append({
                "model": model, "input_variant": variant, "metric": metric,
                "old_reported": old, "new_ultralytics_val": newv,
                "new_metrics_lib": newo, "delta_new_minus_old": delta,
                "old_source": "results/ablation_table.csv" if model in old_abl else "results/eval_results.csv",
                "new_source": "results_v3/audit/val_point_estimates.csv | own_point_estimates.csv",
            })
    with open(AUDIT_DIR / "reconciliation_neu.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(neu_rows[0].keys()))
        w.writeheader()
        w.writerows(neu_rows)

    # ------------------------------------------------- GC10 reconciliation
    gc_rows = []
    for model in ["M1", "M2", "M3", "M4"]:
        old = old_gc10.get(model, {})
        key_v = (model, "gc10", "raw", "test")
        key_test = (model, "gc10", "raw", "test")
        key_all = (model, "gc10", "raw", "all")
        new_o_all = own.get(key_all, {})
        for metric, oldk, newk in [("mAP@0.5", "mAP@0.5", "mAP@0.5"),
                                   ("precision", "Precision", "precision_mean"),
                                   ("recall", "Recall", "recall_mean"),
                                   ("F1", "F1", "f1_mean")]:
            ov = f(old.get(oldk))
            nv_test = val[key_v].get(newk if newk != "f1_mean" else "mAP@0.5")  # val has no F1
            if metric == "F1":
                nv_test = None
            nv_all = new_o_all.get(newk)
            gc_rows.append({
                "model": model, "metric": metric,
                "old_reported_test230": ov,
                "new_val_test230_single_cls": nv_test,
                "new_own_test230": own.get(key_test, {}).get(newk),
                "new_own_all2300": nv_all,
                "note": "old protocol = 230 test images, class-agnostic; new primary = ALL 2300 images",
            })
    with open(AUDIT_DIR / "reconciliation_gc10.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gc_rows[0].keys()))
        w.writeheader()
        w.writerows(gc_rows)

    # --------------------------------------- class-wise M1 vs ANCHOR check
    cls_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    cw_rows = []
    for ci, cname in enumerate(cls_names):
        row = {"class": cname}
        for mi, model in enumerate(REPORT_CLASS_ORDER):
            row[f"report_{model}"] = REPORT_CLASSWISE[cname][mi]
        for model in ["M1", "ANCHOR", "M2", "M3", "M4"]:
            variant = primary_variant[model]
            v = percls.get((model, "neu", variant, "test", cname))
            row[f"recomputed_{model}"] = v
        cw_rows.append(row)
    # means
    mean_row = {"class": "MEAN"}
    for mi, model in enumerate(REPORT_CLASS_ORDER):
        vals = [REPORT_CLASSWISE[c][mi] for c in cls_names]
        mean_row[f"report_{model}"] = round(sum(vals) / len(vals), 4)
    for model in ["M1", "ANCHOR", "M2", "M3", "M4"]:
        variant = primary_variant[model]
        vals = [percls.get((model, "neu", variant, "test", c)) for c in cls_names]
        vals = [v for v in vals if v is not None]
        mean_row[f"recomputed_{model}"] = round(sum(vals) / len(vals), 4) if vals else None
    cw_rows.append(mean_row)
    with open(AUDIT_DIR / "reconciliation_classwise.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cw_rows[0].keys()))
        w.writeheader()
        w.writerows(cw_rows)

    # --------------------------------------------- which ckpt matches M1 col?
    m1_report = [REPORT_CLASSWISE[c][0] for c in cls_names]
    best_match, best_err = None, None
    for model in ["M1", "ANCHOR"]:
        variant = primary_variant[model]
        rec = [percls.get((model, "neu", variant, "test", c)) for c in cls_names]
        if any(v is None for v in rec):
            continue
        err = sum(abs(a - b) for a, b in zip(m1_report, rec))
        if best_err is None or err < best_err:
            best_match, best_err = model, err

    # ---------------------------------------------------------- summary md
    pc = {r["model"]: r for r in read_csv(AUDIT_DIR / "param_counts.csv")} if (AUDIT_DIR / "param_counts.csv").exists() else {}
    lines.append("# Reconciliation summary (auto-generated by scripts_v3/reconciliation.py)\n")
    lines.append("## NEU-DET test (old vs new)\n")
    lines.append("| model | metric | old | new val | new own | delta |")
    lines.append("|---|---|---|---|---|---|")
    for r in neu_rows:
        lines.append(f"| {r['model']} | {r['metric']} | {r['old_reported']} | {r['new_ultralytics_val']} | {r['new_metrics_lib']} | {r['delta_new_minus_old']} |")
    lines.append("\n## GC10 (old test-230 protocol vs new)\n")
    lines.append("| model | metric | old test230 | new val test230 | new own test230 | new own ALL 2300 |")
    lines.append("|---|---|---|---|---|---|")
    for r in gc_rows:
        lines.append(f"| {r['model']} | {r['metric']} | {r['old_reported_test230']} | {r['new_val_test230_single_cls']} | {r['new_own_test230']} | {r['new_own_all2300']} |")
    lines.append("\n## Class-wise AP50: report table vs recomputed\n")
    lines.append("| class | report M1 | recomp M1 | report M2 | recomp M2 | report M3 | recomp M3 | report M4 | recomp M4 | recomp ANCHOR |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in cw_rows[:-1]:
        lines.append(f"| {r['class']} | {r['report_M1']} | {r['recomputed_M1']} | {r['report_M2']} | {r['recomputed_M2']} | {r['report_M3']} | {r['recomputed_M3']} | {r['report_M4']} | {r['recomputed_M4']} | {r['recomputed_ANCHOR']} |")
    lines.append(f"| MEAN | {mean_row['report_M1']} | {mean_row['recomputed_M1']} | {mean_row['report_M2']} | {mean_row['recomputed_M2']} | {mean_row['report_M3']} | {mean_row['recomputed_M3']} | {mean_row['report_M4']} | {mean_row['recomputed_M4']} | {mean_row['recomputed_ANCHOR']} |")
    lines.append(f"\nReport 'M1' class-wise column best matches recomputed checkpoint: **{best_match}** (total abs err {best_err:.4f})\n")
    lines.append("## Parameter counts\n")
    for m, r in pc.items():
        lines.append(f"- {m}: total={r['total_params']}, fused={r.get('fused_params','')}, attention={r['attention_params']}, eca_k={r['eca_kernel_sizes']}")
    (AUDIT_DIR / "reconciliation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\nBest match for report M1 class column: {best_match} (abs err {best_err})")


if __name__ == "__main__":
    main()
