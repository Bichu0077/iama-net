"""Phase 2.4 aggregation: build results_v3/ablation_matched.csv and
results_v3/cross_dataset_matched.csv from the matched-protocol evaluation
artifacts. Single seed (0) per user decision 2026-10-05, so mean=value and
std across seeds is not available; image-level bootstrap CIs are included.

Inputs (results_v3/audit/):
  matched_val_point_estimates.csv, matched_bootstrap_ci.csv,
  matched_per_class_ap.csv, matched_fps_benchmark_summary.csv,
  matched_paired_bootstrap.csv (M4m vs M1m),
  matched_pM2_paired_bootstrap.csv, matched_pM3_paired_bootstrap.csv,
  gc10_restricted_matched.csv
"""

import csv
import json
from pathlib import Path

from iama_env import AUDIT_DIR, PROJECT_ROOT

OUT_ROOT = PROJECT_ROOT / "results_v3"

VARIANTS = ["M1m", "M2m", "M3m", "M4m"]
META = {
    "M1m": dict(variant="M1", arch="baseline", train_inputs="raw"),
    "M2m": dict(variant="M2", arch="baseline", train_inputs="preprocessed"),
    "M3m": dict(variant="M3", arch="iama_attention", train_inputs="raw"),
    "M4m": dict(variant="M4", arch="iama_attention", train_inputs="preprocessed"),
}
NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
NEU_EVAL_INPUTS = {"M1m": "raw", "M2m": "pp", "M3m": "raw", "M4m": "pp"}


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    val = read_csv(AUDIT_DIR / "matched_val_point_estimates.csv")
    boot = read_csv(AUDIT_DIR / "matched_bootstrap_ci.csv")
    percls = read_csv(AUDIT_DIR / "matched_per_class_ap.csv")
    fps = read_csv(AUDIT_DIR / "matched_fps_benchmark_summary.csv")
    paired = (read_csv(AUDIT_DIR / "matched_paired_bootstrap.csv")
              + read_csv(AUDIT_DIR / "matched_pM2_paired_bootstrap.csv")
              + read_csv(AUDIT_DIR / "matched_pM3_paired_bootstrap.csv"))
    restr = read_csv(AUDIT_DIR / "gc10_restricted_matched.csv")

    def val_get(model, dataset, variant, scope, metric):
        for r in val:
            if (r["model"], r["dataset"], r["variant"], r["scope"], r["metric"]) == \
               (model, dataset, variant, scope, metric):
                return r["value"]
        return ""

    def boot_get(model, dataset, variant, scope, metric, cls=""):
        for r in boot:
            if (r["model"], r["dataset"], r["variant"], r["scope"], r["metric"], r["cls"]) == \
               (model, dataset, variant, scope, metric, cls):
                return r
        return {}

    # ------------------------------------------------ ablation_matched.csv
    rows = []
    for m in VARIANTS:
        meta = META[m]
        ev = NEU_EVAL_INPUTS[m]
        b50 = boot_get(m, "neu", ev, "all", "mAP@0.5")
        b5095 = boot_get(m, "neu", ev, "all", "mAP@0.5:0.95")
        row = {
            "model_id": m,
            "variant": meta["variant"],
            "seed": 0,
            "epochs": 100,
            "arch": meta["arch"],
            "train_inputs": meta["train_inputs"],
            "eval_inputs": ev,
            "seeds_note": "single seed per user decision 2026-10-05; no between-seed std available",
            "mAP50_val": val_get(m, "neu", ev, "test", "mAP@0.5"),
            "mAP50_boot_point": b50.get("point", ""),
            "mAP50_ci95_low": b50.get("ci95_low", ""),
            "mAP50_ci95_high": b50.get("ci95_high", ""),
            "mAP5095_val": val_get(m, "neu", ev, "test", "mAP@0.5:0.95"),
            "mAP5095_boot_point": b5095.get("point", ""),
            "mAP5095_ci95_low": b5095.get("ci95_low", ""),
            "mAP5095_ci95_high": b5095.get("ci95_high", ""),
            "precision_val": val_get(m, "neu", ev, "test", "precision_mean"),
            "recall_val": val_get(m, "neu", ev, "test", "recall_mean"),
        }
        for cname in NEU_CLASSES:
            r = [x for x in percls if (x["model"], x["dataset"], x["variant"], x["class"]) == (m, "neu", ev, cname)]
            row[f"AP50_{cname}"] = r[0]["AP50"] if r else ""
        for f_ in fps:
            if f_["model"] == m and f_["source"] == "real_neu_test_image":
                row[f"fps_{f_['precision']}"] = f_["fps_mean"]
                row[f"fps_{f_['precision']}_std"] = f_["fps_std"]
                row[f"inf_ms_{f_['precision']}"] = f_["inf_ms_mean"]
        rows.append(row)

    # paired deltas vs M1m (NEU)
    for row in rows:
        m = row["model_id"]
        if m == "M1m":
            continue
        for metric, key in [("mAP@0.5", "mAP50"), ("mAP@0.5:0.95", "mAP5095")]:
            pr = [r for r in paired
                  if r["B_model"] == m and r["dataset"] == "neu" and r["metric"] == metric]
            if pr:
                p = pr[0]
                row[f"delta_{key}_vs_M1_pp"] = p["delta_point_pp"]
                row[f"delta_{key}_ci_low_pp"] = p["delta_ci95_low_pp"]
                row[f"delta_{key}_ci_high_pp"] = p["delta_ci95_high_pp"]
                row[f"delta_{key}_p"] = p["p_two_sided"]

    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT_ROOT / "ablation_matched.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("saved results_v3/ablation_matched.csv")

    # -------------------------------------------- cross_dataset_matched.csv
    xrows = []
    for m in VARIANTS:
        protocols = [
            ("agnostic_test230_raw", "gc10", "raw", "test", None),
            ("agnostic_all2300_raw", "gc10", "raw", "all", "all"),
            ("agnostic_all2300_raw_testsubset", "gc10", "raw", "all", "test"),
        ]
        if m in ("M2m", "M4m"):
            protocols += [("agnostic_all2300_pp", "gc10", "pp", "all", "all"),
                          ("agnostic_test230_pp", "gc10", "pp", "test", None)]
        for pname, ds, variant, val_scope, boot_scope in protocols:
            row = {"model_id": m, "variant": META[m]["variant"], "protocol": pname, "seed": 0}
            row["mAP50_val"] = val_get(m, ds, variant, val_scope, "mAP@0.5")
            row["precision_val"] = val_get(m, ds, variant, val_scope, "precision_mean")
            row["recall_val"] = val_get(m, ds, variant, val_scope, "recall_mean")
            if boot_scope:
                b = boot_get(m, ds, variant, boot_scope, "mAP@0.5")
                row["mAP50_boot_point"] = b.get("point", "")
                row["mAP50_ci95_low"] = b.get("ci95_low", "")
                row["mAP50_ci95_high"] = b.get("ci95_high", "")
            xrows.append(row)
        # restricted overlap rows
        for r in restr:
            if r["model"] != m:
                continue
            if r["metric"] == "map50":
                xrows.append({"model_id": m, "variant": META[m]["variant"],
                              "protocol": f"restricted_{r['scope']}_{r['input_variant']}",
                              "seed": 0, "mAP50_val": "", "precision_val": "", "recall_val": "",
                              "mAP50_boot_point": r["point"], "mAP50_ci95_low": r["ci95_low"],
                              "mAP50_ci95_high": r["ci95_high"],
                              "n_gt_restricted": r["n_gt_restricted"]})
        # paired deltas on GC10 (vs M1m)
        if m != "M1m":
            for pr in paired:
                if pr["B_model"] == m and pr["dataset"] == "gc10" and pr["metric"] == "mAP@0.5":
                    scope_tag = "all2300" if "all 2300" in pr["label"] else "test230"
                    xrows.append({"model_id": m, "variant": META[m]["variant"],
                                  "protocol": f"paired_delta_vs_M1_{scope_tag}", "seed": 0,
                                  "mAP50_val": "", "mAP50_boot_point": pr["B_point"],
                                  "delta_pp": pr["delta_point_pp"],
                                  "delta_ci_low_pp": pr["delta_ci95_low_pp"],
                                  "delta_ci_high_pp": pr["delta_ci95_high_pp"],
                                  "p_two_sided": pr["p_two_sided"]})
    xkeys = []
    for r in xrows:
        for k in r:
            if k not in xkeys:
                xkeys.append(k)
    with open(OUT_ROOT / "cross_dataset_matched.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=xkeys)
        w.writeheader()
        w.writerows(xrows)
    print("saved results_v3/cross_dataset_matched.csv")


if __name__ == "__main__":
    main()
