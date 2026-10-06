"""Phase 4.1: regenerate ALL report tables and figures from evaluation CSVs.

No number is hand-typed in the final report: it embeds the markdown table
fragments written here (results_v3/report/tables/*.md) and the figures
(results_v3/report/figures/*.png).

Reads (whatever exists): results_v3/audit/*.csv, results_v3/*.csv,
results_v3/phase3/*.csv. Missing inputs are skipped with a warning so the
script can also run on partial results.

Usage: python scripts_v3/make_report_tables.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from iama_env import AUDIT_DIR, PROJECT_ROOT

OUT = PROJECT_ROOT / "results_v3" / "report"
TAB = OUT / "tables"
FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RV3 = PROJECT_ROOT / "results_v3"


def read_csv(path):
    if not Path(path).exists():
        print(f"WARN missing {path}")
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_md(name, lines):
    (TAB / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"table -> {TAB / name} ({len(lines)} lines)")


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


def fnum(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x) if x not in (None, "") else "—"


# ------------------------------------------------------------------ tables

def t_old_vs_new():
    recon = read_csv(AUDIT_DIR / "reconciliation_neu.csv")
    cw = read_csv(AUDIT_DIR / "reconciliation_classwise.csv")
    rows = []
    for r in recon:
        if r["metric"] in ("mAP@0.5", "mAP@0.5:0.95", "precision", "recall"):
            rows.append([r["model"], r["metric"], fnum(r["old_reported"]),
                         fnum(r["new_ultralytics_val"]), fnum(r["delta_new_minus_old"])])
    write_md("old_vs_new_neu.md", md_table(
        ["model", "metric", "old reported", "new val()", "Δ(new−old)"], rows))

    cw_rows = []
    for r in cw:
        cw_rows.append([r["class"], r.get("report_M1", ""), fnum(r.get("recomputed_M1"), 3),
                        fnum(r.get("recomputed_ANCHOR"), 3), r.get("report_M2", ""),
                        fnum(r.get("recomputed_M2"), 3), r.get("report_M3", ""),
                        fnum(r.get("recomputed_M3"), 3), r.get("report_M4", ""),
                        fnum(r.get("recomputed_M4"), 3)])
    write_md("old_vs_new_classwise.md", md_table(
        ["class", "rep M1", "new M1", "new ANCHOR", "rep M2", "new M2", "rep M3", "new M3", "rep M4", "new M4"],
        cw_rows))


def t_original_audit():
    boot = read_csv(AUDIT_DIR / "bootstrap_ci.csv")
    paired = read_csv(AUDIT_DIR / "paired_bootstrap_m4_vs_m1.csv")
    restr = read_csv(AUDIT_DIR / "gc10_restricted_overlap.csv")
    rows = []
    for r in boot:
        if r["cls"] == "" and r["metric"] in ("mAP@0.5", "mAP@0.5:0.95"):
            rows.append([r["model"], r["dataset"], r["variant"], r["scope"], r["metric"],
                         fnum(r["point"]), f"[{fnum(r['ci95_low'])}, {fnum(r['ci95_high'])}]"])
    write_md("original_models_ci.md", md_table(
        ["model", "dataset", "inputs", "scope", "metric", "point", "95% CI"], rows))
    prows = []
    for r in paired:
        prows.append([r["label"], r["metric"], fnum(r["M1_point"]), fnum(r["M4_point"]),
                      f"{fnum(r['delta_point_pp'],2)} pp",
                      f"[{fnum(r['delta_ci95_low_pp'],2)}, {fnum(r['delta_ci95_high_pp'],2)}] pp",
                      fnum(r["p_two_sided"], 4)])
    write_md("original_paired.md", md_table(
        ["comparison", "metric", "M1", "M4", "Δ", "Δ 95% CI", "p (two-sided)"], prows))
    rrows = []
    for r in restr:
        if r["metric"] == "map50":
            rrows.append([r["model"], r["input_variant"], r["scope"], r["n_gt_restricted"],
                          fnum(r["point"], 5), f"[{fnum(r['ci95_low'],5)}, {fnum(r['ci95_high'],5)}]"])
    write_md("original_restricted_overlap.md", md_table(
        ["model", "inputs", "scope", "n GT (mapped)", "mAP@0.5", "95% CI"], rrows))


def t_matched():
    abl = read_csv(RV3 / "ablation_matched.csv")
    if not abl:
        return
    rows = []
    for r in abl:
        rows.append([r["variant"], r["arch"], r["train_inputs"],
                     fnum(r["mAP50_val"]), f"[{fnum(r['mAP50_ci95_low'],3)}, {fnum(r['mAP50_ci95_high'],3)}]",
                     fnum(r["mAP5095_val"]), f"[{fnum(r['mAP5095_ci95_low'],3)}, {fnum(r['mAP5095_ci95_high'],3)}]",
                     fnum(r["precision_val"], 3), fnum(r["recall_val"], 3),
                     r.get("delta_mAP50_vs_M1_pp", ""), r.get("delta_mAP50_p", "")])
    write_md("matched_main.md", md_table(
        ["variant", "arch", "train inputs", "mAP@0.5", "95% CI", "mAP@0.5:0.95", "95% CI",
         "P", "R", "Δ mAP50 vs M1 (pp)", "p"], rows))

    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    prows = []
    for r in abl:
        prows.append([r["variant"]] + [fnum(r.get(f"AP50_{c}"), 3) for c in classes])
    write_md("matched_perclass.md", md_table(["variant"] + classes, prows))

    fps_rows = []
    for r in abl:
        fps_rows.append([r["variant"], r.get("fps_fp32", ""), r.get("fps_fp32_std", ""),
                         r.get("fps_fp16", ""), r.get("fps_fp16_std", ""), r.get("inf_ms_fp32", "")])
    write_md("matched_fps.md", md_table(
        ["variant", "FPS fp32", "±std", "FPS fp16", "±std", "inference ms fp32"], fps_rows))

    cross = read_csv(RV3 / "cross_dataset_matched.csv")
    crows = []
    for r in cross:
        crows.append([r["variant"], r["protocol"], fnum(r.get("mAP50_val"), 4) or fnum(r.get("mAP50_boot_point"), 4),
                      f"[{fnum(r.get('mAP50_ci95_low'),4)}, {fnum(r.get('mAP50_ci95_high'),4)}]"
                      if r.get("mAP50_ci95_low") else "",
                      fnum(r.get("precision_val"), 4), fnum(r.get("recall_val"), 4),
                      r.get("delta_pp", ""), r.get("p_two_sided", "")])
    write_md("matched_cross_dataset.md", md_table(
        ["variant", "protocol", "mAP@0.5", "95% CI", "P", "R", "Δ vs M1 (pp)", "p"], crows))


def t_phase3():
    attn = read_csv(RV3 / "attention_alternatives.csv")
    if attn:
        rows = []
        for r in attn:
            rows.append([r["attention"], r["model_id"], f"{int(float(r['params_total'])):,}"
                         if str(r["params_total"]).replace(',', '').isdigit() else r["params_total"],
                         fnum(r["neu_mAP50"]), r["neu_mAP50_ci"], fnum(r["neu_mAP5095"]),
                         fnum(r["gc10_all_mAP50"]), r.get("delta_neu_mAP50_vs_M3m_pp", ""),
                         r.get("delta_neu_p", ""), r.get("fps_fp32", "")])
        write_md("attention_alternatives.md", md_table(
            ["attention", "id", "params", "NEU mAP@0.5", "95% CI", "NEU mAP@0.5:0.95",
             "GC10-all mAP@0.5", "Δ NEU vs ECA+spatial (pp)", "p", "FPS fp32"], rows))

    pp = read_csv(RV3 / "preprocessing_controls.csv")
    if pp:
        rows = []
        for r in pp:
            rows.append([r["preprocessing"], r["model_id"], fnum(r["neu_mAP50"]), r["neu_mAP50_ci"],
                         fnum(r["neu_mAP5095"]), fnum(r["AP50_crazing"], 3),
                         r.get("delta_neu_mAP50_vs_M2m_pp", ""), r.get("delta_neu_p", "")])
        write_md("preprocessing_controls.md", md_table(
            ["preprocessing", "id", "NEU mAP@0.5", "95% CI", "NEU mAP@0.5:0.95",
             "AP50 crazing", "Δ vs LAB+bilateral (pp)", "p"], rows))

    ext = read_csv(RV3 / "external_reference.csv")
    if ext:
        rows = []
        for r in ext:
            rows.append([r["detector"], fnum(r["neu_mAP50"]), r["neu_mAP50_ci"],
                         fnum(r["neu_mAP5095"]), fnum(r["gc10_all_mAP50"]),
                         r.get("delta_neu_mAP50_vs_M1m_pp", ""), r.get("delta_neu_p", ""),
                         r.get("fps_fp32", "")])
        write_md("external_reference.md", md_table(
            ["detector", "NEU mAP@0.5", "95% CI", "NEU mAP@0.5:0.95", "GC10-all mAP@0.5",
             "Δ vs YOLO11s (pp)", "p", "FPS fp32"], rows))

    tc = read_csv(RV3 / "transfer_controls.csv")
    if tc:
        rows = []
        for r in tc:
            rows.append([r["control"], r["model"], r["setting"], r.get("metric", ""),
                         r.get("value", ""), r.get("ci95", ""), r.get("recall_at_0.25", "")])
        write_md("transfer_controls.md", md_table(
            ["control", "model", "setting", "metric", "value", "95% CI", "R@0.25"], rows))

    gc = read_csv(RV3 / "phase3" / "gradcam_quant.csv")
    if gc:
        rows = []
        for r in gc:
            rows.append([r["model"], r["scale"], r["pointing_game_all"],
                         f"[{r['pg_all_ci_low']}, {r['pg_all_ci_high']}]",
                         r["pointing_game_cond"], f"[{r['pg_cond_ci_low']}, {r['pg_cond_ci_high']}]",
                         r["energy_in_gt"], f"[{r['energy_ci_low']}, {r['energy_ci_high']}]"])
        write_md("gradcam_quant.md", md_table(
            ["model", "scale", "PG all", "95% CI", "PG cond (TP@0.25)", "95% CI",
             "energy-in-GT", "95% CI"], rows))


# ------------------------------------------------------------------ figures

def fig_matched():
    abl = read_csv(RV3 / "ablation_matched.csv")
    if not abl:
        return
    variants = [r["variant"] + ("*" if r["variant"] in ("M2", "M4") else "") for r in abl]
    m50 = [float(r["mAP50_val"]) for r in abl]
    m50lo = [float(r["mAP50_val"]) - float(r["mAP50_ci95_low"]) for r in abl]
    m50hi = [float(r["mAP50_ci95_high"]) - float(r["mAP50_val"]) for r in abl]
    m95 = [float(r["mAP5095_val"]) for r in abl]
    m95lo = [float(r["mAP5095_val"]) - float(r["mAP5095_ci95_low"]) for r in abl]
    m95hi = [float(r["mAP5095_ci95_high"]) - float(r["mAP5095_val"]) for r in abl]
    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.bar(x - 0.18, m50, 0.36, yerr=[m50lo, m50hi], capsize=3, label="mAP@0.5")
    ax.bar(x + 0.18, m95, 0.36, yerr=[m95lo, m95hi], capsize=3, label="mAP@0.5:0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("metric (NEU-DET test)")
    ax.set_title("Matched-protocol ablation (100 epochs, seed 0)\n* = trained/evaluated on preprocessed inputs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_matched_ablation.png")
    plt.close(fig)
    print("figure -> fig_matched_ablation.png")

    # per-class
    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    w = 0.8 / len(abl)
    for i, r in enumerate(abl):
        vals = [float(r.get(f"AP50_{c}", 0) or 0) for c in classes]
        ax.bar(np.arange(len(classes)) + i * w - 0.4 + w / 2, vals, w, label=r["variant"])
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels([c.replace("_", "\n") for c in classes])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AP@0.5")
    ax.set_title("Per-class AP@0.5, matched protocol (NEU-DET test)")
    ax.legend(ncol=len(abl))
    fig.tight_layout()
    fig.savefig(FIG / "fig_matched_perclass.png")
    plt.close(fig)
    print("figure -> fig_matched_perclass.png")


def fig_gc10():
    cross = read_csv(RV3 / "cross_dataset_matched.csv")
    if not cross:
        return
    protos = ["agnostic_test230_raw", "agnostic_all2300_raw_testsubset", "agnostic_all2300_raw"]
    labels = ["test-230 (old protocol)", "test-230 subset of all-run", "ALL 2300 images"]
    variants = sorted({r["variant"] for r in cross})
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=150)
    w = 0.8 / max(len(variants), 1)
    for i, vname in enumerate(variants):
        vals, los, his = [], [], []
        for p in protos:
            r = next((x for x in cross if x["variant"] == vname and x["protocol"] == p), None)
            if r and r.get("mAP50_boot_point"):
                val = float(r["mAP50_boot_point"])
                vals.append(val)
                los.append(val - float(r["mAP50_ci95_low"]))
                his.append(float(r["mAP50_ci95_high"]) - val)
            else:
                vals.append(0)
                los.append(0)
                his.append(0)
        ax.bar(np.arange(len(protos)) + i * w - 0.4 + w / 2, vals, w,
               yerr=[los, his], capsize=2, label=vname)
    ax.set_xticks(np.arange(len(protos)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("class-agnostic mAP@0.5 (GC10 zero-shot)")
    ax.set_title("GC10-DET zero-shot: small test split vs all images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_gc10_scope.png")
    plt.close(fig)
    print("figure -> fig_gc10_scope.png")


def fig_attention():
    attn = read_csv(RV3 / "attention_alternatives.csv")
    if not attn:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    names = [r["attention"] for r in attn]
    vals = [float(r["neu_mAP50"]) for r in attn]
    los, his = [], []
    for r in attn:
        ci = r["neu_mAP50_ci"].strip("[]").split(",")
        try:
            los.append(vals[-1] - float(ci[0]))
            his.append(float(ci[1]) - vals[-1])
        except (ValueError, IndexError):
            los.append(0)
            his.append(0)
    ax1.barh(names, vals, xerr=[los, his], capsize=3)
    ax1.set_xlim(0.5, 0.85)
    ax1.set_xlabel("NEU-DET test mAP@0.5")
    ax1.set_title("Attention alternatives (100 ep, seed 0)")
    prms = []
    for r in attn:
        try:
            prms.append(int(float(r["params_total"])))
        except (TypeError, ValueError):
            prms.append(0)
    overhead = [p - min(prms) for p in prms]
    ax2.barh(names, overhead)
    ax2.set_xscale("symlog")
    ax2.set_xlabel("attention overhead (params, symlog)")
    ax2.set_title("Parameter overhead")
    for i, v in enumerate(overhead):
        ax2.text(v, i, f" {v:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_attention_alternatives.png")
    plt.close(fig)
    print("figure -> fig_attention_alternatives.png")


def fig_gradcam():
    gc = read_csv(RV3 / "phase3" / "gradcam_quant.csv")
    if not gc:
        return
    models = sorted({r["model"] for r in gc})
    scales = ["P3", "P4", "P5", "fused"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        pg, en = [], []
        for s in scales:
            r = next((x for x in gc if x["model"] == m and x["scale"] == s), None)
            pg.append(float(r["pointing_game_all"]) if r else 0)
            en.append(float(r["energy_in_gt"]) if r else 0)
        ax1.bar(np.arange(len(scales)) + i * w - 0.4 + w / 2, pg, w, label=m)
        ax2.bar(np.arange(len(scales)) + i * w - 0.4 + w / 2, en, w, label=m)
    for ax, t in ((ax1, "Pointing game (all images)"), (ax2, "Energy inside GT boxes")):
        ax.set_xticks(np.arange(len(scales)))
        ax.set_xticklabels(scales)
        ax.set_ylim(0, 1)
        ax.set_title(t)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_gradcam_quant.png")
    plt.close(fig)
    print("figure -> fig_gradcam_quant.png")


def t_datasets_params():
    import json
    da = AUDIT_DIR / "data_audit_summary.json"
    if da.exists():
        s = json.loads(da.read_text())
        rows = []
        for ds in ("NEU-DET", "NEU-DET_preprocessed", "GC10-DET"):
            d = s["datasets"][ds]
            for sp in ("train", "val", "test"):
                v = d["splits"][sp]
                sizes = ", ".join(v["image_sizes"].keys())
                chans = ", ".join(v["channel_kinds"].keys())
                rows.append([ds, sp, v["n_images"], v["n_boxes"], sizes, chans])
        write_md("datasets.md", md_table(
            ["dataset", "split", "images", "boxes", "size (px)", "channels"], rows))
        gc = s.get("gc10_metadata_jsonl", {})
        if gc:
            write_md("gc10_metadata.md", md_table(
                ["scope", "boxes"],
                [["total", gc["total_boxes"]]] +
                [[k, v["n_boxes"]] for k, v in gc["per_split"].items()] +
                [["per class", "; ".join(f"{k}={v}" for k, v in gc["boxes_per_class"].items())]]))
    pc = read_csv(AUDIT_DIR / "param_counts.csv")
    if pc:
        rows = []
        for r in pc:
            rows.append([r["model"], f"{int(r['total_params']):,}",
                         f"{int(float(r['fused_params'])):,}" if r.get("fused_params") else "—",
                         r.get("attention_params", "0"), r.get("eca_kernel_sizes", "[]")])
        write_md("params.md", md_table(
            ["model", "params (unfused)", "params (fused)", "attention params", "ECA kernel sizes"], rows))


def main():
    t_old_vs_new()
    t_original_audit()
    t_matched()
    t_phase3()
    t_datasets_params()
    fig_matched()
    fig_gc10()
    fig_attention()
    fig_gradcam()
    print("\nAll available tables/figures regenerated in results_v3/report/")


if __name__ == "__main__":
    main()
