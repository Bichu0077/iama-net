# AGENTS.md — IAMA-Net project working agreement

Created 2026-10-05 as STEP 0 of the audit-and-fix session. All agents/sessions working in this repo must follow this file.

## 1. Project summary

IAMA-Net is a computer-vision course project: an "illumination-aware multi-scale attention" variant of YOLOv11s (Ultralytics) for steel surface defect detection. It combines (a) CLAHE-on-LAB + bilateral-filter preprocessing and (b) ECA + CBAM-style spatial attention inserted at P3/P4/P5, trained on NEU-DET (6 classes) with a zero-shot class-agnostic transfer test on GC10-DET (10 classes). An external review found serious flaws (overclaims, internal inconsistencies, confounded ablation, single seed); this session audits and fixes them.

## 2. Directory map

```
iama-net/
├── AGENTS.md                  # this file (rules + checklist)
├── README.md                  # quick start (references a venv that does not currently exist)
├── requirements.txt           # pinned: torch 2.11.0 (cu128), ultralytics 8.3.40, opencv, numpy 2.2.6...
├── yolo11n.pt                 # stray YOLO11n weights at repo root (unused by pipeline)
├── configs/
│   ├── yolo11s_baseline.yaml  # M1 architecture (stock YOLO11s, nc=6)
│   └── yolo11s_iama.yaml      # M3/M4 architecture: ECA_SpatialAttn at layers 5, 8, 13 (P3/P4/P5)
├── models/
│   └── attention.py           # ECA (k adaptive, min 3, bias init 3.0), SpatialAttn (7x7), ECA_SpatialAttn
├── preprocessing/
│   └── illumination.py        # LAB CLAHE(clip 2.0, 8x8) + bilateralFilter(d=5, 50, 50) defaults
├── data/
│   ├── NEU-DET/               # raw: images+labels, train 1257 / val 270 / test 270 (200x200 grayscale jpg)
│   ├── NEU-DET_preprocessed/  # CLAHE+bilateral copy, same split sizes, own data.yaml
│   ├── GC10-DET/              # images train 1841 / val 231 / test 231; labels train 1840 / val 230 / test 230 (mismatch to audit)
│   ├── samples_annotated/
│   ├── download_datasets.py
│   └── convert_to_yolo.py
├── scripts/
│   ├── train.py / train_optimized.py / train_proposal_aligned.py   # training (M2–M4 via proposal-aligned)
│   ├── evaluate.py            # val() on NEU-DET test + naive FPS (10 warmup / 50 timed predict calls, batch 1 dummy)
│   ├── evaluate_all.py        # wrote results/ablation_table.csv and cross_dataset_results.csv
│   ├── cross_dataset_eval.py  # GC10 class-agnostic (single_cls=True), test split only (230 images)
│   ├── gradcam.py             # P5 Grad-CAM, hand-picked examples
│   ├── grid_search_preprocessing.py
│   ├── detect.py
│   └── convert_md_to_pdf.py
├── runs/                      # ORIGINAL artifacts — read-only, never overwrite
│   ├── M1_baseline_seed42_20260924_195541/  # weights: best, last, epoch0/25/50/75; args.yaml
│   ├── M2_aligned_seed42/     # best, last; args.yaml (25 epochs, init = M1 best, AdamW lr0=8e-4)
│   ├── M3_aligned_seed42/     # best, last; args.yaml (25 epochs, mapped init, freeze=[1,2,3,4,6,7,9,10,11,12])
│   ├── M4_aligned_seed42/     # best, last; args.yaml (30 epochs, mapped init, same freeze)
│   ├── m3_aligned_init_seed42.pt   # identity-mapped anchor (M1 best → IAMA yaml, identity attention)
│   └── m4_aligned_init_seed42.pt   # same mapping used to init M4
├── results/                   # ORIGINAL artifacts — read-only
│   ├── ablation_table.csv / eval_results.csv / cross_dataset_results.csv / preprocessing_grid_search.csv
│   ├── IAMA_Net_Academic_Research_Report.md (+ .pdf)  # the report under review
│   ├── results_summary.md / project_progress_report.md
│   ├── gradcam_examples/ / detection_tests/ / preprocessing_comparisons/
├── results_v3/                # NEW outputs of this session (audit/, figures/, CSVs) — dated, never overwrite runs/ or results/
└── runs_v3/                   # NEW training runs of this session
```

Key facts recorded from `runs/*/args.yaml` (evidence for review finding C2, confounded protocol):
- M1: 100 epochs, optimizer=auto, lr0=0.01, cos_lr=false, close_mosaic=10, warmup_epochs=3.0, from `configs/yolo11s_baseline.yaml`, raw NEU-DET.
- M2: 25 epochs, init = M1 `best.pt`, AdamW lr0=0.0008, cos_lr=true, close_mosaic=0, warmup 1.0, no freeze, preprocessed NEU-DET.
- M3: 25 epochs, init = identity-mapped M1 best (`m3_aligned_init_seed42.pt`), freeze=[1,2,3,4,6,7,9,10,11,12], raw NEU-DET.
- M4: 30 epochs, init = identity-mapped M1 best (`m4_aligned_init_seed42.pt`), same freeze, preprocessed NEU-DET.

## 3. Hardware / environment constraints (fixed; do not change across variants unless a task says so)

- GPU: NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM (driver 596.36, CUDA 13.2 capable).
- OS: Windows 11. Shell: PowerShell 5.1.
- Ultralytics **8.3.40** (required — 8.1.29 cannot even load these YOLO11 checkpoints: `C3k2` missing).
- `workers=2`, `batch=8` with `accumulate=2`, `imgsz=640`.
- Optimizer AdamW, `lr0=0.0008`, `lrf=0.01`, `weight_decay=5e-4`, cosine LR, `warmup_epochs=1.0`, `close_mosaic=0`.
- Known environment conflict (2026-10-05): no venv exists; global `C:\Python310` has torch 2.9.0+**cpu** and ultralytics 8.1.29. A repo-local venv must be rebuilt per `requirements.txt` (torch 2.11.0 cu128 + ultralytics 8.3.40) before any GPU work.

## 4. Hard rules

1. **No fabricated numbers.** Every metric in any report must be produced by a script from a CSV/JSON emitted by an actual evaluation run; record the script path next to the table.
2. **Never overwrite or delete** existing checkpoints or CSVs (`runs/`, `results/` are read-only). Write new outputs to new dated folders (`runs_v3/`, `results_v3/`).
3. **Every training run** is resumable and logs its full command, seed, git commit, and epoch count to a run manifest (JSON).
4. **Statistics:** report mean ± std over seeds and 95% bootstrap CIs. Never claim "significant" without a statistical test whose result is saved.
5. **Units:** percentage points (pp) for absolute differences, % only for relative differences; always label which.
6. **Honesty:** if a result contradicts the hypothesis, report it as is. Negative results are acceptable.
7. **Long jobs:** ask the user before starting any single job expected to run > 3 hours; print the time estimate first.
8. Training jobs run through a queue script that survives interruptions and logs to files (no live-terminal-only jobs).

## 5. Task checklist

### Phase 1 — Audit and zero-training fixes (< 1 GPU-hour)
- [x] 1.0 Rebuild environment (venv, torch cu128, ultralytics 8.3.40), verify checkpoint loading + GPU
- [x] 1.1 One unified eval script: recompute per-class AP, mAP@0.5, mAP@0.5:0.95, P, R, F1 for M1–M4 + identity anchor on NEU-DET test; reconciliation table old-vs-new, explain every mismatch (esp. M1 0.7527 vs 0.7608)
- [x] 1.2 Parameter recount script (M1, M3/M4); fix 309-vs-315 and ECA kernel sizes
- [x] 1.3 GC10 zero-shot re-eval of all 5 checkpoints on ALL images: (a) class-agnostic all-GT-collapsed; (b) restricted-overlap protocol with documented class mapping
- [x] 1.4 Bootstrap 95% CIs (1000 resamples over images) for all metrics; paired bootstrap M4 vs M1 on NEU-DET and GC10; save CSVs
- [x] 1.5 Proper FPS benchmark (batch 1, fp32+fp16, 50 warmup/300 timed, pre/inference/NMS split, clocks+thermal note, 5 repeats, mean±std), all models one session
- [x] 1.6 Data audit: image sizes/channels both datasets, split duplicate/leakage check, split file hashes
- [x] 1.7 Write results_v3/audit/audit_report.md

### Phase 2 — Matched ablation (ask before launching)
- [x] 2.1 Time one epoch of M1 and M4; print projected wall-clock for full plan; WAIT for user go-ahead
  (measured 2026-10-05: M1 110.3 s/epoch, M4 101.5 s/epoch, matched protocol incl. per-epoch val)
- [x] 2.2 Matched protocol: all 4 variants from same COCO-pretrained YOLOv11s, same epochs/hparams/split/augmentation; aligned semantic layer mapping for attention variants (log transferred tensor count); keep +4.0 bias identity init
  — **USER-APPROVED PLAN (2026-10-05): 100 epochs × 1 seed (seed 0), 4 runs (M1–M4), ≈12 h total. User waived the 3-seed minimum ("completion fast and correctly is the priority"). Consequence: no seed-variance estimate; report must state single-seed limitation; image-level bootstrap CIs still computed. Each single run ≈3 h (>3 h rule waived by this approval).**
- [x] 2.3 ~~Seeds 0/1/2 minimum (12 runs)~~ → seed 0 only per user decision; cache preprocessed images offline (DONE: NEU-DET_preprocessed verified clip2.0/d5/σ50; GC10-DET_preprocessed_v3 built); identical preprocessing at train/val/test
- [x] 2.4 Evaluate every run with Phase 1 scripts (NEU-DET test, GC10 all-images zero-shot, FPS); aggregate + CIs → results_v3/ablation_matched.csv, cross_dataset_matched.csv
- [x] 2.5 Diagnose patches/crazing/rolled-in-scale regressions under matched protocol

### Phase 3 — Controls and comparisons (ask before launching)
**USER BLANKET APPROVAL (2026-10-05 17:10 UTC): proceed through Phases 2–4 autonomously.**
**USER SCOPE TRIM (2026-10-05 17:55 UTC): "don't overdo the optional stuff — main objective is a good/very good course project, maybe small-journal publishable; extensive stuff later." Trimmed Phase 3: KEEP = SE + CBAM alternative attention (1 seed, 100 ep), grayscale-CLAHE preprocessing control (1 seed, 100 ep), quantified Grad-CAM on matched models, transfer controls (tiling + multi-scale) for M1m/M4m only, per-GC10-class results, preprocessing on/off at test time. DROPPED (deferred, not deleted from plan): Coordinate-Attention run, YOLOv8s external reference, LAB-no-bilateral run, tiling for all 4 models, Grad-CAM on original checkpoints, EMA. Revised completion ETA ≈ 2026-10-06 05:00 UTC.**
- [x] 3.1 Preprocessing sanity (gray-CLAHE arm done; sigma variant deferred): grayscale CLAHE (clip 2.0, 8x8, OpenCV semantics) vs LAB route; bilateral off; smaller sigma_color; select on VALIDATION only
- [x] 3.2 Alternative attention (SE+CBAM done; CA/EMA deferred) (SE, CBAM, CA, EMA if time) at same insertion points, same init, attention-only, 3 seeds
- [ ] 3.3 External reference (DEFERRED by user trim 2026-10-05) detector (YOLOv8s) same protocol, 3 seeds
- [x] 3.4 Transfer confound controls: GC10 at multiple input scales incl. defect-scale-matched tiling; per-GC10-class results; preprocessing on/off at test time per model
- [x] 3.5 Quantified Grad-CAM: pointing-game / energy-in-GT-box on whole NEU-DET test set, P3/P4/P5 or eigen-CAM; mean ± CI

### Phase 4 — Honest rewrite
- [x] 4.1 make_report_tables.py — all tables/figures regenerated from CSVs, no hand-typed numbers
- [x] 4.2 Rewrite abstract/verdicts/conclusions/§6; supported/not-supported/inconclusive wording; remove banned phrases; fix GC10 facts, LAB-chromaticity argument, CLAHE formula (OpenCV semantics), "ray-traced"
- [x] 4.3 Add related work (2 refs flagged for manual verification), verified references, full training details, limitations/threats-to-validity
- [x] 4.4 Reframe as empirical study (in-domain + cross-dataset shift question)
- [x] 4.5 viva_prep.md: 20 examiner Q&A grounded in new numbers + remaining weaknesses
- [x] 4.6 CHANGELOG.md: every number that changed vs old report, and why

## 6. Known flaws (copied from the external review)

### A. Claims vs evidence
- The "full" model M4 has the LOWEST NEU-DET mAP@0.5 (0.733 vs baseline 0.753), lowest mAP@0.5:0.95 (0.406 vs 0.420), lowest F1; it loses 3 of 6 classes by 9–13 pp (crazing −10.4, patches −13.2, rolled-in scale −8.9).
- The abstract cherry-picks: "inclusion 87.8%" and "pitted 83.1%" come from M3 (attention only), not M4. "+9.7%" is percentage points but is shown beside a relative "+49.8%".
- H1 (preprocessing) and H2 (attention) are NOT supported by the report's own tables (mAP, F1, precision all drop). H3 is trivially true (baseline is faster). H4 is only suggestive. "Strongly confirmed", "statistically significant", "the only deployable model" are unsupported.

### B. Internal inconsistencies
- Class-wise APs average to M1=0.7608, M2=0.7338, M3=0.7442, M4=0.7293, but the master table states 0.7527, 0.7444, 0.7484, 0.7333. The M1 class-wise average (0.7608) equals the reported mAP of the identity-mapped anchor model, so the M1 column may come from the wrong checkpoint. Find the cause.
- Parameter overhead is 309 in the abstract but 315 in section 2.2. Recount from the actual model (also check ECA kernel size: log2(128)/2+0.5 = 4, so k is 3 or 5, not clearly 5).
- GC10-DET facts are off or uncertain: the 10th class is "rolled pit" (not "rolled-in scale"); the public copy has ~2,300 images, roughly 3,563 boxes. Verify against the local copy and fix.
- Report calls NEU-DET "high-resolution" but it is 200×200 grayscale.

### C. Methodological flaws
- Single seed (42). The identity-init model is nearly a no-op (scale ~0.964) yet changes mAP by +0.8 pp, and the same architecture is 57.2 FPS in one row and 44.4 FPS in another. Noise is about 1 pp or more; FPS measurement is unreliable.
- Training protocol is probably confounded: M3/M4 seem fine-tuned from converged M1 weights (different init and budget than M1/M2). Epoch counts are not stated anywhere.
- The crazing explanation ("bilateral filter + bicubic upscaling") fails: M3 has no preprocessing yet also drops crazing (−5.6 pp), patches (−13.1 pp), rolled-in scale (−7.5 pp); patches are large defects, not 1-px cracks; upscaling 200→640 also applies to the baseline.
- GC10 transfer: mAP 0.0311 → 0.0466 and recall FALLS 0.0956 → 0.0792 (about 35 vs 29 true detections of 366). M2 and M3 are each below baseline but M4 is +50%, which looks like seed noise, not synergy. Precision 8.6% → 16.6% means false detections go from 91% to 83% of detections, not "halved". Confounds not controlled: label-space shift (most GC10 classes do not exist in NEU-DET), resolution shift (2048×1000 downscaled to 640 vs 200×200 upscaled to 640), and only 230 of ~2,300 images used even though the test is zero-shot.
- Both datasets are grayscale, so the "decouple chromaticity via CIE L*a*b*" rationale is moot. "Ray-traced contrast limit" is not a CLAHE concept. The stated clip-limit formula does not match OpenCV's definition (clipLimit is a multiplier of the mean bin count). Bilateral filter sigma_color=50 smooths the faint (<15 gray level) cracks it is meant to preserve.
- The real-time requirement (20–30 m/s → 30–45 FPS) is asserted, not derived; line-scan systems are not specified in FPS. FPS protocol is undefined (batch size, warm-up, whether preprocessing and NMS are included, thermal state).
- Grad-CAM is on the 20×20 P5 map with 3 hand-picked examples; boundary-level claims are not supportable and nothing is quantified.
- Novelty claims ("first joint illumination-attention study", "first zero-shot cross-mill protocol") are not credible: CLAHE inside YOLO pipelines and CBAM/ECA/CA-style attention in YOLO on NEU-DET already exist. Only rigorous cross-dataset evaluation is a real gap. No related-work section, no references, no SOTA or alternative-attention comparison.
