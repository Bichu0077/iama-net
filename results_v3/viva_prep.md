# Viva Prep — IAMA-Net v3 (2026-10-06)

20 likely examiner questions with short, truthful answers grounded in the v3 numbers
(sources: `results_v3/ablation_matched.csv`, `cross_dataset_matched.csv`,
`attention_alternatives.csv`, `preprocessing_controls.csv`, `transfer_controls.csv`,
`phase3/gradcam_quant.csv`, `results_v3/audit/*`). Followed by the remaining weaknesses
and how to present them honestly.

## Q&A

**1. What is IAMA-Net in one sentence?**
A YOLOv11s variant with (a) CLAHE+bilateral "illumination" preprocessing and (b) 315-parameter
ECA+spatial-attention blocks at P3/P4/P5, studied as an empirical question: do these two cheap
additions help steel defect detection in-domain and under cross-dataset shift?

**2. What is your headline result?**
Under a matched protocol (100 epochs, seed 0, same COCO init, same hyperparameters), the full
model gains +2.02 pp mAP@0.5 over the baseline (0.7547 vs 0.7345; paired image bootstrap
p=0.038), driven mainly by crazing (+10.5 pp AP50); mAP@0.5:0.95 (+0.46 pp, p=0.41) is
inconclusive. Preprocessing alone: no measurable effect. Zero-shot transfer: no variant helps;
modifications slightly hurt.

**3. Your earlier report claimed the opposite for some classes. What happened?**
The old ablation was confounded: M2–M4 were 25–30-epoch fine-tunes of the converged M1 (M3/M4
with 10 frozen backbone layers) vs M1's 100 epochs. It measured fine-tuning damage, not the
interventions — the identity-mapped anchor (≈M1, zero extra training) outscored all trained
variants (0.7608). Under the matched protocol the "regressions" (crazing −10.4, patches −13.2,
rolled-in scale −8.9 pp) reverse or vanish.

**4. How do you know the old numbers weren't fabricated?**
The master-table mAP@0.5 values all reproduce exactly from the original checkpoints (±0.0002).
What did NOT reproduce: the class-wise table (it came from the wrong checkpoint — the
identity-mapped anchor, total deviation 0.0019 vs 0.0646 for real M1), the mAP@0.5:0.95 column
(no CSV provenance; M2 reported 0.4070 vs recomputed 0.4275), and the "+9.7 pp inclusion" claim
(recomputed inclusion AP50 spans 0.782–0.788 across all five original checkpoints).

**5. Single seed — why should I believe +2.02 pp?**
You should believe it conditionally. The paired image-bootstrap CI excludes zero (p=0.038) for
this seed, but seed variance is unestimated; the original study's own anchor-vs-M1 gap (~0.8 pp
from an almost no-op change) suggests seed noise around ~1 pp. The crazing gain (+8.5…+10.6 pp)
is consistent across four independent attention runs (ECA, SE, CBAM, full model), which is
stronger evidence than the mAP headline. A 3-seed sweep is the first extension for publication.

**6. Is your attention module novel?**
No, and we don't claim it. ECA (Wang et al. 2020) and CBAM-style spatial attention (Woo et al.
2018) are standard; insertion into YOLO necks for defect detection exists in the 2022–2025
literature. The contribution is the rigorously controlled evaluation and the audit itself.

**7. Is ECA+spatial the right module choice?**
Not demonstrably. SE at the same insertion points ties in-domain (+0.30 pp vs ECA+spatial,
p=0.80) and is significantly better on GC10 zero-shot (+0.41 pp, p=0.006), at 50k vs 315 extra
parameters. CBAM is statistically indistinguishable in-domain. The honest claim: "adding
attention at P3/P4/P5 helps mildly; the specific module is not justified by our data."

**8. Does the illumination preprocessing work?**
No measurable in-domain benefit (+0.31 pp, p=0.67; mAP@0.5:0.95 −0.22 pp, p=0.72), a
significant transfer cost (−0.73 pp on GC10 all-images, p=0.014), and a hard input-distribution
dependency (PP-trained models lose ≈20 pp mAP@0.5 on raw test inputs). A "principled" grayscale
CLAHE control (no LAB, no bilateral) is worse than the original recipe (−1.19 pp, p=0.26) —
so even the recipe's accidental choices matter more than its rationale.

**9. Both datasets are grayscale — why was LAB used at all?**
That was a design error in the original work: the "decouple chromaticity" rationale is
inapplicable, and the LAB round trip measurably injects chroma noise (308 of 1,797 preprocessed
NEU images come out with non-identical BGR channels). We keep the original recipe as the object
of study and add the grayscale-CLAHE control; the control does not rescue the idea.

**10. What about the +49.8% transfer improvement claim?**
Dead. It was computed on 230 of 2,300 GC10 images under the confounded protocol. Original
checkpoints on all 2,300 images: +0.32 pp (p=0.17, CI spans 0). Matched models: M4 is
significantly WORSE than baseline on all images (−1.03 pp, p<0.001) and null on the test split
(+0.06 pp, p=0.79). Attention-only is worst (−1.62 pp, p<0.001).

**11. Why is zero-shot transfer so bad (mAP ≈ 0.03–0.05)?**
Label-space shift (6 NEU classes vs 10 GC10 classes with little overlap), scale shift (200×200
upscaled vs 2048×1000 downscaled to 640), and domain shift (different mills/imaging). Our
controls show scale isn't the whole story: imgsz 1024 lifts baseline transfer (0.0448→0.0644),
and defect-scale-matched tiling lowers absolute AP for everyone (0.0244 baseline / 0.0150 full)
but preserves the ordering.

**12. Which GC10 classes transfer at all?**
Blob/texture-like ones: oil_spot (AP50 0.043 baseline), punching_hole (0.036), crescent_gap
(0.017–0.043), water_spot (0.011). The semantically closest classes are near zero: inclusion
(0.001), rolled_pit (0.0008), welding_line (0.0000). Transfer follows superficial texture
similarity, not class semantics — which is why the restricted-overlap protocol (mapping NEU
inclusion/pitted/rolled-in-scale → GC10 inclusion/rolled_pit) is at noise level (≤0.0022) for
every model.

**13. Your GC10 test split has how many rolled_pit instances?**
One (of 366 boxes). Dataset-wide there are 85. Any per-class claim on the test split for that
class is meaningless; we evaluate on all 2,300 images for exactly this reason.

**14. What do the Grad-CAM numbers say?**
On the whole NEU test split: pointing game is best at P3 (0.49–0.54) and worst at P5
(0.22–0.30) — the old report's P5-only gallery was the coarsest possible evidence. Fused
pointing games overlap across models (no localisation advantage from attention). Energy-in-GT
is highest for the full model (fused 0.546 [0.518, 0.573] vs baseline 0.480 [0.449, 0.510],
CIs disjoint): it concentrates more attribution mass inside defects without a sharper peak.
Conditional variants have unequal denominators (229/242/163) and are not model-comparable.

**15. FPS claims?**
Controlled protocol (batch 1, 50 warm-up + 300 timed × 5 repeats, fp32/fp16, one session):
58–67 FPS wall for all variants on an RTX 3050 Laptop; attention costs ≈ +13% inference time
(14.1→16.0 ms in the audit session). The old table (43.5–59.5) is not reproducible — same
architecture varied up to 25% by session/input; wall-clock on this WDDM laptop GPU carries
±10–20% session uncertainty. fp16 gives no reliable batch-1 speedup.

**16. Parameter overhead — 309 or 315?**
315 (3 blocks × 105: ECA Conv1d k=5 → 5 weights + 1 bias; spatial Conv2d(2→1, 7×7) → 98 + 1).
309 assumed k=3. The ECA kernel is k=5 at all scales because the yaml hardcodes channels=512
(ultralytics doesn't width-scale unknown modules); actual channels are 256/256/512 at P3/P4/P5,
for which the adaptive formula also gives 5. Baseline is 9,415,122 fused / 9,430,114 unfused —
the old report quoted the fused number without saying so.

**17. How do you guarantee reproducibility now?**
Every run has a manifest (full command, seed, git commit, epochs, versions, timings) in
`runs_v3/*/manifest.json`; training runs through a resumable queue with state and logs; every
reported number is generated by a script from evaluation CSVs (`make_report_tables.py`,
`make_changelog.py`); the environment is pinned (`requirements.txt`, torch 2.11.0+cu128,
ultralytics 8.3.40). One caveat, stated in the report: the original venv patched ultralytics
site-packages and that class was lost; we reconstructed it and validated the reconstruction by
exactly reproducing the original checkpoints' metrics.

**18. Did you check data integrity?**
Yes: SHA-256 over file bytes and decoded pixels for all 4,097+2,300 images — no duplicates
within/across splits in either dataset, no orphan/missing labels, YOLO labels match GC10's own
metadata.jsonl exactly (3,563 boxes). Split hashes are archived under
`results_v3/audit/split_hashes/`.

**19. What is the practical recommendation from this study?**
For a deployment on NEU-DET-like data: add cheap attention at P3/P4/P5 (SE is fine and arguably
better), skip the illumination preprocessing (it adds an input-distribution dependency and hurts
transfer), and expect no zero-shot capability on a new mill's data — collect and label local
data instead. All variants are comfortably real-time on a 4 GB laptop GPU.

**20. What would you do next (if this became a paper)?**
(i) 3–5 seeds per variant (the single-seed limitation is the biggest weakness); (ii) a
proper resolution-matched transfer study (train on GC10 tiles, test on NEU, both directions);
(iii) test-time augmentation / domain-adaptation baselines for the transfer question;
(iv) replace the heuristic ECA kernel derivation with the correct per-scale channel counts;
(v) quantify the preprocessing on/off decision on the validation split per dataset rather than
fixing it a priori.

## Remaining weaknesses (say these before the examiner does)

1. **Single seed** — all matched results conditional on seed 0; seed noise plausibly ~1 pp;
   only crazing's +8.5…+10.6 pp consistency across 4 attention runs is multi-model evidence.
2. **p=0.038 headline is marginal** — one seed sweep could move it either way; mAP@0.5:0.95 is
   already inconclusive (p=0.41).
3. **pitted_surface −3.5…−4.9 pp** appears under ECA/SE attention AND both preprocessing
   variants but not CBAM; with 67 test instances and one seed it is unattributable.
4. **Reconstructed attention class** for the original checkpoints (validated by exact metric
   reproduction, but still a reconstruction).
5. **GC10 "all images"** includes its train/val splits — legitimate zero-shot, but not
   comparable to supervised GC10 literature numbers.
6. **Tiling control is non-overlapping** — boundary defects penalised for all models equally.
7. **Grad-CAM target choice** (max pre-NMS anchor score) is one defensible option among several.
8. **FPS on a laptop WDDM GPU shared with the display** — within-session comparisons only.
9. **Two references still need manual bibliographic verification** (GC10-DET TII paper;
   Zuiderveld CLAHE DOI) — flagged in `results_v3/report/references.md`.
10. **Scope trims** (user-directed): no Coordinate-Attention/EMA runs, no YOLOv8s external
    reference, no LAB-no-bilateral arm, no 5-seed extension — all listed as future work.
