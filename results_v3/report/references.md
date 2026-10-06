# References for the IAMA-Net report rewrite (Phase 4.3)

Verification status recorded per AGENTS.md hard rule (verify every reference exists).
Verified 2026-10-05 via arXiv export API and Crossref REST API from this environment.
Items marked UNVERIFIED must be checked manually before submission.

## Verified (arXiv API / Crossref)

1. **ECA-Net** — Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, Q. Hu. "ECA-Net: Efficient
   Channel Attention for Deep Convolutional Neural Networks." CVPR 2020.
   arXiv:1910.03151. [verified: arXiv API title search]

2. **CBAM** — S. Woo, J. Park, J.-Y. Lee, I. S. Kweon. "CBAM: Convolutional Block
   Attention Module." ECCV 2018. arXiv:1807.06521. [verified: arXiv API]

3. **SE** — J. Hu, L. Shen, S. Albanie, G. Sun, E. Wu. "Squeeze-and-Excitation
   Networks." CVPR 2018 (journal version TPAMI 2020). arXiv:1709.01507.
   [verified: arXiv API]

4. **Coordinate Attention** — Q. Hou, D. Zhou, J. Feng. "Coordinate Attention for
   Efficient Mobile Network Design." CVPR 2021. arXiv:2103.02907.
   [verified: arXiv API]

5. **Grad-CAM** — R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh,
   D. Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
   Localization." ICCV 2017; IJCV 2019. arXiv:1610.02391,
   DOI 10.1007/s11263-019-01228-7. [verified: arXiv API incl. DOI]

6. **NEU-DET dataset** — K. Song, Y. Yan. "A noise robust method based on completed
   local binary patterns for hot-rolled steel strip surface defects." Applied Surface
   Science, 2013. DOI 10.1016/j.apsusc.2013.09.002. [verified: Crossref]

7. **Bilateral filtering** — C. Tomasi, R. Manduchi. "Bilateral Filtering for Gray and
   Color Images." ICCV 1998. DOI 10.1109/ICCV.1998.710815. [verified: Crossref]

8. **Steel-defect detector w/ NEU-DET + GC10-DET (2025)** — S. Sun, M. Deng, X. Yu,
   X. Xi, L. Zhao. "Self-Adaptive Gamma Context-Aware SSM-based Model for Metal
   Defect Detection" (GCM-DET). IJCNN 2025. arXiv:2503.01234. [verified: arXiv API]

9. **YOLOv5 variant w/ NEU-DET + GC10-DET (2023/24)** — S. M. Yasir, H. Ahn. "Faster
   Metallic Surface Defect Detection Using Deep Learning with Channel Shuffling."
   Computers, Materials & Continua 2023. arXiv:2406.14582,
   DOI 10.32604/cmc.2023.035698. [verified: arXiv API incl. DOI]

10. **Ultralytics YOLO (v8/v11)** — Ultralytics. "Ultralytics YOLO" software
    documentation, https://docs.ultralytics.com (version used: 8.3.40).
    [URL is the canonical software citation; no paper]

## UNVERIFIED — check manually before submission

11. **GC10-DET dataset** — commonly cited as: X. Lv, B. Dazi, L. He, D. Wang, S. Jin.
    "GC10-DET: An Industrial Surface Defect Detection Dataset Using Two-Stage Cascade
    Network." IEEE Transactions on Industrial Informatics, 2023.
    STATUS: title/author list NOT confirmable via Crossref or arXiv from this
    environment (proxy 403 on Semantic Scholar/OpenAlex). A verified adjacent source
    that documents and benchmarks GC10-DET: PeerJ Computer Science 2024,
    DOI 10.7717/peerjcs.1727 (Crossref-confirmed components: "GC10-DET deep metal
    surface defect dataset defect types", "Detection results of GC10-DET dataset").
    ACTION: confirm the TII citation (authors, volume, DOI) from the GC10-DET GitHub
    README (https://github.com/lvxiaoming2020/GC10-DET) before citing; otherwise cite
    the PeerJ CS paper + GitHub release as the dataset provenance.

12. **CLAHE** — K. Zuiderveld. "Contrast Limited Adaptive Histogram Equalization."
    Graphics Gems IV, Academic Press, 1994, pp. 474–485.
    STATUS: canonical chapter; widely cited DOI 10.1016/B978-0-12-336156-1.50061-7
    not re-verified here (Crossref search rate-limited). ACTION: verify DOI string.

13. **1–2 more 2022–2025 steel-defect YOLO papers using attention or CLAHE**
    (for related work): candidates to search on arXiv when the proxy allows:
    "steel surface defect detection YOLO attention", "CLAHE YOLO defect".
    Must only cite after verification.

## Notes for the rewrite

- Do NOT claim "first joint illumination-attention study" or "first zero-shot
  cross-mill protocol" — items 1, 2, 4, 8, 9 already establish prior art for
  attention-in-YOLO on NEU-DET/GC10-DET, and CLAHE-in-YOLO pipelines are common.
- The defensible novelty statement: a rigorously audited, matched-protocol,
  CI-reported empirical study of illumination normalisation + lightweight attention
  for steel defect detection, including whole-dataset class-agnostic zero-shot
  transfer evaluation with explicit confound controls (scale, tiling, label space,
  input distribution).
