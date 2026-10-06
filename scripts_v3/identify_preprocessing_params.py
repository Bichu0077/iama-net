"""Identify which preprocessing parameters were actually used to build
data/NEU-DET_preprocessed, by re-applying candidate parameter sets to raw
images and comparing against the stored preprocessed files (MAE / exact-match).

JPEG re-encoding prevents exact matching; the correct parameter set should
stand out with a clearly lower MAE. Writes results_v3/audit/preprocessing_param_id.csv.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from iama_env import NEU_RAW, NEU_PP, AUDIT_DIR, PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))
from preprocessing.illumination import illumination_normalize  # noqa: E402

CANDIDATES = {
    "clahe2.0_d5_s50 (illumination.py defaults)": dict(clip_limit=2.0, tile_grid=(8, 8), d=5, sigma_color=50.0, sigma_space=50.0),
    "clahe3.0_d7_s50 (grid-search best)": dict(clip_limit=3.0, tile_grid=(8, 8), d=7, sigma_color=50.0, sigma_space=50.0),
    "clahe2.0_d9_s75 (__main__ defaults)": dict(clip_limit=2.0, tile_grid=(8, 8), d=9, sigma_color=75.0, sigma_space=75.0),
    "clahe1.5_d7_s50": dict(clip_limit=1.5, tile_grid=(8, 8), d=7, sigma_color=50.0, sigma_space=50.0),
}


def grayscale_clahe(img, clip=2.0, tile=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(img)


def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for split in ("train", "val", "test"):
        raw_dir = NEU_RAW / "images" / split
        pp_dir = NEU_PP / "images" / split
        files = sorted(raw_dir.glob("*.jpg"))[:8]
        for f in files:
            pp = pp_dir / f.name
            if pp.exists():
                samples.append((f, pp))
    print(f"Using {len(samples)} sample images")

    rows = []
    for label, params in CANDIDATES.items():
        maes = []
        for raw_p, pp_p in samples:
            raw = cv2.imread(str(raw_p), cv2.IMREAD_COLOR)
            stored = cv2.imread(str(pp_p), cv2.IMREAD_COLOR)
            cand = illumination_normalize(raw, **params)
            maes.append(float(np.mean(np.abs(cand.astype(np.float32) - stored.astype(np.float32)))))
        rows.append({"method": label, "mean_abs_err_vs_stored": round(float(np.mean(maes)), 4),
                     "max_abs_err": round(float(np.max(maes)), 4)})
        print(f"{label}: MAE={np.mean(maes):.4f}")

    # extra controls: CLAHE only (no bilateral), grayscale CLAHE, and identity
    for label, fn in [
        ("clahe2.0-LAB-no-bilateral", lambda img: _clahe_lab(img, 2.0)),
        ("grayscale-clahe2.0-only", lambda img: cv2.merge([grayscale_clahe(img[..., 0])]*3)),
        ("identity (raw unchanged)", lambda img: img),
    ]:
        maes = []
        for raw_p, pp_p in samples:
            raw = cv2.imread(str(raw_p), cv2.IMREAD_COLOR)
            stored = cv2.imread(str(pp_p), cv2.IMREAD_COLOR)
            cand = fn(raw)
            maes.append(float(np.mean(np.abs(cand.astype(np.float32) - stored.astype(np.float32)))))
        rows.append({"method": label, "mean_abs_err_vs_stored": round(float(np.mean(maes)), 4),
                     "max_abs_err": round(float(np.max(maes)), 4)})
        print(f"{label}: MAE={np.mean(maes):.4f}")

    with open(AUDIT_DIR / "preprocessing_param_id.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "mean_abs_err_vs_stored", "max_abs_err"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["mean_abs_err_vs_stored"]))
    print(f"Saved {AUDIT_DIR / 'preprocessing_param_id.csv'}")


def _clahe_lab(img, clip):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


if __name__ == "__main__":
    main()
