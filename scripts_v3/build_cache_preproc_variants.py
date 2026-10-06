"""Phase 3.1: build preprocessing-variant caches of NEU-DET (in-domain controls).

Variants (all clip=2.0, tile 8x8 — OpenCV semantics):
  NEU-DET_gray_clahe : CLAHE directly on the single grayscale channel, no LAB
                       round-trip, NO bilateral filter. Saved as 3 identical
                       channels (matches how the network consumes JPEGs).
  NEU-DET_lab_nobil  : the OLD LAB route (CLAHE on L*, merge, back to BGR),
                       but NO bilateral filter — isolates the bilateral term.

Both copy labels unchanged and write their own data.yaml.
Rationale (review finding C5): images are grayscale, so the LAB detour is
moot and may inject chroma noise; bilateral sigma_color=50 may smooth the
faint (<15 gray levels) crazing cracks it was supposed to protect.
"""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from iama_env import NEU_RAW, PROJECT_ROOT

SPLITS = ("train", "val", "test")
VARIANTS = {
    "NEU-DET_gray_clahe": "gray_clahe",
    "NEU-DET_lab_nobil": "lab_clahe_nobilateral",
}


def gray_clahe(img_bgr, clip=2.0, tile=(8, 8)):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    g = clahe.apply(g)
    return cv2.merge([g, g, g])


def lab_clahe_nobilateral(img_bgr, clip=2.0, tile=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


FNS = {"gray_clahe": gray_clahe, "lab_clahe_nobilateral": lab_clahe_nobilateral}


def main():
    for ds_name, kind in VARIANTS.items():
        dst_root = PROJECT_ROOT / "data" / ds_name
        fn = FNS[kind]
        for split in SPLITS:
            src_img = NEU_RAW / "images" / split
            src_lab = NEU_RAW / "labels" / split
            dst_img = dst_root / "images" / split
            dst_lab = dst_root / "labels" / split
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lab.mkdir(parents=True, exist_ok=True)
            files = sorted(src_img.glob("*.jpg"))
            for n, p in enumerate(files):
                dst = dst_img / p.name
                if dst.exists() and dst.stat().st_size > 0:
                    continue
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                out = fn(img)
                # verify grayscale variants stay channel-identical (gray_clahe) or not (lab)
                cv2.imwrite(str(dst), out)
                if n % 400 == 0:
                    print(f"{ds_name}/{split}: {n}/{len(files)}", flush=True)
            for p in src_lab.glob("*.txt"):
                d = dst_lab / p.name
                if not d.exists():
                    shutil.copy2(str(p), str(d))
            print(f"{ds_name}/{split}: done {len(files)}", flush=True)
        yaml_text = f'''path: "{dst_root.as_posix()}"
train: images/train
val: images/val
test: images/test
nc: 6
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
'''
        (dst_root / "data.yaml").write_text(yaml_text)
        print("wrote", dst_root / "data.yaml", flush=True)

    # channel-structure verification (feeds the audit claim about chroma noise)
    for ds_name in VARIANTS:
        counts = {"identical": 0, "color": 0}
        for split in SPLITS:
            for p in (PROJECT_ROOT / "data" / ds_name / "images" / split).glob("*.jpg"):
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2]):
                    counts["identical"] += 1
                else:
                    counts["color"] += 1
        print(f"{ds_name} channel check: {counts}", flush=True)


if __name__ == "__main__":
    main()
