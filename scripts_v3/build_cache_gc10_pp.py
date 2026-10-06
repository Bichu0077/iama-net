"""Build an offline CLAHE+bilateral preprocessed cache of GC10-DET.

Same parameters verified for NEU-DET_preprocessed: CLAHE clip=2.0 tile 8x8 on
LAB L-channel + bilateral d=5 sigma_color=50 sigma_space=50
(see results_v3/audit/preprocessing_param_id.csv).

GC10 images are single-channel JPEGs; cv2.imread(IMREAD_COLOR) replicates them
to 3 identical channels, matching how the NEU-DET cache was produced from
3-channel-identical JPEGs.

Writes:
  data/GC10-DET_preprocessed_v3/images/{train,val,test}/*.jpg
  data/GC10-DET_preprocessed_v3/labels/{train,val,test}/*.txt  (copies)
  data/GC10-DET_preprocessed_v3/data.yaml
"""

import shutil
import sys
import time
from pathlib import Path

import cv2

from iama_env import GC10_RAW, GC10_PP, PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))
from preprocessing.illumination import illumination_normalize  # noqa: E402

SPLITS = ("train", "val", "test")
PARAMS = dict(clip_limit=2.0, tile_grid=(8, 8), d=5, sigma_color=50.0, sigma_space=50.0)


def main():
    for split in SPLITS:
        src_img = GC10_RAW / "images" / split
        src_lab = GC10_RAW / "labels" / split
        dst_img = GC10_PP / "images" / split
        dst_lab = GC10_PP / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lab.mkdir(parents=True, exist_ok=True)

        files = sorted(p for p in src_img.glob("*.jpg"))
        t0 = time.time()
        for n, p in enumerate(files):
            dst = dst_img / p.name
            if dst.exists() and dst.stat().st_size > 0:
                continue
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)  # 1ch jpg -> 3 identical channels
            assert img is not None, f"cannot read {p}"
            out = illumination_normalize(img, **PARAMS)
            if not cv2.imwrite(str(dst), out):
                raise RuntimeError(f"cannot write {dst}")
            if n % 200 == 0:
                print(f"{split}: {n}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
        # copy labels
        for p in src_lab.glob("*.txt"):
            dst = dst_lab / p.name
            if not dst.exists():
                shutil.copy2(str(p), str(dst))
        print(f"{split}: done {len(files)} images in {time.time()-t0:.0f}s", flush=True)

    yaml_text = f'''path: "{GC10_PP.as_posix()}"
train: images/train
val: images/val
test: images/test
nc: 10
names:
  0: crease
  1: crescent_gap
  2: inclusion
  3: oil_spot
  4: punching_hole
  5: rolled_pit
  6: silk_spot
  7: waist_folding
  8: water_spot
  9: welding_line
'''
    (GC10_PP / "data.yaml").write_text(yaml_text)
    print("wrote", GC10_PP / "data.yaml")


if __name__ == "__main__":
    main()
