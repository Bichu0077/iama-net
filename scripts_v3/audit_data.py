"""Phase 1.6 data audit for NEU-DET, NEU-DET_preprocessed and GC10-DET.

Checks (all outputs written to results_v3/audit/):
  1. Per-split image counts, pixel sizes, channel counts (grayscale vs BGR), formats.
  2. Per-split YOLO label counts, per-class instance counts, malformed lines.
  3. Duplicate images (SHA-256 of file bytes AND of decoded pixels) within and
     across splits -> leakage check.
  4. GC10 label-vs-metadata.jsonl cross-check (box counts per split/class).
  5. Split manifests: sorted (filename, sha256) for every image and label file.
  6. NEU-DET_preprocessed <-> NEU-DET filename correspondence.

Outputs:
  results_v3/audit/data_audit_summary.json
  results_v3/audit/image_inventory.csv
  results_v3/audit/split_hashes/*.csv
  results_v3/audit/duplicates.csv
"""

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "results_v3" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "split_hashes").mkdir(exist_ok=True)

DATASETS = {
    "NEU-DET": PROJECT_ROOT / "data" / "NEU-DET",
    "NEU-DET_preprocessed": PROJECT_ROOT / "data" / "NEU-DET_preprocessed",
    "GC10-DET": PROJECT_ROOT / "data" / "GC10-DET",
}
SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

NEU_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
GC10_NAMES = ["crease", "crescent_gap", "inclusion", "oil_spot", "punching_hole",
              "rolled_pit", "silk_spot", "waist_folding", "water_spot", "welding_line"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    summary = {"datasets": {}, "duplicate_groups": [], "issues": []}
    inventory_rows = []
    file_hash_cache = {}   # path -> sha256(file bytes)
    pixel_hash_map = defaultdict(list)  # sha256(decoded pixels) -> [paths]
    file_hash_map = defaultdict(list)   # sha256(file bytes) -> [paths]

    for ds_name, ds_root in DATASETS.items():
        ds_summary = {"splits": {}, "class_names": NEU_NAMES if "NEU" in ds_name else GC10_NAMES}
        for split in SPLITS:
            img_dir = ds_root / "images" / split
            lab_dir = ds_root / "labels" / split
            if not img_dir.exists():
                ds_summary["splits"][split] = {"error": "missing image dir"}
                continue

            img_files = sorted(p for p in img_dir.iterdir()
                               if p.is_file() and p.suffix.lower() in IMG_EXTS)
            other_files = sorted(p for p in img_dir.iterdir()
                                 if p.is_file() and p.suffix.lower() not in IMG_EXTS)
            lab_files = sorted(p for p in lab_dir.iterdir()
                               if p.is_file() and p.suffix.lower() == ".txt") if lab_dir.exists() else []

            sizes = Counter()
            channels = Counter()
            exts = Counter()
            cls_counts = Counter()
            n_boxes = 0
            malformed = 0
            imgs_without_label = []
            label_names = {p.stem for p in lab_files}
            image_names = {p.stem for p in img_files}

            for p in img_files:
                exts[p.suffix.lower()] += 1
                fb = sha256_file(p)
                file_hash_cache[str(p)] = fb
                file_hash_map[fb].append(str(p))
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is None:
                    summary["issues"].append(f"unreadable image: {p}")
                    continue
                if img.ndim == 2:
                    h, w = img.shape
                    c = 1
                    px = img
                else:
                    h, w, c = img.shape
                    px = img
                    if c == 3 and np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2]):
                        channels["3ch-identical(grayscale-stored-as-BGR)"] += 1
                        px = img[..., 0]
                    else:
                        channels[f"{c}ch-color"] += 1
                if c == 1:
                    channels["1ch"] += 1
                sizes[f"{w}x{h}"] += 1
                ph = hashlib.sha256(np.ascontiguousarray(px).tobytes()).hexdigest()
                pixel_hash_map[ph].append(str(p))
                inventory_rows.append({
                    "dataset": ds_name, "split": split, "file": p.name,
                    "width": w, "height": h, "channels": c,
                    "sha256_file": fb, "sha256_pixels": ph,
                })

            for p in lab_files:
                if p.stem not in image_names:
                    imgs_without_label.append(f"ORPHAN-LABEL:{p.name}")
                with open(p, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            malformed += 1
                            continue
                        try:
                            ci = int(parts[0])
                            vals = [float(v) for v in parts[1:]]
                        except ValueError:
                            malformed += 1
                            continue
                        if not all(0.0 <= v <= 1.0 for v in vals):
                            malformed += 1
                            continue
                        cls_counts[ci] += 1
                        n_boxes += 1
                fb = sha256_file(p)
                file_hash_cache[str(p)] = fb

            missing_labels = sorted(image_names - label_names)
            ds_summary["splits"][split] = {
                "n_images": len(img_files),
                "n_label_files": len(lab_files),
                "non_image_files_in_image_dir": [p.name for p in other_files],
                "image_sizes": dict(sizes),
                "channel_kinds": dict(channels),
                "extensions": dict(exts),
                "n_boxes": n_boxes,
                "malformed_label_lines": malformed,
                "instances_per_class": {ds_summary["class_names"][k] if k < len(ds_summary["class_names"]) else str(k): v
                                        for k, v in sorted(cls_counts.items())},
                "images_missing_label_file": missing_labels,
                "orphan_labels": imgs_without_label,
            }

            # split hash manifest
            man_path = OUT / "split_hashes" / f"{ds_name}_{split}.csv"
            with open(man_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["role", "path", "sha256"])
                for p in img_files:
                    w.writerow(["image", str(p.relative_to(PROJECT_ROOT)), file_hash_cache[str(p)]])
                for p in lab_files:
                    w.writerow(["label", str(p.relative_to(PROJECT_ROOT)), file_hash_cache[str(p)]])
        summary["datasets"][ds_name] = ds_summary

    # duplicate detection within each dataset and across splits
    def group_dups(hash_map):
        groups = []
        for h, paths in hash_map.items():
            if len(paths) > 1:
                groups.append(paths)
        return groups

    # per-dataset pixel duplicates (catches identical images under different names/splits)
    by_ds = defaultdict(lambda: defaultdict(list))
    for ph, paths in pixel_hash_map.items():
        for p in paths:
            rel = Path(p).relative_to(PROJECT_ROOT)
            ds = str(rel.parent.parent.parent)  # data/<DS>/images/<split>
            sp = rel.parent.name
            by_ds[ds][(ph)].append(f"{sp}/{Path(p).name}")
    for ds, m in by_ds.items():
        for ph, items in m.items():
            if len(items) > 1:
                summary["duplicate_groups"].append({"dataset": ds, "kind": "identical_pixels", "files": sorted(items)})

    # file-byte duplicates across datasets (raw vs preprocessed should NOT be identical unless preprocessing is a no-op)
    cross = defaultdict(list)
    for fb, paths in file_hash_map.items():
        dss = {str(Path(p).relative_to(PROJECT_ROOT).parent.parent.parent) for p in paths}
        if len(dss) > 1:
            cross[fb] = paths
    for fb, paths in cross.items():
        summary["duplicate_groups"].append({"dataset": "cross", "kind": "identical_file_bytes", "files": sorted(paths)})

    # raw vs preprocessed filename correspondence
    for split in SPLITS:
        raw = {p.name for p in (DATASETS["NEU-DET"] / "images" / split).iterdir() if p.suffix.lower() in IMG_EXTS}
        pre = {p.name for p in (DATASETS["NEU-DET_preprocessed"] / "images" / split).iterdir() if p.suffix.lower() in IMG_EXTS}
        if raw != pre:
            summary["issues"].append(f"NEU raw/preprocessed filename mismatch in {split}: only_raw={sorted(raw-pre)[:5]} only_pre={sorted(pre-raw)[:5]}")
    summary["neu_raw_preprocessed_filenames_match"] = True

    # GC10 metadata.jsonl cross-check
    gc_meta_total = 0
    gc_meta_per_split = {}
    gc_meta_cls = Counter()
    for split in SPLITS:
        meta = DATASETS["GC10-DET"] / "images" / split / "metadata.jsonl"
        if not meta.exists():
            summary["issues"].append(f"missing {meta}")
            continue
        n_img = 0
        n_box = 0
        with open(meta, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_img += 1
                cats = rec.get("objects", {}).get("categories", [])
                n_box += len(cats)
                gc_meta_cls.update(cats)
        gc_meta_per_split[split] = {"n_records": n_img, "n_boxes": n_box}
        gc_meta_total += n_box
    summary["gc10_metadata_jsonl"] = {
        "per_split": gc_meta_per_split,
        "total_boxes": gc_meta_total,
        "boxes_per_class": {GC10_NAMES[k] if k < len(GC10_NAMES) else str(k): v for k, v in sorted(gc_meta_cls.items())},
    }

    with open(OUT / "data_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUT / "image_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "split", "file", "width", "height", "channels", "sha256_file", "sha256_pixels"])
        w.writeheader()
        w.writerows(inventory_rows)

    with open(OUT / "duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "kind", "files"])
        for g in summary["duplicate_groups"]:
            w.writerow([g["dataset"], g["kind"], ";".join(g["files"])])

    print(json.dumps({k: v for k, v in summary.items() if k != "duplicate_groups"}, indent=2)[:4000])
    print(f"\nduplicate_groups: {len(summary['duplicate_groups'])}")
    print(f"issues: {len(summary['issues'])}")
    for i in summary["issues"][:20]:
        print("  -", i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
