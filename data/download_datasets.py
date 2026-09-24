"""Download and prepare NEU-DET and GC10-DET datasets.

- NEU-DET: Downloaded from verified open mirror (KeenForgeAI/NEU-DET-corrected),
  pooled across splits and re-split with a fixed 70/15/15 train/val/test
  class-stratified split (seed 42).
- GC10-DET: Downloaded from dronefreak/GC10-DET in YOLO format. Used strictly
  for cross-dataset evaluation (never trained on).
- Visual verification: Renders sample images with ground truth bounding boxes.

Usage:
    python data/download_datasets.py --data-dir ./data
"""

import argparse
import os
import shutil
import random
import cv2
import yaml
from pathlib import Path
from collections import defaultdict
from huggingface_hub import snapshot_download


NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

GC10_CLASSES = [
    "crease",
    "crescent_gap",
    "inclusion",
    "oil_spot",
    "punching_hole",
    "rolled_pit",
    "silk_spot",
    "waist_folding",
    "water_spot",
    "welding_line",
]


def download_and_prepare_neu_det(data_dir: Path, seed: int = 42) -> Path:
    """Download NEU-DET and establish a fixed 70/15/15 train/val/test split."""
    target_dir = data_dir / "NEU-DET"
    raw_dir = data_dir / "raw_neu_det"

    if (target_dir / "data.yaml").exists() and (target_dir / "images" / "train").exists():
        train_count = len(list((target_dir / "images" / "train").glob("*.jpg")))
        val_count = len(list((target_dir / "images" / "val").glob("*.jpg")))
        test_count = len(list((target_dir / "images" / "test").glob("*.jpg")))
        if train_count > 0 and val_count > 0 and test_count > 0:
            print(f"NEU-DET already prepared at {target_dir}: train={train_count}, val={val_count}, test={test_count}")
            return target_dir

    print("Downloading NEU-DET from HuggingFace mirror (KeenForgeAI/NEU-DET-corrected)...")
    snapshot_download(
        repo_id="KeenForgeAI/NEU-DET-corrected",
        repo_type="dataset",
        local_dir=str(raw_dir),
        allow_patterns=["images/**", "labels/**", "data.yaml"],
        max_workers=16,
    )

    print("Collecting all images and labels for 70/15/15 stratified split...")
    # Gather all images across raw folders
    all_images = {}
    for img_path in raw_dir.rglob("*.jpg"):
        all_images[img_path.stem] = img_path

    # Gather matching label files
    all_labels = {}
    for lbl_path in raw_dir.rglob("*.txt"):
        if lbl_path.stem in all_images:
            all_labels[lbl_path.stem] = lbl_path

    print(f"Found {len(all_images)} images and {len(all_labels)} matching labels.")

    # Group by primary class prefix (e.g. 'crazing', 'inclusion', etc.)
    class_groups = defaultdict(list)
    for stem, img_path in all_images.items():
        if stem not in all_labels:
            continue
        # Extract class prefix from filename
        prefix = stem.split("_")[0]
        if prefix == "rolled-in":
            cls_name = "rolled-in_scale"
        elif prefix == "pitted":
            cls_name = "pitted_surface"
        else:
            cls_name = prefix
        class_groups[cls_name].append(stem)

    print("Images per class category:")
    for cls_name, items in sorted(class_groups.items()):
        print(f"  {cls_name}: {len(items)}")

    # Deterministic stratified 70/15/15 split
    rng = random.Random(seed)
    train_stems, val_stems, test_stems = [], [], []

    for cls_name, stems in sorted(class_groups.items()):
        shuffled = list(stems)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        train_stems.extend(shuffled[:n_train])
        val_stems.extend(shuffled[n_train:n_train + n_val])
        test_stems.extend(shuffled[n_train + n_val:])

    print(f"Split results (seed {seed}): train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)}")

    # Create destination directories
    splits = {
        "train": train_stems,
        "val": val_stems,
        "test": test_stems,
    }

    for split_name, stems in splits.items():
        img_dst = target_dir / "images" / split_name
        lbl_dst = target_dir / "labels" / split_name
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            shutil.copy2(str(all_images[stem]), str(img_dst / f"{stem}.jpg"))
            shutil.copy2(str(all_labels[stem]), str(lbl_dst / f"{stem}.txt"))

    # Create data.yaml
    data_yaml_content = {
        "path": str(target_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(NEU_CLASSES),
        "names": {i: name for i, name in enumerate(NEU_CLASSES)},
    }
    with open(target_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml_content, f, sort_keys=False)

    print(f"NEU-DET prepared successfully at {target_dir}")
    return target_dir


def download_and_prepare_gc10_det(data_dir: Path) -> Path:
    """Download GC10-DET dataset for zero-shot cross-dataset evaluation."""
    target_dir = data_dir / "GC10-DET"
    raw_dir = data_dir / "raw_gc10_det"

    if (target_dir / "data.yaml").exists() and (target_dir / "images" / "test").exists():
        test_count = len(list((target_dir / "images" / "test").glob("*.jpg")))
        if test_count > 0:
            print(f"GC10-DET already prepared at {target_dir} ({test_count} test images)")
            return target_dir

    print("Downloading GC10-DET from HuggingFace mirror (dronefreak/GC10-DET)...")
    snapshot_download(
        repo_id="dronefreak/GC10-DET",
        repo_type="dataset",
        local_dir=str(raw_dir),
        allow_patterns=["data/**"],
        max_workers=16,
    )

    data_src = raw_dir / "data"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Move images and labels into target_dir
    for split in ["train", "valid", "test"]:
        dst_split = "val" if split == "valid" else split
        src_img = data_src / "images" / split
        src_lbl = data_src / "labels" / split
        dst_img = target_dir / "images" / dst_split
        dst_lbl = target_dir / "labels" / dst_split

        if src_img.exists():
            dst_img.mkdir(parents=True, exist_ok=True)
            for f in src_img.glob("*.*"):
                shutil.copy2(str(f), str(dst_img / f.name))

        if src_lbl.exists():
            dst_lbl.mkdir(parents=True, exist_ok=True)
            for f in src_lbl.glob("*.txt"):
                shutil.copy2(str(f), str(dst_lbl / f.name))

    data_yaml_content = {
        "path": str(target_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(GC10_CLASSES),
        "names": {i: name for i, name in enumerate(GC10_CLASSES)},
    }
    with open(target_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml_content, f, sort_keys=False)

    print(f"GC10-DET prepared successfully at {target_dir}")
    return target_dir


def render_sample_annotations(
    dataset_dir: Path,
    output_dir: Path,
    classes: list,
    num_samples: int = 6,
    split: str = "train",
):
    """Render sample images with bounding boxes overlaid to visually verify annotations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    img_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not img_files:
        print(f"No images found in {images_dir}")
        return

    # Pick samples from different classes
    samples_by_class = defaultdict(list)
    for img_p in img_files:
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        with open(lbl_p, "r") as f:
            lines = [line.strip().split() for line in f if line.strip()]
        if not lines:
            continue
        cls_id = int(lines[0][0])
        if len(samples_by_class[cls_id]) < 2:
            samples_by_class[cls_id].append((img_p, lines))

    rendered_count = 0
    colors = [
        (0, 255, 0),
        (0, 165, 255),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
        (0, 128, 255),
        (128, 255, 0),
    ]

    for cls_id in sorted(samples_by_class.keys()):
        for img_p, lines in samples_by_class[cls_id]:
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            h, w = img.shape[:2]

            for line in lines:
                cid = int(line[0])
                cx = float(line[1]) * w
                cy = float(line[2]) * h
                bw = float(line[3]) * w
                bh = float(line[4]) * h
                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2)
                y2 = int(cy + bh / 2)

                cname = classes[cid] if cid < len(classes) else str(cid)
                color = colors[cid % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img,
                    cname,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            out_name = f"{dataset_dir.name}_{split}_{img_p.stem}_verified.jpg"
            out_p = output_dir / out_name
            cv2.imwrite(str(out_p), img)
            rendered_count += 1
            if rendered_count >= num_samples:
                break
        if rendered_count >= num_samples:
            break

    print(f"Rendered {rendered_count} annotation samples to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument("--data-dir", type=str, default="iama-net/data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: NEU-DET ===")
    neu_path = download_and_prepare_neu_det(data_dir, seed=args.seed)

    print("\n=== Step 2: GC10-DET ===")
    gc10_path = download_and_prepare_gc10_det(data_dir)

    print("\n=== Step 3: Annotation Verification Samples ===")
    samples_dir = data_dir / "samples_annotated"
    render_sample_annotations(neu_path, samples_dir, NEU_CLASSES, num_samples=6, split="train")
    render_sample_annotations(gc10_path, samples_dir, GC10_CLASSES, num_samples=6, split="test")

    print("\nData preparation complete!")
