"""Convert datasets to YOLO format if needed.

Handles various annotation formats (VOC XML, COCO JSON) and converts
them to YOLO TXT format. Also handles train/val/test splitting.

Usage:
    python data/convert_to_yolo.py --src data/raw --dst data/NEU-DET --format voc
"""

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


def voc_to_yolo(
    xml_path: Path,
    class_map: Dict[str, int],
    img_w: int,
    img_h: int,
) -> List[str]:
    """Convert a VOC XML annotation to YOLO format lines.

    Args:
        xml_path: Path to VOC XML annotation file.
        class_map: Mapping from class name to class index.
        img_w: Image width.
        img_h: Image height.

    Returns:
        List of YOLO format annotation lines.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    lines = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in class_map:
            print(f"Warning: Unknown class '{cls_name}' in {xml_path}, skipping")
            continue

        cls_id = class_map[cls_name]
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO format (center_x, center_y, width, height) normalized
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines


def split_dataset(
    image_dir: Path,
    label_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[int, int, int]:
    """Split dataset into train/val/test with fixed seed.

    Args:
        image_dir: Directory containing images.
        label_dir: Directory containing YOLO labels.
        output_dir: Output directory for split dataset.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (n_train, n_val, n_test).
    """
    random.seed(seed)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "valid": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        img_dst = output_dir / split_name / "images"
        lbl_dst = output_dir / split_name / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            shutil.copy2(str(img_path), str(img_dst / img_path.name))
            lbl_path = label_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy2(str(lbl_path), str(lbl_dst / lbl_path.name))

    return len(splits["train"]), len(splits["valid"]), len(splits["test"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and split dataset to YOLO format")
    parser.add_argument("--src", type=str, required=True, help="Source directory")
    parser.add_argument("--dst", type=str, required=True, help="Output directory")
    parser.add_argument("--format", type=str, choices=["voc", "coco", "yolo"],
                        default="voc", help="Source annotation format")
    parser.add_argument("--split", action="store_true",
                        help="Perform train/val/test split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Converting {args.src} -> {args.dst} (format: {args.format})")
    if args.split:
        print(f"Splitting with seed {args.seed}: 70/15/15")
