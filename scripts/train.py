"""Training script for IAMA-Net ablation study.

Supports configs M1-M4:
    M1: Baseline YOLOv11s (no preprocessing, no added attention)
    M2: CLAHE+bilateral preprocessing, no added attention
    M3: No preprocessing, with ECA+SpatialAttn attention
    M4: Full IAMA-Net (preprocessing + attention)

Usage:
    python scripts/train.py --config M1 --seed 42
    python scripts/train.py --config M4 --seed 42 --epochs 100
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_config(config_name: str) -> dict:
    """Get training configuration for the specified ablation config.

    Args:
        config_name: One of M1, M2, M3, M4.

    Returns:
        Dictionary with config parameters.
    """
    configs = {
        "M1": {
            "name": "M1_baseline",
            "description": "Baseline YOLOv11s",
            "use_preprocessing": False,
            "use_attention": False,
            "model_yaml": str(PROJECT_ROOT / "configs" / "yolo11s_baseline.yaml"),
        },
        "M2": {
            "name": "M2_preprocessing",
            "description": "YOLOv11s + CLAHE/bilateral",
            "use_preprocessing": True,
            "use_attention": False,
            "model_yaml": str(PROJECT_ROOT / "configs" / "yolo11s_baseline.yaml"),
        },
        "M3": {
            "name": "M3_attention",
            "description": "YOLOv11s + ECA+SpatialAttn",
            "use_preprocessing": False,
            "use_attention": True,
            "model_yaml": str(PROJECT_ROOT / "configs" / "yolo11s_iama.yaml"),
        },
        "M4": {
            "name": "M4_full",
            "description": "Full IAMA-Net (preprocessing + attention)",
            "use_preprocessing": True,
            "use_attention": True,
            "model_yaml": str(PROJECT_ROOT / "configs" / "yolo11s_iama.yaml"),
        },
    }
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Must be one of {list(configs.keys())}")
    return configs[config_name]


def train(config_name: str, seed: int, epochs: int, batch_size: int,
          imgsz: int, data_yaml: str, device: str, pretrained: str) -> Path:
    """Run a single training run.

    Args:
        config_name: Ablation config (M1/M2/M3/M4).
        seed: Random seed.
        epochs: Number of training epochs.
        batch_size: Batch size.
        imgsz: Image size.
        data_yaml: Path to dataset YAML.
        device: Device string (e.g. '0' for GPU 0, 'cpu').
        pretrained: Path to pretrained weights.

    Returns:
        Path to the training run output directory.
    """
    from ultralytics import YOLO
    import torch
    import numpy as np
    import random

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cfg = get_config(config_name)
    run_name = f"{cfg['name']}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = PROJECT_ROOT / "runs" / run_name

    print(f"\n{'='*60}")
    print(f"Training: {cfg['description']}")
    print(f"Config: {config_name} | Seed: {seed} | Epochs: {epochs}")
    print(f"Model YAML: {cfg['model_yaml']}")
    print(f"Data YAML: {data_yaml}")
    print(f"Run dir: {run_dir}")
    print(f"{'='*60}\n")

    # Handle preprocessing: if needed, preprocess the dataset first
    actual_data_yaml = data_yaml
    if cfg["use_preprocessing"]:
        from preprocessing.illumination import preprocess_dataset
        actual_data_yaml = _preprocess_data(data_yaml, seed)
        print(f"Using preprocessed data: {actual_data_yaml}")

    # Build model
    if pretrained:
        print(f"Loading {cfg['model_yaml']} with pretrained weights: {pretrained}")
        model = YOLO(cfg["model_yaml"]).load(pretrained)
    else:
        print(f"Building {cfg['model_yaml']} from scratch (no pretrained weights)")
        model = YOLO(cfg["model_yaml"])

    # Train
    results = model.train(
        data=actual_data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        seed=seed,
        device=device,
        workers=2,
        project=str(PROJECT_ROOT / "runs"),
        name=run_name,
        exist_ok=True,
        save=True,
        save_period=25,  # Save checkpoint every 25 epochs
        plots=True,
        val=True,
        verbose=True,
        deterministic=True,
        # Fixed augmentation settings for fair comparison
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    print(f"\nTraining complete. Results saved to: {run_dir}")
    return run_dir


def _preprocess_data(data_yaml: str, seed: int) -> str:
    """Create a preprocessed copy of the dataset and return new data YAML path.

    Args:
        data_yaml: Path to original data YAML.
        seed: Random seed.

    Returns:
        Path to new data YAML pointing to preprocessed images.
    """
    from preprocessing.illumination import preprocess_dataset

    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)

    data_yaml_path = Path(data_yaml).resolve()
    base_path = data_yaml_path.parent
    preproc_dir = base_path.parent / f"{base_path.name}_preprocessed"
    new_yaml_path = preproc_dir / "data.yaml"

    if not new_yaml_path.exists() or not (preproc_dir / "images").exists():
        print(f"Preprocessing dataset: {base_path} -> {preproc_dir}")
        preprocess_dataset(base_path, preproc_dir)
        new_cfg = data_cfg.copy()
        new_cfg["path"] = str(preproc_dir)
        with open(new_yaml_path, 'w') as f:
            yaml.dump(new_cfg, f, default_flow_style=False)
        print(f"Preprocessed dataset ready at: {preproc_dir}")
    else:
        print(f"Preprocessed dataset already exists at: {preproc_dir}")

    return str(new_yaml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IAMA-Net ablation configs")
    parser.add_argument("--config", type=str, required=True,
                        choices=["M1", "M2", "M3", "M4"],
                        help="Ablation config to train")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to dataset YAML")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: 0 for GPU, cpu for CPU")
    parser.add_argument("--pretrained", type=str, default="yolo11s.pt",
                        help="Pretrained weights (default: yolo11s.pt)")
    args = parser.parse_args()

    # Default data path
    if args.data is None:
        args.data = str(PROJECT_ROOT / "data" / "NEU-DET" / "data.yaml")

    train(
        config_name=args.config,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        data_yaml=args.data,
        device=args.device,
        pretrained=args.pretrained,
    )
