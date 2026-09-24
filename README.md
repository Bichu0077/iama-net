# IAMA-Net: Illumination-Aware Multi-scale Attention Network

Real-time metal surface defect detection using YOLOv11s with CLAHE preprocessing
and ECA + Spatial Attention modules.

## Project Structure

```
iama-net/
├── README.md                  # This file
├── requirements.txt           # Pinned dependencies
├── data/
│   ├── download_datasets.py   # Download NEU-DET and GC10-DET
│   └── convert_to_yolo.py     # Format conversion and splitting
├── configs/
│   ├── yolo11s_baseline.yaml  # Baseline YOLOv11s config
│   └── yolo11s_iama.yaml      # IAMA-Net config (with ECA + SpatialAttn)
├── preprocessing/
│   └── illumination.py        # CLAHE + bilateral filter
├── models/
│   └── attention.py           # ECA, SpatialAttn modules
├── scripts/
│   ├── train.py               # Training script (--config M1/M2/M3/M4)
│   ├── evaluate.py            # Evaluation metrics
│   ├── gradcam.py             # Grad-CAM visualization
│   └── cross_dataset_eval.py  # Cross-dataset generalization test
├── runs/                      # Training checkpoints and logs
└── results/
    ├── ablation_table.csv     # Final ablation results
    ├── gradcam_examples/      # Grad-CAM heatmap overlays
    └── results_summary.md     # Hypothesis validation summary
```

## Ablation Study

| Config | CLAHE+bilateral | Added Attention | Description |
|--------|----------------|-----------------|-------------|
| M1     | No             | No              | Baseline YOLOv11s |
| M2     | Yes            | No              | Preprocessing only |
| M3     | No             | Yes             | Attention only |
| M4     | Yes            | Yes             | Full IAMA-Net |

## Quick Start

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Download datasets
python data/download_datasets.py

# Train M1 baseline
python scripts/train.py --config M1 --seed 42 --epochs 100

# Evaluate
python scripts/evaluate.py --weights runs/M1_baseline_seed42/weights/best.pt --data data/NEU-DET/data.yaml
```

## Datasets

- **NEU-DET**: 1,800 images, 6 classes (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches)
- **GC10-DET**: ~2,300-3,570 images, 10 classes (cross-dataset eval only, never trained on)

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- See requirements.txt for full dependency list
