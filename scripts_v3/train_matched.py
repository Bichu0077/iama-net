"""Phase 2.2/2.3: matched-protocol training for the fair ablation.

ALL variants:
  - start from the SAME COCO-pretrained YOLO11s weights (yolo11s.pt):
      M1/M2: direct load (ultralytics auto-transfers matching tensors,
             Detect head re-initialised for nc=6)
      M3/M4: aligned semantic layer mapping (scripts_v3/map_pretrained.py,
             runs_v3/init/iama_coco_mapped_init.pt) + identity attention init
             (zero weights, +4.0 bias). Transfer count logged in its manifest.
  - identical optimizer/schedule/augmentation/split/epochs
  - NO freezing, NO fine-tuning from converged NEU weights
  - seed per run; deterministic=True
  - writes runs_v3/{name}/manifest.json with full command, seed, git commit,
    epoch count, versions, timings; supports resume=True from last.pt.

Fixed environment constraints (AGENTS.md section 3):
  workers=2, batch=8, nbs=16 (accumulate=2), imgsz=640, AdamW lr0=0.0008
  lrf=0.01 wd=5e-4 cosine LR warmup_epochs=1.0 close_mosaic=0.

Usage:
  python scripts_v3/train_matched.py --variant M4 --seed 0 --epochs 50
  python scripts_v3/train_matched.py --variant M4 --seed 0 --resume
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from iama_env import register_iama_modules, PROJECT_ROOT, NEU_RAW, NEU_PP

register_iama_modules()

from ultralytics import YOLO  # noqa: E402
import torch, ultralytics  # noqa: E402

COCO_WEIGHTS = PROJECT_ROOT / "yolo11s.pt"
MAPPED_INIT = PROJECT_ROOT / "runs_v3" / "init" / "iama_coco_mapped_init.pt"

TRAIN_ARGS = dict(
    imgsz=640,
    batch=8,
    nbs=16,               # accumulate = nbs/batch = 2  (AGENTS.md constraint)
    optimizer="AdamW",
    lr0=0.0008,
    lrf=0.01,
    weight_decay=5e-4,
    cos_lr=True,
    warmup_epochs=1.0,
    close_mosaic=0,
    workers=2,
    device="0",
    deterministic=True,
    val=True,
    plots=True,
    save=True,
    exist_ok=False,
    # augmentation: ultralytics defaults, fixed explicitly for the record
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0,
)

VARIANTS = {
    # Phase 2 matched ablation
    "M1": dict(data=NEU_RAW / "data.yaml", init=("coco", "yolo11s")),
    "M2": dict(data=NEU_PP / "data.yaml", init=("coco", "yolo11s")),
    "M3": dict(data=NEU_RAW / "data.yaml", init=("mapped", "iama_coco_mapped_init.pt")),
    "M4": dict(data=NEU_PP / "data.yaml", init=("mapped", "iama_coco_mapped_init.pt")),
    # Phase 3.2 alternative attention (attention-only, raw inputs)
    "M3se": dict(data=NEU_RAW / "data.yaml", init=("mapped", "se_coco_mapped_init.pt")),
    "M3cbam": dict(data=NEU_RAW / "data.yaml", init=("mapped", "cbam_coco_mapped_init.pt")),
    "M3ca": dict(data=NEU_RAW / "data.yaml", init=("mapped", "ca_coco_mapped_init.pt")),
    # Phase 3.3 external reference detector
    "M1v8": dict(data=NEU_RAW / "data.yaml", init=("coco", "yolov8s")),
    # Phase 3.1 preprocessing controls (baseline arch)
    "M2gray": dict(data=PROJECT_ROOT / "data" / "NEU-DET_gray_clahe" / "data.yaml",
                   init=("coco", "yolo11s")),
    "M2nobil": dict(data=PROJECT_ROOT / "data" / "NEU-DET_lab_nobil" / "data.yaml",
                    init=("coco", "yolo11s")),
}


def build_model(variant: str) -> YOLO:
    kind, ref = VARIANTS[variant]["init"]
    if kind == "mapped":
        p = PROJECT_ROOT / "runs_v3" / "init" / ref
        if not p.exists():
            raise SystemExit(f"missing mapped init {p}; run scripts_v3/map_pretrained.py first")
        return YOLO(str(p))
    if ref == "yolo11s":
        coco = PROJECT_ROOT / "yolo11s.pt"
        if not coco.exists():
            YOLO("yolo11s.pt")  # trigger download
        return YOLO(str(PROJECT_ROOT / "configs" / "yolo11s_baseline.yaml")).load(str(coco))
    if ref == "yolov8s":
        coco = PROJECT_ROOT / "yolov8s.pt"
        if not coco.exists():
            YOLO("yolov8s.pt")  # trigger download
        return YOLO("yolov8s.yaml").load(str(coco))
    raise ValueError(ref)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=str(PROJECT_ROOT)).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--timing-only", action="store_true",
                    help="train 1 epoch into runs_v3/timing_* for wall-clock estimation")
    args = ap.parse_args()

    v = VARIANTS[args.variant]
    model = build_model(args.variant)

    name = (f"timing_{args.variant}_seed{args.seed}" if args.timing_only
            else f"{args.variant}_matched_seed{args.seed}")
    run_dir = PROJECT_ROOT / "runs_v3" / name
    epochs = 1 if args.timing_only else args.epochs

    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"cannot resume: {last} missing")
        print(f"resuming {name} from {last}")
        model = YOLO(str(last))
        model.train(resume=True)
        return

    cmd = " ".join(sys.orig_argv)
    manifest = {
        "run_name": name,
        "variant": args.variant,
        "seed": args.seed,
        "epochs_requested": epochs,
        "timing_only": args.timing_only,
        "command": cmd,
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "init_weights": str((PROJECT_ROOT / "runs_v3" / "init" / v["init"][1])
                            if v["init"][0] == "mapped" else
                            (PROJECT_ROOT / ("yolo11s.pt" if v["init"][1] == "yolo11s" else "yolov8s.pt"))),
        "data_yaml": str(v["data"]),
        "train_args": {k: str(x) for k, x in TRAIN_ARGS.items()},
        "started": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    t0 = time.time()
    model.train(
        data=str(v["data"]),
        epochs=epochs,
        seed=args.seed,
        project=str(PROJECT_ROOT / "runs_v3"),
        name=name,
        **{**TRAIN_ARGS, "exist_ok": True},
    )
    wall = time.time() - t0
    # record the ACTUAL save dir (ultralytics may suffix the name on collision)
    actual_dir = Path(getattr(model.trainer, "save_dir", run_dir))
    manifest["status"] = "completed"
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest["wall_clock_s"] = round(wall, 1)
    manifest["s_per_epoch"] = round(wall / epochs, 2)
    manifest["actual_save_dir"] = str(actual_dir)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if actual_dir != run_dir:
        actual_dir.mkdir(parents=True, exist_ok=True)
        (actual_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{name}: completed in {wall/60:.1f} min ({wall/epochs:.1f} s/epoch) -> {actual_dir}")


if __name__ == "__main__":
    main()
