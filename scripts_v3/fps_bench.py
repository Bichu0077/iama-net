"""Phase 1.5: proper FPS benchmark for all five original checkpoints.

Protocol (fixed, documented):
  - batch size 1, single real NEU-DET test image (200x200 -> letterboxed 640)
    AND a random-noise 640x640x3 uint8 array (to show input-content effects
    and to replicate what the OLD scripts/evaluate.py measured)
  - fp32 and fp16 reported separately
  - 50 warm-up + 300 timed iterations per repeat, 5 repeats
  - wall-clock (perf_counter + cuda.synchronize) AND ultralytics internal
    stage breakdown (preprocess / inference / postprocess-NMS, ms)
  - nvidia-smi GPU state (SM clock, temp, power, util) captured before and
    after every repeat
  - all models timed in ONE session, fixed order M1, M2, M3, M4, ANCHOR,
    fp32 before fp16
  - mean +/- std over the 5 repeats

Outputs results_v3/audit/fps_benchmark.csv (per repeat) and
fps_benchmark_summary.csv (mean/std), plus gpu_state.csv.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from iama_env import register_iama_modules, CHECKPOINTS, AUDIT_DIR, NEU_RAW, PROJECT_ROOT

register_iama_modules()

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

WARMUP = 50
TIMED = 300
REPEATS = 5
IMGSZ = 640
DEVICE = "0"
ORDER = ["M1", "M2", "M3", "M4", "ANCHOR"]


def gpu_state(tag):
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True).strip()
        vals = [v.strip() for v in out.split(",")]
        return dict(tag=tag, clock_sm_mhz=vals[0], clock_max_mhz=vals[1],
                    temp_c=vals[2], power_w=vals[3], util_pct=vals[4], mem_used_mib=vals[5])
    except Exception as e:  # noqa: BLE001
        return dict(tag=tag, clock_sm_mhz=f"err:{e}")


def bench(model, source, half, tag, gpu_rows):
    g0 = gpu_state(f"{tag}_before")
    t_start = time.time()
    # warm-up
    for _ in range(WARMUP):
        model.predict(source, imgsz=IMGSZ, device=DEVICE, half=half, verbose=False)
    repeats = []
    for r in range(REPEATS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        speed_acc = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        for _ in range(TIMED):
            res = model.predict(source, imgsz=IMGSZ, device=DEVICE, half=half, verbose=False)[0]
            speed_acc["preprocess"] += res.speed["preprocess"]
            speed_acc["inference"] += res.speed["inference"]
            speed_acc["postprocess"] += res.speed["postprocess"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        repeats.append({
            "repeat": r + 1, "wall_s": round(wall, 4),
            "fps_wall": round(TIMED / wall, 2),
            "pre_ms": round(speed_acc["preprocess"] / TIMED, 3),
            "inf_ms": round(speed_acc["inference"] / TIMED, 3),
            "nms_ms": round(speed_acc["postprocess"] / TIMED, 3),
        })
    g1 = gpu_state(f"{tag}_after")
    g1["wall_time_total_s"] = round(time.time() - t_start, 1)
    gpu_rows.append(g0)
    gpu_rows.append(g1)
    return repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=None, help="JSON {model_id: weights_path}")
    ap.add_argument("--order", default=None, help="comma-separated model ids (default: registry order)")
    ap.add_argument("--out-prefix", default="", help="prefix for output CSV names")
    args = ap.parse_args()

    registry = {k: (Path(v) if Path(v).is_absolute() else PROJECT_ROOT / v)
                for k, v in json.loads(Path(args.registry).read_text()).items()} \
        if args.registry else dict(CHECKPOINTS)
    order = args.order.split(",") if args.order else list(registry)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())

    real_img = str(sorted((NEU_RAW / "images" / "test").glob("*.jpg"))[0])
    noise_img = np.random.default_rng(0).integers(0, 256, (IMGSZ, IMGSZ, 3), dtype=np.uint8)
    sources = {"real_neu_test_image": real_img, "random_noise_640": noise_img}

    rows, gpu_rows = [], []
    for model_id in order:
        print(f"\n=== {model_id} ===", flush=True)
        model = YOLO(str(registry[model_id]))
        # NOTE: no explicit model.fuse() — ultralytics fuses at checkpoint load;
        # a second fuse() crashes on the hand-saved ANCHOR checkpoint
        # ("requires_grad flags of leaf variables"). Fusion state is recorded.
        n_fused = sum(1 for m in model.model.modules()
                      if type(m).__name__ == "Conv" and hasattr(m, "forward_fuse"))
        print(f"  modules: {len(list(model.model.modules()))}, conv count: {n_fused}", flush=True)
        for half in (False, True):
            prec = "fp16" if half else "fp32"
            for src_name, src in sources.items():
                tag = f"{model_id}_{prec}_{src_name}"
                reps = bench(model, src, half, tag, gpu_rows)
                for rep in reps:
                    rows.append({"model": model_id, "precision": prec, "source": src_name, **rep})
                fps = [r["fps_wall"] for r in reps]
                print(f"  {prec} {src_name}: fps={np.mean(fps):.2f} +/- {np.std(fps, ddof=1):.2f}", flush=True)
                with open(AUDIT_DIR / f"{args.out_prefix}fps_benchmark.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
        del model
        torch.cuda.empty_cache()
        time.sleep(15)  # cool-down between models to reduce thermal order effects

    # summary
    summary = []
    for model_id in order:
        for prec in ("fp32", "fp16"):
            for src_name in sources:
                sel = [r for r in rows if r["model"] == model_id and r["precision"] == prec
                       and r["source"] == src_name]
                if not sel:
                    continue
                fps = np.array([r["fps_wall"] for r in sel])
                summary.append({
                    "model": model_id, "precision": prec, "source": src_name,
                    "repeats": len(sel),
                    "fps_mean": round(float(fps.mean()), 2),
                    "fps_std": round(float(fps.std(ddof=1)), 2),
                    "pre_ms_mean": round(float(np.mean([r["pre_ms"] for r in sel])), 3),
                    "inf_ms_mean": round(float(np.mean([r["inf_ms"] for r in sel])), 3),
                    "nms_ms_mean": round(float(np.mean([r["nms_ms"] for r in sel])), 3),
                    "protocol": f"batch1,warmup{WARMUP},timed{TIMED},repeats{REPEATS},imgsz{IMGSZ}",
                })
    with open(AUDIT_DIR / f"{args.out_prefix}fps_benchmark_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    gpu_keys = []
    for g in gpu_rows:
        for k in g:
            if k not in gpu_keys:
                gpu_keys.append(k)
    with open(AUDIT_DIR / f"{args.out_prefix}gpu_state.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gpu_keys)
        w.writeheader()
        w.writerows(gpu_rows)
    print("\nsaved fps_benchmark.csv, fps_benchmark_summary.csv, gpu_state.csv")


if __name__ == "__main__":
    main()



