"""Phase 2 job queue runner (AGENTS.md hard rule 8).

Runs a list of training jobs sequentially as subprocesses, logging each to
results_v3/logs/<run>.log and tracking completion in
results_v3/logs/queue_state.json. Survives interruptions: rerunning this
script skips completed jobs and resumes interrupted ones (ultralytics
resume=True from last.pt) automatically.

Job list JSON format:
  [{"variant": "M1", "seed": 0, "epochs": 50}, ...]

Usage:
  python scripts_v3/run_queue.py --jobs results_v3/queue_phase2.json
  python scripts_v3/run_queue.py --jobs results_v3/queue_phase2.json --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from iama_env import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "results_v3" / "logs"
STATE_PATH = LOG_DIR / "queue_state.json"
PY = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"jobs": {}}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def run_dir_for(job):
    return PROJECT_ROOT / "runs_v3" / f"{job['variant']}_matched_seed{job['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = json.loads(Path(args.jobs).read_text())
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()

    for job in jobs:
        key = f"{job['variant']}_seed{job['seed']}"
        entry = state["jobs"].get(key, {})
        run_dir = run_dir_for(job)
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        manifest = run_dir / "manifest.json"

        if entry.get("status") == "completed" and best.exists():
            print(f"[skip] {key} already completed")
            continue

        cmd = [PY, str(PROJECT_ROOT / "scripts_v3" / "train_matched.py"),
               "--variant", job["variant"], "--seed", str(job["seed"]),
               "--epochs", str(job["epochs"])]
        if entry.get("status") == "running" and last.exists():
            cmd.append("--resume")
            print(f"[resume] {key} from last.pt")
        else:
            print(f"[start] {key} epochs={job['epochs']}")

        if args.dry_run:
            print("   DRY:", " ".join(cmd))
            continue

        state["jobs"][key] = {"status": "running", "started": datetime.now().isoformat(timespec="seconds"),
                              "command": " ".join(cmd)}
        save_state(state)

        log_path = LOG_DIR / f"{key}.log"
        t0 = time.time()
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n===== {datetime.now().isoformat()} CMD: {' '.join(cmd)} =====\n")
            lf.flush()
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
        wall = time.time() - t0

        ok = proc.returncode == 0 and best.exists()
        state["jobs"][key].update({
            "status": "completed" if ok else "failed",
            "returncode": proc.returncode,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "wall_clock_min": round(wall / 60, 1),
            "log": str(log_path.relative_to(PROJECT_ROOT)),
        })
        save_state(state)
        print(f"[{'done' if ok else 'FAIL'}] {key} in {wall/60:.1f} min -> {log_path.name}")
        if not ok:
            print(f"  job failed (rc={proc.returncode}); continuing with next job. "
                  f"Inspect {log_path}")

    print("\nQueue finished. State:", json.dumps(state["jobs"], indent=1)[:2000])


if __name__ == "__main__":
    main()
