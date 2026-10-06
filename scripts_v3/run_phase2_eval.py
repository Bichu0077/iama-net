"""Phase 2.4 orchestrator: run the full evaluation battery on the matched
checkpoints (runs_v3/*_matched_seed0) once training has completed.

Steps (each logged to results_v3/logs/phase2_eval_<step>.log):
  1. eval_unified   (val + per-image caches; NEU test, GC10 test/all/per-class, pp variants)
  2. bootstrap_ci   (CIs; 1000 resamples)
  3. paired tests   M2m vs M1m, M3m vs M1m, M4m vs M1m (paired-only runs)
  4. restricted overlap on GC10
  5. FPS benchmark  (one session, fixed order)
  6. aggregate      -> results_v3/ablation_matched.csv + cross_dataset_matched.csv

Usage:  python scripts_v3/run_phase2_eval.py [--skip-to STEP]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from iama_env import PROJECT_ROOT

PY = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
M = PROJECT_ROOT / "results_v3" / "matched"
LOGS = PROJECT_ROOT / "results_v3" / "logs"


def run(step, args):
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"phase2_eval_{step}.log"
    print(f"\n===== STEP {step}: {' '.join(args)} =====", flush=True)
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = "scripts_v3"
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.run([PY] + args, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=str(PROJECT_ROOT), env=env)
    ok = proc.returncode == 0
    print(f"step {step}: rc={proc.returncode} in {(time.time()-t0)/60:.1f} min -> {log.name}", flush=True)
    if not ok:
        tail = log.read_text(encoding="utf-8", errors="replace")[-2000:]
        print(tail, flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-to", type=int, default=1)
    args = ap.parse_args()
    s = args.skip_to

    registry = str(M / "registry_matched.json")
    ok = True
    if s <= 1:
        ok &= run("eval", ["scripts_v3/eval_unified.py", "--registry", registry,
                           "--combos", str(M / "combos_eval.json"), "--out-prefix", "matched_"])
    if s <= 2:
        ok &= run("bootstrap", ["scripts_v3/bootstrap_ci.py",
                                "--combos", str(M / "combos_bootstrap.json"),
                                "--paired", str(M / "paired_bootstrap.json"),
                                "--models-a", "M1m", "--models-b", "M4m",
                                "--out-prefix", "matched_"])
    if s <= 3:
        ok &= run("paired_m2m", ["scripts_v3/bootstrap_ci.py",
                                 "--combos", str(M / "combos_empty.json"),
                                 "--paired", str(M / "paired_m2m.json"),
                                 "--models-a", "M1m", "--models-b", "M2m",
                                 "--out-prefix", "matched_pM2_"])
        ok &= run("paired_m3m", ["scripts_v3/bootstrap_ci.py",
                                 "--combos", str(M / "combos_empty.json"),
                                 "--paired", str(M / "paired_m3m.json"),
                                 "--models-a", "M1m", "--models-b", "M3m",
                                 "--out-prefix", "matched_pM3_"])
    if s <= 4:
        ok &= run("restricted", ["scripts_v3/gc10_restricted_overlap.py",
                                 "--combos", str(M / "combos_restricted.json"),
                                 "--out", "gc10_restricted_matched.csv"])
    if s <= 5:
        ok &= run("fps", ["scripts_v3/fps_bench.py", "--registry", registry,
                          "--order", "M1m,M2m,M3m,M4m", "--out-prefix", "matched_"])
    if s <= 6:
        ok &= run("aggregate", ["scripts_v3/aggregate_matched.py"])
    print("\nPHASE 2 EVAL", "COMPLETE" if ok else "FINISHED WITH ERRORS (see logs)")


if __name__ == "__main__":
    main()
