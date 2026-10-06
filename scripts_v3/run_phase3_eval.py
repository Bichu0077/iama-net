"""Phase 3 evaluation orchestrator (run AFTER Phase 3 training queue).

Steps (logs in results_v3/logs/phase3_eval_<step>.log):
  1. eval_unified on Phase-3 checkpoints + pp-at-test-time controls for M1m/M3m
  2. bootstrap CIs for all Phase-3 combos
  3. paired tests: each alt-attention vs M3m, M1v8 vs M1m, M2gray/M2nobil vs M2m
  4. restricted-overlap GC10 for alt-attention models
  5. FPS benchmark (Phase-3 models, one session)
  6. gradcam_quant (whole NEU test set; needs Phase-2 matched checkpoints)
  7. transfer_controls (multiscale + tiling + per-class; needs Phase-2 checkpoints)
  8. aggregate_phase3 -> results_v3/attention_alternatives.csv,
     preprocessing_controls.csv, external_reference.csv, transfer_controls.csv
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

from iama_env import PROJECT_ROOT

PY = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
M = PROJECT_ROOT / "results_v3" / "matched"
LOGS = PROJECT_ROOT / "results_v3" / "logs"


def run(step, args):
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"phase3_eval_{step}.log"
    print(f"\n===== P3 STEP {step}: {' '.join(args)} =====", flush=True)
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = "scripts_v3"
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.run([PY] + args, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=str(PROJECT_ROOT), env=env)
    ok = proc.returncode == 0
    print(f"step {step}: rc={proc.returncode} in {(time.time()-t0)/60:.1f} min -> {log.name}", flush=True)
    if not ok:
        print(log.read_text(encoding="utf-8", errors="replace")[-2000:], flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-to", type=int, default=1)
    args = ap.parse_args()
    s = args.skip_to
    registry = str(M / "registry_phase3.json")
    ok = True

    if s <= 1:
        ok &= run("eval", ["scripts_v3/eval_unified.py", "--registry", registry,
                           "--combos", str(M / "combos_phase3.json"), "--out-prefix", "p3_"])
    if s <= 2:
        ok &= run("bootstrap", ["scripts_v3/bootstrap_ci.py",
                                "--combos", str(M / "combos_bootstrap_phase3.json"),
                                "--paired", str(M / "paired_empty.json"),
                                "--out-prefix", "p3_"])
    if s <= 3:
        for tag, a, b in [("se", "M3m", "M3se"), ("cbam", "M3m", "M3cbam"),
                          ("gray", "M2m", "M2gray")]:
            ok &= run(f"paired_{tag}", ["scripts_v3/bootstrap_ci.py",
                                        "--combos", str(M / "combos_empty.json"),
                                        "--paired", str(M / f"paired_p3_{tag}.json"),
                                        "--models-a", a, "--models-b", b,
                                        "--out-prefix", f"p3_{tag}_"])
    if s <= 4:
        ok &= run("restricted", ["scripts_v3/gc10_restricted_overlap.py",
                                 "--combos", str(M / "combos_restricted_p3.json"),
                                 "--out", "gc10_restricted_p3.csv"])
    if s <= 5:
        ok &= run("fps", ["scripts_v3/fps_bench.py", "--registry", registry,
                          "--order", "M3se,M3cbam,M2gray",
                          "--out-prefix", "p3_"])
    if s <= 6:
        ok &= run("gradcam", ["scripts_v3/gradcam_quant.py"])
    if s <= 7:
        ok &= run("transfer", ["scripts_v3/transfer_controls.py", "--models", "M1m,M4m"])
    if s <= 8:
        ok &= run("aggregate", ["scripts_v3/aggregate_phase3.py"])
    print("\nPHASE 3 EVAL", "COMPLETE" if ok else "FINISHED WITH ERRORS (see logs)")


if __name__ == "__main__":
    main()
