"""Master pipeline: waits for the Phase-2 training queue, then chains
Phase 2 eval -> Phase 3 training -> Phase 3 eval. Resumable: step statuses
persist in results_v3/logs/pipeline_state.json; rerunning skips finished steps.

Runs as a persistent background process; all output logged under
results_v3/logs/pipeline_*.log. Writes results_v3/PIPELINE_DONE.json at the end.
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from iama_env import PROJECT_ROOT

PY = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
LOGS = PROJECT_ROOT / "results_v3" / "logs"
STATE = LOGS / "pipeline_state.json"
PHASE2_VARIANTS = ["M1", "M2", "M3", "M4"]
# Phase-3 variants are derived from the queue file so trims apply automatically
PHASE3_VARIANTS = [j["variant"] for j in json.loads(
    (PROJECT_ROOT / "results_v3" / "queue_phase3.json").read_text())]
POLL_S = 120
MAX_WAIT_H = 14


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOGS / "pipeline_master.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"steps": {}}


def set_step(st, name, status, **extra):
    st["steps"][name] = {"status": status, "at": datetime.now().isoformat(timespec="seconds"), **extra}
    STATE.write_text(json.dumps(st, indent=2))


def run_script(name, script_args, st):
    if st["steps"].get(name, {}).get("status") == "completed":
        log(f"skip completed step {name}")
        return True
    log(f"START {name}: {' '.join(script_args)}")
    set_step(st, name, "running")
    env = dict(os.environ)
    env["PYTHONPATH"] = "scripts_v3"
    t0 = time.time()
    with open(LOGS / f"pipeline_{name}.log", "w", encoding="utf-8") as lf:
        proc = subprocess.run([PY] + script_args, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=str(PROJECT_ROOT), env=env)
    mins = (time.time() - t0) / 60
    ok = proc.returncode == 0
    set_step(st, name, "completed" if ok else "failed", rc=proc.returncode, minutes=round(mins, 1))
    log(f"{'DONE' if ok else 'FAILED'} {name} rc={proc.returncode} in {mins:.1f} min")
    return ok


def queue_done(variants, state_path=None):
    p = Path(state_path or (LOGS / "queue_state.json"))
    if not p.exists():
        return False, []
    js = json.loads(p.read_text())
    jobs = js.get("jobs", {})
    done, failed = [], []
    for v in variants:
        e = jobs.get(f"{v}_seed0", {})
        if e.get("status") == "completed":
            done.append(v)
        elif e.get("status") == "failed":
            failed.append(v)
    return len(done) == len(variants), failed


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    st = state()

    # STEP 1: wait for the Phase-2 training queue (started separately)
    t0 = time.time()
    retried = False
    while True:
        done, failed = queue_done(PHASE2_VARIANTS)
        if done:
            log("Phase-2 training queue complete")
            break
        if failed and not retried:
            log(f"Phase-2 queue has failed jobs {failed}; relaunching queue once (resume)")
            retried = True
            env = dict(os.environ)
            env["PYTHONPATH"] = "scripts_v3"
            subprocess.run([PY, "scripts_v3/run_queue.py", "--jobs", "results_v3/queue_phase2.json"],
                           cwd=str(PROJECT_ROOT), env=env,
                           stdout=open(LOGS / "pipeline_queue_retry.log", "w", encoding="utf-8"),
                           stderr=subprocess.STDOUT)
            continue
        if (time.time() - t0) / 3600 > MAX_WAIT_H:
            log(f"ABORT: Phase-2 training not complete after {MAX_WAIT_H}h wait")
            set_step(st, "wait_phase2_training", "timeout")
            return
        time.sleep(POLL_S)
    set_step(st, "wait_phase2_training", "completed")

    # STEP 2: Phase-2 evaluation battery
    run_script("phase2_eval", ["scripts_v3/run_phase2_eval.py"], st)

    # STEP 3: Phase-3 training queue
    if st["steps"].get("phase3_training", {}).get("status") != "completed":
        set_step(st, "phase3_training", "running")
        done3 = False
        retried3 = False
        while not done3:
            done3, failed3 = queue_done(PHASE2_VARIANTS + PHASE3_VARIANTS)
            if done3:
                break
            if failed3 and not retried3:
                log(f"Phase-3 queue failed jobs {failed3}; relaunching queue once (resume)")
                retried3 = True
            env = dict(os.environ)
            env["PYTHONPATH"] = "scripts_v3"
            with open(LOGS / "pipeline_phase3_queue.log", "a", encoding="utf-8") as lf:
                subprocess.run([PY, "scripts_v3/run_queue.py", "--jobs", "results_v3/queue_phase3.json"],
                               cwd=str(PROJECT_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
            done3, failed3 = queue_done(PHASE2_VARIANTS + PHASE3_VARIANTS)
            if not done3 and failed3:
                log(f"Phase-3 jobs still failed after retry: {failed3} — continuing to eval what exists")
                break
        set_step(st, "phase3_training", "completed" if done3 else "partial")

    # STEP 4: Phase-3 evaluation battery
    run_script("phase3_eval", ["scripts_v3/run_phase3_eval.py"], st)

    done_path = PROJECT_ROOT / "results_v3" / "PIPELINE_DONE.json"
    done_path.write_text(json.dumps({"finished": datetime.now().isoformat(timespec="seconds"),
                                     "steps": st["steps"]}, indent=2))
    log("PIPELINE COMPLETE -> results_v3/PIPELINE_DONE.json")


if __name__ == "__main__":
    main()
