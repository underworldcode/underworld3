"""Bounded smoother sweep for the TI weak-fault Stokes. Runs each
GAMG/FMG-smoother config as its OWN subprocess with a hard wall-clock
timeout (kills the whole process group), so slow configs report 'DNF >Ns'
instead of hanging the sweep. Fast configs report Newton + total KSP iters.

Documents how FMG (various smoothers) and GAMG behave on a localized
sharp fault contrast. Writes ~/+Simulations/StagnantLid/fault_smoother_sweep/table.txt
"""
import os, signal, subprocess, time

WT = "/Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/fault-convection"
OUT = os.path.expanduser("~/+Simulations/StagnantLid/fault_smoother_sweep")
os.makedirs(OUT, exist_ok=True)
TIMEOUT = 180   # seconds per config

BASE = ["pixi", "run", "-e", "amr-dev", "python", "-u", "scripts/fault_fmg_test.py",
        "--base-res", "4", "--refinement", "3", "--mode", "fault", "--contrast", "1000"]

CONFIGS = [
    ("gamg (reference)",   ["--gamg"]),
    ("fmg sor   x8",       ["--smooth", "8",  "--smoother-pc", "sor",     "--smoother-ksp", "richardson"]),
    ("fmg sor   x16",      ["--smooth", "16", "--smoother-pc", "sor",     "--smoother-ksp", "richardson"]),
    ("fmg sor   x32",      ["--smooth", "32", "--smoother-pc", "sor",     "--smoother-ksp", "richardson"]),
    ("fmg ilu   x4",       ["--smooth", "4",  "--smoother-pc", "ilu",     "--smoother-ksp", "richardson"]),
    ("fmg asm   x4",       ["--smooth", "4",  "--smoother-pc", "asm",     "--smoother-ksp", "richardson"]),
    ("fmg bjacobi x4",     ["--smooth", "4",  "--smoother-pc", "bjacobi", "--smoother-ksp", "richardson"]),
    ("fmg cheby/sor x8",   ["--smooth", "8",  "--smoother-pc", "sor",     "--smoother-ksp", "chebyshev"]),
]

table = os.path.join(OUT, "table.txt")
rows = [f"TI fault res32 / 4-level / contrast 1000 — per-config timeout {TIMEOUT}s",
        f"{'config':20s} | result"]


def emit(line):
    print(line, flush=True)
    rows.append(line)
    with open(table, "w") as fh:
        fh.write("\n".join(rows) + "\n")


emit(rows[1])
for label, extra in CONFIGS:
    cmd = BASE + extra
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=WT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=TIMEOUT)
        dt = time.time() - t0
        res_lines = [l for l in out.splitlines() if l.startswith("RESULT")]
        if res_lines:
            r = res_lines[-1].replace("RESULT ", "")
            emit(f"{label:20s} | {r}")
        elif "DIVERGED" in out:
            dl = [l for l in out.splitlines() if "DIVERGED" in l]
            emit(f"{label:20s} | DIVERGED ({dt:.0f}s): {dl[-1][:55]}")
        else:
            tail = out.strip().splitlines()[-1][:60] if out.strip() else "(no output)"
            emit(f"{label:20s} | no RESULT ({dt:.0f}s): {tail}")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
        p.communicate()
        emit(f"{label:20s} | DNF (> {TIMEOUT}s) — too slow / not converging")

emit("DONE")
