import json, glob, os
import numpy as np
D = os.path.expanduser("~/+Simulations/fault_mg_exploration")   # where stage6_moving.py writes
runs = {}
for f in sorted(glob.glob(f"{D}/moving_*_k*.json")):
    d = json.load(open(f)); runs[(d["strategy"], d["k"])] = d
phases = ["addon_adapt", "split_finest", "split_levels", "level_frames", "background_TI", "solver_build", "jit_build", "first_solve"]
for k in sorted({k for _, k in runs}):
    print(f"\n=== k = {k} (mean seconds per step over {len(next(iter(runs.values()))['records'])} steps) ===")
    hdr = f"{'':6s}" + "".join(f"{p[:12]:>13s}" for p in phases) + f"{'transfers':>11s}{'TOTAL':>9s}{'vel its':>11s}{'levels':>8s}"
    print(hdr)
    for s in ("A", "C", "D", "gamg"):
        if (s, k) not in runs: continue
        R = runs[(s, k)]["records"]
        row = f"{s:6s}"
        tot = 0.0
        for p in phases:
            v = np.mean([r["times"].get(p, 0.0) for r in R]); tot += v
            row += f"{v:13.2f}"
        tr = np.mean([r.get("inside_solve", {}).get("transfers", 0.0) for r in R])
        its = [r["vel_its"][-1] for r in R]
        lev = [r["levels"] for r in R]
        conv = all(r["conv"] for r in R)
        row += f"{tr:11.2f}{tot:9.2f}{min(its):>6d}-{max(its):<4d}{min(lev):>4d}-{max(lev)}" + ("" if conv else "  NOT CONVERGED")
        print(row)
    # correctness: peak slip per step, all strategies
    print("peak slip per step:")
    for s in ("A", "C", "D", "gamg"):
        if (s, k) in runs:
            print(f"  {s:5s}", " ".join(f"{r['peak_slip']:.5f}" for r in runs[(s, k)]["records"]))
