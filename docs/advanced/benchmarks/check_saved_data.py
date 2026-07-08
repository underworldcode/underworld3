"""Verify that the on-disk benchmark npz files contain everything
needed for any plot we'd want — without re-running the simulations.

Lists keys in each .npz and asserts the per-config files have
``sigma_bdf1``, ``sigma_bdf2``, and ``sigma_ana`` (so the BDF-1 vs
BDF-2 overlay is reproducible from saved data alone), and that the
convergence file has ``trace_*`` arrays for every (order, dt) pair
recorded in the metrics arrays (so any per-run trace from the
convergence sweep is reproducible too).

Run::

    pixi run -e amr-dev python docs/advanced/benchmarks/check_saved_data.py
"""

import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bench_helpers import OUTPUT_DIR, load_run


def _list_npz_keys(name):
    path = f"{OUTPUT_DIR}/{name}.npz"
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as f:
        return list(f.keys())


def _check_per_case(name):
    arrays, params, extra = load_run(name)
    needed_arrays = {"times", "dts", "gamma_dot", "sigma_ana",
                     "sigma_bdf1", "sigma_bdf2"}
    have = set(arrays.keys())
    missing = needed_arrays - have
    print(f"\n[{name}.npz]")
    print(f"  arrays: {sorted(have)}")
    print(f"  params keys: {sorted(params.keys())}")
    print(f"  extra keys:  {sorted(extra.keys())}")
    if missing:
        print(f"  MISSING: {sorted(missing)}")
        return False
    print("  OK — has both BDF traces + analytical reference")
    return True


def _check_convergence(name):
    arrays, params, extra = load_run(name)
    n_runs = len(arrays["order"])
    print(f"\n[{name}.npz]")
    print(f"  metrics arrays: order, dt, n_steps, max_abs, rms, wall  ({n_runs} runs)")
    expected_traces = []
    for order, dt in zip(arrays["order"], arrays["dt"]):
        tag = f"o{int(order)}_dt{float(dt):.4f}"
        expected_traces += [f"trace_t_{tag}", f"trace_sigma_{tag}", f"trace_ana_{tag}"]
    have = set(arrays.keys())
    missing = [t for t in expected_traces if t not in have]
    print(f"  expected {len(expected_traces)} trace arrays; have {len(expected_traces) - len(missing)}")
    if missing:
        print(f"  MISSING: {missing[:6]}{' …' if len(missing) > 6 else ''}")
        return False
    print("  OK — every (order, dt) trace is on disk")
    return True


def main():
    ok = True
    for name in ("ve_harmonic", "ve_square", "vep_square"):
        if _list_npz_keys(name) is None:
            print(f"\n[{name}.npz]  not on disk — skipping")
            continue
        ok = _check_per_case(name) and ok
    for name in ("convergence_ve_harmonic", "convergence_ve_square",
                 "convergence_vep_square"):
        if _list_npz_keys(name) is None:
            print(f"\n[{name}.npz]  not on disk — skipping")
            continue
        ok = _check_convergence(name) and ok
    print("\n=== overall:", "OK" if ok else "FAIL", "===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
