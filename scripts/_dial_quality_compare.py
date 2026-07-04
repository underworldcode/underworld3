"""Single-dial default decision (mesh quality only — solver
robustness is now a separable, fixed layer). Compares the settled
mesh of each candidate via the new mesh.quality() API + settled Nu
from history. Existing checkpoints only (no re-runs; the mover
produces the same mesh for a given dial regardless of the Stokes
solver config).

  cc=2     (a16c2v)  legacy TWO-knob (aniso_cap=4, coarsen_cap=2)
  R=1.5    (a16r15d)  single-knob equidist resolution_ratio=1.5
  R=2      (a16e)     single-knob equidist resolution_ratio=2
"""
import glob
import re
import os
import numpy as np
import underworld3 as uw

D = "/tmp/metric_mesh/sat"
CAND = [("cc=2  (2-knob)", "a16c2v"),
        ("R=1.5 (1-knob)", "a16r15d"),
        ("R=2   (1-knob)", "a16e")]


def latest(tag):
    ix = [int(re.search(r"\.mesh\.T\.(\d+)\.h5$", f).group(1))
          for f in glob.glob(f"{D}/sat_{tag}.mesh.T.*.h5")]
    return max(ix) if ix else None


def settled_nu(tag):
    hp = f"{D}/sat_{tag}_hist.npz"
    if not os.path.exists(hp):
        return float("nan")
    z = np.load(hp)
    nu = z["Nu"]
    k = max(1, len(nu) // 7)
    return float(nu[-k:].mean())


hdr = ("dial            | ckpt | volMin/mn  q_min  q_mean  "
       "q<0.3  aspMax  jumpMax  BIG&THIN | settledNu")
print(hdr)
print("-" * len(hdr))
for label, tag in CAND:
    idx = latest(tag)
    if idx is None:
        print(f"{label:15s} |  (no checkpoints)")
        continue
    m = uw.discretisation.Mesh(f"{D}/sat_{tag}.mesh.{idx:05}.h5")
    Q = m.quality()
    print(f"{label:15s} | {idx:4d} | "
          f"{Q['vol_min_over_mean']:.3f}    "
          f"{Q['q_min']:.3f}  {Q['q_mean']:.3f}  "
          f"{Q['n_q_lt_0p3']:4d}  {Q['aspect_max']:5.1f}  "
          f"{Q['sizejump_max']:6.1f}  {Q['n_big_thin']:4d}     "
          f"| {settled_nu(tag):+.3f}")
print("\nde-resolution strength ∝ (lower volMin/mn, higher jumpMax); "
      "quality ∝ (higher q_min/q_mean, low q<0.3 & BIG&THIN). "
      "cc=2 settled-Nu uses OLD cached stencil if not recomputed — "
      "compare 1-knob R=1.5 vs R=2 on the proper-Nu runs.")
