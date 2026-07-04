"""Smoke + cross-validation for the new mesh.quality() API.
Loads a known checkpoint and checks the headline numbers match
the scratch _cellquality.py results for the same mesh, exercises
per_cell=True, and renders mesh.view() so the summary line shows.

Reference (scripts/_cellquality.py, sat_a16e step 20):
  n=1522  q_min≈0.237  n(q<0.2)=0  aspect_max≈8.5  ang_max≈147.5
"""
import numpy as np
import underworld3 as uw

D = "/tmp/metric_mesh/sat"
m = uw.discretisation.Mesh(f"{D}/sat_a16e.mesh.00020.h5")

Q = m.quality()
print("mesh.quality() →")
for k, v in Q.items():
    if k == "per_cell":
        continue
    print(f"  {k:20s} = {v:.4f}" if isinstance(v, float)
          else f"  {k:20s} = {v}")

ref = dict(n_cells=1522, q_min=0.237, n_q_lt_0p2=0)
ok = (Q["n_cells"] == ref["n_cells"]
      and abs(Q["q_min"] - ref["q_min"]) < 0.01
      and Q["n_q_lt_0p2"] == ref["n_q_lt_0p2"])
print(f"\ncross-check vs scratch _cellquality (a16e s20): "
      f"{'PASS' if ok else 'FAIL'}  "
      f"(n={Q['n_cells']} q_min={Q['q_min']:.3f} "
      f"n_q<0.2={Q['n_q_lt_0p2']})")

pc = m.quality(per_cell=True)["per_cell"]
print(f"per_cell arrays: q{pc['q'].shape} angle{pc['angle_deg'].shape}"
      f" aspect{pc['aspect'].shape} volume{pc['volume'].shape} "
      f"(q.min={pc['q'].min():.3f} == headline {Q['q_min']:.3f}: "
      f"{np.isclose(pc['q'].min(), Q['q_min'])})")

print("\n--- mesh.view() (expect a 'Cell quality:' line) ---")
m.view()
