"""Run the mover with verbose=True and capture the per-iter
max|Δx| trajectory. If displacement stays LARGE across iters
(doesn't decrease), the Winslow solve is not converging — D is
'remembering' the previous mesh state because it's stored on
a MeshVariable (Lagrangian).

If displacement DECREASES geometrically, the solver IS
converging and the iteration is doing what it should."""
import os
import sys
import io
import contextlib
import re
import numpy as np
import underworld3 as uw

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes import build_mesh_with_field


for nit in [12]:
    for relax in [1.0, 0.5, 0.2]:
        m, T = build_mesh_with_field()
        rho = uw.meshing.metric_density_from_gradient(
            m, T, refinement=3.0, name=f"disp_{nit}_{relax}")
        # Capture verbose output
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            uw.meshing.smooth_mesh_interior(
                m, metric=rho, method="anisotropic",
                strategy="med",
                method_kwargs=dict(n_outer=nit, relax=relax),
                verbose=True)
        out = buf.getvalue()
        # Extract per-iter max|Δx|
        rows = []
        for line in out.splitlines():
            mch = re.search(
                r"outer\s+(\d+)/\d+:.*max\|Δx\|=([0-9.e+-]+)",
                line)
            if mch:
                rows.append((int(mch.group(1)),
                             float(mch.group(2))))
        print(f"\n=== n_outer={nit}, relax={relax} — "
              f"per-iter max|Δx|:")
        for it, d in rows:
            print(f"  iter {it:2d}:  {d:.4e}")
