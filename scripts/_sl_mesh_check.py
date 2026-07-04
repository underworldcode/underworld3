"""Quick verification: load the step-125 snapshot, print mesh
stats, and render JUST the geometric mesh (no T fill, no
high-order DOF Delaunay) so we can see exactly what the loaded
mesh looks like."""
from __future__ import annotations
import os
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True

# uniform
src_u = os.path.expanduser(
    '~/+Simulations/StagnantLid/uniform_res16_Ra1e7_dEta1e4')
stem_u = "sl_uniform_res16_Ra1e7_dEta1e4_step00125"
mu = uw.discretisation.Mesh(
    os.path.join(src_u, f"{stem_u}.mesh.00000.h5"))

# adapted
src_a = os.path.expanduser(
    '~/+Simulations/StagnantLid/adapted_R15_Ra1e7_dEta1e4')
ma = uw.discretisation.Mesh(
    os.path.join(src_a, "adapted.mesh.00000.h5"))

for label, m in [("uniform", mu), ("adapted", ma)]:
    nv = m.X.coords.shape[0]
    cStart, cEnd = m.dm.getHeightStratum(0)
    pStart, pEnd = m.dm.getDepthStratum(0)
    print(f"{label}: vertices={pEnd - pStart}, cells={cEnd - cStart}, "
          f"X.coords.shape={m.X.coords.shape}")

# Render both: pure geometric mesh, no fills
pl = pv.Plotter(shape=(1, 2), off_screen=True,
                window_size=(1800, 900), border=False)
pl.set_background("white")

for k, (label, m) in enumerate([("uniform", mu),
                                 ("adapted", ma)]):
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(0, k)
    pl.add_text(label, font_size=14, color="black")
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False)
    pl.view_xy()
    pl.camera.zoom(1.3)

out = "/tmp/sl_mesh_check.png"
pl.screenshot(out)
print(f"wrote {out}")
