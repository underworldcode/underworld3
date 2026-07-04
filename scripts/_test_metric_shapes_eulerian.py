"""Lagrangian vs Eulerian metric on the synthetic shapes.

Hypothesis: the metric stored on a MeshVariable moves with the
mover's mesh deformation, creating a feedback loop. Storing
the metric on a SWARM (particles that stay put in physical
space) breaks this — the metric is Eulerian.

Three movers tested (anisotropic, spring, MA-like via spring
fallback), each with Lagrangian and Eulerian metric storage.
"""
import os
import sys
import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
from underworld3.meshing import smoothing as _sm
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_eulerian')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes import build_mesh_with_field


def lagrangian_metric(m, T, refinement=3.0):
    """Standard metric_density_from_gradient — stores ρ on a
    MeshVariable that moves with mesh deformation."""
    return uw.meshing.metric_density_from_gradient(
        m, T, refinement=refinement, name="lag")


# Module-level cache to keep swarms alive (otherwise they get
# garbage-collected when eulerian_metric() returns and the swarm
# variable can't query its parent).
_SWARM_STORE = []


def eulerian_metric(m, T, refinement=3.0):
    """Same metric values, but snapshot onto a swarm and serve
    via the swarm's proxy. Particles stay put in physical
    space, so the metric is Eulerian wrt the deforming mesh.
    Returns a sympy expression (the swarm proxy variable's sym)."""
    rho_mesh = uw.meshing.metric_density_from_gradient(
        m, T, refinement=refinement, name="eul_src")
    swarm = uw.swarm.Swarm(m)
    rho_var = uw.swarm.SwarmVariable(
        f"rho_E_{id(m):x}_{len(_SWARM_STORE)}", swarm,
        vtype=uw.VarType.SCALAR,
        proxy_degree=1, proxy_continuous=True)
    swarm.populate(fill_param=4)
    coords_p = np.asarray(swarm.data)
    rho_var.data[:, 0] = np.asarray(
        uw.function.evaluate(rho_mesh, coords_p)).reshape(-1)
    _SWARM_STORE.append((swarm, rho_var))  # keep alive
    return rho_var.sym[0]


# Cases: (label, metric_kind, mover_method, mover_kwargs)
# Now testing 2x2: {Lagrangian, Eulerian} × {frozen-M, refresh-per-iter}
CASES = [
    ("aniso + Lag (frozen M)", "L", "anisotropic",
     dict(strategy="med",
          method_kwargs=dict(relax=0.2, n_outer=12,
                              metric_refresh_per_iter=False))),
    ("aniso + Eul (frozen M)", "E", "anisotropic",
     dict(strategy="med",
          method_kwargs=dict(relax=0.2, n_outer=12,
                              metric_refresh_per_iter=False))),
    ("aniso + Lag (refresh M)", "L", "anisotropic",
     dict(strategy="med",
          method_kwargs=dict(relax=0.2, n_outer=12,
                              metric_refresh_per_iter=True))),
    ("aniso + Eul (refresh M)", "E", "anisotropic",
     dict(strategy="med",
          method_kwargs=dict(relax=0.2, n_outer=12,
                              metric_refresh_per_iter=True))),
]


def build_metric(m, T, kind):
    """Build metric, optionally with 'sharpened' bulk = 1.0."""
    if kind.startswith("L"):
        rho = lagrangian_metric(m, T, refinement=3.0)
    else:
        rho = eulerian_metric(m, T, refinement=3.0)
    if kind.endswith("sharp"):
        # Sharpen: clip ρ to {1.0, ρ_max} via Piecewise where
        # ρ < threshold (mostly-bulk) → 1.0 exactly.
        # We can't easily do this on the symbolic expression,
        # but we can sharpen by raising to a higher power so
        # the bulk (ρ ≈ 1) stays 1 and the feature regions
        # (ρ > 1) get amplified. ρ_sharp = ρ^2.
        rho = rho ** 2
    return rho


def adapt_one(label, kind, method, kw):
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace("+", "")
              .replace("(", "").replace(")", "")
              .replace(".", "p"))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(
            os.path.join(out_dir, "adapted.mesh.00000.h5")):
        print(f"{label}: cached")
        return out_dir
    print(f"{label}: adapting")
    m, T = build_mesh_with_field()
    rho = build_metric(m, T, kind)
    uw.meshing.smooth_mesh_interior(
        m, metric=rho, method=method, verbose=False, **kw)
    m.write_timestep(filename="adapted", index=0,
                     outputPath=out_dir, meshVars=[T],
                     meshUpdates=True, create_xdmf=True)
    return out_dir


for label, kind, method, kw in CASES:
    adapt_one(label, kind, method, kw)


# Render 2x2 grid
ncols, nrows = 2, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1500 * ncols, 1500 * nrows),
                border=False)
pl.set_background("white")
for i, (label, kind, method, kw) in enumerate(CASES):
    row, col = i // ncols, i % ncols
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace("+", "")
              .replace("(", "").replace(")", "")
              .replace(".", "p"))
    m = uw.discretisation.Mesh(
        os.path.join(out_dir, "adapted.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        f"T_view_{i}", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep("adapted", "T_shapes", 0, outputPath=out_dir)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(label, font_size=22, color='black')
    pl.add_mesh(pv_T, scalars="T", cmap="Blues",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_shapes_eulerian_sweep.png")
pl.screenshot(out_png)
pl.close()
print(f"wrote {out_png}")
