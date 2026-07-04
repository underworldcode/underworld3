"""Compare adapted-mesh layouts for different `power` values of
metric_density_from_gradient on the step-80 R1.0_aniso T field
(uniform mesh at t=0.0075).

power=1 = front-following (the historical default; refine the
          sharp boundaries)
power=2 = gradient-uniform (every cell carries the same ΔT)
power=0.5 = even softer than power=1

Adapts the same uniform res-16 mesh with each metric and
renders:
  - the resulting node distribution
  - the |∇T| field overlaid with mesh edges
"""
import os
import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv


pv.OFF_SCREEN = True
BASE = os.path.expanduser(
    '~/+Simulations/StagnantLid/aniso_dt_validate')
SRC_DIR = os.path.join(BASE, 'R1.0_aniso')
SRC_STEM = "step0080"   # uniform mesh, T at t=0.0075
OUT_BASE = os.path.expanduser(
    '~/+Simulations/StagnantLid/power_metric_compare')
os.makedirs(OUT_BASE, exist_ok=True)

POWERS = [0.5, 1.0, 2.0]
RESOLUTION_RATIO = 1.5    # same R as the R1.5 production point


def load_uniform_step80():
    m = uw.discretisation.Mesh(
        os.path.join(SRC_DIR, f"{SRC_STEM}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(SRC_STEM, "T_v2p1", 0, outputPath=SRC_DIR)
    return m, T


# Adapt for each power, FE-remap T onto the adapted nodes so
# all panels show the SAME physical T field (just sampled on
# differently-placed nodes — what we actually want to compare).
for p in POWERS:
    out = os.path.join(OUT_BASE, f"power{p}")
    os.makedirs(out, exist_ok=True)
    print(f"power={p}: adapting from uniform-step80...")
    m, T = load_uniform_step80()
    # Capture uniform geometry + T values BEFORE the mover moves
    # the nodes
    old_X = np.asarray(m.X.coords).copy()
    old_T = np.asarray(T.data).copy()
    rho = uw.meshing.metric_density_from_gradient(
        m, T, amp=8.0, lo_percentile=50.0, hi_percentile=97.0,
        power=p, name=f"pwr{p}")
    uw.meshing.smooth_mesh_interior(
        m, metric=rho, method="anisotropic",
        method_kwargs=dict(resolution_ratio=RESOLUTION_RATIO,
                           relax=0.2, n_outer=12))
    new_X = np.asarray(m.X.coords).copy()
    new_Tx = np.asarray(T.coords).copy()
    # FE-remap: restore the OLD geometry, FE-eval the OLD T at
    # the NEW dof coords, then put the mesh back to NEW geometry
    # and write the resampled T.
    m._deform_mesh(old_X)
    T.data[...] = old_T
    rT = np.asarray(uw.function.evaluate(
        T.sym[0], new_Tx)).reshape(-1)
    m._deform_mesh(new_X)
    T.data[:, 0] = rT
    m.write_timestep(filename="adapted", index=0, outputPath=out,
                     meshVars=[T], meshUpdates=True,
                     create_xdmf=True)
    print("  done")


# Render: 2 rows × len(POWERS) cols
# Row 0: |∇T| field (no edges) for each power (same on all because
#        starting field is the same — but the mesh DEFORMATION
#        moves nodes so the visualisation samples differ).
# Row 1: |∇T| field + mesh edges


def gradT_mag_sym(mesh, T):
    X = mesh.CoordinateSystem.X
    return sympy.sqrt(T.sym[0].diff(X[0]) ** 2
                      + T.sym[0].diff(X[1]) ** 2)


# Pre-pass: shared |∇T| clim
g_max = 0.0
loaded = []
for p in POWERS:
    out = os.path.join(OUT_BASE, f"power{p}")
    m = uw.discretisation.Mesh(
        os.path.join(out, "adapted.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        f"T_view_p{p}", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep("adapted", "T_v2p1", 0, outputPath=out)
    loaded.append((p, m, T, out))
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    g = vis.scalar_fn_to_pv_points(pv_T, gradT_mag_sym(m, T))
    g_max = max(g_max, float(np.nanmax(g)))
print(f"global |∇T|max = {g_max:.3e}")

ncols = len(POWERS)
pl = pv.Plotter(shape=(2, ncols), off_screen=True,
                window_size=(500 * ncols, 1000),
                border=False)
pl.set_background("white")

for c, (p, m, T, out) in enumerate(loaded):
    # Row 0: T field (no mesh) so we can see the physical state
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    pl.subplot(0, c)
    title = ("front-following" if p == 1.0
             else "grad-uniform (2D)" if p == 2.0
             else f"softer (power={p})")
    pl.add_text(f"power={p}  ({title})\nT field",
                font_size=11, color='black')
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False,
                show_scalar_bar=(c == ncols - 1),
                scalar_bar_args=dict(title="T", color="black"))
    pl.view_xy(); pl.camera.zoom(1.25)

    # Row 1: |∇T| field + mesh edges
    pv_g = vis.meshVariable_to_pv_mesh_object(T)
    pv_g.point_data["gradT"] = vis.scalar_fn_to_pv_points(
        pv_g, gradT_mag_sym(m, T))
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(1, c)
    pl.add_text(f"|∇T| + mesh  (power={p})",
                font_size=11, color='black')
    pl.add_mesh(pv_g, scalars="gradT", cmap="Greens",
                clim=(0.0, g_max), show_edges=False,
                lighting=False,
                show_scalar_bar=(c == ncols - 1),
                scalar_bar_args=dict(title=r"|∇T|",
                                     color="black"))
    pl.add_mesh(edges, color="black", line_width=0.7,
                lighting=False, opacity=0.6)
    pl.view_xy(); pl.camera.zoom(1.25)

out = os.path.join(OUT_BASE, "plot_power_compare.png")
pl.screenshot(out)
pl.close()
print(f"wrote {out}")
