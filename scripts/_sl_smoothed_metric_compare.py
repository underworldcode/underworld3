"""Compare three approaches to suppressing the metric/mesh
feedback in metric_density_from_gradient:

  1. raw            — gradient on the adapted mesh directly
                      (the susceptible case)
  2. smoothed_L1h0  — gradient on screened-Poisson-smoothed T
                      with L = 1×h0 (background mean cell size)
  3. smoothed_L2h0  — same, L = 2×h0 (more aggressive)

Each adapts the step-80 R1/uniform T field with strategy='med'
and FE-remaps T onto the resulting graded mesh, so the rendered
T fields are physically the same and we're comparing **only**
the mesh response to different gradient-input choices.
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
SRC_STEM = "step0080"
OUT_BASE = os.path.expanduser(
    '~/+Simulations/StagnantLid/smoothed_metric_compare')
os.makedirs(OUT_BASE, exist_ok=True)


def load_uniform():
    m = uw.discretisation.Mesh(
        os.path.join(SRC_DIR, f"{SRC_STEM}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(SRC_STEM, "T_v2p1", 0, outputPath=SRC_DIR)
    return m, T


# Background mean cell size from a clean uniform load
m_ref, _T_ref = load_uniform()
H0 = float(m_ref._radii.mean())
print(f"background mean cell size h0 = {H0:.4f}")

CASES = [
    dict(label="raw",            kind=None,         L=None),
    dict(label="smoothT_L1h0",   kind="field",      L=1.0 * H0),
    dict(label="smoothT_L2h0",   kind="field",      L=2.0 * H0),
    dict(label="smoothGrad_L1h0",kind="gradient",   L=1.0 * H0),
    dict(label="smoothGrad_L2h0",kind="gradient",   L=2.0 * H0),
]


for c in CASES:
    out = os.path.join(OUT_BASE, c['label'])
    os.makedirs(out, exist_ok=True)
    if os.path.exists(
            os.path.join(out, "adapted.mesh.00000.h5")):
        print(f"{c['label']}: already adapted")
        continue
    print(f"{c['label']}: adapting "
          f"(kind={c['kind']}, L={c['L']})")
    m, T = load_uniform()
    old_X = np.asarray(m.X.coords).copy()
    old_T = np.asarray(T.data).copy()
    kw = dict(strategy="med", name=f"smetric_{c['label']}")
    if c['kind'] == "field":
        kw['smoothing_length'] = c['L']
    elif c['kind'] == "gradient":
        kw['gradient_smoothing_length'] = c['L']
    rho = uw.meshing.metric_density_from_gradient(
        m, T, **kw)
    uw.meshing.smooth_mesh_interior(
        m, metric=rho, method="anisotropic",
        strategy="med",
        method_kwargs=dict(relax=0.2, n_outer=12))
    new_X = np.asarray(m.X.coords).copy()
    new_Tx = np.asarray(T.coords).copy()
    m._deform_mesh(old_X)
    T.data[...] = old_T
    rT = np.asarray(uw.function.evaluate(
        T.sym[0], new_Tx)).reshape(-1)
    m._deform_mesh(new_X)
    T.data[:, 0] = rT
    m.write_timestep(filename="adapted", index=0,
                     outputPath=out, meshVars=[T],
                     meshUpdates=True, create_xdmf=True)
    print("  done")


# Render
def gradT_mag_sym(mesh, T):
    X = mesh.CoordinateSystem.X
    return sympy.sqrt(T.sym[0].diff(X[0]) ** 2
                      + T.sym[0].diff(X[1]) ** 2)


g_max = 0.0
loaded = []
for c in CASES:
    out = os.path.join(OUT_BASE, c['label'])
    m = uw.discretisation.Mesh(
        os.path.join(out, "adapted.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        f"T_view_{c['label']}", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep("adapted", "T_v2p1", 0, outputPath=out)
    loaded.append((c, m, T))
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    g = vis.scalar_fn_to_pv_points(pv_T, gradT_mag_sym(m, T))
    g_max = max(g_max, float(np.nanmax(g)))
print(f"global |∇T|max = {g_max:.3e}")

ncols = len(CASES)
pl = pv.Plotter(shape=(2, ncols), off_screen=True,
                window_size=(450 * ncols, 900), border=False)
pl.set_background("white")

for c_idx, (c, m, T) in enumerate(loaded):
    if c['L'] is None:
        title = f"{c['label']}\n(no smoothing)"
    else:
        title = (f"{c['label']}\n"
                 f"kind={c['kind']}  L={c['L']:.4f}")
    # Row 0: T field
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    pl.subplot(0, c_idx)
    pl.add_text(f"{title}\nT field", font_size=11, color='black')
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False,
                show_scalar_bar=(c_idx == ncols - 1),
                scalar_bar_args=dict(title="T", color="black"))
    pl.view_xy(); pl.camera.zoom(1.25)
    # Row 1: |∇T| + mesh
    pv_g = vis.meshVariable_to_pv_mesh_object(T)
    pv_g.point_data["gradT"] = vis.scalar_fn_to_pv_points(
        pv_g, gradT_mag_sym(m, T))
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(1, c_idx)
    pl.add_text(f"|∇T| + mesh  ({c['label']})",
                font_size=11, color='black')
    pl.add_mesh(pv_g, scalars="gradT", cmap="Greens",
                clim=(0.0, g_max), show_edges=False,
                lighting=False,
                show_scalar_bar=(c_idx == ncols - 1),
                scalar_bar_args=dict(title=r"|∇T|",
                                     color="black"))
    pl.add_mesh(edges, color="black", line_width=0.7,
                lighting=False, opacity=0.6)
    pl.view_xy(); pl.camera.zoom(1.25)

out_png = os.path.join(OUT_BASE, "plot_smoothed_compare.png")
pl.screenshot(out_png)
pl.close()
print(f"wrote {out_png}")
