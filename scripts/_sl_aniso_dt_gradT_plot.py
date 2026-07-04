"""|∇T| field + mesh edges for the aniso-dt validation snapshots.

Tests the user's hypothesis: in a stagnant lid problem the
maximum |∇T| should be in the cool sub-layer BELOW the lid
(where T transitions from ~0 to active-layer values), NOT at
the outer cold boundary itself. The mesh should refine there.

Renders a 4×5 grid (rows = runs, cols = step snapshots) with
|∇T| in greens (white → dark green) and mesh edges overlaid.
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

ROWS = ['R1.0_aniso', 'R1.0_iso', 'R3.0_aniso', 'R3.0_iso']
STEPS = ['init', 'step0020', 'step0040', 'step0060', 'step0080']


def step_index_from_label(label):
    return 0 if label == 'init' else int(label.replace('step', ''))


# Pre-pass: load history.npz per row for t lookup
t_lookup = {}
for row in ROWS:
    h = os.path.join(BASE, row, 'history.npz')
    if not os.path.exists(h):
        t_lookup[row] = {}
        continue
    z = np.load(h)
    t_lookup[row] = dict(zip(
        z['step'].astype(int).tolist(), z['t'].tolist()))


# Pre-pass: shared |∇T| clim across the whole grid
def gradT_mag_sym(mesh, T):
    X = mesh.CoordinateSystem.X
    return sympy.sqrt(T.sym[0].diff(X[0]) ** 2
                      + T.sym[0].diff(X[1]) ** 2)


print("  scanning |∇T|max across all snapshots...")
g_max = 0.0
for r, row in enumerate(ROWS):
    rowdir = os.path.join(BASE, row)
    for label in STEPS:
        mp = os.path.join(rowdir, f"{label}.mesh.00000.h5")
        if not os.path.exists(mp):
            continue
        m = uw.discretisation.Mesh(mp)
        T = uw.discretisation.MeshVariable(
            f"T_scan_{row}_{label}", m,
            vtype=uw.VarType.SCALAR, degree=3, continuous=True)
        T.read_timestep(label, "T_v2p1", 0, outputPath=rowdir)
        pv_T = vis.meshVariable_to_pv_mesh_object(T)
        g = vis.scalar_fn_to_pv_points(pv_T, gradT_mag_sym(m, T))
        m_max = float(np.nanmax(g))
        if m_max > g_max:
            g_max = m_max
print(f"  global |∇T|max = {g_max:.3e}")


pl = pv.Plotter(shape=(len(ROWS), len(STEPS)), off_screen=True,
                window_size=(380 * len(STEPS), 380 * len(ROWS)),
                border=False)
pl.set_background('white')

for r, row in enumerate(ROWS):
    rowdir = os.path.join(BASE, row)
    for c, label in enumerate(STEPS):
        mp = os.path.join(rowdir, f"{label}.mesh.00000.h5")
        if not os.path.exists(mp):
            continue
        m = uw.discretisation.Mesh(mp)
        T = uw.discretisation.MeshVariable(
            f"T_grad_{row}_{label}", m,
            vtype=uw.VarType.SCALAR, degree=3, continuous=True)
        T.read_timestep(label, "T_v2p1", 0, outputPath=rowdir)
        pv_g = vis.meshVariable_to_pv_mesh_object(T)
        pv_g.point_data["gradT"] = vis.scalar_fn_to_pv_points(
            pv_g, gradT_mag_sym(m, T))
        edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
        s_idx = step_index_from_label(label)
        t_val = t_lookup.get(row, {}).get(s_idx, None)
        title = (f"{row}\n{label}"
                 + (f"  t={t_val:.4f}" if t_val is not None
                    else ""))
        pl.subplot(r, c)
        pl.add_text(title, font_size=9, color='black')
        pl.add_mesh(pv_g, scalars="gradT", cmap="Greens",
                    clim=(0.0, g_max), show_edges=False,
                    lighting=False,
                    show_scalar_bar=(r == 0
                                     and c == len(STEPS) - 1),
                    scalar_bar_args=dict(title=r"|∇T|",
                                         color="black"))
        pl.add_mesh(edges, color="black", line_width=0.6,
                    lighting=False, opacity=0.6)
        pl.view_xy()
        pl.camera.zoom(1.25)

out = os.path.join(BASE, 'plot_gradT_mesh.png')
pl.screenshot(out)
pl.close()
print(f"wrote {out}")
