"""|∇T| field comparison at matched simulated times for R1-aniso,
R1-iso, R3-aniso. Writes two grids — one with mesh overlay,
one without.
"""
import os
import re
import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv


pv.OFF_SCREEN = True
BASE = os.path.expanduser('~/+Simulations/StagnantLid')
ROWS = ['R1.0_iso', 'adapt_loop_med_every5', 'adapt_loop_med_every5_aniso']
TARGET_T = [0.001, 0.002, 0.003, 0.004, 0.005]


def _rowdir(row):
    if row.startswith('adapt_loop'):
        return os.path.join(BASE, row)
    return os.path.join(BASE, 'aniso_dt_validate', row)


def collect_snapshots(row):
    rowdir = _rowdir(row)
    hpath = os.path.join(rowdir, 'history.npz')
    z = np.load(hpath)
    step_to_t = dict(zip(
        z['step'].astype(int).tolist(), z['t'].tolist()))
    out = []
    for f in sorted(os.listdir(rowdir)):
        m = re.match(r"step(\d+)\.mesh\.00000\.h5$", f)
        if m:
            s = int(m.group(1))
            t = step_to_t.get(s, None)
            if t is not None:
                out.append((s, t, f"step{s:04d}"))
    return out


def find_closest(snaps, target_t):
    if not snaps:
        return None
    return min(snaps, key=lambda x: abs(x[1] - target_t))


sources = {row: collect_snapshots(row) for row in ROWS}
for row, snaps in sources.items():
    print(f"  {row}: {len(snaps)} snapshots, "
          f"t range {snaps[0][1]:.4f}..{snaps[-1][1]:.4f}")

selected = []
for tt in TARGET_T:
    picks = {}
    keep = False
    for row in ROWS:
        c = find_closest(sources[row], tt)
        if c is not None and abs(c[1] - tt) < 0.5 * tt:
            picks[row] = c
            keep = True
    if keep:
        selected.append((tt, picks))

ncols = len(selected)


def gradT_mag_sym(mesh, T):
    X = mesh.CoordinateSystem.X
    return sympy.sqrt(T.sym[0].diff(X[0]) ** 2
                      + T.sym[0].diff(X[1]) ** 2)


# Pre-pass: shared |∇T| clim across the grid
print("  scanning |∇T|max...")
g_max = 0.0
for r, row in enumerate(ROWS):
    rowdir = _rowdir(row)
    for c, (tt, picks) in enumerate(selected):
        if row not in picks:
            continue
        s_idx, t_val, label = picks[row]
        m = uw.discretisation.Mesh(
            os.path.join(rowdir, f"{label}.mesh.00000.h5"))
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


def render(with_edges, out_name):
    pl = pv.Plotter(shape=(len(ROWS), ncols), off_screen=True,
                    window_size=(400 * ncols, 400 * len(ROWS)),
                    border=False)
    pl.set_background('white')
    for r, row in enumerate(ROWS):
        rowdir = _rowdir(row)
        for c, (tt, picks) in enumerate(selected):
            if row not in picks:
                continue
            s_idx, t_val, label = picks[row]
            m = uw.discretisation.Mesh(
                os.path.join(rowdir, f"{label}.mesh.00000.h5"))
            T = uw.discretisation.MeshVariable(
                f"T_render_{row}_{label}_{out_name}", m,
                vtype=uw.VarType.SCALAR, degree=3,
                continuous=True)
            T.read_timestep(label, "T_v2p1", 0,
                            outputPath=rowdir)
            pv_g = vis.meshVariable_to_pv_mesh_object(T)
            pv_g.point_data["gradT"] = vis.scalar_fn_to_pv_points(
                pv_g, gradT_mag_sym(m, T))
            pl.subplot(r, c)
            pl.add_text(f"{row}\n{label}  t={t_val:.4f}",
                        font_size=10, color='black')
            pl.add_mesh(pv_g, scalars="gradT", cmap="Greens",
                        clim=(0.0, g_max), show_edges=False,
                        lighting=False,
                        show_scalar_bar=(r == 0
                                         and c == ncols - 1),
                        scalar_bar_args=dict(title="|∇T|",
                                             color="black"))
            if with_edges:
                edges = (vis.mesh_to_pv_mesh(m)
                         .extract_all_edges())
                pl.add_mesh(edges, color="black",
                            line_width=0.5, lighting=False,
                            opacity=0.55)
            pl.view_xy()
            pl.camera.zoom(1.25)
    out = os.path.join(BASE, 'adapt_loop_med_every5', out_name)
    pl.screenshot(out)
    pl.close()
    print(f"wrote {out}")


render(with_edges=False, out_name='plot_gradT_compare_clean.png')
render(with_edges=True, out_name='plot_gradT_compare_mesh.png')
