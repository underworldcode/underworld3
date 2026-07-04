"""T-field comparison at matched simulated times for R1-aniso
vs R3-aniso.

Picks target t values across the overlap region, finds the
closest snapshot in each run, renders a 2-row grid:
  Row 1: R1.0_aniso (uniform mesh)
  Row 2: R3.0_aniso (adapted R=3 mesh)
"""
import os
import re
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv


pv.OFF_SCREEN = True
BASE = os.path.expanduser(
    '~/+Simulations/StagnantLid')


def _rowdir(row):
    """adapt_loop_* lives one level up from aniso_dt_validate."""
    if row.startswith('adapt_loop'):
        return os.path.join(BASE, row)
    return os.path.join(BASE, 'aniso_dt_validate', row)
ROWS = ['R1.0_iso', 'adapt_loop_med_every5', 'adapt_loop_med_every5_aniso']
TARGET_T = [0.001, 0.002, 0.003, 0.004, 0.005]


def collect_snapshots(row):
    """Find available step snapshots + their t values."""
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


# Gather snapshots
sources = {row: collect_snapshots(row) for row in ROWS}
for row, snaps in sources.items():
    print(f"  {row}: {len(snaps)} snapshots, "
          f"t range {snaps[0][1]:.4f}..{snaps[-1][1]:.4f}")

# Decide cols: for each TARGET_T, find closest in each row;
# drop a target if neither row has data within ±10% of TARGET_T
selected = []   # list of (target_t, {row: (s, t, label)})
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
        print(f"  target t={tt:.4f}: "
              + ", ".join(f"{row}→step{picks[row][0]} "
                          f"(t={picks[row][1]:.4f})"
                          for row in ROWS if row in picks))

ncols = len(selected)

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
            f"T_{row}_{label}", m, vtype=uw.VarType.SCALAR,
            degree=3, continuous=True)
        T.read_timestep(label, "T_v2p1", 0, outputPath=rowdir)
        pv_T = vis.meshVariable_to_pv_mesh_object(T)
        pv_T.point_data["T"] = np.asarray(T.data[:, 0])
        pl.subplot(r, c)
        pl.add_text(f"{row}\n{label}  t={t_val:.4f}",
                    font_size=10, color='black')
        pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                    clim=(0.0, 1.0), show_edges=False,
                    lighting=False,
                    show_scalar_bar=(r == 0 and c == ncols - 1),
                    scalar_bar_args=dict(title="T",
                                         color="black"))
        pl.view_xy()
        pl.camera.zoom(1.25)

out = os.path.join(
    BASE, 'adapt_loop_med_every5', 'plot_T_compare_to_iso.png')
pl.screenshot(out)
pl.close()
print(f"wrote {out}")
