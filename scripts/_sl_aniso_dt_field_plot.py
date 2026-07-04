"""4×5 grid of T-field snapshots for the aniso-dt validation runs.

Rows: R1_aniso, R1_iso, R3_aniso, R3_iso
Cols: init, step20, step40, step60, step80

Writes TWO grids:
  - plot_T_fields.png       — T + mesh edges overlay
  - plot_T_fields_clean.png — T only (no edges) for trajectory
                              comparison without the mesh visually
                              biasing the eye

Annotates each cell with the simulated time t (read from
history.npz) so the comparison is on a sim-time axis as well
as step count.
"""
import os
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv


pv.OFF_SCREEN = True
BASE = os.path.expanduser(
    '~/+Simulations/StagnantLid/aniso_dt_validate')

ROWS = ['R1.0_aniso', 'R1.0_iso', 'R3.0_aniso', 'R3.0_iso']
STEPS = ['init', 'step0020', 'step0040', 'step0060', 'step0080']

# Pre-pass: load step→t map per row
t_lookup = {}
for row in ROWS:
    h = os.path.join(BASE, row, 'history.npz')
    if not os.path.exists(h):
        t_lookup[row] = {}
        continue
    z = np.load(h)
    t_lookup[row] = dict(zip(
        z['step'].astype(int).tolist(), z['t'].tolist()))


def step_index_from_label(label):
    return 0 if label == 'init' else int(label.replace('step', ''))


def render(with_edges, out_name):
    pl = pv.Plotter(shape=(len(ROWS), len(STEPS)), off_screen=True,
                    window_size=(380 * len(STEPS), 380 * len(ROWS)),
                    border=False)
    pl.set_background('white')
    for r, row in enumerate(ROWS):
        rowdir = os.path.join(BASE, row)
        for c, label in enumerate(STEPS):
            mesh_path = os.path.join(rowdir,
                                     f"{label}.mesh.00000.h5")
            if not os.path.exists(mesh_path):
                continue
            m = uw.discretisation.Mesh(mesh_path)
            T = uw.discretisation.MeshVariable(
                f"T_{row}_{label}_{out_name}", m,
                vtype=uw.VarType.SCALAR, degree=3,
                continuous=True)
            T.read_timestep(label, "T_v2p1", 0, outputPath=rowdir)
            pv_T = vis.meshVariable_to_pv_mesh_object(T)
            pv_T.point_data["T"] = np.asarray(T.data[:, 0])
            s_idx = step_index_from_label(label)
            t_val = t_lookup.get(row, {}).get(s_idx, None)
            title = (f"{row}\n{label}"
                     + (f"  t={t_val:.4f}" if t_val is not None
                        else ""))
            pl.subplot(r, c)
            pl.add_text(title, font_size=9, color='black')
            pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                        clim=(0.0, 1.0), show_edges=False,
                        lighting=False,
                        show_scalar_bar=(r == 0
                                         and c == len(STEPS) - 1),
                        scalar_bar_args=dict(title="T",
                                             color="black"))
            if with_edges:
                edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
                pl.add_mesh(edges, color="#202020", line_width=0.5,
                            lighting=False, opacity=0.5)
            pl.view_xy()
            pl.camera.zoom(1.25)
    out = os.path.join(BASE, out_name)
    pl.screenshot(out)
    pl.close()
    print(f"wrote {out}")


render(with_edges=True, out_name='plot_T_fields.png')
render(with_edges=False, out_name='plot_T_fields_clean.png')
