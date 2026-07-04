"""Pyvista comparison of T (P3) + V (P2) at step 15 for FE vs RBF runs.

T is plotted on its own P3 DOF cloud (Delaunay-triangulated) so the
high-order interpolant is faithfully rendered. The original deformed
mesh edges are overlaid on top so the cell quality is visible.
"""
import os
import numpy as np
import pyvista as pv
import underworld3 as uw
import underworld3.visualisation as vis

pv.OFF_SCREEN = True

RUNS = (
    ("output/convection_zoo_snapshots_sanity_baseline",
     "FE baseline"),
    ("output/convection_zoo_snapshots_rbf_advection_smoke",
     "RBF advection"),
)
STEP = 15
OUT_PNG = "output/T_V_compare_step15.png"


def load_run(snap_dir, step):
    root = f"uw_bdf2_sl_step{step:04d}"
    mesh_h5 = os.path.join(snap_dir, f"{root}.mesh.00000.h5")
    print(f"loading {mesh_h5}")
    mesh = uw.discretisation.Mesh(mesh_h5)
    T = uw.discretisation.MeshVariable(
        "T_conv_v2p1", mesh, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    V = uw.discretisation.MeshVariable(
        "V_conv_v2p1", mesh, vtype=uw.VarType.VECTOR,
        degree=2, continuous=True)
    T.read_timestep(root, "T_conv_v2p1", 0, outputPath=snap_dir)
    V.read_timestep(root, "V_conv_v2p1", 0, outputPath=snap_dir)
    return mesh, T, V


def stream_seeds(n=350, r_in=0.52, r_out=0.98):
    rng = np.random.default_rng(42)
    out = []
    while len(out) < n:
        cand = rng.uniform(-1.1, 1.1, size=(n * 3, 3))
        cand[:, 2] = 0.0
        r = np.linalg.norm(cand[:, :2], axis=1)
        good = cand[(r > r_in) & (r < r_out)]
        out.extend(good[:n - len(out)])
    return pv.PolyData(np.asarray(out))


plotter = pv.Plotter(shape=(2, 2), window_size=(1800, 1800),
                     border=False, off_screen=True)

for col, (snap_dir, label) in enumerate(RUNS):
    mesh, T, V = load_run(snap_dir, STEP)
    print(f"  T DOFs: {T.coords.shape[0]}  "
          f"T.data range=[{T.data[:,0].min():.4f},"
          f"{T.data[:,0].max():.4f}]")

    # P3 T field rendered on its own DOF Delaunay
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])

    # Original (linear-vertex) mesh edges for cell-quality overlay
    pv_mesh = vis.mesh_to_pv_mesh(mesh)
    edges = pv_mesh.extract_all_edges()

    # P2 V field on its own DOF cloud
    pv_V = vis.meshVariable_to_pv_mesh_object(V)
    Vfull = np.zeros((pv_V.n_points, 3))
    Vfull[:, :2] = np.asarray(V.data)
    pv_V.point_data["V"] = Vfull
    vmag = np.linalg.norm(V.data, axis=1)

    # --- T panel ---
    plotter.subplot(0, col)
    plotter.set_background("white")
    plotter.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                     clim=(0, 1), show_edges=False,
                     lighting=False,
                     scalar_bar_args={"title": "T",
                                      "vertical": False,
                                      "position_x": 0.2,
                                      "position_y": 0.02,
                                      "width": 0.6, "height": 0.04,
                                      "title_font_size": 18,
                                      "label_font_size": 16})
    plotter.add_mesh(edges, color="black", line_width=0.5,
                     lighting=False)
    plotter.add_text(
        f"{label} — step {STEP}\nT (P3) on its DOF cloud + mesh edges",
        position="upper_edge", font_size=14, color="black")
    plotter.view_xy()

    # --- V streamlines panel ---
    plotter.subplot(1, col)
    plotter.set_background("white")
    plotter.add_mesh(edges, color="gray", line_width=0.5,
                     lighting=False)
    seeds = stream_seeds(n=350)
    try:
        stream = pv_V.streamlines_from_source(
            seeds, vectors="V",
            integration_direction="both",
            max_steps=2000,
            initial_step_length=0.005,
            terminal_speed=1.0e-8,
        )
        plotter.add_mesh(stream, color="navy",
                         line_width=1.2, lighting=False)
    except Exception as e:
        print(f"streamlines failed: {e}")
    plotter.add_text(
        f"{label} — step {STEP}\n"
        f"V (P2) streamlines (|V|max={vmag.max():.0f})",
        position="upper_edge", font_size=14, color="black")
    plotter.view_xy()

plotter.screenshot(OUT_PNG, transparent_background=False,
                   window_size=(1800, 1800))
plotter.close()
print(f"wrote {OUT_PNG}")
