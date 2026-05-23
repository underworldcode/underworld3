"""OT-improve + heuristic interleave sweep on synthetic shapes.

Tests the architectural payoff of "OT as a composable step":
each row of the grid is a different recipe alternating OT
improvements with shape-quality moves. We track the OT imbalance
metric (std of log(V·ρ/K)) per step and render the final mesh.
"""
import os
import sys
import io
import re
import contextlib
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_OT_interleave')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes_analytic_disp import analytic_rho


def build_uniform_mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
        cellSize=0.04, qdegree=3)


# ρ background (on its own undeformed reference)
m_bg = build_uniform_mesh()
T_bg = uw.discretisation.MeshVariable(
    "T_bg_rho", m_bg, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
T_bg.data[:, 0] = np.asarray(uw.function.evaluate(
    analytic_rho(m_bg), np.asarray(T_bg.coords))).reshape(-1)
rho_clip = (1.0, float(T_bg.data[:, 0].max()))
pv_bg = vis.meshVariable_to_pv_mesh_object(T_bg)
pv_bg.point_data["rho"] = np.asarray(T_bg.data[:, 0])


def step_OT(mesh, rho, relax=0.1):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        uw.meshing.smooth_mesh_interior(
            mesh, metric=rho, method="ot", verbose=True,
            boundary_slip="box",
            method_kwargs=dict(n_outer=1, relax=relax,
                                step_frac=0.3))
    # Pull the imbalance from the verbose line.
    out = buf.getvalue()
    m = re.search(r"imb=([0-9.e+-]+)", out)
    return float(m.group(1)) if m else None


def step_spring_metric(mesh, rho):
    # spring with metric — uses ρ to grade per-cell rest lengths
    with contextlib.redirect_stdout(io.StringIO()):
        uw.meshing.smooth_mesh_interior(
            mesh, metric=rho, method="spring",
            boundary_slip="box")


def step_jacobi(mesh):
    # plain graph-Laplacian smoothing (no metric)
    with contextlib.redirect_stdout(io.StringIO()):
        uw.meshing.smooth_mesh_interior(
            mesh, n_iters=3, alpha=0.5)


# Recipes: each is a list of strings 'OT'|'spring'|'jacobi'
# describing the per-call action. Total #OT steps kept ≈ 5 across
# recipes (apples-to-apples on "OT budget").
RECIPES = [
    ("A. OT x 5 (baseline)",
     ["OT"] * 5),
    ("B. OT x 15 (raw push)",
     ["OT"] * 15),
    ("C. (OT, Jacobi) x 5",
     ["OT", "jacobi"] * 5),
    ("D. (OT, spring) x 5",
     ["OT", "spring"] * 5),
    ("E. (OT x 3, Jacobi) x 3",
     (["OT"] * 3 + ["jacobi"]) * 3),
    ("F. (OT x 3, spring) x 3",
     (["OT"] * 3 + ["spring"]) * 3),
]


def run_recipe(label, actions, rho_sym):
    m = build_uniform_mesh()
    # Build a sympy ρ on THIS mesh's coord system.
    rho = analytic_rho(m)
    imb_traj = []
    for step in actions:
        if step == "OT":
            imb = step_OT(m, rho)
            if imb is not None:
                imb_traj.append(imb)
        elif step == "spring":
            step_spring_metric(m, rho)
        elif step == "jacobi":
            step_jacobi(m)
        else:
            raise ValueError(step)
    return m, imb_traj


# Run all recipes
results = {}
for label, actions in RECIPES:
    print(f"\n=== {label} ({len(actions)} actions) ===")
    m, imb = run_recipe(label, actions, None)
    final_imb = imb[-1] if imb else float("nan")
    n_ot = sum(1 for a in actions if a == "OT")
    print(f"  {n_ot} OT steps; imb traj first/last: "
          f"{imb[0] if imb else 'NA'} → {final_imb}")
    results[label] = (m, imb, final_imb, n_ot)


# Render 2×3 grid
ncols, nrows = 3, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1200 * ncols, 1200 * nrows),
                border=False)
pl.set_background("white")

for i, (label, _) in enumerate(RECIPES):
    row, col = i // ncols, i % ncols
    m, imb, final_imb, n_ot = results[label]
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(f"{label}\nimb={final_imb:.3f}  "
                f"({n_ot} OT)",
                font_size=20, color='black')
    pl.add_mesh(pv_bg, scalars="rho", cmap="Blues",
                clim=rho_clip, show_edges=False,
                lighting=False, show_scalar_bar=False,
                opacity=0.85)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_OT_interleave.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")

# Print imb trajectories for the record
print("\n--- imbalance trajectories (per OT step) ---")
for label, _ in RECIPES:
    _, imb, _, _ = results[label]
    traj_str = " ".join(f"{v:.3f}" for v in imb)
    print(f"  {label}\n    {traj_str}")
