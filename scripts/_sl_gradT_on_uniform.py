"""Plot |∇T| on the UNIFORM uniform-res16 mesh at the step-80
snapshot — to see what the gradient really looks like at the
outer (cold lid) boundary before any adaptation biases it.

Two panels:
  - |∇T| via symbolic uw.function.evaluate (per-cell P2,
    cell-ordering-dependent at shared DOFs)
  - |∇T| via Clement L2-projected nodal recovery
    (continuous P1 reconstruction)

This lets us tell whether outer-boundary refinement is
responding to a real signal or a projection artefact.
"""
import os
import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv


pv.OFF_SCREEN = True
SRC_DIR = os.path.expanduser(
    '~/+Simulations/StagnantLid/aniso_dt_validate/R1.0_aniso')
SRC_STEM = "step0080"
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/gradT_diagnosis.png')

mesh = uw.discretisation.Mesh(
    os.path.join(SRC_DIR, f"{SRC_STEM}.mesh.00000.h5"))
T = uw.discretisation.MeshVariable(
    "T_v2p1", mesh, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
T.read_timestep(SRC_STEM, "T_v2p1", 0, outputPath=SRC_DIR)
X = mesh.CoordinateSystem.X
gradT_sym = sympy.sqrt(T.sym[0].diff(X[0]) ** 2
                       + T.sym[0].diff(X[1]) ** 2)


# Method 1: symbolic per-cell evaluation
pv_T1 = vis.meshVariable_to_pv_mesh_object(T)
g_sym_vals = vis.scalar_fn_to_pv_points(pv_T1, gradT_sym)
pv_T1.point_data["gradT"] = g_sym_vals


# Method 2: Clement L2-projection — continuous P1 reconstruction
clem = uw.function.compute_clement_gradient_at_nodes(T)
# clem shape: (n_p1_nodes, dim) — at mesh.X.coords (degree=1 cloud)
clem_mag = np.linalg.norm(clem, axis=1)
# Render on degree-1 cloud
pv_geo = vis.mesh_to_pv_mesh(mesh)
pv_geo.point_data["gradT_clem"] = clem_mag


# Method 3: L2-projection onto P1 vector (the same projection
# that metric_density_from_gradient uses internally)
g_var = uw.discretisation.MeshVariable(
    "g_proj", mesh, vtype=uw.VarType.VECTOR, degree=1,
    continuous=True)
gp = uw.systems.Vector_Projection(mesh, g_var)
gp.smoothing = 0.0
gp.uw_function = sympy.Matrix(
    [T.sym[0].diff(X[i]) for i in range(2)]).T
gp.solve()
proj_mag = np.linalg.norm(
    np.asarray(g_var.data), axis=1)
pv_geo2 = vis.mesh_to_pv_mesh(mesh)
pv_geo2.point_data["gradT_proj"] = proj_mag


g_max = max(float(np.nanmax(g_sym_vals)),
            float(np.nanmax(clem_mag)),
            float(np.nanmax(proj_mag)))
print(f"|∇T|max: symbolic={float(np.nanmax(g_sym_vals)):.2e}  "
      f"clement={float(np.nanmax(clem_mag)):.2e}  "
      f"proj={float(np.nanmax(proj_mag)):.2e}")


pl = pv.Plotter(shape=(1, 3), off_screen=True,
                window_size=(1500, 500), border=False)
pl.set_background("white")
edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()


def panel(col, pv_obj, scalar, title):
    pl.subplot(0, col)
    pl.add_text(title, font_size=11, color='black')
    pl.add_mesh(pv_obj, scalars=scalar, cmap="Greens",
                clim=(0.0, g_max), show_edges=False,
                lighting=False,
                show_scalar_bar=(col == 2),
                scalar_bar_args=dict(title=r"|∇T|",
                                     color="black"))
    pl.add_mesh(edges, color="black", line_width=0.5,
                lighting=False, opacity=0.4)
    pl.view_xy(); pl.camera.zoom(1.25)


panel(0, pv_T1,    "gradT",      "symbolic (P2, per-cell)")
panel(1, pv_geo,   "gradT_clem", "Clement P1 (averaged at nodes)")
panel(2, pv_geo2,  "gradT_proj", "L2 projection P1 (what mover uses)")
pl.screenshot(OUT)
pl.close()
print(f"wrote {OUT}")
