"""Watch the mesh through the FIRST adaptation jump of an Ra=1e5
adaptive convection run (res-16, harness BCs — BCs not the point).

Runs 5 convection steps, UW-checkpoints the pre-adapt state, fires
the anisotropic mover once, UW-checkpoints the post-adapt state.
Visualisation follows the UW/pyvista requirement: load each
checkpoint mesh, read_timestep the P3 T onto it, render T on its
own DOF cloud (faithful high-order, NOT vertex-only) overlaid with
the deformed-mesh edges; white bg, lighting off, off-screen.

The h5 checkpoints ARE the cache — re-running only re-renders.
"""
from __future__ import annotations
import os
import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
from underworld3.meshing import (
    smooth_mesh_interior, metric_density_from_gradient)

RA, RES, N_PRE, AMP = 1.0e5, 16, 5, 8.0
r_inner, r_o = 0.5, 1.0
SNAP = "/tmp/metric_mesh/aj_snaps"
os.makedirs(SNAP, exist_ok=True)
TNAME, VNAME = "T", "V"


def build():
    mesh = uw.meshing.Annulus(
        radiusOuter=r_o, radiusInner=r_inner,
        cellSize=1.0 / RES, qdegree=3)
    r, th = mesh.CoordinateSystem.R
    v = uw.discretisation.MeshVariable(
        VNAME, mesh, vtype=uw.VarType.VECTOR, degree=2,
        continuous=True)
    P = uw.discretisation.MeshVariable(
        "P", mesh, vtype=uw.VarType.SCALAR, degree=1,
        continuous=True)
    T = uw.discretisation.MeshVariable(
        TNAME, mesh, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v,
                               pressureField=P)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.tolerance = 1.0e-5
    stokes.penalty = 0.0
    unit_r = mesh.CoordinateSystem.unit_e_0
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    stokes.add_natural_bc(1.0e6 * v.sym.dot(unit_r) * unit_r,
                          mesh.boundaries.Upper.name)
    T_cond = (r_o - r) / (r_o - r_inner)
    stokes.bodyforce = RA * (T.sym[0] - T_cond) * unit_r
    adv = uw.systems.AdvDiffusionSLCN(
        mesh, u_Field=T, V_fn=v.sym, verbose=False,
        theta=0.5, monotone_mode="clamp")
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0
    adv.tolerance = 1.0e-4
    adv.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)
    adv.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)
    init_t = (0.01 * sympy.sin(5.0 * th)
              * sympy.sin(np.pi * (r - r_inner) / (r_o - r_inner))
              + (r_o - r) / (r_o - r_inner))
    T.data[...] = np.asarray(uw.function.evaluate(
        init_t, T.coords)).reshape(-1, 1)
    return mesh, v, P, T, stokes, adv


after_h5 = f"{SNAP}/aj_after.mesh.00000.h5"
if not os.path.exists(after_h5):
    mesh, v, P, T, stokes, adv = build()
    stokes.solve(zero_init_guess=True)
    t_sim = 0.0
    for s in range(N_PRE):
        dt = adv.estimate_dt()
        adv.solve(timestep=dt, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)
        t_sim += dt
        print(f"  step {s+1} t={t_sim:.4f}", flush=True)
    mesh.write_timestep("aj_before", 0, outputPath=SNAP,
                        meshVars=[T, v], meshUpdates=True,
                        create_xdmf=True)
    print("checkpointed pre-adapt (aj_before)")
    rho = metric_density_from_gradient(mesh, T, amp=AMP,
                                       name="aj")
    smooth_mesh_interior(
        mesh, metric=rho, method="anisotropic",
        method_kwargs=dict(aniso_cap=2.0, relax=0.2, n_outer=8),
        verbose=True)
    mesh.write_timestep("aj_after", 0, outputPath=SNAP,
                        meshVars=[T, v], meshUpdates=True,
                        create_xdmf=True)
    print("checkpointed post-adapt (aj_after)")
else:
    print(f"using cached checkpoints in {SNAP}")

# ---- UW/pyvista render: T on its DOF cloud + deformed edges -----
import pyvista as pv

pv.OFF_SCREEN = True
pl = pv.Plotter(shape=(1, 2), off_screen=True,
                window_size=(2000, 1000))
pl.set_background("white")
for col, tag in enumerate(("aj_before", "aj_after")):
    m = uw.discretisation.Mesh(f"{SNAP}/{tag}.mesh.00000.h5")
    Tv = uw.discretisation.MeshVariable(
        TNAME, m, vtype=uw.VarType.SCALAR, degree=3,
        continuous=True)
    Tv.read_timestep(tag, TNAME, 0, outputPath=SNAP)
    pv_T = vis.meshVariable_to_pv_mesh_object(Tv)
    pv_T.point_data["T"] = np.asarray(Tv.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(0, col)
    pl.add_text("before 1st jump" if col == 0
                else "after 1st jump (mesh snuggled)",
                font_size=12, color="black")
    pl.add_mesh(pv_T, scalars="T", cmap="inferno",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False, show_scalar_bar=(col == 1),
                scalar_bar_args=dict(title="T", color="black"))
    pl.add_mesh(edges, color="black", line_width=0.6,
                lighting=False)
    pl.view_xy()
    pl.camera.zoom(1.3)
out = "/tmp/metric_mesh/adaptive_first_jump.png"
pl.screenshot(out)
print(f"saved {out}")
