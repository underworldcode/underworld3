"""Solver-cost probe for the dipping-fault (transverse-isotropic) Stokes
operator: time a SINGLE initial Stokes solve across viscosity-contrast ×
preconditioner, to see whether the 1000× anisotropic contrast is
intrinsically hard or specific to geometric FMG, and pick the production PC.

Writes a timing table to ~/+Simulations/StagnantLid/fault_solver_probe/.
"""
from __future__ import annotations
import os, time
import numpy as np
import sympy
import underworld3 as uw

OUT = os.path.expanduser('~/+Simulations/StagnantLid/fault_solver_probe')
os.makedirs(OUT, exist_ok=True)
rows = []

CELL_O, CELL_I = 0.04, 0.10          # moderate graded mesh (fast-ish solves)
DIP, DEPTH, WIDTH = 30.0, 0.3, 0.05
Ra = 1.0e5
theta_FK = float(np.log(100.0))      # lid contrast (FK), fixed


def build_mesh(fmg_levels):
    return uw.meshing.Annulus(
        radiusOuter=1.0, radiusInner=0.5,
        cellSizeOuter=CELL_O, cellSizeInner=CELL_I, qdegree=3,
        refinement=(fmg_levels if fmg_levels > 0 else None))


def trial(contrast, fmg_levels, label):
    inv_c = 1.0 / contrast
    mesh = build_mesh(fmg_levels)
    X = mesh.CoordinateSystem.X
    r = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    unit_r = mesh.CoordinateSystem.unit_e_0
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
    V = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
    P = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    gfac = uw.discretisation.MeshVariable("g", mesh, 1, degree=2)
    # perturbed conductive T (as in the driver)
    th = sympy.atan2(X[1], X[0])
    T_cond = sympy.log(r / 1.0) / sympy.log(0.5 / 1.0)
    initT = 0.01 * sympy.sin(5 * th) * sympy.sin(np.pi * (r - 0.5) / 0.5) + T_cond
    T.data[:] = np.asarray(uw.function.evaluate(initT, T.coords)).reshape(-1, 1)
    # fault
    delta = np.deg2rad(DIP)
    P0 = np.array([0.0, 1.0]); t_hat = np.array([-1.0, 0.0]); e_hat = np.array([0.0, 1.0])
    dhat = np.cos(delta) * t_hat - np.sin(delta) * e_hat
    L = DEPTH / np.sin(delta)
    s = np.linspace(0, L, 25)[:, None]
    pts = np.column_stack([P0[None, :] + s * dhat[None, :], np.zeros(25)])
    fault = uw.meshing.Surface("fault", mesh, pts, symbol="F")
    fault.discretize()
    ff = fault.influence_function(width=WIDTH, value_near=inv_c, value_far=1.0,
                                  profile="gaussian")
    _ = fault.distance
    gfac.data[:, 0] = np.asarray(uw.function.evaluate(ff, gfac.coords)).reshape(-1)
    n = np.array([-dhat[1], dhat[0]]); n /= np.linalg.norm(n)
    director = sympy.Matrix([float(n[0]), float(n[1])])
    eta_FK = sympy.exp(theta_FK * (1 - T.sym[0]))

    st = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    st.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = eta_FK
    st.constitutive_model.Parameters.shear_viscosity_1 = eta_FK * gfac.sym[0]
    st.constitutive_model.Parameters.director = director
    st.tolerance = 1.0e-4
    st.penalty = 0.0
    st.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    st.add_nitsche_bc(mesh.boundaries.Upper.name, gamma=10.0)
    st.bodyforce = Ra * (T.sym[0] - T_cond) * unit_r
    if fmg_levels > 0:
        st.preconditioner = "auto"

    t0 = time.time()
    converged = True
    try:
        st.solve(zero_init_guess=True)
    except Exception as e:
        converged = False
        err = str(e)[:60]
    dt = time.time() - t0
    try:
        reason = int(st.snes.getConvergedReason())
        its = int(st.snes.getIterationNumber())
    except Exception:
        reason, its = -99, -1
    vmax = float(np.sqrt(V.data[:, 0] ** 2 + V.data[:, 1] ** 2).max())
    msg = (f"contrast={contrast:>6g}  {label:<10s}  "
           f"{'OK ' if converged else 'FAIL'}  wall={dt:6.1f}s  "
           f"snes_its={its:>2d} reason={reason:>3d}  |v|max={vmax:.2e}")
    print(msg, flush=True)
    rows.append(msg)


for contrast in (100.0, 1000.0):
    trial(contrast, 0, "GAMG")
    trial(contrast, 1, "FMG-1")

with open(os.path.join(OUT, "probe.txt"), "w") as f:
    f.write("\n".join(rows) + "\n")
print(f"\n→ {os.path.join(OUT, 'probe.txt')}")
