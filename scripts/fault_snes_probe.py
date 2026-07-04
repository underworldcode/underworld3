"""Why does the TI Stokes take ~20 SNES iterations for a LINEAR problem?
Compare isotropic vs transverse-isotropic, and newtonls vs ksponly, on the
same graded annulus + fault. GAMG throughout (the probe showed GAMG ≫ FMG
for this operator). Report snes/ksp iterations, wall time, |v|max.

Writes to ~/+Simulations/StagnantLid/fault_snes_probe/.
"""
from __future__ import annotations
import os, time
import numpy as np, sympy, underworld3 as uw

OUT = os.path.expanduser('~/+Simulations/StagnantLid/fault_snes_probe')
os.makedirs(OUT, exist_ok=True)
rows = []
CONTRAST = 1000.0
theta_FK = float(np.log(100.0))
Ra = 1.0e5
inv_c = 1.0 / CONTRAST


def setup():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSizeOuter=0.04, cellSizeInner=0.10, qdegree=3)
    X = mesh.CoordinateSystem.X
    r = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    th = sympy.atan2(X[1], X[0])
    unit_r = mesh.CoordinateSystem.unit_e_0
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
    V = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
    P = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    gfac = uw.discretisation.MeshVariable("g", mesh, 1, degree=2)
    T_cond = sympy.log(r / 1.0) / sympy.log(0.5 / 1.0)
    initT = 0.01 * sympy.sin(5 * th) * sympy.sin(np.pi * (r - 0.5) / 0.5) + T_cond
    T.data[:] = np.asarray(uw.function.evaluate(initT, T.coords)).reshape(-1, 1)
    delta = np.deg2rad(30.0)
    P0 = np.array([0.0, 1.0]); t_hat = np.array([-1.0, 0.0]); e_hat = np.array([0.0, 1.0])
    dhat = np.cos(delta) * t_hat - np.sin(delta) * e_hat
    s = np.linspace(0, 0.3 / np.sin(delta), 25)[:, None]
    pts = np.column_stack([P0[None, :] + s * dhat[None, :], np.zeros(25)])
    fault = uw.meshing.Surface("fault", mesh, pts, symbol="F")
    fault.discretize()
    ff = fault.influence_function(width=0.05, value_near=inv_c, value_far=1.0, profile="gaussian")
    _ = fault.distance
    gfac.data[:, 0] = np.asarray(uw.function.evaluate(ff, gfac.coords)).reshape(-1)
    n = np.array([-dhat[1], dhat[0]]); n /= np.linalg.norm(n)
    director = sympy.Matrix([float(n[0]), float(n[1])])
    return mesh, X, r, unit_r, T, V, P, gfac, T_cond, director


def run(label, model, ksponly):
    mesh, X, r, unit_r, T, V, P, gfac, T_cond, director = setup()
    eta_FK = sympy.exp(theta_FK * (1 - T.sym[0]))
    st = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    st.constitutive_model = model
    st.constitutive_model.Parameters.shear_viscosity_0 = eta_FK
    if model is uw.constitutive_models.TransverseIsotropicFlowModel:
        st.constitutive_model.Parameters.shear_viscosity_1 = eta_FK * gfac.sym[0]
        st.constitutive_model.Parameters.director = director
    st.tolerance = 1.0e-4
    st.penalty = 0.0
    st.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    st.add_nitsche_bc(mesh.boundaries.Upper.name, gamma=10.0)
    st.bodyforce = Ra * (T.sym[0] - T_cond) * unit_r
    if ksponly:
        st.petsc_options["snes_type"] = "ksponly"
    t0 = time.time()
    st.solve(zero_init_guess=True)
    dt = time.time() - t0
    its = int(st.snes.getIterationNumber())
    try:
        ksp_its = int(st.snes.getKSP().getIterationNumber())
    except Exception:
        ksp_its = -1
    vmax = float(np.sqrt(V.data[:, 0] ** 2 + V.data[:, 1] ** 2).max())
    msg = (f"{label:<28s} wall={dt:6.1f}s  snes_its={its:>2d}  "
           f"ksp_its={ksp_its:>4d}  |v|max={vmax:.4e}")
    print(msg, flush=True); rows.append(msg)
    return vmax


run("isotropic / newtonls", uw.constitutive_models.ViscousFlowModel, False)
run("isotropic / ksponly ", uw.constitutive_models.ViscousFlowModel, True)
run("TI        / newtonls", uw.constitutive_models.TransverseIsotropicFlowModel, False)
run("TI        / ksponly ", uw.constitutive_models.TransverseIsotropicFlowModel, True)

with open(os.path.join(OUT, "snes_probe.txt"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("\n→", os.path.join(OUT, "snes_probe.txt"))
