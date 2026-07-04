"""Track down the velocity blotches: take step-60 T and re-solve the SAME TI
Stokes on (a) the saved ADAPTED mesh and (b) a clean UNIFORM res32 mesh, each
with a DIRECT (LU) solve, then render projected |v| via pyvista. If uniform is
smooth and adapted is blotchy -> the adaptation/mesh is implicated; if adapted
is smooth under LU -> the production FMG+penalty solve was under-converged.
"""
import os, glob, re, numpy as np, sympy, underworld3 as uw, pyvista as pv
pv.OFF_SCREEN = True
DIR = os.path.expanduser('~/+Simulations/StagnantLid/fault_ti_Ra1e6_fmg')
SRC = os.path.expanduser('~/+Simulations/StagnantLid/fault_ti_Ra1e6_fmg')
lab = "step0060"
Ra, dEta, floor = 1.0e6, 1000.0, 1.0
theta_FK = float(np.log(dEta))

# fault geometry (12 o'clock, 30 deg, depth 0.225, width 0.05)
delta = np.deg2rad(30.0); P0 = np.array([0., 1.])
dh = np.cos(delta)*np.array([-1., 0.]) - np.sin(delta)*np.array([0., 1.])
nvec = np.array([-dh[1], dh[0]]); nvec /= np.linalg.norm(nvec)
L = 0.225/np.sin(delta); xy = P0[None, :] + np.linspace(0, L, 25)[:, None]*dh[None, :]
director = sympy.Matrix([float(nvec[0]), float(nvec[1])])

# the adapted T (degree 3) from the saved snapshot
amesh = uw.discretisation.Mesh(os.path.join(DIR, f"{lab}.mesh.00000.h5"))
aT = uw.discretisation.MeshVariable("T_v2p1", amesh, 1, degree=3, varsymbol="T")
aT.read_timestep(lab, "T_v2p1", 0, outputPath=SRC)


def solve_and_render(mesh, Tsrc, name):
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3, varsymbol="T")
    V = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
    P = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    gfac = uw.discretisation.MeshVariable("gfac", mesh, 1, degree=2)
    T.data[:, 0] = np.asarray(uw.function.evaluate(Tsrc.sym[0], T.coords)).reshape(-1)
    fault = uw.meshing.Surface("f", mesh, np.column_stack([xy, np.zeros(25)]), symbol="F")
    fault.discretize(); _ = fault.distance
    finf_expr = fault.influence_function(width=0.05, value_near=1.0, value_far=0.0, profile="gaussian")
    gfac.data[:, 0] = np.asarray(uw.function.evaluate(finf_expr, gfac.coords)).reshape(-1)
    X = mesh.CoordinateSystem.X; r = sympy.sqrt(X[0]**2 + X[1]**2)
    unit_r = mesh.CoordinateSystem.unit_e_0
    eta_FK = sympy.exp(theta_FK*(1 - T.sym[0])); finf = gfac.sym[0]
    eta_weak = eta_FK*(1.0 - finf) + floor*finf
    st = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    st.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = eta_FK
    st.constitutive_model.Parameters.shear_viscosity_1 = eta_weak
    st.constitutive_model.Parameters.director = director
    st.tolerance = 1.0e-6; st.penalty = 0.0
    st.add_essential_bc((0.0, 0.0), "Lower")
    st.add_natural_bc(1.0e7 * V.sym.dot(unit_r) * unit_r, "Upper")
    T_cond = sympy.log(r/1.0)/sympy.log(0.5/1.0)
    st.bodyforce = Ra*(T.sym[0] - T_cond)*unit_r
    # Default fieldsplit solver (handles the saddle + pressure nullspace), tight tol.
    st.petsc_use_pressure_nullspace = True
    st.tolerance = 1.0e-7
    st.solve()
    Vm = uw.discretisation.MeshVariable(f"Vm_{name}", mesh, 1, degree=2)
    pr = uw.systems.Projection(mesh, Vm); pr.uw_function = sympy.sqrt(V.sym.dot(V.sym))
    pr.smoothing = 0.0; pr.solve()
    print(f"[{name}] LU solve: |v|max={np.sqrt((V.data**2).sum(1)).max():.2f} "
          f"reason={st.snes.getConvergedReason()}", flush=True)
    uw.visualisation.plot_scalar(mesh, Vm.sym, "Vmag", cmap="magma", clim=(0.0, 80.0),
        save_png=True, dir_fname=os.path.join(DIR, f"cmp_V_{name}.png"))
    print("→", os.path.join(DIR, f"cmp_V_{name}.png"), flush=True)


# (a) adapted mesh, LU
solve_and_render(amesh, aT, "adapted_LU")
# (b) uniform res32 mesh, LU
umesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1/32, qdegree=3)
solve_and_render(umesh, aT, "uniform_LU")
