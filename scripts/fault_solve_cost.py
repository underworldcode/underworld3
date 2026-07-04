"""Why is the TI Stokes solve slow on the mmpde-fault-adapted mesh, and what
fixes it? Build uniform res24 + fault, mmpde-adapt once, then time a cold TI
Stokes solve under a few {contrast, preconditioner} configs. Reports wall,
SNES (Picard) iters, and KSP iters so we can see whether it's the contrast,
the Picard count, or the GAMG conditioning on the graded mesh.
"""
from __future__ import annotations
import time
import numpy as np, sympy, underworld3 as uw


def build(contrast, pc, adapt, tol=1e-5, maxit=50, linesearch=None, floor_interior=False):
    m = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1/24, qdegree=3)
    X = m.CoordinateSystem.X; r = sympy.sqrt(X[0]**2 + X[1]**2); ur = m.CoordinateSystem.unit_e_0
    th = sympy.atan2(X[1], X[0]); Tc = sympy.log(r) / sympy.log(0.5)
    T = uw.discretisation.MeshVariable('T', m, 1, degree=3)
    V = uw.discretisation.MeshVariable('V', m, m.dim, degree=2)
    P = uw.discretisation.MeshVariable('P', m, 1, degree=1)
    g = uw.discretisation.MeshVariable('g', m, 1, degree=2)
    T.data[:] = np.asarray(uw.function.evaluate(
        0.01*sympy.sin(5*th)*sympy.sin(np.pi*(r-0.5)/0.5) + Tc, T.coords)).reshape(-1, 1)
    delta = np.deg2rad(30.); P0 = np.array([0., 1.]); tt = np.array([-1., 0.]); e = np.array([0., 1.])
    dh = np.cos(delta)*tt - np.sin(delta)*e
    s = np.linspace(0, 0.3/np.sin(delta), 25)[:, None]; xy = P0[None, :] + s*dh[None, :]
    f = uw.meshing.Surface('f', m, np.column_stack([xy, np.zeros(25)]), symbol='F'); f.discretize()
    ff = f.influence_function(width=0.05, value_near=1.0/contrast, value_far=1.0, profile='gaussian'); _ = f.distance
    g.data[:, 0] = np.asarray(uw.function.evaluate(ff, g.coords)).reshape(-1)
    if adapt:
        d = f.distance.sym[0]; rho = 1.0 + 18*sympy.exp(-(d/0.075)**2)
        uw.meshing.smooth_mesh_interior(m, metric=rho, method='mmpde', skip_threshold=None,
                                        slip_surfaces=True, method_kwargs=dict(relax=0.2, n_outer=12))
        f._distance_stale = True; _ = f.distance
        g.data[:, 0] = np.asarray(uw.function.evaluate(ff, g.coords)).reshape(-1)
    n = np.array([-dh[1], dh[0]]); n /= np.linalg.norm(n)
    eta = sympy.exp(np.log(100.)*(1 - T.sym[0]))
    st = uw.systems.Stokes(m, velocityField=V, pressureField=P)
    st.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    eta_1 = eta*g.sym[0]
    if floor_interior:
        eta_1 = sympy.Max(1.0, eta_1)   # weak plane floored at interior viscosity
    st.constitutive_model.Parameters.shear_viscosity_0 = eta
    st.constitutive_model.Parameters.shear_viscosity_1 = eta_1
    st.constitutive_model.Parameters.director = sympy.Matrix([float(n[0]), float(n[1])])
    st.tolerance = tol
    st.petsc_options["snes_max_it"] = maxit
    st.add_essential_bc((0., 0.), m.boundaries.Lower.name)
    st.add_nitsche_bc(m.boundaries.Upper.name, gamma=10.)
    st.bodyforce = 1e5*(T.sym[0] - Tc)*ur
    if pc == "lu":
        # direct factorisation of the velocity block (small mesh)
        st.petsc_options["fieldsplit_velocity_ksp_type"] = "preonly"
        st.petsc_options["fieldsplit_velocity_pc_type"] = "lu"
    if linesearch is not None:
        st.petsc_options["snes_linesearch_type"] = linesearch
    return st, V


# Test the interior-viscosity FLOOR on the refined mesh (no near-zero shear mode):
CONFIGS = [
    ("adapt c1000 NOfloor gamg", 1000.0, "gamg", True, 1e-4, 50, None, False),
    ("adapt c1000 FLOOR   gamg", 1000.0, "gamg", True, 1e-4, 50, None, True),
    ("adapt c1000 FLOOR   lu/bt",1000.0, "lu",   True, 1e-4, 50, "bt", True),
]
for label, c, pc, adapt, tol, maxit, ls, floor in CONFIGS:
    st, V = build(c, pc, adapt, tol=tol, maxit=maxit, linesearch=ls, floor_interior=floor)
    t0 = time.time()
    try:
        st.solve(zero_init_guess=True)
        dt = time.time() - t0
        its = int(st.snes.getIterationNumber())
        try:
            kit = int(st.snes.getKSP().getIterationNumber())
        except Exception:
            kit = -1
        vmax = float(np.sqrt(V.data[:, 0]**2 + V.data[:, 1]**2).max())
        print(f"{label:22s} wall={dt:7.1f}s  snes={its:>3d}  ksp={kit:>4d}  |v|max={vmax:.3e}", flush=True)
    except Exception as ex:
        print(f"{label:22s} FAIL {type(ex).__name__}: {str(ex)[:60]}", flush=True)
