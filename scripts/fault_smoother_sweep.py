"""Smoother sweep for the TI weak-fault Stokes on a uniform res32 / 4-level
hierarchy. Compares GAMG vs geometric FMG with several velocity-block
smoothers. Each solve is BOUNDED (snes_max_it + inner ksp_max_it) so a poor
config reports quickly instead of grinding. Reports SNES (Newton) iters, total
inner KSP iters (the smoother-quality diagnostic), converged reason, wall.

Mesh + fields built ONCE; a fresh Stokes solver per config.
Writes ~/+Simulations/StagnantLid/fault_smoother_sweep/sweep.txt
"""
from __future__ import annotations
import os, time
import numpy as np, sympy, underworld3 as uw

OUT = os.path.expanduser('~/+Simulations/StagnantLid/fault_smoother_sweep')
os.makedirs(OUT, exist_ok=True)
results = []

CONTRAST = 1000.0
theta_FK = float(np.log(100.0))
Ra = 1e5
inv_c = 1.0 / CONTRAST
SNES_MAXIT = 25
KSP_MAXIT = 200          # bound the inner velocity solve

# ---- build mesh + fields ONCE (base4 refine3 → res32 fine, 4 levels) ----
mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1/4,
                          refinement=3, qdegree=3)
nlev = len(mesh.dm_hierarchy)
X = mesh.CoordinateSystem.X; r = sympy.sqrt(X[0]**2+X[1]**2); ur = mesh.CoordinateSystem.unit_e_0
th = sympy.atan2(X[1], X[0]); Tc = sympy.log(r)/sympy.log(0.5)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
g = uw.discretisation.MeshVariable("g", mesh, 1, degree=2)
T.data[:] = np.asarray(uw.function.evaluate(
    0.01*sympy.sin(5*th)*sympy.sin(np.pi*(r-0.5)/0.5) + Tc, T.coords)).reshape(-1, 1)
delta = np.deg2rad(30.); P0 = np.array([0., 1.]); tt = np.array([-1., 0.]); e = np.array([0., 1.])
dh = np.cos(delta)*tt - np.sin(delta)*e
s = np.linspace(0, 0.3/np.sin(delta), 25)[:, None]; xy = P0[None, :] + s*dh[None, :]
f = uw.meshing.Surface("f", mesh, np.column_stack([xy, np.zeros(25)]), symbol="F"); f.discretize()
ff = f.influence_function(width=0.05, value_near=inv_c, value_far=1.0, profile='gaussian'); _ = f.distance
g.data[:, 0] = np.asarray(uw.function.evaluate(ff, g.coords)).reshape(-1)
nrm = np.array([-dh[1], dh[0]]); nrm /= np.linalg.norm(nrm)
print(f"[mesh] res32 fine, levels={nlev}, vertices={mesh.X.coords.shape[0]}", flush=True)

FMG = dict(pc_type="mg", pc_mg_type="full", pc_mg_galerkin="both",
           mg_coarse_pc_type="redundant", mg_coarse_redundant_pc_type="lu")

# label -> (use_gamg, smoother-option-dict over the FMG base)
CONFIGS = [
    ("gamg",            True,  None),
    ("fmg sor x8",      False, dict(mg_levels_ksp_type="richardson", mg_levels_pc_type="sor",  mg_levels_ksp_max_it=8)),
    ("fmg sor x16",     False, dict(mg_levels_ksp_type="richardson", mg_levels_pc_type="sor",  mg_levels_ksp_max_it=16)),
    ("fmg sor x32",     False, dict(mg_levels_ksp_type="richardson", mg_levels_pc_type="sor",  mg_levels_ksp_max_it=32)),
    ("fmg ilu x4",      False, dict(mg_levels_ksp_type="richardson", mg_levels_pc_type="ilu",  mg_levels_ksp_max_it=4)),
    ("fmg asm x4",      False, dict(mg_levels_ksp_type="richardson", mg_levels_pc_type="asm",  mg_levels_ksp_max_it=4)),
    ("fmg cheby-sor x6",False, dict(mg_levels_ksp_type="chebyshev",  mg_levels_pc_type="sor",  mg_levels_ksp_max_it=6)),
]


def make_solver():
    V = uw.discretisation.MeshVariable(f"V{make_solver.i}", mesh, mesh.dim, degree=2)
    P = uw.discretisation.MeshVariable(f"P{make_solver.i}", mesh, 1, degree=1)
    make_solver.i += 1
    st = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    st.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = sympy.exp(theta_FK*(1-T.sym[0]))
    st.constitutive_model.Parameters.shear_viscosity_1 = sympy.exp(theta_FK*(1-T.sym[0]))*g.sym[0]
    st.constitutive_model.Parameters.director = sympy.Matrix([float(nrm[0]), float(nrm[1])])
    st.tolerance = 1e-4
    st.add_essential_bc((0., 0.), mesh.boundaries.Lower.name)
    st.add_nitsche_bc(mesh.boundaries.Upper.name, gamma=10.)
    st.bodyforce = Ra*(T.sym[0]-Tc)*ur
    st.petsc_options["snes_max_it"] = SNES_MAXIT
    st.petsc_options["fieldsplit_velocity_ksp_max_it"] = KSP_MAXIT
    return st, V
make_solver.i = 0


for label, use_gamg, smoother in CONFIGS:
    st, V = make_solver()
    p = st._pc_option_prefix
    if use_gamg:
        st.preconditioner = "gamg"
    else:
        st._pc_user_override = True
        o = st.petsc_options
        for k, v in FMG.items():
            o[f"{p}{k}"] = v
        for k, v in smoother.items():
            o[f"{p}{k}"] = v
        o[f"{p}mg_levels_ksp_converged_maxits"] = None
    t0 = time.time()
    try:
        st.solve(zero_init_guess=True)
        dt = time.time() - t0
        its = int(st.snes.getIterationNumber())
        kit = int(st.snes.getLinearSolveIterations())
        reason = int(st.snes.getConvergedReason())
        vmax = float(np.sqrt(V.data[:, 0]**2 + V.data[:, 1]**2).max())
        msg = (f"{label:18s} wall={dt:7.1f}s  newton={its:>2d}  ksp_tot={kit:>5d}  "
               f"reason={reason:>3d}  |v|max={vmax:.4e}  {'CONV' if reason>0 else 'FAIL'}")
    except Exception as ex:
        msg = f"{label:18s} EXC {type(ex).__name__}: {str(ex)[:50]}"
    print(msg, flush=True)
    results.append(msg)

with open(os.path.join(OUT, "sweep.txt"), "w") as fh:
    fh.write(f"TI fault, res32, {nlev} levels, contrast {CONTRAST:g}, "
             f"snes_max_it={SNES_MAXIT}, ksp_max_it={KSP_MAXIT}\n")
    fh.write("\n".join(results) + "\n")
print("DONE", flush=True)
