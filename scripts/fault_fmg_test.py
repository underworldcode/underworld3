"""Validate the FMG (geometric multigrid, Galerkin) solver on the annulus
convection problem, FIRST without the fault, then with the anisotropic fault.
Galerkin coarse operators are built from the (deformed) fine operator, so the
anisotropy should be captured across levels — testing the hypothesis that
proper deep FMG handles the TI fault where single-level GAMG/LU did not.

base-res = gmsh coarse resolution (well resolved); refinement=N adds N uniform
levels (N+1 grids), fine res = base * 2^N.

Reports per config: hierarchy levels, fine vertex count, wall, SNES (Picard)
iters, converged reason, |v|max. Writes to ~/+Simulations/StagnantLid/fault_fmg.
"""
from __future__ import annotations
import os, time, argparse
import numpy as np, sympy, underworld3 as uw

ap = argparse.ArgumentParser()
ap.add_argument('--base-res', type=int, default=32)
ap.add_argument('--refinement', type=int, default=3)   # N levels up; 3 → coarse res32, fine res256, 4 grids
ap.add_argument('--Ra', type=float, default=1e5)
ap.add_argument('--delta-eta', type=float, default=100.0)
ap.add_argument('--contrast', type=float, default=1000.0)
ap.add_argument('--fault-width', type=float, default=0.05)
ap.add_argument('--mode', type=str, default='iso',
                choices=['iso', 'iso-fault', 'fault', 'fault-adapt'])
ap.add_argument('--lag', action='store_true',
                help='snes_lag_jacobian=-2: build the Jacobian + FMG PC once.')
ap.add_argument('--smooth', type=int, default=0,
                help='mg_levels_ksp_max_it (SOR sweeps per level). 0 = FMG '
                     'default (4). Higher = stronger smoothing of the fault '
                     'contrast on each level.')
ap.add_argument('--coarse', type=str, default='lu', choices=['lu', 'gamg'],
                help='MG coarse solve: redundant LU (default) or GAMG.')
ap.add_argument('--smoother-pc', type=str, default='sor',
                help='mg_levels_pc_type: sor, ilu, asm, bjacobi, jacobi ...')
ap.add_argument('--smoother-ksp', type=str, default='richardson',
                help='mg_levels_ksp_type: richardson, chebyshev, gmres ...')
ap.add_argument('--gamg', action='store_true',
                help='Force GAMG (algebraic) instead of geometric FMG.')
args = ap.parse_args()
OUT = os.path.expanduser('~/+Simulations/StagnantLid/fault_fmg')
os.makedirs(OUT, exist_ok=True)

theta_FK = float(np.log(args.delta_eta))
inv_c = 1.0 / args.contrast

print(f"[build] Annulus base-res={args.base_res} refinement={args.refinement} "
      f"mode={args.mode} ...", flush=True)
t0 = time.time()
mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                          cellSize=1.0/args.base_res, refinement=args.refinement,
                          qdegree=3)
nlev = len(mesh.dm_hierarchy) if getattr(mesh, "dm_hierarchy", None) else 1
print(f"[build] {time.time()-t0:.1f}s  hierarchy levels={nlev}  "
      f"fine vertices={mesh.X.coords.shape[0]}", flush=True)

X = mesh.CoordinateSystem.X; r = sympy.sqrt(X[0]**2+X[1]**2); ur = mesh.CoordinateSystem.unit_e_0
th = sympy.atan2(X[1], X[0]); Tc = sympy.log(r)/sympy.log(0.5)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
V = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
P = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
T.data[:] = np.asarray(uw.function.evaluate(
    0.01*sympy.sin(5*th)*sympy.sin(np.pi*(r-0.5)/0.5) + Tc, T.coords)).reshape(-1, 1)
eta_FK = sympy.exp(theta_FK*(1 - T.sym[0]))

stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
if args.mode == 'iso':
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_FK
else:
    g = uw.discretisation.MeshVariable("g", mesh, 1, degree=2)
    delta = np.deg2rad(30.); P0 = np.array([0., 1.]); tt = np.array([-1., 0.]); e = np.array([0., 1.])
    dh = np.cos(delta)*tt - np.sin(delta)*e
    s = np.linspace(0, 0.3/np.sin(delta), 25)[:, None]; xy = P0[None, :] + s*dh[None, :]
    f = uw.meshing.Surface("f", mesh, np.column_stack([xy, np.zeros(25)]), symbol="F"); f.discretize()
    ff = f.influence_function(width=args.fault_width, value_near=inv_c, value_far=1.0, profile='gaussian'); _ = f.distance
    g.data[:, 0] = np.asarray(uw.function.evaluate(ff, g.coords)).reshape(-1)
    if args.mode == 'fault-adapt':
        print("[adapt] mmpde fault refinement ...", flush=True)
        ta = time.time()
        d = f.distance.sym[0]; rho = 1.0 + 18*sympy.exp(-(d/0.075)**2)
        uw.meshing.smooth_mesh_interior(mesh, metric=rho, method='mmpde', skip_threshold=None,
                                        slip_surfaces=True, method_kwargs=dict(relax=0.2, n_outer=12))
        f._distance_stale = True; _ = f.distance
        g.data[:, 0] = np.asarray(uw.function.evaluate(ff, g.coords)).reshape(-1)
        print(f"[adapt] {time.time()-ta:.1f}s  hierarchy levels now={len(mesh.dm_hierarchy) if getattr(mesh,'dm_hierarchy',None) else 1}", flush=True)
    if args.mode == 'iso-fault':
        # ISOTROPIC weak fault: same sharp 1000x contrast, but isotropic
        # (exact Newton, no director/TI tensor) — isolates whether FMG's
        # trouble is the TI defect-correction or the contrast itself.
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_FK*g.sym[0]
    else:
        n = np.array([-dh[1], dh[0]]); n /= np.linalg.norm(n)
        stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_FK
        stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_FK*g.sym[0]
        stokes.constitutive_model.Parameters.director = sympy.Matrix([float(n[0]), float(n[1])])

stokes.tolerance = 1e-4
stokes.add_essential_bc((0., 0.), mesh.boundaries.Lower.name)
stokes.add_nitsche_bc(mesh.boundaries.Upper.name, gamma=10.)
stokes.bodyforce = args.Ra*(T.sym[0] - Tc)*ur
stokes.preconditioner = "gamg" if args.gamg else "auto"
if args.smooth > 0 and not args.gamg:
    # Manual FMG bundle with STRONGER smoothing; _pc_user_override stops
    # _build re-applying the 4-sweep default. DM hierarchy is attached in
    # _build regardless of PC choice, so geometric MG still works.
    p = stokes._pc_option_prefix          # "fieldsplit_velocity_"
    stokes._pc_user_override = True
    o = stokes.petsc_options
    o[f"{p}pc_type"] = "mg"
    o[f"{p}pc_mg_type"] = "full"
    o[f"{p}pc_mg_galerkin"] = "both"
    o[f"{p}mg_levels_ksp_type"] = args.smoother_ksp
    o[f"{p}mg_levels_pc_type"] = args.smoother_pc
    o[f"{p}mg_levels_ksp_max_it"] = args.smooth
    o[f"{p}mg_levels_ksp_converged_maxits"] = None
    if args.coarse == "gamg":
        o[f"{p}mg_coarse_pc_type"] = "gamg"
    else:
        o[f"{p}mg_coarse_pc_type"] = "redundant"
        o[f"{p}mg_coarse_redundant_pc_type"] = "lu"
if args.lag:
    stokes.petsc_options["snes_lag_jacobian"] = -2
    stokes.petsc_options["snes_lag_jacobian_persists"] = True
print(f"[solve] preconditioner={stokes.preconditioner} smooth={args.smooth} "
      f"coarse={args.coarse} lag={args.lag} ...", flush=True)

t0 = time.time()
stokes.solve(zero_init_guess=True)
dt = time.time() - t0
its = int(stokes.snes.getIterationNumber())
try:
    ksp_tot = int(stokes.snes.getLinearSolveIterations())
except Exception:
    ksp_tot = -1
reason = int(stokes.snes.getConvergedReason())
vmax = float(np.sqrt(V.data[:, 0]**2 + V.data[:, 1]**2).max())
line = (f"RESULT mode={args.mode} levels={nlev}: wall={dt:.1f}s snes={its} "
        f"ksp_tot={ksp_tot} reason={reason} |v|max={vmax:.4e}")
print(line, flush=True)
with open(os.path.join(OUT, "fmg_results.txt"), "a") as fh:
    fh.write(line + "\n")
