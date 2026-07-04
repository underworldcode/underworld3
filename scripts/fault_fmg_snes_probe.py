"""Does the geometric FMG preconditioner converge under snes_type=newtonls
(linesearch=basic) vs snes_type=ksponly, on the fault Stokes operator?

Builds the production fault Stokes problem (isotropic FLOORED weak fault,
Nitsche free-slip, buoyancy) on a COARSE-base annulus with a refinement
hierarchy (dm_hierarchy → FMG velocity-block MG), loads a developed
convection T for a realistic stress test, and solves under each
(preconditioner x snes_type) combo. Reports converged reason, SNES iters,
KSP iters, wall time, |v|max.

  ./uw build NOT needed (script only). Run:
  pixi run -e amr-dev python scripts/fault_fmg_snes_probe.py --base-res 4 --levels 3
"""
from __future__ import annotations
import os, glob, re, time, argparse
import numpy as np, sympy, underworld3 as uw

ap = argparse.ArgumentParser()
ap.add_argument('--base-res', type=int, default=4)      # coarse mesh (FMG coarsest)
ap.add_argument('--levels', type=int, default=3)        # refinements: 4->8->16->32
ap.add_argument('--Ra', type=float, default=1.0e6)
ap.add_argument('--delta-eta', type=float, default=1000.0)
ap.add_argument('--fault-floor', type=float, default=1.0)
ap.add_argument('--fault-depth', type=float, default=0.225)
ap.add_argument('--fault-dip-deg', type=float, default=30.0)
ap.add_argument('--fault-width', type=float, default=0.05)
ap.add_argument('--nitsche-gamma', type=float, default=10.0)
ap.add_argument('--src-tag', type=str, default='fault_iso_Ra1e6')   # developed T
ap.add_argument('--only-ksponly', action='store_true',
                help='skip newtonls rows (scaling sweeps use the production '
                     'ksponly solver only)')
ap.add_argument('--rheology', type=str, default='isotropic',
                choices=['isotropic', 'ti'])
ap.add_argument('--ti-no-weak', action='store_true',
                help='TI code path but shear_viscosity_1 = shear_viscosity_0 '
                     '(eta_1=eta_0, mechanically isotropic). Isolates whether the '
                     'Schur count is driven by the 2nd viscosity eta_1 (which acts '
                     'on a divergence-free shear mode and SHOULD NOT couple to '
                     'pressure) or by the TI formulation/bulk contrast itself.')
ap.add_argument('--snes-max-it', type=int, default=40,
                help='SNES iteration cap (bounds a TI stall so it cannot hang).')
ap.add_argument('--penalty', type=float, default=0.0,
                help='Augmented-Lagrangian grad-div coefficient stokes.penalty '
                     '(γ(∇·u)I in the stress). Conditions the Schur complement; '
                     '∇·u=0 at the solution so the velocity is unchanged.')
ap.add_argument('--schur-precond', type=str, default='',
                help='Override pc_fieldsplit_schur_precondition (a11 default; '
                     'selfp sees the penalty-inflated diag(A); self/full).')
ap.add_argument('--pressure-ksp-maxit', type=int, default=0,
                help='Cap the outer Schur (pressure) KSP iterations (0=default 200).')
ap.add_argument('--tol', type=float, default=1.0e-5,
                help='stokes.tolerance — tighten (e.g. 1e-8) so the Krylov outer '
                     'drives div(u)->0 and a large AL penalty stays consistent.')
ap.add_argument('--saddle-pc', type=str, default='default',
                choices=['default', 'weak', 'floor', 'al'],
                help='Pressure Schur scaling stokes.saddle_preconditioner. '
                     'default=auto 1/K=1/eta_FK (bulk — WRONG in the fault zone for '
                     'TI). weak=1/eta_weak (floored effective viscosity, reflects the '
                     'fault). floor=1/fault_floor. A PRECONDITIONER: cannot change |v|.')
args = ap.parse_args()
theta_FK = float(np.log(args.delta_eta))

# developed convection T (fallback: mode-1 perturbation) -------------------
SRC = os.path.expanduser(f'~/+Simulations/StagnantLid/{args.src_tag}')
_snap = sorted(glob.glob(os.path.join(SRC, "step*.mesh.00000.h5")),
               key=lambda c: int(re.search(r"step(\d+)\.mesh", c).group(1)))
Tsnap = None
if _snap:
    _lab = re.search(r"(step\d+)\.mesh", os.path.basename(_snap[-1])).group(1)
    _ms = uw.discretisation.Mesh(_snap[-1])
    Tsnap = uw.discretisation.MeshVariable("Ts", _ms, 1, degree=3, varsymbol="T")
    Tsnap.read_timestep(_lab, "T_v2p1", 0, outputPath=SRC)
    print(f"developed T from {args.src_tag}/{_lab}", flush=True)
else:
    print("no snapshot — using mode-1 perturbation T", flush=True)

# fault geometry
dlt = np.deg2rad(args.fault_dip_deg)
P0 = np.array([0.0, 1.0]); t_hat = np.array([-1.0, 0.0]); e_hat = np.array([0.0, 1.0])
dhat = np.cos(dlt) * t_hat - np.sin(dlt) * e_hat
L = args.fault_depth / np.sin(dlt)
xy = P0[None, :] + np.linspace(0, L, 25)[:, None] * dhat[None, :]
_n = np.array([-dhat[1], dhat[0]]); _n /= np.linalg.norm(_n)
director = sympy.Matrix([float(_n[0]), float(_n[1])])


def build_and_solve(pc, snes_type, linesearch):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSize=1.0 / args.base_res, qdegree=3,
                              refinement=args.levels)
    X = mesh.CoordinateSystem.X
    r_sym = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    unit_r = mesh.CoordinateSystem.unit_e_0
    T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, varsymbol="T")
    V = uw.discretisation.MeshVariable("V_v2p1", mesh, vtype=uw.VarType.VECTOR, degree=2)
    P = uw.discretisation.MeshVariable("P_v2p1", mesh, 1, degree=1)
    gfac = uw.discretisation.MeshVariable("eta_fac", mesh, 1, degree=2)
    # T field
    T_cond = sympy.log(r_sym / 1.0) / sympy.log(0.5 / 1.0)
    if Tsnap is not None:
        T.data[:, 0] = np.asarray(uw.function.evaluate(Tsnap.sym[0], T.coords)).reshape(-1)
    else:
        th = sympy.atan2(X[1], X[0])
        ic = (0.05 * sympy.sin(1.0 * th)
              * sympy.sin(np.pi * (r_sym - 0.5) / 0.5) + T_cond)
        T.data[:, 0] = np.asarray(uw.function.evaluate(ic, T.coords)).reshape(-1)
    # fault influence -> gfac
    fault = uw.meshing.Surface("fault", mesh, np.column_stack([xy, np.zeros(25)]), symbol="F")
    fault.discretize(); _ = fault.distance
    finf_expr = fault.influence_function(width=args.fault_width, value_near=1.0,
                                         value_far=0.0, profile="gaussian")
    gfac.data[:, 0] = np.asarray(uw.function.evaluate(finf_expr, gfac.coords)).reshape(-1)
    # isotropic floored weak fault
    eta_FK = sympy.exp(theta_FK * (1 - T.sym[0]))
    finf = gfac.sym[0]
    eta_weak = eta_FK * (1.0 - finf) + float(args.fault_floor) * finf
    stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
    if args.rheology == "ti":
        # transverse-isotropic: weak ONLY to fault-parallel shear (η_1 floored),
        # isotropic η_FK away. Defect-corrected anisotropic stress → Newton.
        stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_FK
        # eta_1 = eta_weak (weak fault) OR = eta_FK (no weak plane — isolate eta_1)
        stokes.constitutive_model.Parameters.shear_viscosity_1 = (
            eta_FK if args.ti_no_weak else eta_weak)
        stokes.constitutive_model.Parameters.director = director
    else:
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_weak
    stokes.tolerance = float(args.tol)
    stokes.penalty = float(args.penalty)        # augmented-Lagrangian grad-div
    # Pressure Schur scaling: the default auto-1/K = 1/eta_FK (bulk) mis-scales
    # the fault zone for TI (soft mode is eta_weak). Override with the floored
    # effective viscosity. Pure preconditioner — does not change the solution.
    if args.saddle_pc == 'weak':
        stokes.saddle_preconditioner = 1.0 / eta_weak
    elif args.saddle_pc == 'floor':
        stokes.saddle_preconditioner = 1.0 / float(args.fault_floor)
    elif args.saddle_pc == 'al':
        # penalty-aware AL Schur scaling: S_gamma ~ 1/((1+lambda) mu)
        stokes.saddle_preconditioner = 1.0 / ((1.0 + float(args.penalty)) * eta_weak)
    stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)
    stokes.add_nitsche_bc(mesh.boundaries.Upper.name, gamma=args.nitsche_gamma)
    stokes.bodyforce = args.Ra * (T.sym[0] - T_cond) * unit_r
    # preconditioner
    if pc == "fmg":
        stokes.preconditioner = "auto"          # geometric MG on the dm_hierarchy
    elif pc == "gamg":
        stokes.preconditioner = "gamg"
    if os.environ.get("TESTJAC"):
        # Definitive Jacobian correctness check: compares the assembled tangent
        # to a finite-difference Jacobian and prints ||J - Jfd||/||J||. Large
        # ratio => the hand-assembled Jacobian is inconsistent with the residual.
        stokes.petsc_options["snes_test_jacobian"] = None
    if os.environ.get("DIRECT"):
        # Monolithic direct LU on the whole saddle system → EXACT linear solve.
        # Isolates Jacobian quality from preconditioner/Schur effects: with an
        # exact tangent and an exact linear solve, Newton converges in 1 step.
        stokes.petsc_options["pc_type"] = "lu"
        stokes.petsc_options["ksp_type"] = "fgmres"
        stokes.petsc_options["pc_factor_mat_solver_type"] = "mumps"
        stokes.petsc_options["pc_fieldsplit_type"] = ""   # neutralise fieldsplit default
        stokes.petsc_use_pressure_nullspace = True
    elif args.schur_precond:
        stokes.petsc_options["pc_fieldsplit_schur_precondition"] = args.schur_precond
    if args.pressure_ksp_maxit > 0:
        stokes.petsc_options["fieldsplit_pressure_ksp_max_it"] = int(args.pressure_ksp_maxit)
    # snes flags
    stokes.petsc_options["snes_type"] = snes_type
    stokes.petsc_options["snes_max_it"] = int(args.snes_max_it)
    if snes_type == "newtonls":
        stokes.petsc_options["snes_linesearch_type"] = linesearch
    if os.environ.get("MONITOR"):
        # Show WHERE the cost is: SNES (Newton) residual per step, and the
        # inner velocity-block KSP convergence (is FMG converging the
        # anisotropic operator, or hitting its iteration cap?).
        stokes.petsc_options["snes_monitor"] = None
        stokes.petsc_options["snes_converged_reason"] = None
        stokes.petsc_options["fieldsplit_velocity_ksp_monitor"] = None
        stokes.petsc_options["fieldsplit_velocity_ksp_converged_reason"] = None
        _vcap = os.environ.get("VEL_KSP_MAXIT")
        if _vcap:
            stokes.petsc_options["fieldsplit_velocity_ksp_max_it"] = int(_vcap)
    t0 = time.time()
    reason, snes_its, ksp_its, p_its, v_its = None, None, None, -1, -1
    try:
        stokes.solve(zero_init_guess=True)
        reason = int(stokes.snes.getConvergedReason())
        snes_its = int(stokes.snes.getIterationNumber())
        try:
            ksp_its = int(stokes.snes.getKSP().getIterationNumber())
        except Exception:
            ksp_its = -1
        # Outer Schur (pressure) and inner velocity sub-KSP iteration counts
        # of the LAST Newton step — the diagnostic that locates the bottleneck.
        try:
            sub = stokes.snes.getKSP().getPC().getFieldSplitSubKSP()
            v_its = int(sub[0].getIterationNumber())     # velocity (A00)
            p_its = int(sub[1].getIterationNumber())     # Schur / pressure
        except Exception:
            pass
    except Exception as e:
        reason = f"EXC:{str(e)[:50]}"
    dt = time.time() - t0
    vmax = float(np.sqrt(V.data[:, 0] ** 2 + V.data[:, 1] ** 2).max())
    return dict(pc=pc, snes=snes_type, ls=linesearch, reason=reason,
                snes_its=snes_its, ksp_its=ksp_its, p_its=p_its, v_its=v_its,
                t=dt, vmax=vmax)


CONFIGS = [
    ("fmg", "ksponly", "-"),
    ("fmg", "newtonls", "basic"),
    ("gamg", "ksponly", "-"),
    ("gamg", "newtonls", "basic"),
]
if args.only_ksponly:
    CONFIGS = [c for c in CONFIGS if c[1] == "ksponly"]
_only_pc = os.environ.get("ONLY_PC")        # "fmg" or "gamg" to isolate one
if _only_pc:
    CONFIGS = [c for c in CONFIGS if c[0] == _only_pc]
_only_snes = os.environ.get("ONLY_SNES")    # "ksponly" or "newtonls"
if _only_snes:
    CONFIGS = [c for c in CONFIGS if c[1] == _only_snes]
print(f"mesh: base res{args.base_res} + {args.levels} refinements "
      f"(coarsest->finest res {args.base_res}->{args.base_res * 2**args.levels}), "
      f"Ra={args.Ra:g} dEta={args.delta_eta:g} floor={args.fault_floor} "
      f"rheology={args.rheology} | schur_precond={args.schur_precond or 'a11(default)'} "
      f"penalty(AL)={args.penalty}\n", flush=True)
rows = []
for pc, st, ls in CONFIGS:
    print(f"--- {pc} / {st}{('/'+ls) if st=='newtonls' else ''} ---", flush=True)
    r = build_and_solve(pc, st, ls)
    rows.append(r)
    print(f"    reason={r['reason']} snes={r['snes_its']} schur/p_ksp={r['p_its']} "
          f"vel_ksp={r['v_its']} t={r['t']:.1f}s |v|max={r['vmax']:.4e}\n", flush=True)

print("=" * 92)
print(f"schur={args.schur_precond or 'a11'} penalty={args.penalty} rheo={args.rheology}")
print(f"{'precond':8s} {'snes':10s} {'ls':6s} {'reason':>7s} {'snes':>5s} "
      f"{'p_ksp':>6s} {'v_ksp':>6s} {'time(s)':>8s} {'|v|max':>11s}")
print("-" * 92)
for r in rows:
    rs = r['reason'] if isinstance(r['reason'], int) else str(r['reason'])
    print(f"{r['pc']:8s} {r['snes']:10s} {str(r['ls']):6s} {str(rs):>7s} "
          f"{str(r['snes_its']):>5s} {str(r['p_its']):>6s} {str(r['v_its']):>6s} "
          f"{r['t']:>8.1f} {r['vmax']:>11.4e}")
print("=" * 92)
print("p_ksp = outer Schur(pressure) KSP iters (last Newton step) — the TI bottleneck. "
      "v_ksp = velocity FMG iters.")
