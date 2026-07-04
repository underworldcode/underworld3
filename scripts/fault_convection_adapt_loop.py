"""Stagnant-lid annulus convection with a DIPPING FAULT, periodic
mesh adaptation, surface slip and FMG.

This is the fault variant of ``stagnant_lid_adapt_loop.py``. The
fault is a low-viscosity weak zone represented with the
**anisotropic (transverse-isotropic) viscous model**: the material
is weak to shear PARALLEL to the fault plane (small ``shear_viscosity_1``
on the fault, isotropic away from it), while the bulk
Frank-Kamenetskii temperature dependence rides on both viscosities.

Key pieces vs the baseline:
  * graded gmsh mesh — fine near the outer (cold) surface where the
    thermal boundary layer and the fault root live, coarse at the base;
    composes with ``refinement=N`` for the FMG dm_hierarchy.
  * a dipping fault embedded as a ``uw.meshing.Surface`` (12 o'clock,
    30 deg dip, finite depth) — the mover refines AND aligns cells along it.
  * ``TransverseIsotropicFlowModel`` weak fault; the fault factor is
    precomputed onto a MeshVariable (no ``exp`` of a MeshVariable in the
    constitutive tensor — see UW3 smoother/constitutive footguns).
  * surface slip (``slip_surfaces=True``) so boundary nodes slide
    tangentially along the annulus arcs (parallel-safe orchestrator;
    Annulus auto-registers Upper/Lower).
  * Nitsche free-slip on the outer boundary (penalty fallback available).
  * geometric FMG on the velocity block (``stokes.preconditioner='auto'``).

Loop pattern per step (unchanged from the baseline):
  1. estimate dt
  2. if (step % adapt_every == 0): build combined metric (|grad T| x fault),
     call the anisotropic mover with surface slip + skip_threshold; the
     mover owns field transfer. Refresh the fault distance + factor, zero V,P.
  3. solve Stokes (warm if no adapt this step; cold if adapt)
  4. solve advdiff
  5. log Nu, vrms, T extents
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import numpy as np
import sympy
import underworld3 as uw


p = argparse.ArgumentParser()
# --- start state ---
p.add_argument('--src-dir', type=str,
               default=os.path.expanduser(
                   '~/+Simulations/StagnantLid/'
                   'uniform_res16_Ra1e7_dEta1e4'))
p.add_argument('--src-stem', type=str,
               default='sl_uniform_res16_Ra1e7_dEta1e4_step00125')
p.add_argument('--from-perturbation', action='store_true',
               help='Start from the near-conductive initial state '
                    '(T_cond + small mode perturbation, V=P=0) on a '
                    'fresh graded Annulus instead of a checkpoint. '
                    'This is the natural entry point for the fault run.')
p.add_argument('--resume', action='store_true')
# --- mesh / FMG ---
p.add_argument('--res', type=int, default=24,
               help='UNIFORM base resolution (cells across the radial '
                    'gap); cellSize = 1/res. Default 24. A uniform mesh '
                    'gives the mover a full node budget to redistribute '
                    'toward the fault (a graded mesh starves the '
                    'interior). Use 24 or 32.')
p.add_argument('--cell-outer', type=float, default=0.0,
               help='OPTIONAL graded override: gmsh cell size at the '
                    'outer surface. 0 (default) = uniform --res. Set '
                    'both --cell-outer and --cell-inner for a graded mesh.')
p.add_argument('--cell-inner', type=float, default=0.0,
               help='OPTIONAL graded override: gmsh cell size at the '
                    'inner boundary. 0 (default) = uniform --res.')
p.add_argument('--fmg-levels', type=int, default=0,
               help='Uniform refinement levels → dm_hierarchy for '
                    'geometric FMG on the velocity block. 0 (DEFAULT) = '
                    'GAMG. NOTE: geometric FMG is COUNTERPRODUCTIVE for the '
                    'transverse-isotropic operator (probe: ~7x slower than '
                    'GAMG); GAMG handles the anisotropic 1000x contrast '
                    'robustly (~same cost as 100x). Leave at 0 unless '
                    'experimenting. Requires the development branch.')
# --- fault geometry ---
p.add_argument('--fault-theta-deg', type=float, default=90.0,
               help='Azimuth (deg) of the fault surface trace. 90 = '
                    '12 o\'clock (top of the annulus).')
p.add_argument('--fault-dip-deg', type=float, default=30.0,
               help='Dip angle (deg) below the local horizontal at the '
                    'surface intersection. 30 = shallow dipping fault.')
p.add_argument('--fault-depth', type=float, default=0.225,
               help='Radial depth of the fault tip below the outer '
                    'surface (r=1.0). Down-dip length L = depth/sin(dip). '
                    'Default 0.225 (≈75%% of the original 0.3) → shorter '
                    'fault, tip near r≈0.90, kept within the lid region.')
p.add_argument('--fault-dip-dir', type=str, default='east',
               choices=['east', 'west'],
               help='Which way the fault dips from 12 o\'clock: east '
                    '(+x, toward 3 o\'clock) or west (−x).')
# --- fault rheology ---
p.add_argument('--rheology', type=str, default='ti',
               choices=['ti', 'isotropic'],
               help='Fault rheology. ti = transverse-isotropic weak plane '
                    '(physically intended; expensive solve on a resolved '
                    'fault). isotropic = whole viscosity drops in the fault '
                    'zone (linear, cheap — use to validate the physics).')
p.add_argument('--no-fault', action='store_true',
               help='Disable the fault entirely: pure Frank-Kamenetskii '
                    'convection (isotropic eta_FK, |grad T|-only adaptation, no '
                    'fault metric). Use as the faultless control — same mesh, '
                    'Ra, viscosity, mover and solver, so the fault is the only '
                    'difference vs a fault run.')
p.add_argument('--fault-passive', action='store_true',
               help='PASSIVE fault: the mesh still REFINES to the fault (gmsh '
                    'base + fault metric/tensor kept) but the fault has NO '
                    'mechanical effect (isotropic eta_FK, finf=0 — same rheology '
                    'as --no-fault). Use to watch the adaptive meshing track both '
                    'the thermal field and the (inert) fault before turning on '
                    'fault mechanics.')
p.add_argument('--old-frame', action='store_true',
               help='Enable old-frame SL reach-back on the T advection (DuDt): the '
                    'semi-Lagrangian trace-back samples psi_star in the PREVIOUS '
                    "(pre-adapt) mesh geometry + direct nodal carry, so mesh motion "
                    "doesn't remap-diffuse the field. This is the free-surface "
                    "exact-carry fix; expected to remove the convection damping a "
                    "moving fault metric otherwise causes. V_fn stays the PHYSICAL "
                    "velocity (contract).")
p.add_argument('--fault-base-smin', type=float, default=0.0,
               help='If >0, build the INITIAL mesh with a gmsh refine_lines fault '
                    'base at this min cell size, so the fault is refined AT '
                    'CONSTRUCTION (the mover only maintains it). The node-moving '
                    'movers cannot CREATE fault refinement from a uniform mesh '
                    '(capped ~2x, and the fault sits in the low-gradient cold '
                    'lid), so a gmsh base is needed for a resolved fault. '
                    '0 = plain mesh (mover-only, weak). Ignored with --no-fault.')
p.add_argument('--fault-floor', type=float, default=1.0,
               help='Viscosity FLOOR in the fault zone — the fault weakens '
                    'the (cold, stiff) lid down to this value, not below. '
                    'Default 1.0 = the interior/asthenosphere viscosity, so '
                    'with --delta-eta 1000 the fault is 1000x weaker than the '
                    'surrounding lid but never near-zero (physical + better '
                    'conditioned).')
p.add_argument('--fault-eta-contrast', type=float, default=1000.0,
               help='(legacy; superseded by --fault-floor) nominal '
                    'bulk/fault ratio, retained for the output tag only.')
p.add_argument('--fault-width', type=float, default=0.04,
               help='Gaussian half-width of the weak zone (model '
                    'coordinates). Default 0.04.')
# --- fault-aware refinement ---
p.add_argument('--fault-anisotropy', type=float, default=8.0,
               help='ANISOTROPIC fault metric ratio Rf for the mmpde mover. '
                    'The metric is the SPD tensor M = rho_T·I + (Rf²-1)·'
                    'exp(-(d/W)²)·n nᵀ — fine ONLY across the fault normal n, '
                    'so cells go thin-across / long-along and a node row lands '
                    'ON the centre-line (a scalar density bump just refines a '
                    'fat isotropic corridor and leaves the centre coarse). '
                    'Rf=8 makes across-fault refinement stronger than the '
                    'hottest plume; 0 disables (fall back to scalar amp).')
p.add_argument('--fault-refine-amp', type=float, default=12.0,
               help='SCALAR fallback density amplitude (only used when '
                    '--fault-anisotropy 0 or a non-mmpde mover): rho_fault '
                    'peaks '
                    'at 1+amp on the fault, → 1 away. 0 = no fault-driven '
                    'refinement (gradient-only). Default 12 (strong, so '
                    'the fault dominates the weak conductive |∇T| signal '
                    'and pulls nodes toward itself).')
p.add_argument('--resolution-ratio', type=float, default=2.5,
               help='Mover cell-size clamp R: cells may shrink to h0/R '
                    '(fine, on the fault) and grow to h0·R (coarse, away). '
                    'This is the HARD cap on how far the mover can deform '
                    '— the named strategies top out at R=2 (extreme). '
                    'Default 2.5 so the fault genuinely carves a refined '
                    'corridor and the outer ring slips toward it. Higher '
                    '(4-8) is more dramatic but can tangle the mesh.')
p.add_argument('--fault-refine-width', type=float, default=-1.0,
               help='Gaussian half-width of the SHARP across-fault (anisotropic) '
                    'localisation band. < 0 (default) ⇒ 1.5·fault_width.')
p.add_argument('--fault-draw-width', type=float, default=-1.0,
               help='Gaussian half-width of the WIDE isotropic DRAW footprint '
                    'that pulls distant nodes toward the fault (gives the fault '
                    'enough weight in the mmpde global functional to be served). '
                    '< 0 (default) ⇒ 4·refine_w. Wider draws from farther but '
                    'competes more with the TBL for nodes.')
p.add_argument('--metric-combine', type=str, default='product',
               choices=['product', 'max'],
               help='How to fuse the |grad T| density with the fault '
                    'density. product = budget shared (default); max = '
                    'cap the fused peak.')
p.add_argument('--method', type=str, default='mmpde',
               choices=['mmpde', 'ot-reset', 'ot', 'anisotropic'],
               help='Mesh mover. mmpde (DEFAULT) = the variational, '
                    'non-folding (G→∞ as detJ→0) Huang–Kamenski mover that '
                    'clusters AND aligns cells to the metric and runs its OWN '
                    'nonlinear-CG iteration (accel=cg). This is the production '
                    'mover for the convection movies. "R" (resolution_ratio) '
                    'is NOT a mover clamp — it is the metric grading ratio fed '
                    'to metric_density_from_gradient(refinement=R, '
                    'front-following). ot-reset = mesh.OT_adapt reset-based '
                    'mover (sliver-free, fault rides as analytic extra_density; '
                    'good but a different feel). anisotropic SHREDS/backtracks '
                    'on a static fault — comparison only.')
# --- free slip BC ---
p.add_argument('--freeslip', type=str, default='nitsche',
               choices=['nitsche', 'penalty', 'constrained'],
               help='Outer free-slip enforcement. nitsche (default, gamma), '
                    'penalty (KFS natural BC), or constrained '
                    '(Stokes_Constrained in-saddle Lagrange multiplier — '
                    'exact TI Jacobian, no penalty stiffness, recoverable '
                    'topography; the clean parallel recipe).')
p.add_argument('--nitsche-gamma', type=float, default=10.0,
               help='Nitsche penalty parameter. Default 10.')
p.add_argument('--kfs', type=float, default=1.0e6,
               help='Penalty free-slip stiffness (penalty mode only).')
# --- mover / adaptation ---
p.add_argument('--strategy', type=str, default='med',
               choices=list(uw.meshing.ADAPT_STRATEGIES.keys()))
p.add_argument('--adapt-every', type=int, default=5)
p.add_argument('--fault-prerefine', type=int, default=0,
               help='Run N fault-focused adapt passes BEFORE the time loop, '
                    'while T is still conductive (no thermal boundary layer '
                    'competing for nodes) — so the mover concentrates on the '
                    'fault unopposed, then convection maintains it. ~8-15 with '
                    'annealing (below) gets past the ridge-crossing barrier.')
p.add_argument('--fault-prerefine-width0', type=float, default=4.0,
               help='Two-stage anneal: the fault metric width starts this '
                    'multiple WIDER on pass 1 and sharpens geometrically to 1.0 '
                    'by the last pass. Wide early = mover-resolvable on the '
                    'coarse mesh (pulls nodes in); sharp late = localised on the '
                    'centre-line. 1.0 disables annealing (constant width).')
p.add_argument('--adapt-start', type=int, default=0,
               help='Do NOT adapt before this step (run on the fixed initial '
                    'mesh first). Use e.g. 10 to let convection develop on a '
                    'static mesh, then enable adaptation — isolates whether a '
                    'failure mode is in the base advection scheme or the '
                    'adapt/history-remap.')
p.add_argument('--skip-threshold', type=float, default=-1.0,
               help='Override the adapt skip threshold. -1 = strategy '
                    'default (~0.9). >10 = never skip. 0 = always skip.')
p.add_argument('--grad-smooth-h0', type=float, default=0.0,
               help='gradient_smoothing_length as a multiple of mean h0. '
                    '0 = no smoothing; 2.0 = production de-noising.')
p.add_argument('--slip-surfaces', dest='slip_surfaces',
               action='store_true',
               help='Enable tangential boundary slip (default ON).')
p.add_argument('--no-slip-surfaces', dest='slip_surfaces',
               action='store_false',
               help='Disable tangential boundary slip (pin boundary '
                    'nodes). Slip is ON by default.')
p.set_defaults(slip_surfaces=True)
# --- physics ---
p.add_argument('--Ra', type=float, default=1.0e7,
               help='Rayleigh number (default 1e7).')
p.add_argument('--delta-eta', type=float, default=1.0e4,
               help='Frank-Kamenetskii viscosity contrast '
                    'eta(cold)/eta(hot). Default 1e4.')
p.add_argument('--pert-mode', type=int, default=5,
               help='Azimuthal wavenumber of the initial T perturbation.')
p.add_argument('--pert-amplitude', type=float, default=0.01)
p.add_argument('--pert-phase-deg', type=float, default=0.0,
               help='Azimuthal phase (deg) of the initial T perturbation: '
                    'sin(mode*(theta - phase)). With mode 3, phase 0 already '
                    'places a downwelling at theta=90 (the fault). Use to put '
                    'an up/downwelling under the fault.')
p.add_argument('--dt-mult', type=float, default=1.0,
               help='Multiplier on estimate_dt (SLCN is unconditionally '
                    'stable; 3-5 gives larger physical-time steps).')
p.add_argument('--mover-accel', type=str, default='none',
               help="mmpde acceleration: 'none' (default; the heavy-ball/CG "
                    "accel diverges — area ratio runs away to ~900 on a fixed "
                    "metric), 'cg', or 'heavy'. Use 'none' until the CG "
                    "acceleration (commit c25a4b7a) is fixed.")
p.add_argument('--cold-stokes', action='store_true',
               help='Force zero_init_guess=True on EVERY Stokes solve (cold '
                    'restart) instead of warm-restarting from the previous V. '
                    'Diagnostic: Nitsche free-slip appears to accumulate a '
                    'spurious v·n throughflow under warm restart.')
# --- run control ---
p.add_argument('--n-steps', type=int, default=100)
p.add_argument('--max-t', type=float, default=0.0)
p.add_argument('--log-every', type=int, default=2)
p.add_argument('--snapshot-every', type=int, default=20)
p.add_argument('--out-tag', type=str, default=None)
p.add_argument('--sim-dir', type=str, default='~/+Simulations/StagnantLid',
               help='Base output directory; runs go in <sim-dir>/<out-tag>. '
                    'Use ~/+Simulations/StagnantLid+Fault for the fault study.')
args = p.parse_args()


tag = args.out_tag or (
    f"fault_dip{int(args.fault_dip_deg)}_c{int(args.fault_eta_contrast)}"
    f"_{args.strategy}")
OUT_DIR = os.path.expanduser(os.path.join(args.sim_dir, tag))
os.makedirs(OUT_DIR, exist_ok=True)

Ra = float(args.Ra)
theta_FK = float(np.log(float(args.delta_eta)))
STRAT = uw.meshing.ADAPT_STRATEGIES[args.strategy]
inv_c = 1.0 / float(args.fault_eta_contrast)
refine_w = (args.fault_refine_width if args.fault_refine_width > 0
            else 1.5 * args.fault_width)
# Wide isotropic DRAW footprint (default 4×refine_w): pulls distant nodes
# toward the fault so it carries enough weight in the mmpde global functional.
draw_w = (args.fault_draw_width if args.fault_draw_width > 0
          else 4.0 * refine_w)
# Annealing scale on the fault widths, driven by the two-stage pre-refine loop
# (1.0 = production sharpness; >1 = wider/easier for the mover to resolve).
_width_scale = 1.0

_rheo_desc = "NO-FAULT (isotropic FK control)" if args.no_fault else args.rheology
print(f"=== fault convection: rheology={_rheo_desc} dip={args.fault_dip_deg}deg "
      f"depth={args.fault_depth} deltaEta={args.delta_eta:g} "
      f"fault_floor={args.fault_floor:g} strategy={args.strategy} ===")
_mesh_desc = (f"graded {args.cell_outer:.4g}/{args.cell_inner:.4g}"
              if (args.cell_outer > 0 and args.cell_inner > 0)
              else f"uniform res{args.res}")
print(f"  mesh={_mesh_desc}, mover={args.method}, "
      f"fault_amp={args.fault_refine_amp}, slip={args.slip_surfaces}, "
      f"freeslip={args.freeslip}, FMG={args.fmg_levels}")
print(f"  out: {OUT_DIR}")


# --- resume / fresh-start logic ---
def _latest_snapshot():
    import glob, re
    fs = glob.glob(os.path.join(OUT_DIR, "step*.mesh.00000.h5"))
    idxs = []
    for f in fs:
        m = re.search(r"step(\d+)\.mesh\.00000\.h5$", os.path.basename(f))
        if m:
            idxs.append(int(m.group(1)))
    if not idxs:
        return None
    s_max = max(idxs)
    return s_max, f"step{s_max:04d}"


resume_info = _latest_snapshot() if args.resume else None
if resume_info is not None:
    resume_step, resume_label = resume_info
    print(f"  resuming from {resume_label}")
    mesh = uw.discretisation.Mesh(
        os.path.join(OUT_DIR, f"{resume_label}.mesh.00000.h5"))
elif args.from_perturbation:
    resume_step = 0
    resume_label = None
    _ref = (args.fmg_levels if args.fmg_levels > 0 else None)
    # gmsh fault base: refine the initial mesh ALONG the fault trace so the
    # fault is resolved at construction (the mover cannot create this from a
    # uniform mesh — it caps ~2x and the fault is in the low-gradient lid).
    _base_lines = None
    if args.fault_base_smin > 0 and not args.no_fault:
        _d, _th = np.deg2rad(args.fault_dip_deg), np.deg2rad(args.fault_theta_deg)
        _P0 = np.array([np.cos(_th), np.sin(_th)])
        _e, _t = np.array([np.cos(_th), np.sin(_th)]), np.array([-np.sin(_th), np.cos(_th)])
        _sd = 1.0 if args.fault_dip_dir == 'east' else -1.0
        _dh = _sd * np.cos(_d) * _t - np.sin(_d) * _e
        _L = args.fault_depth / np.sin(_d)
        _base_lines = [np.array([_P0]) + np.linspace(0.0, _L, 40)[:, None] * _dh[None, :]]
        print(f"  fault gmsh base: refine_lines smin={args.fault_base_smin}", flush=True)
    _rl = dict(refine_lines=_base_lines, refine_size_min=args.fault_base_smin,
               refine_dist_min=0.02, refine_dist_max=0.12) if _base_lines else {}
    if args.cell_outer > 0 and args.cell_inner > 0:
        # optional graded override
        mesh = uw.meshing.Annulus(
            radiusOuter=1.0, radiusInner=0.5,
            cellSizeOuter=args.cell_outer, cellSizeInner=args.cell_inner,
            qdegree=3, refinement=_ref, **_rl)
    else:
        # uniform mesh (default) + optional gmsh fault base via _rl.
        mesh = uw.meshing.Annulus(
            radiusOuter=1.0, radiusInner=0.5,
            cellSize=1.0 / args.res, qdegree=3, refinement=_ref, **_rl)
else:
    resume_step = 0
    resume_label = None
    mesh = uw.discretisation.Mesh(
        os.path.join(args.src_dir, f"{args.src_stem}.mesh.00000.h5"))

T = uw.discretisation.MeshVariable(
    "T_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=3,
    continuous=True, varsymbol="T")
V = uw.discretisation.MeshVariable(
    "V_v2p1", mesh, vtype=uw.VarType.VECTOR, degree=2,
    continuous=True, varsymbol=r"\mathbf{v}")
P = uw.discretisation.MeshVariable(
    "P_v2p1", mesh, vtype=uw.VarType.SCALAR, degree=1,
    continuous=True, varsymbol="p")
# Precomputed fault INFLUENCE f: 1 on the fault → 0 away. Created up front
# (before any mover call) so the DM is stable for the loop. (Var name kept
# as 'eta_fac' for snapshot/restart compatibility.)
# DEGREE 1 (P1), NOT P2: viscosity-bearing factors must stay positive. P1 is a
# convex (barycentric) interpolant, so sampling the gaussian in [0,1] at the
# nodes keeps f in [0,1] everywhere → eta_weak = eta_FK·(1-f)+floor·f is a
# guaranteed-positive blend. P2's negative basis lobes overshoot in under-
# resolved fault cells and can drive the weakening factor (hence viscosity)
# out of range. Low order is the principled choice for material properties.
gfac = uw.discretisation.MeshVariable(
    "eta_fac", mesh, vtype=uw.VarType.SCALAR, degree=1,
    continuous=True, varsymbol=r"f_F")
# Companion UNSIGNED, edge-clamped DISTANCE field for the ADAPT METRIC. The
# Surface's own fault.distance is a *signed* P1 field whose zero-contour follows
# the fault's INFINITE line, so interpolating it lights up the metric along the
# line EXTENSION (beyond the tip and radially inward) — the mover then refines
# the extension, MISSING the actual finite fault (user-observed). Computing the
# unsigned geometric distance directly at the nodes (like gfac) is edge-aware:
# beyond the segment it is the radial distance to the endpoint (grows), so no
# spurious near-zero off the fault. The metric gaussian uses THIS field.
dfac = uw.discretisation.MeshVariable(
    "dist_fac", mesh, vtype=uw.VarType.SCALAR, degree=1,
    continuous=True, varsymbol=r"d_F")

if resume_label:
    T.read_timestep(resume_label, "T_v2p1", 0, outputPath=OUT_DIR)
    V.read_timestep(resume_label, "V_v2p1", 0, outputPath=OUT_DIR)
    try:
        P.read_timestep(resume_label, "P_v2p1", 0, outputPath=OUT_DIR)
    except Exception:
        P.data[...] = 0.0
elif args.from_perturbation:
    r_inner, r_o = 0.5, 1.0
    X = mesh.CoordinateSystem.X
    r_sym = sympy.sqrt(X[0]**2 + X[1]**2)
    th_sym = sympy.atan2(X[1], X[0])
    T_cond = sympy.log(r_sym/r_o) / sympy.log(r_inner/r_o)
    # Phase shift (radians): sin(mode*(theta - phase)) lets us place a
    # downwelling (negative perturbation = cold = sinking) at a chosen azimuth,
    # e.g. under the fault. A negative perturbation sits where mode*(theta-phase)
    # = -pi/2 (mod 2pi).
    _pert_phase = np.deg2rad(float(args.pert_phase_deg))
    init_T = (float(args.pert_amplitude)
              * sympy.sin(float(args.pert_mode) * (th_sym - _pert_phase))
              * sympy.sin(np.pi * (r_sym - r_inner) / (r_o - r_inner))
              + T_cond)
    T.data[...] = np.asarray(uw.function.evaluate(
        init_T, T.coords)).reshape(-1, 1)
    V.data[...] = 0.0
    P.data[...] = 0.0
else:
    T.read_timestep(args.src_stem, "T_v2p1", 0, outputPath=args.src_dir)
    V.read_timestep(args.src_stem, "V_v2p1", 0, outputPath=args.src_dir)
    P.read_timestep(args.src_stem, "P_v2p1", 0, outputPath=args.src_dir)
print(f"  loaded T=[{T.data.min():.3f},{T.data.max():.3f}]  "
      f"|v|max={float(np.sqrt(V.data[:,0]**2+V.data[:,1]**2).max()):.2e}")


X = mesh.CoordinateSystem.X
r_sym = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
unit_r = mesh.CoordinateSystem.unit_e_0


# ============================================================
#  Dipping fault: embedded Surface + anisotropic weak zone
# ============================================================
delta = np.deg2rad(args.fault_dip_deg)
theta0 = np.deg2rad(args.fault_theta_deg)
# Surface intersection point and local frame at the trace.
P0 = np.array([np.cos(theta0), np.sin(theta0)])          # on r=1.0
e_hat = np.array([np.cos(theta0), np.sin(theta0)])        # local radial (out)
t_hat = np.array([-np.sin(theta0), np.cos(theta0)])       # local horizontal
side = 1.0 if args.fault_dip_dir == 'east' else -1.0
# Down-dip unit vector: dip below the local horizontal, into the medium.
dhat = side * np.cos(delta) * t_hat - np.sin(delta) * e_hat
L = args.fault_depth / np.sin(delta)                      # tip at radial depth
s = np.linspace(0.0, L, 25)[:, None]
xy = P0[None, :] + s * dhat[None, :]
fault_pts = np.column_stack([xy, np.zeros(len(xy))])      # (N,3) model coords

fault = uw.meshing.Surface("fault", mesh, fault_pts, symbol="F")
fault.discretize()

# Fault INFLUENCE expression (gaussian): 1 on the fault, 0 far away.
# Sampled onto gfac (NOT fed as exp() into the C-tensor — UW3 footgun rule).
fault_influence_expr = fault.influence_function(
    width=args.fault_width, value_near=1.0, value_far=0.0,
    profile="gaussian")


def _refresh_fault_influence():
    """Sample the gaussian fault influence onto gfac (1 on fault → 0 away) on
    the CURRENT node positions (the mover does not notify surfaces).

    CRITICAL: compute the distance DIRECTLY (geometry_tools) at the gfac nodes,
    NOT via fault.influence_function / fault.distance.sym. The Surface stores a
    *signed* distance as a P1 FE field; the sign FLIPS across the fault, so
    linear interpolation of that field along the fault's infinite-line
    EXTENSION beyond the tip crosses through near-zero |distance| — a spurious
    weak-zone streak past the fault end (and worse at P2, whose mid-edge nodes
    miss the P1 distance nodes). Evaluating the geometric distance at each gfac
    node directly is exact and degree-independent; the gaussian squares the
    distance so the sign is irrelevant. gfac stays in [0, 1]."""
    from underworld3.utilities.geometry_tools import (
        signed_distance_pointcloud_polyline_2d)
    coords = np.asarray(gfac.coords)[:, :2]
    d = signed_distance_pointcloud_polyline_2d(coords, xy[:, :2])
    gfac.data[:, 0] = np.exp(-(d / args.fault_width) ** 2)
    # UNSIGNED edge-clamped distance for the adapt metric (gfac/dfac share P1
    # nodes). |d| is the edge-aware distance to the finite segment, so the metric
    # gaussian built from it does NOT bleed along the fault's line extension.
    dfac.data[:, 0] = np.abs(d)
    # Keep the Surface's own (signed) distance field fresh (other consumers).
    fault._distance_stale = True
    _ = fault.distance


_refresh_fault_influence()

# Director = unit normal to the (weak) fault plane. The straight fault has
# a constant in-plane normal perpendicular to dhat; sign is irrelevant to
# the Mühlhaus–Moresi tensor (it enters as even products of n).
n_vec = np.array([-dhat[1], dhat[0]])
n_vec = n_vec / np.linalg.norm(n_vec)
director = sympy.Matrix([float(n_vec[0]), float(n_vec[1])])
# Outer product n nᵀ (constant for the straight fault) for the ANISOTROPIC
# mmpde metric tensor: refining ONLY the fault-normal direction aligns thin
# cell rows onto the centre-line (a scalar bump refines an isotropic corridor
# and leaves the centre coarse — see _winslow_mmpde docstring).
_nnT = director * director.T

# --- Analytic fault refinement band (mesh-coordinate expression) ----------
# The OT reset mover deforms the mesh to its uniform reference canvas before
# building the metric, so a cached Surface-distance proxy would be stale on
# that canvas. A straight dipping segment has a CLOSED-FORM signed distance,
# so express the band directly in mesh.X — it then evaluates correctly at the
# reference positions and survives the reset. No sympy.Abs (the mover
# differentiates the metric; Abs emits an unsupported `re` in the JIT).
_Xc = mesh.CoordinateSystem.X
_perp = ((_Xc[0] - float(P0[0])) * float(n_vec[0])
         + (_Xc[1] - float(P0[1])) * float(n_vec[1]))      # ⟂ signed distance
_salong = ((_Xc[0] - float(P0[0])) * float(dhat[0])
           + (_Xc[1] - float(P0[1])) * float(dhat[1]))      # along-dip coord
_ws = refine_w                                              # along-dip taper
# Smooth flat-top window ≈1 on the segment s∈[0,L], →0 beyond either end, so
# the perpendicular gaussian refines the FAULT, not the whole infinite chord.
_window = (sympy.Rational(1, 4)
           * (1 + sympy.tanh(_salong / _ws))
           * (1 - sympy.tanh((_salong - float(L)) / _ws)))
fault_band = 1.0 + args.fault_refine_amp * sympy.exp(
    -(_perp / refine_w) ** 2) * _window

# Frank-Kamenetskii bulk viscosity (rides the live T): eta_FK in [1, deltaEta]
# (1 at the hot interior, deltaEta in the cold lid). The fault weakens the
# (cold, stiff) lid DOWN TO the interior viscosity FLOOR — not below — which
# is physical (a fault zone is weak rock ≈ asthenosphere viscosity, not near-
# zero) and avoids a near-zero conditioning mode. finf = gfac.sym[0] is the
# fault influence: 1 on the fault, 0 away.
#   eta_weak = eta_FK*(1-finf) + floor*finf   →  floor on the fault, eta_FK away.
NO_FAULT = bool(args.no_fault)
FAULT_PASSIVE = bool(args.fault_passive)
# MECHANICS off for both --no-fault (no fault at all) and --fault-passive (fault
# still REFINES the mesh but is mechanically inert): finf=0 → eta_weak=eta_FK,
# isotropic FK. The fault METRIC stays unless NO_FAULT (see the metric block).
_mech_off = NO_FAULT or FAULT_PASSIVE
_rheology = "isotropic" if _mech_off else args.rheology
eta_FK = sympy.exp(theta_FK * (1 - T.sym[0]))
finf = 0 if _mech_off else gfac.sym[0]
eta_weak = eta_FK * (1.0 - finf) + float(args.fault_floor) * finf

if args.freeslip == "constrained":
    # In-saddle Lagrange-multiplier free slip: a scalar multiplier h is coupled
    # into the saddle-point system so u·n=0 on Upper is enforced exactly (no
    # penalty stiffness, exact TI Jacobian, recoverable dynamic topography).
    # The clean PARALLEL recipe — runs at np>1 (PR #240, or the
    # UW_CONSTRAINED_ALLOW_PARALLEL escape hatch on pre-#240 builds).
    stokes = uw.systems.Stokes_Constrained(mesh, velocityField=V, pressureField=P)
else:
    stokes = uw.systems.Stokes(mesh, velocityField=V, pressureField=P)
if _rheology == "isotropic":
    # ISOTROPIC weak fault (or --no-fault: eta_weak == eta_FK, pure FK).
    # Linear, exact Newton — far more tractable than TI.
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_weak
else:
    # ANISOTROPIC (transverse-isotropic) weak fault: weak only to
    # fault-parallel shear (η_1 floored at `floor`), isotropic away.
    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_FK
    stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_weak
    stokes.constitutive_model.Parameters.director = director
stokes.tolerance = 1.0e-5
stokes.penalty = 0.0
if _rheology == "isotropic" and args.freeslip != "constrained":
    # Isotropic Stokes is LINEAR (viscosity independent of V) → ksponly: one
    # exact linear KSP solve, no Newton/line-search. The default newtonls
    # line search REJECTS steps on vigorous high-Ra flow
    # (DIVERGED_LINE_SEARCH) and wastes iterations; ksponly is the correct,
    # clean, fast solver for the linear problem.
    # (Excluded for --freeslip constrained: the grouped [p,h] Schur solve
    # gives the WRONG answer under ksponly — needs the default outer iterate.)
    stokes.petsc_options["snes_type"] = "ksponly"
# NOTE (TI only): keep the default newtonls for --rheology ti. Its flux is
# linear too, but the inexact GAMG inner solve makes Newton iterate; do NOT
# blindly copy ksponly to the TI path without re-checking convergence.
stokes.add_essential_bc((0.0, 0.0), mesh.boundaries.Lower.name)  # no-slip inner
if args.freeslip == "nitsche":
    # u·n̂ = 0 on the outer boundary (direction defaults to the surface normal).
    stokes.add_nitsche_bc(mesh.boundaries.Upper.name, gamma=args.nitsche_gamma)
elif args.freeslip == "constrained":
    # In-saddle multiplier free slip: u·n=0 on Upper, n = outward radial.
    # No penalty stiffness — h carries the exact constraint and, at
    # convergence, the boundary normal traction (dynamic topography).
    h_topo = stokes.add_constraint_bc(
        mesh.boundaries.Upper.name, g=0.0, normal=unit_r)
    # Enclosed cavity (no-slip inner + normal-constrained outer) → the
    # pressure has a constant nullspace; the grouped [p,h] Schur needs it.
    stokes.petsc_use_pressure_nullspace = True
else:
    fs = (args.kfs * V.sym.dot(unit_r) * unit_r)
    stokes.add_natural_bc(fs, mesh.boundaries.Upper.name)
T_cond = sympy.log(r_sym / 1.0) / sympy.log(0.5 / 1.0)
stokes.bodyforce = Ra * (T.sym[0] - T_cond) * unit_r

# Geometric FMG on the velocity block (auto-detects the dm_hierarchy).
if args.fmg_levels > 0:
    if not hasattr(stokes, "preconditioner"):
        sys.exit("FMG requested but stokes.preconditioner is unavailable — "
                 "this needs the development branch (PR #230/#231).")
    if args.freeslip == "constrained":
        # FMG on the velocity block of the 3-field grouped [p,h] Schur is
        # UNVALIDATED — the multiplier field changes the fieldsplit. Allow it
        # (opt-in) but flag so a divergence here is attributed correctly.
        uw.pprint("  [warn] FMG + --freeslip constrained is unvalidated "
                  "(extra multiplier field in the fieldsplit); watch convergence.")
    stokes.preconditioner = "auto"

adv = uw.systems.AdvDiffusionSLCN(
    mesh, u_Field=T, V_fn=V.sym, verbose=False,
    theta=1.0, monotone_mode='clamp')
if args.old_frame:
    # Exact-carry SL reach-back (free-surface fix): sample psi_star in the OLD
    # (pre-adapt) geometry so the mmpde mesh motion doesn't remap-diffuse T.
    # DuDt only (NOT the diffusive DFDt). V_fn must stay PHYSICAL (it is).
    adv.DuDt.old_frame_traceback = True
    print("  old-frame SL reach-back: ON (exact carry through mesh motion)", flush=True)
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0
adv.tolerance = 1.0e-4
adv.add_dirichlet_bc(1.0, mesh.boundaries.Lower.name)   # T=1 hot
adv.add_dirichlet_bc(0.0, mesh.boundaries.Upper.name)   # T=0 cold


# --- Nu evaluator: surface heat flux on the cold (Upper) boundary ---
Q_COND = 2.0 * np.pi / np.log(1.0 / 0.5)
_X = mesh.CoordinateSystem.X
_n = mesh.Gamma_N
_qn_outer = -(T.sym[0].diff(_X[0]) * _n[0]
              + T.sym[0].diff(_X[1]) * _n[1])
_bd_qn_upper = uw.maths.BdIntegral(
    mesh=mesh, fn=_qn_outer, boundary=mesh.boundaries.Upper.name)


def _nu_surface():
    return float(_bd_qn_upper.evaluate()) / Q_COND


_nu = _nu_surface


# --- Achieved-refinement diagnostic (thread 1) ------------------------------
# Measure how finely the mover actually resolves the fault and the thermal
# boundary layer UNDER LIVE CONVECTION, step by step. The right measure is the
# RATIO of nearest-neighbour spacing in a feature region to the bulk spacing
# (NOT misalignment — misalignment is a global scalar and the fault is a tiny
# fraction of nodes, so it stays ~0.9 even with the fault unrefined). A
# fault/bulk ratio of ~0.2 means ~5x finer at the fault; ~0.6-0.8 is the
# mmpde-from-uniform "creation cap" regime; ~1.0 means no fault refinement.
from scipy.spatial import cKDTree as _cKDTree
from underworld3.utilities.geometry_tools import (
    signed_distance_pointcloud_polyline_2d as _sd_poly)

_FAULT_XY = xy[:, :2]   # fault polyline (model coords), defined above


def _refine_quality():
    """Fault/bulk and thermal-BL/bulk NN-spacing ratios on the CURRENT mesh
    vertices. Returns a dict; ratios < 1 mean finer than bulk (more refined)."""
    C = np.asarray(mesh.X.coords)[:, :2]
    r = np.sqrt((C ** 2).sum(axis=1))
    dd, _ = _cKDTree(C).query(C, k=2)
    nn = dd[:, 1]                                    # nearest-neighbour spacing
    # Bulk reference = mid-radius interior, away from the fault (excludes both
    # the BL shell and the fault corridor) so it's a clean background spacing.
    if not NO_FAULT:
        dist = np.abs(_sd_poly(C, _FAULT_XY))
        near = dist < (1.5 * args.fault_width)
    else:
        dist = np.full(len(C), 1.0e9)
        near = np.zeros(len(C), dtype=bool)
    bulk_mask = (dist > 0.3) & (r > 0.55) & (r < 0.85)
    if not bulk_mask.any():
        bulk_mask = dist > 0.3
    bulk = float(np.median(nn[bulk_mask])) if bulk_mask.any() else float('nan')
    fault_med = float(np.median(nn[near])) if near.any() else float('nan')
    bl_mask = r > 0.90                               # outer thermal-BL shell
    bl_med = float(np.median(nn[bl_mask])) if bl_mask.any() else float('nan')
    fin = np.isfinite(bulk) and bulk > 0
    return dict(
        fault_ratio=(fault_med / bulk if (near.any() and fin) else float('nan')),
        bl_ratio=(bl_med / bulk if fin else float('nan')),
        fault_med=fault_med, bl_med=bl_med, bulk_med=bulk,
        n_fault=int(near.sum()))


def snapshot(step):
    label = "init" if step == 0 else f"step{step:04d}"
    mesh.write_timestep(filename=label, index=0, outputPath=OUT_DIR,
                        meshVars=[T, V, P, gfac], meshUpdates=True,
                        create_xdmf=True)


def _adapt_step():
    """Build the combined fault×|∇T| metric, move with surface slip; the
    mover owns field transfer. Refresh the fault distance + factor on the
    moved nodes and zero V,P for a cold flow restart. Returns
    (moved, misalignment)."""
    old_X = np.asarray(mesh.X.coords).copy()
    h0 = float(mesh._radii.mean())
    grad_L = (args.grad_smooth_h0 * h0 if args.grad_smooth_h0 > 0 else None)
    if args.skip_threshold >= 0:
        sk = (None if args.skip_threshold > 10.0 else args.skip_threshold)
    else:
        sk = STRAT["skip_threshold"]

    # |∇T| density (frozen Lagrangian) × fault-distance density.
    # The "effective R" lives HERE, in the metric — resolution_ratio is passed
    # as refinement= to metric_density_from_gradient (finest/coarsest cell-size
    # grading ratio), front-following. R=5 grades far harder than any named
    # strategy (extreme caps at 2). The MMPDE mover then equidistributes to
    # this density with its OWN CG iteration; R is NOT a mover clamp.
    _Rmet = float(args.resolution_ratio)
    if _Rmet > 0:
        rho_T = uw.meshing.metric_density_from_gradient(
            mesh, T, refinement=_Rmet, coarsening="auto",
            metric_choice="front-following", name="loop",
            gradient_smoothing_length=grad_L)
    else:
        rho_T = uw.meshing.metric_density_from_gradient(
            mesh, T, strategy=args.strategy, name="loop",
            gradient_smoothing_length=grad_L)
    if NO_FAULT:
        # Faultless control: pure |∇T| refinement, no fault density/tensor.
        rho = rho_T
        rho_tensor = None
    else:
        # Distance for the metric gaussians = the DIRECT unsigned distance field
        # dfac (computed geometrically at the nodes), NOT fault.distance.sym (the
        # interpolated SIGNED field, which bleeds the gaussian along the fault's
        # line extension → the mover refines the extension and MISSES the fault).
        # TWO widths: a WIDE isotropic DRAW gaussian (draw_w) that pulls distant
        # nodes toward the fault — giving it enough weight in the mmpde GLOBAL
        # functional to be served (a thin sharp band is invisible to the
        # variational mover) — and a SHARP across-fault gaussian (refine_w) that
        # localises the drawn nodes onto the centre-line.
        # _width_scale (module global, default 1.0) anneals the fault widths for
        # the two-stage pre-refine: start WIDE (mover-resolvable on the coarse
        # mesh → pulls nodes into the neighbourhood, beats the ridge-crossing
        # barrier) and SHARPEN toward 1.0 as nodes accumulate.
        _ws = _width_scale
        _gauss_sharp = sympy.exp(-(dfac.sym[0] / (refine_w * _ws)) ** 2)
        _gauss_wide = sympy.exp(-(dfac.sym[0] / (draw_w * _ws)) ** 2)
        # SCALAR fault density: WIDE draw footprint (ot-reset / non-mmpde / the
        # diagnostic mismatch + the tensor isotropic part).
        fault_rho = 1.0 + args.fault_refine_amp * _gauss_wide
        if args.metric_combine == "max":
            rho = sympy.Max(rho_T, fault_rho)
        else:
            rho = rho_T * fault_rho

        # ANISOTROPIC TENSOR fault metric for the mmpde mover:
        #   M = rho·I + (Rf²-1)·exp(-(d/refine_w)²)·n nᵀ
        # Isotropic part `rho` carries the WIDE draw (pulls nodes in); the n nᵀ
        # term uses the SHARP gaussian to thin them ACROSS the centre-line.
        _Rf = float(args.fault_anisotropy)
        rho_tensor = (rho * sympy.eye(2)
                      + (_Rf ** 2 - 1.0) * _gauss_sharp * _nnT) if _Rf > 0 else None

    # Diagnostic: misalignment BEFORE adapting (logged whether or not we move).
    mm = uw.meshing.mesh_metric_mismatch(
        mesh, rho, resolution_ratio=args.resolution_ratio)
    misalign = float(mm["misalignment"])
    print(f"  mismatch before adapt: misalignment={misalign:.3f} "
          f"(skip threshold {sk})", flush=True)

    # --- ot-reset (DEFAULT): the production reset-based OT mover -----------
    # mesh.OT_adapt re-meshes FRESH to the current |∇T| each cycle (cannot
    # lag), is sliver-free, slips the radial rings, and OWNS field transfer
    # (remaps T + SLCN history, zeros V,P via fields_to_zero). The fault rides
    # along as the analytic extra_density band (valid on the reset canvas).
    if args.method == "ot-reset":
        moved = mesh.OT_adapt(
            T,
            refinement=args.resolution_ratio,
            coarsening="auto",
            metric_choice="front-following",
            grad_smoothing_length=(grad_L if grad_L else "auto"),
            extra_density=fault_band,
            fields_to_zero=[V, P],
            skip_threshold=sk,
            verbose=True)
        if not moved:
            return False, misalign
        # OT_adapt remapped T and zeroed V,P; refresh the (custom) fault
        # viscosity factor on the moved nodes.
        _refresh_fault_influence()
        return True, misalign

    # --- MMPDE (DEFAULT) and other in-place movers ------------------------
    # CRITICAL: each mover takes its OWN method_kwargs.
    #  * mmpde (the variational, non-folding, cluster+align mover — the one we
    #    use): runs its OWN nonlinear-CG iteration to outer_tol (n_outer~150
    #    internally). Pass step_frac/accel/momentum; do NOT inject relax/n_outer
    #    (those are the anisotropic mover's knobs and STARVE the mmpde solve).
    #  * anisotropic: the clean config is relax=1.0, n_outer=1 (NOT 0.2/12 —
    #    that over-iterates and slivers; see project_winslow_mover_iteration_bug).
    # R (resolution_ratio) is already baked into the metric grading above; it is
    # NOT a mover argument for mmpde. do NOT pass strategy= (it setdefaults
    # resolution_ratio into method_kwargs, which mmpde rejects).
    if args.method in ("mmpde", "variational"):
        _accel = None if args.mover_accel in ("none", "off", "") else args.mover_accel
        mk = dict(step_frac=0.2, accel=_accel, momentum=0.0)
        # mmpde consumes the ANISOTROPIC TENSOR metric (small across the fault);
        # fall back to the isotropic scalar only if --fault-anisotropy 0.
        metric = rho_tensor if rho_tensor is not None else rho
    elif args.method in ("anisotropic", "aniso", "tensor"):
        mk = dict(relax=1.0, n_outer=1, resolution_ratio=args.resolution_ratio)
        metric = rho
    else:  # raw 'ot' and others: library defaults
        mk = dict()
        metric = rho
    # The mover's internal skip-check calls mesh_metric_mismatch(metric); a TENSOR
    # metric breaks it (per-cell scalar areas vs d×d entries). Do the skip decision
    # HERE using the scalar-rho misalignment already computed, then pass
    # skip_threshold=None so the mover doesn't re-check with the tensor.
    _tensor_metric = isinstance(metric, sympy.MatrixBase)
    if _tensor_metric and sk is not None and misalign < float(sk):
        return False, misalign
    uw.meshing.smooth_mesh_interior(
        mesh, metric=metric, method=args.method,
        skip_threshold=(None if _tensor_metric else sk),
        slip_surfaces=args.slip_surfaces,
        method_kwargs=mk, verbose=True)
    new_X = np.asarray(mesh.X.coords).copy()
    if np.allclose(new_X, old_X):
        return False, misalign

    # Mover moved the nodes (T + SLCN history already transferred). Recompute
    # the fault distance/influence on the new positions, then cold-restart flow.
    _refresh_fault_influence()
    V.data[...] = 0.0
    P.data[...] = 0.0
    return True, misalign


# Pre-convection fault refinement: with T still conductive there is NO thermal
# boundary layer competing for nodes, so the mmpde mover can concentrate on the
# fault unopposed. Run N fault-focused adapt passes BEFORE the time loop; the
# combined metric is fault-dominated here (conductive |∇T| is smooth). Once
# convection starts the ongoing adapts maintain this refinement while tracking
# the thermal field (and would let nodes follow a moving fault via the metric).
if args.fault_prerefine > 0 and not NO_FAULT and args.strategy != "off":
    _np = int(args.fault_prerefine)
    _w0 = float(args.fault_prerefine_width0)
    print(f"  pre-refining mesh to the fault ({_np} passes, width anneal "
          f"{_w0:g}->1.0, T conductive — no TBL competition)...", flush=True)
    for _i in range(_np):
        # Geometric anneal of the width scale w0 -> 1.0 over the passes. Wide
        # early passes are mover-resolvable (pull nodes in past the ridge-
        # crossing barrier); later passes sharpen onto the centre-line.
        _width_scale = (_w0 ** (1.0 - _i / max(_np - 1, 1))) if _np > 1 else 1.0
        _moved, _mis = _adapt_step()
        print(f"    prerefine pass {_i+1}: w_scale={_width_scale:.2f} "
              f"moved={_moved} misalign={_mis:.3f}", flush=True)
    _width_scale = 1.0
    _refresh_fault_influence()

# Initial Stokes solve
print("  initial Stokes solve...", flush=True)
t0 = time.time()
stokes.solve(zero_init_guess=False)
print(f"  init done {time.time()-t0:.1f}s "
      f"|v|max={float(np.sqrt(V.data[:,0]**2+V.data[:,1]**2).max()):.2e}",
      flush=True)


hist = []
rq_hist = []          # achieved-refinement diagnostic (thread 1)
t_sim = 0.0
if resume_label:
    hpath = os.path.join(OUT_DIR, "history.npz")
    if os.path.exists(hpath):
        z = np.load(hpath)
        for i in range(len(z['step'])):
            if int(z['step'][i]) > resume_step:
                continue
            _mis = (float(z['misalignment'][i])
                    if 'misalignment' in z.files else float('nan'))
            hist.append((int(z['step'][i]), float(z['t'][i]),
                         float(z['dt'][i]), float(z['wall'][i]),
                         float(z['vrms'][i]), float(z['Nu'][i]),
                         float(z['Tmin'][i]), float(z['Tmax'][i]),
                         int(z['adapted'][i]), _mis))
        if hist:
            t_sim = hist[-1][1]
            print(f"  resumed history: {len(hist)} entries, t={t_sim:.5f}")
else:
    snapshot(0)

START_STEP = resume_step + 1 if resume_label else 1
END_STEP = (resume_step if resume_label else 0) + args.n_steps + 1

print(f"  running steps {START_STEP}..{END_STEP - 1} "
      f"(snapshot every {args.snapshot_every}, log every {args.log_every})")
print(f"{'step':>5} {'t':>9} {'dt':>10} {'wall':>7} "
      f"{'vrms':>10} {'Nu':>8} {'T[min,max]':>22} {'refine(f/b bl/b)':>18} {'adapt'}")

n_adapt_skipped = 0
n_adapt_done = 0
for s in range(START_STEP, END_STEP):
    t_step_0 = time.time()
    did_adapt = False
    misalign = float('nan')
    if (args.strategy != "off" and s >= args.adapt_start
            and (s % args.adapt_every == 0)):
        did_adapt, misalign = _adapt_step()
        if did_adapt:
            n_adapt_done += 1
        else:
            n_adapt_skipped += 1
    # Stokes BEFORE AdvDiff so the post-adapt AdvDiff step sees a freshly
    # solved V from the just-remapped T (avoids a one-step diffusion smear).
    try:
        stokes.solve(zero_init_guess=(did_adapt or args.cold_stokes))
        # MEDIAN element-size CFL (percentile=50): the tiny fault/BL cells do NOT
        # control the timestep (they're outliers above the median), matching the
        # "fixed or mean-size, not fault-controlled" requirement.
        dt = adv.estimate_dt(direction_aware=True,
                             percentile=50.0) * float(args.dt_mult)
        adv.solve(timestep=dt, zero_init_guess=False)
    except Exception as e:
        print(f"  EXCEPTION at step {s}: {e}", flush=True)
        break
    t_sim += dt
    wall = time.time() - t_step_0

    T_arr = T.data[:, 0]
    if np.isnan(T_arr).any() or np.isinf(T_arr).any():
        print(f"  step {s}: NaN/Inf in T — ABORT", flush=True)
        break
    Tmin, Tmax = float(T_arr.min()), float(T_arr.max())
    if Tmax > 1.1 or Tmin < -0.1:
        print(f"  step {s}: T overshoot [{Tmin:+.4f},{Tmax:+.4f}] — ABORT",
              flush=True)
        break

    v_sq = np.asarray(uw.function.evaluate(V.sym.dot(V.sym), mesh.X.coords))
    vrms = float(np.sqrt(np.mean(v_sq)))
    Nu_val = _nu()

    # Achieved-refinement ratios on the current (possibly just-adapted) mesh.
    rq = _refine_quality()
    rq_hist.append((s, rq['fault_ratio'], rq['bl_ratio'],
                    rq['fault_med'], rq['bl_med'], rq['bulk_med'],
                    rq['n_fault'], int(did_adapt)))
    _rq = np.asarray(rq_hist)
    np.savez(os.path.join(OUT_DIR, "refine_quality.npz"),
             step=_rq[:, 0], fault_ratio=_rq[:, 1], bl_ratio=_rq[:, 2],
             fault_med=_rq[:, 3], bl_med=_rq[:, 4], bulk_med=_rq[:, 5],
             n_fault=_rq[:, 6], adapted=_rq[:, 7])

    hist.append((s, t_sim, dt, wall, vrms, Nu_val,
                 Tmin, Tmax, int(did_adapt), misalign))
    _h = np.asarray(hist)
    np.savez(os.path.join(OUT_DIR, "history.npz"),
             step=_h[:, 0], t=_h[:, 1], dt=_h[:, 2], wall=_h[:, 3],
             vrms=_h[:, 4], Nu=_h[:, 5], Tmin=_h[:, 6], Tmax=_h[:, 7],
             adapted=_h[:, 8], misalignment=_h[:, 9])
    if s % args.snapshot_every == 0:
        snapshot(s)
    if s % args.log_every == 0:
        _fr = rq['fault_ratio']
        _frs = f"{_fr:.2f}" if np.isfinite(_fr) else "  - "
        print(f"{s:>5d} {t_sim:>9.5f} {dt:>10.3e} {wall:>6.2f}s "
              f"{vrms:>10.3e} {Nu_val:>+8.3f} [{Tmin:+.3f},{Tmax:+.3f}]  "
              f"f/b={_frs} bl/b={rq['bl_ratio']:.2f} "
              f"{'ADAPT' if did_adapt else ''}", flush=True)
    if args.max_t > 0 and t_sim >= args.max_t:
        print(f"  reached max_t={args.max_t} at step {s} "
              f"(t_sim={t_sim:.5f}) — STOPPING", flush=True)
        if s % args.snapshot_every != 0:
            snapshot(s)
        break

print(f"=== done; adapts done={n_adapt_done}, skipped={n_adapt_skipped} ===",
      flush=True)
if hist:
    snapshot(int(hist[-1][0]))
