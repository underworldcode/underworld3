"""Benchmark: Transverse-isotropic VEP fault under harmonic shear.

Sister to ``bench_ve_harmonic.py`` (isotropic Maxwell): same peak-start
initial condition and cosine forcing, but with an embedded fault
modelled by ``TransverseIsotropicVEPFlowModel``.  The point of the
benchmark is to confirm that BDF-1 / BDF-2 time integration are as
robust on the angled-fault problem as on the isotropic case — no new
SNES instabilities expected.

Three fault angles run side-by-side: θ ∈ {0°, +15°, -15°}.

Probes:
* ``sigma_xy``      — global shear stress at the fault centre (fault frame)
* ``tau_resolved``  — shear on the fault plane: t·σ·n  with t the fault
                      tangent and n the fault normal.

For θ = 0° the resolved shear equals σ_xy.  For θ ≠ 0° the resolved
shear caps at τ_y while σ_xy keeps growing, since only the fault-plane
component yields.

Forcing: V_top(t) = V0·cos(ωt + φ) with the same Deborah-number / phase
as the isotropic case so the analytical (sub-yield) reference is
identical: the resolved shear should track A_∞·cos(ωt) once any
plastic transients die out.

Output: one ``.npz`` per (angle, τ_y) pair, BDF-1 and BDF-2 traces.
"""

import os
import time
import numpy as np
import sympy

import underworld3 as uw
from underworld3.function import expression
from _bench_helpers import OUTPUT_DIR


# ---------------------------------------------------------------------------
# Run-specific parameters (kept aligned with bench_ve_harmonic.py)
# ---------------------------------------------------------------------------

V0 = 0.5
OMEGA = np.pi / 2.0          # period 4·t_r
DT = 0.05
N_PERIODS = 4
T_END = N_PERIODS * 2.0 * np.pi / OMEGA

ETA_0 = 1.0                  # bulk shear viscosity
ETA_1 = 1.0                  # fault-plane shear viscosity
MU = 1.0                     # elastic shear modulus
TAU_Y_BULK = 200.0           # effectively infinite away from the fault

# Geometry
RES = 16                     # mesh resolution (RES x RES) — kept modest for benchmark turnaround
H = 1.0; W = 1.0             # domain size [0, W] × [0, H]
FAULT_LENGTH = 0.6
FAULT_WIDTH = 0.06           # influence-function half-width

# Sweep
ANGLES_DEG = (0.0, 15.0, -15.0)
TAU_Y_LIST = (0.15, 0.30)
BDF_ORDERS = (1, 2)


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------

def build_ti_stokes(label, theta_deg, tau_y, bdf_order):
    """Construct a TI-VEP Stokes problem with an embedded fault.

    Parameters
    ----------
    label : str
        Used to namespace mesh-variable names.
    theta_deg : float
        Fault angle from horizontal, in degrees.
    tau_y : float
        Fault-plane yield stress.
    bdf_order : int
        BDF time-integration order (1 or 2).

    Returns
    -------
    mesh, stokes, V_top_expr, n_vec
    """
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(RES, RES),
        minCoords=(0.0, 0.0), maxCoords=(W, H),
        qdegree=3,
    )
    v = uw.discretisation.MeshVariable(
        f"U_{label}", mesh, 2, degree=2, vtype=uw.VarType.VECTOR,
    )
    p = uw.discretisation.MeshVariable(
        f"P_{label}", mesh, 1, degree=1,
        continuous=True, vtype=uw.VarType.SCALAR,
    )

    theta = np.radians(theta_deg)
    cx, cy = 0.5 * W, 0.5 * H
    dx = 0.5 * FAULT_LENGTH * np.cos(theta)
    dy = 0.5 * FAULT_LENGTH * np.sin(theta)
    fault = uw.meshing.Surface(
        f"fault_{label}", mesh,
        np.array([[cx - dx, cy - dy], [cx + dx, cy + dy]]),
        symbol=f"F{label}",
    )
    fault.discretize()

    n_x = -np.sin(theta)
    n_y =  np.cos(theta)
    director = sympy.Matrix([n_x, n_y])

    weakness = fault.influence_function(
        width=FAULT_WIDTH,
        value_near=1.0 / tau_y,
        value_far=1.0 / TAU_Y_BULK,
        profile="gaussian",
    )
    tau_y_field = 1.0 / weakness

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = uw.constitutive_models.TransverseIsotropicVEPFlowModel(
        stokes.Unknowns, order=bdf_order,
    )
    stokes.constitutive_model = cm
    cm.Parameters.shear_viscosity_0 = ETA_0
    cm.Parameters.shear_viscosity_1 = ETA_1
    cm.Parameters.shear_modulus = MU
    cm.Parameters.yield_stress = tau_y_field
    cm.Parameters.director = director
    cm.Parameters.shear_viscosity_min = ETA_0 * 1.0e-3
    cm.Parameters.strainrate_inv_II_min = 1.0e-6
    cm.yield_mode = "softmin"  # default; smooth and robust

    stokes.saddle_preconditioner = 1.0 / cm.K
    stokes.tolerance = 1.0e-4
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["snes_force_iteration"] = True

    V_top = expression(
        rf"V_{{top,{label}}}", sympy.Float(0.0), "Top BC velocity",
    )
    stokes.add_essential_bc(sympy.Matrix([V_top, 0.0]), "Top")
    stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")
    stokes.bodyforce = sympy.Matrix([0.0, 0.0])

    return mesh, stokes, V_top, np.array([n_x, n_y])


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def probe_stress(stokes, n_vec, c=np.array([[0.5, 0.5]])):
    """Return σ_xy and resolved fault-plane shear at the fault centre."""
    tau = stokes.tau
    dists = np.linalg.norm(tau.coords - c, axis=1)
    idx = int(np.argmin(dists))
    s_xx, s_yy, s_xy = tau.data[idx, 0], tau.data[idx, 1], tau.data[idx, 2]
    n_x, n_y = n_vec
    t_x, t_y = n_y, -n_x  # fault tangent (perp to normal)
    resolved = (s_xx * t_x * n_x + s_xy * (t_x * n_y + t_y * n_x)
                + s_yy * t_y * n_y)
    return float(s_xy), float(resolved)


# ---------------------------------------------------------------------------
# Time-stepping core
# ---------------------------------------------------------------------------

def _run_one(theta_deg, tau_y, bdf_order, label):
    """One run.  Returns dict of arrays."""
    mesh, stokes, V_top, n_vec = build_ti_stokes(
        label, theta_deg, tau_y, bdf_order,
    )

    # Maxwell relaxation time and steady-state amplitude (sub-yield)
    t_r = ETA_1 / MU
    De = OMEGA * t_r
    # BCs: Top moves at V_top, Bottom fixed → engineering shear rate
    # γ̇_0 = V0/H (NOT 2·V0/H — that would be the antisymmetric case
    # used by bench_ve_harmonic.py).  Steady VE amplitude is then
    # σ_∞ = 2η·ε̇/sqrt(1+De²) = η·γ̇_0/sqrt(1+De²) since ε̇ = γ̇/2.
    gamma_dot_0 = V0 / H
    A_inf = ETA_1 * gamma_dot_0 / np.sqrt(1.0 + De**2)
    phi = float(np.arctan(De))

    # Peak-start: plant ψ*[k] = (resolved shear at t=-k·dt) on the fault
    # tangent direction in the SYM_TENSOR slot.  For a 2D tensor, with
    # the resolved shear along (t_x, t_y) and normal (n_x, n_y), the
    # corresponding stress contribution is τ·(t_i n_j + n_i t_j).
    n_x, n_y = n_vec
    t_x, t_y = n_y, -n_x
    n_nodes = stokes.DFDt.psi_star[0].array.shape[0]
    history = []
    for k in range(stokes.DFDt.order):
        val_k = A_inf * float(np.cos(OMEGA * k * DT))
        # symmetric tensor: σ = τ_resolved * (t⊗n + n⊗t)
        arr = np.zeros((n_nodes, 2, 2))
        sxx = val_k * 2.0 * t_x * n_x
        syy = val_k * 2.0 * t_y * n_y
        sxy = val_k * (t_x * n_y + t_y * n_x)
        arr[:, 0, 0] = sxx
        arr[:, 1, 1] = syy
        arr[:, 0, 1] = sxy
        arr[:, 1, 0] = sxy
        history.append(arr)
    stokes.DFDt.set_initial_history(history, dt=DT)

    times, sxy_h, tres_h, reasons, iters = [], [], [], [], []
    t_cur = 0.0
    t0 = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end_step = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end_step + phi))
        V_top.sym = sympy.Float(v_now)
        stokes.constitutive_model.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)
        sxy, tres = probe_stress(stokes, n_vec)
        t_cur = t_end_step
        times.append(t_cur); sxy_h.append(sxy); tres_h.append(tres)
        reasons.append(int(stokes.snes.getConvergedReason()))
        iters.append(int(stokes.snes.getIterationNumber()))
    wall = time.time() - t0

    times = np.array(times); sxy_h = np.array(sxy_h); tres_h = np.array(tres_h)
    reasons = np.array(reasons); iters = np.array(iters)

    # Sub-yield analytical: A_∞·cos(ωt).  Above yield, this is the VE
    # "no-yield" envelope and the actual response should track it until
    # |τ| reaches τ_y, then plateau.
    sigma_ve = A_inf * np.cos(OMEGA * times)

    return dict(
        times=times, sigma_xy=sxy_h, tau_resolved=tres_h,
        sigma_ve=sigma_ve, reasons=reasons, iters=iters,
        wall=wall, A_inf=A_inf, phi=phi, De=De, gamma_dot_0=gamma_dot_0,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []
    for theta_deg in ANGLES_DEG:
        for tau_y in TAU_Y_LIST:
            results = {}
            for bdf in BDF_ORDERS:
                lbl = f"tivep_o{bdf}_th{theta_deg:+.0f}_ty{tau_y:.2f}".replace(
                    ".", "p"
                )
                print(f"\n--- {lbl}: θ={theta_deg}°, τ_y={tau_y}, BDF-{bdf} ---",
                      flush=True)
                results[bdf] = _run_one(theta_deg, tau_y, bdf, lbl)
                r = results[bdf]
                ndiv = int((r["reasons"] < 0).sum())
                print(f"    wall={r['wall']:.1f}s  steps={len(r['times'])}  "
                      f"diverged={ndiv}  mean_its={float(r['iters'].mean()):.2f}  "
                      f"peak|τ_resolved|={float(np.abs(r['tau_resolved']).max()):.4f}  "
                      f"peak|σ_xy|={float(np.abs(r['sigma_xy']).max()):.4f}",
                      flush=True)
                summary.append(dict(
                    label=lbl, theta=theta_deg, tau_y=tau_y, bdf=bdf,
                    wall=r["wall"], diverged=ndiv,
                    mean_its=float(r["iters"].mean()),
                    peak_resolved=float(np.abs(r["tau_resolved"]).max()),
                    peak_sxy=float(np.abs(r["sigma_xy"]).max()),
                ))

            # Save BDF-1 and BDF-2 traces side by side
            tag = f"ti_vep_harmonic_th{theta_deg:+.0f}_ty{tau_y:.2f}".replace(
                ".", "p"
            )
            np.savez(
                os.path.join(OUTPUT_DIR, f"{tag}.npz"),
                theta_deg=theta_deg, tau_y=tau_y,
                times=results[1]["times"],
                sigma_xy_bdf1=results[1]["sigma_xy"],
                sigma_xy_bdf2=results[2]["sigma_xy"],
                tau_resolved_bdf1=results[1]["tau_resolved"],
                tau_resolved_bdf2=results[2]["tau_resolved"],
                sigma_ve=results[1]["sigma_ve"],
                reasons_bdf1=results[1]["reasons"],
                reasons_bdf2=results[2]["reasons"],
                iters_bdf1=results[1]["iters"],
                iters_bdf2=results[2]["iters"],
                A_inf=results[1]["A_inf"], De=results[1]["De"],
                gamma_dot_0=results[1]["gamma_dot_0"],
                wall_bdf1=results[1]["wall"], wall_bdf2=results[2]["wall"],
                V0=V0, OMEGA=OMEGA, DT=DT, T_END=T_END,
                ETA_0=ETA_0, ETA_1=ETA_1, MU=MU,
                FAULT_WIDTH=FAULT_WIDTH, FAULT_LENGTH=FAULT_LENGTH, RES=RES,
            )
            print(f"  saved → {tag}.npz", flush=True)

    print("\n=== summary ===", flush=True)
    print(f"{'label':<36} {'θ°':>4} {'τ_y':>5} {'BDF':>4} {'wall':>6} "
          f"{'div':>4} {'its':>5} {'peak|τ_res|':>11} {'peak|σ_xy|':>10}",
          flush=True)
    for s in summary:
        print(f"{s['label']:<36} {s['theta']:>4.0f} {s['tau_y']:>5.2f} "
              f"{s['bdf']:>4d} {s['wall']:>6.1f} {s['diverged']:>4d} "
              f"{s['mean_its']:>5.2f} {s['peak_resolved']:>11.4f} "
              f"{s['peak_sxy']:>10.4f}", flush=True)


if __name__ == "__main__":
    main()
