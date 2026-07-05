"""Phase B validator for MaxwellExponentialFlowModel — VE harmonic.

Mirrors ``docs/advanced/benchmarks/bench_ve_harmonic.py`` but assigns the
new ``MaxwellExponentialFlowModel`` (ETD-2) instead of the BDF-style VEP
model. Decision gate: max|err| must match or beat BDF-2's 1.34e-3 baseline.

The peak-start IC plants ``σ⁰ = A_∞·cos(0) = A_∞`` and the matching
``ε̇⁰ = γ̇₀/(2√(1+De²))`` so step 1 starts on the analytical steady cycle
with no homogeneous transient.

Run::

    pixi run -e amr-dev python -u docs/developer/design/_exp_integrator_phase_b_validate.py
"""

import time
import numpy as np
import sympy

import underworld3 as uw
from underworld3 import VarType
from underworld3.function import expression


# Parameters — same as bench_ve_harmonic
ETA = 1.0
MU = 1.0
V0 = 0.5
OMEGA = np.pi / 2.0
DT = 0.05
N_PERIODS = 4
T_END = N_PERIODS * 2.0 * np.pi / OMEGA
H = 1.0
W = 2.0


def run_exp():
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 8), minCoords=(-W / 2, -H / 2), maxCoords=(W / 2, H / 2)
    )
    v = uw.discretisation.MeshVariable("U_exp_b", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_exp_b", mesh, 1, degree=1)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.MaxwellExponentialFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = ETA
    cm.Parameters.shear_modulus = MU
    stokes.tolerance = 1e-7
    stokes.petsc_options["snes_force_iteration"] = True

    # Antisymmetric BCs (matches bench_ve_harmonic)
    V_top = expression(r"V_{top}^{exp}", sympy.Float(0.0), "Top BC for exp validator")
    stokes.add_essential_bc((V_top, 0.0), "Top")
    stokes.add_essential_bc((-V_top, 0.0), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")

    t_r = ETA / MU
    De = OMEGA * t_r
    gamma_dot_0 = 2.0 * V0 / H
    A_inf = ETA * gamma_dot_0 / np.sqrt(1.0 + De ** 2)
    phi_lag = float(np.arctan(De))

    DFDt = stokes.Unknowns.DFDt
    n_nodes = DFDt.psi_star[0].array.shape[0]

    # Plant σ_xy = A_inf at t=0 (peak-start)
    sigma0 = np.zeros((n_nodes, 2, 2))
    sigma0[:, 0, 1] = A_inf
    sigma0[:, 1, 0] = A_inf
    history = [sigma0]
    DFDt.set_initial_history(history, dt=DT)

    # Plant ε̇⁰ = γ̇₀/(2√(1+De²)) (i.e. shear-only) so step 1's history
    # term references the analytical ε̇ at t=0, not zero.
    edot0 = gamma_dot_0 / (2.0 * np.sqrt(1.0 + De ** 2))
    f0 = np.zeros((n_nodes, 2, 2))
    f0[:, 0, 1] = edot0
    f0[:, 1, 0] = edot0
    DFDt.forcing_star.array[...] = f0

    times, dts, sigmas, reasons = [], [], [], []
    t_cur = 0.0
    t0_wall = time.time()
    while t_cur < T_END - 1e-9:
        dt = min(DT, T_END - t_cur)
        t_end_step = t_cur + dt
        v_now = V0 * float(np.cos(OMEGA * t_end_step + phi_lag))
        V_top.sym = sympy.Float(v_now)
        cm.Parameters.dt_elastic = dt
        stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=2)

        coords = DFDt.psi_star[0].coords
        centre = np.array([[0.0, 0.0]])
        idx = int(np.argmin(np.linalg.norm(coords - centre, axis=1)))
        sigmas.append(float(DFDt.psi_star[0].array[idx, 0, 1]))
        reasons.append(int(stokes.snes.getConvergedReason()))
        times.append(t_end_step)
        dts.append(dt)
        t_cur = t_end_step

    times = np.array(times)
    dts = np.array(dts)
    sigmas = np.array(sigmas)
    reasons = np.array(reasons)
    sigma_ana = A_inf * np.cos(OMEGA * times)
    err = np.abs(sigmas - sigma_ana)
    max_err = float(err.max())
    rms = float(np.sqrt((err ** 2).mean()))
    wall = time.time() - t0_wall
    diverged = int((reasons < 0).sum())
    return dict(
        times=times,
        dts=dts,
        sigmas=sigmas,
        sigma_ana=sigma_ana,
        max_err=max_err,
        rms=rms,
        wall=wall,
        diverged=diverged,
        A_inf=A_inf,
        De=De,
    )


def main():
    print(f"[ve_harmonic_exp] dt={DT} T_end={T_END:.4f} (4 periods)", flush=True)
    res = run_exp()
    print(f"  steps={len(res['times'])} A_inf={res['A_inf']:.4f} De={res['De']:.4f}")
    print(f"  ETD-2  wall={res['wall']:.1f}s  max|err|={res['max_err']:.4e}  rms={res['rms']:.4e}")
    print(f"  diverged: {res['diverged']}/{len(res['times'])}")
    print(f"  baseline (bench_ve_harmonic BDF-2 peak-start): max|err| = 1.34e-3", flush=True)
    out = dict(
        times=res["times"], dts=res["dts"],
        sigma_exp=res["sigmas"], sigma_ana=res["sigma_ana"],
        A_inf=res["A_inf"], De=res["De"],
        max_err=res["max_err"], rms=res["rms"],
    )
    np.savez("output/exp_integrator_phase_b_ve_harmonic.npz", **out)


if __name__ == "__main__":
    main()
