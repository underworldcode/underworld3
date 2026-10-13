"""Phase D — 1D cleanroom bench for the per-component ETD-2 scheme.

Two parallel Maxwell branches with disparate relaxation times — the
exact analogue of the rank-4 TI tensor split into matrix-aligned (η_⊥)
and director-aligned (η_∥) channels:

    σ̇_⊥ + σ_⊥/τ_⊥ = μ ε̇,   τ_⊥ = η_⊥ / μ     (slow, matrix)
    σ̇_∥ + σ_∥/τ_∥ = μ ε̇,   τ_∥ = η_∥ / μ     (fast, post-yield clamp)
    σ_total = σ_⊥ + σ_∥

Both branches see the same engineering shear rate ε̇ = γ̇₀ cos(ωt).
The analytical solution is the sum of two independent Maxwell phasor
responses — fully closed-form, no numerical reference needed.

Three integrators run on the *total* stress:

  1. Per-component ETD-2 — propose, integrate σ_⊥ and σ_∥ separately
     with their own (α_⊥, φ_⊥) and (α_∥, φ_∥), then sum. (Phase D.)
  2. Lumped-effective ETD-2 — Phase B's current shape, one (α, φ) from
     τ_eff = (η_⊥ + η_∥) / μ on the total stress.
  3. Lumped-min ETD-2 — single (α, φ) from τ_min = min(τ_⊥, τ_∥); a
     prior lagged-τ experiment we already tried.

τ_∥ = 0.05 (post-yield-clamp regime), τ_⊥ = 1.0, μ = 1, ω = π/2,
Δt swept from 0.005 to 0.5.  Headline metric: max-|err|/A_∞_total.

Run::

    pixi run -e amr-dev python -u docs/developer/design/experiments/exp-integrator/_exp_integrator_phase_d_split.py
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Parameters ─────────────────────────────────────────────────────
MU = 1.0
ETA_PERP = 1.0     # matrix viscosity (slow)
ETA_PAR = 0.05     # post-yield-clamp director viscosity (fast)
TAU_PERP = ETA_PERP / MU
TAU_PAR = ETA_PAR / MU

GAMMA_DOT_0 = 1.0
OMEGA = np.pi / 2.0
N_PERIODS = 4.0
T_END = N_PERIODS * 2.0 * np.pi / OMEGA

OUT_DIR = "output"


def maxwell_sin(t, tau):
    """σ(t) for ε̇(t) = γ̇₀ cos(ωt), one Maxwell branch with relaxation τ.
    Phasor steady state plus decaying transient at σ(0) = 0."""
    De = OMEGA * tau
    A_inf = (MU * tau) * GAMMA_DOT_0 / np.sqrt(1 + De ** 2)
    phi = np.arctan(De)
    sigma_ss = A_inf * np.cos(OMEGA * t - phi)
    transient = -A_inf * np.cos(-phi) * np.exp(-t / tau)
    return sigma_ss + transient


def analytical_total(t):
    return maxwell_sin(t, TAU_PERP) + maxwell_sin(t, TAU_PAR)


# ── Integrators ────────────────────────────────────────────────────


def _alpha_phi(dt, tau):
    x = dt / tau
    alpha = np.exp(-x)
    if x > 1e-12:
        phi = (1 - alpha) / x
    else:
        phi = 1.0 - x / 2 + x * x / 6
    return alpha, phi


def etd2_step(gdot_n, gdot_np1, sigma_n, dt, tau, eta):
    """Single Maxwell branch: σⁿ⁺¹ = α σⁿ + μ[A γ̇ⁿ⁺¹ + B γ̇ⁿ]."""
    alpha, phi = _alpha_phi(dt, tau)
    A = tau * (1 - phi)
    B = tau * (phi - alpha)
    return alpha * sigma_n + MU * (A * gdot_np1 + B * gdot_n)


def run_per_component(dt):
    """Per-component scheme: integrate the two branches separately."""
    t = np.arange(0.0, T_END + 1e-12, dt)
    eps_dot = GAMMA_DOT_0 * np.cos(OMEGA * t)
    sigma_perp = np.zeros_like(t)
    sigma_par = np.zeros_like(t)
    for i in range(1, len(t)):
        sigma_perp[i] = etd2_step(
            eps_dot[i - 1], eps_dot[i], sigma_perp[i - 1], dt, TAU_PERP, ETA_PERP
        )
        sigma_par[i] = etd2_step(
            eps_dot[i - 1], eps_dot[i], sigma_par[i - 1], dt, TAU_PAR, ETA_PAR
        )
    return t, sigma_perp + sigma_par, sigma_perp, sigma_par


def run_lumped(dt, tau_choice):
    """Single-(α, φ) lump applied to the total stress.

    The effective viscosity in the lumped picture is η_⊥ + η_∥ (the
    instantaneous viscous stress is the sum of the two branches at
    γ̇₀), so the model is σ̇ + σ/τ_choice = (η_⊥ + η_∥) γ̇ / τ_choice
    — i.e. μ_eff γ̇ in our shorthand, where μ_eff = (η_⊥ + η_∥)/τ_choice.
    """
    t = np.arange(0.0, T_END + 1e-12, dt)
    eps_dot = GAMMA_DOT_0 * np.cos(OMEGA * t)
    sigma = np.zeros_like(t)
    eta_eff = ETA_PERP + ETA_PAR
    mu_eff = eta_eff / tau_choice
    alpha, phi = _alpha_phi(dt, tau_choice)
    A = tau_choice * (1 - phi)
    B = tau_choice * (phi - alpha)
    for i in range(1, len(t)):
        sigma[i] = (
            alpha * sigma[i - 1]
            + mu_eff * (A * eps_dot[i] + B * eps_dot[i - 1])
        )
    return t, sigma


# ── Main bench ─────────────────────────────────────────────────────


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Pick one Δt for the trajectory plot; sweep for the error figure.
    dt_show = 0.05
    dt_sweep = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])

    # --- trajectory at dt_show ---
    t_pc, s_pc_total, s_pc_perp, s_pc_par = run_per_component(dt_show)
    _, s_lump_eff = run_lumped(dt_show, TAU_PERP + TAU_PAR)   # naive sum
    _, s_lump_min = run_lumped(dt_show, TAU_PAR)              # min-τ
    _, s_lump_slow = run_lumped(dt_show, TAU_PERP)            # pick-the-slow

    s_ana = analytical_total(t_pc)
    s_perp_ana = maxwell_sin(t_pc, TAU_PERP)
    s_par_ana = maxwell_sin(t_pc, TAU_PAR)

    A_inf_total = np.max(np.abs(s_ana[len(s_ana) // 2:]))

    # --- err sweep ---
    err_pc, err_lump_eff, err_lump_min, err_lump_slow = [], [], [], []
    for dt in dt_sweep:
        t, s_pc, _, _ = run_per_component(dt)
        _, s_le = run_lumped(dt, TAU_PERP + TAU_PAR)
        _, s_lm = run_lumped(dt, TAU_PAR)
        _, s_ls = run_lumped(dt, TAU_PERP)
        ana = analytical_total(t)
        err_pc.append(np.max(np.abs(s_pc - ana)) / A_inf_total)
        err_lump_eff.append(np.max(np.abs(s_le - ana)) / A_inf_total)
        err_lump_min.append(np.max(np.abs(s_lm - ana)) / A_inf_total)
        err_lump_slow.append(np.max(np.abs(s_ls - ana)) / A_inf_total)

    err_pc = np.array(err_pc); err_lump_eff = np.array(err_lump_eff)
    err_lump_min = np.array(err_lump_min); err_lump_slow = np.array(err_lump_slow)

    print("Phase D 1D bench — two parallel Maxwell branches", flush=True)
    print(f"  τ_⊥={TAU_PERP}, τ_∥={TAU_PAR}, η_⊥={ETA_PERP}, η_∥={ETA_PAR}", flush=True)
    print(f"  ω={OMEGA:.4f}, γ̇₀={GAMMA_DOT_0}, T_END={T_END:.2f}", flush=True)
    print(f"  A_∞_total ≈ {A_inf_total:.4f}", flush=True)
    print(flush=True)
    print(f"{'dt':>8s} {'per-comp':>11s} {'lump-eff':>11s} {'lump-min':>11s} {'lump-slow':>11s}",
          flush=True)
    for i, dt in enumerate(dt_sweep):
        print(
            f"{dt:8.4f} {err_pc[i]:11.4e} {err_lump_eff[i]:11.4e} "
            f"{err_lump_min[i]:11.4e} {err_lump_slow[i]:11.4e}",
            flush=True,
        )

    # --- plots ---
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0])
    ax_traj = fig.add_subplot(gs[0, :])
    ax_split = fig.add_subplot(gs[1, 0])
    ax_err = fig.add_subplot(gs[1, 1])

    # Total trajectories
    ax_traj.plot(t_pc, s_ana, "-", color="black", lw=2.0, alpha=0.8,
                 label=f"analytical (total, A∞={A_inf_total:.3f})")
    ax_traj.plot(t_pc, s_pc_total, "--", color="#1f77b4", lw=1.5,
                 label=f"per-component ETD-2 "
                       f"(max|err|/A∞={err_pc[np.where(dt_sweep==dt_show)[0][0]]:.2e})")
    idx = np.where(dt_sweep == dt_show)[0][0]
    ax_traj.plot(t_pc, s_lump_eff, "--", color="#d62728", lw=1.2,
                 label=f"lumped τ=τ_⊥+τ_∥ "
                       f"(max|err|/A∞={err_lump_eff[idx]:.2e})")
    ax_traj.plot(t_pc, s_lump_slow, ":", color="#9467bd", lw=1.2,
                 label=f"lumped τ=τ_⊥ "
                       f"(max|err|/A∞={err_lump_slow[idx]:.2e})")
    ax_traj.plot(t_pc, s_lump_min, ":", color="#2ca02c", lw=1.2,
                 label=f"lumped τ=τ_∥ "
                       f"(max|err|/A∞={err_lump_min[idx]:.2e})")
    ax_traj.set_xlabel("time")
    ax_traj.set_ylabel(r"σ_total")
    ax_traj.set_title(rf"Total stress — Δt={dt_show}, τ_⊥={TAU_PERP}, τ_∥={TAU_PAR}")
    ax_traj.legend(loc="upper right", fontsize=8.5, ncol=1)
    ax_traj.grid(alpha=0.3)

    # Per-component split
    ax_split.plot(t_pc, s_perp_ana, "-", color="black", lw=1.6,
                  label=r"σ_⊥ analytical")
    ax_split.plot(t_pc, s_pc_perp, "--", color="#1f77b4", lw=1.2,
                  label=r"σ_⊥ ETD-2")
    ax_split.plot(t_pc, s_par_ana, "-", color="#444444", lw=1.6,
                  label=r"σ_∥ analytical")
    ax_split.plot(t_pc, s_pc_par, "--", color="#ff7f0e", lw=1.2,
                  label=r"σ_∥ ETD-2")
    ax_split.set_xlabel("time")
    ax_split.set_ylabel(r"branch stress")
    ax_split.set_title("Per-component branches resolved separately")
    ax_split.legend(loc="upper right", fontsize=8)
    ax_split.grid(alpha=0.3)

    # Error sweep (log-log)
    ax_err.loglog(dt_sweep, err_pc, "o-", color="#1f77b4", label="per-component")
    ax_err.loglog(dt_sweep, err_lump_eff, "s--", color="#d62728", label=r"lumped τ_⊥+τ_∥")
    ax_err.loglog(dt_sweep, err_lump_slow, "^:", color="#9467bd", label=r"lumped τ_⊥")
    ax_err.loglog(dt_sweep, err_lump_min, "v:", color="#2ca02c", label=r"lumped τ_∥")
    ax_err.set_xlabel(r"Δt")
    ax_err.set_ylabel(r"max|err|/A∞_total")
    ax_err.set_title(r"Error vs Δt (log-log)")
    ax_err.legend(loc="lower right", fontsize=8)
    ax_err.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Phase D — per-component ETD-2 vs lumped variants  "
        rf"(parallel Maxwell branches, τ_⊥={TAU_PERP}, τ_∥={TAU_PAR})",
        y=0.995, fontsize=11,
    )
    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, "exp_integrator_phase_d_split.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\n  wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
