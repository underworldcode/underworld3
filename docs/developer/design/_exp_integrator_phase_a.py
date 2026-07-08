"""Phase A — 1D linear Maxwell exponential integrator validator.

Engineering form throughout: σ̇ + σ/τ = μ γ̇  (γ̇ engineering shear rate).
Steady-state under constant γ̇ → σ = η γ̇.  Compare:
  - Exponential integrator (proposed)
  - BDF-1
  - BDF-2 (constant Δt)
  - Analytical reference

Forcings:
  - sinusoidal: ε̇ = γ̇₀ cos(ωt)         → analytical Maxwell phasor
  - square-wave: ε̇ = ±γ̇₀                → piecewise exponential

Sweep Δt/τ from 0.01 to 10.  Output: max|err|, RMS, behaviour at large dt.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# ── Parameters (η = μ = 1 for clarity) ─────────────────────────────
ETA = 1.0; MU = 1.0; TAU = ETA / MU            # relaxation time = 1
GAMMA_DOT_0 = 1.0
T_END = 8 * TAU


# ── Analytical references ──────────────────────────────────────────

def maxwell_sin(t, omega, gamma_dot_0):
    """σ(t) for ε̇(t) = γ̇₀ cos(ωt), σ(0) = 0."""
    De = omega * TAU
    A_inf = ETA * gamma_dot_0 / np.sqrt(1 + De**2)
    phi = np.arctan(De)
    # σ_ss(t) = A_∞ cos(ωt - φ); transient = -A_∞ cos(-φ) e^(-t/τ)
    sigma_ss = A_inf * np.cos(omega * t - phi)
    transient = -A_inf * np.cos(-phi) * np.exp(-t / TAU)
    return sigma_ss + transient


def maxwell_square(t, half_period, gamma_dot_0):
    """σ(t) for square-wave γ̇, σ(0) = 0.  σ_start_n = value at start
    of period n; updated period-by-period by relaxing toward target."""
    sigma_ss = ETA * gamma_dot_0
    out = np.zeros_like(t)
    decay_full = np.exp(-half_period / TAU)
    n_prev = -1
    sigma_start = 0.0  # σ at start of period 0
    for i, ti in enumerate(t):
        n = int(ti // half_period)
        # Advance sigma_start to start-of-period-n if we crossed boundaries
        while n_prev < n - 1:
            n_prev += 1
            sign = 1.0 if n_prev % 2 == 0 else -1.0
            target = sign * sigma_ss
            sigma_start = target + (sigma_start - target) * decay_full
        n_prev = n
        sign = 1.0 if n % 2 == 0 else -1.0
        target = sign * sigma_ss
        t_local = ti - n * half_period
        out[i] = target + (sigma_start - target) * np.exp(-t_local / TAU)
    return out


# ── Integrators ────────────────────────────────────────────────────

def exp_integrator(gdot_n, gdot_np1, sigma_n, dt):
    """One step of σⁿ⁺¹ = α σⁿ + μ(A γ̇ⁿ⁺¹ + B γ̇ⁿ).  Engineering form."""
    x = dt / TAU
    alpha = np.exp(-x)
    phi = (1 - alpha) / x if x > 1e-12 else 1.0 - x/2 + x*x/6
    A = TAU * (1 - phi)
    B = TAU * (phi - alpha)
    return alpha * sigma_n + MU * (A * gdot_np1 + B * gdot_n)


def bdf1_step(gdot_np1, sigma_n, dt):
    """Backward Euler: σⁿ⁺¹ = (σⁿ + μΔt γ̇ⁿ⁺¹) / (1 + Δt/τ)."""
    return (sigma_n + MU * dt * gdot_np1) / (1 + dt / TAU)


def bdf2_step(gdot_np1, sigma_n, sigma_nm1, dt):
    """BDF-2 (constant dt): σⁿ⁺¹ (3/(2Δt) + 1/τ) = (4σⁿ - σⁿ⁻¹)/(2Δt) + μ γ̇ⁿ⁺¹"""
    lhs = 1.5 / dt + 1.0 / TAU
    rhs = (2 * sigma_n - 0.5 * sigma_nm1) / dt + MU * gdot_np1
    return rhs / lhs


# ── Run a forcing through each integrator ──────────────────────────

def run_sinusoidal(omega, dt):
    """Return (t, σ_exp, σ_bdf1, σ_bdf2, σ_ana)."""
    t = np.arange(0.0, T_END + 1e-12, dt)
    eps = GAMMA_DOT_0 * np.cos(omega * t)

    sig_exp = np.zeros_like(t)
    sig_b1 = np.zeros_like(t)
    sig_b2 = np.zeros_like(t)

    for i in range(1, len(t)):
        sig_exp[i] = exp_integrator(eps[i-1], eps[i], sig_exp[i-1], dt)
        sig_b1[i] = bdf1_step(eps[i], sig_b1[i-1], dt)
        if i == 1:
            # BDF-2 startup: do BDF-1 for the very first step
            sig_b2[i] = bdf1_step(eps[i], sig_b2[i-1], dt)
        else:
            sig_b2[i] = bdf2_step(eps[i], sig_b2[i-1], sig_b2[i-2], dt)

    sig_ana = maxwell_sin(t, omega, GAMMA_DOT_0)
    return t, sig_exp, sig_b1, sig_b2, sig_ana


def run_square(half_period, dt):
    t = np.arange(0.0, T_END + 1e-12, dt)
    # sign flips at integer multiples of half_period
    n_period = (t // half_period).astype(int)
    eps_at = GAMMA_DOT_0 * np.where(n_period % 2 == 0, 1.0, -1.0)
    # ε̇ at step boundaries: take the value at t (right-continuous)

    sig_exp = np.zeros_like(t)
    sig_b1 = np.zeros_like(t)
    sig_b2 = np.zeros_like(t)

    for i in range(1, len(t)):
        sig_exp[i] = exp_integrator(eps_at[i-1], eps_at[i], sig_exp[i-1], dt)
        sig_b1[i] = bdf1_step(eps_at[i], sig_b1[i-1], dt)
        if i == 1:
            sig_b2[i] = bdf1_step(eps_at[i], sig_b2[i-1], dt)
        else:
            sig_b2[i] = bdf2_step(eps_at[i], sig_b2[i-1], sig_b2[i-2], dt)

    sig_ana = maxwell_square(t, half_period, GAMMA_DOT_0)
    return t, sig_exp, sig_b1, sig_b2, sig_ana


def errors(sig, sig_ana):
    err = np.abs(sig - sig_ana)
    return float(err.max()), float(np.sqrt((err**2).mean()))


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Sinusoidal sweep over Δt/τ
    omega = np.pi / 2  # period 4τ
    dt_ratios = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    print("\n=== Sinusoidal ε̇(t) = cos(πt/2),  T = 8τ,  η = μ = τ = 1 ===")
    print(f"{'Δt/τ':>6} | {'exp max|err|':>12} {'bdf1 max':>10} {'bdf2 max':>10} | "
          f"{'exp rms':>10} {'bdf1 rms':>10} {'bdf2 rms':>10}")
    print("-" * 90)
    rows = []
    for r in dt_ratios:
        dt = r * TAU
        if dt > T_END / 4:
            continue
        t, se, s1, s2, sa = run_sinusoidal(omega, dt)
        em = errors(se, sa); e1 = errors(s1, sa); e2 = errors(s2, sa)
        print(f"{r:>6.3g} | {em[0]:>12.3e} {e1[0]:>10.3e} {e2[0]:>10.3e} | "
              f"{em[1]:>10.3e} {e1[1]:>10.3e} {e2[1]:>10.3e}")
        rows.append((r, dt, em[0], e1[0], e2[0], em[1], e1[1], e2[1]))

    # Square-wave (just one dt — focus on flip handling)
    half_period = 2 * TAU
    print("\n=== Square-wave (half-period = 2τ) ===")
    for dt in (0.05, 0.1, 0.2, 0.5):
        t, se, s1, s2, sa = run_square(half_period, dt)
        em = errors(se, sa); e1 = errors(s1, sa); e2 = errors(s2, sa)
        print(f"  dt = {dt:>4.2f}: exp max={em[0]:.3e}  bdf1 max={e1[0]:.3e}  "
              f"bdf2 max={e2[0]:.3e}")

    # Plot the dt-sweep convergence
    rs = np.array([row[0] for row in rows])
    em_max = np.array([row[2] for row in rows])
    e1_max = np.array([row[3] for row in rows])
    e2_max = np.array([row[4] for row in rows])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: convergence
    ax_l.loglog(rs, em_max, 'o-', label='Exponential', color='C0')
    ax_l.loglog(rs, e1_max, 's-', label='BDF-1', color='C1')
    ax_l.loglog(rs, e2_max, '^-', label='BDF-2', color='C2')
    # Reference slopes
    ax_l.loglog(rs, 1e-3 * rs / rs[0], 'k:', alpha=0.4, label='slope 1')
    ax_l.loglog(rs, 1e-4 * (rs / rs[0])**2, 'k--', alpha=0.4, label='slope 2')
    ax_l.set_xlabel(r'$\Delta t / \tau$')
    ax_l.set_ylabel(r'max $|\sigma_{\rm sim} - \sigma_{\rm ana}|$')
    ax_l.set_title('Sinusoidal forcing — dt convergence')
    ax_l.grid(True, which='both', alpha=0.3)
    ax_l.legend(fontsize=9)

    # Right: trace at large dt (1.0 if we have it)
    if 1.0 in rs:
        idx = list(rs).index(1.0)
        dt = TAU * 1.0
        t, se, s1, s2, sa = run_sinusoidal(omega, dt)
        ax_r.plot(t, sa, 'k-', label='analytical', linewidth=1.5)
        ax_r.plot(t, se, 'o-', label='Exponential', color='C0', markersize=4)
        ax_r.plot(t, s1, 's-', label='BDF-1', color='C1', markersize=4)
        ax_r.plot(t, s2, '^-', label='BDF-2', color='C2', markersize=4)
        ax_r.set_xlabel(r'$t / \tau$')
        ax_r.set_ylabel(r'$\sigma$')
        ax_r.set_title(rf'Trace at $\Delta t/\tau = 1$ (= 1/4 period)')
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(fontsize=9)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "exp_integrator_phase_a.png")
    fig.savefig(fig_path, dpi=140)
    print(f"\nWrote {fig_path}")


if __name__ == "__main__":
    main()
