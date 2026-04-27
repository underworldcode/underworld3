"""Phase B evaluation — exponential integrator under VEP, yield-active.

Two questions to answer numerically before committing to the UW3
``MaxwellExponentialFlowModel`` design:

  1. Does the exponential integrator + softmin yield (lagged-τ) handle
     a sub-/super-yield harmonic problem cleanly?  Compare against BDF-1
     (the current safe choice for fault problems).

  2. At Δt/τ ≥ 1 — the regime where BDF-1/2 collapse to no-amplitude
     output — does the exponential integrator give a physically
     meaningful answer?  This is the most interesting regime for
     mantle/lithosphere coupling where τ can be small.

Engineering form throughout: σ̇ + σ/τ = μ γ̇.  Steady viscous limit
σ → η γ̇.  Yield surface: |σ| ≤ τ_y.

The "VEP" treatment here uses **lagged-τ**: each step uses τ from the
previous step's η_eff (= softmin(η_ve_exp, η_pl)).  η_pl is the
Drucker-Prager-style instantaneous limiter.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# ── Parameters ─────────────────────────────────────────────────────
ETA = 1.0; MU = 1.0
TAU_VE = ETA / MU            # = 1
GAMMA_DOT_0 = 1.0
T_END = 16 * TAU_VE


# ── softmin yield (matches UW3's formula at δ=0.1) ─────────────────

def eta_eff_softmin(eta_ve, eta_pl, delta=0.1):
    """η_eff = η_ve / g(f) where f = η_ve/η_pl,
       g(f) = 1 + softplus(f-1) - softplus(-1) = 1 + (f-1+sqrt((f-1)²+δ²))/2 - offset
       This is a smooth approximation to min(η_ve, η_pl)."""
    f = eta_ve / eta_pl
    offset = (-1 + np.sqrt(1 + delta**2)) / 2
    g = 1 + (f - 1 + np.sqrt((f - 1)**2 + delta**2)) / 2 - offset
    return eta_ve / g


def eta_pl_DP(sigma, gamma_dot, tau_y, eps_min=1e-6):
    """Drucker-Prager-like plastic viscosity: τ_y = η_pl · γ̇  →  η_pl = τ_y/|γ̇|.
       Use |γ̇| including a tiny floor to avoid 1/0 at γ̇=0."""
    return tau_y / (abs(gamma_dot) + eps_min)


# ── Integrators (engineering Maxwell, optional yield) ──────────────

def step_exp_VEP(sigma_n, gdot_n, gdot_np1, dt, tau_y=None,
                 tau_prev=TAU_VE, delta=0.1):
    """One step of the exponential integrator with optional yield clip.

    For the prototype: predictor-corrector return mapping.
    1. Predict σ_pred via pure VE exponential update
    2. If |σ_pred| > τ_y: smoothly clip via softmin so |σ| → τ_y
    3. Update τ for next step based on (yielded or not)
    """
    x = dt / tau_prev
    alpha = np.exp(-x)
    phi = (1 - alpha) / x if x > 1e-12 else 1.0 - x/2 + x*x/6
    A = tau_prev * (1 - phi)
    B = tau_prev * (phi - alpha)
    sigma_pred = alpha * sigma_n + MU * (A * gdot_np1 + B * gdot_n)
    if tau_y is None:
        return sigma_pred, ETA / MU  # pure VE, full elastic τ
    # Smooth return-mapping clip via softmin on |σ|/τ_y
    # f = |σ_pred|/τ_y.  If f<1, no change.  If f>1, scale toward τ_y.
    f = abs(sigma_pred) / tau_y
    if f <= 1.0:
        return sigma_pred, ETA / MU  # below yield, full elastic relaxation
    offset = (-1 + np.sqrt(1 + delta**2)) / 2
    g = 1 + (f - 1 + np.sqrt((f - 1)**2 + delta**2)) / 2 - offset
    sigma_clipped = sigma_pred / g
    # During yield, effective relaxation time τ = η_pl/μ.
    # η_pl ≈ τ_y/|γ̇| (Drucker-Prager).  Use γ̇ⁿ⁺¹ for the lagged update.
    eta_pl = tau_y / (abs(gdot_np1) + 1e-6)
    tau_new = max(eta_pl / MU, 1e-3)  # floor to avoid α→1 numerical issues
    return sigma_clipped, tau_new


def step_bdf1_VEP(sigma_n, gdot_np1, dt, tau_y=None, delta=0.1):
    """BDF-1 with softmin return-mapping yield (parallel to step_exp_VEP)."""
    sigma_pred = (sigma_n + MU * dt * gdot_np1) / (1 + dt / TAU_VE)
    if tau_y is None:
        return sigma_pred
    f = abs(sigma_pred) / tau_y
    if f <= 1.0:
        return sigma_pred
    offset = (-1 + np.sqrt(1 + delta**2)) / 2
    g = 1 + (f - 1 + np.sqrt((f - 1)**2 + delta**2)) / 2 - offset
    return sigma_pred / g


# ── Square-wave analyticals and tests ──────────────────────────────

def maxwell_square_analytical(t, half_period, gamma_dot_0, tau_y=None):
    """σ(t) for square-wave γ̇, σ(0) = 0.  Optional yield clip at τ_y.

    Within period n: σ(t) = target_n + (σ_start_n - target_n) e^{-(t-n·HP)/τ}
    σ_start_(n+1) = target_n + (σ_start_n - target_n) e^{-HP/τ}

    For yielding: clip σ to [-τ_y, +τ_y] post-hoc.  This is approximate
    (real yielding clamps σ̇=0 once at yield, doesn't blend) but good
    enough for cross-checking integrators against the same model.
    """
    sigma_ss = ETA * gamma_dot_0
    decay = np.exp(-half_period / TAU_VE)
    out = np.zeros_like(t)
    n_prev = 0
    sigma_start = 0.0
    for i, ti in enumerate(t):
        n = int(ti // half_period)
        while n_prev < n:                         # advance one period at a time
            sign_p = 1.0 if n_prev % 2 == 0 else -1.0
            target_p = sign_p * sigma_ss
            sigma_start = target_p + (sigma_start - target_p) * decay
            n_prev += 1
        sign = 1.0 if n % 2 == 0 else -1.0
        target = sign * sigma_ss
        t_local = ti - n * half_period
        s = target + (sigma_start - target) * np.exp(-t_local / TAU_VE)
        if tau_y is not None and abs(s) > tau_y:
            s = np.sign(s) * tau_y
        out[i] = s
    return out


def test_square_VE_VEP(half_period, dt, tau_y=None):
    """Run all three integrators on square-wave forcing.  Returns
    (t, sig_exp, sig_b1, sig_b2, sig_ana)."""
    t = np.arange(0.0, T_END + 1e-12, dt)
    n_period = (t // half_period).astype(int)
    gdot_at = GAMMA_DOT_0 * np.where(n_period % 2 == 0, 1.0, -1.0)

    sig_exp = np.zeros_like(t)
    sig_b1 = np.zeros_like(t)
    tau_lag = TAU_VE
    for i in range(1, len(t)):
        sig_exp[i], tau_lag = step_exp_VEP(
            sig_exp[i-1], gdot_at[i-1], gdot_at[i], dt,
            tau_y=tau_y, tau_prev=tau_lag,
        )
        sig_b1[i] = step_bdf1_VEP(sig_b1[i-1], gdot_at[i], dt, tau_y=tau_y)
    sig_ana = maxwell_square_analytical(t, half_period, GAMMA_DOT_0, tau_y=tau_y)
    return t, sig_exp, sig_b1, sig_ana


# ── Test 1: yield-active sinusoidal forcing ────────────────────────

def test_yield_sin(omega, dt, tau_y):
    t = np.arange(0.0, T_END + 1e-12, dt)
    gdot = GAMMA_DOT_0 * np.cos(omega * t)

    sig_exp = np.zeros_like(t)
    sig_b1 = np.zeros_like(t)
    tau_lag = TAU_VE  # initial relaxation time
    for i in range(1, len(t)):
        sig_exp[i], tau_lag = step_exp_VEP(
            sig_exp[i-1], gdot[i-1], gdot[i], dt,
            tau_y=tau_y, tau_prev=tau_lag,
        )
        sig_b1[i] = step_bdf1_VEP(sig_b1[i-1], gdot[i], dt, tau_y=tau_y)
    return t, sig_exp, sig_b1


# ── Test 2: large-dt regime (Δt = τ, 2τ, 5τ) ───────────────────────

def test_largedt_sin(omega, dt):
    """Sinusoidal forcing, no yield (pure VE), large dt."""
    t = np.arange(0.0, T_END + 1e-12, dt)
    gdot = GAMMA_DOT_0 * np.cos(omega * t)
    sig_exp = np.zeros_like(t)
    sig_b1 = np.zeros_like(t)
    for i in range(1, len(t)):
        sig_exp[i], _ = step_exp_VEP(sig_exp[i-1], gdot[i-1], gdot[i], dt)
        sig_b1[i] = step_bdf1_VEP(sig_b1[i-1], gdot[i], dt)
    De = omega * TAU_VE
    A_inf = ETA * GAMMA_DOT_0 / np.sqrt(1 + De**2)
    phi = np.arctan(De)
    sig_ana = A_inf * (np.cos(omega * t - phi) - np.cos(phi) * np.exp(-t/TAU_VE))
    return t, sig_exp, sig_b1, sig_ana


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Test 1: yield-active VEP ─────────────────────────────────
    omega = np.pi / 4   # period 8τ — generous timestep window
    dt = 0.1 * TAU_VE
    print(f"\n=== Test 1: VEP harmonic, ω = π/4, dt = {dt} ===")
    print(f"{'τ_y':>5} | {'A_∞':>6} {'sub/sup':>8} | "
          f"{'Exp peak|σ|':>11} {'BDF-1 peak|σ|':>13} {'ratio':>6}")
    for tau_y in (0.10, 0.20, 0.30, 0.50):
        t, sig_exp, sig_b1 = test_yield_sin(omega, dt, tau_y)
        De = omega * TAU_VE
        A_inf = ETA * GAMMA_DOT_0 / np.sqrt(1 + De**2)
        regime = "sub" if A_inf <= tau_y else "sup"
        peak_e = np.abs(sig_exp).max()
        peak_b = np.abs(sig_b1).max()
        print(f"{tau_y:>5.2f} | {A_inf:>6.3f} {regime:>8} | "
              f"{peak_e:>11.4f} {peak_b:>13.4f} {peak_e/peak_b:>6.3f}")

    # ── Test 2: large dt ─────────────────────────────────────────
    print(f"\n=== Test 2: Pure VE harmonic at large Δt/τ ===")
    print(f"{'Δt/τ':>5} | {'Exp max|err|':>12} {'Exp peak':>9} | "
          f"{'BDF-1 max|err|':>14} {'BDF-1 peak':>10} | {'analytical peak':>15}")
    for dt_over_tau in (0.5, 1.0, 2.0, 5.0):
        dt = dt_over_tau * TAU_VE
        t, sig_exp, sig_b1, sig_ana = test_largedt_sin(omega, dt)
        peak_ana = np.abs(sig_ana).max()
        peak_e = np.abs(sig_exp).max()
        peak_b = np.abs(sig_b1).max()
        err_e = np.abs(sig_exp - sig_ana).max()
        err_b = np.abs(sig_b1 - sig_ana).max()
        print(f"{dt_over_tau:>5.2g} | {err_e:>12.3e} {peak_e:>9.4f} | "
              f"{err_b:>14.3e} {peak_b:>10.4f} | {peak_ana:>15.4f}")

    # ── Plot 1: yield-active VEP traces ──────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    omega = np.pi / 4
    dt = 0.1 * TAU_VE
    De = omega * TAU_VE
    A_inf = ETA * GAMMA_DOT_0 / np.sqrt(1 + De**2)
    for ax, tau_y in zip(axes.ravel(), (0.10, 0.20, 0.30, 0.50)):
        t, sig_exp, sig_b1 = test_yield_sin(omega, dt, tau_y)
        # Reference: pure-VE no-yield analytical
        phi = np.arctan(De)
        sig_ve = A_inf * (np.cos(omega * t - phi) - np.cos(phi) * np.exp(-t/TAU_VE))
        ax.plot(t, sig_ve, ':', color='0.4', linewidth=1, label='VE (no yield)')
        ax.axhline(+tau_y, color='gray', linestyle=':', alpha=0.6, linewidth=1)
        ax.axhline(-tau_y, color='gray', linestyle=':', alpha=0.6, linewidth=1)
        ax.plot(t, sig_exp, '-', color='C0', linewidth=1.4, label='Exponential')
        ax.plot(t, sig_b1, '-', color='C1', linewidth=1.4, label='BDF-1', alpha=0.85)
        regime = "sub-yield" if A_inf <= tau_y else "yielding"
        ax.set_title(rf'$\tau_y = {tau_y}$  ({regime}; $A_\infty = {A_inf:.3f}$)')
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9, loc='upper right')
    for ax in axes[1]:
        ax.set_xlabel(r'$t/\tau$')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\sigma$')
    fig.suptitle("VEP harmonic — Exponential vs BDF-1 (lagged-τ softmin yield, δ=0.1)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "exp_integrator_phase_b_yield.png"), dpi=140)
    print(f"\nWrote phase_b_yield.png")

    # ── Test 3: Square-wave VE and VEP ───────────────────────────
    half_period = 2 * TAU_VE
    print(f"\n=== Test 3: Square-wave VE  (half-period = 2τ) ===")
    print(f"{'dt':>5} | {'Exp max|err|':>12} {'Exp peak':>9} | "
          f"{'BDF-1 max|err|':>14} {'BDF-1 peak':>10} | {'Ana peak':>8}")
    for dt in (0.05, 0.1, 0.25, 0.5, 1.0):
        t, se, s1, sa = test_square_VE_VEP(half_period, dt, tau_y=None)
        eme = np.abs(se - sa).max(); em1 = np.abs(s1 - sa).max()
        print(f"{dt:>5.2f} | {eme:>12.3e} {np.abs(se).max():>9.4f} | "
              f"{em1:>14.3e} {np.abs(s1).max():>10.4f} | {np.abs(sa).max():>8.4f}")

    print(f"\n=== Test 4: Square-wave VEP (half-period = 2τ, τ_y = 0.4) ===")
    tau_y = 0.4
    print(f"{'dt':>5} | {'Exp max|err|':>12} {'Exp peak':>9} | "
          f"{'BDF-1 max|err|':>14} {'BDF-1 peak':>10} | {'τ_y':>5}")
    for dt in (0.05, 0.1, 0.25, 0.5):
        t, se, s1, sa = test_square_VE_VEP(half_period, dt, tau_y=tau_y)
        eme = np.abs(se - sa).max(); em1 = np.abs(s1 - sa).max()
        print(f"{dt:>5.2f} | {eme:>12.3e} {np.abs(se).max():>9.4f} | "
              f"{em1:>14.3e} {np.abs(s1).max():>10.4f} | {tau_y:>5.2f}")

    # ── Plot 3: square-wave VE/VEP traces ────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    half_period = 2 * TAU_VE
    for ax, dt, tau_y_plot, title in [
        (axes[0, 0], 0.1, None, "VE  Δt=0.1τ"),
        (axes[0, 1], 0.5, None, "VE  Δt=0.5τ (large)"),
        (axes[1, 0], 0.1, 0.4,  "VEP Δt=0.1τ, τ_y=0.4"),
        (axes[1, 1], 0.5, 0.4,  "VEP Δt=0.5τ, τ_y=0.4"),
    ]:
        t, se, s1, sa = test_square_VE_VEP(half_period, dt, tau_y=tau_y_plot)
        ax.plot(t, sa, 'k-', lw=1.5, label='analytical')
        ax.plot(t, se, 'o-', color='C0', ms=3, label='Exp', alpha=0.85)
        ax.plot(t, s1, 's-', color='C1', ms=3, label='BDF-1', alpha=0.85)
        if tau_y_plot is not None:
            ax.axhline(+tau_y_plot, color='gray', ls=':', alpha=0.5)
            ax.axhline(-tau_y_plot, color='gray', ls=':', alpha=0.5)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    for ax in axes[1]: ax.set_xlabel(r'$t/\tau$')
    for ax in axes[:, 0]: ax.set_ylabel(r'$\sigma$')
    fig.suptitle("Square-wave forcing: VE & VEP — Exponential vs BDF-1",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "exp_integrator_phase_b_square.png"), dpi=140)
    print(f"\nWrote phase_b_square.png")

    # ── Plot 2: large-dt traces ──────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    for ax, dt_over_tau in zip(axes.ravel(), (0.5, 1.0, 2.0, 5.0)):
        dt = dt_over_tau * TAU_VE
        t, sig_exp, sig_b1, sig_ana = test_largedt_sin(omega, dt)
        ax.plot(t, sig_ana, 'k-', linewidth=1.5, label='analytical')
        ax.plot(t, sig_exp, 'o-', color='C0', markersize=3, label='Exp')
        ax.plot(t, sig_b1, 's-', color='C1', markersize=3, label='BDF-1', alpha=0.85)
        ax.set_title(rf'$\Delta t/\tau = {dt_over_tau}$')
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Pure VE harmonic at large Δt/τ — Exponential vs BDF-1",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "exp_integrator_phase_b_largedt.png"), dpi=140)
    print(f"Wrote phase_b_largedt.png")


if __name__ == "__main__":
    main()
