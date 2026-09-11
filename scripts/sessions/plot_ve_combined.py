"""Combined VE benchmark figure: oscillatory + square-wave shear.

Two rows, each with:
  - Left axis: stress response (analytical + numerical)
  - Right axis: strain rate (filled grey)

Reads from saved .npz data — run the benchmarks first:
  python tests/plot_ve_oscillatory_validation.py
  python docs/advanced/benchmarks/run_ve_square_wave.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def maxwell_oscillatory_stress(t, eta, mu, gamma_dot_0, omega):
    """Full analytical stress including startup transient."""
    t_r = eta / mu
    De = omega * t_r
    coeff = eta * gamma_dot_0 / (1 + De**2)
    return coeff * (np.sin(omega * t) - De * np.cos(omega * t) + De * np.exp(-t / t_r))


def square_wave_stress(t, eta, mu, gamma_dot_0, omega, n_harmonics=10):
    """Analytical stress for Fourier-truncated square wave."""
    stress = np.zeros_like(t)
    t_r = eta / mu
    for k in range(1, n_harmonics + 1):
        n = 2 * k - 1
        a_k = 4 * gamma_dot_0 / (np.pi * n)
        omega_k = n * omega
        De_k = omega_k * t_r
        coeff = eta * a_k / (1 + De_k**2)
        stress += coeff * (np.sin(omega_k * t) - De_k * np.cos(omega_k * t) + De_k * np.exp(-t / t_r))
    return stress


def discontinuous_square_wave_stress(t, eta, mu, gamma_dot_0, omega):
    """Exact analytical stress for true discontinuous square-wave shear.

    On each half-period the shear rate is constant (±γ̇₀), and the
    Maxwell stress relaxes exponentially toward ±η γ̇₀.  The stress
    at the start of each half-period is inherited from the end of
    the previous one, giving a recursive formula.

    σ(t) on the n-th half-period [n·T/2, (n+1)·T/2]:
        σ(t) = σ_∞ + (σ_n - σ_∞) exp(-(t - n·T/2) / t_r)

    where σ_∞ = (-1)^n · η γ̇₀ is the viscous target, σ_n is the
    stress at the start of the interval, and t_r = η/μ.
    """
    t_r = eta / mu
    half_period = np.pi / omega
    sigma_ss = eta * gamma_dot_0  # viscous steady state magnitude

    stress = np.zeros_like(t)
    for i, ti in enumerate(t):
        # Which half-period are we in?
        n = int(ti / half_period)
        t_local = ti - n * half_period

        # Build up σ_n by recursion from σ_0 = 0
        sigma_start = 0.0
        for j in range(n):
            sign = 1.0 if j % 2 == 0 else -1.0
            sigma_target = sign * sigma_ss
            sigma_start = sigma_target + (sigma_start - sigma_target) * np.exp(-half_period / t_r)

        # Current half-period target
        sign = 1.0 if n % 2 == 0 else -1.0
        sigma_target = sign * sigma_ss

        stress[i] = sigma_target + (sigma_start - sigma_target) * np.exp(-t_local / t_r)

    return stress


def square_wave_shear_rate(t, gamma_dot_0, omega, n_harmonics=10):
    """Fourier-truncated square wave shear rate."""
    rate = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        n = 2 * k - 1
        rate += np.sin(n * omega * t) / n
    return rate * 4 * gamma_dot_0 / np.pi


def true_square_wave_rate(t, gamma_dot_0, omega):
    """True discontinuous square wave shear rate."""
    half_period = np.pi / omega
    phase = t % (2 * half_period)
    return np.where(phase < half_period, gamma_dot_0, -gamma_dot_0)


# ── Load data ──

# Oscillatory
osc = {}
for order in [1, 2]:
    d = np.load(f"tests/ve_oscillatory_order{order}_final.npz")
    osc[order] = d

eta = float(osc[1]["ETA"])
mu = float(osc[1]["MU"])
omega_osc = float(osc[1]["omega"])
gamma_dot_0_osc = float(osc[1]["gamma_dot_0"])
t_r = eta / mu
De_osc = omega_osc * t_r

# Square wave
sq = np.load("tests/ve_square_wave_10h.npz")
# Reconstruct parameters from the oscillatory data (same material)
De_sq = 1.5  # from the benchmark script
omega_sq = De_sq / t_r
gamma_dot_0_sq = gamma_dot_0_osc  # same V0/H

# ── Figure ──

fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

# Shared style
ana_style = dict(color="black", linewidth=1.5, zorder=8)
fill_color = "0.88"
fill_edge = "0.45"
stress_colors = {1: "#E91E63", 2: "#2196F3"}
stress_markers = {1: "o", 2: "s"}
marker_kw = dict(markersize=3, alpha=0.8, linewidth=0, zorder=6)


def setup_twin_axes(ax, ax_r, stress_lim, rate_lim):
    """Align zero for both axes and set symmetric limits.
    Put the twin (strain rate) behind the primary (stress)."""
    ax.set_ylim(-stress_lim, stress_lim)
    ax_r.set_ylim(-rate_lim, rate_lim)
    ax.axhline(0, color="grey", linewidth=0.3, zorder=0)
    # Twin axis renders on top by default — move it behind
    ax.set_zorder(ax_r.get_zorder() + 1)
    ax.set_frame_on(False)  # make primary axis background transparent


# ────────────────────────────
# Panel 1: Oscillatory shear
# ────────────────────────────
ax1 = axes[0]
ax1r = ax1.twinx()

t1 = osc[1]["times"]
t_fine = np.linspace(0, t1.max(), 2000)

# Strain rate — background layer (filled + dark outline)
strain_rate_osc = gamma_dot_0_osc * np.sin(omega_osc * t_fine)
ax1r.fill_between(t_fine / t_r, strain_rate_osc, color=fill_color, zorder=1)
ax1r.plot(t_fine / t_r, strain_rate_osc, color=fill_edge, linewidth=0.8, zorder=2)
ax1r.set_ylabel(r"Shear rate $\dot\gamma$", color="0.4")
ax1r.tick_params(axis="y", labelcolor="0.4")

# Analytical stress — foreground
sigma_ana = maxwell_oscillatory_stress(t_fine, eta, mu, gamma_dot_0_osc, omega_osc)
ax1.plot(t_fine / t_r, sigma_ana, label="Analytical (Maxwell)", **ana_style)

# Numerical dots — on top
for order in [1, 2]:
    t_num = osc[order]["times"]
    s_num = osc[order]["stress"]
    ax1.plot(t_num / t_r, s_num, stress_markers[order],
             color=stress_colors[order],
             label=f"BDF-{order}", **marker_kw)

# Align zero, symmetric limits
stress_max = max(abs(sigma_ana.max()), abs(sigma_ana.min())) * 1.3
rate_max = gamma_dot_0_osc * 1.3
setup_twin_axes(ax1, ax1r, stress_max, rate_max)

ax1.set_ylabel(r"Stress $\sigma_{xy}$")
ax1.set_xlabel(r"Time $t / t_r$")
ax1.legend(loc="lower right", fontsize=9)

phase_lag = np.degrees(np.arctan(De_osc))
steady_amp = eta * gamma_dot_0_osc / np.sqrt(1 + De_osc**2)
ax1.set_title(f"Oscillatory shear: De = {De_osc:.1f}, "
              f"phase lag = {phase_lag:.0f}\u00b0, "
              f"steady amplitude = {steady_amp:.3f}",
              fontsize=10)

# ────────────────────────────
# Panel 2: Square-wave shear
# ────────────────────────────
ax2 = axes[1]
ax2r = ax2.twinx()

t_sq_fine = np.linspace(0, sq["uniform_times"].max(), 4000)

# Strain rate — true discontinuous square wave (background)
sr_disc = true_square_wave_rate(t_sq_fine, gamma_dot_0_sq, omega_sq)
ax2r.fill_between(t_sq_fine / t_r, sr_disc, color=fill_color, zorder=1, step="mid")
ax2r.step(t_sq_fine / t_r, sr_disc, color=fill_edge, linewidth=0.8, zorder=2, where="mid")
ax2r.set_ylabel(r"Shear rate $\dot\gamma$", color="0.4")
ax2r.tick_params(axis="y", labelcolor="0.4")

# Determine time range from data
if "disc_times" in sq:
    t_max_plot = sq["disc_times"].max()
    t_plot_fine = np.linspace(0, t_max_plot, 4000)
else:
    t_max_plot = sq["adaptive_times"].max()
    t_plot_fine = t_sq_fine

# Analytical stress — discontinuous exact (foreground)
sigma_disc_ana = discontinuous_square_wave_stress(t_plot_fine, eta, mu, gamma_dot_0_sq, omega_sq)
ax2.plot(t_plot_fine / t_r, sigma_disc_ana, label="Analytical (exact)", **ana_style)

# Also show Fourier analytical for comparison (dashed)
sigma_fourier_ana = square_wave_stress(t_plot_fine, eta, mu, gamma_dot_0_sq, omega_sq)
ax2.plot(t_plot_fine / t_r, sigma_fourier_ana, "--", color="0.5", linewidth=1,
         label="Analytical (10 harmonics)", zorder=7)

# Numerical: Fourier BCs
ax2.plot(sq["adaptive_times"] / t_r, sq["adaptive_numerical"],
         "o", color=stress_colors[1],
         label=f"Fourier BCs ({len(sq['adaptive_times'])} steps)",
         **marker_kw)

# Numerical: discontinuous BCs
if "disc_times" in sq:
    ax2.plot(sq["disc_times"] / t_r, sq["disc_numerical"],
             "^", color="#4CAF50", markersize=3.5, alpha=0.8,
             linewidth=0, zorder=6,
             label=f"Discontinuous BCs ({len(sq['disc_times'])} steps)")

# Align zero, symmetric limits
stress_sq_max = max(abs(sigma_disc_ana).max(), abs(sigma_fourier_ana).max()) * 1.3
rate_sq_max = gamma_dot_0_sq * 1.5
setup_twin_axes(ax2, ax2r, stress_sq_max, rate_sq_max)
ax2.set_xlim(0, t_max_plot / t_r)

ax2.set_ylabel(r"Stress $\sigma_{xy}$")
ax2.set_xlabel(r"Time $t / t_r$")
ax2.legend(loc="lower right", fontsize=8)
ax2.set_title(f"Square-wave shear: De = {De_sq:.1f}, "
              f"discontinuous vs Fourier BCs, BDF-2",
              fontsize=10)

# ── Save ──

output_path = "tests/ve_benchmarks_combined.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
