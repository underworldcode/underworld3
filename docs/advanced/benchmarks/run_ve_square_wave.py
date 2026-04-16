"""Variable-timestep VE benchmark: square-wave forcing.

Maxwell material under square-wave shear rate (Fourier series, N harmonics):

    γ̇(t) = (4γ̇₀/π) Σ_{k=1..N} sin((2k-1)ωt) / (2k-1)

Since the Maxwell equation is linear, the analytical stress is the
superposition of single-frequency solutions:

    σ(t) = Σ_{k=1..N} maxwell_oscillatory(t, η, μ, aₖ, ωₖ)

where aₖ = 4γ̇₀/(π(2k-1)) and ωₖ = (2k-1)ω.

The sharp transitions demand small dt at the edges, large dt in the
flat regions — testing variable-dt BDF correctness.

Usage:
    python tests/run_ve_square_wave.py
"""

import time as timer
import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import expression


def maxwell_oscillatory(t, eta, mu, gamma_dot_0, omega):
    """Full analytical σ_xy for single-frequency oscillatory Maxwell shear."""
    t_r = eta / mu
    De = omega * t_r
    prefactor = eta * gamma_dot_0 / (1.0 + De**2)
    return prefactor * (np.sin(omega * t) - De * np.cos(omega * t) + De * np.exp(-t / t_r))


def square_wave_analytical(t, eta, mu, gamma_dot_0, omega, n_harmonics=20):
    """Analytical stress for square-wave forcing via Fourier superposition."""
    sigma = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        n = 2 * k - 1  # odd harmonics: 1, 3, 5, ...
        a_k = 4.0 * gamma_dot_0 / (np.pi * n)
        omega_k = n * omega
        sigma += maxwell_oscillatory(t, eta, mu, a_k, omega_k)
    return sigma


def square_wave_shear_rate(t, gamma_dot_0, omega, n_harmonics=20):
    """Square-wave shear rate via truncated Fourier series."""
    rate = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        n = 2 * k - 1
        rate += 4.0 * gamma_dot_0 / (np.pi * n) * np.sin(n * omega * t)
    return rate


def adaptive_dt(t_current, omega, dt_min, dt_max):
    """Adaptive timestep: small near square-wave transitions, large on plateaux.

    Transitions occur at t = (2m+1)·π/(2ω) for integer m, i.e. at odd
    multiples of quarter-period. We use distance to nearest transition
    to interpolate between dt_min and dt_max.
    """
    half_period = np.pi / omega
    # Phase within half-period [0, half_period)
    phase = t_current % half_period
    # Distance to nearest transition (0 or half_period boundary)
    dist = min(phase, half_period - phase)
    # Normalise to [0, 1] where 0 = at transition, 1 = mid-plateau
    frac = dist / (half_period / 2.0)
    # Smooth interpolation
    return dt_min + (dt_max - dt_min) * frac**2


def run_square_wave(order, De, n_periods, dt_min_over_tr, dt_max_over_tr,
                    n_harmonics=20, uniform=False):
    """Run VE square-wave shear box with adaptive or uniform timestep."""

    ETA, MU, H, W = 1.0, 1.0, 1.0, 2.0
    t_r = ETA / MU
    omega = De / t_r
    V0 = 0.5
    gamma_dot_0 = 2.0 * V0 / H
    dt_min = dt_min_over_tr * t_r
    dt_max = dt_max_over_tr * t_r
    T = 2.0 * np.pi / omega
    t_end = n_periods * T

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 8), minCoords=(-W / 2, -H / 2), maxCoords=(W / 2, H / 2),
    )
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

    stokes = uw.systems.VE_Stokes(mesh, velocityField=v, pressureField=p, order=order)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    stokes.constitutive_model.Parameters.shear_modulus = MU

    # Boundary conditions: simple shear driven by top/bottom velocity
    # V_top updated numerically each timestep to produce square-wave γ̇
    V_top = expression(R"V_{\mathrm{top}}", sympy.Float(0.0), "Top boundary velocity")

    stokes.add_dirichlet_bc((V_top, 0.0), "Top")
    stokes.add_dirichlet_bc((-V_top, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6

    # Time loop
    times = []
    numerical_stress = []
    timesteps_used = []
    t_current = 0.0
    step = 0

    while t_current < t_end:
        # Determine timestep
        if uniform:
            dt = dt_min
        else:
            dt = adaptive_dt(t_current, omega, dt_min, dt_max)

        t_next = t_current + dt

        # Update boundary velocity: V_top(t) such that γ̇ = 2·V_top/H = square wave
        # square_wave_shear_rate returns the actual γ̇, so V_top = γ̇ · H/2
        gamma_dot_t = square_wave_shear_rate(
            np.array([t_next]), gamma_dot_0, omega, n_harmonics
        )[0]
        V_top.sym = sympy.Float(gamma_dot_t * H / 2.0)

        stokes.constitutive_model.Parameters.dt_elastic = dt

        stokes.solve(zero_init_guess=False, timestep=dt)

        t_current += dt
        step += 1

        # Extract stress at mesh centre
        centre = np.array([[0.0, 0.0]])
        tau_xy = uw.function.evaluate(stokes.tau.sym[0, 1], centre)
        sigma_xy = float(tau_xy.flatten()[0])

        times.append(t_current)
        numerical_stress.append(sigma_xy)
        timesteps_used.append(dt)

        if step % 50 == 0:
            ana = square_wave_analytical(
                np.array([t_current]), ETA, MU, gamma_dot_0, omega, n_harmonics
            )[0]
            print(f"  Step {step:4d}: t/t_r = {t_current / t_r:.3f}, "
                  f"dt/t_r = {dt / t_r:.4f}, "
                  f"σ_xy = {sigma_xy:.6f}, ana = {ana:.6f}")

    times = np.array(times)
    numerical_stress = np.array(numerical_stress)
    timesteps_used = np.array(timesteps_used)

    # Analytical solution
    analytical_stress = square_wave_analytical(times, ETA, MU, gamma_dot_0, omega, n_harmonics)

    # Error metrics (skip first period for startup transient)
    mask = times > T
    if mask.sum() > 0:
        l2_err = np.sqrt(np.mean((numerical_stress[mask] - analytical_stress[mask]) ** 2))
        linf_err = np.max(np.abs(numerical_stress[mask] - analytical_stress[mask]))
    else:
        l2_err = linf_err = np.nan

    return {
        "times": times,
        "numerical": numerical_stress,
        "analytical": analytical_stress,
        "timesteps": timesteps_used,
        "l2_error": l2_err,
        "linf_error": linf_err,
        "n_steps": step,
    }


if __name__ == "__main__":
    De = 1.5
    order = 2
    n_periods = 3
    n_harmonics = 10

    print("=" * 60)
    print(f"Square-wave VE benchmark: De={De}, order={order}")
    print(f"  {n_harmonics} Fourier harmonics, {n_periods} periods")
    print("=" * 60)

    # Run with adaptive dt
    print("\n--- Adaptive timestep ---")
    t0 = timer.time()
    result_adaptive = run_square_wave(
        order=order, De=De, n_periods=n_periods,
        dt_min_over_tr=0.02, dt_max_over_tr=0.15,
        n_harmonics=n_harmonics, uniform=False,
    )
    t_adaptive = timer.time() - t0
    print(f"  {result_adaptive['n_steps']} steps in {t_adaptive:.1f}s")
    print(f"  L2 error:   {result_adaptive['l2_error']:.6e}")
    print(f"  Linf error: {result_adaptive['linf_error']:.6e}")
    print(f"  dt range:   [{result_adaptive['timesteps'].min():.4f}, "
          f"{result_adaptive['timesteps'].max():.4f}]")

    # Run with uniform dt (reference)
    print("\n--- Uniform timestep (dt_min) ---")
    t0 = timer.time()
    result_uniform = run_square_wave(
        order=order, De=De, n_periods=n_periods,
        dt_min_over_tr=0.02, dt_max_over_tr=0.02,
        n_harmonics=n_harmonics, uniform=True,
    )
    t_uniform = timer.time() - t0
    print(f"  {result_uniform['n_steps']} steps in {t_uniform:.1f}s")
    print(f"  L2 error:   {result_uniform['l2_error']:.6e}")
    print(f"  Linf error: {result_uniform['linf_error']:.6e}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Adaptive: {result_adaptive['n_steps']} steps, "
          f"L2 = {result_adaptive['l2_error']:.2e}")
    print(f"  Uniform:  {result_uniform['n_steps']} steps, "
          f"L2 = {result_uniform['l2_error']:.2e}")
    ratio = result_adaptive['l2_error'] / result_uniform['l2_error']
    print(f"  Error ratio (adaptive/uniform): {ratio:.2f}")
    print(f"  Step savings: {1 - result_adaptive['n_steps'] / result_uniform['n_steps']:.0%}")

    # Save results
    np.savez(
        f"tests/ve_square_wave_{n_harmonics}h.npz",
        adaptive_times=result_adaptive["times"],
        adaptive_numerical=result_adaptive["numerical"],
        adaptive_analytical=result_adaptive["analytical"],
        adaptive_timesteps=result_adaptive["timesteps"],
        uniform_times=result_uniform["times"],
        uniform_numerical=result_uniform["numerical"],
        uniform_analytical=result_uniform["analytical"],
        uniform_timesteps=result_uniform["timesteps"],
    )
    print(f"\nResults saved to tests/ve_square_wave_{n_harmonics}h.npz")
