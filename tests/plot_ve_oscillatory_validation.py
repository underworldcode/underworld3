"""Viscoelastic oscillatory shear — validation plot for PR.

Produces a figure showing:
  - Top panel: applied shear rate γ̇(t) = γ̇₀ sin(ωt)
  - Bottom panel: stress σ_xy(t) for order 1 and order 2 vs analytical

Maxwell analytical solution including startup transient:
    σ_xy(t) = η γ̇₀/(1+De²) [sin(ωt) - De cos(ωt) + De exp(-t/t_r)]

Results are saved as .npz checkpoint files for re-analysis.

Usage:
    python tests/plot_ve_oscillatory_validation.py
    python tests/plot_ve_oscillatory_validation.py --replot   # replot from saved data
"""

import numpy as np
import sympy
import os
import sys
import underworld3 as uw
from underworld3.function import expression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def maxwell_oscillatory(t, eta, mu, gamma_dot_0, omega):
    """Full analytical σ_xy for oscillatory Maxwell shear (incl. transient)."""
    t_r = eta / mu
    De = omega * t_r
    prefactor = eta * gamma_dot_0 / (1.0 + De**2)
    return prefactor * (np.sin(omega * t) - De * np.cos(omega * t) + De * np.exp(-t / t_r))


def run_oscillatory(order, n_steps, dt, V0, H, ETA, MU, omega, save_prefix=None):
    """Run oscillatory VE shear box, return time series and save checkpoints."""
    W = 2.0
    gamma_dot_0 = 2.0 * V0 / H

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 8), minCoords=(-W / 2, -H / 2), maxCoords=(W / 2, H / 2),
    )
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

    stokes = uw.systems.VE_Stokes(mesh, velocityField=v, pressureField=p, order=order)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = ETA
    stokes.constitutive_model.Parameters.shear_modulus = MU
    stokes.constitutive_model.Parameters.dt_elastic = dt

    V_bc = expression(r"{V_{bc}}", 0.0, "Time-dependent boundary velocity")

    stokes.add_dirichlet_bc((V_bc, 0.0), "Top")
    stokes.add_dirichlet_bc((-V_bc, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    stokes.tolerance = 1.0e-6

    centre = np.array([[0.0, 0.0]])
    times, stress_num = [], []
    time_phys = 0.0

    checkpoint_interval = max(1, n_steps // 20)

    for step in range(n_steps):
        time_phys += dt

        V_t = V0 * np.sin(omega * time_phys)
        V_bc.sym = V_t

        stokes.solve(zero_init_guess=False, timestep=dt, evalf=False)

        val = uw.function.evaluate(stokes.tau.sym[0, 1], centre)
        sigma_xy = float(val.flatten()[0])
        times.append(time_phys)
        stress_num.append(sigma_xy)

        if save_prefix and (step + 1) % checkpoint_interval == 0:
            np.savez(
                f"{save_prefix}_order{order}_checkpoint.npz",
                times=np.array(times),
                stress=np.array(stress_num),
                step=step + 1,
                dt=dt,
                omega=omega,
                ETA=ETA,
                MU=MU,
                V0=V0,
                H=H,
                order=order,
            )

    del stokes, mesh

    t_arr = np.array(times)
    s_arr = np.array(stress_num)

    if save_prefix:
        np.savez(
            f"{save_prefix}_order{order}_final.npz",
            times=t_arr,
            stress=s_arr,
            dt=dt,
            omega=omega,
            ETA=ETA,
            MU=MU,
            V0=V0,
            H=H,
            order=order,
            gamma_dot_0=gamma_dot_0,
        )

    return t_arr, s_arr


def make_plot(results, params, output_path):
    """Generate the validation figure from results dict."""

    ETA = params["ETA"]
    MU = params["MU"]
    V0 = params["V0"]
    H = params["H"]
    omega = params["omega"]
    dt = params["dt"]
    t_r = ETA / MU
    De = omega * t_r
    gamma_dot_0 = 2.0 * V0 / H

    # Determine time range from results
    t_max = max(r[0][-1] for r in results.values())

    # Analytical solution (fine sampling)
    t_fine = np.linspace(0, t_max, 1000)
    sigma_analytical = maxwell_oscillatory(t_fine, ETA, MU, gamma_dot_0, omega)
    shear_rate = gamma_dot_0 * np.sin(omega * t_fine)

    phase_lag_deg = np.degrees(np.arctan(De))
    steady_amp = ETA * gamma_dot_0 / np.sqrt(1 + De**2)
    period = 2.0 * np.pi / omega

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.5]},
    )

    # Top panel: applied shear rate
    ax1.plot(t_fine / t_r, shear_rate, "k-", linewidth=1.2,
             label=r"$\dot{\gamma}(t) = \dot{\gamma}_0 \sin(\omega t)$")
    ax1.set_ylabel(r"Shear rate $\dot{\gamma}$")
    ax1.axhline(0, color="grey", linewidth=0.5)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlim(0, t_max / t_r)

    # Bottom panel: stress response
    ax2.plot(t_fine / t_r, sigma_analytical, "k-", linewidth=1.5,
             label="Analytical (Maxwell)", zorder=3)

    colors = {1: "#2196F3", 2: "#E91E63", 3: "#4CAF50"}
    markers = {1: "o", 2: "s", 3: "^"}

    for order in sorted(results.keys()):
        t, stress = results[order]
        ax2.plot(t / t_r, stress, markers.get(order, "o"),
                 color=colors.get(order, "grey"),
                 markersize=2.5, alpha=0.7,
                 label=f"Order {order} (BDF-{order})", zorder=2)

    ax2.set_xlabel(r"Time $t / t_r$")
    ax2.set_ylabel(r"Stress $\sigma_{xy}$")
    ax2.axhline(0, color="grey", linewidth=0.5)
    ax2.legend(loc="lower right", fontsize=9)

    ax2.set_title(
        f"Maxwell viscoelastic shear: De = {De:.1f}, "
        r"$\delta t / t_r$" + f" = {dt / t_r:.2f}, "
        f"phase lag = {phase_lag_deg:.0f}\u00b0, "
        f"steady amplitude = {steady_amp:.3f}",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":

    output_dir = "tests"
    save_prefix = os.path.join(output_dir, "ve_oscillatory")

    replot = "--replot" in sys.argv

    # Physical parameters
    ETA = 1.0
    MU = 1.0
    V0 = 0.5
    H = 1.0
    t_r = ETA / MU

    De = 5.0  # High Deborah number — strong elastic effects, 79° phase lag
    omega = De / t_r
    period = 2.0 * np.pi / omega
    n_periods = 4
    dt = 0.02 * t_r  # Fine enough to resolve the fast oscillations
    n_steps = int(n_periods * period / dt)

    params = dict(ETA=ETA, MU=MU, V0=V0, H=H, omega=omega, dt=dt)

    print(f"Maxwell oscillatory shear: De={De}, omega={omega:.3f}, period={period:.3f}")
    print(f"dt={dt}, dt/t_r={dt/t_r}, n_steps={n_steps}, t_end={n_steps * dt:.2f}")
    print(f"Phase lag = {np.degrees(np.arctan(De)):.1f} deg")
    print()

    if replot:
        # Load saved data
        results = {}
        for order in [1, 2]:
            fpath = f"{save_prefix}_order{order}_final.npz"
            if os.path.exists(fpath):
                data = np.load(fpath)
                results[order] = (data["times"], data["stress"])
                print(f"Loaded order {order} from {fpath}")
            else:
                print(f"Missing {fpath} — run without --replot first")
                sys.exit(1)
    else:
        results = {}
        for order in [1, 2]:
            print(f"Running order {order} ({n_steps} steps)...", end=" ", flush=True)
            t, stress = run_oscillatory(
                order, n_steps, dt, V0, H, ETA, MU, omega,
                save_prefix=save_prefix,
            )
            results[order] = (t, stress)
            print("done")

    make_plot(results, params, os.path.join(output_dir, "ve_oscillatory_validation.png"))
