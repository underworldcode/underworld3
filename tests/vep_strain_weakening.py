"""VEP shear box with strain-weakening yield stress.

Plastic fraction computed directly from stored data:
  eta_ve = eta * mu * dt / (eta + mu * dt)              (known scalar)
  edot_II_eff = edot_kin + sigma_star / (2 * mu * dt)   (from psi_star data)
  eta_vep = min(eta_ve, tau_y / (2 * edot_II_eff))      (yield viscosity)
  plastic_fraction = max(0, 1 - eta_vep / eta_ve)

No evaluate, no projection — pure numpy on stored data.

Run: pixi run -e amr-dev python tests/vep_strain_weakening.py
"""

import time
import numpy as np
import sympy
import underworld3 as uw

ETA = 1.0
MU = 1.0
TAU_Y0 = 0.3
TAU_RESIDUAL = 0.1
EPS_CRIT = 0.5
DT = 0.1
NSTEPS = 30
V_TOP = 0.5

t0 = time.time()

mesh = uw.meshing.StructuredQuadBox(
    elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=2)
v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, vtype=uw.VarType.VECTOR)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True,
                                    vtype=uw.VarType.SCALAR)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
cm = stokes.constitutive_model
cm.Parameters.shear_viscosity_0 = ETA
cm.Parameters.shear_modulus = MU
cm.Parameters.yield_stress = TAU_Y0
cm.Parameters.shear_viscosity_min = ETA * 1.0e-3
cm.Parameters.strainrate_inv_II_min = 1.0e-10
stokes.tolerance = 1.0e-4

stokes.add_essential_bc(sympy.Matrix([V_TOP, 0.0]), "Top")
stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
stokes.add_essential_bc((sympy.oo, 0.0), "Left")
stokes.add_essential_bc((sympy.oo, 0.0), "Right")
stokes.bodyforce = sympy.Matrix([0.0, 0.0])
stokes.petsc_options["ksp_type"] = "fgmres"

print(f"Setup: {time.time()-t0:.1f}s")
print(f"eta={ETA}, mu={MU}, t_relax={ETA/MU}")
print(f"tau_y0={TAU_Y0}, tau_residual={TAU_RESIDUAL}, eps_crit={EPS_CRIT}")
print()

times = []
max_stresses = []
tau_y_values = []
eps_p_cum = 0.0
eps_p_values = []
edot_p_values = []

eta_ve = ETA * MU * DT / (ETA + MU * DT)
edot_kin = V_TOP / 2.0  # tensor shear rate for simple shear

print(f"{'step':>4} {'t':>5} {'sigma_xy':>10} {'tau_y':>8} {'eps_p':>8} {'edot_p':>8}")
print("-" * 60)

current_tau_y = TAU_Y0

for step in range(NSTEPS):
    cm.Parameters.yield_stress.sym = current_tau_y

    t1 = time.time()
    stokes.solve(timestep=DT, zero_init_guess=False)
    solve_time = time.time() - t1

    t = (step + 1) * DT
    times.append(t)

    sigma_xy = stokes.tau.data[:, 2].max()
    max_stresses.append(sigma_xy)

    # Compute plastic fraction from stored data
    sigma_star_xy = stokes.DFDt.psi_star[0].data[:, 2].mean()
    edot_history = sigma_star_xy / (2 * MU * DT)
    edot_II_eff = edot_kin + edot_history

    if edot_II_eff > 0:
        eta_vep = min(eta_ve, current_tau_y / (2 * edot_II_eff))
    else:
        eta_vep = eta_ve

    plastic_fraction = max(0.0, 1.0 - eta_vep / eta_ve)
    edot_p = edot_II_eff * plastic_fraction
    edot_p_values.append(edot_p)

    # Accumulate and weaken
    eps_p_cum += edot_p * DT
    weakening = min(eps_p_cum / EPS_CRIT, 1.0)
    current_tau_y = TAU_Y0 + (TAU_RESIDUAL - TAU_Y0) * weakening

    tau_y_values.append(current_tau_y)
    eps_p_values.append(eps_p_cum)

    if (step + 1) % 5 == 0 or step < 12:
        print(f"{step+1:4d} {t:5.2f} {sigma_xy:10.4f} {current_tau_y:8.4f} {eps_p_cum:8.4f} {edot_p:8.4f}  ({solve_time:.1f}s)")

print()
print(f"Total: {time.time()-t0:.0f}s")

# --- Plot ---

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

t_anal = np.linspace(0.001, max(times), 200)
t_r = ETA / MU
maxwell = ETA * V_TOP * (1 - np.exp(-t_anal / t_r))

axes[0].plot(t_anal, maxwell, 'k--', linewidth=1, alpha=0.4, label="Maxwell (no yield)")
axes[0].plot(times, max_stresses, 'r-', linewidth=2, label=r"$\sigma_{xy}$")
axes[0].plot(times, tau_y_values, 'b--', linewidth=1.5, label=r"$\tau_y(\varepsilon_p)$")
axes[0].axhline(TAU_Y0, color='gray', linestyle=':', alpha=0.5)
axes[0].axhline(TAU_RESIDUAL, color='gray', linestyle='-.', alpha=0.5)
axes[0].set_ylabel("Stress")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_title(f"VEP strain weakening: $\\eta$={ETA}, $\\mu$={MU}")

axes[1].plot(times, edot_p_values, 'm-', linewidth=2)
axes[1].set_ylabel(r"Plastic $\dot{\varepsilon}$")
axes[1].grid(True, alpha=0.3)

axes[2].plot(times, eps_p_values, 'g-', linewidth=2)
axes[2].axhline(EPS_CRIT, color='gray', linestyle=':', alpha=0.5, label=f"$\\varepsilon_{{crit}}$={EPS_CRIT}")
axes[2].set_xlabel("Time")
axes[2].set_ylabel(r"Accumulated $\varepsilon_p$")
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
out_path = "/Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/solver-unification/vep_strain_weakening.png"
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
