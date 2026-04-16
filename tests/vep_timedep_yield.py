"""VEP with time-dependent yield stress (no feedback).

Prescribed tau_y(t) that decreases linearly over time.
VE stress builds, hits tau_y, then tau_y drops and stress follows.

No strain accumulation, no projection of viscosity ratios —
just the solver responding to a changing yield stress parameter.

Run: pixi run -e amr-dev python tests/vep_timedep_yield.py
"""

import time
import numpy as np
import sympy
import underworld3 as uw

ETA = 1.0
MU = 1.0
V_TOP = 0.5
DT = 0.1
NSTEPS = 25

# tau_y schedule: starts above Maxwell steady state, drops below it
TAU_Y_START = 1.5   # well above Maxwell steady state (2*eta*edot = 1.0)
TAU_Y_END = 0.2
TAU_Y_DROP_START = 0.8   # start dropping at t=0.8
TAU_Y_DROP_END = 1.8     # reach minimum at t=1.8

t0 = time.time()

mesh = uw.meshing.StructuredQuadBox(
    elementRes=(4, 4),
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    qdegree=2,
)

v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, vtype=uw.VarType.VECTOR)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True,
                                    vtype=uw.VarType.SCALAR)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
cm = stokes.constitutive_model
cm.Parameters.shear_viscosity_0 = ETA
cm.Parameters.shear_modulus = MU
cm.Parameters.shear_viscosity_min = ETA * 1.0e-3
cm.Parameters.strainrate_inv_II_min = 1.0e-10
stokes.saddle_preconditioner = 1.0
stokes.tolerance = 1.0e-4

# Start with high yield stress
cm.Parameters.yield_stress = TAU_Y_START

stokes.add_essential_bc(sympy.Matrix([V_TOP, 0.0]), "Top")
stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
stokes.add_essential_bc((sympy.oo, 0.0), "Left")
stokes.add_essential_bc((sympy.oo, 0.0), "Right")
stokes.bodyforce = sympy.Matrix([0.0, 0.0])
stokes.petsc_options["ksp_type"] = "fgmres"

print(f"Setup: {time.time()-t0:.1f}s")
print(f"eta={ETA}, mu={MU}, t_relax={ETA/MU}")
print(f"tau_y: {TAU_Y_START} -> {TAU_Y_END} (t={TAU_Y_DROP_START}..{TAU_Y_DROP_END})")
print(f"Maxwell steady state: eta*gamma_dot = {ETA*V_TOP}")
print()

times = []
max_stresses = []
tau_y_values = []

print(f"{'step':>4} {'t':>5} {'sigma_xy':>10} {'tau_y':>8} {'phase':>10}")
print("-" * 50)

for step in range(NSTEPS):
    t = (step + 1) * DT

    # Update tau_y for this step
    if t < TAU_Y_DROP_START:
        current_tau_y = TAU_Y_START
    elif t > TAU_Y_DROP_END:
        current_tau_y = TAU_Y_END
    else:
        frac = (t - TAU_Y_DROP_START) / (TAU_Y_DROP_END - TAU_Y_DROP_START)
        current_tau_y = TAU_Y_START + (TAU_Y_END - TAU_Y_START) * frac

    cm.Parameters.yield_stress = current_tau_y

    t1 = time.time()
    stokes.solve(timestep=DT, zero_init_guess=False)
    solve_time = time.time() - t1

    times.append(t)
    tau_y_values.append(current_tau_y)

    sd = stokes.tau.data
    sigma_xy = sd[:, 2].max()
    max_stresses.append(sigma_xy)

    # Determine phase
    if sigma_xy > 0.95 * current_tau_y:
        phase = "yield"
    else:
        phase = "elastic"

    print(f"{step+1:4d} {t:5.2f} {sigma_xy:10.4f} {current_tau_y:8.3f} {phase:>10}  ({solve_time:.1f}s)")

print()
print(f"Total: {time.time()-t0:.0f}s")

# --- Plot ---

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

t_anal = np.linspace(0.001, max(times), 200)
t_r = ETA / MU
# Maxwell: sigma_xy = eta * gamma_dot * (1 - exp(-t/t_r))
# gamma_dot = V_TOP / H = V_TOP (H=1)
maxwell = ETA * V_TOP * (1 - np.exp(-t_anal / t_r))

ax.plot(t_anal, maxwell, 'k--', linewidth=1, alpha=0.4, label="Maxwell (no yield)")
ax.plot(times, max_stresses, 'r-o', linewidth=2, markersize=3, label=r"$\sigma_{xy}$")
ax.plot(times, tau_y_values, 'b--', linewidth=1.5, label=r"$\tau_y(t)$")
ax.set_xlabel("Time")
ax.set_ylabel("Stress")
ax.set_title(f"VEP with prescribed weakening: $\\eta$={ETA}, $\\mu$={MU}")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = "/Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/solver-unification/vep_timedep_yield.png"
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
