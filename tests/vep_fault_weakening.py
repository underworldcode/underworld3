"""VEP shear box with embedded fault — convergence study.

Horizontal fault at y=0.5 using Surface gaussian influence function.
Runs at two vertical resolutions to check convergence.

Run: pixi run -e amr-dev python tests/vep_fault_weakening.py
"""

import time
import numpy as np
import sympy
import underworld3 as uw

ETA = 1.0
MU = 1.0
TAU_Y_FAULT = 0.2
TAU_Y_BULK = 2.0
FAULT_WIDTH = 0.08
DT = 0.1
NSTEPS = 25
V_TOP = 0.5


def run_fault_model(res_x, res_y):
    """Run the fault model at given resolution, return time series."""

    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res_x, res_y), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=2)
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, vtype=uw.VarType.VECTOR)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True,
                                        vtype=uw.VarType.SCALAR)

    fault_points = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
    fault = uw.meshing.Surface("fault", mesh, fault_points)
    fault.discretize()

    tau_y_field = fault.influence_function(
        width=FAULT_WIDTH, value_near=TAU_Y_FAULT, value_far=TAU_Y_BULK, profile="gaussian")

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
    cm = stokes.constitutive_model
    cm.Parameters.shear_viscosity_0 = ETA
    cm.Parameters.shear_modulus = MU
    cm.Parameters.yield_stress = tau_y_field
    cm.Parameters.shear_viscosity_min = ETA * 1.0e-2
    cm.Parameters.strainrate_inv_II_min = 1.0e-10
    # saddle_preconditioner left at default (uses constitutive stiffness)
    stokes.tolerance = 1.0e-4

    stokes.add_essential_bc(sympy.Matrix([V_TOP, 0.0]), "Top")
    stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")
    stokes.bodyforce = sympy.Matrix([0.0, 0.0])
    stokes.petsc_options["ksp_type"] = "fgmres"

    fault_sample = np.array([[0.5, 0.5]])
    bulk_sample = np.array([[0.5, 0.25]])

    times = []
    fault_stresses = []
    bulk_stresses = []
    converged = []

    for step in range(NSTEPS):
        stokes.solve(timestep=DT, zero_init_guess=False)
        t = (step + 1) * DT
        times.append(t)

        reason = stokes.snes.getConvergedReason()
        converged.append(reason)

        sd = stokes.tau.data
        coords = stokes.tau.coords
        fault_idx = np.argmin(np.sum((coords - fault_sample)**2, axis=1))
        bulk_idx = np.argmin(np.sum((coords - bulk_sample)**2, axis=1))
        fault_stresses.append(sd[fault_idx, 2])
        bulk_stresses.append(sd[bulk_idx, 2])

        flag = "" if reason > 0 else f"  SNES={reason}"
        if (step + 1) % 5 == 0 or step < 3 or reason < 0:
            uw.pprint(0, f"  [{res_x}x{res_y}] step {step+1}, t={t:.2f}, "
                      f"fault={fault_stresses[-1]:.4f}, bulk={bulk_stresses[-1]:.4f}{flag}")

    # Cross-section at final step
    y_coords = coords[:, 1]
    x_coords = coords[:, 0]
    near_centre = np.abs(x_coords - 0.5) < 0.1
    y_profile = y_coords[near_centre]
    sigma_profile = sd[near_centre, 2]
    sort_idx = y_profile.argsort()

    return {
        "times": np.array(times),
        "fault": np.array(fault_stresses),
        "bulk": np.array(bulk_stresses),
        "converged": np.array(converged),
        "profile_y": y_profile[sort_idx],
        "profile_sigma": sigma_profile[sort_idx],
    }


# --- Run both resolutions ---

t0 = time.time()
results = {}
for res_x, res_y in [(16, 16), (16, 32)]:
    uw.pprint(0, f"\n=== Resolution {res_x}x{res_y} ===")
    t1 = time.time()
    results[(res_x, res_y)] = run_fault_model(res_x, res_y)
    uw.pprint(0, f"  done ({time.time()-t1:.0f}s)")

uw.pprint(0, f"\nTotal: {time.time()-t0:.0f}s")

# --- Plot ---

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

t_anal = np.linspace(0.001, NSTEPS * DT, 200)
t_r = ETA / MU
maxwell = ETA * V_TOP * (1 - np.exp(-t_anal / t_r))

axes[0].plot(t_anal, maxwell, 'k--', linewidth=1, alpha=0.4, label="Maxwell")
for (rx, ry), r in results.items():
    axes[0].plot(r["times"], r["fault"], linewidth=2, label=f"fault {rx}x{ry}")
    axes[0].plot(r["times"], r["bulk"], linewidth=2, linestyle='--', label=f"bulk {rx}x{ry}")
    # Mark diverged steps
    div_mask = r["converged"] < 0
    if div_mask.any():
        axes[0].plot(r["times"][div_mask], r["fault"][div_mask], 'x', color='red', markersize=6)
axes[0].axhline(TAU_Y_FAULT, color='gray', linestyle=':', alpha=0.5)
axes[0].set_xlabel("Time")
axes[0].set_ylabel(r"$\sigma_{xy}$")
axes[0].set_title("Stress history")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

for (rx, ry), r in results.items():
    axes[1].plot(r["profile_y"], r["profile_sigma"], '-o', markersize=2,
                 linewidth=2, label=f"{rx}x{ry}")
axes[1].axvline(0.5, color='gray', linestyle=':', alpha=0.5, label="fault")
axes[1].set_xlabel("y")
axes[1].set_ylabel(r"$\sigma_{xy}$ (final step)")
axes[1].set_title("Stress profile across fault")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle(f"VEP embedded fault convergence: $\\tau_y$={TAU_Y_FAULT}/{TAU_Y_BULK}, width={FAULT_WIDTH}",
             fontsize=12, y=1.02)
fig.tight_layout()
out_path = "/Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/solver-unification/vep_fault.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
uw.pprint(0, f"Saved {out_path}")
