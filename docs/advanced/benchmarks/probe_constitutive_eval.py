"""Evaluate the constitutive expressions DIRECTLY at the centre, using
specified pre-solve psi_star values. No FE solve involved — just plug
numbers into the symbolic stress() and viscosity formulas. If the
formulas give σ = τ_y under Min mode, the formulas are right. If the
SIM gives σ ≠ τ_y at the same input state, the bug is in solve/project
not in the formulas.
"""

import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import expression


# Build a fresh stokes problem
mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 8),
    minCoords=(-1, -0.5), maxCoords=(1, 0.5))
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
    stokes.Unknowns, order=2,
)
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.constitutive_model.Parameters.shear_modulus = 1.0
stokes.constitutive_model.Parameters.yield_stress = 0.5
stokes.constitutive_model.Parameters.strainrate_inv_II_min = 1.0e-6
stokes.constitutive_model._yield_mode = "min"
cm = stokes.constitutive_model
V_top = expression(R"V_{top}", sympy.Float(0.5), "Top V")
stokes.add_dirichlet_bc((V_top, 0.0), "Top")
stokes.add_dirichlet_bc((-V_top, 0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")
stokes.tolerance = 1.0e-6
stokes.petsc_options["snes_force_iteration"] = True

# Solve a few steps at dt=0.20 to populate u to uniform shear at yield
for _ in range(5):
    cm.Parameters.dt_elastic = 0.20
    stokes.solve(zero_init_guess=False, timestep=0.20, divergence_retries=1)

centre = np.array([[0.0, 0.0]])
print(f"\nAfter coarse warm-up:")
print(f"  σ at centre = {float(uw.function.evaluate(stokes.tau.sym[0,1], centre).flatten()[0]):.4f}")

# === EXPERIMENT: directly inject specific psi_star values, then EVALUATE
# the constitutive law without solving. We'll then compare with an
# actual solve at the same state.

# Set ψ*[0] = 0.5 (yielded), ψ*[1] = 0.4268 (pre-yield from coarse step)
stokes.DFDt.psi_star[0].array[:] = 0
stokes.DFDt.psi_star[0].array[:, 0, 1] = 0.5
stokes.DFDt.psi_star[0].array[:, 1, 0] = 0.5
stokes.DFDt.psi_star[1].array[:] = 0
stokes.DFDt.psi_star[1].array[:, 0, 1] = 0.4268
stokes.DFDt.psi_star[1].array[:, 1, 0] = 0.4268
# Force dt_history[0] = 0.20 so that the next step at dt=0.10 is "halving"
stokes.DFDt._dt_history[0] = 0.20
stokes.DFDt._dt_history[1] = 0.20

# Set dt = 0.10 and update BDF coefficients
cm.Parameters.dt_elastic = 0.10
cm._update_bdf_coefficients()

print(f"\nAfter setting state:")
print(f"  ψ*[0] at centre = {float(uw.function.evaluate(stokes.DFDt.psi_star[0].sym[0,1], centre).flatten()[0]):.4f}")
print(f"  ψ*[1] at centre = {float(uw.function.evaluate(stokes.DFDt.psi_star[1].sym[0,1], centre).flatten()[0]):.4f}")
print(f"  dt_h[0] = {stokes.DFDt._dt_history[0]}, dt_e = {float(cm.Parameters.dt_elastic.sym):.3f}")
print(f"  c_0 = {float(cm._bdf_c0.sym):.4f}")
print(f"  c_1 = {float(cm._bdf_c1.sym):.4f}")
print(f"  c_2 = {float(cm._bdf_c2.sym):.4f}")

# Now WITHOUT solving, evaluate the symbolic formulas at the centre.
print("\n=== Direct evaluation of symbolic constitutive formulas ===")
print("(no FE solve; using the velocity field from the warm-up which is uniform shear)")
edot_xy = float(uw.function.evaluate(cm.grad_u[0,1], centre).flatten()[0])
print(f"  ε̇_xy = {edot_xy:.4f}")
E_eff_xy = float(uw.function.evaluate(cm.E_eff.sym[0,1], centre).flatten()[0])
E_eff_inv_II = float(uw.function.evaluate(cm.E_eff_inv_II.sym, centre).flatten()[0])
print(f"  E_eff_xy   = {E_eff_xy:.4f}")
print(f"  E_eff_inv_II = {E_eff_inv_II:.4f}")
eta_ve = float(uw.function.evaluate(cm.Parameters.ve_effective_viscosity.sym, centre).flatten()[0])
eta_pl = float(uw.function.evaluate(cm._plastic_effective_viscosity, centre).flatten()[0])
eta_min = float(uw.function.evaluate(cm.viscosity, centre).flatten()[0])
print(f"  η_ve = {eta_ve:.4f}")
print(f"  η_pl = {eta_pl:.4f}")
print(f"  η = Min(η_ve, η_pl) = {eta_min:.4f}  (expected min: {min(eta_ve, eta_pl):.4f})")

# Direct evaluation of the stress() formula
stress_formula = cm.stress()
sigma_xy_direct = float(uw.function.evaluate(stress_formula[0,1], centre).flatten()[0])
print(f"  σ_xy direct evaluation of stress() formula = {sigma_xy_direct:.4f}")
print(f"  Predicted from 2·η·E_eff = {2*eta_min*E_eff_xy:.4f}")

# === Now solve and see what comes out
print("\n=== Run a SOLVE with ψ*[1] = 0.4268 (pre-yield) ===")
stokes.solve(zero_init_guess=False, timestep=0.10, divergence_retries=1)
sigma_after_solve_a = float(uw.function.evaluate(stokes.tau.sym[0,1], centre).flatten()[0])
print(f"  σ_xy after solve = {sigma_after_solve_a:.4f}")

# === User's hypothesis: ψ*[1] = pre-yield is the issue. Try ψ*[1] = ψ*[0]
# (matches what FINE has — both at yield)
print("\n=== Reset and re-solve with ψ*[1] = ψ*[0] = 0.5 (both at yield) ===")
stokes.DFDt.psi_star[0].array[:] = 0
stokes.DFDt.psi_star[0].array[:, 0, 1] = 0.5
stokes.DFDt.psi_star[0].array[:, 1, 0] = 0.5
stokes.DFDt.psi_star[1].array[:] = 0
stokes.DFDt.psi_star[1].array[:, 0, 1] = 0.5
stokes.DFDt.psi_star[1].array[:, 1, 0] = 0.5
stokes.DFDt._dt_history[0] = 0.20  # still dt-halving
cm.Parameters.dt_elastic = 0.10
cm._update_bdf_coefficients()
stokes.solve(zero_init_guess=False, timestep=0.10, divergence_retries=1)
sigma_after_solve_b = float(uw.function.evaluate(stokes.tau.sym[0,1], centre).flatten()[0])
print(f"  σ_xy after solve = {sigma_after_solve_b:.4f}")

# === And also: ψ*[1] = ψ*[0] = 0.5 with dt_history = [0.10, 0.10] (consistent)
print("\n=== Reset, ψ*[1] = ψ*[0] = 0.5, dt_history = [0.10, 0.10] (fully consistent) ===")
stokes.DFDt.psi_star[0].array[:] = 0
stokes.DFDt.psi_star[0].array[:, 0, 1] = 0.5
stokes.DFDt.psi_star[0].array[:, 1, 0] = 0.5
stokes.DFDt.psi_star[1].array[:] = 0
stokes.DFDt.psi_star[1].array[:, 0, 1] = 0.5
stokes.DFDt.psi_star[1].array[:, 1, 0] = 0.5
stokes.DFDt._dt_history[0] = 0.10
stokes.DFDt._dt_history[1] = 0.10
cm.Parameters.dt_elastic = 0.10
cm._update_bdf_coefficients()

# Probe the symbolic formula once more to confirm Min selection is correct
sigma_direct = float(uw.function.evaluate(cm.stress()[0,1], centre).flatten()[0])
eta_ve = float(uw.function.evaluate(cm.Parameters.ve_effective_viscosity.sym, centre).flatten()[0])
eta_pl = float(uw.function.evaluate(cm._plastic_effective_viscosity, centre).flatten()[0])
eta_min = float(uw.function.evaluate(cm.viscosity, centre).flatten()[0])
print(f"  Direct symbolic eval BEFORE solve: σ = {sigma_direct:.4f}, η_ve={eta_ve:.4f}, η_pl={eta_pl:.4f}, η_min={eta_min:.4f}")

stokes.solve(zero_init_guess=False, timestep=0.10, divergence_retries=1)
sigma_after_solve_c = float(uw.function.evaluate(stokes.tau.sym[0,1], centre).flatten()[0])
print(f"  σ_xy after solve = {sigma_after_solve_c:.4f}")

# === Also: solve TWICE (force re-iteration to settle)
print("\n=== Re-solve with same state to see if it self-corrects ===")
stokes.DFDt.psi_star[0].array[:] = 0
stokes.DFDt.psi_star[0].array[:, 0, 1] = 0.5
stokes.DFDt.psi_star[0].array[:, 1, 0] = 0.5
stokes.DFDt.psi_star[1].array[:] = 0
stokes.DFDt.psi_star[1].array[:, 0, 1] = 0.5
stokes.DFDt.psi_star[1].array[:, 1, 0] = 0.5
stokes.DFDt._dt_history[0] = 0.10
stokes.DFDt._dt_history[1] = 0.10
cm.Parameters.dt_elastic = 0.10
cm._update_bdf_coefficients()
for k in range(5):
    stokes.solve(zero_init_guess=False, timestep=0.10, divergence_retries=1)
    s = float(uw.function.evaluate(stokes.tau.sym[0,1], centre).flatten()[0])
    print(f"  After solve {k+1}: σ_xy = {s:.4f}")

print(f"\n=== Summary ===")
print(f"  (a) ψ*=[0.5, 0.4268], dt_h0=0.20 → σ = {sigma_after_solve_a:.4f}  (large overshoot)")
print(f"  (b) ψ*=[0.5, 0.5000], dt_h0=0.20 → σ = {sigma_after_solve_b:.4f}  (clean halving)")
print(f"  (c) ψ*=[0.5, 0.5000], dt_h0=0.10 → σ = {sigma_after_solve_c:.4f}  (no dt change)")
