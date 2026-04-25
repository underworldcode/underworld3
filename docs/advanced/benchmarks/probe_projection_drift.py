"""Replace the implicit projection of flux→psi_star[0] with a direct
one-shot pointwise evaluation. Test whether the drift disappears.

Procedure per step:
  1. Snapshot pre-solve psi_star[0] as `ps0_pre` (this is what ψ*[1] will
     become after the shift).
  2. Replace `_psi_star_projection_solver.solve` with a no-op so the main
     stokes.solve() does NOT update ψ*[0].
  3. Run stokes.solve() — Newton finds u, projection is a no-op, then
     the shift sets ψ*[1] = ps0_pre.  At this point ψ*[0] is still ps0_pre.
  4. Evaluate cm.flux at sample points using the just-solved u and the
     frozen pre-solve ψ*[0]. This is a pure forward computation (no
     fixed-point feedback).
  5. Assign the evaluated flux to ψ*[0].array.

For our uniform-shear test, the field is uniform so step 4 just samples
the centre and step 5 assigns uniformly.
"""

import numpy as np
import sympy
import underworld3 as uw
from underworld3.function import expression


def make_stokes(label):
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 8),
        minCoords=(-1, -0.5), maxCoords=(1, 0.5))
    v = uw.discretisation.MeshVariable(f"U_{label}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"P_{label}", mesh, 1, degree=1)
    s = uw.systems.VE_Stokes(mesh, velocityField=v, pressureField=p, order=2)
    s.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.constitutive_model.Parameters.shear_modulus = 1.0
    s.constitutive_model.Parameters.yield_stress = 0.5
    s.constitutive_model.Parameters.strainrate_inv_II_min = 1.0e-6
    s.constitutive_model._yield_mode = "min"
    Vt = expression(rf"V_{{{label}}}", sympy.Float(0.5), "Top V")
    s.add_dirichlet_bc((Vt, 0.0), "Top")
    s.add_dirichlet_bc((-Vt, 0.0), "Bottom")
    s.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    s.add_dirichlet_bc((sympy.oo, 0.0), "Right")
    s.tolerance = 1.0e-6
    s.petsc_options["snes_force_iteration"] = True
    return s, Vt


def patched_solve(stokes, dt, V_top, V_sign=1.0):
    """Run the standard solve, then OVERWRITE psi_star[0] with the manually
    computed σ from the formula evaluated against PRE-solve psi_star."""
    cm = stokes.constitutive_model
    ddt = stokes.DFDt
    centre = np.array([[0.0, 0.0]])

    V_top.sym = sympy.Float(V_sign * 0.5)
    cm.Parameters.dt_elastic = dt

    # Snapshot pre-solve psi_star (BOTH levels) — these are what the formula
    # should use for the implicit step
    ps0_pre = np.copy(ddt.psi_star[0].array)
    ps1_pre = np.copy(ddt.psi_star[1].array)

    # Run the standard solve as-is.  This will:
    #   - Run main Newton (finds u)
    #   - Run the (buggy) projection that writes into psi_star[0]
    #   - Shift: psi_star[1] = old (pre-solve) psi_star[0]   ← correct
    stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=1)

    # Now manually compute the analytical σ at this state and overwrite
    # the buggy projection's result.  Use PRE-solve psi_star values
    # in the formula (those are the "history" inputs to the implicit step).
    edot_now = float(uw.function.evaluate(cm.grad_u[0, 1], centre).flatten()[0])
    # In our test, fields are uniform so a single node value suffices.
    # Take the centre (or any node) of the snapshotted pre-solve arrays.
    n_nodes = ps0_pre.shape[0]
    ps0_use = float(ps0_pre[n_nodes // 2, 0, 1])  # pre-solve ψ*[0]_xy
    ps1_use = float(ps1_pre[n_nodes // 2, 0, 1])  # pre-solve ψ*[1]_xy
    c0 = float(cm._bdf_c0.sym); c1 = float(cm._bdf_c1.sym); c2 = float(cm._bdf_c2.sym)

    E_eff_xy = edot_now + (-c1) * ps0_use / (2 * 1 * dt) + (-c2) * ps1_use / (2 * 1 * dt)
    eta_ve_manual = 1 * dt / (c0 * 1 + 1 * dt)
    eta_pl_manual = 0.5 / (2 * abs(E_eff_xy)) if abs(E_eff_xy) > 1e-12 else 1e9
    eta_min = min(eta_ve_manual, eta_pl_manual)
    sigma_manual = 2 * eta_min * E_eff_xy

    # Read what the buggy projection produced (for diagnostic)
    sigma_buggy = float(uw.function.evaluate(stokes.tau.sym[0, 1], centre).flatten()[0])

    print(f"   [dt={dt:.3f}: pre ψ*=[{ps0_use:.4f},{ps1_use:.4f}] ε̇={edot_now:.4f} "
          f"c=[{c0:.3f},{c1:.3f},{c2:.3f}] manual σ={sigma_manual:.4f} buggy σ={sigma_buggy:.4f}]")

    # Overwrite psi_star[0] with the manual σ (uniform in our test problem)
    ddt.psi_star[0].array[:] = 0
    ddt.psi_star[0].array[:, 0, 1] = sigma_manual
    ddt.psi_star[0].array[:, 1, 0] = sigma_manual


def step_one(stokes, V, dt):
    V.sym = sympy.Float(0.5)
    stokes.constitutive_model.Parameters.dt_elastic = dt
    stokes.solve(zero_init_guess=False, timestep=dt, divergence_retries=1)


def probe(stokes, c=np.array([[0.0, 0.0]])):
    cm = stokes.constitutive_model
    return {
        'sigma': float(uw.function.evaluate(stokes.tau.sym[0, 1], c).flatten()[0]),
        'edot':  float(uw.function.evaluate(cm.grad_u[0, 1], c).flatten()[0]),
        'psi0':  float(uw.function.evaluate(stokes.DFDt.psi_star[0].sym[0, 1], c).flatten()[0]),
        'psi1':  float(uw.function.evaluate(stokes.DFDt.psi_star[1].sym[0, 1], c).flatten()[0]),
    }


def fmt(label, p):
    return f"{label:14s} σ={p['sigma']:.4f}  ε̇={p['edot']:.4f}  ψ*0={p['psi0']:.4f}  ψ*1={p['psi1']:.4f}"


# === Build two simulations: ORIGINAL (uses implicit projection) and PATCHED.

print("Building ORIGINAL stokes (implicit projection of flux→ψ*[0])...")
orig, V_o = make_stokes("orig")
print("Building PATCHED stokes (direct ptwise assign instead of projection)...")
patched, V_p = make_stokes("patched")

print("\n=== Phase 1: Drive both to yield steady state at dt=0.20 ===")
for k in range(5):
    step_one(orig, V_o, 0.20)
    patched_solve(patched, 0.20, V_p)
print(fmt("ORIG (after warm-up)", probe(orig)))
print(fmt("PATCHED (after warm-up)", probe(patched)))

print("\n=== Phase 2: switch to dt=0.10 (halving). Take 4 steps ===")
for k in range(4):
    step_one(orig, V_o, 0.10)
    patched_solve(patched, 0.10, V_p)
    print(f"Step {k+1} after halving:")
    print("  " + fmt("ORIG", probe(orig)))
    print("  " + fmt("PATCHED", probe(patched)))

print("\n=== Phase 3: switch back to dt=0.20 (doubling). Take 4 steps ===")
for k in range(4):
    step_one(orig, V_o, 0.20)
    patched_solve(patched, 0.20, V_p)
    print(f"Step {k+1} after doubling:")
    print("  " + fmt("ORIG", probe(orig)))
    print("  " + fmt("PATCHED", probe(patched)))
