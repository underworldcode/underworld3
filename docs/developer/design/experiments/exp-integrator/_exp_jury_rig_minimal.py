"""Minimal jury-rig: build up the exponential constitutive model term by term.

  Step A: pure Newton fluid (sanity check the custom-class plumbing)
  Step B: Newton fluid + constant σⁿ history (additive stress)
  Step C: Add α·σⁿ with α<1 (real exp)
  Step D: Add ε̇ⁿ history term (full ETD-2)

If any step diverges, the failing addition is identified.
"""

import numpy as np
import sympy
import underworld3 as uw
from underworld3 import VarType
from underworld3.function import expression
from underworld3.constitutive_models import ViscousFlowModel


ETA = 1.0; MU = 1.0
V0 = 0.5
DT = 0.05
T_END = 0.5  # short — just need a few steps


def setup():
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 4), minCoords=(-1, -0.5), maxCoords=(1, 0.5),
    )
    v = uw.discretisation.MeshVariable("U_min", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_min", mesh, 1, degree=1)
    sigma_n = uw.discretisation.MeshVariable(
        "sigma_n_m", mesh, 2, degree=2, vtype=VarType.SYM_TENSOR,
    )
    epsdot_n = uw.discretisation.MeshVariable(
        "epsdot_n_m", mesh, 2, degree=2, vtype=VarType.SYM_TENSOR,
    )
    # Initialise to zero
    sigma_n.array[...] = 0.0
    epsdot_n.array[...] = 0.0
    return mesh, v, p, sigma_n, epsdot_n


def make_solver(mesh, v, p, custom_flux_fn):
    """Build a Stokes solver with a custom-flux constitutive model."""

    class _Custom(ViscousFlowModel):
        @property
        def K(self):
            return self.Parameters.shear_viscosity_0

        @property
        def flux(self):
            return custom_flux_fn(self)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    cm = _Custom(stokes.Unknowns)
    cm.Parameters.shear_viscosity_0 = ETA
    stokes.constitutive_model = cm
    stokes.tolerance = 1e-6
    stokes.petsc_options["snes_force_iteration"] = True

    V_top = expression(r"V_t", sympy.Float(0.0), "Top BC")
    stokes.add_essential_bc((V_top, 0.0), "Top")
    stokes.add_essential_bc((-V_top, 0.0), "Bottom")
    stokes.add_essential_bc((sympy.oo, 0.0), "Left")
    stokes.add_essential_bc((sympy.oo, 0.0), "Right")
    return stokes, cm, V_top


def run_a_few_steps(stokes, V_top, label, n_steps=4):
    print(f"\n--- {label} ---", flush=True)
    diverged = 0
    last_iters = 0
    for step in range(n_steps):
        V_top.sym = sympy.Float(V0)
        try:
            stokes.solve(zero_init_guess=(step == 0))
        except Exception as exc:
            print(f"    step {step+1}: solve raised: {exc}", flush=True)
            diverged += 1
            break
        reason = int(stokes.snes.getConvergedReason())
        last_iters = int(stokes.snes.getIterationNumber())
        if reason < 0:
            diverged += 1
            print(f"    step {step+1}: SNES diverged (reason={reason})", flush=True)
        else:
            print(f"    step {step+1}: converged in {last_iters} its (reason={reason})",
                  flush=True)


def main():
    # Step A: pure Newton fluid
    mesh, v, p, sigma_n, epsdot_n = setup()
    stokes, cm, V_top = make_solver(
        mesh, v, p,
        custom_flux_fn=lambda self: 2 * self.Parameters.shear_viscosity_0 * self.Unknowns.E,
    )
    run_a_few_steps(stokes, V_top, "A: pure Newton (2η·ε̇)")

    # Step B: Newton + uniform constant σ added (use sigma_n with σ_xy=0.3 baked in)
    mesh, v, p, sigma_n, epsdot_n = setup()
    sigma_n.array[:, 0, 1] = 0.3
    sigma_n.array[:, 1, 0] = 0.3
    stokes, cm, V_top = make_solver(
        mesh, v, p,
        custom_flux_fn=lambda self: (
            2 * self.Parameters.shear_viscosity_0 * self.Unknowns.E
            + sigma_n.sym
        ),
    )
    run_a_few_steps(stokes, V_top, "B: Newton + σⁿ (constant uniform σ_xy=0.3)")

    # Step C: scaled-down viscosity with α·σⁿ history (representative of exp)
    mesh, v, p, sigma_n, epsdot_n = setup()
    sigma_n.array[:, 0, 1] = 0.3
    sigma_n.array[:, 1, 0] = 0.3
    alpha_expr = expression(r"\alpha", sympy.Float(0.95), "α")
    phi_expr = expression(r"\varphi", sympy.Float(0.975), "φ")
    stokes, cm, V_top = make_solver(
        mesh, v, p,
        custom_flux_fn=lambda self: (
            2 * self.Parameters.shear_viscosity_0 * (1 - phi_expr) * self.Unknowns.E
            + alpha_expr * sigma_n.sym
        ),
    )
    run_a_few_steps(stokes, V_top, "C: 2η(1-φ)·ε̇ + α·σⁿ  (φ=0.975, α=0.95)")

    # Step D: full ETD-2 form including ε̇ⁿ history
    mesh, v, p, sigma_n, epsdot_n = setup()
    sigma_n.array[:, 0, 1] = 0.3
    sigma_n.array[:, 1, 0] = 0.3
    epsdot_n.array[:, 0, 1] = 0.5
    epsdot_n.array[:, 1, 0] = 0.5
    alpha_expr = expression(r"\alpha", sympy.Float(0.95), "α")
    phi_expr = expression(r"\varphi", sympy.Float(0.975), "φ")
    stokes, cm, V_top = make_solver(
        mesh, v, p,
        custom_flux_fn=lambda self: (
            2 * self.Parameters.shear_viscosity_0 * (1 - phi_expr) * self.Unknowns.E
            + alpha_expr * sigma_n.sym
            + 2 * self.Parameters.shear_viscosity_0
              * (phi_expr - alpha_expr) * epsdot_n.sym
        ),
    )
    run_a_few_steps(stokes, V_top, "D: full ETD-2 form")


if __name__ == "__main__":
    main()
