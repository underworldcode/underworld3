"""Benchmark for issue #123: parameter-update recompile.

Reproduces the VE square-wave toggle pattern from
https://github.com/underworldcode/underworld3/issues/123 — the original
report measured ~459 s of JIT time vs ~3.7 s of actual SNES solve over
99 timesteps that only changed dt_elastic and the top BC sign.

With the C-source-hash JIT cache in place, the JIT time should collapse
to roughly the cost of the first cold compile (~10 s on this hardware),
because every later step generates the same C source and hits the cache.

Run with:

    pixi run -e amr-dev python -u docs/advanced/benchmarks/jit_cache_vs_recompile.py

Optional: set ``UW_JIT_CACHE=0`` to disable the disk cache and time the
in-memory cache only; or ``rm -rf ~/.cache/underworld3/jit`` first to
force a full cold start.
"""

import time

import sympy

import underworld3 as uw
import underworld3.timing as uw_timing
from underworld3.function import expression
from underworld3.utilities._jitextension import _ext_dict


N_STEPS = 30
ELEMENT_RES = (16, 8)


def main():
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=ELEMENT_RES,
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 0.5),
    )

    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=2,
    )
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.constitutive_model.Parameters.shear_modulus = 1.0

    V_top = expression("V_top", 0.5, "Top BC amplitude")
    stokes.add_dirichlet_bc((V_top, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")

    uw_timing.start()
    t_total = time.time()

    n_compiles_initial = len(_ext_dict)
    step_times = []
    for step in range(N_STEPS):
        V_top.sym = sympy.Float(0.5 * (-1) ** step)
        stokes.constitutive_model.Parameters.dt_elastic = 0.02
        t_step = time.time()
        stokes.solve(zero_init_guess=False, timestep=0.02)
        step_times.append(time.time() - t_step)

    elapsed = time.time() - t_total
    n_compiles_after = len(_ext_dict)

    uw.pprint(f"\n=== JIT cache benchmark — issue #123 reproducer ===")
    uw.pprint(f"Steps                : {N_STEPS}")
    uw.pprint(f"Mesh                 : {ELEMENT_RES}, BDF-2 VE Stokes")
    uw.pprint(f"Wall time (total)    : {elapsed:.1f} s")
    uw.pprint(f"First step (cold)    : {step_times[0]:.2f} s")
    uw.pprint(f"Mean of remaining    : {sum(step_times[1:]) / (N_STEPS - 1):.2f} s/step")
    uw.pprint(f"Compiled bundles     : {n_compiles_after - n_compiles_initial}")
    uw.pprint(f"  (issue #123: would have shown ~3 new compiles per step "
              f"= ~{N_STEPS * 3} total before this fix)")

    uw_timing.print_summary()


if __name__ == "__main__":
    main()
