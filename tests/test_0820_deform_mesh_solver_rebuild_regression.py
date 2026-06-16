"""Regression test for the re-opened portion of issue #122.

Before the fix, a second call to mesh._deform_mesh() followed by
stokes.solve(zero_init_guess=False) returned the pre-deform velocity
field unchanged. The solver's Fast Path 1 (introduced with PR #127)
short-circuited on a matching JIT cache key without checking the
_needs_dm_rebuild flag, so the cached DM (pre-deform coords) was reused
and F(v_prev) ≈ 0 → zero SNES iterations → no update.

Two-part fix:
  1. _deform_mesh now marks registered solvers is_setup=False, matching
     the behaviour of mesh.adapt().
  2. Fast Path 1 now respects _needs_dm_rebuild (matching Fast Path 2).

Either fix alone is insufficient; both must be present.
"""

import pytest
import numpy as np
import sympy

import underworld3 as uw


pytestmark = pytest.mark.level_1


def test_stokes_velocity_updates_after_second_deform():
    """Second mesh deform + re-solve must produce a velocity consistent with
    the new boundary amplitude, not the previous solution.
    """
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(50, 50), minCoords=(-0.5, -1.0), maxCoords=(0.5, 0.0)
    )
    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=1, continuous=True)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=0, continuous=False)

    def deform_with_amplitude(amp, var_name):
        """Solve Poisson for a cosine boundary and apply the result as a
        vertical displacement to the mesh."""
        disp_fn = amp * sympy.cos(2.0 * sympy.pi * mesh.X[0] / 1.0)
        Dz = uw.discretisation.MeshVariable(var_name, mesh, 1, degree=1)
        diff = uw.systems.Poisson(mesh, Dz)
        diff.constitutive_model = uw.constitutive_models.DiffusionModel
        diff.constitutive_model.Parameters.diffusivity = 1.0
        diff.add_essential_bc((disp_fn,), "Top")
        diff.add_essential_bc((0.0,), "Bottom")
        diff.solve()
        disp = np.zeros((mesh.X.coords.shape[0], mesh.dim))
        disp[:, -1] = uw.function.evaluate(Dz.sym[0], mesh.X.coords)[:, 0, 0]
        # This regression deliberately tests the gated internal primitive's
        # solver-rebuild (is_setup=False) contract — open the sanctioned scope.
        with mesh._coord_mutation():
            mesh._deform_mesh(mesh.X.coords + disp)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.bodyforce = sympy.Matrix([0, -1.0])
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 4e-5
    stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.shear_viscosity_0
    stokes.add_essential_bc((0.0, None), "Left")
    stokes.add_essential_bc((0.0, None), "Right")
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.tolerance = 1.0e-6
    stokes.petsc_options["ksp_rtol"] = 1.0e-6
    stokes.petsc_options["ksp_atol"] = 1.0e-6

    # --- first deform (amp 0.02) ---
    deform_with_amplitude(0.02, "Dz_a")
    stokes.solve(zero_init_guess=False)
    vel_first = uw.function.evaluate(v, mesh.X.coords)[:, 0, :].copy()
    vmax_first = abs(vel_first).max()
    assert vmax_first > 1.0, (
        f"first solve should produce nontrivial velocity (got {vmax_first:.3f})"
    )

    # --- second deform (additional amp 0.05 → total boundary amp ~0.07) ---
    deform_with_amplitude(0.05, "Dz_b")
    stokes.solve(zero_init_guess=False)
    vel_second = uw.function.evaluate(v, mesh.X.coords)[:, 0, :].copy()
    vmax_second = abs(vel_second).max()

    # Pre-fix: vmax_second == vmax_first (solver reused stale DM).
    # Post-fix: velocity scales with boundary amplitude (~3.3-3.5× here).
    ratio = vmax_second / vmax_first
    assert ratio > 1.5, (
        f"velocity did not respond to second deform — ratio {ratio:.2f} "
        f"(vmax_first={vmax_first:.3f}, vmax_second={vmax_second:.3f}). "
        "Symptom of the #122 re-opened bug: solver cached pre-deform DM."
    )
