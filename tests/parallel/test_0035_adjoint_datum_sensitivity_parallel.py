"""The Dirichlet-datum part of an adjoint sensitivity, on every rank count.

``_essential_datum_owner`` asked PETSc for each boundary's points with
``getStratumIS``. On a rank holding no point of that boundary, PETSc hard-aborts
for a value missing from the rank's own value set, and otherwise returns an IS
wrapping a NULL handle, not ``None``, whose first use segfaults. Every datum
sensitivity died at np 2, with or without units. The baseline is a central
difference of the forward solve, which uses no adjoint machinery.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0035_adjoint_datum_sensitivity_parallel.py
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(600)]


@pytest.mark.parametrize("references", [False, True], ids=["no units", "units"])
def test_a_lid_datum_gradient_matches_central_differences(references):
    if references:
        uw.get_default_model().set_reference_quantities(
            length=uw.quantity(30, "km"), viscosity=uw.quantity(1e21, "Pa*s"),
            time=uw.quantity(1, "Myr"))
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1 / 8)
    x, y = mesh.X
    alpha = uw.expression(r"\alpha", 2.0, "lid speed")
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0, -sympy.sin(sympy.pi * x)])
    stokes.add_dirichlet_bc((alpha * x * (1 - x), 0.0), "Top")
    for wall in ("Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.tolerance = 1.0e-11
    stokes.solve()
    u = stokes.Unknowns.u
    misfit = u.sym[1] * sympy.sin(sympy.pi * x) + u.sym[0] * y

    adjoint = stokes.gradient(misfit, parameters=[alpha])["parameters"][alpha]

    h = 1e-4 * 2.0
    values = []
    for a in (2.0 + h, 2.0 - h):
        alpha.sym = a
        stokes.solve(zero_init_guess=True)
        values.append(float(stokes.gradient(misfit, parameters=[])["J"]))
    alpha.sym = 2.0
    assert adjoint == pytest.approx((values[0] - values[1]) / (2 * h), rel=1e-6)
