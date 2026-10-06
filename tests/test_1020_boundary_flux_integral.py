"""Direct scalar reaction integrals, with exact signed conduction references.

The same small tests run in serial or under mpirun; no serial golden file is
required. Only Top and Bottom are driven, avoiding mixed corner reactions.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("degree", [1, 2])
def test_signed_reaction_integral(dim, degree):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim,
        cellSize=0.25, regular=True, qdegree=3,
    )
    temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=degree)
    poisson = uw.systems.Poisson(mesh, u_Field=temperature)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 2.5
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")
    poisson.tolerance = 1e-11
    poisson.petsc_options["snes_type"] = "ksponly"
    poisson.solve()

    fields = tuple(mesh.vars)
    bottom = poisson.boundary_flux_integral("Bottom")
    top = poisson.boundary_flux_integral("Top")
    np.testing.assert_allclose([bottom, top], [-2.5, 2.5], rtol=0, atol=1e-8)
    assert tuple(mesh.vars) == fields
    assert abs(bottom + top) < 1e-8
    with pytest.raises(ValueError, match="Unknown boundary"):
        poisson.boundary_flux_integral("NotABoundary")
    uw.pprint(f"DIRECT_FLUX dim={dim} degree={degree} ranks={uw.mpi.size} "
              f"bottom={bottom:.12g} top={top:.12g}")


@pytest.mark.level_1
def test_vector_rejected_before_reaction_assembly():
    from underworld3.utilities.boundary_flux import boundary_flux_integral

    # No assembly method: validation must reject the vector before attempting it.
    section = SimpleNamespace(getFieldComponents=lambda field: 2)
    solver = SimpleNamespace(dm=SimpleNamespace(getLocalSection=lambda: section))
    with pytest.raises(ValueError, match="requires a scalar solver field"):
        boundary_flux_integral(solver, "Top")
