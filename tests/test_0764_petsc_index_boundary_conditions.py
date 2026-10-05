"""Boundary registration must preserve PETSc's configured integer width.

The exact Couette solution is u=(y, 0), p=constant on the unit square.
Exercise both inferred and legacy explicit component lists, including [0, 1],
which a 64-bit PETSc build misread as 4294967296 from a 32-bit buffer.
"""

import numpy as np
import pytest
from petsc4py import PETSc

import underworld3 as uw

pytestmark = pytest.mark.level_1


@pytest.mark.tier_a
@pytest.mark.parametrize("explicit_components", [False, True])
def test_couette_boundary_components_use_petsc_integer_width(explicit_components):
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), qdegree=3)
    velocity = uw.discretisation.MeshVariable("index_velocity", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("index_pressure", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1
    stokes.bodyforce = (0, 0)
    y = mesh.N.y
    for boundary in ("Top", "Bottom", "Left", "Right"):
        if explicit_components:
            with pytest.warns(DeprecationWarning, match="components"):
                stokes.add_dirichlet_bc((y, 0), boundary, (0, 1))
        else:
            stokes.add_dirichlet_bc((y, 0), boundary)

    for bc in stokes.essential_bcs:
        assert bc.components.dtype == np.dtype(PETSc.IntType)
        np.testing.assert_array_equal(bc.components, [0, 1])

    stokes.tolerance = 1e-9
    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0
    coords = velocity.coords
    expected = np.column_stack((coords[:, 1], np.zeros(len(coords))))
    np.testing.assert_allclose(velocity.array[:, 0, :], expected, rtol=1e-7, atol=1e-8)
