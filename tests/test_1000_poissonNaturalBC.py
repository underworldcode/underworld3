import pytest
import underworld3 as uw
import sympy
import numpy as np

"""
Unit test for Natural BCs in a Poisson (scalar) problem.

Tests that natural (flux) boundary conditions work correctly by using:
- Dirichlet BCs on Top/Bottom (to pin the solution)
- Natural BCs on Left/Right (to test flux specification)

Analytical solution: T = x² * y
- ∇²T = 2y, so source f = -2y
- dT/dx = 2xy (flux in x-direction)
- dT/dy = x² (flux in y-direction)

Boundary conditions:
- Bottom (y=0): Dirichlet T = 0
- Top (y=1): Dirichlet T = x²
- Left (x=0): Natural flux dT/dx = 0
- Right (x=1): Natural flux dT/dx = 2y
"""

# Physics solver tests - full solver execution
pytestmark = pytest.mark.level_3

res = 16

width = 1
height = 1

minX, maxX = 0, width
minY, maxY = 0, height

mesh_simp_reg = uw.meshing.UnstructuredSimplexBox(
    minCoords=(minX, minY),
    maxCoords=(maxX, maxY),
    cellSize=1 / res,
    qdegree=3,
    regular=True,
)
mesh_simp_irreg = uw.meshing.UnstructuredSimplexBox(
    minCoords=(minX, minY),
    maxCoords=(maxX, maxY),
    cellSize=1 / res,
    regular=False,
)
mesh_quad = uw.meshing.StructuredQuadBox(
    minCoords=(minX, minY),
    maxCoords=(maxX, maxY),
    elementRes=(res, res),
)


@pytest.mark.skip(reason="Poisson natural BC setup failing - needs solver investigation (PETSc error 73: Object in wrong state)")
@pytest.mark.parametrize("mesh", [mesh_simp_reg, mesh_simp_irreg, mesh_quad])
def test_poisson_natural_bc(mesh):
    """Test Poisson solver with natural (flux) boundary conditions."""

    T_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

    poisson = uw.systems.Poisson(
        mesh=mesh,
        u_Field=T_soln,
        degree=2,
        verbose=False,
    )

    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.petsc_options.delValue("ksp_monitor")

    # Analytical solution: T = x² * y
    x, y = mesh.N.x, mesh.N.y
    ana_soln = (x**2) * y

    # Source term: f = -∇²T = -2y
    poisson.f = -2 * y

    # Dirichlet BCs on Top/Bottom (pin the solution in y-direction)
    poisson.add_dirichlet_bc(0.0, "Bottom")  # T(y=0) = 0
    poisson.add_dirichlet_bc([x**2], "Top")  # T(y=1) = x²

    # Natural (flux) BCs on Left/Right (test flux specification)
    # Note: Natural BC specifies the outward flux = -k * dT/dn
    # For diffusivity k=1 and outward normal:
    #   Left (n = [-1,0]): flux = -(-dT/dx) = dT/dx = 2xy = 0 at x=0
    #   Right (n = [1,0]): flux = -dT/dx = -2xy = -2y at x=1
    poisson.add_natural_bc(0.0, "Left")  # Zero flux at x=0
    poisson.add_natural_bc([-2 * y], "Right")  # Outward flux = -2y at x=1

    poisson.solve()

    # Verify convergence
    assert poisson.snes.getConvergedReason() > 0, "Solver did not converge"

    # Compare numerical to analytical solution
    num = T_soln.data[:].squeeze()
    ana = uw.function.evaluate(ana_soln, T_soln.coords).squeeze()

    assert np.allclose(ana, num, atol=1e-4), "Numerical and analytical solutions differ!"

    del poisson
    del mesh
