"""NavierStokes sizes its velocity-shaped fields by the velocity, not mesh.dim (#872).

On a surface mesh (dim 2 in three-space) the velocity has ``cdim`` components.
The SUPG solver built its advecting-velocity and previous-level fields, the
body-force row, the pressure gradient and the speed estimates with
``mesh.dim``, so they had one component too few. EulerianSUPG and the
Eulerian AdvDiffusion were fixed the same way in the thin-sheet work. A
Stokes-family solve on a surface mesh is not supported yet (it fails in the
saddle-point base), so this checks the shapes the solver builds, which are
its own.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1]


@pytest.mark.tier_a
def test_velocity_shaped_fields_and_residual_rows_follow_the_embedding_dimension():
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.4)
    assert (mesh.dim, mesh.cdim) == (2, 3)
    v = uw.discretisation.MeshVariable("v872", mesh, mesh.cdim, degree=2)
    p = uw.discretisation.MeshVariable("p872", mesh, 1, degree=1)
    ns = uw.systems.NavierStokes(mesh, velocityField=v, pressureField=p, rho=1.0)
    assert ns._a_var.num_components == v.num_components == 3
    assert ns._u_prev.num_components == 3
    x, y, z = mesh.X
    ns.bodyforce = sympy.Matrix([[-y, x, 0]])
    # the strong residual SUPG reads: body force and pressure gradient as cdim rows
    assert ns._strong_residual(with_pressure=True).shape == (1, 3)


@pytest.mark.tier_a
def test_a_flat_mesh_is_unchanged():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25)
    v = uw.discretisation.MeshVariable("vflat", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("pflat", mesh, 1, degree=1)
    ns = uw.systems.NavierStokes(mesh, velocityField=v, pressureField=p, rho=1.0)
    assert ns._a_var.num_components == ns._u_prev.num_components == 2
