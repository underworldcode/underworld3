import pytest
from pyvista import cell_array
import sympy
import underworld3 as uw

# These are tested by test_001_meshes.py

structured_quad_box = uw.meshing.StructuredQuadBox(elementRes=(5,) * 2)
unstructured_quad_box_irregular = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, regular=False)
unstructured_quad_box_regular = uw.meshing.UnstructuredSimplexBox(cellSize=0.1, regular=True)
unstructured_quad_box_irregular_3D = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.33, regular=False
)


spherical_shell = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5, cellSize=0.33)
annulus = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.1)
cubed_sphere = uw.meshing.CubedSphere(radiusOuter=1.0, radiusInner=0.5, numElements=3)


# Maybe lower and upper would work better for the names of the box mesh boundaries too.


@pytest.mark.parametrize(
    "mesh",
    [
        structured_quad_box,
        unstructured_quad_box_irregular,
        unstructured_quad_box_regular,
        unstructured_quad_box_irregular_3D,
    ],
)
def test_stokes_boxmesh(mesh):

    if mesh.dim == 2:
        x, y = mesh.X
    else:
        x, y, z = mesh.X

    u = uw.discretisation.MeshVariable(r"mathbf{u}", mesh, mesh.dim, vtype=uw.VarType.VECTOR, degree=2)
    p = uw.discretisation.MeshVariable(r"mathbf{p}", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
    stokes.constitutive_model.material_properties = stokes.constitutive_model.Parameters(viscosity=1)

    if mesh.dim == 2:
        stokes.bodyforce = sympy.Matrix([0, x])
    else:
        stokes.bodyforce = sympy.Matrix([0, x, 0])

    stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Bottom", (0, 1, 2))
    stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Top", (0, 1, 2))
    stokes.solve()

    return


@pytest.mark.parametrize("mesh", [spherical_shell, annulus, cubed_sphere])
def test_stokes_sphere(mesh):
    if mesh.dim == 2:
        x, y = mesh.X
    else:
        x, y, z = mesh.X

    u = uw.discretisation.MeshVariable(r"mathbf{u}", mesh, mesh.dim, vtype=uw.VarType.VECTOR, degree=2)
    p = uw.discretisation.MeshVariable(r"mathbf{p}", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
    stokes.constitutive_model.material_properties = stokes.constitutive_model.Parameters(viscosity=1)

    if mesh.dim == 2:
        stokes.bodyforce = sympy.Matrix([0, x])
    else:
        stokes.bodyforce = sympy.Matrix([0, x, 0])

    stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Lower", (0, 1, 2))
    stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Upper", (0, 1, 2))
    stokes.solve()

    return
