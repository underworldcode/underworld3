import pytest
import sympy
import underworld3 as uw

# These are tested by test_001_meshes.py

structured_quad_box = uw.meshing.StructuredQuadBox(elementRes=(5,) * 2)

unstructured_quad_box_irregular = uw.meshing.UnstructuredSimplexBox(
    cellSize=0.1, regular=False, qdegree=2
)
unstructured_quad_box_regular = uw.meshing.UnstructuredSimplexBox(
    cellSize=0.1, regular=True, qdegree=2
)

unstructured_quad_box_irregular_3D = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0, 0.0),
    maxCoords=(1.0, 1.0, 1.0),
    cellSize=0.5,
    regular=False,
    qdegree=2,
)

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

    print(f"Mesh - Coordinates: {mesh.CoordinateSystem.type}")
    mesh.dm.view()

    if mesh.dim == 2:
        x, y = mesh.X
    else:
        x, y, z = mesh.X

    u = uw.discretisation.MeshVariable(
        r"mathbf{u}", mesh, mesh.dim, vtype=uw.VarType.VECTOR, degree=2
    )
    p = uw.discretisation.MeshVariable(
        r"mathbf{p}", mesh, 1, vtype=uw.VarType.SCALAR, degree=1
    )

    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(
        mesh.dim
    )
    stokes.constitutive_model.Parameters.viscosity = 1

    if mesh.dim == 2:
        stokes.bodyforce = sympy.Matrix([0, x])
        stokes.add_dirichlet_bc((0.0, 0.0), "Bottom", (0, 1))
        stokes.add_dirichlet_bc((0.0, 0.0), "Top", (0, 1))
        stokes.add_dirichlet_bc(
            (0.0,), ["Left", "Right"], 0
        )  # left/right: components, function, markers
    else:
        stokes.bodyforce = sympy.Matrix([0, x, 0])
        stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Bottom", (0, 1, 2))
        stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Top", (0, 1, 2))
        stokes.add_dirichlet_bc((0.0,), ["Left", "Right"], (0,))
        stokes.add_dirichlet_bc((0.0,), ["Front", "Back"], (2,))

    stokes.solve()

    assert stokes.snes.getConvergedReason() > 0

    del mesh
    del stokes

    return


del structured_quad_box
del unstructured_quad_box_regular
del unstructured_quad_box_irregular
del unstructured_quad_box_irregular_3D
