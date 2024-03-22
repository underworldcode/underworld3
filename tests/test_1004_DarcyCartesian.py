import underworld3 as uw
import numpy as np
import pytest
from sympy import Piecewise
import sympy


# ### Set up variables of the model
res = 15

# ### Set up the mesh
minX, maxX = -1.0, 0.0
minY, maxY = -1.0, 0.0

# +
### Quads
meshStructuredQuadBox = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)),
    minCoords=(minX, minY),
    maxCoords=(maxX, maxY),
    qdegree=2,
)

### Tris
meshSimplex_box_irregular = uw.meshing.UnstructuredSimplexBox(
    minCoords=(minX, minY),
    maxCoords=(maxX, maxY),
    cellSize=1 / res,
    qdegree=2,
    regular=True,
)

meshSimplex_box_regular = uw.meshing.UnstructuredSimplexBox(
    minCoords=(minX, minY),
    maxCoords=(maxX, maxY),
    cellSize=1 / res,
    qdegree=2,
    regular=False,
)


# +
@pytest.mark.parametrize(
    "mesh",
    [
        meshStructuredQuadBox,
        meshSimplex_box_irregular,
        meshSimplex_box_regular,
    ],
)
def test_Darcy_boxmesh_G_and_noG(mesh):
    # Reset the mesh if it still has things lying around from earlier tests
    mesh.dm.clearDS()
    mesh.dm.clearFields()
    mesh.nuke_coords_and_rebuild()
    mesh.dm.createDS()

    p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)
    v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)

    # x and y coordinates
    x = mesh.N.x
    y = mesh.N.y

    # #### Set up the Darcy solver
    darcy = uw.systems.SteadyStateDarcy(mesh, p_soln, v_soln)
    darcy.petsc_options.delValue("ksp_monitor")
    darcy.petsc_options["snes_rtol"] = (
        1.0e-6  # Needs to be smaller than the contrast in properties
    )
    darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel

    # #### Set up the hydraulic conductivity layout
    ### Groundwater pressure boundary condition on the bottom wall

    max_pressure = 0.5

    # +
    # set up two materials

    interfaceY = -0.25

    k1 = 1.0
    k2 = 1.0e-4

    #### The piecewise version
    kFunc = Piecewise((k1, y >= interfaceY), (k2, y < interfaceY), (1.0, True))

    darcy.constitutive_model.Parameters.permeability = kFunc
    darcy.constitutive_model.Parameters.s = sympy.Matrix([0, 0]).T
    darcy.f = 0.0

    # set up boundary conditions
    darcy.add_dirichlet_bc([0.0], "Top")
    darcy.add_dirichlet_bc([-1.0 * minY * max_pressure], "Bottom")

    # Zero pressure gradient at sides / base (implied bc)

    darcy._v_projector.petsc_options["snes_rtol"] = 1.0e-6
    darcy._v_projector.smoothing = 1.0e-6
    # darcy._v_projector.add_dirichlet_bc(0.0, "Left", [0])
    # darcy._v_projector.add_dirichlet_bc(0.0, "Right", [0])

    # Solve darcy
    darcy.solve(verbose=True)

    # set up interpolation coordinates
    ycoords = np.linspace(minY + 0.01 * (maxY - minY), maxY - 0.01 * (maxY - minY), 100)
    xcoords = np.full_like(ycoords, -0.5)
    xy_coords = np.column_stack([xcoords, ycoords])

    pressure_interp = uw.function.evaluate(p_soln.sym[0], xy_coords)

    # #### Get analytical solution
    La = -1.0 * interfaceY
    Lb = 1.0 + interfaceY
    dP = max_pressure

    S = 0
    Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
    pressure_analytic_noG = np.piecewise(
        ycoords,
        [ycoords >= -La, ycoords < -La],
        [
            lambda ycoords: -Pa * ycoords / La,
            lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb,
        ],
    )

    # print(pressure_analytic_noG)
    # print(pressure_interp)

    # ### Compare analytical and numerical solution
    assert np.allclose(pressure_analytic_noG, pressure_interp, atol=3e-2)

    ## Suggest we re-solve right here for version with G to avoid all the re-definitions

    S = 1
    Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
    pressure_analytic = np.piecewise(
        ycoords,
        [ycoords >= -La, ycoords < -La],
        [
            lambda ycoords: -Pa * ycoords / La,
            lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb,
        ],
    )

    darcy.constitutive_model.Parameters.s = sympy.Matrix([0, -1]).T
    darcy.solve()

    pressure_interp = uw.function.evalf(p_soln.sym[0], xy_coords)

    # ### Compare analytical and numerical solution
    assert np.allclose(pressure_analytic, pressure_interp, atol=3e-2)


#
