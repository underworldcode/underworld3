import underworld3 as uw
import numpy as np
import pytest
from sympy import Piecewise
import sympy 


# ### Set up variables of the model
res = 20

# ### Set up the mesh
minX, maxX = -1.0, 0.0
minY, maxY = -1.0, 0.0

### Quads
meshStructuredQuadBox = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)), minCoords=(minX, minY), maxCoords=(maxX, maxY), qdegree=2,
)

unstructured_quad_box_irregular = uw.meshing.UnstructuredSimplexBox(minCoords=(minX, minY), maxCoords=(maxX, maxY), 
                                                                    cellSize=1/res, qdegree=2, regular=True)

unstructured_quad_box_regular = uw.meshing.UnstructuredSimplexBox(minCoords=(minX, minY), maxCoords=(maxX, maxY), 
                                                                    cellSize=1/res, qdegree=2, regular=False)

@pytest.mark.parametrize(
    "mesh",
    [
        meshStructuredQuadBox,
        unstructured_quad_box_irregular,
        unstructured_quad_box_regular,
    ],
)


def test_Darcy_boxmesh_noG(mesh):
    print(f"Mesh - Coordinates: {mesh.CoordinateSystem.type}")



    p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)
    v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)

    # x and y coordinates
    x = mesh.N.x
    y = mesh.N.y

    # #### Set up the Darcy solver
    darcy = uw.systems.SteadyStateDarcy(mesh, p_soln, v_soln)
    darcy.petsc_options.delValue("ksp_monitor")
    darcy.petsc_options["snes_rtol"] = 1.0e-6  # Needs to be smaller than the contrast in properties
    darcy.constitutive_model = uw.constitutive_models.DiffusionModel


    # #### Set up the hydraulic conductivity layout 
    ### Groundwater pressure boundary condition on the bottom wall

    max_pressure = 0.5
    initialPressure = -1.0 * y * max_pressure

    # +
    # set up two materials

    interfaceY = -0.25


    k1 = 1.0
    k2 = 1.0e-4


    #### The piecewise version
    kFunc = Piecewise((k1, y >= interfaceY), (k2, y < interfaceY), (1.0, True))
    


    darcy.constitutive_model.Parameters.diffusivity=kFunc


    darcy.f = 0.0

    darcy.s = sympy.Matrix([0, 0]).T


    # set up boundary conditions
    darcy.add_dirichlet_bc([0.0], "Top")
    darcy.add_dirichlet_bc([-1.0 * minY * max_pressure], "Bottom")

    # Zero pressure gradient at sides / base (implied bc)

    darcy._v_projector.petsc_options["snes_rtol"] = 1.0e-6
    darcy._v_projector.smoothing = 1.0e-6
    darcy._v_projector.add_dirichlet_bc(0.0, "Left",  [0])
    darcy._v_projector.add_dirichlet_bc(0.0, "Right", [0])

    # Solve darcy
    darcy.solve()

    # set up interpolation coordinates
    ycoords = np.linspace(minY + 0.01 * (maxY - minY), maxY - 0.01 * (maxY - minY), 100)
    xcoords = np.full_like(ycoords, -1)
    xy_coords = np.column_stack([xcoords, ycoords])

    pressure_interp = uw.function.evaluate(p_soln.sym[0], xy_coords)
    
    # #### Get analytical solution
    La = -1.0 * interfaceY
    Lb = 1.0 + interfaceY
    dP = max_pressure

    S = 1
    Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
    pressure_analytic = np.piecewise(
        ycoords,
        [ycoords >= -La, ycoords < -La],
        [lambda ycoords: -Pa * ycoords / La, lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb],
    )

    S = 0
    Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
    pressure_analytic_noG = np.piecewise(
        ycoords,
        [ycoords >= -La, ycoords < -La],
        [lambda ycoords: -Pa * ycoords / La, lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb],
    )
    # -

    # ### Compare analytical and numerical solution
    assert np.allclose(pressure_analytic_noG, pressure_interp, atol=1e-2)


def test_Darcy_boxmesh_G(mesh):
    print(f"Mesh - Coordinates: {mesh.CoordinateSystem.type}")



    p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)
    v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)

    # x and y coordinates
    x = mesh.N.x
    y = mesh.N.y

    minX, maxX = 0, 1
    minY, maxY = 0, 1

    # #### Set up the Darcy solver
    darcy = uw.systems.SteadyStateDarcy(mesh, p_soln, v_soln)
    darcy.petsc_options.delValue("ksp_monitor")
    darcy.petsc_options["snes_rtol"] = 1.0e-6  # Needs to be smaller than the contrast in properties
    darcy.constitutive_model = uw.constitutive_models.DiffusionModel


    # #### Set up the hydraulic conductivity layout 
    ### Groundwater pressure boundary condition on the bottom wall

    max_pressure = 0.5
    initialPressure = -1.0 * y * max_pressure

    # +
    # set up two materials

    interfaceY = -0.25

    k1 = 1.0
    k2 = 1.0e-4


    #### The piecewise version
    kFunc = Piecewise((k1, y >= interfaceY), (k2, y < interfaceY), (1.0, True))
    


    darcy.constitutive_model.Parameters.diffusivity=kFunc

    ### add bodyforce term
    darcy.s = sympy.Matrix([0, -1]).T


    # set up boundary conditions
    darcy.add_dirichlet_bc([0.0], "Top")
    darcy.add_dirichlet_bc([-1.0 * minY * max_pressure], "Bottom")

    # Zero pressure gradient at sides / base (implied bc)

    darcy._v_projector.petsc_options["snes_rtol"] = 1.0e-6
    darcy._v_projector.smoothing = 1.0e-6
    darcy._v_projector.add_dirichlet_bc(0.0, "Left",  [0])
    darcy._v_projector.add_dirichlet_bc(0.0, "Right", [0])
# -

    # Solve darcy
    darcy.solve()

    # set up interpolation coordinates
    ycoords = np.linspace(minY + 0.01 * (maxY - minY), maxY - 0.01 * (maxY - minY), 100)
    xcoords = np.full_like(ycoords, -1)
    xy_coords = np.column_stack([xcoords, ycoords])

    pressure_interp = uw.function.evaluate(p_soln.sym[0], xy_coords)
    
    # #### Get analytical solution
    La = -1.0 * interfaceY
    Lb = 1.0 + interfaceY
    dP = max_pressure

    S = 1
    Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
    pressure_analytic = np.piecewise(
        ycoords,
        [ycoords >= -La, ycoords < -La],
        [lambda ycoords: -Pa * ycoords / La, lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb],
    )

    S = 0
    Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
    pressure_analytic_noG = np.piecewise(
        ycoords,
        [ycoords >= -La, ycoords < -La],
        [lambda ycoords: -Pa * ycoords / La, lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb],
    )
    # -

    # ### Compare analytical and numerical solution
    assert np.allclose(pressure_analytic, pressure_interp, atol=1e-2)


