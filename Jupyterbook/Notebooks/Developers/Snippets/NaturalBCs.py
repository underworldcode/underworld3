import sympy
import underworld3 as uw
import underworld3.maths
import numpy as np
import os 

os.environ["UW_TIMING_ENABLE"] = "1"
# os.environ["UW_JITNAME"] = "TEST_PF"

# #### These are tested by test_001_meshes.py

# +
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(2,2), minCoords=(0.0,0.0), maxCoords=(1.0,1.0),
    qdegree=4, refinement=4,
)

# mesh = uw.meshing.UnstructuredSimplexBox(
#     minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 0.75
# )
# -


with uw.utilities.CaptureStdout(split=False) as captured_dm_view:
    mesh.dm.view()

captured_dm_view.splitlines()


if mesh.dim == 2:
    x, y = mesh.X
else:
    x, y, z = mesh.X


u = uw.discretisation.MeshVariable("U", mesh, mesh.dim, 
                                   vtype=uw.VarType.VECTOR, 
                                   degree=2, varsymbol=r"\mathbf{u}",
)
p = uw.discretisation.MeshVariable("P",
    mesh, 1, vtype=uw.VarType.SCALAR, degree=1, continuous=True, 
                                   varsymbol=r"\mathbf{p}", 
)

mesh.qdegree

hw =  mesh.get_min_radius() / (mesh.qdegree + 1)
surface_fn = 2 * uw.maths.delta_function(y-1, hw) # / uw.maths.delta_function(0.0, hw)
surface_fn = sympy.Piecewise((1.0,1-y < hw), (0.0, True))
I = uw.maths.Integral(mesh, surface_fn)
norm = I.evaluate()


norm

# +
# uw.function.evalf(surface_fn, np.array([[0.0,0.9]]))
# -

stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(u)
stokes.constitutive_model.Parameters.viscosity=1

if mesh.dim == 2:
    stokes.bodyforce = sympy.Matrix([0.0, 1.0*sympy.sin(x)])

    stokes.add_dirichlet_bc((0.0, 0.0), "Top")
    stokes.add_dirichlet_bc((sympy.oo,0.0), "Bottom")
    stokes.add_dirichlet_bc((0.0,sympy.oo), "Left")
    stokes.add_dirichlet_bc((0.0,sympy.oo), "Right")


stokes.petsc_options["snes_monitor"]= None
stokes.petsc_options["ksp_monitor"] = None

# +
stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# gasm is super-fast ... but mg seems to be bulletproof
# gamg is toughest wrt viscosity

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# -

eta_0 = 1.0
x_c = 0.5
f_0 = 1.0

stokes.penalty = 0.0
stokes.bodyforce = sympy.Matrix([[0.0, (1-y)*(1-x)]])


converged = stokes.solve(verbose=False, debug=False)


# +
stokes.essential_bcs = []
stokes.natural_bcs = []

stokes.bodyforce[1] -= 1.0e6 * u.sym[1] * (surface_fn)

stokes.dm = None

# stokes.add_natural_bc((0.0, pen * u[1].sym), "Top")
stokes.add_dirichlet_bc((sympy.oo,0.0), "Bottom")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Right")

converged = stokes.solve(zero_init_guess=False, picard=0, verbose=False, debug=True, _force_setup=True)

# -


surface_fn.subs(y,1).evalf()

# Residual term
#
# $$\int_\Gamma \phi {\vec f}_0(u, u_t, \nabla u, x, t) \cdot \hat n + \nabla\phi \cdot {\overleftrightarrow f}_1(u, u_t, \nabla u, x, t) \cdot \hat n$$
#
#
# Jacobian term
#
# $$\int_\Gamma \phi {\vec g}_0(u, u_t, \nabla u, x, t) \cdot \hat n \psi + \phi {\vec g}_1(u, u_t, \nabla u, x, t) \cdot \hat n \nabla \psi + \nabla\phi \cdot {\vec g}_2(u, u_t, \nabla u, x, t) \cdot \hat n \psi + \nabla\phi \cdot {\overleftrightarrow g}_3(u, u_t, \nabla u, x, t) \cdot \hat n \cdot \nabla \psi$$
#
#

import mpi4py

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    V = u

    pvmesh.point_data["P"] = uw.function.evalf(stokes.p.sym[0], mesh.data)
    pvmesh.point_data["V"] = uw.function.evalf(V.sym.dot(V.sym), mesh.data)
    pvmesh.point_data["S"] = uw.function.evalf(surface_fn, mesh.data)

    arrow_loc = np.zeros((V.coords.shape[0], 3))
    arrow_loc[:, 0:2] = V.coords[...]

    V_anti_coords = (1.0, 1.0) - V.coords

    arrow_diff = np.zeros((V.coords.shape[0], 3))
    arrow_diff[:, 0] = uw.function.evalf(V.sym[0], V.coords) + uw.function.evalf(V.sym[0], V_anti_coords)
    arrow_diff[:, 1] = uw.function.evalf(V.sym[1], V.coords) + uw.function.evalf(V.sym[1], V_anti_coords)

    arrow_length = np.zeros((V.coords.shape[0], 3))
    arrow_length[:, 0] = uw.function.evalf(V.sym[0], V.coords) 
    arrow_length[:, 1] = uw.function.evalf(V.sym[1], V.coords) 


    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="V",
        use_transparency=False,
        opacity=1.0,
    )

    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
    #               use_transparency=False, opacity=1.0)

    pl.add_arrows(arrow_loc, arrow_length, mag=10)

    pl.show(cpos="xy")

(1.0, 1.0) - V.coords


V.coords

stokes.natural_bcs[0].fns["uu_G0"]

# ##### 
