# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Stokes (Cartesian formulation) in Elliptical Domain
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

import os
os.environ["UW_TIMING_ENABLE"] = "1"

# +

free_slip_upper = True
free_slip_lower = True

# Define the problem size
#      1 - ultra low res for automatic checking
#      2 - low res problem to play with this notebook
#      3 - medium resolution (be prepared to wait)
#      4 - highest resolution (benchmark case from Spiegelman et al)

problem_size = 2

# For testing and automatic generation of notebook output,
# over-ride the problem size if the UW_TESTING_LEVEL is set

uw_testing_level = os.environ.get("UW_TESTING_LEVEL")
if uw_testing_level:
    try:
        problem_size = int(uw_testing_level)
    except ValueError:
        # Accept the default value
        pass

r_o = 1.0
r_i = 0.5

if problem_size <= 1:
    res = 0.5
elif problem_size == 2:
    res = 0.1
elif problem_size == 3:
    res = 0.05
elif problem_size == 4:
    res = 0.025
elif problem_size == 5:
    res = 0.01
elif problem_size >= 6:
    res = 0.005


cellSizeOuter = res
cellSizeInner = res/2
ellipticityOuter = 1.5
ellipticityInner = 1.0

radiusOuter = r_o
radiusInner = r_i


# +
from enum import Enum

class boundaries(Enum):
    Inner = 1
    Outer = 2

if uw.mpi.rank == 0:
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("Annulus")

    p0 = gmsh.model.geo.add_point(0.00, 0.00, 0.00, meshSize=cellSizeInner)

    loops = []

    p1 = gmsh.model.geo.add_point(radiusInner*ellipticityInner, 0.0, 0.0, meshSize=cellSizeInner)
    p2 = gmsh.model.geo.add_point(0.0, radiusInner, 0.0, meshSize=cellSizeInner)
    p3 = gmsh.model.geo.add_point(-radiusInner*ellipticityInner, 0.0, 0.0, meshSize=cellSizeInner)
    p4 = gmsh.model.geo.add_point(0.0, -radiusInner, 0.0, meshSize=cellSizeInner)
        
    c1 = gmsh.model.geo.add_ellipse_arc(p1, p0, p1, p2)
    c2 = gmsh.model.geo.add_ellipse_arc(p2, p0, p3, p3)
    c3 = gmsh.model.geo.add_ellipse_arc(p3, p0, p3, p4)
    c4 = gmsh.model.geo.add_ellipse_arc(p4, p0, p1, p1)

    cl1 = gmsh.model.geo.add_curve_loop([c1, c2, c3, c4], tag=boundaries.Inner.value)

    loops = [cl1] + loops

    p5 = gmsh.model.geo.add_point(radiusOuter*ellipticityOuter, 0.0, 0.0, meshSize=cellSizeOuter)
    p6 = gmsh.model.geo.add_point(0.0, radiusOuter, 0.0, meshSize=cellSizeOuter)
    p7 = gmsh.model.geo.add_point(-radiusOuter*ellipticityOuter, 0.0, 0.0, meshSize=cellSizeOuter)
    p8 = gmsh.model.geo.add_point(0.0, -radiusOuter, 0.0, meshSize=cellSizeOuter)
        
    c5 = gmsh.model.geo.add_ellipse_arc(p5, p0, p5, p6)
    c6 = gmsh.model.geo.add_ellipse_arc(p6, p0, p7, p7)
    c7 = gmsh.model.geo.add_ellipse_arc(p7, p0, p7, p8)
    c8 = gmsh.model.geo.add_ellipse_arc(p8, p0, p5, p5)

    # l1 = gmsh.model.geo.add_line(p5, p4)

    cl2 = gmsh.model.geo.add_curve_loop([c5, c6, c7, c8], tag=boundaries.Outer.value)

    loops = [cl2] + loops

    s = gmsh.model.geo.add_plane_surface(loops)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(0, [p0], 2, s)

    gmsh.model.addPhysicalGroup(
        1,
        [c1, c2, c3, c4],
        boundaries.Inner.value,
        name=boundaries.Inner.name,
    )

    gmsh.model.addPhysicalGroup(
        1, [c5, c6, c7, c8], 
        boundaries.Outer.value, 
        name=boundaries.Outer.name
    )
    gmsh.model.addPhysicalGroup(2, [s], 666666, "Elements")

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write("tmp_elliptical_mesh.msh")
    gmsh.finalize()

elliptical_mesh = uw.discretisation.Mesh(
        "tmp_elliptical_mesh.msh",
        degree=1,
        qdegree=3,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
        boundaries=boundaries,
        coordinate_system_type=None,
        refinement=0,
        refinement_callback=None,
        return_coords_to_bounds=None,
    )

x,y = elliptical_mesh.X

# -


x

# +
# Analytic expression for surface normals

Gamma_N_Outer = sympy.Matrix([2 * x / ellipticityOuter**2, 2 * y ]).T
Gamma_N_Outer = Gamma_N_Outer / sympy.sqrt(Gamma_N_Outer.dot(Gamma_N_Outer))
Gamma_N_Inner = sympy.Matrix([2 * x / ellipticityInner**2, 2 * y ]).T
Gamma_N_Inner = Gamma_N_Inner / sympy.sqrt(Gamma_N_Inner.dot(Gamma_N_Inner))


# +
# check the mesh if in a notebook / serial and look at the surface normals (check them)

if 1 and uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(elliptical_mesh)

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, opacity=0.5)

    pvmesh.point_data["Gamma"] = vis.vector_fn_to_pv_points(pvmesh, Gamma_N_Outer, dim=2)
    pvmesh.point_data["Gamma2"] = vis.vector_fn_to_pv_points(pvmesh, Gamma_N_Inner, dim=2)

    pl.add_arrows(pvmesh.points,pvmesh.point_data["Gamma"], mag=0.1, color="Red")
    pl.add_arrows(pvmesh.points,pvmesh.point_data["Gamma2"], mag=0.1, color="Blue")

    pl.show()

# +
# Test that the second one is skipped

v_soln = uw.discretisation.MeshVariable(r"U", elliptical_mesh, 2, degree=2, continuous=True, varsymbol=r"\mathbf{u}")
p_soln = uw.discretisation.MeshVariable(r"P", elliptical_mesh, 1, degree=1, continuous=True, varsymbol=r"\mathbf{p}")
t_soln = uw.discretisation.MeshVariable(r"T", elliptical_mesh, 1, degree=3, varsymbol="\Delta T")

# -


v_soln.sym

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

x, y = elliptical_mesh.CoordinateSystem.X

radius_fn = sympy.sqrt(x**2+y**2)
unit_rvec = elliptical_mesh.CoordinateSystem.X / radius_fn
gravity_fn = 1  # radius_fn / r_o

# Some useful coordinate stuff

Rayleigh = 1.0e5

hw = 10000.0 / res
# -


elliptical_mesh.Gamma

# +
# Create Stokes object

stokes = Stokes(elliptical_mesh, velocityField=v_soln, pressureField=p_soln, solver_name="stokes")
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
stokes.penalty = 1.0

# Surface normals provided by DMPLEX

Gamma = elliptical_mesh.Gamma
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Outer")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Inner")

# Or, use the analytic version instead

# stokes.add_natural_bc(10000 * Gamma_N_Outer.dot(v_soln.sym) *  Gamma_N_Outer, "Outer")
# stokes.add_natural_bc(10000 * Gamma_N_Inner.dot(v_soln.sym) *  Gamma_N_Inner, "Inner")
    
stokes.saddle_preconditioner = sympy.simplify(1 / (stokes.constitutive_model.viscosity + stokes.penalty))



# +

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# gasm is super-fast ... but mg seems to be bulletproof
# gamg is toughest wrt viscosity

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "kaskade")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")
# -


stokes._setup_pointwise_functions(verbose=False)
stokes._setup_discretisation(verbose=False)

# t_init = 10.0 * sympy.exp(-5.0 * (x**2 + (y - 0.5) ** 2))
t_init = sympy.cos(4 * sympy.atan2(y,x))

# +
# Write density into a variable for saving

with elliptical_mesh.access(t_soln):
    t_soln.data[:, 0] = uw.function.evaluate(
        t_init, coords=t_soln.coords, coord_sys=elliptical_mesh.N
    )


# +

buoyancy_force = Rayleigh * gravity_fn * t_init

stokes.bodyforce = unit_rvec * buoyancy_force
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_monitor"] = None
stokes.tolerance = 1.0e-4
# -


buoyancy_force

# +
from underworld3 import timing

timing.reset()
timing.start()
# +
stokes.solve(zero_init_guess=True, debug=False)

timing.print_table()
# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(elliptical_mesh)
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_init)

    points = np.zeros((elliptical_mesh._centroids.shape[0], 3))
    points[:, 0] = elliptical_mesh._centroids[:, 0]
    points[:, 1] = elliptical_mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", 
        integration_direction="forward", 
        integrator_type=2,
        surface_streamlines=True,
        initial_step_length=0.01,
        max_time=1.0,
        max_steps=2000
    )
    
    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
                pvmesh,
                cmap="coolwarm",
                edge_color="Black",
                scalars="T",
                show_edges=True,
                use_transparency=False,
                opacity=0.75,
               )
    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=1e-4)
    
    pl.add_mesh(pvstream)

    pl.show(cpos="xy")
# -

# From the `PETSc` docs, the form of the boundary integral (residual, jacobian, preconditioner) and the form of the interior integrals
#
# ## Neumann terms (boundary integrals)
#
# Boundary integral in mathematical form.
#
# $$\int_\Gamma \phi {\vec f}_0(u, u_t, \nabla u, x, t) \cdot \hat n + \nabla\phi \cdot {\overleftrightarrow f}_1(u, u_t, \nabla u, x, t) \cdot \hat n$$
#
#
# ## Interior integrals
#
# $$\int_\Omega \phi f_0(u, u_t, \nabla u, x, t) + \nabla\phi \cdot {\vec f}_1(u, u_t, \nabla u, x, t)$$
#
#
#


