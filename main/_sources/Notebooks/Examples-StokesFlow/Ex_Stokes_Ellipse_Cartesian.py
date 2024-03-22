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

problem_size = 3

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


elliptical_mesh.dm.view()

# +
# Analytic expression for surface normals

Gamma_N_Outer = sympy.Matrix([2 * x / ellipticityOuter**2, 2 * y ]).T
Gamma_N_Outer = Gamma_N_Outer / sympy.sqrt(Gamma_N_Outer.dot(Gamma_N_Outer))
Gamma_N_Inner = sympy.Matrix([2 * x / ellipticityInner**2, 2 * y ]).T
Gamma_N_Inner = Gamma_N_Inner / sympy.sqrt(Gamma_N_Inner.dot(Gamma_N_Inner))


# +
# Some geometry things

x, y = elliptical_mesh.CoordinateSystem.X

radius_fn = sympy.sqrt(x**2+y**2)
unit_rvec = elliptical_mesh.CoordinateSystem.X / radius_fn

# Some useful coordinate stuff


# +
# Test that the second one is skipped

v_soln = uw.discretisation.MeshVariable("U", elliptical_mesh, 2, degree=2, continuous=True, varsymbol=r"\mathbf{u}")
v_soln_1 = uw.discretisation.MeshVariable("U1", elliptical_mesh, 2, degree=2, continuous=True, varsymbol=r"{\mathbf{u}^[1]}")
v_soln_0 = uw.discretisation.MeshVariable("U0", elliptical_mesh, 2, degree=2, continuous=True, varsymbol=r"{\mathbf{u}^[0]}")
p_soln = uw.discretisation.MeshVariable("P", elliptical_mesh, 1, degree=1, continuous=True, varsymbol=r"\mathbf{p}")



# +
n_vect = uw.discretisation.MeshVariable("Gamma", elliptical_mesh, 2, degree=2, varsymbol="{\Gamma_N}")

projection = uw.systems.Vector_Projection(elliptical_mesh, n_vect)
projection.uw_function = sympy.Matrix([[0,0]])

# r.dot(Gamma) Ensure consistent orientation (not needed for mesh boundary surfaces)

GammaNorm = unit_rvec.dot(elliptical_mesh.Gamma) / sympy.sqrt(elliptical_mesh.Gamma.dot(elliptical_mesh.Gamma))

projection.add_natural_bc(elliptical_mesh.Gamma * GammaNorm, "Outer")
projection.add_natural_bc(elliptical_mesh.Gamma * GammaNorm, "Inner")

projection.solve()

# Ensure n_vect are unit vectors 
with elliptical_mesh.access(n_vect):
    n_vect.data[:,:] /= np.sqrt(n_vect.data[:,0]**2 + n_vect.data[:,1]**2).reshape(-1,1)



# +
# Create Stokes object

stokes = Stokes(elliptical_mesh, velocityField=v_soln, pressureField=p_soln, solver_name="stokes")
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
stokes.penalty = 1.0
stokes.saddle_preconditioner = sympy.simplify(1 / (stokes.constitutive_model.viscosity + stokes.penalty))

stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

# Surface normals provided by DMPLEX

Gamma = elliptical_mesh.Gamma
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Outer")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Inner")


# +
t_init = sympy.cos(4 * sympy.atan2(y,x))

stokes.bodyforce = unit_rvec * t_init
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_monitor"] = None
stokes.tolerance = 1.0e-6


# +
from underworld3 import timing

timing.reset()
timing.start()

stokes.solve(zero_init_guess=True, debug=False)

with elliptical_mesh.access(v_soln_1):
    v_soln_1.data[...] = v_soln.data[...]

timing.print_table()
# +
# Create Stokes object

stokes._reset()

# Surface normals (computed)

stokes.add_natural_bc(10000 * n_vect.sym.dot(v_soln.sym) * n_vect.sym, "Outer")
stokes.add_natural_bc(10000 * n_vect.sym.dot(v_soln.sym) * n_vect.sym, "Inner")
    

stokes.solve(zero_init_guess=False)

with elliptical_mesh.access(v_soln_0):
    v_soln_0.data[...] = v_soln.data[...]


# +
# Create Stokes object

stokes._reset()

# Surface normals (computed analytically)
stokes.add_natural_bc(10000 * Gamma_N_Outer.dot(v_soln.sym) *  Gamma_N_Outer, "Outer")
stokes.add_natural_bc(10000 * Gamma_N_Inner.dot(v_soln.sym) *  Gamma_N_Inner, "Inner")
    
stokes.solve(zero_init_guess=False)

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(elliptical_mesh)
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["Va"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)
    velocity_points.point_data["Vn"] = vis.vector_fn_to_pv_points(velocity_points, v_soln_1.sym)
    velocity_points.point_data["Vp"] = vis.vector_fn_to_pv_points(velocity_points, v_soln_0.sym)
    velocity_points.point_data["dVn"] = velocity_points.point_data["Vn"] - velocity_points.point_data["Va"]
    velocity_points.point_data["dVp"] = velocity_points.point_data["Vp"] - velocity_points.point_data["Va"]

    
    pvmesh.point_data["Va"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vn"] = vis.vector_fn_to_pv_points(pvmesh, v_soln_0.sym)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_init)

    points = np.zeros((elliptical_mesh._centroids.shape[0], 3))
    points[:, 0] = elliptical_mesh._centroids[:, 0]
    points[:, 1] = elliptical_mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="Va", 
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
    pl.add_arrows(velocity_points.points, velocity_points.point_data["Va"], mag=3, color="Green")
    pl.add_arrows(velocity_points.points, velocity_points.point_data["Vn"], mag=3, color="Black")
    pl.add_arrows(velocity_points.points, velocity_points.point_data["Vp"], mag=3, color="Blue")
    pl.add_arrows(velocity_points.points, velocity_points.point_data["dVp"], mag=1000, color="Yellow")
    
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


