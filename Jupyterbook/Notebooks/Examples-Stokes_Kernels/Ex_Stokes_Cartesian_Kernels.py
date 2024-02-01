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

# # Cartesian Stokes Kernels
#
# Mesh with embedded internal surface
#
# This allows us to introduce an internal force integral

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

res = 0.05
resH = 0.25

options = PETSc.Options()

import os

os.environ["SYMPY_USE_CACHE"] = "no"

# +
from enum import Enum

class boundaries(Enum):
    Top = 1
    Bottom = 2
    Internal = 3
    Left = 4
    Right = 5


xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0
yint = 0.66

cellSize = res

if uw.mpi.rank == 0:
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("KernelBox")

    p1 = gmsh.model.geo.add_point(xmin, ymin, 0.0, meshSize=cellSize)
    p2 = gmsh.model.geo.add_point(xmax, ymin, 0.0, meshSize=cellSize)
    p3 = gmsh.model.geo.add_point(xmin, ymax, 0.0, meshSize=cellSize)
    p4 = gmsh.model.geo.add_point(xmax, ymax, 0.0, meshSize=cellSize)

    # Internal surface points
    p5 = gmsh.model.geo.add_point(xmin, yint, 0.0, meshSize=cellSize)
    p6 = gmsh.model.geo.add_point(xmax, yint, 0.0, meshSize=cellSize)

    l1 = gmsh.model.geo.add_line(p1, p2)
    l2 = gmsh.model.geo.add_line(p3, p4) 
    l3 = gmsh.model.geo.add_line(p1, p5)
    l4 = gmsh.model.geo.add_line(p5, p3)
    l5 = gmsh.model.geo.add_line(p2, p6)
    l6 = gmsh.model.geo.add_line(p6, p4) 
    l7 = gmsh.model.geo.add_line(p5, p6)

    cl1 = gmsh.model.geo.add_curve_loop((l1, l5, -l7, -l3))
    cl2 = gmsh.model.geo.add_curve_loop((-l2, -l4, l7, l6))
    
    gmsh.model.geo.synchronize()

    # gmsh.model.geo.add_curve_loops([cl1,cl2])
    surface1 = gmsh.model.geo.add_plane_surface([cl1])
    surface2 = gmsh.model.geo.add_plane_surface([cl2])

    gmsh.model.geo.synchronize()

    # Add Physical groups for boundaries
    gmsh.model.add_physical_group(1, [l1,], boundaries.Bottom.value)
    gmsh.model.set_physical_name(1, l1, boundaries.Bottom.name)
    gmsh.model.add_physical_group(1, [l2], boundaries.Top.value)
    gmsh.model.set_physical_name(1, l2, boundaries.Top.name)
    gmsh.model.add_physical_group(1, [l3, l4], boundaries.Left.value)
    gmsh.model.set_physical_name(1, l3, boundaries.Left.name)
    gmsh.model.add_physical_group(1, [l5,l6], boundaries.Right.value)
    gmsh.model.set_physical_name(1, l4, boundaries.Right.name)            
    
    gmsh.model.add_physical_group(1, [l7], boundaries.Internal.value)
    gmsh.model.set_physical_name(1, l7, boundaries.Internal.name)

    gmsh.model.addPhysicalGroup(2, [surface1,surface2], 99999)
    gmsh.model.setPhysicalName(2, 99999, "Elements")

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write("tmp_cart_kernel_mesh.msh")

    # gmsh.fltk.run()

    gmsh.finalize()

kernel_mesh = uw.discretisation.Mesh(
        "tmp_cart_kernel_mesh.msh",
        degree=1,
        qdegree=3,
        useMultipleTags=True,
        useRegions=False,
        markVertices=True,
        boundaries=boundaries,
        coordinate_system_type=None,
        refinement=0,
        refinement_callback=None,
        return_coords_to_bounds=None,
    )

x,y = kernel_mesh.X

## The internal bc is not read correctly
## We set useRegions=False and then unstack the boundaries
## OK because we know (because it's our mesh) that the face-sets capture the boundaries 

uw.adaptivity._dm_unstack_bcs(kernel_mesh.dm, kernel_mesh.boundaries, "Face Sets")

kernel_mesh.dm.view()

# -
v_soln = uw.discretisation.MeshVariable(r"\mathbf{u}", kernel_mesh, 2, degree=2)
p_soln = uw.discretisation.MeshVariable(r"p", kernel_mesh, 1, degree=1, continuous=False)
p_cont = uw.discretisation.MeshVariable(r"pc", kernel_mesh, 1, degree=1, continuous=True)
syy = uw.discretisation.MeshVariable(r"Syy", kernel_mesh, 1, degree=1, continuous=True)

if 0 and uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(kernel_mesh)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(pvmesh, "Grey", "wireframe")

    pl.add_mesh(
        pvmesh,
        cmap="Greens",
        edge_color="Grey",
        show_edges=True,
        use_transparency=False,
        clim=[0.66, 1],
        opacity=0.75,
    )

    pl.show(cpos="xy")

# +
# Create Stokes object

stokes = Stokes(
    kernel_mesh, velocityField=v_soln, pressureField=p_soln, solver_name="stokes"
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

t_init = sympy.cos(3*x*sympy.pi) * sympy.exp(-1000.0 * ((y - yint) ** 2)) 

stokes.add_essential_bc(sympy.Matrix([sympy.oo, 0.0]), "Top")
stokes.add_essential_bc(sympy.Matrix([sympy.oo, 0.0]), "Bottom")
stokes.add_essential_bc(sympy.Matrix([0.0,sympy.oo]), "Left")
stokes.add_essential_bc(sympy.Matrix([0.0,sympy.oo]), "Right")

stokes.add_natural_bc(sympy.Matrix([0.0, -t_init]), "Internal")

stokes.bodyforce = sympy.Matrix([0,0])


# +
pressure_solver = uw.systems.Projection(kernel_mesh, p_cont)
pressure_solver.uw_function = p_soln.sym[0]
pressure_solver.smoothing = 1.0e-3

stress_solver = uw.systems.Projection(kernel_mesh, syy)
stress_solver.uw_function = stokes.constitutive_model.flux[1,1]
stress_solver.smoothing = 0.0e-6


# -

stokes.petsc_options.setValue("ksp_monitor", None)
stokes.petsc_options.setValue("snes_monitor", None)
stokes.petsc_options.delValue("snes_converged_reason")
stokes.solve()

# Pressure at mesh nodes
pressure_solver.solve()
stress_solver.solve()

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(kernel_mesh)
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym * sympy.exp(-100000.0 * ((y - 1) ** 2) ))
    velocity_points.point_data["V"][:,0] = 0
    velocity_points.point_data["Syy"] = vis.scalar_fn_to_pv_points(velocity_points, syy.sym)
    velocity_points.point_data["SyyV"] = velocity_points.point_data["V"].copy() * 0.0
    velocity_points.point_data["SyyV"][:,1] =  vis.scalar_fn_to_pv_points(velocity_points, syy.sym[0] * sympy.exp(-100000.0 * ((y - 1) ** 2)))


    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_cont.sym)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_init)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

    points = np.zeros((kernel_mesh._centroids.shape[0], 3))
    points[:, 0] = kernel_mesh._centroids[:, 0]
    points[:, 1] = kernel_mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", 
        integration_direction="both", 
        integrator_type=2,
        surface_streamlines=True,
        initial_step_length=0.01,
        max_time=1.0,
        max_steps=500
    )
   
    pl = pv.Plotter(title="Stokes Greens Functions", window_size=(750, 750))


    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        scalars="P",
        show_edges=True,
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=False,
    )

    pl.add_mesh(pvstream, opacity=0.3, show_scalar_bar=False)
    pl.add_arrows(velocity_points.points, velocity_points.point_data["SyyV"], mag=-1, show_scalar_bar=False)


    pl.show(cpos="xy")
# -
stokes.view()


stokes.constitutive_model.view()

stokes.constitutive_model.C

stokes.constitutive_model.c




