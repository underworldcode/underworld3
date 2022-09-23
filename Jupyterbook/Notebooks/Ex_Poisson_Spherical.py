# %%
import pygmsh, meshio

from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Poisson
import numpy as np


# %%
# Set some things
k = 1. 
f = 0.
t_i = 2.
t_o = 1.
r_i = 0.5
r_o = 1.0

# %%
from underworld3.util_mesh import Annulus

# %%
# first do 2D
cell_size=0.02

mesh = Annulus(radiusInner=r_i, 
               radiusOuter=r_o,
               cellSize=cell_size)

t_soln  = uw.mesh.MeshVariable("T", mesh, 1, degree=2 )


# Create Poisson object
poisson = Poisson(mesh, u_Field=t_soln)
poisson.k = 1.0 
poisson.f = f

poisson.petsc_options["snes_rtol"] = 1.0e-6
poisson.petsc_options.delValue("ksp_monitor")
poisson.petsc_options.delValue("ksp_rtol")


# %%
import sympy
abs_r = sympy.sqrt(mesh.rvec.dot(mesh.rvec))
bc = sympy.Piecewise( ( t_i,  abs_r < 0.5*(r_i+r_o) ),
                      ( t_o,                 True ) )
poisson.add_dirichlet_bc( bc, "All_dm_boundaries" )

# %%
poisson.solve()

# %%
# Check. Construct simple solution for above config.
import math
A = (t_i-t_o)/(sympy.log(r_i)-math.log(r_o))
B = t_o - A*sympy.log(r_o)
sol = A*sympy.log(sympy.sqrt(mesh.N.x**2+mesh.N.y**2)) + B

with mesh.access():
    mesh_analytic_soln = uw.function.evaluate(sol,mesh.data)
    mesh_numerical_soln = uw.function.evaluate(t_soln.fn,mesh.data)

import numpy as np
if not np.allclose(mesh_analytic_soln,mesh_numerical_soln,rtol=0.001):
    raise RuntimeError("Unexpected values encountered.")

# %%
poisson.k = 1.0 + 0.1 * poisson.u.fn**1.5
poisson.f = 0.01 * poisson.u.fn**0.5
poisson.solve(zero_init_guess=False, _force_setup=True)

# %%
# Validate

from mpi4py import MPI

if MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    mesh.vtk("mesh_tmp.vtk")
    pvmesh = pv.read("mesh_tmp.vtk")

    with mesh.access():
        pvmesh.point_data["T"]  = mesh_analytic_soln
        pvmesh.point_data["T2"] = mesh_numerical_soln
        pvmesh.point_data["DT"] = pvmesh.point_data["T"] - pvmesh.point_data["T2"] 
    
    pl = pv.Plotter()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T2",
                  use_transparency=False, opacity=0.5)
    
    pl.camera_position="xy"
     
    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")  

# %%
savefile = "output/poisson_disc.h5" 
mesh.save(savefile)
poisson.u.save(savefile)
mesh.generate_xdmf(savefile)

# %%
from underworld3.util_mesh import SphericalShell

# %%
# now do 3D
cell_size=0.1
mesh_3d = SphericalShell(radiusInner=r_i, 
                         radiusOuter=r_o,
                         cellSize=cell_size)
t_soln_3d  = uw.mesh.MeshVariable("T", mesh_3d, 1, degree=2 )

# Create Poisson object
poisson = Poisson(mesh_3d, u_Field=t_soln_3d)
poisson.k = k
poisson.f = f

poisson.petsc_options["snes_rtol"] = 1.0e-6
poisson.petsc_options.delValue("ksp_monitor")
poisson.petsc_options.delValue("ksp_rtol")

# %%
import sympy
abs_r = sympy.sqrt(mesh.rvec.dot(mesh.rvec))
bc = sympy.Piecewise( ( t_i,  abs_r < 0.5*(r_i+r_o) ),
                      ( t_o,                 True ) )
poisson.add_dirichlet_bc( bc, "All_dm_boundaries" )

# %%
bc

# %%
# Solve time
poisson.solve()

# %%
# Check. Construct simple solution for above config.

A = (t_i-t_o)/(1/r_i-1/r_o)
B =  t_o - A / r_o
sol = A/(sympy.sqrt(mesh_3d.N.x**2+mesh_3d.N.y**2+mesh_3d.N.z**2)) + B

with mesh.access():
    mesh_analytic_soln = uw.function.evaluate(sol,mesh_3d.data)
    mesh_numerical_soln = uw.function.evaluate(t_soln_3d.fn,mesh_3d.data)

import numpy as np
if not np.allclose(mesh_analytic_soln,mesh_numerical_soln,rtol=0.01):
    raise RuntimeError("Unexpected values encountered.")

# %%
# Validate

from mpi4py import MPI

if MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    mesh_3d.vtk("mesh_tmp.vtk")
    pvmesh = pv.read("mesh_tmp.vtk")

    with mesh_3d.access():
        pvmesh.point_data["T"]  = mesh_analytic_soln
        pvmesh.point_data["T2"] = mesh_numerical_soln
        pvmesh.point_data["DT"] = pvmesh.point_data["T"] - pvmesh.point_data["T2"] 
        
    clipped = pvmesh.clip(origin=(0.001,0.0,0.0), normal=(1, 0, 0), invert=True)

    
    pl = pv.Plotter()
    

    pl.add_mesh(clipped, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="DT",
                  use_transparency=False, opacity=1.0)
 

    
    pl.camera_position="xy"
     
    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")  

# %%
savefile = "output/poisson_spherical_3d.h5" 
mesh.save(savefile)
poisson.u.save(savefile)
mesh.generate_xdmf(savefile)
