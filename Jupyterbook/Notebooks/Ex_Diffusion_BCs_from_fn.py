# # Cylindrical 2D Diffusion 

# +
from petsc4py import PETSc
import mpi4py
import underworld3 as uw
from underworld3.systems import Poisson
import numpy as np

options = PETSc.Options()

options["dm_plex_check_all"] = None

# options["poisson_pc_type"]  = "svd"
options["poisson_ksp_rtol"] = 1.0e-3
# options["poisson_ksp_monitor_short"] = None
# options["poisson_nes_type"]  = "fas"
options["poisson_snes_converged_reason"] = None
options["poisson_snes_monitor_short"] = None
# options["poisson_snes_view"]=None
options["poisson_snes_rtol"] = 1.0e-3

import os
os.environ["SYMPY_USE_CACHE"]="no"

# -

# Set some things
k = 1. 
h = 5.  
t_i = 2.
t_o = 1.
r_i = 0.5
r_o = 1.0

# +
dim=2
radius_inner = 0.1
radius_outer = 1.0

import pygmsh
# Generate local mesh.
with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_max = 0.1
    if dim==2:
        ndimspherefunc = geom.add_disk
    else:
        ndimspherefunc = geom.add_ball
    ball_outer = ndimspherefunc([0.0,]*dim, radius_outer)
    
    if radius_inner > 0.:
        ball_inner = ndimspherefunc([0.0,]*dim, radius_inner)
        geom.boolean_difference(ball_outer,ball_inner)

    

    pygmesh0 = geom.generate_mesh()

# -

import meshio
mesh = uw.meshes.SphericalShell(dim=2, radius_outer=1.0, radius_inner=0.0, cell_size=0.05, degree=1, verbose=True)                       

t_soln = uw.mesh.MeshVariable("T", mesh, 1, degree=3)

# +
# check the mesh if in a notebook / serial

if PETSc.Comm.size == 1:
    
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pvmesh = mesh.mesh2pyvista()
 
    clipped_stack = pvmesh.clip(origin=(0.0,0.0,0.0), normal=(-1, -1, 0), invert=False)

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Blue', 'wireframe' )
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, 
                  use_transparency=False)
    pl.show()


# +
# Create Poisson object

poisson = uw.systems.Poisson(mesh, u_Field=t_soln, 
                             solver_name="poisson", degree=3)
poisson.k = k
poisson.f = 0.0

bcs_var = uw.mesh.MeshVariable("bcs",mesh, 1)

# +
import sympy
abs_r = sympy.sqrt(mesh.rvec.dot(mesh.rvec))
bc = sympy.cos(2.0*mesh.N.y)

with mesh.access(bcs_var):
    bcs_var.data[:,0] = uw.function.evaluate(bc, mesh.data)

poisson.add_dirichlet_bc( bcs_var.fn, "Upper", components=0 )
poisson.add_dirichlet_bc( 1.0, "Centre" )
# -

poisson.petsc_options.getAll()

poisson.solve()

# +
# check the mesh if in a notebook / serial

if mpi4py.MPI.COMM_WORLD.size==1:
    
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pvmesh = mesh.mesh2pyvista()
    
    pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, mesh.data)

    # clipped_stack = pvmesh.clip(origin=(0.001,0.0,0.0), normal=(1, 0, 0), invert=False)

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Blue', 'wireframe' )
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, 
                  use_transparency=False)
    pl.show(cpos="xy")
# +
# savefile = "output/poisson_cyindrical_2d.h5" 
# mesh.save(savefile)
# poisson.u.save(savefile)
# mesh.generate_xdmf(savefile)


# +
## We should try the non linear version of this next ... 
