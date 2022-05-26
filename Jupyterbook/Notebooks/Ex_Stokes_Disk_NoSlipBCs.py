# # Cylindrical Stokes

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np

# options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None


# +
import meshio

meshball = uw.meshes.SphericalShell(dim=2, radius_outer=1.0, radius_inner=0.0, cell_size=0.05, degree=1, verbose=True)                       
# -

v_soln = uw.mesh.MeshVariable('U',meshball, 2, degree=2 )
p_soln = uw.mesh.MeshVariable('P',meshball, 1, degree=1 )
t_soln = uw.mesh.MeshVariable("T",meshball, 1, degree=3 )


# +
# check the mesh if in a notebook / serial

if uw.mpi.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    
    pvmesh = meshball.mesh2pyvista()
    pvmesh.plot(show_edges=True, cpos="xy")


# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre 
# of the sphere to (say) 1 at the surface

import sympy

radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec)) # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10+radius_fn)
gravity_fn = radius_fn

# Some useful coordinate stuff 

x = meshball.N.x
y = meshball.N.y

r  = sympy.sqrt(x**2+y**2)
th = sympy.atan2(y+1.0e-5,x+1.0e-5)

# 

Rayleigh = 1.0e2

# +
# Surface-driven flow, use this bc

vtheta = r * sympy.sin(th)

vx = -vtheta*sympy.sin(th)
vy =  vtheta*sympy.cos(th)

# +
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, 
                u_degree=2, p_degree=1, solver_name="stokes")

# stokes.petsc_options.getAll() # to see the defaults 

stokes.petsc_options["fieldsplit_velocity_pc_type"]="lu" # serial
stokes.petsc_options["snes_converged_reason"]=None
stokes.petsc_options["snes_monitor"]=None


# Constant visc
stokes.viscosity = 1.

# Velocity boundary conditions

stokes.add_dirichlet_bc( (0.0, 0.0), "Upper",  (0,1))
stokes.add_dirichlet_bc( (0.0, 0.0), "Centre", (0,1))

# -

t_init = sympy.cos(3*th)

# +
# Write density into a variable for saving

with meshball.access(t_soln):
    t_soln.data[:,0] = uw.function.evaluate(t_init, t_soln.coords)
    print(t_soln.data.min(), t_soln.data.max())
# -
stokes.bodyforce = Rayleigh * unit_rvec * t_init # minus * minus
uw.function.evaluate(-unit_rvec * t_init, meshball.data)[0:10,:]

stokes.solve()

# +
# An alternative is to use the swarm project_from method using these points to make a swarm

# +
# check the mesh if in a notebook / serial


if uw.mpi.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 600]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    
    pvmesh = meshball.mesh2pyvista()
    
    with meshball.access():
        pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, meshball.data)
 
    with meshball.access():
        usol = stokes.u.data

    arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
    arrow_loc[:,0:2] = stokes.u.coords[...]
    
    arrow_length = np.zeros((stokes.u.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...]
# -


    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, 
                  use_transparency=False, opacity=0.5)
    pl.add_arrows(arrow_loc, arrow_length, mag=0.1)
    pl.show(cpos="xy")


