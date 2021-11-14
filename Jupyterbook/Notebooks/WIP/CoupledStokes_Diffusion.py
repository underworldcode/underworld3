# # Spherical Stokes coupled with Poisson for Buoyancy
#
# This is a simple example in which we try to instantiate two solvers on the mesh and have them use a common set of variables.
#
# We set up a v, p, T system in which we will solve for a steady-state T field in response to thermal boundary conditions and then we will 

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np

options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None


# Options directed at the Stokes solver

options["stokes_ksp_rtol"] =  1.0e-3
options["stokes_ksp_monitor"] = None
options["stokes_snes_converged_reason"] = None
options["stokes_snes_monitor_short"] = None

# options["stokes_snes_view"]=None
# options["stokes_snes_test_jacobian"] = None
options["stokes_snes_max_it"] = 1
options["stokes_pc_type"] = "fieldsplit"
options["stokes_pc_fieldsplit_type"] = "schur"
options["stokes_pc_fieldsplit_schur_factorization_type"] ="full"
options["stokes_pc_fieldsplit_schur_precondition"] = "a11"
options["stokes_fieldsplit_velocity_pc_type"] = "lu"
options["stokes_fieldsplit_pressure_ksp_rtol"] = 1.e-3
options["stokes_fieldsplit_pressure_pc_type"] = "lu"


# Options directed at the poisson solver

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

# options.getAll()

# +
import meshio

meshball = uw.mesh.SphericalShell(dim=3, radius_inner=0.0, radius_outer=1.0, cell_size=0.05)
# -

v_soln = uw.mesh.MeshVariable('U',meshball, 3, degree=2 )
p_soln = uw.mesh.MeshVariable('P',meshball, 1, degree=1 )
t_soln = uw.mesh.MeshVariable("T",meshball, 1, degree=3)


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pvmesh = meshball.mesh2pyvista()
    pvmesh.plot(show_edges=True)


# +
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, 
                u_degree=2, p_degree=1, solver_name="stokes")

# Constant visc
stokes.viscosity = 1.

# Velocity boundary conditions
stokes.add_dirichlet_bc( (0.,0.,0.), meshball.boundary.ALL_BOUNDARIES, (0,1,2) )

# +
# Create Poisson object

# Set some things
k = 1. 
h = -5.  # Note there is a sign error in the implementation, apparently
t_i = 2.
t_o = 1.
r_i = 0.5
r_o = 1.0

poisson = uw.systems.Poisson(meshball, phiField=t_soln, solver_name="poisson", degree=3)
poisson.k = k
poisson.h = h

bcs_var = uw.mesh.MeshVariable("bcs",meshball, 1)

# +

import sympy
abs_r = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
bc = sympy.cos(meshball.rvec.dot(meshball.N.k))

with meshball.access(bcs_var):
    bcs_var.data[:,0] = uw.function.evaluate(bc, meshball.data)

poisson.add_dirichlet_bc( bcs_var.fn, meshball.boundary.ALL_BOUNDARIES )
# -

poisson.solve()

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
z = meshball.N.z

r  = sympy.sqrt(x**2+y**2+z**2)
th = sympy.atan2(y+1.0e-5,z+1.0e-5)
ph = sympy.acos(z/(r+1.0e-5))

# -




stokes.bodyforce = unit_rvec * t_soln.fn # minus * minus
# uw.function.evaluate(-unit_rvec * t_soln.fn, meshball.data)[0:10,:]

stokes.solve()

# +
kd = uw.algorithms.KDTree(meshball.data)
kd.build_index()
n,d,b = kd.find_closest_point(t_soln.coords)

t_mesh = np.zeros((meshball.data.shape[0]))
w = np.zeros((meshball.data.shape[0]))

with meshball.access():
    for i in range(0,n.shape[0]):
        t_mesh[n[i]] += t_soln.data[i]
        w[n[i]] += 1.0
        
t_mesh /= w


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 600]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    
    pvmesh = meshball.mesh2pyvista()
    
    pvmesh.point_data["T"] = t_mesh
    # pvmesh.cell_data["T"] = uw.function.evaluate(t_soln.fn, pvmesh.cell_centers().points)
    # pvmesh.plot(scalars='my cell values', show_edges=True)
    
    with meshball.access():
        usol = stokes.u.data

    clipped_stack = pvmesh.clip(origin=(0.001,0.0,0.0), normal=(1, 0, 0), invert=False)

# -

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(clipped_stack, cmap="coolwarm", edge_color="Black", show_edges=False, 
                  use_transparency=False, opacity=0.5)
    pl.add_arrows(stokes.u.coords, usol, mag=100.0)
    pl.show()

usol.max()


