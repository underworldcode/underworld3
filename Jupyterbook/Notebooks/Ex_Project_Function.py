# # Use SNES solvers to project sympy / mesh variable functions and derivatives to nodes
#
# Pointwise / symbolic functions cannot always be evaluated using `uw.function.evaluate` because they contain a mix of mesh variables, derivatives and symbols which may not be defined everywhere. 
#
# Our solution is to use a projection of the function to a continuous mesh variable with the SNES machinery performing all of the background operations to determine the values and the optimal fitting. 
#
# This approach also allows us to include boundary conditions, smoothing, and constraint terms (e.g. remove a null space) in cases (like piecewise continuous swarm variables) where this is difficult in the original form.
#
# We'll demonstrate this using a swarm variable (scalar / vector), but the same approach is useful for gradient recovery.

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

# -

meshbox = uw.meshes.Unstructured_Simplex_Box(dim=2, 
                                             minCoords=(0.0,0.0,0.0), 
                                             maxCoords=(1.0,1.0,1.0), 
                                             cell_size=1.0/32.0, 
                                             regular=True)

# +
import sympy

# Some useful coordinate stuff 

x = meshbox.N.x
y = meshbox.N.y
z = meshbox.N.z
# -

s_soln  = uw.mesh.MeshVariable("T",    meshbox,  1,            degree=2 )
v_soln  = uw.mesh.MeshVariable('U',    meshbox,  meshbox.dim,  degree=2 )
iv_soln = uw.mesh.MeshVariable('IU',   meshbox,  meshbox.dim,  degree=2 )


s_fn = sympy.cos(5.0*sympy.pi * x) * sympy.cos(5.0*sympy.pi * y)
sv_fn = sympy.vector.curl(v_soln.fn)



# +
swarm  = uw.swarm.Swarm(mesh=meshbox)
s_values  = uw.swarm.SwarmVariable("Ss", swarm, 1,           proxy_degree=3)
v_values  = uw.swarm.SwarmVariable("Vs", swarm, meshbox.dim, proxy_degree=3)
iv_values = uw.swarm.SwarmVariable("Vi", swarm, meshbox.dim, proxy_degree=3)

swarm.populate(fill_param=3)
# -


scalar_projection = uw.systems.Projection(meshbox, s_soln)
scalar_projection.uw_function = s_values.fn
scalar_projection.smoothing = 1.0e-6

# +
vector_projection = uw.systems.Vector_Projection(meshbox, v_soln)
vector_projection.uw_function = v_values.fn
vector_projection.smoothing = 1.0e-3  # see how well it works !

# Velocity boundary conditions (compare left / right walls in the soln !)

# vector_projection.add_dirichlet_bc( (0.0,), "Left" ,   (0,) )
vector_projection.add_dirichlet_bc( (0.0,), "Right" ,  (0,) )
vector_projection.add_dirichlet_bc( (0.0,), "Top" ,    (1,) )
vector_projection.add_dirichlet_bc( (0.0,), "Bottom" , (1,) )

# +
# try to enforce incompressibility

incompressible_vector_projection = uw.systems.Solenoidal_Vector_Projection(meshbox, iv_soln)
incompressible_vector_projection.uw_function =  v_soln.fn # + sv_fn
incompressible_vector_projection.smoothing = 1.0e-2  # see how well it works !

# Velocity boundary conditions (compare left / right walls in the soln !)
# incompressible_vector_projection.add_dirichlet_bc( (0.0,), "Left" ,   (0,) )
incompressible_vector_projection.add_dirichlet_bc( (0.0,), "Right" ,  (0,) )
incompressible_vector_projection.add_dirichlet_bc( (0.0,), "Top" ,    (1,) )
incompressible_vector_projection.add_dirichlet_bc( (0.0,), "Bottom" , (1,) )
# -

with swarm.access(s_values, v_values, iv_values):
    s_values.data[:,0]  = uw.function.evaluate(s_fn, swarm.data)    
    v_values.data[:,0]  = uw.function.evaluate(sympy.cos(5.0*sympy.pi * x) * sympy.cos(5.0*sympy.pi * y), swarm.data)    
    v_values.data[:,1]  = uw.function.evaluate(sympy.sin(5.0*sympy.pi * x) * sympy.sin(5.0*sympy.pi * y), swarm.data)
    iv_values.data[:,0] = uw.function.evaluate(sympy.vector.curl(v_soln.fn).dot(meshbox.N.i), swarm.data)    
    iv_values.data[:,1] = uw.function.evaluate(sympy.vector.curl(v_soln.fn).dot(meshbox.N.j), swarm.data)    

scalar_projection.solve()

vector_projection.solve()

incompressible_vector_projection.solve()

scalar_projection.uw_function = sympy.vector.divergence(iv_soln.fn)
scalar_projection.solve()

s_soln.stats()

# +
# check the projection

import mpi4py

if mpi4py.MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pv.start_xvfb()
    
    pvmesh = meshbox.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

    with meshbox.access():
        vsol = iv_soln.data.copy()
  
    pvmesh.point_data["S"]  = uw.function.evaluate(s_soln.fn, meshbox.data)

    arrow_loc = np.zeros((iv_soln.coords.shape[0],3))
    arrow_loc[:,0:2] = iv_soln.coords[...]
    
    arrow_length = np.zeros((iv_soln.coords.shape[0],3))
    arrow_length[:,0:2] = vsol[...] 
    
    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="S",
                  use_transparency=False, opacity=0.5)
    
    pl.add_arrows(arrow_loc, arrow_length, mag=1.0e-1, opacity=0.5)
    #pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)
    
    # pl.add_points(pdata)

    pl.show(cpos="xy")


# -
scalar_projection.uw_function = sympy.vector.divergence(v_soln.fn)
scalar_projection.solve()

s_soln.stats()

# +
# check the projection

import mpi4py

if mpi4py.MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pv.start_xvfb()
    
    pvmesh = meshbox.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

    with meshbox.access():
        vsol = v_soln.data.copy()
        ivsol = iv_soln.data.copy()
  
    pvmesh.point_data["S"]  = uw.function.evaluate(s_soln.fn, meshbox.data)

    arrow_loc = np.zeros((v_soln.coords.shape[0],3))
    arrow_loc[:,0:2] = v_soln.coords[...]
    
    arrow_length = np.zeros((v_soln.coords.shape[0],3))
    arrow_length[:,0:2] = vsol[...] - ivsol[...]
    
    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="S",
                  use_transparency=False, opacity=0.5)
    
    pl.add_arrows(arrow_loc, arrow_length, mag=1.0e-1, opacity=0.5)
    #pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)
    
    # pl.add_points(pdata)

    pl.show(cpos="xy")
# -




