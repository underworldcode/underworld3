# %% [markdown]
# # Poisson Cartesian Solve

# %%
from petsc4py import PETSc
import underworld3 as uw
import numpy as np


# %%
from underworld3.util_mesh import UnstructuredSimplexBox

# %%
mesh = UnstructuredSimplexBox(minCoords=(0.0,0.0), maxCoords=(1.0,1.0), cellize=1.0/16) 

# %%
# Create Poisson object
poisson = uw.systems.Poisson(mesh)

# %%
import sympy
k = 1.0 
k

# %%
# Set some things
poisson.k = k
poisson.f = 0.
poisson.add_dirichlet_bc( 1., "Bottom" )  
poisson.add_dirichlet_bc( 0., "Top" )  

# %%
# Solve time
poisson.solve()

# %%
# Check. Construct simple linear which is solution for 
# above config.  Exclude boundaries from mesh data. 
import numpy as np
with mesh.access():
    mesh_numerical_soln = uw.function.evaluate(poisson.u.fn, mesh.data)
    mesh_analytic_soln = uw.function.evaluate(1.0-mesh.N.y, mesh.data)
    if not np.allclose(mesh_analytic_soln, mesh_numerical_soln, rtol=0.01):
        raise RuntimeError("Unexpected values encountered.")

# %%
# Validate

from mpi4py import MPI

if MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    mesh.vtk("mesh.tmp.vtk")
    pvmesh = pv.read("mesh.tmp.vtk")

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
# Now let's construct something a little more complex.
# First get the coord system off the mesh/dm.
N = mesh.N

# %%
# Create some function using one of the base scalars N.x/N.y/N.z
import sympy
k = sympy.exp(-N.y)

# %%
# View
k

# %%
# Don't forget to set the diffusivity
poisson.k = k

# %%
with mesh.access():
    orig_soln = poisson.u.data.copy()
poisson.solve()

# %%
# Simply confirm different results
with mesh.access():
    if not np.allclose(poisson.u.data, orig_soln):
        raise RuntimeError("Unexpected values encountered.")


# %%
from underworld3.util_mesh import UnstructuredSimplexBox

# %%
# Nonlinear example
mesh = UnstructuredSimplexBox(minCoords=(0.0,0.0), maxCoords=(1.0,1.0), cellSize=0.05) 
mesh.dm.view()


# %%
poisson = uw.systems.Poisson(mesh, degree=1)


# %%
u = poisson.u.fn


# %%
from sympy.vector import gradient
nabla_u = gradient(u)
poisson.k = 0.5*(nabla_u.dot(nabla_u))
poisson.k


# %%
N = mesh.N
abs_r2 = (N.x**2 + N.y**2)
poisson.f = -16*abs_r2
poisson.f


# %%
poisson.add_dirichlet_bc(abs_r2, "All_dm_boundaries" )

# %%
# First solve linear to get reasonable initial guess.
k_keep = poisson.k
poisson.k = 1.
poisson.solve()
# Now solve non-linear
poisson.k = k_keep
poisson.solve(zero_init_guess=False)

# %%
with mesh.access():
    exact = mesh.data[:,0]**2 + mesh.data[:,1]**2
    if not np.allclose(poisson.u.data[:,0],exact[:],rtol=7.e-2):
        l2 = np.linalg.norm(exact[:]-poisson.u.data[:,0])
        raise RuntimeError(f"Unexpected values encountered. Diff norm = {l2}")

