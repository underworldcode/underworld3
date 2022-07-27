# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Poisson
import numpy as np

options = PETSc.Options()
# options["pc_type"]  = "svd"

# options["ksp_rtol"] = 1.0e-7
# # options["ksp_monitor_short"] = None

# # options["snes_type"]  = "fas"
# options["snes_converged_reason"] = None
# options["snes_monitor_short"] = None
# # options["snes_view"]=None
# options["snes_rtol"] = 1.0e-7

# %%
mesh = uw.meshes.Unstructured_Simplex_Box(dim=2, minCoords=(0.0,0.0), maxCoords=(1.0,1,0), cell_size=0.05) 
mesh.dm.view()

# %%
# Create Poisson object
poisson = Poisson(mesh)

# %%
# Set some things
poisson.k = 1. 
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
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    pvmesh = mesh.mesh2pyvista()

    with mesh.access():
        pvmesh.point_data["T"]  = mesh_analytic_soln
        pvmesh.point_data["T2"] = mesh_numerical_soln
        pvmesh.point_data["DT"] = pvmesh.point_data["T"] - pvmesh.point_data["T2"] 
    
    pl = pv.Plotter()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="DT",
                  use_transparency=False, opacity=0.5)
    
    pl.camera_position="xy"
     
    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")  

# %%
pvmesh.point_data["DT"].min()

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
# Nonlinear example
mesh = uw.meshes.Unstructured_Simplex_Box(dim=2, minCoords=(0.0,0.0), maxCoords=(1.0,1,0), cell_size=0.05) 
mesh.dm.view()


# %%
poisson = Poisson(mesh, degree=1)


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


# %%
poisson._f1

# %%
0/0

# %%
# Now create system with mesh variable as source term.

mesh = uw.discretisation.Box(elementRes=(9,9), minCoords=(-2.2,-.4))
bnds = mesh.boundary
# Create Poisson object
u_degree = 1
poisson = Poisson(mesh, degree=u_degree)

# %%
# Model parameters
T1 = -1.0   # top surface temperature
T0 =  7.0   # bottom surface temperature
k =   3.0   # diffusivity
h =  10.0   # heat production, source term
y1 = mesh.maxCoords[1]
y0 = mesh.minCoords[1]
diff = uw.discretisation.MeshVariable( mesh=mesh, num_components=1, name="diff", vtype=uw.VarType.SCALAR, degree=u_degree )
# example of setting the auxiliary field by numpy array, a.k.a by hand
with mesh.access(diff):
    diff.data[:] = k # just set every aux dof to k

# %%
# Set some things
poisson.k = diff.fn   # Note the `.fn` here
poisson.f = h
poisson.add_dirichlet_bc( T0, bnds.BOTTOM )
poisson.add_dirichlet_bc( T1, bnds.TOP )

# %%
# Solve time
poisson.solve()

# %%
# analytic solution definitions
def analyticTemperature(y, h, k, c0, c1):
     return -h/(2.*k)*y**2 + c0*y + c1

# arbitrary constant given the 2 dirichlet conditions
c0 = (T1-T0+h/(2*k)*(y1**2-y0**2)) / (y1-y0)
c1 = T1 + h/(2*k)*y1**2 - c0*y1

# Check. Construct simple linear which is solution for 
# above config.  Exclude boundaries from mesh data. 
import numpy as np
with mesh.access():
    if not np.allclose(analyticTemperature(mesh.data[:,1], h, k, c0, c1),poisson.u.data[:,0]):
        raise RuntimeError("Unexpected values encountered.")


# %%

# %%

# %%
