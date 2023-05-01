# +
## Mesh refinement ... 

import os
os.environ["UW_TIMING_ENABLE"] = "1"
os.environ["SYMPY_USE_CACHE"] = "no"

import petsc4py
from petsc4py import PETSc

from underworld3 import timing
from underworld3 import adaptivity

import underworld3 as uw
from underworld3 import function

import numpy as np
import sympy

free_slip_upper = True


# +
# Earth-like ratio of inner to outer
r_o = 1.0
r_i = 0.547
res = 500 / 6730 

mesh0 = uw.meshing.SphericalShell(radiusOuter=r_o, 
                           radiusInner=r_i, 
                           cellSize=res,
                           filename="tmp_low_r.msh")

H = uw.discretisation.MeshVariable("H", mesh0, 1)
Metric = uw.discretisation.MeshVariable("M", mesh0, 1, degree=1)
grad = uw.discretisation.MeshVariable(r"\nabla~T", mesh0, 1)
U = uw.discretisation.MeshVariable(r"U", mesh0, mesh0.dim, degree=2)

# Add a swarm to this mesh

swarm = uw.swarm.Swarm(mesh=mesh0)
gradS = uw.swarm.SwarmVariable(r"\nabla~T_s", swarm, proxy_degree=1, num_components=1)
swarm.populate(fill_param=3)

# +
# Mesh independent buoyancy force

x, y, z = mesh0.CoordinateSystem.N

t_forcing_fn = 1.0 * (
    + sympy.exp(-10.0 * (x**2 + (y - 0.8) ** 2 + z**2))
    + sympy.exp(-10.0 * ((x - 0.8) ** 2 + y**2 + z**2))
    + sympy.exp(-10.0 * (x**2 + y**2 + (z - 0.8) ** 2))
)

grad_fn = 1.0 + mesh0.vector.gradient(t_forcing_fn**2).dot(mesh0.vector.gradient(t_forcing_fn**2))
# -

mesh0.dm.view()

mesh0.view()

# +
# gradient = uw.systems.Projection(mesh0, grad, solver_name="gradient")
# gradient.uw_function = grad_fn
# gradient.petsc_options["snes_rtol"] = 1.0e-6
# gradient.smoothing = 1.0e-3
# gradient.solve()
# -

with mesh0.access(grad):
    grad.data[:,0] = uw.function.evaluate(grad_fn, mesh0.data, mesh0.N)


with mesh0.access():
    print(grad.data.min(), grad.data.max(), flush=True)

with mesh0.access(H):
    H.data[:,0] = 10 + grad.data[:,0] * 80
    # print(H.data.min())

with swarm.access(gradS):
    gradS.data[:] = 0.0 # grad.rbf_interpolate(swarm.particle_coordinates.data)


print("Mesh adaptation ", flush=True)

mesh0.dm.removeLabel("CellAdaptor")
mesh0.dm.createLabel("CellAdaptor")
ca_label = mesh0.dm.getLabel("CellAdaptor")
depth_label = mesh0.dm.getLabel("depth")
elt_is = depth_label.getStratumIS(3)
# ca_label.setStratumIS(0, elt_is)


# +
centroids = mesh0._get_mesh_centroids()
cStart, cEnd = mesh0.dm.getHeightStratum(0)

for c in elt_is.getIndices():
    cell = c - cStart
    if np.abs(centroids[cell, 0]) > 0.75:
        ca_label.setValue(cell, 10)
    else:
        ca_label.setValue(cell, 1)

         
# -

dmR = mesh0.dm.adaptLabel("CellAdaptor")

mesh0.dm.view()

dmR.view()

meshR = uw.meshing.Mesh(
        dmR,
        simplex=mesh0.dm.isSimplex,
        coordinate_system_type=mesh0.CoordinateSystem.coordinate_type,
        qdegree=mesh0.qdegree,
        refinement=None,
        refinement_callback=mesh0.refinement_callback,
        boundaries=mesh0.boundaries,
        distribute=False,
    )

# +
import mpi4py

if mpi4py.MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.anti_aliasing = "ssaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh0.vtk("tmp_meshball0.vtk")
    pvmesh0 = pv.read("tmp_meshball0.vtk")

    meshR.vtk("tmp_meshball.vtk")
    pvmeshR = pv.read("tmp_meshball.vtk")
    
    pvmesh0.points *= 0.995
    
    clipped = pvmeshR.clip(origin=(0.0, 0.0, 0.0), normal=(0.1, 0, 1), invert=False)


# +

if mpi4py.MPI.COMM_WORLD.size == 1:

    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(
        clipped, 
        cmap="coolwarm",
        edge_color="black",
        style="surface",
        show_edges=True,
    )

    pl.add_mesh(
        pvmesh0, 
        edge_color="black",
        cmap="coolwarm",
        style="surface",
        show_edges=True,)

    # pl.add_arrows(arrow_loc, arrow_length, mag=10, opacity=0.5)

    # pl.screenshot(filename="sphere.png", window_size=(1000, 1000), return_img=False)
    # OR
    pl.show(cpos="xy")
# -





0/0

# +
# This is how we adapt the mesh

icoord, meshA = adaptivity.mesh_adapt_meshVar(mesh0, H, Metric, redistribute=False)

# Add the variables we need to carry over 
gradA = uw.discretisation.MeshVariable(r"\nabla~T", meshA, 1)
v_soln = uw.discretisation.MeshVariable(r"u", meshA, meshA.dim, degree=2, vtype=uw.VarType.VECTOR)
p_soln = uw.discretisation.MeshVariable(r"p", meshA, 1, degree=1, continuous=True)

# We switch the swarm to the new mesh before we 
# map values on the new variables.

# swarm.mesh = meshA # Maybe this works. perhaps not though
meshA.dm.setName("uw_mesh_A")
# -
meshA.dm.view()

# +
swarmA = uw.swarm.Swarm(mesh=meshA)
gradSA = uw.swarm.SwarmVariable(r"\nabla~T_{SA}", swarmA, proxy_degree=1, num_components=1)
swarmA.populate(fill_param=2)

with swarmA.access(gradSA), swarm.access():
    gradSA.data[:] = grad.rbf_interpolate(swarmA.data)

swarmA.dm.migrate(remove_sent_points=True)
print("Swarm Migrated", flush=True)


# -

with swarmA.access():
    print(f"{uw.mpi.rank}: {gradSA.data.min()}, {gradSA.data.min()}")

# +
swarm0 = uw.swarm.Swarm(mesh=mesh0)
gradS0 = uw.swarm.SwarmVariable(r"\nabla~T_{SA}", swarm0, _proxy=False, num_components=1 )
swarm0.dm.finalizeFieldRegister()

adds = swarmA.dm.getLocalSize()+1
# adds = 1000 

swarm0.dm.addNPoints( adds )

with  swarm0.access(swarm0.particle_coordinates, gradS0), swarmA.access():
         swarm0.data[:,:] = swarmA.data[:,:] 
         gradS0.data[:,:] = gradSA.data[:,:]
        
# Be cautious here
# uw.mpi.barrier()
        
swarm0.dm.migrate(remove_sent_points=True)


# +
# gradS0.save("adaptor_write.gradS0.h5")
# swarm0.particle_coordinates.save("adaptor_write.X.h5")
# -

print(f"{uw.mpi.rank} swarmA -> {swarmA.dm.getLocalSize()} / {swarmA.dm.getSize()}", flush=True)
print(f"{uw.mpi.rank} swarm0 -> {swarm0.dm.getLocalSize()} / {swarm0.dm.getSize()}", flush=True)

print(f"------", flush=True)

# +
# gradS0.save("adaptor_write.gradS0.h5")
# swarm0.particle_coordinates.save("adaptor_write.X.h5")

# +
# We map any variables we want on the new mesh
# Maybe not that many (T, for example) but others
# will be swarm based

# adaptivity.mesh2mesh_mapVar(grad, gradA)    


# +
# Any solvers ? they will also need rebuilding

stokes = uw.systems.Stokes(
    meshA,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
    solver_name="stokes",
)

stokes.tolerance = 1.0e-3
stokes.petsc_options["ksp_monitor"] = None
stokes.penalty = 0.1

x, y, z = meshA.CoordinateSystem.N

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(
    meshA.dim
)
stokes.constitutive_model.Parameters.viscosity = 1

# thermal buoyancy force
t_forcing_fn = 1.0 * (
    + sympy.exp(-10.0 * (x**2 + (y - 0.8) ** 2 + z**2))
    + sympy.exp(-10.0 * ((x - 0.8) ** 2 + y**2 + z**2))
    + sympy.exp(-10.0 * (x**2 + y**2 + (z - 0.8) ** 2))
)

unit_rvec = meshA.CoordinateSystem.unit_e_0
stokes.bodyforce = t_forcing_fn * unit_rvec
stokes.saddle_preconditioner = 1.0

# Velocity boundary conditions - upper / lower fixed
stokes.add_dirichlet_bc( (0.0, 0.0, 0.0), "Upper", (0,1,2))
stokes.add_dirichlet_bc( (0.0, 0.0, 0.0), "Lower", (0,1,2))

# +
# stokes.solve()

# +
# Save results

print("Save output ...", flush=True)

meshA.write_timestep_xdmf("adaptor_write_xdmf", meshUpdates=True, 
                          meshVars=[gradA, v_soln, p_soln], swarmVars=[gradSA])

print("Saved output ...", flush=True)



# +
import mpi4py

if mpi4py.MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.anti_aliasing = "ssaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh0.vtk("tmp_meshball0.vtk")
    pvmesh0 = pv.read("tmp_meshball0.vtk")

    meshR.vtk("tmp_meshball.vtk")
    pvmeshR = pv.read("tmp_meshball.vtk")
    
    pvmeshR.points *= 0.999
    
# +
if mpi4py.MPI.COMM_WORLD.size == 1:

#     with meshA.access():
#         pvmeshA.point_data["gradS"] = gradA.data.copy()

#     arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
#     arrow_loc[...] = stokes.u.coords[...]

#     arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
#     arrow_length[...] = stokes.u.rbf_interpolate(stokes.u.coords)

    clipped = pvmeshR.clip(origin=(0.0, 0.0, 0.0), normal=(0.1, 0, 1), invert=False)


# +

if mpi4py.MPI.COMM_WORLD.size == 1:

    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(
        clipped, 
        cmap="coolwarm",
        edge_color="black",
        style="surface",
        show_edges=True,
    )

    # pl.add_mesh(
    #     pvmesh0, 
    #     edge_color="grey",
    #     color="grey",
    #     style="wireframe",
    #     render_lines_as_tubes=False,
    # )

    # pl.add_arrows(arrow_loc, arrow_length, mag=10, opacity=0.5)

    # pl.screenshot(filename="sphere.png", window_size=(1000, 1000), return_img=False)
    # OR
    pl.show(cpos="xy")
# -




