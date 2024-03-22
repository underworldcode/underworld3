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

mesh0 = uw.meshing.SphericalShell(
    radiusOuter=r_o, radiusInner=r_i, cellSize=res, filename="tmp_low_r.msh"
)

H = uw.discretisation.MeshVariable("H", mesh0, 1)
grad = uw.discretisation.MeshVariable(r"\nabla~T", mesh0, 1)

# Add a swarm to this mesh

swarm = uw.swarm.Swarm(mesh=mesh0)
gradS = uw.swarm.SwarmVariable(r"\nabla~T_S", swarm, proxy_degree=1, num_components=1)
swarm.populate(fill_param=2)

# +
# Mesh independent buoyancy force

x, y, z = mesh0.CoordinateSystem.N

t_forcing_fn = 1.0 * (
    +sympy.exp(-10.0 * (x**2 + (y - 0.8) ** 2 + z**2))
    + sympy.exp(-10.0 * ((x - 0.8) ** 2 + y**2 + z**2))
    + sympy.exp(-10.0 * (x**2 + y**2 + (z - 0.8) ** 2))
)

# -

with swarm.access(gradS):
    print(f"Swarm coords -> {swarm.particle_coordinates.data.shape}", flush=True)
    print(f"Swarm data -> {gradS.data.shape}", flush=True)


mesh0.dm.view()

options = PETSc.Options()
options.setValue("dm_adaptor", "parmmg")

# +
# mesh0.dm.setFromOptions()
# -

print(f"{uw.mpi.rank} - Coarsen", flush=True)
mesh0.dm.coarsen()
print(f"{uw.mpi.rank} - Coarsen / done", flush=True)


print(f"{uw.mpi.rank} - Refine", flush=True)
mesh0.dm.refine()
print(f"{uw.mpi.rank} - Refine / done", flush=True)


print(f"{uw.mpi.rank} - Coarsen Hierarchy", flush=True)
mesh0.dm.coarsenHierarchy(2)
print(f"{uw.mpi.rank} - Coarsen Hierarchy / done", flush=True)


print(f"{uw.mpi.rank} - Refine Hierarchy", flush=True)
dm2h = mesh0.dm.refineHierarchy(2)
print(f"{uw.mpi.rank} - Refine Hierarchy / done", flush=True)

with swarm.access(gradS):
    print(f"Swarm coords -> {swarm.particle_coordinates.data.shape}", flush=True)
    print(f"Swarm data -> {gradS.data.shape}", flush=True)

gradient = uw.systems.Projection(mesh0, grad, solver_name="gradient")
gradient.uw_function = 1.0 + mesh0.vector.gradient(t_forcing_fn**2).dot(
    mesh0.vector.gradient(t_forcing_fn**2)
)
gradient.smoothing = 1.0e-2
gradient.solve()

with mesh0.access():
    print(grad.data.min(), grad.data.max(), flush=True)

with mesh0.access(H):
    H.data[:, 0] = 5 + grad.data[:, 0] * 10

with swarm.access(gradS):
    print(f"Swarm coords -> {swarm.particle_coordinates.data.shape}", flush=True)
    print(f"Swarm data -> {gradS.data.shape}", flush=True)

# +
# with swarm.access(gradS):
#     gradS.data[:] = grad.rbf_interpolate(swarm.particle_coordinates.data)


# +
##

# +
##

# +
# This is how we adapt the mesh

print("Mesh adaptation ... start", flush=True)

meshA = adaptivity.mesh_adapt_meshVar(mesh0, H)

print("Mesh adaptation ... complete", flush=True)


# -


# +

# Add the variables we need to carry over
gradA = uw.discretisation.MeshVariable("\nabla~T", meshA, 1)

v_soln = uw.discretisation.MeshVariable(
    r"u", meshA, meshA.dim, degree=2, vtype=uw.VarType.VECTOR
)
p_soln = uw.discretisation.MeshVariable(r"p", meshA, 1, degree=1, continuous=True)

# -

0 / 0

meshA.dm.view()

# +
# We switch the swarm to the new mesh

swarm.mesh = meshA

# +
# This is how to nterpolate mesh variables

with meshA.access(gradA):
    gradA.data[:] = grad.rbf_interpolate(meshA.data)

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

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(meshA.dim)
stokes.constitutive_model.Parameters.viscosity = 1

# thermal buoyancy force
t_forcing_fn = 1.0 * (
    +sympy.exp(-10.0 * (x**2 + (y - 0.8) ** 2 + z**2))
    + sympy.exp(-10.0 * ((x - 0.8) ** 2 + y**2 + z**2))
    + sympy.exp(-10.0 * (x**2 + y**2 + (z - 0.8) ** 2))
)

unit_rvec = meshA.CoordinateSystem.unit_e_0
stokes.bodyforce = t_forcing_fn * unit_rvec
stokes.saddle_preconditioner = 1.0

# Velocity boundary conditions - upper / lower fixed
stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Upper", (0, 1, 2))
stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Lower", (0, 1, 2))
# -

stokes.solve()

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

    meshA.vtk("tmp_meshball.vtk")
    pvmeshA = pv.read("tmp_meshball.vtk")

    pvmeshA.points *= 0.999


# -


if mpi4py.MPI.COMM_WORLD.size == 1:
    with meshA.access():
        pvmeshA.point_data["gradS"] = gradS._meshVar.data.copy()

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[...] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[...] = stokes.u.rbf_interpolate(stokes.u.coords)

    clipped = pvmeshA.clip(origin=(0.0, 0.0, 0.0), normal=(0.1, 0, 1), invert=False)


# +

if mpi4py.MPI.COMM_WORLD.size == 1:
    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(
        clipped,
        cmap="coolwarm",
        edge_color="black",
        style="surface",
        scalars="gradS",
        show_edges=True,
    )

    pl.add_mesh(
        pvmesh0,
        edge_color="grey",
        color="grey",
        style="wireframe",
        render_lines_as_tubes=False,
    )

    pl.add_arrows(arrow_loc, arrow_length, mag=33, opacity=0.5)

    # pl.screenshot(filename="sphere.png", window_size=(1000, 1000), return_img=False)
    # OR
    pl.show(cpos="xy")
# -
