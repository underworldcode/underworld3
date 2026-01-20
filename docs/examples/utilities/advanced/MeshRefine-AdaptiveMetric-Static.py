# %% [markdown]
"""
# 🎓 MeshRefine-AdaptiveMetric-Static

**PHYSICS:** utilities  
**DIFFICULTY:** advanced  
**MIGRATED:** From underworld3-documentation/Notebooks

## Description
This example has been migrated from the original UW3 documentation.
Additional documentation and parameter annotations will be added.

## Migration Notes
- Original complexity preserved
- Parameters to be extracted and annotated
- Claude hints to be added in future update
"""

# %% [markdown]
"""
## Original Code
The following is the migrated code with minimal modifications.
"""

# %%
# +
## Mesh refinement ...

import os

os.environ["UW_TIMING_ENABLE"] = "1"
os.environ["SYMPY_USE_CACHE"] = "no"

import numpy as np
import petsc4py
import sympy
import underworld3 as uw
from petsc4py import PETSc
from underworld3 import adaptivity, function, timing

free_slip_upper = True


# +
# Earth-like ratio of inner to outer
r_o = 1.0
r_i = 0.547
res = 333 / 6730
res = 0.15

mesh0 = uw.meshing.SphericalShell(
    radiusOuter=r_o,
    radiusInner=r_i,
    cellSize=res)

H = uw.discretisation.MeshVariable("H", mesh0, 1)
Metric = uw.discretisation.MeshVariable("M", mesh0, 1, degree=1)
grad = uw.discretisation.MeshVariable(r"\nabla~T", mesh0, 1)
U = uw.discretisation.MeshVariable(r"U", mesh0, mesh0.dim, degree=2)

# Add a swarm to this mesh

swarm = uw.swarm.Swarm(mesh=mesh0)
gradS = uw.swarm.SwarmVariable(
    r"\nabla~T_s", swarm, vtype=uw.VarType.SCALAR, proxy_degree=1
)
swarm.populate(fill_param=1)


# +
# Mesh independent buoyancy force

x, y, z = mesh0.CoordinateSystem.N

t_forcing_fn = 1.0 * (
    +sympy.exp(-1.0 * (x**2 + (y - 1) ** 2 + z**2))
    # + sympy.exp(-3.0 * ((x - 0.8) ** 2 + y**2 + z**2))
    # + sympy.exp(-3.0 * (x**2 + y**2 + (z - 0.8) ** 2))
)

grad_fn = 0.01 + (
    (t_forcing_fn.diff(x)) ** 2
    + (t_forcing_fn.diff(y)) ** 2
    + (t_forcing_fn.diff(z)) ** 2
)

# grad_fn = 1.0 + mesh0.vector.gradient(t_forcing_fn**2).dot(mesh0.vector.gradient(t_forcing_fn**2))
# -
if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh0 = vis.mesh_to_pv_mesh(mesh0)
    pvmesh0.point_data["T"] = uw.function.evaluate(
        t_forcing_fn, pvmesh0.points, rbf=True
    )
    pvmesh0.point_data["gradT"] = uw.function.evaluate(
        grad_fn, pvmesh0.points, rbf=True
    )

    pvmesh0.points

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh0,
        cmap="coolwarm",
        edge_color="Black",
        scalars="gradT",
        show_edges=True,
        use_transparency=False,
        opacity=1,
        show_scalar_bar=False)

    pl.show(jupyter_backend="html")

# +
# gradient = uw.systems.Projection(mesh0, grad)
# gradient.uw_function = grad_fn
# gradient.petsc_options["snes_rtol"] = 1.0e-6
# gradient.smoothing = 1.0e-3
# gradient.solve()
# -

with mesh0.access(grad):
    grad.data[:, 0] = uw.function.evaluate(grad_fn, mesh0.data, mesh0.N)


with mesh0.access(H):
    H.data[:, 0] = 0 + grad.data[:, 0] * 2000
    # print(H.data.min())

H.stats()

# +
# with swarm.access(gradS):
#     gradS.data[:] = grad.rbf_interpolate(swarm._particle_coordinates.data)


# +
# This is how we adapt the mesh

icoord, meshA = adaptivity.mesh_adapt_meshVar(mesh0, H, Metric, redistribute=True)

# Add the variables we need to carry over
# gradA = uw.discretisation.MeshVariable(r"\nabla~T", meshA, 1)
# v_soln = uw.discretisation.MeshVariable(
#     r"u", meshA, meshA.dim, degree=2, vtype=uw.VarType.VECTOR
# )
# p_soln = uw.discretisation.MeshVariable(r"p", meshA, 1, degree=1, continuous=True)

# swarmA = uw.adaptivity.mesh2mesh_swarm(
#     mesh0, meshA, swarm, swarmVarList=[gradS], verbose=True
# )
# -
mesh0.view()
meshA.view()

# +
# gradSA = swarmA.vars["nablaT_s"]
# -


if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh0 = vis.mesh_to_pv_mesh(mesh0)
    pvmesh0.point_data["gradT"] = vis.scalar_fn_to_pv_points(pvmesh0, grad.sym)

    pvmeshA = vis.mesh_to_pv_mesh(meshA)
    pvmeshA.point_data["gradT"] = vis.scalar_fn_to_pv_points(pvmeshA, grad.sym)


    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmeshA,  # .clip(crinkle=True),
        style="surface",
        colormap="coolwarm",
        scalars="gradT",
        show_edges=True,
        line_width=2,
        use_transparency=False,
        opacity=1,
        clim=[-0.1, 0.9],
        show_scalar_bar=True)

    # pl.add_mesh(
    #             pvmesh0,
    #             style="wireframe",
    #             color="Black",
    #             use_transparency=False,
    #             opacity=1,
    #             show_scalar_bar=False,
    #            )

    # pvmesh0.points = pvmesh0.points * 0.99

    # pl.add_mesh(
    #     pvmesh0,  # .clip(origin=(-0.02, 0.0, 0.0)),
    #     colormap="coolwarm",
    #     scalars="gradT",
    #     edge_color="Grey",
    #     show_edges=True,
    #     line_width=0.5,
    #     use_transparency=False,
    #     opacity=1,
    #     clim=[-0.5, 1.5],
    #     show_scalar_bar=True,
    # )

    pl.camera.position = (0.0, 0.0, 4)

    pl.export_html("AdaptedSphere.html")

    pl.show(jupyter_backend="html")

# So now we have two meshes that probably have a different decompostions across the available processes.
# We pass swarms back and forth to carry the information between decompositions.
#
# There seems to be some issue when we pass a swarm across the domain - we lose particles if they are not in neighbouring patches. Looking at the source code for DMSwarm, this is probably because there is not a global search implemented in DMPlex and so particles are only handed to neighbours by DMSwarm. Sigh.
#
# A global accumulation of mesh points is not too bad though, because this is quite light compared to other information in the mesh. It may be more problematic for dense swarms.
#

0 / 0

# +
print(f"{uw.mpi.rank}: {gradSA.data.min()}, {gradSA.data.max()}")

print(f"{uw.mpi.rank}: {gradS.data.min()}, {gradS.data.max()}")


# +
# We map any variables we want on the new mesh
# Maybe not that many (T, for example) but others
# will be swarm based

adaptivity.mesh2mesh_meshVariable(grad, gradA, verbose=True)


# +
# Any solvers ? they will also need rebuilding

stokes = uw.systems.Stokes(
    meshA,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False)

stokes.tolerance = 1.0e-3
stokes.petsc_options["ksp_monitor"] = None
stokes.penalty = 0.1

x, y, z = meshA.CoordinateSystem.N

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
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

# +
# stokes.solve()

# +
# Save results

# meshA.write_timeste(
#     "adaptor_write_xdmf",
#     meshUpdates=True,
#     meshVars=[gradA, v_soln, p_soln],
#     swarmVars=[gradSA],
# )
# -


with meshA.access():
    print(gradSA._meshVar.data.min(), gradSA._meshVar.data.max())

if mpi4py.MPI.COMM_WORLD.size == 1:
    with meshA.access():
        pvmeshA.point_data["gradS"] = gradSA._meshVar.data

    with meshA.access():
        pvmeshA.point_data["gradSi"] = gradA.data

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[...] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[...] = stokes.u.rbf_interpolate(stokes.u.coords)

    clipped = pvmeshA.clip(origin=(0.0, 0.0, 0.0), normal=(0.1, 0, 1), invert=False)


if mpi4py.MPI.COMM_WORLD.size == 1:
    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(
        clipped,
        cmap="coolwarm",
        edge_color="black",
        style="surface",
        scalars="gradSi",
        show_edges=True)

    # pl.add_mesh(
    #     pvmesh0,
    #     edge_color="grey",
    #     color="grey",
    #     style="wireframe",
    #     render_lines_as_tubes=False,
    # )

    pl.add_arrows(arrow_loc, arrow_length, mag=10, opacity=0.5)

    # pl.screenshot(filename="sphere.png", window_size=(1000, 1000), return_img=False)
    # OR
    pl.show(cpos="xy")
