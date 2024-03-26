# %%
import underworld3 as uw
import numpy as np
import math
import petsc4py

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt
# -
# #### Setup the mesh params


# %%
# Set the resolution.
res = 64

Tdegree = 3

### diffusivity constant
k = 1

mesh_qdegree = Tdegree
mesh_qdegree


# %%
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

tmin, tmax = 0.5, 1
# -

# #### Set up the mesh

mesh = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)),
    minCoords=(xmin, ymin),
    maxCoords=(xmax, ymax),
    qdegree=mesh_qdegree,
)


# %%
petsc4py.PETSc.garbage_view()

# %%
passiveSwarm = uw.swarm.Swarm(mesh=mesh)
# test_ps = passiveSwarm.add_variable('test_data', vtype=uw.VarType.MATRIX,
#                                )

test_ps = uw.swarm.SwarmVariable(
    "test_data", passiveSwarm, size=(10000, 1), vtype=uw.VarType.MATRIX, _proxy=False,
)

# add particles to the swarm
num_p = 100

coords = np.zeros((num_p, 2))
coords[:, 0] = np.linspace(
    xmin + mesh.get_min_radius(), xmax - mesh.get_min_radius(), num_p
)
coords[:, 1] = np.linspace(
    ymin + mesh.get_min_radius(), ymax - mesh.get_min_radius(), num_p
)



# %%
petsc4py.PETSc.garbage_view()

# %%
kdt = uw.kdtree.KDTree(coords)

# %%
passiveSwarm.add_particles_with_coordinates(coords)


# %%
passiveSwarm.dm.view()

# %%

with passiveSwarm.access():
    print(test_ps.data.shape)

# %%
with passiveSwarm.access(test_ps):
    print(uw.mpi.rank, passiveSwarm.data.shape)
    test_ps.data[:, 0] = 1.0

# %%
passiveSwarm.write_timestep(
    "TEST_SWARM",
    "swarm",
    swarmVars=[],
    outputPath="./output",
    index=0,
    force_sequential=True,
)


# %%

# %%
