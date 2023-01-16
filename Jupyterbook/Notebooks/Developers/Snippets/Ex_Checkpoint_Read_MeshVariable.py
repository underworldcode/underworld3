# %%
# Enable timing (before uw imports)

import os

os.environ["UW_TIMING_ENABLE"] = "1"

from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import timing

import numpy as np
import sympy
from mpi4py import MPI

if uw.mpi.size == 1:
    os.makedirs("output", exist_ok=True)
else:
    os.makedirs(f"output_np{uw.mpi.size}", exist_ok=True)

# %%
# Define the problem size
#      1  - ultra low res for automatic checking
#      2  - low res problem to play with this notebook
#      3  - medium resolution (be prepared to wait)
#      4  - highest resolution
#      5+ - v. high resolution (parallel only)

problem_size = uw.options.getInt("model_size", default=2)

if problem_size <= 1:
    res = 8
elif problem_size == 2:
    res = 16
elif problem_size == 3:
    res = 32
elif problem_size == 4:
    res = 48
elif problem_size == 5:
    res = 64
elif problem_size >= 6:
    res = 128

# %%
# Set size and position of dense sphere.
sphereRadius = 0.1
sphereCentre = (0.0, 0.7)

# define some names
materialLightIndex = 0
materialHeavyIndex = 1

# Set constants for the viscosity and density of the sinker.
viscBG = 1.0
viscSphere = 1.0e6

densityBG = 1.0
densitySphere = 10.0

expt_name = f"output/sinker_eta{viscSphere}_rho10_res{res}"

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / res,
    regular=False,
    qdegree=3,
)

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)


# %%

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)

stokes.add_dirichlet_bc(
    (0.0), ["Top", "Bottom"], [1]
)  
stokes.add_dirichlet_bc(
    (0.0), ["Left", "Right"], [0]
)  

# %%
swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.IndexSwarmVariable(
    "M", swarm, indices=2, proxy_continuous=True, proxy_degree=1
)
swarm.populate(fill_param=5)

blob = np.array([[sphereCentre[0], sphereCentre[1], sphereRadius, 1]])

with swarm.access(material):
    material.data[...] = materialLightIndex

    for i in range(blob.shape[0]):
        cx, cy, r, m = blob[i, :]
        inside = (swarm.data[:, 0] - cx) ** 2 + (swarm.data[:, 1] - cy) ** 2 < r**2
        material.data[inside] = m

# %%
mat_density = np.array([densityBG, densitySphere])
density = mat_density[0] * material.sym[0] + mat_density[1] * material.sym[1]

mat_viscosity = np.array([viscBG, viscSphere])
viscosityMat = mat_viscosity[0] * material.sym[0] + mat_viscosity[1] * material.sym[1]

viscosity = viscBG * material.sym[0] + viscSphere * material.sym[1]

# %%
stokes.constitutive_model.Parameters.viscosity = viscosity
stokes.bodyforce = sympy.Matrix([0, -1 * density])
stokes.penalty = 1.0
stokes.saddle_preconditioner = 1.0 / viscosity

snes_rtol = 1.0e-6
stokes.tolerance = snes_rtol
stokes.petsc_options["ksp_monitor"] = None

stokes._setup_terms()
stokes.solve(zero_init_guess=True)

# %%
## Now we have data we can save
mesh.write_checkpoint("test_save_load", meshUpdates=False, meshVars=[p, v])

# %%
v_stats = mesh.stats(v.sym[0])
p_stats = mesh.stats(p.sym[0])

if uw.mpi.rank == 0:
    print(v_stats, flush=True)
    print(p_stats, flush=True)


# %%
if uw.mpi.rank == 0:
    print("New mesh to match existing")

mesh2 = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=1.0 / res,
    regular=False,
    qdegree=3,
)

v2 = uw.discretisation.MeshVariable("U2", mesh2, mesh2.dim, degree=2)
p2 = uw.discretisation.MeshVariable("P2", mesh2, 1, degree=1, continuous=True)

if uw.mpi.rank == 0:
    print("reload U,P from h5 files")

v2.load("test_save_load.U.0.h5", data_name="U")
p2.load("test_save_load.P.0.h5", data_name="P")

# %%
v_stats = mesh.stats(v2.sym[0])
p_stats = mesh.stats(p2.sym[0])

if uw.mpi.rank == 0:
    print(v_stats, flush=True)
    print(p_stats, flush=True)


# %%
if uw.mpi.rank == 0:
    print("reload mesh from h5 file")

mesh3 = uw.discretisation.Mesh("test_save_load.mesh.0.h5")
v3 = uw.discretisation.MeshVariable("U3", mesh3, mesh3.dim, degree=2)
p3 = uw.discretisation.MeshVariable("P3", mesh3, 1, degree=1, continuous=True)
   
if uw.mpi.rank == 0:
    print("reload U,P from h5 files")

v3.load("test_save_load.U.0.h5", data_name="U")
p3.load("test_save_load.P.0.h5", data_name="P")

# %%
v_stats = mesh.stats(v3.sym[0])
p_stats = mesh.stats(p3.sym[0])

if uw.mpi.rank == 0:
    print(v_stats, flush=True)
    print(p_stats, flush=True)


# %%
