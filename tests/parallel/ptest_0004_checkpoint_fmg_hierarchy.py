"""Parallel (MPI) test: FMG mesh hierarchy survives a checkpoint round-trip.

A mesh built with ``refinement`` carries a geometric-multigrid hierarchy. On
reload from a checkpoint, ``Mesh(file)`` must transparently restore that
hierarchy in parallel too — without the cross-partition interpolation hang that
a naive (graph-partitioned) reload of independently-loaded levels produces.

The fix co-locates the levels by reconstructing the hierarchy the same way a
fresh ``refinement`` mesh is built: distribute the coarsest level first, then
``refine()`` it locally. ``refine()`` never moves points across the
decomposition (only ``distribute()`` does), so every coarse cell and all of its
children are guaranteed co-resident on one rank — the multigrid interpolation is
rank-local at any np. (The earlier approach distributed independently-built
levels and misaligned at parallel np, aborting inside
``DMPlexComputeInterpolatorNested``.) The saved deformed fine coordinates are
then stamped back on via an exact reference-coordinate lookup.

Run:

    cd tests/parallel
    mpirun -np 2 python ./ptest_0004_checkpoint_fmg_hierarchy.py

Asserts (checked on rank 0):
  1. The reloaded mesh has its hierarchy restored (levels > 1).
  2. Geometric FMG converges on the reloaded mesh — proving the levels are
     co-located and linked (a hang would block here, caught by the CI timeout).
"""

import os
import glob

import underworld3 as uw
from petsc4py import PETSc

rank = uw.mpi.rank
size = uw.mpi.size

fn = "/tmp/_ptest_0004_fmg_ckpt.h5"
if rank == 0:
    for p in glob.glob("/tmp/_ptest_0004_fmg_ckpt*"):
        os.remove(p)
uw.mpi.barrier()

# Build a refinement mesh (3-level hierarchy) and checkpoint it.
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
    cellSize=0.2, refinement=2, qdegree=2,
)
mesh.write(fn)
uw.mpi.barrier()

# Reload in parallel — same call as a non-hierarchical mesh.
mesh2 = uw.discretisation.Mesh(fn)
assert len(mesh2.dm_hierarchy) == len(mesh.dm_hierarchy) == 3, (
    f"hierarchy not restored: got {len(mesh2.dm_hierarchy)} levels"
)

# Geometric FMG must converge on the reloaded mesh (no cross-partition hang).
poisson = uw.systems.Poisson(mesh2)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1
poisson.f = 0.0
poisson.add_dirichlet_bc(0.0, "Bottom")
poisson.add_dirichlet_bc(1.0, "Top")
for k, v in {
    "pc_type": "mg", "pc_mg_type": "full", "pc_mg_galerkin": "both",
    "mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "sor",
    "mg_coarse_pc_type": "redundant", "mg_coarse_redundant_pc_type": "lu",
}.items():
    poisson.petsc_options[k] = v
poisson.solve()

assert poisson.petsc_options.getString("pc_type") == "mg"
assert poisson.snes.getConvergedReason() > 0

if rank == 0:
    print(
        f"ptest_0004 OK (np={size}): hierarchy restored to "
        f"{len(mesh2.dm_hierarchy)} levels, FMG converged in "
        f"{poisson.snes.getKSP().getIterationNumber()} iters",
        flush=True,
    )
