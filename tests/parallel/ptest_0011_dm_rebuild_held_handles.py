"""MPI test for the issue #492 DM-rebuild contract.

Run via mpi_runner.sh (mpirun -np N python ptest_0011_*.py).

Creating a second MeshVariable rebuilds ``mesh.dm`` collectively. Asserts on
EVERY rank:
  - a handle captured before the rebuild stays valid (stale, not blinded);
  - the captured wrapper is the old DM's last holder (refcount 1);
  - the rebuilt DM carries both fields and supports a solve.

Pre-fix, the eager ``dm_old.destroy()`` zeroed the captured wrapper's handle
on every rank (SIGSEGV on next use — design probe exit -11, issue #492).
"""
import numpy as np
import underworld3 as uw

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.3,
    regular=False, qdegree=2)

held = mesh.dm
dim0 = held.getDimension()

v = uw.discretisation.MeshVariable("v1", mesh, mesh.dim, degree=2)
assert mesh.dm is held, f"rank {uw.mpi.rank}: first variable must not rebuild"

p = uw.discretisation.MeshVariable("p1", mesh, 1, degree=1)
assert mesh.dm is not held, f"rank {uw.mpi.rank}: second variable must rebuild"
assert held.handle != 0, f"rank {uw.mpi.rank}: held DM handle was blinded (#492)"
assert held.getDimension() == dim0
assert held.getRefCount() == 1, (
    f"rank {uw.mpi.rank}: expected the captured wrapper to be the last "
    f"holder, refcount {held.getRefCount()}")
assert mesh.dm.getNumFields() == 2

# the rebuilt DM must be collectively functional
poisson = uw.systems.Poisson(mesh, u_Field=p)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1.0
poisson.f = 0.0
poisson.add_dirichlet_bc(0.0, "Bottom")
poisson.add_dirichlet_bc(1.0, "Top")
poisson.petsc_options["ksp_rtol"] = 1e-8
poisson.solve()

err = float(np.linalg.norm(p.data[:, 0] - p.coords[:, 1]))
nrm = float(np.linalg.norm(p.coords[:, 1])) + 1e-30
# rank-local relative error on an exact linear solution
assert err / nrm < 1e-8, f"rank {uw.mpi.rank}: solve wrong on rebuilt DM"

if uw.mpi.rank == 0:
    print(f"OK: held DM handle valid after rebuild on {uw.mpi.size} ranks; "
          f"solve on rebuilt DM exact", flush=True)
