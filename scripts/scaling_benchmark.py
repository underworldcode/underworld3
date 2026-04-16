#!/usr/bin/env python
"""
Parallel scaling benchmark for Underworld3 Stokes solver.

Solves a 3D buoyancy-driven Stokes problem and reports timing.
Use run_scaling_test.sh to sweep across rank counts, or run directly:

    mpirun -n N pixi run -e amr-dev python scripts/scaling_benchmark.py

Adjust CELL_SIZE to control problem size:
  0.1  → ~4,600 elements  (quick, but too small for scaling)
  0.05 → ~36,000 elements (good for scaling tests)
  0.03 → ~150k+ elements  (larger machines)

Reference results (2026-03-11, Apple M4 Max, OpenMPI 5.0.10, 36k elements):
  1 rank: 54.2s | 2: 30.4s (1.78x) | 4: 17.7s (3.06x) | 8: 13.3s (4.08x)
"""

import time
import sympy
import underworld3 as uw
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# ----- Problem setup -----
# 3D unstructured simplex mesh — large enough to show scaling
# cellSize=0.05 gives ~36k elements — good balance of size vs. runtime
CELL_SIZE = 0.05

t0 = time.perf_counter()

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0, 0.0),
    maxCoords=(1.0, 1.0, 1.0),
    cellSize=CELL_SIZE,
    regular=False,
    qdegree=2,
)

t_mesh = time.perf_counter() - t0

x, y, z = mesh.X

u = uw.discretisation.MeshVariable(
    "u", mesh, mesh.dim, vtype=uw.VarType.VECTOR, degree=2
)
p = uw.discretisation.MeshVariable(
    "p", mesh, 1, vtype=uw.VarType.SCALAR, degree=1
)

stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"
stokes.tolerance = 1.0e-3

stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 7
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None
stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# Buoyancy-driven flow
stokes.bodyforce = 1.0e6 * sympy.Matrix([0, x + z, 0])

stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Bottom")
stokes.add_dirichlet_bc((0.0, None, 0.0), "Top")
stokes.add_dirichlet_bc((0.0, None, None), "Left")
stokes.add_dirichlet_bc((0.0, None, None), "Right")
stokes.add_dirichlet_bc((None, None, 0.0), "Front")
stokes.add_dirichlet_bc((None, None, 0.0), "Back")

# ----- Solve and time -----
comm.Barrier()
t1 = time.perf_counter()

stokes.solve()

comm.Barrier()
t_solve = time.perf_counter() - t1

converged = stokes.snes.getConvergedReason()
ksp_its = stokes.snes.getLinearSolveIterations()
snes_its = stokes.snes.getIterationNumber()

# Gather DOF info
n_elements = mesh.dm.getHeightStratum(0)[1] - mesh.dm.getHeightStratum(0)[0]
total_elements = comm.reduce(n_elements, op=MPI.SUM, root=0)

if rank == 0:
    print(f"\n{'='*60}")
    print(f"  UNDERWORLD3 PARALLEL SCALING BENCHMARK")
    print(f"{'='*60}")
    print(f"  MPI ranks:        {size}")
    print(f"  Cell size:        {CELL_SIZE}")
    print(f"  Total elements:   {total_elements}")
    print(f"  Mesh time:        {t_mesh:.2f} s")
    print(f"  Solve time:       {t_solve:.2f} s")
    print(f"  SNES iterations:  {snes_its}")
    print(f"  KSP iterations:   {ksp_its}")
    print(f"  Converged:        {converged > 0} (reason={converged})")
    print(f"{'='*60}\n")
