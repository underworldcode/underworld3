"""Custom-P FMG on coarse levels with 0, 1 and a few cells per rank.

A Poisson solve on a box refined three times above a coarsest level of
``uw_coarse_cells`` cells (approximately), with the custom-P transfers, against
a GAMG reference. Records per level the cells per rank (min, max, how many
ranks hold none), then the installed PC, iterations, time, the recorded PC
fallbacks and the error against the reference. Run under mpirun.
"""
import time
import numpy as np
import underworld3 as uw
from underworld3.utilities import custom_mg

params = uw.Params(uw_cell_size=0.5, uw_levels=3)


def _wrap(dm, like):
    return uw.discretisation.Mesh(
        dm.clone(), simplex=True,
        coordinate_system_type=like.CoordinateSystem.coordinate_type,
        qdegree=2, boundaries=like.boundaries)


def _poisson(mesh):
    p = uw.systems.Poisson(mesh)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1
    p.f = 1.0
    p.add_dirichlet_bc(0.0, "Bottom")
    p.add_dirichlet_bc(1.0, "Top")
    p.petsc_options["ksp_rtol"] = 1e-8
    p.petsc_options["ksp_type"] = "cg"
    return p


m0 = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                       cellSize=params.uw_cell_size, regular=True, qdegree=2)
dms = [m0.dm]
for _ in range(int(params.uw_levels)):
    dms.append(dms[-1].refine())
for k, dm in enumerate(dms):
    s, e = dm.getHeightStratum(0)
    n = np.array(uw.mpi.comm.allgather(e - s))
    uw.pprint(f"MEASURE level {k}: cells {n.sum()}  per rank min {n.min()} max {n.max()} "
              f"empty {int((n == 0).sum())}/{len(n)}")

reference = _poisson(_wrap(dms[-1], m0))
reference.preconditioner = "gamg"
reference.solve()

solver = _poisson(_wrap(dms[-1], m0))
t0 = time.perf_counter()
status = "ok"
try:
    custom_mg.set_custom_fmg(solver, [_wrap(d, m0) for d in dms[:-1]], builder="barycentric")
    solver.solve()
except Exception as exc:
    status = f"{type(exc).__name__}: {str(exc)[:200]}"
elapsed = uw.mpi.comm.allreduce(time.perf_counter() - t0, op=__import__("mpi4py").MPI.MAX)
ksp = solver.snes.getKSP()
diff = np.asarray(solver.Unknowns.u.data) - np.asarray(reference.Unknowns.u.data)
num = uw.mpi.comm.allreduce(float(np.sum(diff ** 2)))
den = uw.mpi.comm.allreduce(float(np.sum(np.asarray(reference.Unknowns.u.data) ** 2)))
uw.pprint(f"MEASURE np {uw.mpi.size}: status {status}; pc {ksp.getPC().getType()}; "
          f"its {ksp.getIterationNumber()}; reason {solver.snes.getConvergedReason()}; "
          f"time {elapsed:.2f}s; rel err {(num / den) ** 0.5:.2e}; "
          f"fallbacks {solver.pc_fallbacks}")
