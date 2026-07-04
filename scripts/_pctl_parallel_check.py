"""Parallel-safe percentile check for metric_density_from_gradient.

The metric must be PARTITION-INDEPENDENT: the same physical |∇f|
⇒ the same density, no matter how many ranks. Before the fix the
percentile window was computed on each rank's LOCAL gmag, so the
density field (hence ∫ρ over the domain) differed with rank count.

Witness = a global FE integral of the density field ∫ρ0 dΩ (and its
global max). Run:

    pixi run -e amr-dev python    scripts/_pctl_parallel_check.py   # n=1
    pixi run -e amr-dev mpirun -n 2 python scripts/_pctl_parallel_check.py

The two ‖∫ρ0‖ must agree (to FP / the documented tiny shared-DOF
over-weight). Pre-fix they diverged with rank count.
"""
import numpy as np
import underworld3 as uw
from underworld3.meshing import metric_density_from_gradient

mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0,
                          cellSize=1.0 / 12, qdegree=3)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
xy = np.asarray(T.coords)[:, :2]
rr = np.sqrt((xy ** 2).sum(1))
# steep localised ring ⇒ a heavy-tailed |∇T| distribution where the
# percentile window genuinely matters (and differs per subdomain)
T.data[:, 0] = np.exp(-((rr - 0.75) / 0.03) ** 2)

rho = metric_density_from_gradient(mesh, T, amp=8.0,
                                   lo_percentile=50.0,
                                   hi_percentile=97.0, name="pchk")

# ∫ρ dΩ — a single global scalar, partition-independent *iff* the
# density field itself is partition-independent.
I = uw.maths.Integral(mesh, rho).evaluate()
# global max of the normalised ramp (allreduce MAX, dedup-safe)
from underworld3.meshing.smoothing import _MDG_CACHE
_, _, rho0 = list(_MDG_CACHE.values())[0]
loc_max = float(np.asarray(rho0.data[:, 0]).max())
if uw.mpi.size > 1:
    from mpi4py import MPI as _MPI
    g_max = uw.mpi.comm.allreduce(loc_max, op=_MPI.MAX)
else:
    g_max = loc_max

uw.mpi.barrier()
if uw.mpi.rank == 0:
    print(f"size={uw.mpi.size}  ∫rho dΩ = {I:.10e}  "
          f"max(rho0_ramp) = {g_max:.10e}", flush=True)
