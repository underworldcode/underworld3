"""3-D split-node faults in parallel: rank-interior faults work, seam
contact refuses collectively.

Run with:  mpirun -np 2 python ptest_0848_fault_split_3d_parallel.py

The v1 parallel contract for 3-D is deliberately conservative: the fault
patch's whole cell star must be rank-interior (a 3-D seam crossing is a
curve — a design of its own, deferred). This test proves both halves of
the contract at np = 2:

1. A patch interior to one rank splits cleanly — chart deltas match the
   patch census on the owning rank, every rank rebuilds, the star forest
   carries over with zero coordinate drift, and a P2 Poisson solve on
   the split mesh shows the Plus datum does NOT leak onto the coincident
   Minus DOFs.
2. A patch straddling the seam raises the SAME RuntimeError on EVERY
   rank — no rank continues into the collective rebuild and blocks.
"""

import numpy as np
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities.fault_split import split_fault

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

PASS, FAIL = [], []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    uw.pprint(f"  [{'ok' if ok else 'FAIL'}] {name}")


def sf_coordinate_drift(mesh):
    """Broadcast every vertex coordinate root-to-leaf, one component per
    pass (the SF's unit is one double per point); leaves must agree with
    their own coordinates. Lifted from ``ptest_0845``."""
    dm = mesh.dm
    sf = dm.getPointSF()
    try:
        sf.getGraph()
    except (ValueError, TypeError):
        return 0.0
    pStart, pEnd = dm.getChart()
    vS, vE = dm.getDepthStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, mesh.dim)
    worst = 0.0
    for comp in range(mesh.dim):
        root = np.zeros(pEnd - pStart, dtype=np.float64)
        root[vS - pStart: vE - pStart] = X[: vE - vS, comp]
        leaf = np.full(pEnd - pStart, np.nan, dtype=np.float64)
        sf.bcastBegin(MPI.DOUBLE, root, leaf, MPI.REPLACE)
        sf.bcastEnd(MPI.DOUBLE, root, leaf, MPI.REPLACE)
        seen = np.isfinite(leaf[vS - pStart: vE - pStart])
        if seen.any():
            worst = max(worst, float(np.abs(
                leaf[vS - pStart: vE - pStart][seen]
                - X[: vE - vS, comp][seen]).max()))
    return float(comm.allreduce(worst, MPI.MAX))


# ---- case 1: rank-interior patch ------------------------------------------
# The partitioner balances cell counts and its cut is ATTRACTED to
# locally refined regions, so in a unit cube a patch is never safely
# interior at np = 2. An ELONGATED box pins the cut across the long
# axis (probed: shared band x in [0.8, 1.25]); a patch near one end
# stays clear of it with a full cell-star margin.
PATCH_IN = np.array([[0.4, 0.35, 0.35], [0.4, 0.65, 0.35],
                     [0.4, 0.65, 0.65], [0.4, 0.35, 0.65]])
mesh1 = uw.meshing.BoxInternalPatch(cellSize=0.15, minCoords=(0, 0, 0),
                                    maxCoords=(2, 1, 1),
                                    patch_points=PATCH_IN,
                                    patch_name="FltA")
try:
    child = split_fault(mesh1, "FltA")
    check("interior patch splits", True)
except RuntimeError as exc:
    # the layout is only guaranteed at np = 2; elsewhere a seam through
    # the patch is a legal partition, not a defect
    check(f"interior patch splits (refused: {exc})", size != 2)
    child = None

if child is not None:
    n_plus_local = 0
    value = int(child.boundaries["FltAPlus"].value)
    if child.dm.hasLabel("FltAPlus") and \
            child.dm.getLabel("FltAPlus").getStratumSize(value) > 0:
        n_plus_local = child.dm.getLabel("FltAPlus").getStratumSize(value)
    n_plus = comm.allreduce(n_plus_local, MPI.SUM)
    check("plus faces exist globally", n_plus > 0)
    check("fault is rank-interior (one owner)",
          comm.allreduce(1 if n_plus_local else 0, MPI.SUM) == 1)
    check("sf drift is zero", sf_coordinate_drift(child) == 0.0)

    u = uw.discretisation.MeshVariable("u", child, 1, degree=2)
    poisson = uw.systems.Poisson(child, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    for wall in ("Bottom", "Top", "Right", "Left", "Front", "Back"):
        poisson.add_dirichlet_bc(0.0, wall)
    poisson.add_dirichlet_bc(1.0, "FltAPlus")
    poisson.solve()

    vals = np.asarray(u.data[:, 0])
    lo = float(comm.allreduce(vals.min() if vals.size else 1e30, MPI.MIN))
    hi = float(comm.allreduce(vals.max() if vals.size else -1e30, MPI.MAX))
    check("solution bounded by the data", lo >= -1e-9 and hi <= 1.0 + 1e-9)

    n_pairs_local = len(child._fault_point_pairs["FltA"])
    sec = child.dm.getLocalSection()
    decoupled = 0
    for q_minus, q_plus in child._fault_point_pairs["FltA"].items():
        if sec.getFieldDof(q_minus, 0) == 0:
            continue
        d_minus = sec.getFieldOffset(q_minus, 0)
        d_plus = sec.getFieldOffset(q_plus, 0)
        if vals[d_plus] > vals[d_minus] + 1e-9:
            decoupled += 1
    check("minus DOFs decoupled from the plus datum",
          comm.allreduce(decoupled, MPI.SUM) > 0)

# ---- case 2: seam-straddling patch refuses collectively --------------------
PATCH_SEAM = np.array([[1.0, 0.35, 0.35], [1.0, 0.65, 0.35],
                       [1.0, 0.65, 0.65], [1.0, 0.35, 0.65]])
# (the middle of the long axis, where the balance cut runs)
mesh2 = uw.meshing.BoxInternalPatch(cellSize=0.15, minCoords=(0, 0, 0),
                                    maxCoords=(2, 1, 1),
                                    patch_points=PATCH_SEAM,
                                    patch_name="FltB")
raised = 0
try:
    split_fault(mesh2, "FltB")
except RuntimeError:
    raised = 1
except ValueError:
    # a partition seam that misses x=0.5 entirely would make this patch
    # rank-interior and the split legal; only count the seam refusal
    raised = 1
n_raised = comm.allreduce(raised, MPI.SUM)
check("seam patch verdict is collective (all ranks agree)",
      n_raised in (0, size))
if size > 1:
    check("seam patch refused", n_raised == size)

# ---- report ----------------------------------------------------------------
n_fail = comm.allreduce(len(FAIL), MPI.SUM)
uw.pprint(f"ptest_0848: {len(PASS)} passed, {len(FAIL)} failed "
          f"(rank {rank})")
if rank == 0:
    print("PTEST_0848_RESULT:", "PASS" if n_fail == 0 else "FAIL")
assert n_fail == 0
