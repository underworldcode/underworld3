"""partition_addon on the 2-D adapt-on-top fault fixture: the invariants, at any np.

mpirun -np N python probe_partition_2d.py [k=2]
"""
import sys
import numpy as np
import underworld3 as uw
from mpi4py import MPI
from petsc4py import PETSc
from underworld3.utilities import addon_partition as ap
from underworld3.utilities.fault_mg import _barycentric, _cell_vertex_table, addon_level_indices
from underworld3.utilities.place_surface import _region_cells

comm = MPI.COMM_WORLD
k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TIP_A, TIP_B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.1, refinement=1, qdegree=2)
pts = np.column_stack([np.linspace(TIP_A[0], TIP_B[0], 9), np.linspace(TIP_A[1], TIP_B[1], 9), np.zeros(9)])
guide = uw.meshing.Surface("guide", base, pts); guide.discretize()
child = base.adapt(guide.refinement_metric_function(h_near=0.05 / 2**k, h_far=0.05, width=0.08, profile="linear"), max_levels=k)
trace = np.vstack([TIP_A, TIP_B])

def cells(m):
    cS, cE = m.dm.getHeightStratum(0); return cE - cS

def report(tag, m):
    tail = m._custom_mg_coarse_meshes
    idx = addon_level_indices(tail)
    per = [comm.gather(cells(tail[i]), root=0) for i in idx] + [comm.gather(cells(m), root=0)]
    if comm.rank == 0:
        print(f"{tag}: add-on levels {len(idx)}; cells per rank per level (coarsest add-on .. finest): {per}", flush=True)

report("before", child)
marks = ap.mark_polylines([trace])
out = ap.partition_addon(child, marks, layers=1, verbose=True)
report("after ", out)
rec = out._addon_partition
if comm.rank == 0:
    la = np.array(rec["load_after"]); lb = np.array(rec["load_before"])
    print(f"record: {rec['regions']} region(s) owners {rec['owners']}, units moved {rec['units_moved']}, sweeps {rec['sweeps']}, load max/mean {lb.max()/lb.mean():.2f} -> {la.max()/la.mean():.2f}", flush=True)

# 1. regions interior on the new finest (the split's precondition): star + 2 layers of the trace on one rank
marks2 = marks(out.dm)
cell_region, canon, regions = _region_cells(out.dm, marks2, layers=2)
counts = np.asarray(comm.allgather([int((cell_region == r).sum()) for r in regions]))
if comm.rank == 0:
    print(f"regions after: {len(regions)}; ranks holding each: {[int(np.count_nonzero(counts[:, j])) for j in range(len(regions))]} (1 = interior)", flush=True)

# 2. add-on pairs co-partitioned: every fine cell's parent is local
tail = out._custom_mg_coarse_meshes
levels = [tail[i] for i in addon_level_indices(tail)] + [out]
static = [tail[i] for i in range(len(tail)) if i not in addon_level_indices(tail)]
ok = True
for l in range(1, len(levels)):
    try:
        ap._parent_cells_local(levels[l - 1], levels[l])
    except RuntimeError as e:
        ok = False
        print(f"rank {comm.rank}: pair {l-1}->{l} NOT co-partitioned: {e}", flush=True)
ok = comm.allreduce(ok, op=MPI.LAND)
if comm.rank == 0:
    print(f"add-on pairs co-partitioned: {ok}", flush=True)

# 3. provenance: fetch each parent's vertex coordinates by its CURRENT address and test containment
def check_parents(below, lev, tag):
    par = lev._addon_parent
    vxy_b, cv_b = _cell_vertex_table(below)
    nb = cv_b.shape[0]
    verts = vxy_b[cv_b].reshape(nb, -1)              # (nb, (dim+1)*cdim)
    sf = PETSc.SF().create(comm=PETSc.COMM_WORLD)
    sf.setGraph(nb, None, np.ascontiguousarray(par, dtype=np.int32)); sf.setUp()
    got = np.zeros((par.shape[0], verts.shape[1]))
    unit = MPI.DOUBLE.Create_contiguous(verts.shape[1]); unit.Commit()
    sf.bcastBegin(unit, np.ascontiguousarray(verts), got, MPI.REPLACE); sf.bcastEnd(unit, np.ascontiguousarray(verts), got, MPI.REPLACE)
    unit.Free()
    fcent, _v, _c = ap._cell_centroids(lev)
    V = got.reshape(par.shape[0], cv_b.shape[1], -1)
    lam = _barycentric(fcent, V)
    bad = int((lam.min(axis=1) < -1e-9).sum())
    remote = int((par[:, 0] != comm.rank).sum())
    tot = comm.allreduce(bad); rem = comm.allreduce(remote); n = comm.allreduce(par.shape[0])
    if comm.rank == 0:
        print(f"provenance {tag}: {n} cells, {rem} with a remote parent, {tot} NOT inside their parent", flush=True)

check_parents(static[-1], levels[0], "base -> first add-on")
for l in range(1, len(levels)):
    check_parents(levels[l - 1], levels[l], f"add-on {l-1} -> {l}")
