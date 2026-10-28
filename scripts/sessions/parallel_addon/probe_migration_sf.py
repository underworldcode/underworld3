"""Does a per-cell integer field ride a shell-partition redistribution?

Checks the two PETSc routes a provenance array could take: the migration star
forest ``distribute()`` returns, read as a point map, and ``distributeField``
on a cell-wise section. The field is each cell's centroid x-coordinate (a
quantity we can re-derive on the new mesh), so a wrong mapping is visible.
"""
import numpy as np
import underworld3 as uw
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                         cellSize=0.1, qdegree=2)
dm = mesh.dm
cS, cE = dm.getHeightStratum(0)
cent = np.array([dm.computeCellGeometryFVM(c)[1][0] for c in range(cS, cE)])
# move the right-hand half of every rank's cells to the next rank
assign = np.full(cE - cS, comm.rank, dtype=np.int32)
assign[cent > 0.5] = (comm.rank + 1) % comm.size
work = dm.clone()
sec = PETSc.Section().create(comm=PETSc.COMM_WORLD)
sec.setChart(*work.getChart())
for c in range(cS, cE):
    sec.setDof(c, 1)
sec.setUp()
vec = PETSc.Vec().createSeq(sec.getStorageSize(), comm=PETSc.COMM_SELF)
arr = vec.getArray()
for c in range(cS, cE):
    arr[sec.getOffset(c)] = cent[c - cS]
vec.setArray(arr)
part = work.getPartitioner()
part.setType(PETSc.Partitioner.Type.SHELL)
order = np.argsort(assign, kind="stable").astype(np.int32)
sizes = np.bincount(assign, minlength=comm.size).astype(np.int32)
part.setShellPartition(comm.size, sizes=sizes, points=order)
sf = work.distribute()
print(f"rank {comm.rank}: migration sf {'None' if sf is None else type(sf).__name__}", flush=True)
ncS, ncE = work.getHeightStratum(0)
new_cent = np.array([work.computeCellGeometryFVM(c)[1][0] for c in range(ncS, ncE)])
# route 1: migration SF as a point map (leaf = new local point, root = old (rank, point))
if sf is not None:
    nroots, ilocal, iremote = sf.getGraph()
    nleaves = len(iremote)
    cells_leaf = [(i if ilocal is None else int(ilocal[i])) for i in range(nleaves)]
    depth_ok = sum(1 for i, q in enumerate(cells_leaf) if (ncS <= q < ncE) == (cS <= int(iremote[i][1]) < cE))
    print(f"rank {comm.rank}: sf nroots {nroots} (old chart size {dm.getChart()[1]}), leaves {nleaves} (new chart size {work.getChart()[1]}), "
          f"depth preserved on {depth_ok}/{nleaves}", flush=True)
    # route 2: distributeField
    newsec, newvec = work.distributeField(sf, sec, vec)
    narr = newvec.getArray()
    got = np.array([narr[newsec.getOffset(c)] for c in range(ncS, ncE)])
    err = np.abs(got - new_cent).max() if len(got) else 0.0
    print(f"rank {comm.rank}: distributeField cells {ncE - ncS}, max |field - recomputed centroid| = {err:.2e}", flush=True)
