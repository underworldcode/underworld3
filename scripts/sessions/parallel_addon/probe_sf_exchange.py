"""petsc4py star-forest semantics for the provenance exchange: bcast root
int64 pairs to leaves, reduce leaf pairs into roots (REPLACE)."""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
comm = MPI.COMM_WORLD
r, n = comm.rank, comm.size
nroots = 4
# every rank's leaves point at the next rank's roots, two leaves per root
remote = np.array([[(r + 1) % n, i // 2] for i in range(2 * nroots)], dtype=np.int32)
sf = PETSc.SF().create(comm=PETSc.COMM_WORLD)
sf.setGraph(nroots, None, remote)
sf.setUp()
root = np.array([[r, 10 * r + i] for i in range(nroots)], dtype=np.int64)
leaf = np.zeros((2 * nroots, 2), dtype=np.int64)
unit = MPI.INT64_T.Create_contiguous(2); unit.Commit()
sf.bcastBegin(unit, root, leaf, MPI.REPLACE); sf.bcastEnd(unit, root, leaf, MPI.REPLACE)
ok_b = bool((leaf[:, 0] == (r + 1) % n).all() and (leaf[:, 1] == 10 * ((r + 1) % n) + np.arange(2 * nroots) // 2).all())
# reduce: leaves write their (rank, index) into roots; two leaves per root -> MAX picks the larger index
leaf2 = np.array([[r, i] for i in range(2 * nroots)], dtype=np.int64)
root2 = np.full((nroots, 2), -1, dtype=np.int64)
unit1 = MPI.INT64_T
# reduce needs contiguous arrays; do it column by column
col = root2[:, 1].copy(); src = leaf2[:, 1].copy()
sf.reduceBegin(unit1, src, col, MPI.MAX); sf.reduceEnd(unit1, src, col, MPI.MAX)
ok_r = bool((col == 2 * np.arange(nroots) + 1).all())
print(f"rank {r}: bcast ok {ok_b}; reduce MAX ok {ok_r} (got {col.tolist()})", flush=True)
