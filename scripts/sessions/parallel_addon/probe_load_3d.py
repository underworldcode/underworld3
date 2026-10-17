"""Per-rank load of the split refined patch box after the add-on partition."""
import numpy as np, underworld3 as uw
from mpi4py import MPI
from underworld3.utilities.fault_split import split_fault
comm = MPI.COMM_WORLD
Lx, Ly, H = 3.0, 2.0, 1.0; h, hf, f = 0.2, 0.1, 4
yf, xa, xb, zb = 1.0, 1.0, 2.0, 0.5; ztop = H - 2 * hf
patch = np.array([[xa, yf, zb], [xb, yf, zb], [xb, yf, ztop], [xa, yf, ztop]])
fine = uw.meshing.BoxInternalPatch(cellSize=f * h, minCoords=(0, 0, 0), maxCoords=(Lx, Ly, H), patch_points=patch, patch_name="Fault",
                                   patch_cellSize=f * hf, grading_distance=8 * hf, refinement=2, qdegree=3)
mesh = split_fault(fine, "Fault", verbose=True)
rec = getattr(mesh.parent, "_addon_partition", None)
nodes = comm.gather(np.asarray(mesh._get_coords_for_basis(2, True)).shape[0], root=0)
cells = comm.gather(mesh.dm.getHeightStratum(0)[1] - mesh.dm.getHeightStratum(0)[0], root=0)
if comm.rank == 0:
    n = np.array(nodes, float); c = np.array(cells, float)
    print(f"np {comm.size}: finest P2 nodes per rank {nodes} max/mean {n.max()/n.mean():.2f}; cells {cells} max/mean {c.max()/c.mean():.2f}", flush=True)
    if rec: print("record:", {k: v for k, v in rec.items() if k != 'owners'}, flush=True)
