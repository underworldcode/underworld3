import numpy as np
from mpi4py import MPI
import underworld3 as uw
from underworld3.utilities.line_cut import _global_extent, _label_cut_edges

comm = MPI.COMM_WORLD
H, WIDTH = 0.03, 0.04
def pieces():
    main = np.column_stack([np.linspace(0.25, 0.50, 12), np.full(12, 0.5)])
    cont = np.column_stack([np.linspace(0.55, 0.75, 9), np.full(9, 0.5)])
    s = np.linspace(0.0, 1.0, 8)
    splay = np.column_stack([0.38 + 0.12 * s, 0.5 + 0.18 * s])
    return [("Main", main), ("Cont", cont), ("Splay", splay)]
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
    cellSize=8 * H, regular=False, refinement=1, qdegree=2)
net = uw.meshing.FaultNetwork(pieces(), hierarchy=[n for n, _p in pieces()])
net.prepare(h=H, ligament=1.0, verbose=False)
net.build(base=base, width=WIDTH, realisation="ti", max_levels=1, seams="ligament")
mesh = net.mesh
dm = mesh.dm
info = net.info
lig = np.asarray(info["ligament"])
band = np.asarray(info["band"])
X = np.asarray(mesh.X.coords)
cS, cE = dm.getHeightStratum(0)
scale = _global_extent(dm)
counts = {}
for name, P in net.prepared:
    marked = _label_cut_edges(dm, [np.asarray(P, float)], 1e-9 * scale, "probe_" + name, 1)
    counts[name] = len(marked)
    for e in marked[:0]:
        pass
xr = (float(X[:, 0].min()), float(X[:, 0].max())) if len(X) else None
for r in range(comm.size):
    if comm.rank == r:
        print(f"rank {r}: cells {cE-cS}, band {band.sum()}, ligament {lig.sum()}, "
              f"x-range {xr}, edges on trace {counts}", flush=True)
    comm.Barrier()
if comm.rank == 0:
    print({k: v for k, v in info.items() if k in ("n_cells", "n_ligament_cells", "seams")})
# where are the ligament cells?
from underworld3.utilities.place_surface import _cell_centroids_of
ids, cen = _cell_centroids_of(dm, lig)
for r in range(comm.size):
    if comm.rank == r and len(ids):
        print(f"rank {r} ligament centroids x: {np.round(np.sort(cen[:,0]),3)}", flush=True)
    comm.Barrier()

# ---- dump for a picture
from underworld3.utilities.place_surface import _shared_point_flags, _cells_anticlockwise
vS, vE = dm.getDepthStratum(0)
pStart, pEnd = dm.getChart()
Xv = np.asarray(X)[:vE - vS]
cells = _cells_anticlockwise(dm, Xv)
shared = _shared_point_flags(dm).astype(bool)[vS - pStart: vE - pStart]
cat = np.zeros(len(cells), dtype=int)
cat[band] = 1
cat[lig] = 2
eS, eE = dm.getDepthStratum(1)
edges = []
for name, P in net.prepared:
    lbl = dm.getLabel("probe_" + name)
    if lbl.getStratumSize(1) == 0:
        continue
    for e in lbl.getStratumIS(1).getIndices():
        a, b = (int(q) - vS for q in dm.getCone(int(e)))
        edges.append((a, b))
import os
np.savez(os.path.join(os.environ.get("SP", "."), f"np2_rank{comm.rank}.npz"), X=Xv, cells=cells,
         cat=cat, shared=shared, edges=np.asarray(edges, dtype=int).reshape(-1, 2))
