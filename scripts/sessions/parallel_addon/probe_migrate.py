import numpy as np, underworld3 as uw
from mpi4py import MPI
from underworld3.utilities.fault_split import migrate_point_pairs, redistribute_by_assignment
comm = MPI.COMM_WORLD
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, refinement=1, qdegree=3)
sheet = base.add_fault([("Flt", np.array([[0.25, 0.5], [0.75, 0.5]]))])
dm = sheet.dm
def coord_of(dm, q):
    cs = dm.getCoordinateSection(); cv = dm.getCoordinatesLocal().array
    vS, vE = dm.getDepthStratum(0); cdim = dm.getCoordinateDim()
    verts = [int(v) for v in dm.getTransitiveClosure(int(q))[0] if vS <= v < vE]
    return np.array([cv[cs.getOffset(v):cs.getOffset(v)+cdim] for v in verts]).mean(axis=0)
pairs = sheet._fault_point_pairs["Flt"]
gap = max([np.linalg.norm(coord_of(dm, a) - coord_of(dm, b)) for a, b in pairs.items()] or [0.0])
print(comm.rank, "before: pairs", len(pairs), "max gap", gap, flush=True)
cS, cE = dm.getHeightStratum(0)
g, sf = redistribute_by_assignment(dm, np.zeros(cE - cS, dtype=np.int32))
print(comm.rank, "sf", None if sf is None else sf.getGraph()[0], "new chart", g.getChart(), flush=True)
newp, dropped = migrate_point_pairs({"Flt": pairs}, sf, dm, g)
np_ = newp["Flt"]
gap2 = max([np.linalg.norm(coord_of(g, a) - coord_of(g, b)) for a, b in np_.items()] or [0.0])
print(comm.rank, "after: pairs", len(np_), "dropped", dropped, "max gap", gap2, flush=True)
