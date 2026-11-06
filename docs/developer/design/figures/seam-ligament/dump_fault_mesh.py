"""Dump the split long-fault mesh (serial or ligament) for the diagram."""
import os, numpy as np
from mpi4py import MPI
import underworld3 as uw
from underworld3.utilities.place_surface import _shared_point_flags, _cells_anticlockwise, _cell_centroids_of
comm = MPI.COMM_WORLD
params = uw.Params(uw_seams="ligament")
SEAMS = str(params.uw_seams)
H = 0.02
main = np.column_stack([np.full(36, 0.5), np.linspace(0.15, 0.85, 36)])
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=4 * H, regular=False, refinement=1, qdegree=2)
net = uw.meshing.FaultNetwork([("Main", main)])
net.prepare(h=H, verbose=False)
net.build(base=base, width=2 * H, realisation="split", max_levels=1, seams=SEAMS, band=2*H, ramp=6*H)
mesh = net.mesh; dm = mesh.dm
lig = net.ligament_cells()
x, y = mesh.X
v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
if lig is not None:
    net.apply(stokes, eta_1=0.01)
    eta = np.asarray(net.ti["eta_1"].array).ravel()
else:
    net.apply(stokes)
    eta = np.ones(int(dm.getHeightStratum(0)[1]))
vS, vE = dm.getDepthStratum(0); pStart, _ = dm.getChart()
X = np.asarray(mesh.X.coords)[:vE - vS]
cells = _cells_anticlockwise(dm, X)
shared = _shared_point_flags(dm).astype(bool)[vS - pStart: vE - pStart]
cat = np.zeros(len(cells), dtype=int); cat[np.asarray(net.info["band"])] = 1
if lig is not None: cat[lig] = 2
edges = {}
for side in ("Plus", "Minus"):
    lbl = "Main" + side; out = []
    if dm.hasLabel(lbl):
        val = int(mesh.boundaries[lbl].value)
        if dm.getLabel(lbl).getStratumSize(val):
            for e in dm.getLabel(lbl).getStratumIS(val).getIndices():
                a, b = (int(q) - vS for q in dm.getCone(int(e)))
                out.append((a, b))
    edges[side] = np.asarray(out, dtype=int).reshape(-1, 2)
# tips: vertices of degree 1 in the Plus-edge graph
from collections import Counter
deg = Counter(edges["Plus"].ravel().tolist())
tips = np.array([vtx for vtx, d in deg.items() if d == 1], dtype=int)
# replica pairs (minus -> plus vertex ids) for drawing the duplicated nodes
pairs = mesh._fault_point_pairs["Main"]
rep = np.array([[qm - vS, qp - vS] for qm, qp in pairs.items() if vS <= qm < vE and vS <= qp < vE], dtype=int).reshape(-1, 2)
np.savez(os.path.join(os.environ.get("SP", "."), f"faultmesh_{SEAMS}_np{comm.size}_rank{comm.rank}.npz"),
         X=X, cells=cells, cat=cat, eta=eta, shared=shared, plus=edges["Plus"], minus=edges["Minus"], tips=tips, rep=rep)
if comm.rank == 0:
    print(f"[{SEAMS} np{comm.size}] dumped", flush=True)
