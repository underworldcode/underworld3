"""np=2: a VERTICAL network (the seam crosses it) — gather vs ligament."""
import os, numpy as np
from mpi4py import MPI
import underworld3 as uw

comm = MPI.COMM_WORLD
params = uw.Params(uw_seams="ligament", uw_eta_1=0.01)
SEAMS = str(params.uw_seams)
ETA1 = float(params.uw_eta_1)
H, WIDTH = 0.03, 0.04

def pieces():
    main = np.column_stack([np.full(12, 0.5), np.linspace(0.25, 0.50, 12)])
    cont = np.column_stack([np.full(9, 0.5), np.linspace(0.55, 0.75, 9)])
    s = np.linspace(0.0, 1.0, 8)
    splay = np.column_stack([0.5 + 0.18 * s, 0.38 + 0.12 * s])
    return [("Main", main), ("Cont", cont), ("Splay", splay)]

base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
    cellSize=8 * H, regular=False, refinement=1, qdegree=2)
net = uw.meshing.FaultNetwork(pieces(), hierarchy=[n for n, _p in pieces()])
net.prepare(h=H, ligament=1.0, verbose=False)
net.build(base=base, width=WIDTH, realisation="split", max_levels=1, seams=SEAMS)
mesh = net.mesh
info = net.info
lig = net.ligament_cells()
n_lig = comm.allreduce(int(lig.sum()) if lig is not None else 0, op=MPI.SUM)
n_cells = comm.allreduce(int(mesh.dm.getHeightStratum(0)[1]), op=MPI.SUM)
pairs = {k: comm.allreduce(len(v), op=MPI.SUM) for k, v in mesh._fault_point_pairs.items()}
uw.pprint(f"[{SEAMS}] cells {n_cells}, ligament cells {n_lig}, pairs {pairs}")

x, y = mesh.X
v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = net.junction_patch(eta_0=1.0)
for wall in ("Bottom", "Top", "Left", "Right"):
    stokes.add_dirichlet_bc((0.0, 2.0 * (x - 0.5)), wall)
stokes.petsc_use_pressure_nullspace = True
stokes.tolerance = 1e-5
if lig is not None:
    net.apply(stokes, eta_1=ETA1)
else:
    net.apply(stokes)
sol = net.solve(stokes)
local = net.slips(stokes)
peaks = {n: comm.allreduce(float(local.get(n, 0.0)), op=MPI.MAX) for n in ("Main", "Cont", "Splay")}
uw.pprint(f"[{SEAMS}] converged {sol.get('converged')}, fallbacks {getattr(stokes, 'pc_fallbacks', {})}, peaks {peaks}")

# dump for a picture
from underworld3.utilities.place_surface import _shared_point_flags, _cells_anticlockwise
dm = mesh.dm
vS, vE = dm.getDepthStratum(0); pStart, _ = dm.getChart()
Xv = np.asarray(mesh.X.coords)[:vE - vS]
cells = _cells_anticlockwise(dm, Xv)
shared = _shared_point_flags(dm).astype(bool)[vS - pStart: vE - pStart]
cat = np.zeros(len(cells), dtype=int); cat[np.asarray(info["band"])] = 1
if lig is not None: cat[lig] = 2
edges = []
for name in ("Main", "Cont", "Splay"):
    lbl = name + "Plus"
    if dm.hasLabel(lbl):
        val = int(mesh.boundaries[lbl].value)
        if dm.getLabel(lbl).getStratumSize(val):
            for e in dm.getLabel(lbl).getStratumIS(val).getIndices():
                a, b = (int(q) - vS for q in dm.getCone(int(e)))
                edges.append((a, b))
np.savez(os.path.join(os.environ.get("SP", "."), f"cross_{SEAMS}_rank{comm.rank}.npz"), X=Xv, cells=cells,
         cat=cat, shared=shared, edges=np.asarray(edges, dtype=int).reshape(-1, 2))
