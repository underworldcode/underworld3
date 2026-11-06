"""One long vertical fault crossed once by the np=2 seam: gather vs ligament."""
import os, numpy as np
from mpi4py import MPI
import underworld3 as uw
from underworld3.utilities.fault_contact import fault_pair_jumps

comm = MPI.COMM_WORLD
params = uw.Params(uw_seams="ligament", uw_eta_1=0.01, uw_h=0.02, uw_levels=1)
SEAMS, ETA1, H, LEVELS = str(params.uw_seams), float(params.uw_eta_1), float(params.uw_h), int(params.uw_levels)
WIDTH = 2 * H
main = np.column_stack([np.full(36, 0.5), np.linspace(0.15, 0.85, 36)])
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
    cellSize=4 * H, regular=False, refinement=1, qdegree=2)
net = uw.meshing.FaultNetwork([("Main", main)])
net.prepare(h=H, verbose=False)
net.build(base=base, width=WIDTH, realisation="split", max_levels=LEVELS, seams=SEAMS, band=2*H, ramp=6*H)
mesh = net.mesh
lig = net.ligament_cells()
n_lig = comm.allreduce(int(lig.sum()) if lig is not None else 0, op=MPI.SUM)
n_cells = comm.allreduce(int(mesh.dm.getHeightStratum(0)[1]), op=MPI.SUM)
n_pairs = comm.allreduce(len(mesh._fault_point_pairs["Main"]), op=MPI.SUM)
local_cells = comm.gather(int(mesh.dm.getHeightStratum(0)[1]), root=0)
x, y = mesh.X
v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
for wall in ("Bottom", "Top", "Left", "Right"):
    stokes.add_dirichlet_bc((0.0, 2.0 * (x - 0.5)), wall)
stokes.petsc_use_pressure_nullspace = True
stokes.tolerance = 1e-6
net.apply(stokes, eta_1=ETA1) if lig is not None else net.apply(stokes)
sol = net.solve(stokes)
info = stokes._rotated_freeslip_info
coords, jumps, normals = fault_pair_jumps(stokes, "Main", info)
if len(jumps):
    jn = np.einsum("ij,ij->i", jumps, normals)
    tang = np.linalg.norm(jumps - jn[:, None] * normals, axis=1)
    prof = np.column_stack([coords[:, 1], tang])
else:
    prof = np.zeros((0, 2))
allp = comm.gather(prof, root=0)
if comm.rank == 0:
    P = np.vstack(allp); P = P[np.argsort(P[:, 0])]
    np.savez(os.path.join(os.environ.get("SP", "."), f"long_{SEAMS}_np{comm.size}.npz"), prof=P, n_cells=n_cells, n_lig=n_lig, n_pairs=n_pairs, local=np.array(local_cells))
    print(f"[{SEAMS} np{comm.size}] cells {n_cells} (per rank {local_cells}), ligament {n_lig}, pairs {n_pairs}, "
          f"converged {sol.get('converged')}, fallbacks {getattr(stokes,'pc_fallbacks',{})}, peak {P[:,1].max():.4f}, "
          f"y-range of pairs {P[:,0].min():.3f}..{P[:,0].max():.3f}", flush=True)
