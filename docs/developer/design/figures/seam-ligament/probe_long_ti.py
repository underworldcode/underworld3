"""One long vertical fault crossed by the seam: the TI realisation, gather vs ligament."""
import os, numpy as np
from mpi4py import MPI
import underworld3 as uw
comm = MPI.COMM_WORLD
params = uw.Params(uw_seams="ligament", uw_eta_1=0.01, uw_h=0.02)
SEAMS, ETA1, H = str(params.uw_seams), float(params.uw_eta_1), float(params.uw_h)
WIDTH = 2 * H
main = np.column_stack([np.full(36, 0.5), np.linspace(0.15, 0.85, 36)])
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=4 * H, regular=False, refinement=1, qdegree=2)
net = uw.meshing.FaultNetwork([("Main", main)])
net.prepare(h=H, verbose=False)
net.build(base=base, width=WIDTH, realisation="ti", max_levels=1, seams=SEAMS, band=2*H, ramp=6*H)
mesh = net.mesh
lig = net.ligament_cells()
n_lig = comm.allreduce(int(lig.sum()) if lig is not None else 0, op=MPI.SUM)
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
net.apply(stokes, eta_1=ETA1)
stokes.solve()
peak = comm.allreduce(float(net.slips(stokes).get("Main", 0.0)), op=MPI.MAX)
# the slip PROFILE: tangential jump across the band at probe points along the trace
ys = np.linspace(0.17, 0.83, 34)
off = 0.5 * WIDTH + H
left = np.column_stack([np.full(len(ys), 0.5 - off), ys]); right = np.column_stack([np.full(len(ys), 0.5 + off), ys])
vl = uw.function.evaluate(v.sym, left); vr = uw.function.evaluate(v.sym, right)
own = (mesh._robust_owning_cells(np.ascontiguousarray(left)) >= 0) & (mesh._robust_owning_cells(np.ascontiguousarray(right)) >= 0)
vl = np.asarray(vl).reshape(len(ys), -1); vr = np.asarray(vr).reshape(len(ys), -1)
jump = np.where(own, np.abs(vr[:, 1] - vl[:, 1]), -1.0)
allj = comm.gather(jump, root=0)
if comm.rank == 0:
    J = np.max(np.vstack(allj), axis=0)
    np.savez(os.path.join(os.environ.get("SP", "."), f"longti_{SEAMS}_np{comm.size}.npz"), ys=ys, jump=J, n_lig=n_lig, local=np.array(local_cells))
    print(f"[ti {SEAMS} np{comm.size}] per rank {local_cells}, ligament {n_lig}, peak(slips) {peak:.4f}, profile max {J.max():.4f}, "
          f"unowned probes {(J < 0).sum()}", flush=True)
