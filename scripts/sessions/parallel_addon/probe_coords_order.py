"""On the split patch box: set v = x from v.coords and evaluate it back at random points.
A mismatch between the coordinate order and the section order shows as evaluation error."""
import numpy as np, underworld3 as uw
from mpi4py import MPI
from underworld3.utilities.fault_split import split_fault
rank0 = MPI.COMM_WORLD.rank == 0
Lx, Ly, H = 3.0, 2.0, 1.0; h, hf, f = 0.2, 0.1, 4
yf, xa, xb, zb = 1.0, 1.0, 2.0, 0.5; ztop = H - 2 * hf
patch = np.array([[xa, yf, zb], [xb, yf, zb], [xb, yf, ztop], [xa, yf, ztop]])
fine = uw.meshing.BoxInternalPatch(cellSize=f * h, minCoords=(0, 0, 0), maxCoords=(Lx, Ly, H), patch_points=patch, patch_name="Fault",
                                   patch_cellSize=f * hf, grading_distance=8 * hf, refinement=2, qdegree=3)
for label, mesh in (("unsplit", fine), ("split", split_fault(fine, "Fault"))):
    v = uw.discretisation.MeshVariable(f"v_{label}", mesh, 3, degree=2)
    c = np.asarray(v.coords)
    v.array[:, 0, 0] = c[:, 0]; v.array[:, 0, 1] = c[:, 1]; v.array[:, 0, 2] = c[:, 2]
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0.05, 2.95, 400), rng.uniform(0.05, 1.95, 400), rng.uniform(0.05, 0.95, 400)])
    U = np.asarray(uw.function.global_evaluate(v.sym, pts)).reshape(-1, 3)
    err = np.abs(U - pts).max(axis=1)
    if rank0:
        print(f"{label} np {MPI.COMM_WORLD.size}: v = x evaluated back: max err {err.max():.2e}, points with err > 1e-6: {(err > 1e-6).sum()} of {len(err)}", flush=True)
