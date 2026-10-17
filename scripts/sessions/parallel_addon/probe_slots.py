import numpy as np, underworld3 as uw
from mpi4py import MPI
from underworld3.utilities.fault_split import split_fault
Lx, Ly, H = 3.0, 2.0, 1.0; h, hf, f = 0.2, 0.1, 4
yf, xa, xb, zb = 1.0, 1.0, 2.0, 0.5; ztop = H - 2 * hf
patch = np.array([[xa, yf, zb], [xb, yf, zb], [xb, yf, ztop], [xa, yf, ztop]])
fine = uw.meshing.BoxInternalPatch(cellSize=f * h, minCoords=(0, 0, 0), maxCoords=(Lx, Ly, H), patch_points=patch, patch_name="Fault",
                                   patch_cellSize=f * hf, grading_distance=8 * hf, refinement=2, qdegree=3)
mesh = split_fault(fine, "Fault")
tail = mesh._custom_mg_coarse_meshes
if MPI.COMM_WORLD.rank == 0:
    print("np", MPI.COMM_WORLD.size, "unsplit slot", getattr(fine, "_refine_slot", None) is not None and fine._refine_slot[1],
          "split slot", getattr(mesh, "_refine_slot", None) is not None and mesh._refine_slot[1],
          "tail slots", [w._refine_slot[1] for w in tail], "same token split/tail", getattr(mesh, "_refine_slot", (None,))[0] is tail[-1]._refine_slot[0] if getattr(mesh, "_refine_slot", None) else None, flush=True)
rp = getattr(mesh, "_adapt_prolongation", None)
rpf = getattr(fine, "_adapt_prolongation", None)
if MPI.COMM_WORLD.rank == 0:
    print("np", MPI.COMM_WORLD.size, "split child _adapt_prolongation:", type(rp).__name__, getattr(rp, "shape", None) if rp is not None else None,
          "| unsplit:", type(rpf).__name__, getattr(rpf, "shape", None) if rpf is not None else None, flush=True)
