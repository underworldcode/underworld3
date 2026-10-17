"""Cost of partition_addon on a larger adapt-on-top fault: wall time of the call
and rank 0's cProfile of it (top cumulative entries).

mpirun -np N python probe_partition_cost.py [cell_size=0.02] [k=2] [dim=2]
"""
import cProfile, io, pstats, sys, time
import numpy as np
import underworld3 as uw
from mpi4py import MPI
from underworld3.utilities import addon_partition as ap

comm = MPI.COMM_WORLD
h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
dim = int(sys.argv[3]) if len(sys.argv) > 3 else 2
if dim == 2:
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=h, refinement=1, qdegree=2)
    A, B = np.array([0.3, 0.45]), np.array([0.7, 0.55])
    pts = np.column_stack([np.linspace(A[0], B[0], 9), np.linspace(A[1], B[1], 9), np.zeros(9)])
    guide = uw.meshing.Surface("guide", base, pts); guide.discretize()
    child = base.adapt(guide.refinement_metric_function(h_near=h / 2**k, h_far=h, width=4 * h, profile="linear"), max_levels=k)
    marks = ap.mark_polylines([np.vstack([A, B])])
    layers = 1
else:
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=h, regular=False, qdegree=2, refinement=1)
    c = np.array([0.5, 0.5, 0.5]); n = np.array([0.0, 1.0, 0.0])
    def metric(X):
        d = np.abs((np.asarray(X)[:, :3] - c) @ n)
        hh = np.clip(h / 2**k + d * 0.5, h / 2**k, h)
        return 1.0 / hh**2
    child = base.adapt(metric, max_levels=k)
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    rim = c + 0.25 * np.column_stack([np.cos(a), np.zeros(24), np.sin(a)])
    sheet_pts = np.vstack([c, rim]); tris = np.array([(0, 1 + i, 1 + (i + 1) % 24) for i in range(24)])
    marks = ap.mark_sheets([(sheet_pts, tris)], 0.6)
    layers = 2
cS, cE = child.dm.getHeightStratum(0)
ncell = comm.allreduce(cE - cS)
prof = cProfile.Profile()
comm.Barrier(); t = time.perf_counter(); prof.enable()
out = ap.partition_addon(child, marks, layers=layers)
prof.disable(); comm.Barrier(); dt = time.perf_counter() - t
rec = out._addon_partition
if comm.rank == 0:
    print(f"np {comm.size} dim {dim} h {h} k {k}: finest cells {ncell}, partition {dt:.2f} s, units moved {rec['units_moved']}, sweeps {rec['sweeps']}", flush=True)
    s = io.StringIO(); st = pstats.Stats(prof, stream=s); st.sort_stats("tottime").print_stats(25); st.print_callers("_get_coords_for_basis")
    print("\n".join(l for l in s.getvalue().splitlines() if ("ncalls" in l or "(" in l)), flush=True)
