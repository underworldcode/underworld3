"""Cost of split_fault: wall time and the top own-time entries, 2-D cut and 3-D patch.

python probe_split_cost.py [dim=3]
"""
import cProfile, io, pstats, sys, time
import numpy as np
import underworld3 as uw
from underworld3.utilities.fault_split import split_fault

dim = int(sys.argv[1]) if len(sys.argv) > 1 else 3
if dim == 3:
    patch = np.array([[1.0, 1.0, 0.5], [2.0, 1.0, 0.5], [2.0, 1.0, 0.8], [1.0, 1.0, 0.8]])
    mesh = uw.meshing.BoxInternalPatch(cellSize=0.8, minCoords=(0, 0, 0), maxCoords=(3, 2, 1), patch_points=patch,
                                       patch_name="Fault", patch_cellSize=0.4, grading_distance=0.8, refinement=2, qdegree=3)
    name = "Fault"
else:
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.005, qdegree=2)
    from underworld3.utilities.fault_split import add_fault
    prof0 = cProfile.Profile()
    mesh = None
cS, cE = (mesh.dm.getHeightStratum(0) if mesh is not None else (0, 0))
prof = cProfile.Profile(); t = time.perf_counter(); prof.enable()
if dim == 3:
    out = split_fault(mesh, name)
else:
    out = base.add_fault([("Flt", np.array([[0.3, 0.45], [0.7, 0.55]]))])
    cS, cE = base.dm.getHeightStratum(0)
prof.disable(); dt = time.perf_counter() - t
print(f"dim {dim}: cells {cE - cS}, split {dt:.2f} s", flush=True)
s = io.StringIO(); pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(14)
print("\n".join(l for l in s.getvalue().splitlines() if "(" in l)[:4000], flush=True)
