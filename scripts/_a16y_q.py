import numpy as np, glob, re, os
import underworld3 as uw
from underworld3.meshing.smoothing import _tri_cells, _signed_areas
D = "/tmp/metric_mesh/sat"
ix = []
for f in glob.glob(f"{D}/sat_a16y.mesh.T.*.h5"):
    m = re.search(r"\.mesh\.T\.(\d+)\.h5$", os.path.basename(f))
    if m:
        ix.append(int(m.group(1)))
print("a16y T-ckpts:", sorted(ix))
for i in sorted(ix):
    m = uw.discretisation.Mesh(f"{D}/sat_a16y.mesh.{i:05}.h5")
    tr = _tri_cells(m.dm)
    A = np.abs(_signed_areas(np.asarray(m.X.coords), tr))
    o = np.sign(np.median(_signed_areas(np.asarray(m.X.coords), tr)))
    valid = bool((_signed_areas(np.asarray(m.X.coords), tr) * o).min() > 0)
    print(f"  ckpt {i:4d}: minA/meanA={A.min()/A.mean():.4f}  "
          f"valid={valid}")
print("[cap=4 a16s plateau ~0.27; cumulative-fail ~0.00; "
      "fold ⇒ minA→~0 / invalid]")
