import numpy as np, glob, re, os
import underworld3 as uw
from underworld3.meshing.smoothing import _tri_cells, _signed_areas
D = "/tmp/metric_mesh/sat"
ix = []
for f in glob.glob(f"{D}/sat_a16x.mesh.T.*.h5"):
    m = re.search(r"\.mesh\.T\.(\d+)\.h5$", os.path.basename(f))
    if m:
        ix.append(int(m.group(1)))
print("a16x T-ckpts:", sorted(ix))
i = max(ix)
m = uw.discretisation.Mesh(f"{D}/sat_a16x.mesh.{i:05}.h5")
tr = _tri_cells(m.dm)
A = np.abs(_signed_areas(np.asarray(m.X.coords), tr))
print(f"a16x (amp=24) ckpt {i}: minA/meanA = {A.min()/A.mean():.4f}"
      f"   [a16s amp=16 plateau ~0.27; cumulative-fail ~0.00]")
