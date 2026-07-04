import numpy as np, glob, re, os
import underworld3 as uw
from underworld3.meshing.smoothing import _tri_cells, _signed_areas
D = "/tmp/metric_mesh/sat"
ix = []
for f in glob.glob(f"{D}/sat_a16z.mesh.T.*.h5"):
    m = re.search(r"\.mesh\.T\.(\d+)\.h5$", os.path.basename(f))
    if m:
        ix.append(int(m.group(1)))
print("a16z T-ckpts:", sorted(ix))
for i in sorted(ix):
    m = uw.discretisation.Mesh(f"{D}/sat_a16z.mesh.{i:05}.h5")
    tr = _tri_cells(m.dm)
    Xc = np.asarray(m.X.coords)
    sa = _signed_areas(Xc, tr)
    A = np.abs(sa)
    o = np.sign(np.median(sa)) or 1.0
    print(f"  ckpt {i:4d}: minA/meanA={A.min()/A.mean():.4f}  "
          f"valid={bool((sa*o).min() > 0)}")
print("[control a16s (pct=50): plateau ~0.27 valid; "
      "fold ⇒ minA→~0 / invalid]")
