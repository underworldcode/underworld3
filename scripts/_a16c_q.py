import sys, numpy as np, glob, re, os
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _tri_cells, _signed_areas, _edge_pairs)
D = "/tmp/metric_mesh/sat"
tag = sys.argv[1] if len(sys.argv) > 1 else "a16c"
ix = []
for f in glob.glob(f"{D}/sat_{tag}.mesh.T.*.h5"):
    m = re.search(r"\.mesh\.T\.(\d+)\.h5$", os.path.basename(f))
    if m:
        ix.append(int(m.group(1)))
print(f"{tag} T-ckpts:", sorted(ix))
for i in sorted(ix):
    m = uw.discretisation.Mesh(f"{D}/sat_{tag}.mesh.{i:05}.h5")
    tr = _tri_cells(m.dm)
    Xc = np.asarray(m.X.coords)
    sa = _signed_areas(Xc, tr)
    A = np.abs(sa)
    o = np.sign(np.median(sa)) or 1.0
    ep = _edge_pairs(m.dm)
    el = np.linalg.norm(Xc[ep[:, 1]] - Xc[ep[:, 0]], axis=1)
    print(f"  ckpt {i:4d}: minA/meanA={A.min()/A.mean():.4f}  "
          f"valid={bool((sa*o).min() > 0)}  "
          f"edge max/min={el.max()/el.min():.2f}  "
          f"edge p95/p05={np.percentile(el,95)/np.percentile(el,5):.2f}")
print("[refine-only a16s: minA/meanA~0.27, p95/p05~2 (no de-res). "
      "a16c cc=4: 0.04-0.14 (OVER-coarse, slivers), p95/p05~5.4. "
      "Want: minA/meanA back toward ~0.2+ WITH p95/p05 > 2.5 "
      "(some de-res, clean mesh) — the quality knee.]")
