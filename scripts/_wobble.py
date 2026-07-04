"""Quantify mesh 'wobble': frame-to-frame variation of the adapted
mesh over the SETTLED window, a16e (equidist, parameter-free) vs
a16c2 (legacy hand-tuned cc=2). Same underlying convection (same
Ra, same steps) ⇒ any difference is the metric strategy.

Metrics over consecutive settled checkpoints:
  • std of minA/meanA           (mesh-quality jitter)
  • mean & std of RMS |Δx|/h0   (how far nodes shuffle per event)
  • std of edge p95/p05         (grading-strength breathing)
Higher std ⇒ wobblier (mesh re-floats more between events)."""
import numpy as np, glob, re, os
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _tri_cells, _signed_areas, _edge_pairs)
D = "/tmp/metric_mesh/sat"


def series(tag, idx_min=150):
    fs = glob.glob(f"{D}/sat_{tag}.mesh.T.*.h5")
    ix = sorted(int(re.search(r"\.T\.(\d+)\.h5$",
                f).group(1)) for f in fs)
    ix = [i for i in ix if i >= idx_min]
    qa, sp, X = [], [], None
    disp = []
    h0 = None
    for i in ix:
        m = uw.discretisation.Mesh(f"{D}/sat_{tag}.mesh.{i:05}.h5")
        Xc = np.asarray(m.X.coords)
        tr = _tri_cells(m.dm)
        A = np.abs(_signed_areas(Xc, tr))
        qa.append(A.min() / A.mean())
        ep = _edge_pairs(m.dm)
        el = np.linalg.norm(Xc[ep[:, 1]] - Xc[ep[:, 0]], axis=1)
        if h0 is None:
            h0 = el.mean()
        sp.append(np.percentile(el, 95) / np.percentile(el, 5))
        if X is not None and X.shape == Xc.shape:
            disp.append(np.sqrt(((Xc - X) ** 2).sum(1)).mean() / h0)
        X = Xc
    qa, sp, disp = map(np.array, (qa, sp, disp))
    print(f"{tag:>6} (settled, {len(ix)} ckpts idx≥{idx_min}):")
    print(f"   minA/meanA  mean={qa.mean():.3f}  std={qa.std():.4f}"
          f"  range=[{qa.min():.3f},{qa.max():.3f}]")
    print(f"   p95/p05     mean={sp.mean():.3f}  std={sp.std():.4f}")
    print(f"   RMS|dx|/h0 per ckpt  mean={disp.mean():.4f}  "
          f"std={disp.std():.4f}  max={disp.max():.4f}")


for t in ("a16e", "a16c2"):
    series(t)
print("Wobble = larger minA/meanA std, p95/p05 std, and per-ckpt "
      "RMS|dx| ⇒ the equidist normalisation re-floats G + the whole "
      "density field every event (more field-responsive); cc=2's "
      "percentile+fixed cap anchors the magnitude (more damped).")
