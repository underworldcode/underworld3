"""Poor-cell detection — the metrics that actually predict FE /
Stokes-saddle-point conditioning, which bulk minA/meanA misses.

Per triangle:
  q   = 4√3·A / Σℓ²            shape quality   (1=equilateral, →0 sliver)
  ang_max                       largest interior angle (deg; →180 = killer)
  aspect = ℓ_max² / (2A)        longest edge / shortest altitude
  relsize = A / mean(A)         (coarsened cells ≫1)
  sizejump                      max area ratio across a shared edge
                                (mesh gradation the solver sees)

The Stokes line-search trips on the *worst* cells, not the mean, so
we report TAILS + COUNTS. The hypothesised equidist failure mode is
cells that are simultaneously LARGE (coarsened) AND STRETCHED (bump
keyed to local iso) → reported as n_big_thin (relsize>2 & aspect>4).

  python scripts/_cellquality.py [tags=a16c2,a16e[,a16ed]] [steps]
"""
import sys
import numpy as np
import underworld3 as uw
from underworld3.meshing.smoothing import _tri_cells
D = "/tmp/metric_mesh/sat"
tags = (sys.argv[1] if len(sys.argv) > 1
        else "a16c2,a16e").split(",")
steps = [int(s) for s in (sys.argv[2].split(",")
         if len(sys.argv) > 2 else "20,70,80,300".split(","))]


def metrics(tag, step):
    try:
        m = uw.discretisation.Mesh(
            f"{D}/sat_{tag}.mesh.{step:05}.h5")
    except Exception:
        return None
    tri = _tri_cells(m.dm)
    X = np.asarray(m.X.coords)[:, :2]
    v0, v1, v2 = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    a = np.linalg.norm(v1 - v2, axis=1)          # edge opp v0
    b = np.linalg.norm(v2 - v0, axis=1)          # opp v1
    c = np.linalg.norm(v0 - v1, axis=1)          # opp v2
    A = 0.5 * np.abs(np.cross(v1 - v0, v2 - v0))
    A = np.maximum(A, 1e-300)
    q = 4.0 * np.sqrt(3.0) * A / (a * a + b * b + c * c)

    def ang(o, p, r):                            # angle opposite o
        cosv = np.clip((p * p + r * r - o * o) / (2 * p * r),
                       -1.0, 1.0)
        return np.degrees(np.arccos(cosv))
    ang_max = np.maximum.reduce([ang(a, b, c), ang(b, c, a),
                                 ang(c, a, b)])
    Lmax = np.maximum.reduce([a, b, c])
    aspect = Lmax * Lmax / (2.0 * A)
    relsize = A / A.mean()

    # neighbour size-jump via shared edges
    et = {}
    for ti, (i, j, k) in enumerate(tri):
        for u, w in ((i, j), (j, k), (k, i)):
            et.setdefault((min(u, w), max(u, w)), []).append(ti)
    jr = [max(A[t]) / min(A[t]) for t in et.values() if len(t) == 2]
    jr = np.array(jr) if jr else np.array([1.0])

    big_thin = int(((relsize > 2.0) & (aspect > 4.0)).sum())
    return dict(
        n=len(tri),
        minA_meanA=A.min() / A.mean(),
        q_min=q.min(), q_p01=np.percentile(q, 1),
        n_q_lt02=int((q < 0.2).sum()),
        n_q_lt01=int((q < 0.1).sum()),
        ang_max=ang_max.max(),
        n_ang_gt150=int((ang_max > 150).sum()),
        n_ang_gt165=int((ang_max > 165).sum()),
        aspect_max=aspect.max(),
        aspect_p99=np.percentile(aspect, 99),
        jump_max=jr.max(), jump_p99=np.percentile(jr, 99),
        big_thin=big_thin)


hdr = ("tag   step |   n  minA/mn | q_min q_p01 n<.2 n<.1 |"
       " angMx >150 >165 | aspMx asp99 | jumpMx j99 | BIG&THIN")
for tag in tags:
    print("\n" + hdr)
    print("-" * len(hdr))
    for s in steps:
        r = metrics(tag, s)
        if r is None:
            print(f"{tag:5s} {s:4d} |  (no ckpt)")
            continue
        print(f"{tag:5s} {s:4d} | {r['n']:4d} {r['minA_meanA']:6.3f}"
              f" | {r['q_min']:.3f} {r['q_p01']:.3f} "
              f"{r['n_q_lt02']:4d} {r['n_q_lt01']:4d} | "
              f"{r['ang_max']:5.1f} {r['n_ang_gt150']:4d} "
              f"{r['n_ang_gt165']:4d} | {r['aspect_max']:5.1f} "
              f"{r['aspect_p99']:5.1f} | {r['jump_max']:5.1f} "
              f"{r['jump_p99']:4.1f} | {r['big_thin']:4d}")
print("\nStokes trips on the TAIL: high ang_max/n>165, high "
      "aspect_max, high jump_max, and BIG&THIN>0 — NOT minA/meanA "
      "(which a16c2≈a16e≈0.20 despite a16c2=0 vs a16e=10 DIVERGED).")
