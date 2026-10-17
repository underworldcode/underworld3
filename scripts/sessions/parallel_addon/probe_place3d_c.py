import numpy as np, underworld3 as uw, time
from underworld3.utilities.place_surface import place_sheet
h_base = 0.075; k = 2
n = 7
yy, zz = np.meshgrid(np.linspace(0.3, 0.7, n), np.linspace(0.3, 0.7, n), indexing="ij")
pts = np.column_stack([np.full(n * n, 0.5), yy.ravel(), zz.ravel()])
rim = lambda i, j: i in (0, n - 1) or j in (0, n - 1)
tris = []
for i in range(n - 1):
    for j in range(n - 1):
        a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
        ra, rb, rc, rd = rim(i, j), rim(i + 1, j), rim(i + 1, j + 1), rim(i, j + 1)
        tris += [[a, b, d], [b, c, d]] if ((ra and rb and rc) or (ra and rc and rd)) else [[a, b, c], [a, c, d]]
tris = np.array(tris)
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=2 * h_base, refinement=1, qdegree=2)
def metric(X):
    d_plane = np.abs(X[:, 0] - 0.5)
    d_rect = np.maximum(0.0, np.maximum(np.abs(X[:, 1] - 0.5) - 0.2, np.abs(X[:, 2] - 0.5) - 0.2))
    d = np.sqrt(d_plane**2 + d_rect**2)
    h_near, h_far, width = h_base / 2**k, h_base, 0.15
    h = np.minimum(h_far, h_near + (h_far - h_near) * d / width)
    return 1.0 / h**2
child = base.adapt(metric, max_levels=k)
print("child cells", child.dm.getHeightStratum(0)[1], flush=True)
for kw in [dict(clearance=1.2), dict(clearance=1.5), dict(size=0.02, clearance=0.9), dict(size=0.02, clearance=1.2), dict(size=0.04, clearance=0.9)]:
    t0 = time.perf_counter()
    try:
        d, info = place_sheet(child.dm, pts, tris, label="Flt", label_value=40, **kw)
        print(kw, "OK", {k_: v for k_, v in info.items() if not hasattr(v, "shape")}, round(time.perf_counter() - t0, 1), flush=True)
    except Exception as e:
        print(kw, "FAIL", str(e)[:160], round(time.perf_counter() - t0, 1), flush=True)
