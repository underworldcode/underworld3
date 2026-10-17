import numpy as np, underworld3 as uw
from underworld3.utilities.place_surface import place_sheet
h_base = 0.1
n = 5
yy, zz = np.meshgrid(np.linspace(0.3, 0.7, n), np.linspace(0.3, 0.7, n), indexing="ij")
pts = np.column_stack([np.full(n * n, 0.5), yy.ravel(), zz.ravel()])
tris = []
for i in range(n - 1):
    for j in range(n - 1):
        a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
        tris += [[a, b, c], [a, c, d]]
tris = np.array(tris)
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=2 * h_base, refinement=1, qdegree=2)
def metric(X):
    d_plane = np.abs(X[:, 0] - 0.5)
    d_rect = np.maximum(0.0, np.maximum(np.abs(X[:, 1] - 0.5) - 0.2, np.abs(X[:, 2] - 0.5) - 0.2))
    d = np.sqrt(d_plane**2 + d_rect**2)
    h = np.minimum(h_base, h_base / 2 + (h_base / 2) * d / 0.15)
    return 1.0 / h**2
child = base.adapt(metric, max_levels=1)
from petsc4py import PETSc
dm = child.dm
cS, cE = dm.getHeightStratum(0)
vols = np.array([dm.computeCellGeometryFVM(c)[0] for c in range(cS, cE)])
print("child cells", cE - cS, "min vol", vols.min(), "neg", (vols < 0).sum(), "sum", vols.sum(), flush=True)
for kw in [dict(clearance=0.3), dict(clearance=0.9), dict(clearance=1.2), dict(size=0.05), dict(size=0.05, clearance=0.9)]:
    try:
        d, info = place_sheet(child.dm, pts, tris, label="Flt", label_value=40, **kw)
        print(kw, "OK", {k: v for k, v in info.items() if not hasattr(v, "shape")}, flush=True)
    except Exception as e:
        print(kw, "FAIL", str(e)[:160], flush=True)
