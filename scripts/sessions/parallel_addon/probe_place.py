import numpy as np, underworld3 as uw, time
Lx, Ly, H = 3.0, 2.0, 1.0; h, hf = 0.15, 0.05
yf = 1.0; xa, xb = 1.0, 2.0; zb = 0.5; setback = 4 * hf; ztop = H - setback
def dist(pts):
    pts = np.asarray(pts, float).reshape(-1, 3)
    dx = np.maximum(np.maximum(xa - pts[:, 0], pts[:, 0] - xb), 0.0)
    dz = np.maximum(zb - pts[:, 2], 0.0)
    return np.sqrt(dx**2 + (pts[:, 1] - yf)**2 + dz**2)
def metric(pts):
    d = dist(pts); hh = np.where(d < hf, hf, np.minimum(hf + (h - hf) * (d - hf) / (6 * hf), h)); return 1 / hh**2
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0, 0), maxCoords=(Lx, Ly, H), cellSize=2 * h, regular=False, qdegree=3, refinement=1)
child = base.adapt(metric, max_levels=2)
X = child.X.coords
print("child cells", child.dm.getHeightStratum(0)[1], "points", len(X))
# local edge lengths near the top above the fault
top = (X[:, 2] > 0.97) & (np.abs(X[:, 1] - yf) < 0.1) & (X[:, 0] > xa) & (X[:, 0] < xb)
print("vertices on top face above the fault:", top.sum())
nx = int(round((xb - xa) / hf)) + 1; nz = int(round((H + 0.3 * hf - zb) / hf)) + 1
xs = np.linspace(xa, xb, nx); zs = np.linspace(zb, H + 0.3 * hf, nz)
XX, ZZ = np.meshgrid(xs, zs, indexing="ij")
pts = np.column_stack([XX.ravel(), np.full(XX.size, yf), ZZ.ravel()])
tris = []
for i in range(nx - 1):
    for j in range(nz - 1):
        a, b, c, d = i * nz + j, (i + 1) * nz + j, (i + 1) * nz + j + 1, i * nz + j + 1
        tris += [[a, b, c], [a, c, d]]
tris = np.array(tris)
for sb, cl in ((5 * hf, 0.9), (6 * hf, 0.9), (6 * hf, 0.6)):
    if sb == setback and cl == 0.9 and 'done_first' in dir(): 
        # variant: sheet NOT running past the top, stopping at ztop exactly
        zs2 = np.linspace(zb, ztop, int(round((ztop - zb) / hf)) + 1)
        XX2, ZZ2 = np.meshgrid(xs, zs2, indexing="ij"); pts2 = np.column_stack([XX2.ravel(), np.full(XX2.size, yf), ZZ2.ravel()])
        nz2 = len(zs2); tris2 = []
        for i in range(nx - 1):
            for j in range(nz2 - 1):
                a, b, c, d = i * nz2 + j, (i + 1) * nz2 + j, (i + 1) * nz2 + j + 1, i * nz2 + j + 1
                tris2 += [[a, b, c], [a, c, d]]
        label = f"interior sheet to z={ztop:.2f}, no setback, clearance {cl}"
        try:
            t = time.perf_counter(); cut = child.add_conforming_sheet(pts2, np.array(tris2), "Fault", setback=0.0, clearance=cl, verbose=True)
            print("OK:", label, f"{time.perf_counter()-t:.1f}s cells", cut.dm.getHeightStratum(0)[1])
        except Exception as e: print("REFUSED:", label, "->", str(e)[-160:])
        continue
    done_first = True
    label = f"setback {sb:.2f} clearance {cl}"
    try:
        t = time.perf_counter(); cut = child.add_conforming_sheet(pts, tris, "Fault", setback=sb, clearance=cl, verbose=True)
        print("OK:", label, f"{time.perf_counter()-t:.1f}s cells", cut.dm.getHeightStratum(0)[1])
    except Exception as e:
        print("REFUSED:", label, "->", str(e)[-160:])
