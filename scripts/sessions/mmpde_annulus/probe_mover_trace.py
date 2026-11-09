"""The MMPDE mover's outer iteration trace on an annulus: why and where it stops."""
import argparse
import json
import numpy as np
import sympy
import underworld3 as uw

ap = argparse.ArgumentParser()
ap.add_argument("--cell", type=float, default=1 / 8)
ap.add_argument("--refinement", type=int, default=2)
ap.add_argument("--R", type=float, default=5.0)
ap.add_argument("--slip", type=int, default=1)
ap.add_argument("--n-outer", type=int, default=150)
ap.add_argument("--save", default="")
ap.add_argument("--metric", default="gradient", choices=["gradient", "analytic"])
ap.add_argument("--kw", default="{}", help="JSON of extra mover kwargs")
ap.add_argument("--second-move", type=int, default=0, help="1: time a second move from the adapted mesh")
args = ap.parse_args()

mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=args.cell,
                          refinement=args.refinement, qdegree=3)
X = mesh.X
r = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
th = sympy.atan2(X[1], X[0])
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
Tfield = 0.5 * (1 - sympy.tanh((r - 0.9) / 0.03)) * (1 + 0.3 * sympy.cos(4 * th))
field_shifted = 0.5 * (1 - sympy.tanh((r - 0.89) / 0.03)) * (1 + 0.3 * sympy.cos(4 * th + 0.05))
T.array[:, 0, 0] = uw.function.evaluate(Tfield, T.coords).reshape(-1)
rho = uw.meshing.metric_density_from_gradient(
    mesh, T, refinement=args.R, coarsening="auto", metric_choice="front-following")
if args.metric == "analytic":
    rho = 1 + (args.R ** 2 - 1) * sympy.exp(-(((r - 0.9) / 0.06) ** 2))
x0 = np.array(mesh.X.coords)
import time
_t0 = time.time()
uw.meshing.node_redistribution(
    mesh, rho, verbose=True,
    method_kwargs={**dict(step_frac=0.2, accel="cg", momentum=0.0, n_outer=args.n_outer),
                   **json.loads(args.kw)},
    slip_surfaces=bool(args.slip), skip_threshold=None)
if uw.mpi.rank == 0:
    print("mover seconds", time.time() - _t0, flush=True)
if args.second_move:
    T.array[:, 0, 0] = uw.function.evaluate(field_shifted, T.coords).reshape(-1)
    rho2 = uw.meshing.metric_density_from_gradient(
        mesh, T, refinement=args.R, coarsening="auto", metric_choice="front-following")
    _t1 = time.time()
    uw.meshing.node_redistribution(
        mesh, rho2, verbose=True,
        method_kwargs={**dict(step_frac=0.2, accel="cg", momentum=0.0, n_outer=args.n_outer),
                       **json.loads(args.kw)},
        slip_surfaces=bool(args.slip), skip_threshold=None)
    if uw.mpi.rank == 0:
        print("second move seconds", time.time() - _t1, flush=True)
from underworld3.meshing.smoothing import graph
_cells = graph._tri_cells(mesh.dm)
_area = graph._signed_areas(mesh.dm.getCoordinatesLocal().array.reshape(-1, 2), _cells)
print("rank", uw.mpi.rank, "folded", int((_area * np.sign(np.median(_area)) <= 0).sum()),
      "area_ratio", float(np.abs(_area).max() / np.abs(_area).min()), flush=True)
d = np.linalg.norm(np.array(mesh.X.coords) - x0, axis=1)
print("rank", uw.mpi.rank, "max displacement", d.max(), "mean", d.mean(), flush=True)
if args.save:
    all_x = uw.mpi.comm.gather(np.array(mesh.X.coords), root=0)
    all_x0 = uw.mpi.comm.gather(x0, root=0)
    if uw.mpi.rank == 0:
        np.savez(args.save, x=np.vstack(all_x), x0=np.vstack(all_x0))
