"""The remap at a slipped wall: the old failure case for the monotone clamp.

Wall vertices slide a set angle along the circle and nothing else moves, so the
motion is the same at any rank count. A smooth P2 field is remapped; its error at
the wall DOFs and inside, and its difference from the serial answer, are the
measures. Saved rows are compared across rank counts afterwards.
"""
import argparse

import numpy as np
import sympy
import underworld3 as uw

ap = argparse.ArgumentParser()
ap.add_argument("--save", default="")
args = ap.parse_args()

mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
x, y = mesh.X
r = sympy.sqrt(x ** 2 + y ** 2)
theta = sympy.atan2(y, x)
field = (r - 0.5) * 2 + 0.5 * sympy.cos(4 * theta) * r
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
T.array[:, 0, 0] = uw.function.evaluate(field, T.coords).reshape(-1)
mesh.update_lvec()

X = np.array(mesh.X.coords)
R = np.linalg.norm(X, axis=1)
wall = (np.abs(R - 1.0) < 1e-9) | (np.abs(R - 0.5) < 1e-9)
angle = 0.2 * (2 * np.pi / (2 * np.pi * 0.5 / (1 / 8)))     # a fifth of an inner-wall cell
c, s = np.cos(angle), np.sin(angle)
X[wall] = X[wall] @ np.array([[c, s], [-s, c]])
mesh.deform(X)

coords = np.asarray(T.coords).reshape(-1, 2)
value = np.asarray(T.array).reshape(-1)
exact = uw.function.evaluate(field, coords).reshape(-1)
rad = np.linalg.norm(coords, axis=1)
dof_wall = (rad > 1 - 0.01) | (rad < 0.5 + 0.01)
err = np.abs(value - exact)
e_w = uw.mpi.comm.allreduce(float(err[dof_wall].max()) if dof_wall.any() else 0.0, op=uw.MPI.MAX)
e_i = uw.mpi.comm.allreduce(float(err[~dof_wall].max()) if (~dof_wall).any() else 0.0, op=uw.MPI.MAX)
uw.pprint(f"RESULT wall err {e_w:.3e} interior err {e_i:.3e} range [{value.min():.3f}, {value.max():.3f}]")
iset, sub = mesh.dm.createSubDM(T.field_id)
g = sub.createGlobalVector()
l = sub.createLocalVector()
sub.localToGlobal(T.vec, g, addv=False)
sub.globalToLocal(g, l, addv=False)
gap = uw.mpi.comm.allreduce(float(np.abs(value - l.array).max()) if value.size else 0.0, op=uw.MPI.MAX)
uw.pprint(f"RESULT ghost gap {gap:.3e}")
rows = uw.mpi.comm.gather(np.column_stack([coords, l.array.copy()]), root=0)
if args.save and uw.mpi.rank == 0:
    np.save(args.save, np.vstack(rows))
