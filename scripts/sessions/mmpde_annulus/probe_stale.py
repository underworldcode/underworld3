"""After a mover call: which copies of a remapped field hold the remapped values?"""
import numpy as np
import sympy
import underworld3 as uw

mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, refinement=1, qdegree=3)
X = mesh.X
r = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
field = 0.5 * (1 - sympy.tanh((r - 0.9) / 0.03))
T.array[:, 0, 0] = uw.function.evaluate(field, T.coords).reshape(-1)
mesh.update_lvec()
before = np.array(T.array).reshape(-1)
rho = 1 + 24 * sympy.exp(-(((r - 0.9) / 0.06) ** 2))
uw.meshing.node_redistribution(mesh, rho, slip_surfaces=True, skip_threshold=None,
                               method_kwargs=dict(accel="none", n_outer=20))
exact = uw.function.evaluate(field, T.coords).reshape(-1)
arr = np.array(T.array).reshape(-1)
vec = np.array(T.vec.array).reshape(-1)
print("stale flag after move:", mesh._stale_lvec)
print("array vs exact  ", np.abs(arr - exact).max())
print("vec vs array    ", np.abs(vec - arr).max())
print("array vs before ", np.abs(arr - before).max())
mesh.update_lvec()
names, isets, dms = mesh.dm.createFieldDecomposition()
g = mesh.dm.getGlobalVec()
mesh.dm.localToGlobal(mesh.lvec, g)
i = list(names).index(T.clean_name if hasattr(T, "clean_name") else T.name)
sub = g.getSubVector(isets[i])
# owned values in the mesh vector against the variable's own global vector
gv = T._gvec
dms[i].localToGlobal(T.vec, gv, addv=False)
print("lvec vs variable", np.abs(np.array(sub.array) - np.array(gv.array)).max())
g.restoreSubVector(isets[i], sub)
mesh.dm.restoreGlobalVec(g)
area = uw.maths.Integral(mesh, T.sym[0]).evaluate()
T.array[:, 0, 0] = arr
print("integral before touch", area, "after", uw.maths.Integral(mesh, T.sym[0]).evaluate())
