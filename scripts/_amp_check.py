"""Does `amp` actually change the anisotropic mover's metric?
The mover builds M = base[I + β ĝĝᵀ (|∇ρ|/gref)²], gref=max|∇ρ|.
With ρ = 1 + amp·t the gradient is amp·∇t and gref is amp·max|∇t|
⇒ (|∇ρ|/gref) and ĝ are amp-INVARIANT ⇒ M independent of amp.
Verify numerically: build the metric at amp=16 and amp=24, project
∇ρ, compare the normalised-gradient field the mover actually uses.
"""
import numpy as np, sympy
import underworld3 as uw
from underworld3.meshing import metric_density_from_gradient

m = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                       cellSize=1/16, qdegree=3)
T = uw.discretisation.MeshVariable("T", m, vtype=uw.VarType.SCALAR,
                                   degree=3, continuous=True)
r = np.sqrt((np.asarray(T.coords) ** 2).sum(1))
T.data[:, 0] = np.exp(-((r - 0.75) / 0.1) ** 2)   # a feature
X = m.CoordinateSystem.X


def norm_grad_field(amp, name):
    rho = metric_density_from_gradient(m, T, amp=amp, name=name)
    g = uw.discretisation.MeshVariable(
        f"g_{name}", m, vtype=uw.VarType.VECTOR, degree=1,
        continuous=True)
    p = uw.systems.Vector_Projection(m, g)
    p.smoothing = 0.0
    p.uw_function = sympy.Matrix([rho.diff(X[i])
                                  for i in range(2)]).T
    p.solve()
    gv = np.asarray(uw.function.evaluate(
        g.sym, np.asarray(g.coords))).reshape(-1, 2)
    gn = np.linalg.norm(gv, axis=1)
    gref = gn.max()
    return gn / gref            # the (|∇ρ|/gref) the mover uses


a16 = norm_grad_field(16.0, "a16")
a24 = norm_grad_field(24.0, "a24")
print(f"max |  (|∇ρ|/gref)_amp24  −  _amp16  | = "
      f"{np.abs(a24 - a16).max():.3e}")
print(f"⇒ metric tensor M is "
      f"{'IDENTICAL (amp is a no-op)' if np.abs(a24-a16).max()<1e-9 else 'DIFFERENT'}"
      f" between amp=16 and amp=24")
