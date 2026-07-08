"""Phase-resolved profile of a single _winslow_elliptic (MA) call,
cold then warm, AMP=8 on the res-16 Annulus. Wraps the inner
solver objects' .solve() with timers (via the cache) so we see
where the ~12 s cold / ~34 s warm goes: φ Poisson, Hessian
recovery, ∇φ projection, evaluate, _deform_mesh, and the
first-iter DM/SNES (re)build.
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import underworld3 as uw
from underworld3.meshing import smoothing as S

R_O, R_I, WIDTH, RES, AMP = 1.0, 0.5, 0.12, 16, 8.0


def case(tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    r0 = uw.discretisation.MeshVariable(
        f"r0_{tag}", m, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True)
    X0 = np.asarray(m.X.coords)
    r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
    f = 1.0 + AMP * sympy.exp(-(((r0.sym[0]) - R_O) / WIDTH) ** 2)
    return m, f


T = {}


def _wrap(obj, label):
    raw = obj.solve
    def timed(*a, **k):
        t = time.perf_counter()
        r = raw(*a, **k)
        T.setdefault(label, []).append(time.perf_counter() - t)
        return r
    obj.solve = timed
    return raw


m, f = case("p0")
pin = S._auto_pinned_labels(m)

t = time.perf_counter()
S._winslow_elliptic(m, f, pin, False)
cold = time.perf_counter() - t

# wrap cached inner solvers for the warm call
key = list(S._WINSLOW_CACHE)[0]
phi, ps, gradphi, gproj, hsolver, vol_field = S._WINSLOW_CACHE[key]
_wrap(ps, "phi_poisson")
_wrap(hsolver, "hessian")
_wrap(gproj, "gradphi_proj")
_dm_raw = m._deform_mesh
def _dm_timed(*a, **k):
    t0 = time.perf_counter()
    r = _dm_raw(*a, **k)
    T.setdefault("deform_mesh", []).append(time.perf_counter() - t0)
    return r
m._deform_mesh = _dm_timed

t = time.perf_counter()
S._winslow_elliptic(m, f, pin, False)
warm = time.perf_counter() - t

print(f"\n=== MA AMP={AMP} RES={RES}  cold={cold:.2f}s  warm={warm:.2f}s ===")
for label in ("phi_poisson", "hessian", "gradphi_proj", "deform_mesh"):
    v = T.get(label, [])
    if not v:
        continue
    a = np.array(v)
    print(f"{label:14s} n={len(a):3d}  total={a.sum():7.2f}s  "
          f"mean={a.mean()*1e3:8.1f}ms  first={a[0]*1e3:8.1f}ms  "
          f"rest_mean={(a[1:].mean()*1e3 if len(a)>1 else 0):8.1f}ms")
acct = sum(np.array(T.get(l, [0])).sum()
           for l in ("phi_poisson", "hessian", "gradphi_proj",
                     "deform_mesh"))
print(f"{'accounted':14s}            total={acct:7.2f}s  "
      f"(warm {warm:.2f}s; unaccounted {warm-acct:.2f}s)")
