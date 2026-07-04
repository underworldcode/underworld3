"""Rigorous gating + clamp probe for the equidistribution metric.

R=1 (regime 3, refine-only): by construction every eigenvalue
∈ [base, aniso_cap·base] with base=1/h0² — the global MIN
eigenvalue IS exactly base (the flat far-field floor; nothing
de-resolves). So use R=1's min eigenvalue as ground-truth `base`,
then test R=2 against it (no median guesswork).

PASS:
  R=1: max/min == aniso_cap (=2) to rounding, and the min is the
       modal eigenvalue (flat majority sits on the floor) ⇒
       refine-only, zero de-resolution ⇒ no-op character.
  R=2: clamp [base/4, 4·base] RESPECTED (no eigenvalue outside,
       to rounding); eigenvalues span base on BOTH sides
       (min < base < max) ⇒ complementary de-resolution, automatic.
"""
import numpy as np
import underworld3 as uw
from underworld3.meshing import (
    smooth_mesh_interior, metric_density_from_gradient)
from underworld3.meshing.smoothing import _ANISO_CACHE


def build(R):
    _ANISO_CACHE.clear()
    mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0,
                              cellSize=0.08, qdegree=3)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    xy = np.asarray(T.coords)[:, :2]
    rr = (xy ** 2).sum(1) ** 0.5
    T.data[:, 0] = np.exp(-((rr - 0.75) / 0.02) ** 2)  # steep ring
    rho = metric_density_from_gradient(mesh, T, amp=16.0,
                                       name="probe")
    smooth_mesh_interior(
        mesh, metric=rho, method="anisotropic",
        method_kwargs=dict(resolution_ratio=R, relax=0.0,
                           n_outer=1, beta=200.0))
    Df = list(_ANISO_CACHE.values())[0][2]
    return np.linalg.eigvalsh(np.asarray(Df.array))  # (N,2) asc


ev1 = build(1.0)
base = ev1.min()                       # exact 1/h0² (regime-3 floor)
r1 = ev1.max() / ev1.min()
# modal check: fraction of node-eigenvalues sitting on the floor
on_floor = np.isclose(ev1, base, rtol=1e-6).mean()
print(f"R=1: min/base={ev1.min()/base:.4f} max/base="
      f"{ev1.max()/base:.4f} ratio={r1:.4f} (aniso_cap=2 expect) "
      f"frac_on_floor={on_floor:.2f} "
      f"any_sub_base={(ev1 < base*(1-1e-6)).any()}")

ev2 = build(2.0)
lo, hi = base / 4.0, base * 4.0        # clamp for R=2
viol = int((ev2 < lo * (1 - 1e-6)).sum()
           + (ev2 > hi * (1 + 1e-6)).sum())
print(f"R=2: min/base={ev2.min()/base:.4f} "
      f"max/base={ev2.max()/base:.4f} "
      f"clamp=[{lo/base:.2f},{hi/base:.2f}]·base "
      f"clamp_violations={viol}  "
      f"spans_base={(ev2.min() < base < ev2.max())}  "
      f"frac_coarsened={(ev2.min(1) < base*(1-1e-6)).mean():.2f}")
print("VERDICT: R=1 no-op iff ratio≈2 & any_sub_base=False & "
      "high frac_on_floor.  R=2 good iff clamp_violations=0 & "
      "spans_base=True & a real coarsened fraction.")
