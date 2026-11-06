"""What transport solvers do to their unknown and history across a mover call.

Prints T's transfer policy, how far T moved from its formula, the gap between T and
its history, and where one tiny time step changes T: a change that does not shrink
with dt collapses the accuracy-based estimate_dt.
"""
import argparse

import numpy as np
import sympy
import underworld3 as uw

ap = argparse.ArgumentParser()
ap.add_argument("--transport", default="supg", choices=["supg", "slcn"])
ap.add_argument("--wall-bc", default="field", choices=["field", "fixed"])
args = ap.parse_args()

mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, refinement=2, qdegree=3)
X = mesh.X
r = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
th = sympy.atan2(X[1], X[0])
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
field = 0.5 * (1 - sympy.tanh((r - 0.9) / 0.03)) * (1 + 0.3 * sympy.cos(4 * th))
T.array[:, 0, 0] = uw.function.evaluate(field, T.coords).reshape(-1)
v = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
if args.transport == "supg":
    adv = uw.systems.AdvDiffusion(mesh, T, v.sym, order=1, theta=1.0)
else:
    adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=v.sym, theta=1.0, monotone_mode="clamp")
adv.constitutive_model = uw.constitutive_models.DiffusionModel
adv.constitutive_model.Parameters.diffusivity = 1.0e-3
print("T policy", getattr(T, "remesh_policy", None), "managed by",
      type(getattr(T, "_remesh_managed_by", None)).__name__)
if args.wall_bc == "field":
    adv.add_dirichlet_bc(sympy.Matrix([field]), "Upper")
    adv.add_dirichlet_bc(sympy.Matrix([field]), "Lower")
else:
    adv.add_dirichlet_bc(0.0, "Upper")
    adv.add_dirichlet_bc(1.0, "Lower")


def report(tag):
    a = np.asarray(T.array).reshape(-1)
    exact = uw.function.evaluate(field, T.coords).reshape(-1)
    print(f"{tag}: |T-formula|max {np.abs(a - exact).max():.3e} T range [{a.min():.4f}, {a.max():.4f}]")
    ddt = adv.DuDt
    stars = getattr(ddt, "psi_star", None) or []
    for i, s in enumerate(stars):
        arr = np.asarray(s.array).reshape(-1)
        if arr.size == a.size:
            print(f"  history {i} ({s.name}) policy {getattr(s, 'remesh_policy', None)}: "
                  f"|T-star|max {np.abs(a - arr).max():.3e}")


def step(dt, tag):
    before = np.array(T.array).reshape(-1)
    adv.solve(timestep=dt, zero_init_guess=False)
    after = np.asarray(T.array).reshape(-1)
    change = np.abs(after - before)
    i = int(change.argmax())
    rad = float(np.linalg.norm(np.asarray(T.coords)[i]))
    print(f"{tag} dt={dt:.1e}: max change {change.max():.3e} at r={rad:.4f}; "
          f"nodes changing >1e-3: {(change > 1e-3).sum()}; "
          f"estimate_dt {float(adv.estimate_dt()) if args.transport == 'supg' else 0:.3e}")


report("start")
step(1e-3, "first step")
report("after step")
rho = uw.meshing.metric_density_from_gradient(mesh, T, refinement=5, coarsening="auto",
                                              metric_choice="front-following")
uw.meshing.node_redistribution(mesh, rho, slip_surfaces=True, skip_threshold=None,
                               method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0))
report("after move")
for dt in (1e-3, 1e-4, 1e-5, 1e-6):
    step(dt, "step after move")
    report("  then")
