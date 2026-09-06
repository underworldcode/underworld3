"""Solid-body rotation of a Gaussian: SUPG (issue #657 branch) vs SLCN at
several Courant numbers.  One full revolution returns the field to its
initial state, so the round-trip L2 error is an absolute measure.

  python rotation_courant_sweep.py -uw_res 32 -uw_courant 4 -uw_scheme both

Rows are appended to results.csv in this directory.
"""
import os, sys, time, csv
import numpy as np
import sympy
import underworld3 as uw

RES = 32
COURANT = 1.0
SIGMA = 0.12
SCHEME = "both"
SUPG_DC = False

params = uw.Params(
    uw_res=RES, uw_courant=COURANT, uw_sigma=SIGMA, uw_scheme=SCHEME, uw_dc=SUPG_DC,
    uw_fault=False, uw_dt=0.0, uw_core=0.0,
)
fault_refine, dt_fixed, core = bool(params.uw_fault), float(params.uw_dt), float(params.uw_core)
res, courant, sigma, scheme, use_dc = (
    int(params.uw_res), float(params.uw_courant), float(params.uw_sigma),
    str(params.uw_scheme), bool(params.uw_dc))

here = os.path.dirname(os.path.abspath(__file__))

if fault_refine:
    # Section 16.1 of the transport note: a fault-like band the scalar does not
    # need. Base at 2x the target size + one uniform refinement gives h = 2/res
    # everywhere; the band at x = 0 is then bisected down to h/8.
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=4.0 / res, qdegree=3,
        regular=False, refinement=1)
    pts = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    fault = uw.meshing.Surface("fault", base, pts, symbol="F")
    fault.discretize()
    h = 2.0 / res
    if core > 0:
        # Flat-core band: h/8 out to |x| < core, then a linear ramp to h.
        def metric(pts, _f=fault, _hn=h / 8, _hf=h, _core=core, _ramp=0.06):
            d = _f.unsigned_distance(pts)
            hh = np.where(d < _core, _hn, np.minimum(_hn + (_hf - _hn) * (d - _core) / _ramp, _hf))
            return 1.0 / hh ** 2
    else:
        metric = fault.refinement_metric_function(h_near=h / 8, h_far=0.5, width=0.06,
                                                  profile="linear")
    mesh = base.adapt(metric, max_levels=3)
    if uw.mpi.rank == 0:
        print(f"fault-refined child: {mesh.dm.getHeightStratum(0)[1]} cells, "
              f"radii min={float(np.min(mesh._radii)):.4g} max={float(np.max(mesh._radii)):.4g}", flush=True)
else:
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0), cellSize=2.0 / res, qdegree=3, regular=False)
x, y = mesh.X

v_expr = sympy.Matrix([[-y, x]])                     # omega = 1, period 2 pi
T_period = 2.0 * np.pi

def gaussian(xx, yy):
    return sympy.exp(-((xx - 0.5) ** 2 + yy ** 2) / (2 * sigma ** 2))

def exact_at(t):
    c, s = sympy.cos(t), sympy.sin(t)
    return gaussian(x * c + y * s, -x * s + y * c)

T0_expr = gaussian(x, y)

def run(name):
    T = uw.discretisation.MeshVariable(f"T_{name}", mesh, 1, degree=2)
    T.data[:, 0] = uw.function.evaluate(T0_expr, T.coords).reshape(-1)

    if name == "supg":
        adv = uw.systems.AdvDiffusionSUPG(mesh, T, v_expr, diffusivity=0.0,
                                          discontinuity_capturing=use_dc)
    else:
        adv = uw.systems.AdvDiffusionSLCN(mesh, T, v_expr, order=1)
        adv.constitutive_model = uw.constitutive_models.DiffusionModel
        adv.constitutive_model.Parameters.diffusivity = 0.0
        adv.f = 0.0
    for b in ("Left", "Right", "Top", "Bottom"):
        adv.add_dirichlet_bc(0.0, b)

    dt_cfl = float(adv.estimate_dt())
    if dt_fixed > 0:
        nsteps = int(np.round(T_period / dt_fixed))
    else:
        nsteps = int(np.ceil(T_period / (courant * dt_cfl)))
    dt = T_period / nsteps

    mass0 = uw.maths.Integral(mesh, T.sym[0, 0]).evaluate()
    norm0 = np.sqrt(uw.maths.Integral(mesh, T.sym[0, 0] ** 2).evaluate())

    t = 0.0
    wall = []
    quarter_err = {}
    for step in range(nsteps):
        t0 = time.perf_counter()
        adv.solve(timestep=dt)
        wall.append(time.perf_counter() - t0)
        t += dt
        # error against the exact rotated Gaussian at each quarter turn
        for q in (1, 2, 3):
            if q not in quarter_err and t >= q * T_period / 4 - 0.5 * dt:
                e = uw.maths.Integral(mesh, (T.sym[0, 0] - exact_at(t)) ** 2).evaluate()
                quarter_err[q] = np.sqrt(max(e, 0.0)) / norm0

    err = np.sqrt(max(uw.maths.Integral(mesh, (T.sym[0, 0] - T0_expr) ** 2).evaluate(), 0.0)) / norm0
    mass = uw.maths.Integral(mesh, T.sym[0, 0]).evaluate()
    d = np.asarray(T.data[:, 0])
    row = dict(scheme=name, res=res, courant=courant, fault=int(fault_refine), core=core, ncells=int(mesh.dm.getHeightStratum(0)[1]), dc=int(use_dc and name == "supg"),
               dt=dt, nsteps=nsteps, dt_cfl=dt_cfl,
               err_q1=quarter_err.get(1, np.nan), err_q2=quarter_err.get(2, np.nan),
               err_q3=quarter_err.get(3, np.nan), err_full=err,
               mass_drift=(mass - mass0) / mass0, Tmin=d.min(), Tmax=d.max(),
               wall_first=wall[0], wall_per_step=float(np.median(wall[1:])) if len(wall) > 1 else wall[0],
               wall_total=float(np.sum(wall)))
    if uw.mpi.rank == 0:
        print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        path = os.path.join(here, "results.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    return row

schemes = ["supg", "slcn"] if scheme == "both" else [scheme]
for s in schemes:
    run(s)
