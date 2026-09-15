"""Strength of a fault, segment by segment, from surface uplift and stress.

A dipping fault under horizontal shortening, represented as a weak plane in a
transversely isotropic viscosity (no cut in the mesh). Its weak-plane
viscosity is different on each of a few segments down dip, and those are the
unknowns. The observations are the uplift rate along the top surface and the
shear stress in the bulk near a handful of points, taken from a run at the
true strengths.

The gradient of the misfit with respect to each segment's strength comes
from the solver's own discrete adjoint: a transpose against the Jacobian it
assembled, and the symbolic derivative of its residual with respect to the
named strength. Nothing here is differenced.

Run it:

    python fault_segments.py                 # twin experiment: gradient check, then the inversion
    python fault_segments.py -uw_check_only 1
"""
import math

import numpy as np
import sympy

import underworld3 as uw
from underworld3.adjoint import misfit_duals, inner

params = uw.Params(
    cell_size=uw.Param(1 / 24, "mesh cell size (box is 2 x 1)"),
    dip=uw.Param(45.0, "fault dip, degrees"),
    n_segments=uw.Param(3, "segments of independent strength down dip"),
    band=uw.Param(0.08, "half-width of the weak band, in box units"),
    true_strengths=uw.Param("0.01,0.1,0.03", "weak-plane viscosity per segment, true run"),
    initial_strength=uw.Param(0.1, "starting guess, every segment"),
    n_stress_points=uw.Param(4, "shear-stress observation points in the bulk"),
    check_only=uw.Param(0, "1: gradient check against finite differences, no inversion"),
)

# --- the model ---------------------------------------------------------------
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0),
                                         cellSize=params.cell_size, qdegree=3)
x, y = mesh.X

v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)
v_obs = uw.discretisation.MeshVariable("v_obs", mesh, mesh.dim, degree=2)

# The fault: a plane through (0.6, 0) at the given dip, a weak band around it,
# and segments of equal length along it.
theta = math.radians(params.dip)
t_hat = sympy.Matrix([[math.cos(theta), math.sin(theta)]])     # along dip
n_hat = sympy.Matrix([[-math.sin(theta), math.cos(theta)]])    # the director
d = (x - 0.6) * n_hat[0] + y * n_hat[1]                         # distance from the plane
s = (x - 0.6) * t_hat[0] + y * t_hat[1]                         # position along it
length = 1.0 / math.sin(theta)
band = sympy.exp(-(d / params.band) ** 2)

n_seg = int(params.n_segments)
strengths = [uw.expression(rf"\eta_{k + 1}", params.initial_strength,
                           f"weak-plane viscosity of segment {k + 1}")
             for k in range(n_seg)]

def segment(k):
    """A smooth indicator for segment k along the fault, in [0, 1]."""
    edge = params.band
    lo, hi = k * length / n_seg, (k + 1) * length / n_seg
    on = 1 if k == 0 else (1 + sympy.tanh((s - lo) / edge)) / 2
    off = 1 if k == n_seg - 1 else (1 - sympy.tanh((s - hi) / edge)) / 2
    return on * off

eta_0 = 1
eta_1 = eta_0 - band * sum((eta_0 - strengths[k]) * segment(k) for k in range(n_seg))

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0
stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_1
stokes.constitutive_model.Parameters.director = n_hat
stokes.tolerance = 1e-8

# Shortening from both sides, a no-slip base, and a free top: the surface
# velocity is the uplift rate.
stokes.add_essential_bc((0.0, 0.0), "Bottom")
stokes.add_essential_bc((0.5, None), "Left")
stokes.add_essential_bc((-0.5, None), "Right")

# --- the observations ------------------------------------------------------
w_top = sympy.exp(-((1 - y) / params.band) ** 2)
rng = np.random.default_rng(7)
points = [(0.3 + 1.4 * i / max(int(params.n_stress_points) - 1, 1), 0.3 + 0.3 * (i % 2))
          for i in range(int(params.n_stress_points))]
w_points = sum(sympy.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * params.band) ** 2)
               for px, py in points)

def shear_stress(field):
    e = mesh.vector.strain_tensor(field.sym)
    return 2 * eta_0 * e[0, 1]

misfit = (w_top * (v.sym[1] - v_obs.sym[1]) ** 2
          + w_points * (shear_stress(v) - shear_stress(v_obs)) ** 2) / 2

def set_strengths(values):
    for expr, value in zip(strengths, values):
        expr.sym = float(value)

def J_and_gradient():
    """The misfit and dJ/d(log strength) for each segment, by the adjoint."""
    stokes.solve(zero_init_guess=True)
    J = float(uw.maths.Integral(mesh, misfit).evaluate())
    dJ_dv = misfit_duals(misfit, [v])[v]
    mu = uw.discretisation.MeshVariable(f"mu_{uw.adjoint._counter()}", mesh, mesh.dim, degree=2)
    lam = uw.discretisation.MeshVariable(f"lam_{uw.adjoint._counter()}", mesh, 1, degree=1)
    dJ_dv.array[...] = -np.asarray(dJ_dv.array)
    _, reason = stokes.adjoint_solve((dJ_dv, None), target=(mu, lam))
    assert reason > 0, reason
    grad = np.array([stokes.sensitivity(mu, expr) * float(expr.sym) for expr in strengths])
    return J, grad

# --- the truth, and the twin -------------------------------------------------
true_values = [float(t) for t in str(params.true_strengths).split(",")][:n_seg]
set_strengths(true_values)
stokes.solve(zero_init_guess=True)
v_obs.array[...] = np.asarray(v.array)
uw.pprint(f"true strengths {true_values}")

set_strengths([params.initial_strength] * n_seg)
J0, g0 = J_and_gradient()
uw.pprint(f"initial J = {J0:.6e}   dJ/dlog eta = {g0}")

# Gradient check: central differences in each log-strength.
h = 1e-3
for k in range(n_seg):
    base = math.log(params.initial_strength)
    fd = []
    for sign in (+1, -1):
        vals = [params.initial_strength] * n_seg
        vals[k] = math.exp(base + sign * h)
        set_strengths(vals)
        stokes.solve(zero_init_guess=True)
        fd.append(float(uw.maths.Integral(mesh, misfit).evaluate()))
    fd = (fd[0] - fd[1]) / (2 * h)
    uw.pprint(f"segment {k + 1}: adjoint {g0[k]: .6e}   finite difference {fd: .6e}   "
              f"ratio {fd / g0[k]:.5f}")
set_strengths([params.initial_strength] * n_seg)

if int(params.check_only):
    raise SystemExit

# --- the inversion ---------------------------------------------------------------
from scipy.optimize import minimize

history = []

def objective(log_eta):
    set_strengths(np.exp(log_eta))
    J, grad = J_and_gradient()
    history.append((J, np.exp(log_eta).copy()))
    uw.pprint(f"  J = {J:.6e}   strengths = {np.exp(log_eta)}")
    return J, grad

result = minimize(objective, np.log([params.initial_strength] * n_seg), jac=True,
                  method="L-BFGS-B", options={"maxiter": 40, "gtol": 1e-10})
uw.pprint(f"recovered {np.exp(result.x)}   true {true_values}   "
          f"after {len(history)} evaluations")

# --- what the figure needs -----------------------------------------------------
# Uplift-rate profiles along the top at the truth, the start and the answer, the
# weak-plane viscosity on a grid, and the path the strengths took.
xs = np.linspace(0.0, 2.0, 161)
top = np.column_stack([xs, np.full_like(xs, 1.0 - 1e-6)])
profiles = {}
for label, values in (("true", true_values), ("initial", [params.initial_strength] * n_seg),
                      ("recovered", list(np.exp(result.x)))):
    set_strengths(values)
    stokes.solve(zero_init_guess=True)
    profiles[label] = np.asarray(uw.function.evaluate(v.sym[1], top)).ravel()
set_strengths(true_values)
gx, gy = np.meshgrid(np.linspace(0, 2, 201), np.linspace(0, 1, 101))
grid = np.column_stack([gx.ravel(), gy.ravel()])
eta_1_grid = np.asarray(uw.function.evaluate(eta_1, grid)).reshape(gx.shape)
np.savez("fault_segments_data.npz", xs=xs, gx=gx, gy=gy, eta_1=eta_1_grid,
         points=np.array(points), true=np.array(true_values),
         history=np.array([[J, *vals] for J, vals in history]),
         **{f"uplift_{k}": val for k, val in profiles.items()})
