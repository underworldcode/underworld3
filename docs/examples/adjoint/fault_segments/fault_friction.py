"""Friction on a listric fault, segment by segment, from surface uplift and stress.

A listric fault under horizontal shortening and gravity: a ramp that steepens
from a flat decollement at depth to a dip of sixty degrees at the surface,
represented as a weak plane in a transversely isotropic viscosity (no cut in
the mesh). The plane yields at the Coulomb stress tau_y = C + mu p, where p
is the pressure, and the friction coefficient mu is different on the flat,
on the lower and upper parts of the ramp, and near the surface. Those four
coefficients are the unknowns. The observations are the uplift rate along
the top surface and the shear stress in the bulk near a handful of points,
taken from a run at the true coefficients.

The residual is nonlinear in the velocity and the pressure, so the forward
solve is Newton. The gradient with respect to each coefficient comes from
one call, stokes.gradient(misfit, parameters=...): the solver assembles the
adjoint operator from its own Jacobian kernels with trial and test
exchanged, solves it, and differentiates its residual symbolically with
respect to each named coefficient. Nothing here is differenced.

Run it:

    python fault_friction.py                 # twin experiment: gradient check, then the inversion
    python fault_friction.py -uw_check_only 1
"""
import math

import numpy as np
import sympy

import underworld3 as uw

params = uw.Params(
    cell_size=uw.Param(1 / 12, "base mesh cell size (box is 2 x 1); refined once, so half this"),
    surface_dip=uw.Param(60.0, "dip of the ramp where it reaches the surface, degrees"),
    flat_depth=uw.Param(0.3, "height of the decollement above the base"),
    surface_x=uw.Param(1.9, "where the fault reaches the surface"),
    band=uw.Param(0.08, "half-width of the weak band, in box units"),
    true_strengths=uw.Param("0.05,0.15,0.25,0.4",
                            "friction coefficient: flat, lower ramp, upper ramp, near surface"),
    initial_strength=uw.Param(0.2, "starting guess, every segment"),
    cohesion=uw.Param(0.05, "cohesion C in tau_y = C + mu p"),
    rho_g=uw.Param(10.0, "body force, so the pressure grows with depth"),
    check_only=uw.Param(0, "1: gradient check against finite differences, no inversion"),
    observations=uw.Param("uplift+stress",
                          "uplift+stress | orientation (principal-stress orientation at the "
                          "points and along the surface) | orientation_surface (along the "
                          "surface only)"),
)

# --- the model ---------------------------------------------------------------
# Refined once from the base size: the refinement gives the velocity block a
# multigrid hierarchy. Without one it falls back to gamg, which hits its
# iteration cap on this problem, and an inexact Newton step converges linearly.
# Every solve starts cold: six Newton iterations, and a misfit that does not
# depend on the previous evaluation, which the finite-difference check needs.
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0),
                                         cellSize=params.cell_size, qdegree=3, refinement=1)
x, y = mesh.X

v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)
v_obs = uw.discretisation.MeshVariable("v_obs", mesh, mesh.dim, degree=2)

# The fault: a flat decollement at y = flat_depth running from the left wall
# to x = xc, then a circular ramp of radius R about (xc, yc) that leaves the
# flat horizontally and reaches the surface at the given dip. The signed
# distance to it and the position along it are exact on each piece, and the
# director is the normal at the nearest point: vertical on the flat, radial on
# the ramp.
phi_top = math.radians(params.surface_dip) - math.pi / 2       # angle of the surface point about the centre
R = (1.0 - params.flat_depth) / (1.0 + math.sin(phi_top))
yc = params.flat_depth + R
xc = params.surface_x - R * math.cos(phi_top)
r = sympy.sqrt((x - xc) ** 2 + (y - yc) ** 2)
phi = sympy.atan2(y - yc, x - xc)
on_flat = x < xc
d = sympy.Piecewise((y - params.flat_depth, on_flat), (R - r, True))
s = sympy.Piecewise((x, on_flat), (xc + R * (phi + sympy.pi / 2), True))
n_hat = sympy.Matrix([[sympy.Piecewise((0, on_flat), ((x - xc) / r, True)),
                       sympy.Piecewise((1, on_flat), ((y - yc) / r, True))]])
ramp = R * (phi_top + math.pi / 2)
length = xc + ramp
band = sympy.exp(-(d / params.band) ** 2)

# Segments along the fault: the flat, then the ramp in three equal parts.
edges = [0.0, xc, xc + ramp / 3, xc + 2 * ramp / 3, length]
names = ["flat", "lower ramp", "upper ramp", "near surface"]
n_seg = len(names)
strengths = [uw.expression(rf"\mu_{{{k + 1}}}", params.initial_strength,
                           f"friction coefficient, {names[k]}")
             for k in range(n_seg)]

def segment(k):
    """A smooth indicator for segment k along the fault, in [0, 1]."""
    edge = params.band
    on = 1 if k == 0 else (1 + sympy.tanh((s - edges[k]) / edge)) / 2
    off = 1 if k == n_seg - 1 else (1 - sympy.tanh((s - edges[k + 1]) / edge)) / 2
    return on * off

eta_0 = 1

# Coulomb yield on the plane. The shear strain rate resolved on the plane is
# t.E.n; the plane's viscosity is the harmonic combination of the bulk
# viscosity and the yield stress over that rate, which is smooth everywhere
# and tends to tau_y / (2 e_s) where the plane slips.
E = mesh.vector.strain_tensor(v.sym)
t_hat = sympy.Matrix([[-n_hat[1], n_hat[0]]])
e_s = sympy.sqrt((t_hat * E * n_hat.T)[0] ** 2 + uw.maths.functions.vanishing)
friction = sum(strengths[k] * segment(k) for k in range(n_seg))
# Compression is positive; where the dynamic pressure is tensile the plane
# keeps its cohesion and no more.
tau_y = params.cohesion + friction * sympy.Max(p.sym[0], 0)
eta_plane = eta_0 * tau_y / (tau_y + 2 * eta_0 * e_s)
eta_1 = eta_0 - band * (eta_0 - eta_plane)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0
stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_1
stokes.constitutive_model.Parameters.director = n_hat
stokes.tolerance = 1e-8
stokes.bodyforce = sympy.Matrix([0, -params.rho_g])

# Shortening from both sides, a no-slip base, and a free top: the surface
# velocity is the uplift rate.
stokes.add_essential_bc((0.0, 0.0), "Bottom")
stokes.add_essential_bc((0.5, None), "Left")
stokes.add_essential_bc((-0.5, None), "Right")

# --- the observations ------------------------------------------------------
points = [(0.3, 0.65), (0.75, 0.12), (1.25, 0.3), (0.9, 0.6), (1.55, 0.85)]
w_points = sum(sympy.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * params.band) ** 2)
               for px, py in points)

def shear_stress(field):
    e = mesh.vector.strain_tensor(field.sym)
    return 2 * eta_0 * e[0, 1]

def orientation(field):
    """The principal-stress orientation as the unit vector (cos 2theta, sin 2theta)
    of the deviatoric strain rate, which is the stress orientation in the
    isotropic bulk. A unit vector rather than an angle, so there is no wrap."""
    e = mesh.vector.strain_tensor(field.sym)
    a, b = e[0, 0] - e[1, 1], 2 * e[0, 1]
    norm = sympy.sqrt(a ** 2 + b ** 2 + uw.maths.functions.vanishing)
    return sympy.Matrix([[a / norm, b / norm]])

# The misfit has a term on the top surface — a true boundary integral of
# the uplift rate, or of the stress orientation — and a term in the volume
# around the stress points. gradient() takes them as {domain: integrand}.
what = str(params.observations)
dq = orientation(v) - orientation(v_obs)
if what == "uplift+stress":
    misfit = {"Top": (v.sym[1] - v_obs.sym[1]) ** 2 / 2,
              None: w_points * (shear_stress(v) - shear_stress(v_obs)) ** 2 / 2}
elif what == "orientation":
    misfit = {"Top": (dq[0] ** 2 + dq[1] ** 2) / 2,
              None: w_points * (dq[0] ** 2 + dq[1] ** 2) / 2}
else:
    misfit = {"Top": (dq[0] ** 2 + dq[1] ** 2) / 2}

def misfit_value():
    return sum(uw.adjoint.integral(mesh, term, where) for where, term in misfit.items())

model = uw.get_default_model()

def set_strengths(values):
    for expr, value in zip(strengths, values):
        expr.sym = float(value)

evaluations = [0]

def J_and_gradient(label=None):
    """The misfit and dJ/d(log strength) for each segment, by the adjoint.

    Each evaluation is one step of zero length in the model's record, so the
    run's transcript lists the forward solve and the adjoint solve it made.
    """
    evaluations[0] += 1
    with model.step(0.0, label=label or f"eval {evaluations[0]}"):
        stokes.solve(zero_init_guess=True)
        out = stokes.gradient(misfit, parameters=strengths)
    # d/d(log mu) = mu d/d(mu)
    grad = np.array([out["parameters"][expr] * float(expr.sym) for expr in strengths])
    return out["J"], grad

def forward(label):
    with model.step(0.0, label=label):
        stokes.solve(zero_init_guess=True)

# --- the truth, and the twin -------------------------------------------------
true_values = [float(t) for t in str(params.true_strengths).split(",")][:n_seg]
set_strengths(true_values)
forward("truth")
v_obs.array[...] = np.asarray(v.array)
uw.pprint(f"true strengths {true_values}")

set_strengths([params.initial_strength] * n_seg)
J0, g0 = J_and_gradient("start")
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
        forward(f"fd {names[k]} {'+' if sign > 0 else '-'}h")
        fd.append(misfit_value())
    fd = (fd[0] - fd[1]) / (2 * h)
    uw.pprint(f"{names[k]:>13}: adjoint {g0[k]: .6e}   finite difference {fd: .6e}   "
              f"ratio {fd / g0[k]:.5f}")
set_strengths([params.initial_strength] * n_seg)

if int(params.check_only):
    raise SystemExit

# --- the inversion ---------------------------------------------------------------
from scipy.optimize import minimize

history = []

# The optimiser sees the misfit relative to its starting value: L-BFGS-B stops
# on the absolute decrease of its objective, and a surface integral of a
# velocity misfit is a small number.
J_scale = J0

def objective(log_eta):
    set_strengths(np.exp(log_eta))
    J, grad = J_and_gradient()
    history.append((J, np.exp(log_eta).copy()))
    uw.pprint(f"  J = {J:.6e}   strengths = {np.exp(log_eta)}")
    return J / J_scale, grad / J_scale

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
    forward(f"profile {label}")
    profiles[label] = np.asarray(uw.function.evaluate(v.sym[1], top)).ravel()
set_strengths(true_values)
forward("truth again")                       # the field on the grid is the truth's
gx, gy = np.meshgrid(np.linspace(0, 2, 201), np.linspace(0, 1, 101))
grid = np.column_stack([gx.ravel(), gy.ravel()])
eta_1_grid = np.asarray(uw.function.evaluate(eta_1, grid)).reshape(gx.shape)
np.savez(f"fault_friction_{what}_data.npz", xs=xs, gx=gx, gy=gy, eta_1=eta_1_grid,
         points=np.array(points), true=np.array(true_values), band=params.band,
         history=np.array([[J, *vals] for J, vals in history]),
         **{f"uplift_{k}": val for k, val in profiles.items()})
