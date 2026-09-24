# %% [markdown]
# # Friction on a listric fault from surface observations
#
# A listric fault under horizontal shortening and gravity: a flat décollement
# at depth that steepens through a circular ramp to a dip of sixty degrees at
# the surface. The fault is a weak plane in a transversely isotropic viscosity,
# with no cut in the mesh. The plane yields at the Coulomb stress
# $\tau_y = C + \mu p$, and the friction coefficient $\mu$ takes a different
# value on the flat, on the lower and upper parts of the ramp, and near the
# surface. Those four coefficients are the unknowns.
#
# The observations are the uplift rate along the top surface and the shear
# stress near a handful of interior points, read from a run at the true
# coefficients. The gradient of the misfit with respect to each coefficient
# comes from one call, `stokes.gradient(misfit, parameters=...)`: the solver
# assembles the adjoint operator from its own Jacobian kernels with trial and
# test functions exchanged, solves it, and differentiates its residual
# symbolically with respect to each named coefficient. Nothing is differenced.
#
# The notebook checks that gradient against central differences, then hands
# the misfit and its gradient to PETSc's TAO to recover the four
# coefficients. Each misfit evaluation is one step of zero length in the
# model's record, so the run's transcript lists the forward solve and the
# adjoint solve it made.

# %%
import math

import numpy as np
import sympy

import underworld3 as uw

# %% [markdown]
# ### Configurable parameters
#
# Default values are defined as named constants below. From the command line,
# override them with PETSc-style flags:
#
# ```bash
# python fault_friction.py -uw_check_only 1
# python fault_friction.py -uw_observations surface_strain
# python fault_friction.py -uw_noise 0.03 -uw_prior_sigma 1 -uw_bounds 0.005,1
# ```
#
# The problem depends on one dimensionless number: the lithostatic pressure at
# the base of the model over the viscous stress of the shortening,
# $\rho g L^2 / (\eta V)$. We fix the depth, the viscosity and the density,
# and that ratio sets the convergence rate.

# %%
# --- Scales ---
DOMAIN_DEPTH = uw.quantity(10, "km")           # the box is 20 km wide and 10 km deep
REF_VISCOSITY = uw.quantity(1e21, "Pa*s")      # the bulk viscosity
DENSITY = uw.quantity(2700, "kg/m**3")
GRAVITY = uw.quantity(9.81, "m/s**2")
PRESSURE_TO_STRESS = 10                        # rho g L^2 / (eta V)
CONVERGENCE_RATE = (DENSITY * GRAVITY * DOMAIN_DEPTH**2 / (PRESSURE_TO_STRESS * REF_VISCOSITY)).to("mm/yr")   # 8.4 mm/yr
REF_STRESS = (REF_VISCOSITY * CONVERGENCE_RATE / DOMAIN_DEPTH).to("MPa")                                       # 26 MPa

# --- Geometry ---
CELL_SIZE = DOMAIN_DEPTH / 12                  # base mesh; refined once, so half this
SURFACE_DIP = 60.0                             # degrees, where the ramp reaches the surface
FLAT_DEPTH = uw.quantity(3, "km")              # height of the décollement above the base
SURFACE_X = uw.quantity(19, "km")              # where the fault reaches the surface
BAND_HALF_WIDTH = uw.quantity(0.8, "km")       # of the weak plane

# --- Rheology ---
COHESION = (0.05 * REF_STRESS).to("MPa")       # C in tau_y = C + mu p; 1.3 MPa
TRUE_FRICTION = "0.05,0.15,0.25,0.4"           # flat, lower ramp, upper ramp, near surface
INITIAL_FRICTION = 0.2                         # the starting guess, every segment

# --- The inversion ---
CHECK_ONLY = 0                                 # 1: the gradient check, no inversion
OPTIMISER = "tao"                              # tao (PETSc quasi-Newton) | scipy (L-BFGS-B)
OBSERVATIONS = "uplift+stress"                 # uplift+stress | orientation_points | surface_strain
NOISE = 0.0                                    # on the observed velocity, as a fraction of its rms
SEED = 7
PRIOR_SIGMA = 0.0                              # width of a Gaussian prior on log mu; 0 for none
BOUNDS = ""                                    # friction bounds lo,hi for TAO's blmvm; empty for none

params = uw.Params(
    uw_cell_size=uw.Param(CELL_SIZE, description="base mesh cell size"),
    uw_surface_dip=uw.Param(SURFACE_DIP, description="dip of the ramp at the surface, degrees"),
    uw_flat_depth=uw.Param(FLAT_DEPTH, description="height of the décollement above the base"),
    uw_surface_x=uw.Param(SURFACE_X, description="where the fault reaches the surface"),
    uw_band=uw.Param(BAND_HALF_WIDTH, description="half-width of the weak plane"),
    uw_cohesion=uw.Param(COHESION, description="cohesion C in tau_y = C + mu p"),
    uw_true_friction=uw.Param(TRUE_FRICTION, description="friction: flat, lower ramp, upper ramp, near surface"),
    uw_initial_friction=uw.Param(INITIAL_FRICTION, description="starting guess, every segment"),
    uw_check_only=uw.Param(CHECK_ONLY, description="1: gradient check only"),
    uw_optimiser=uw.Param(OPTIMISER, description="tao | scipy"),
    uw_observations=uw.Param(OBSERVATIONS, description="uplift+stress | orientation_points | surface_strain"),
    uw_noise=uw.Param(NOISE, description="noise on the observed velocity, fraction of its rms"),
    uw_seed=uw.Param(SEED, description="seed for the noise"),
    uw_prior_sigma=uw.Param(PRIOR_SIGMA, description="width of the prior on log mu; 0 for none"),
    uw_bounds=uw.Param(BOUNDS, description="friction bounds lo,hi; empty for none"),
)

# %% [markdown]
# ## The model and its scales
#
# The model is declared first, with the three reference quantities that fix
# the scaling: the depth, the viscosity and the convergence rate. The solver
# then works in units of those, so the box is $2 \times 1$, the bulk viscosity
# is one, the walls close at one, and the body force is the pressure-to-stress
# ratio. `coords` on a variable come back in physical units, and `coords_nd`
# are the model's own.

# %%
orchestration_model = uw.get_default_model()
orchestration_model.set_reference_quantities(
    domain_depth=DOMAIN_DEPTH,
    viscosity=REF_VISCOSITY,
    convergence_rate=CONVERGENCE_RATE,
)


def _nd(quantity):
    """The plain number a dimensionless quantity stands for."""
    try:
        return float(quantity.to("dimensionless").magnitude)
    except AttributeError:
        return float(quantity)


def _lengths(quantity):
    """A length in units of the model depth."""
    return _nd(quantity / DOMAIN_DEPTH)


mm_per_yr = float(CONVERGENCE_RATE.to("mm/yr").magnitude)     # one model velocity unit
km = float(DOMAIN_DEPTH.to("km").magnitude)                    # one model length unit

# %% [markdown]
# ## The mesh
#
# Refined once from the base cell size, which gives the velocity block a
# multigrid hierarchy. Without one it falls back to an algebraic coarsening
# that hits its iteration cap on this problem, and an inexact Newton step then
# converges only linearly.

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0),
    cellSize=_lengths(params.uw_cell_size), qdegree=3, refinement=1,
)
x, y = mesh.X

v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)
v_obs = uw.discretisation.MeshVariable("v_obs", mesh, mesh.dim, degree=2)

# %% [markdown]
# ## The fault
#
# A flat décollement runs from the left wall to $x = x_c$, then a circular
# ramp of radius $R$ about $(x_c, y_c)$ leaves the flat horizontally and
# reaches the surface at the given dip. The signed distance to the fault and
# the position along it are exact on each piece. The director is the normal
# at the nearest point: vertical on the flat, radial on the ramp.

# %%
flat_depth = _lengths(params.uw_flat_depth)
band_width = _lengths(params.uw_band)
phi_top = math.radians(params.uw_surface_dip) - math.pi / 2     # angle of the surface point about the centre
R = (1.0 - flat_depth) / (1.0 + math.sin(phi_top))
yc = flat_depth + R
xc = _lengths(params.uw_surface_x) - R * math.cos(phi_top)
r = sympy.sqrt((x - xc) ** 2 + (y - yc) ** 2)
phi = sympy.atan2(y - yc, x - xc)
on_flat = x < xc
d = sympy.Piecewise((y - flat_depth, on_flat), (R - r, True))
s = sympy.Piecewise((x, on_flat), (xc + R * (phi + sympy.pi / 2), True))
n_hat = sympy.Matrix([[sympy.Piecewise((0, on_flat), ((x - xc) / r, True)),
                       sympy.Piecewise((1, on_flat), ((y - yc) / r, True))]])
ramp = R * (phi_top + math.pi / 2)
length = xc + ramp
band = sympy.exp(-(d / band_width) ** 2)

# %% [markdown]
# ## The segments and their friction coefficients
#
# Four segments along the fault: the flat, then the ramp in three equal
# parts. Each coefficient is a named expression, so the solver can
# differentiate its residual with respect to it.

# %%
edges = [0.0, xc, xc + ramp / 3, xc + 2 * ramp / 3, length]
names = ["flat", "lower ramp", "upper ramp", "near surface"]
n_seg = len(names)
friction_coefficients = [
    uw.expression(rf"\mu_{{{k + 1}}}", params.uw_initial_friction, f"friction coefficient, {names[k]}")
    for k in range(n_seg)
]


def segment(k):
    """A smooth indicator for segment k along the fault, in [0, 1]."""
    on = 1 if k == 0 else (1 + sympy.tanh((s - edges[k]) / band_width)) / 2
    off = 1 if k == n_seg - 1 else (1 - sympy.tanh((s - edges[k + 1]) / band_width)) / 2
    return on * off


def set_friction(values):
    for coefficient, value in zip(friction_coefficients, values):
        coefficient.sym = float(value)

# %% [markdown]
# ## Coulomb yield on the plane
#
# The shear strain rate resolved on the plane is $\hat t \cdot E \cdot \hat n$.
# The plane's viscosity is the harmonic combination of the bulk viscosity and
# the yield stress over that rate, which is smooth everywhere and tends to
# $\tau_y / 2\dot\varepsilon_s$ where the plane slips. Compression is positive.
# Where the dynamic pressure is tensile the plane keeps its cohesion and no
# more.

# %%
eta_0 = 1
cohesion = _nd(params.uw_cohesion / REF_STRESS)
rho_g = _nd(DENSITY * GRAVITY * DOMAIN_DEPTH / REF_STRESS)

E = mesh.vector.strain_tensor(v.sym)
t_hat = sympy.Matrix([[-n_hat[1], n_hat[0]]])
e_s = sympy.sqrt((t_hat * E * n_hat.T)[0] ** 2 + uw.maths.functions.vanishing)
friction = sum(friction_coefficients[k] * segment(k) for k in range(n_seg))
tau_y = cohesion + friction * sympy.Max(p.sym[0], 0)
eta_plane = eta_0 * tau_y / (tau_y + 2 * eta_0 * e_s)
eta_1 = eta_0 - band * (eta_0 - eta_plane)

# %% [markdown]
# ## The Stokes solver
#
# Shortening from both sides, a no-slip base, and a free top: the surface
# velocity is the uplift rate. The residual is nonlinear in the velocity and
# the pressure, so the forward solve is Newton.

# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0
stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_1
stokes.constitutive_model.Parameters.director = n_hat
stokes.tolerance = 1e-8
stokes.bodyforce = sympy.Matrix([0, -rho_g])

stokes.add_essential_bc((0.0, 0.0), "Bottom")
stokes.add_essential_bc((0.5, None), "Left")
stokes.add_essential_bc((-0.5, None), "Right")

# %% [markdown]
# ## The observations
#
# The misfit has a term on the top surface, a boundary integral of the uplift
# rate, and a term in the volume around five interior points where the shear
# stress is read. `gradient()` takes the terms as `{domain: integrand}`.
#
# On a traction-free surface the shear strain rate vanishes, so a stress
# orientation read there is only a sign. Orientation is an interior
# observable (boreholes, focal mechanisms). The surface gives velocities and
# their tangential derivative, the geodetic strain rate. Both alternatives
# are available through `uw_observations`.

# %%
points = [(0.3, 0.65), (0.75, 0.12), (1.25, 0.3), (0.9, 0.6), (1.55, 0.85)]     # in units of the depth
w_points = sum(sympy.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * band_width) ** 2)
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


what = str(params.uw_observations)
dq = orientation(v) - orientation(v_obs)
if what == "uplift+stress":
    misfit = {"Top": (v.sym[1] - v_obs.sym[1]) ** 2 / 2,
              None: w_points * (shear_stress(v) - shear_stress(v_obs)) ** 2 / 2}
elif what == "orientation_points":
    misfit = {None: w_points * (dq[0] ** 2 + dq[1] ** 2) / 2}
elif what == "surface_strain":
    misfit = {"Top": (v.sym[0].diff(x) - v_obs.sym[0].diff(x)) ** 2 / 2}
else:
    raise ValueError(f"observations: {what!r}")


def misfit_value():
    return sum(uw.adjoint.integral(mesh, term, where) for where, term in misfit.items())

# %% [markdown]
# ## The forward solve and the gradient
#
# Every solve starts cold: six Newton iterations, and a misfit that does not
# depend on the previous evaluation, which the finite-difference check needs.
# Each evaluation is one step of zero length in the model's record.

# %%
evaluations = [0]


def forward(label):
    with orchestration_model.step(0.0, label=label):
        stokes.solve(zero_init_guess=True)


def misfit_and_gradient(label=None):
    """The misfit and dJ/d(log mu) for each segment, by the adjoint."""
    evaluations[0] += 1
    with orchestration_model.step(0.0, label=label or f"eval {evaluations[0]}"):
        stokes.solve(zero_init_guess=True)
        out = stokes.gradient(misfit, parameters=friction_coefficients)
    # d/d(log mu) = mu d/d(mu)
    grad = np.array([out["parameters"][c] * float(c.sym) for c in friction_coefficients])
    return out["J"], grad

# %% [markdown]
# ## The twin
#
# The observations are the velocity field at the true coefficients, with
# optional noise drawn once at a fraction of each component's rms. Every
# observation set reads `v_obs`, so the uplift, the stress and the strain
# rate all inherit the same noise.

# %%
true_values = [float(t) for t in str(params.uw_true_friction).split(",")][:n_seg]
set_friction(true_values)
forward("truth")
v_obs.array[...] = np.asarray(v.array)
if float(params.uw_noise) > 0:
    rng = np.random.default_rng(int(params.uw_seed))
    obs = np.asarray(v_obs.array)
    rms = np.sqrt(np.mean(obs ** 2, axis=0, keepdims=True))
    v_obs.array[...] = obs + float(params.uw_noise) * rms * rng.standard_normal(obs.shape)
    print(f"noise: {float(params.uw_noise):.3f} of the rms per component, seed {int(params.uw_seed)}")
J_truth = misfit_value()                     # v still holds the truth's velocity

set_friction([params.uw_initial_friction] * n_seg)
J0, g0 = misfit_and_gradient("start")
print(f"true friction {true_values}")
print(f"initial J = {J0:.6e}   dJ/dlog mu = {g0}")

# %% [markdown]
# ## The gradient check
#
# Central differences in each log-coefficient, against the adjoint gradient.

# %%
h = 1e-3
for k in range(n_seg):
    base = math.log(params.uw_initial_friction)
    fd = []
    for sign in (+1, -1):
        values = [params.uw_initial_friction] * n_seg
        values[k] = math.exp(base + sign * h)
        set_friction(values)
        forward(f"fd {names[k]} {'+' if sign > 0 else '-'}h")
        fd.append(misfit_value())
    fd = (fd[0] - fd[1]) / (2 * h)
    print(f"{names[k]:>13}: adjoint {g0[k]: .6e}   finite difference {fd: .6e}   ratio {fd / g0[k]:.5f}")
set_friction([params.uw_initial_friction] * n_seg)

if int(params.uw_check_only):
    raise SystemExit

# %% [markdown]
# ## The objective
#
# Without noise the misfit has no natural scale and the optimiser sees it
# relative to its starting value, $J/J_0$. With noise the objective is a
# negative log posterior. The data term is $\chi^2/2$: the misfit scaled by
# its expected value at the truth under the noise, which a twin experiment
# can read directly, times the number of independent data, the surface nodes
# and the nodes under the point weights. The prior term is
# $(\log\mu - \log\mu_0)^2 / 2\sigma_m^2$ with $\sigma_m$ in log units. The
# weight between the two is then a statement about the noise and the prior,
# and which coefficients the data move is decided by their sensitivities
# against that.

# %%
history = []
noisy = float(params.uw_noise) > 0
sigma_m = float(params.uw_prior_sigma)
log_prior = np.log([params.uw_initial_friction] * n_seg)
if sigma_m > 0 and not noisy:
    raise ValueError("a prior is weighed against the noise: give uw_noise as well")

if noisy:
    X = np.asarray(v.coords_nd)
    on_surface = X[:, 1] > 1.0 - 1e-6
    near_points = np.zeros(len(X), dtype=bool)
    for px, py in points:
        near_points |= (X[:, 0] - px) ** 2 + (X[:, 1] - py) ** 2 < (2 * band_width) ** 2
    N_eff = {"uplift+stress": on_surface.sum() + near_points.sum(),
             "orientation_points": near_points.sum(),
             "surface_strain": on_surface.sum()}[what]
    chi2_scale = N_eff / J_truth                # chi^2 = J * chi2_scale
    print(f"N_eff = {N_eff}, misfit floor at the truth = {J_truth:.4e}")


def objective(log_mu):
    set_friction(np.exp(log_mu))
    J, grad = misfit_and_gradient()
    history.append((J, np.exp(log_mu).copy()))
    if not noisy:
        print(f"  J/J0 = {J / J0:.6e}   friction = {np.exp(log_mu)}")
        return J / J0, grad / J0
    print(f"  J = {J:.6e}   chi2/N = {J * chi2_scale / N_eff:.4f}   friction = {np.exp(log_mu)}")
    value = J * chi2_scale / 2
    g = grad * chi2_scale / 2
    if sigma_m > 0:
        value += np.sum((log_mu - log_prior) ** 2) / (2 * sigma_m ** 2)
        g = g + (log_mu - log_prior) / sigma_m ** 2
    return value, g

# %% [markdown]
# ## The inversion
#
# PETSc's TAO drives it by default: the same objective and gradient, a
# limited-memory quasi-Newton update and TAO's line search, and the bounded
# variant when bounds are given. SciPy's L-BFGS-B is the alternative.

# %%
x0 = np.log([params.uw_initial_friction] * n_seg)
if str(params.uw_optimiser) == "tao":
    bounds = None
    if str(params.uw_bounds).strip():
        lo, hi = (float(b) for b in str(params.uw_bounds).split(","))
        bounds = (np.log([lo] * n_seg), np.log([hi] * n_seg))
    x_best, info = uw.adjoint.minimise(objective, x0, max_evaluations=60,
                                       gradient_tolerance=1e-10, bounds=bounds,
                                       method="blmvm" if bounds else "lmvm")
else:
    from scipy.optimize import minimize
    result = minimize(objective, x0, jac=True, method="L-BFGS-B",
                      options={"maxiter": 40, "gtol": 1e-10})
    x_best = result.x
print(f"recovered {np.exp(x_best)}   true {true_values}   after {len(history)} evaluations")

# %% [markdown]
# ## What the figure needs
#
# Uplift-rate profiles along the top at the truth, the start and the answer,
# the weak-plane viscosity on a grid, and the path the coefficients took.
# Lengths are saved in kilometres and velocities in millimetres per year.

# %%
xs = np.linspace(0.0, 2.0, 161)
top = np.column_stack([xs, np.full_like(xs, 1.0 - 1e-6)])
profiles = {}
for label, values in (("true", true_values), ("initial", [params.uw_initial_friction] * n_seg),
                      ("recovered", list(np.exp(x_best)))):
    set_friction(values)
    forward(f"profile {label}")
    profiles[label] = np.asarray(uw.function.evaluate(v.sym[1], top)).ravel() * mm_per_yr
set_friction(true_values)
forward("truth again")                       # the field on the grid is the truth's
gx, gy = np.meshgrid(np.linspace(0, 2, 201), np.linspace(0, 1, 101))
grid = np.column_stack([gx.ravel(), gy.ravel()])
eta_1_grid = np.asarray(uw.function.evaluate(eta_1, grid)).reshape(gx.shape)

tag = (f"{what}" + (f"_noise{float(params.uw_noise):g}" if float(params.uw_noise) > 0 else "")
       + (f"_prior{sigma_m:g}" if sigma_m > 0 else ""))
np.savez(f"fault_friction_{tag}_data.npz", xs=xs * km, gx=gx * km, gy=gy * km, eta_1=eta_1_grid,
         points=np.array(points) * km, true=np.array(true_values), band=band_width * km,
         length_unit="km", velocity_unit="mm/yr",
         history=np.array([[J, *vals] for J, vals in history]),
         **{f"uplift_{k}": val for k, val in profiles.items()})
