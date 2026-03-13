# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tracy (2006) 2D Richards Equation Benchmark
#
# Validates the Underworld Richards solver against the **closed-form analytical
# solution** of Tracy (2006) for 2D steady-state and transient flow in
# unsaturated porous media with Gardner (exponential) soil properties.
#
# The benchmark uses a square domain $L \times L$ with the Gardner conductivity
# $K(\psi) = K_s \exp(\alpha\,\psi)$ and moisture content
# $\theta(\psi) = \theta_r + (\theta_s - \theta_r)\exp(\alpha\,\psi)$.
#
# Two boundary condition cases are tested:
#
# 1. **Specified head** — Dirichlet on all four boundaries ($\psi = h_r$ on
#    bottom, left, right; spatially varying on top)
# 2. **No-flux** — zero-flux on lateral boundaries, Dirichlet on top/bottom
#    ($\psi = h_r$ at bottom; spatially varying at top)
#
# The top boundary carries a spatially varying wetting profile that drives
# infiltration into an initially dry column ($\psi = h_r$).
#
# ### Reference
#
# Tracy, F. T. (2006). Clean two- and three-dimensional analytical solutions of
# Richards' equation for testing numerical solvers. *Water Resources Research*, 42(8).
# https://doi.org/10.1029/2005WR004638

# %%
import nest_asyncio

nest_asyncio.apply()

import underworld3 as uw
import numpy as np
import sympy
from math import sqrt, sin, cos, exp, sinh, pi, log


# %% [markdown]
# ## Analytical solution (Tracy 2006)
#
# The Tracy solution applies the Kirchhoff transform $u = \exp(\alpha\psi)$ to
# linearise the Richards equation. With Gardner properties, the transformed PDE
# is linear and has exact Fourier series solutions on a square domain.
#
# The key parameter grouping is $\alpha L$ — the ratio of domain size to the
# Gardner capillary length. The gwassess package uses $\alpha L = 5.0$ and
# $\alpha h_r = -5.0$; we adopt the same dimensionless combination with
# rescaled dimensional parameters for fast convergence.


# %%
def tracy_specified_head(x, y, t, alpha, hr, L, theta_r, theta_s, Ks):
    """Analytical pressure head — specified head BCs on all boundaries.

    Tracy (2006), equations (15)–(17).

    BCs: psi = hr on bottom (y=0), left (x=0), right (x=L).
         psi = (1/alpha)*log(exp(alpha*hr) + h0*sin(pi*x/L)) on top (y=L).
    IC:  psi = hr everywhere at t=0.
    """
    h0 = 1 - exp(alpha * hr)
    c = alpha * (theta_s - theta_r) / Ks

    beta = sqrt(alpha**2 / 4 + (pi / L) ** 2)
    hss = (
        h0
        * sin(pi * x / L)
        * exp((alpha / 2) * (L - y))
        * sinh(beta * y)
        / sinh(beta * L)
    )

    phi = 0.0
    for k in range(1, 200):
        lambdak = k * pi / L
        gamma = (beta**2 + lambdak**2) / c
        phi += ((-1) ** k) * (lambdak / gamma) * sin(lambdak * y) * exp(-gamma * t)
    phi *= ((2 * h0) / (L * c)) * sin(pi * x / L) * exp(alpha * (L - y) / 2)

    hBar = hss + phi
    return (1 / alpha) * log(exp(alpha * hr) + hBar)


def tracy_no_flux(x, y, t, alpha, hr, L, theta_r, theta_s, Ks):
    """Analytical pressure head — no-flux lateral, specified head top/bottom.

    Tracy (2006), equations (18)–(20).

    BCs: psi = hr on bottom (y=0).
         Zero flux on left (x=0) and right (x=L).
         psi = (1/alpha)*log(exp(alpha*hr) + (h0/2)*(1-cos(2*pi*x/L))) on top (y=L).
    IC:  psi = hr everywhere at t=0.
    """
    h0 = 1 - exp(alpha * hr)
    c = alpha * (theta_s - theta_r) / Ks

    beta = sqrt(alpha**2 / 4 + (2 * pi / L) ** 2)
    hss = (h0 / 2) * exp((alpha / 2) * (L - y)) * (
        sinh(alpha * y / 2) / sinh(alpha * L / 2)
        - cos(2 * pi * x / L) * sinh(beta * y) / sinh(beta * L)
    )

    phi = 0.0
    for k in range(1, 200):
        lambdak = k * pi / L
        gamma1 = (lambdak**2 + alpha**2 / 4) / c
        gamma2 = ((2 * pi / L) ** 2 + lambdak**2 + alpha**2 / 4) / c
        phi += ((-1) ** k) * lambdak * (
            (1 / gamma1) * exp(-gamma1 * t)
            - (1 / gamma2) * cos(2 * pi * x / L) * exp(-gamma2 * t)
        ) * sin(lambdak * y)
    phi *= (h0 / (L * c)) * exp(alpha * (L - y) / 2)

    hBar = hss + phi
    return (1 / alpha) * log(exp(alpha * hr) + hBar)


def tracy_solution_on_grid(sample_pts, t, alpha, hr, L, theta_r, theta_s, Ks, bc_type):
    """Evaluate Tracy solution at an array of (x, y) sample points."""
    func = tracy_specified_head if bc_type == "specified_head" else tracy_no_flux
    return np.array(
        [func(p[0], p[1], t, alpha, hr, L, theta_r, theta_s, Ks) for p in sample_pts]
    )


# %% [markdown]
# ### Configurable parameters
#
# Default values are defined as named constants below. From the command line,
# override them with PETSc-style flags:
#
# ```bash
# python Ex_Richards_Tracy_Benchmark.py -uw_res 48 -uw_bc_type no_flux
# ```
#
# The default parameters give $\alpha L = 5$ and $\alpha h_r = -5$, matching
# the dimensionless grouping used in the gwassess benchmark suite.

# %%
# --- Default values (edit these in a notebook) ---
RES        = 32       # elements per side
ALPHA      = 5.0      # 1/m – Gardner sorptive number
HR         = -1.0     # m – reference pressure head (dry end)
L          = 1.0      # m – domain size (square L×L)
THETA_R    = 0.15     # – residual water content
THETA_S    = 0.45     # – saturated water content
KS         = 1.0      # m/s – saturated hydraulic conductivity
DT         = 0.05     # s – timestep
N_STEPS    = 40       # – number of timesteps to reach steady state
BC_TYPE    = "no_flux" # – boundary condition type

params = uw.Params(
    uw_res      = RES,
    uw_alpha    = uw.Param(ALPHA,   units="1/m",  description="Gardner sorptive number"),
    uw_hr       = uw.Param(HR,      units="m",    description="reference pressure head"),
    uw_L        = uw.Param(L,       units="m",    description="domain size"),
    uw_theta_r  = THETA_R,
    uw_theta_s  = THETA_S,
    uw_Ks       = uw.Param(KS,      units="m/s",  description="saturated conductivity"),
    uw_dt       = uw.Param(DT,      units="s",    description="timestep"),
    uw_n_steps  = N_STEPS,
    uw_bc_type  = BC_TYPE,
)

res     = int(params.uw_res)
alpha   = float(params.uw_alpha)
hr      = float(params.uw_hr)
L_dom   = float(params.uw_L)
theta_r = float(params.uw_theta_r)
theta_s = float(params.uw_theta_s)
Ks      = float(params.uw_Ks)
dt      = float(params.uw_dt)
n_steps = int(params.uw_n_steps)
bc_type = str(params.uw_bc_type)

# Derived parameters
c_time = alpha * (theta_s - theta_r) / Ks   # characteristic time scale
h0 = 1 - np.exp(alpha * hr)                 # driving amplitude

print(f"alpha*L = {alpha * L_dom:.1f},  alpha*hr = {alpha * hr:.1f}")
print(f"c (time scale) = {c_time:.2f} s,  h0 = {h0:.4f}")
print(f"Total time = {n_steps * dt:.1f} s  ({n_steps * dt / c_time:.1f} × c)")

# %% [markdown]
# ## Mesh and variables

# %%
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(res, res),
    minCoords=(0.0, 0.0),
    maxCoords=(L_dom, L_dom),
    qdegree=3,
)

psi = uw.discretisation.MeshVariable(r"\psi", mesh, 1, degree=2)
v_soln = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=1)

# %% [markdown]
# ## Solver setup
#
# The Tracy benchmark uses the Gardner exponential model, which we take
# from the retention curves module.

# %%
from underworld3.utilities.retention_curves import gardner_K, gardner_theta

psi_sym = psi.sym[0]

richards = uw.systems.Richards(mesh, psi, v_soln, order=1, theta=0.5)
richards.petsc_options.delValue("ksp_monitor")
richards.petsc_options["snes_rtol"] = 1.0e-6
richards.petsc_options["snes_max_it"] = 30

richards.constitutive_model = uw.constitutive_models.DarcyFlowModel
richards.constitutive_model.Parameters.permeability = gardner_K(psi_sym, Ks=Ks, alpha=alpha)
richards.constitutive_model.Parameters.s = sympy.Matrix([0, -1]).T

# Mixed form — mass-conservative
richards.water_content = gardner_theta(psi_sym, theta_r=theta_r, theta_s=theta_s, alpha=alpha)
richards.f = 0.0

richards._v_projector.petsc_options["snes_rtol"] = 1.0e-6
richards._v_projector.smoothing = 1.0e-6

# %% [markdown]
# ## Boundary conditions
#
# The Tracy analytical solution has $\psi = h_r$ (dry state) on the bottom and
# lateral boundaries, but a **spatially varying** wetting profile on the top
# boundary.
#
# For the **no-flux** case:
# - Bottom ($y=0$): $\psi = h_r$
# - Left, Right ($x=0, L$): zero flux (natural BC — no Dirichlet)
# - Top ($y=L$): $\psi(x) = \frac{1}{\alpha}\ln\!\left[e^{\alpha h_r}
#   + \frac{h_0}{2}\left(1 - \cos\frac{2\pi x}{L}\right)\right]$
#
# For the **specified-head** case:
# - Bottom, Left, Right: $\psi = h_r$
# - Top: $\psi(x) = \frac{1}{\alpha}\ln\!\left[e^{\alpha h_r}
#   + h_0 \sin\frac{\pi x}{L}\right]$
#
# where $h_0 = 1 - e^{\alpha h_r}$.

# %%
x_sym = mesh.X[0]
h0_sym = 1 - sympy.exp(alpha * hr)

if bc_type == "no_flux":
    psi_top = (1 / alpha) * sympy.log(
        sympy.exp(alpha * hr) + (h0_sym / 2) * (1 - sympy.cos(2 * sympy.pi * x_sym / L_dom))
    )
elif bc_type == "specified_head":
    psi_top = (1 / alpha) * sympy.log(
        sympy.exp(alpha * hr) + h0_sym * sympy.sin(sympy.pi * x_sym / L_dom)
    )
else:
    raise ValueError(f"Unknown bc_type: {bc_type!r}. Use 'no_flux' or 'specified_head'.")

richards.add_dirichlet_bc([psi_top], "Top")
richards.add_dirichlet_bc([hr], "Bottom")

if bc_type == "specified_head":
    richards.add_dirichlet_bc([hr], "Left")
    richards.add_dirichlet_bc([hr], "Right")

# %% [markdown]
# ## Initial condition
#
# Start from the uniform dry state $\psi = h_r$ everywhere. The transient
# solution evolves from this initial condition towards steady state driven
# by the wetting profile on the top boundary.

# %%
psi.array[:, 0, 0] = hr

# %% [markdown]
# ## Time stepping
#
# We step forward in time until the solution approaches steady state. The
# Tracy analytical solution gives us the exact transient profile at each
# time level.

# %%
time = 0.0

for step in range(n_steps):
    richards.solve(timestep=dt)
    time += dt

    if step % 10 == 0 or step == n_steps - 1:
        # Sample interior points to check convergence
        n_sample = 20
        sx = np.linspace(0.1 * L_dom, 0.9 * L_dom, n_sample)
        sy = np.linspace(0.1 * L_dom, 0.9 * L_dom, n_sample)
        xx, yy = np.meshgrid(sx, sy)
        sample_pts = np.column_stack([xx.ravel(), yy.ravel()])

        psi_num = uw.function.evaluate(psi.sym[0], sample_pts).squeeze()
        psi_exact = tracy_solution_on_grid(
            sample_pts, time, alpha, hr, L_dom, theta_r, theta_s, Ks, bc_type
        )
        max_err = np.max(np.abs(psi_num - psi_exact))
        l2_err = np.sqrt(np.mean((psi_num - psi_exact) ** 2))

        print(
            f"Step {step:3d}, t = {time:8.3f} s  |  "
            f"max |err| = {max_err:.4e}, L2 err = {l2_err:.4e}"
        )

# %% [markdown]
# ## Final comparison against analytical solution

# %%
# Dense grid for final comparison
n_final = 40
sx = np.linspace(0.05 * L_dom, 0.95 * L_dom, n_final)
sy = np.linspace(0.05 * L_dom, 0.95 * L_dom, n_final)
xx, yy = np.meshgrid(sx, sy)
sample_pts = np.column_stack([xx.ravel(), yy.ravel()])

psi_numerical = uw.function.evaluate(psi.sym[0], sample_pts).squeeze()
psi_analytical = tracy_solution_on_grid(
    sample_pts, time, alpha, hr, L_dom, theta_r, theta_s, Ks, bc_type
)

max_error = np.max(np.abs(psi_numerical - psi_analytical))
l2_error = np.sqrt(np.mean((psi_numerical - psi_analytical) ** 2))
rel_error = l2_error / np.sqrt(np.mean(psi_analytical**2))

print(f"\nFinal comparison at t = {time:.3f} s ({bc_type} BCs, {res}×{res} mesh)")
print(f"  Max absolute error:  {max_error:.4e}")
print(f"  L2 error:            {l2_error:.4e}")
print(f"  Relative L2 error:   {rel_error:.4e}")

# %% [markdown]
# ## Visualisation

# %%
if uw.is_notebook():
    import pyvista as pv
    import underworld3.visualisation as vis

    pv_mesh = vis.mesh_to_pv_mesh(mesh)
    pv_mesh.point_data["psi_numerical"] = vis.scalar_fn_to_pv_points(pv_mesh, psi.sym[0])

    # Evaluate analytical on pv mesh vertices
    pv_coords = np.array(pv_mesh.points[:, :2])
    pv_mesh.point_data["psi_analytical"] = tracy_solution_on_grid(
        pv_coords, time, alpha, hr, L_dom, theta_r, theta_s, Ks, bc_type
    )
    pv_mesh.point_data["error"] = (
        pv_mesh.point_data["psi_numerical"] - pv_mesh.point_data["psi_analytical"]
    )

    pl = pv.Plotter(shape=(1, 3), window_size=(1200, 400))

    pl.subplot(0, 0)
    pl.add_mesh(
        pv_mesh.copy(),
        scalars="psi_numerical",
        cmap="Blues_r",
        show_edges=False,
        scalar_bar_args={"title": "Numerical"},
    )
    pl.add_text(f"Numerical psi", font_size=10)
    pl.view_xy()

    pl.subplot(0, 1)
    pl.add_mesh(
        pv_mesh.copy(),
        scalars="psi_analytical",
        cmap="Blues_r",
        show_edges=False,
        scalar_bar_args={"title": "Analytical"},
    )
    pl.add_text(f"Analytical psi (Tracy)", font_size=10)
    pl.view_xy()

    pl.subplot(0, 2)
    pl.add_mesh(
        pv_mesh.copy(),
        scalars="error",
        cmap="RdBu_r",
        show_edges=False,
        scalar_bar_args={"title": "Error"},
    )
    pl.add_text(f"Error", font_size=10)
    pl.view_xy()

    pl.show()

# %% [markdown]
# ## Vertical profile comparison

# %%
if uw.is_notebook():
    import matplotlib.pyplot as plt

    # Profile at x = L/2
    n_profile = 100
    y_profile = np.linspace(0.02 * L_dom, 0.98 * L_dom, n_profile)
    x_profile = np.full_like(y_profile, 0.5 * L_dom)
    profile_pts = np.column_stack([x_profile, y_profile])

    psi_profile_num = uw.function.evaluate(psi.sym[0], profile_pts).squeeze()
    psi_profile_exact = tracy_solution_on_grid(
        profile_pts, time, alpha, hr, L_dom, theta_r, theta_s, Ks, bc_type
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(psi_profile_exact, y_profile, "k-", label="Tracy analytical", linewidth=2)
    ax1.plot(psi_profile_num, y_profile, "ro", label="UW3 numerical", markersize=4)
    ax1.set_xlabel(r"Pressure head $\psi$ [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title(f"Vertical profile at x = L/2 (t = {time:.2f} s)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(psi_profile_num - psi_profile_exact, y_profile, "b-", linewidth=1.5)
    ax2.set_xlabel(r"Error $\psi_{num} - \psi_{exact}$ [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Error profile")
    ax2.axvline(0, color="k", linestyle="--", alpha=0.3)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## Quantitative assessment
#
# For a production-quality benchmark, the error should decrease with mesh
# refinement. Run at multiple resolutions to check convergence:
#
# ```bash
# python Ex_Richards_Tracy_Benchmark.py -uw_res 16
# python Ex_Richards_Tracy_Benchmark.py -uw_res 32
# python Ex_Richards_Tracy_Benchmark.py -uw_res 64
# ```
#
# With degree-2 elements, we expect roughly 4th-order convergence
# (halving the element size should reduce the error by ~16×).

# %%
print(f"\nBenchmark: Tracy (2006) 2D Richards equation")
print(f"BC type:     {bc_type}")
print(f"Resolution:  {res}×{res}")
print(f"Parameters:  alpha={alpha}, hr={hr}, L={L_dom}, Ks={Ks}")
print(f"             alpha*L={alpha*L_dom:.1f}, alpha*hr={alpha*hr:.1f}, c={c_time:.2f}")
print(f"Time:        t={time:.3f} s ({n_steps} steps × dt={dt})")
print(f"Max error:   {max_error:.4e}")
print(f"L2 error:    {l2_error:.4e}")
print(f"Rel L2 err:  {rel_error:.4e}")

if max_error < 0.05:
    print("\nPASSED: Max error < 0.05")
else:
    print(f"\nFAILED: Max error = {max_error:.4e} (threshold 0.05)")
