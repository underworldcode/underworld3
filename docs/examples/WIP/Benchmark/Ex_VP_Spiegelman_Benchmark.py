# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Spiegelman et al, notch-deformation benchmark
#
# This example is for the notch-localization test of [Spiegelman et al., 2016](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015GC006228) For which they supply a [geometry file](https://bitbucket.org/mspieg/plasticitymodels/src/master/) which gmsh can use to construct meshes at various resolutions. The same setup is used in [Fraters et al., 2018](https://academic.oup.com/gji/article/218/2/873/5475649)
#
#
# - [Aspect setup file](https://github.com/geodynamics/aspect/tree/main/benchmarks/newton_solver_benchmark_set/spiegelman_et_al_2016)
# - [G-Adopt setup file](https://github.com/g-adopt/g-adopt/blob/v1.2.0/Davies_etal_GMD_2021/Drucker-Prager_rheology/spiegelman.py)
#
#
# The `.geo` file is provided and we show how to make this into a `.msh` file and
# how to read that into a `uw.discretisation.Mesh` object. The `.geo` file has header parameters to control the mesh refinement, and we provide a coarse version and the original version.
#
# After that, there is some cell data which we can assign to a data structure on the elements (such as a swarm).

# %%
#|  echo: false  # Hide in html version

# This is required to fix pyvista 
# (visualisation) crashes in interactive notebooks (including on binder)

import nest_asyncio
nest_asyncio.apply()

# %%
import os
import underworld3 as uw
import numpy as np
import sympy
import gmsh

if uw.is_notebook():
    import matplotlib.pyplot as plt


# %% [markdown]
# ### Configurable parameters
#
# All tuneable settings are collected here in a single `uw.Params` block.
# Default values are defined as named constants so they are easy to find
# and adjust in a notebook.  From the command line, override them with
# PETSc-style flags.  Unit-aware parameters (`uw.Param` with `units=`)
# **require explicit units on the CLI** so that dimension mismatches are
# caught early.
#
# ```bash
# # Override viscosity and convergence rate
# python Ex_VP_Spiegelman_Benchmark.py \
#     -uw_eta_background "5e23 Pa*s" \
#     -uw_convergence_rate "5 mm/yr"
#
# # Change mesh resolution and solver settings
# python Ex_VP_Spiegelman_Benchmark.py \
#     -uw_problem_size 2 -uw_smoothing 1e-4
#
# # Works with MPI
# mpirun -np 4 python Ex_VP_Spiegelman_Benchmark.py \
#     -uw_eta_base "1e20 Pa*s" -uw_problem_size 3
# ```
#
# Use `params.cli_help()` to see all available options with their units and bounds.

# %%

# --- Default values (edit these in a notebook) ---
ETA_BACKGROUND   = 1e24   # Pa·s – background viscosity
ETA_BASE         = 1e21   # Pa·s – base (weak) viscosity
CONVERGENCE_RATE = 2.5    # mm/yr – convergence velocity
PROBLEM_SIZE     = 2      # mesh resolution level (1=ultra-low, 4=benchmark)
SMOOTHING        = 3e-5   # regularisation parameter
N_ITER           = 10     # DD→DP continuation steps
USE_HOMOTOPY     = True   # yield homotopy (ramp soft-min δ→0) in place of Picard
DELTA_START      = 1.0    # initial soft-min δ for the homotopy ramp
REFINEMENT       = 0      # geometric-MG refinement levels on the gmsh base (0 = none)
PRECONDITIONER   = "auto" # "auto"|"fmg"|"gamg" velocity-block preconditioner

params = uw.Params(
    # Physical parameters (unit-aware)
    uw_eta_background   = uw.Param(ETA_BACKGROUND,   units="Pa*s", description="background viscosity"),
    uw_eta_base         = uw.Param(ETA_BASE,          units="Pa*s", description="base (weak) viscosity"),
    uw_convergence_rate = uw.Param(CONVERGENCE_RATE,  units="mm/yr", description="convergence velocity"),

    # Solver / discretisation settings (plain types)
    uw_problem_size     = uw.Param(PROBLEM_SIZE, bounds=(1, 4), description="mesh resolution level (1=ultra-low, 4=benchmark)"),
    uw_smoothing        = SMOOTHING,
    uw_p_cont           = True,
    uw_p_deg            = 1,       # pressure polynomial degree
    uw_v_deg            = 2,       # velocity polynomial degree
    uw_niter            = N_ITER,  # nonlinear iterations for DD→DP transition
    uw_refinement       = REFINEMENT,      # geometric-MG hierarchy levels on the gmsh base
    uw_preconditioner   = PRECONDITIONER,  # "auto"|"fmg"|"gamg"
)

# Convenience aliases for plain-type parameters used throughout
problem_size = params.uw_problem_size
smoothing    = params.uw_smoothing
p_cont       = params.uw_p_cont
p_deg        = params.uw_p_deg
v_deg        = params.uw_v_deg
niter        = params.uw_niter
refinement   = params.uw_refinement
preconditioner = params.uw_preconditioner

d_eta = np.log10(params.uw_eta_background.magnitude) - np.log10(params.uw_eta_base.magnitude)

# %% [markdown]
# ### Model scaling
#
# The unit-aware parameters from `uw.Params` are already `UWQuantity` objects,
# so they plug directly into the `Model` scaling and into `uw.expression()`
# wrappers. The solver non-dimensionalises automatically — no manual `nd()`
# calls needed.
#
# The Model must be set up **before** mesh creation so that the mesh
# inherits the correct coordinate units (km).

# %%
# Reference quantities that define the scaling (Table 1, Spiegelman et al. 2016)
H = uw.quantity(30, "km")   # model height

# Set up scaling via the Model — the solver non-dimensionalises automatically
model = uw.Model()
model.set_reference_quantities(
    length    = H,
    velocity  = params.uw_convergence_rate,
    viscosity = params.uw_eta_background,
)

# Physical parameters as named expressions (for display and symbolic debugging)
V_conv   = uw.expression(r"V_{\mathrm{conv}}", params.uw_convergence_rate,  "convergence velocity")
rho_0    = uw.expression(r"\rho_0",    uw.quantity(2700, "kg/m**3"),        "reference density")
g        = uw.expression(r"g",         uw.quantity(9.81, "m/s**2"),         "gravitational acceleration")
C        = uw.expression(r"C",         uw.quantity(1e8, "Pa"),              "cohesion")
eta_bg   = uw.expression(r"\eta_{bg}", params.uw_eta_background,            "background viscosity")
eta_b    = uw.expression(r"\eta_{b}",  params.uw_eta_base,                  "base viscosity")

# %%
os.makedirs("meshes", exist_ok=True)

outputPath = (
    f"./output/plasticityBenchmark/"
    f"SpiegelmanBenchmark_size={problem_size}"
    f"_eta_base={params.uw_eta_base.magnitude:g}"
    f"_eta_background={params.uw_eta_background.magnitude:g}"
    f"_conv_v={params.uw_convergence_rate.magnitude:g}"
    f"_smoothing={smoothing}"
    f"_vdeg={v_deg}_pdeg={p_deg}_pcont={p_cont}/"
)
if uw.mpi.rank == 0:
    os.makedirs(outputPath, exist_ok=True)



# %% jupyter={"source_hidden": true}
### Set up the mesh — geometry in km (Table 1, Spiegelman et al. 2016)
#
# The gmsh geometry is defined in non-dimensional units of H (model height),
# then scaled to km.  Domain is 4H × H = 120 km × 30 km.

S = H.magnitude  # 30.0 — scale factor: non-dimensional → km

if problem_size <= 1:
    cl_1 = 0.25 * S
    cl_2 = 0.15 * S
    cl_2a = 0.1 * S
    cl_3 = 0.25 * S
    cl_4 = 0.15 * S
elif problem_size == 2:
    cl_1 = 0.1 * S
    cl_2 = 0.05 * S
    cl_2a = 0.03 * S
    cl_3 = 0.1 * S
    cl_4 = 0.05 * S
elif problem_size == 3:
    cl_1 = 0.06 * S
    cl_2 = 0.03 * S
    cl_2a = 0.015 * S
    cl_3 = 0.04 * S
    cl_4 = 0.02 * S
else:
    cl_1 = 0.04 * S
    cl_2 = 0.005 * S
    cl_2a = 0.003 * S
    cl_3 = 0.02 * S
    cl_4 = 0.01 * S

if uw.mpi.rank == 0:

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("Notch")

    # Domain outline (non-dimensional × S → km)
    Point1  = gmsh.model.geo.addPoint(-2 * S,      -1 * S,    0, cl_1)
    Point3  = gmsh.model.geo.addPoint(+2 * S,      -1 * S,    0, cl_1)
    Point4  = gmsh.model.geo.addPoint( 2 * S,    -3/4 * S,    0, cl_1)
    Point5  = gmsh.model.geo.addPoint( 2 * S,           0,    0, cl_1)
    Point6  = gmsh.model.geo.addPoint(-2 * S,           0,    0, cl_1)
    Point7  = gmsh.model.geo.addPoint(-2 * S,    -3/4 * S,    0, cl_1)
    # Top-surface refinement zone
    Point25 = gmsh.model.geo.addPoint(-3/4 * S,         0,    0, cl_4)
    Point26 = gmsh.model.geo.addPoint( 3/4 * S,         0,    0, cl_4)
    Point27 = gmsh.model.geo.addPoint(       0,         0,    0, cl_3)

    Line1 = gmsh.model.geo.addLine(Point1, Point3)
    Line2 = gmsh.model.geo.addLine(Point3, Point4)
    Line3 = gmsh.model.geo.addLine(Point4, Point5)
    Line4 = gmsh.model.geo.addLine(Point5, Point26)
    Line8 = gmsh.model.geo.addLine(Point26, Point27)
    Line9 = gmsh.model.geo.addLine(Point27, Point25)
    Line10 = gmsh.model.geo.addLine(Point25, Point6)
    Line6 = gmsh.model.geo.addLine(Point6, Point7)
    Line7 = gmsh.model.geo.addLine(Point7, Point1)

    # Rounded notch corners — r = 0.02H (corner radius)
    rc = 0.02
    # Points on the notch boundary (tangent points of the arcs)
    Point12 = gmsh.model.geo.addPoint(-(1/12 + rc) * S,   -3/4 * S,        0, cl_2a)
    Point13 = gmsh.model.geo.addPoint( -1/12 * S,        -(3/4 - rc) * S,  0, cl_2a)
    Point14 = gmsh.model.geo.addPoint( -1/12 * S,        -(2/3 + rc) * S,  0, cl_2a)
    Point15 = gmsh.model.geo.addPoint(-(1/12 - rc) * S,   -2/3 * S,        0, cl_2a)
    Point16 = gmsh.model.geo.addPoint( (1/12 - rc) * S,   -2/3 * S,        0, cl_2a)
    Point17 = gmsh.model.geo.addPoint(  1/12 * S,        -(2/3 + rc) * S,  0, cl_2a)
    Point18 = gmsh.model.geo.addPoint(  1/12 * S,        -(3/4 - rc) * S,  0, cl_2a)
    Point19 = gmsh.model.geo.addPoint( (1/12 + rc) * S,   -3/4 * S,        0, cl_2a)
    # Arc centres
    Point20 = gmsh.model.geo.addPoint(-(1/12 + rc) * S,  -(3/4 - rc) * S,  0, cl_2a)
    Point21 = gmsh.model.geo.addPoint(-(1/12 - rc) * S,  -(2/3 + rc) * S,  0, cl_2a)
    Point22 = gmsh.model.geo.addPoint( (1/12 - rc) * S,  -(2/3 + rc) * S,  0, cl_2a)
    Point24 = gmsh.model.geo.addPoint( (1/12 + rc) * S,  -(3/4 - rc) * S,  0, cl_2a)

    Circle22 = gmsh.model.geo.addCircleArc(Point12, Point20, Point13)
    Circle23 = gmsh.model.geo.addCircleArc(Point14, Point21, Point15)
    Circle24 = gmsh.model.geo.addCircleArc(Point16, Point22, Point17)
    Circle25 = gmsh.model.geo.addCircleArc(Point18, Point24, Point19)

    Line26 = gmsh.model.geo.addLine(Point7, Point12)
    Line27 = gmsh.model.geo.addLine(Point13, Point14)
    Line28 = gmsh.model.geo.addLine(Point15, Point16)
    Line29 = gmsh.model.geo.addLine(Point17, Point18)
    Line30 = gmsh.model.geo.addLine(Point19, Point4)

    LineLoop31 = gmsh.model.geo.addCurveLoop(
        [
            Line1,
            Line2,
            -Line30,
            -Circle25,
            -Line29,
            -Circle24,
            -Line28,
            -Circle23,
            -Line27,
            -Circle22,
            -Line26,
            Line7,
        ],
    )

    LineLoop33 = gmsh.model.geo.addCurveLoop(
        [
            Line6,
            Line26,
            Circle22,
            Line27,
            Circle23,
            Line28,
            Circle24,
            Line29,
            Circle25,
            Line30,
            Line3,
            Line4,
            Line8,
            Line9,
            Line10,
        ],
    )

    Surface32 = gmsh.model.geo.addPlaneSurface([LineLoop31])
    Surface34 = gmsh.model.geo.addPlaneSurface([LineLoop33])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [Line1], tag=3, name="Bottom")
    gmsh.model.addPhysicalGroup(1, [Line2, Line3], tag=2, name="Right")
    gmsh.model.addPhysicalGroup(1, [Line7, Line6], tag=1, name="Left")
    gmsh.model.addPhysicalGroup(1, [Line4, Line8, Line9, Line10], tag=4, name="Top")

    gmsh.model.addPhysicalGroup(
        1,
        [
            Line26,
            Circle22,
            Line27,
            Circle23,
            Line28,
            Circle24,
            Line29,
            Circle25,
            Line30,
        ],
        tag=5,
        name="Inner",
    )

    gmsh.model.addPhysicalGroup(2, [Surface32], tag=100, name="Weak")
    gmsh.model.addPhysicalGroup(2, [Surface34], tag=101, name="Strong")

    gmsh.model.mesh.generate(2)

    gmsh.write(f"{outputPath}notch_mesh{problem_size}.msh")
    gmsh.finalize()


plex = uw.discretisation._from_gmsh(f"{outputPath}notch_mesh{problem_size}.msh",
            useMultipleTags=True,
            useRegions=True,)

if uw.is_notebook():
    plex[1].view()

# %%
from petsc4py import PETSc
mesh_file = PETSc.DMPlex().createFromFile(f"{outputPath}notch_mesh{problem_size}.msh")


### match boundary labels from plex to the boundaries class
from enum import Enum

class boundaries(Enum):
    Bottom = 3
    Top = 4
    Right = 2
    Left = 1
    Inner = 5



# refinement > 0 builds a geometric-MG hierarchy by uniformly refining the gmsh
# base (the coarse level). The new boundary vertices are NOT snapped back onto
# the constructive (gmsh CAD) geometry — the resulting chord-arc error on the
# tiny rounded notch corners is benign (cf. the annulus FMG study) and the
# straight notch flanks / interior interface refine exactly. The hierarchy lets
# the velocity block use geometric Full Multigrid (preconditioner="fmg"/"auto").
mesh1 = uw.discretisation.Mesh(plex[1], boundaries=boundaries, useMultipleTags=True, useRegions=True, coordinate_system_type=uw.coordinates.CoordinateSystemType.CARTESIAN, qdegree=3, refinement=refinement,)

### view mesh to make sure boundaries are labeled correctly
if uw.is_notebook():
    mesh1.view()


# %%
### stokes mesh vars
v_soln = uw.discretisation.MeshVariable(r"U", mesh1, mesh1.dim, degree=v_deg, continuous=True)

p_soln = uw.discretisation.MeshVariable(r"P", mesh1, 1, degree=p_deg, continuous=p_cont)

mat = uw.discretisation.MeshVariable(r"mat", mesh1, 1, degree=0, continuous=False)


### model parameters for visualisation
edot = uw.discretisation.MeshVariable(
    r"\dot\varepsilon", mesh1, 1, degree=0)

edot_0 = uw.discretisation.MeshVariable(
    r"\dot\varepsilon_0", mesh1, 1, degree=0)  # linear (no plasticity) strain rate

visc = uw.discretisation.MeshVariable(r"\eta", mesh1, 1, degree=0)
stress = uw.discretisation.MeshVariable(r"\sigma", mesh1, 1, degree=0)



# %%
if uw.is_notebook():
    v_soln.view()

# %% [markdown]
# This is how we extract cell data from the mesh. We can map it to the swarm data structure and use this to
# build material properties that depend on cell type.

# %%
indexSetW = mesh1.dm.getStratumIS("Weak", 100)
indexSetS = mesh1.dm.getStratumIS("Strong", 101)


# %%
# Direct array access. The "Weak"/"Strong" region labels propagate through the
# refinement hierarchy, but refine() spreads them onto the cell *closure* too
# (vertices/edges), so getStratumIS returns more than just cells. Keep only the
# cell points (height-0 stratum [cStart,cEnd)) and index the DG0 `mat` data by
# cell. On the unrefined gmsh mesh the label was cells-only with cStart==0, so
# the original raw indexing happened to work.
cStart, cEnd = mesh1.dm.getHeightStratum(0)


def _cells_only(indexSet):
    if indexSet is None:
        return np.empty(0, dtype=np.int32)
    pts = indexSet.getIndices()
    return pts[(pts >= cStart) & (pts < cEnd)] - cStart


mat.data[...] = 1  # default: Strong
mat.data[_cells_only(indexSetW)] = 0
mat.data[_cells_only(indexSetS)] = 1
uw.pprint(0, f"material: {int((mat.data < 0.5).sum())} weak / {int((mat.data > 0.5).sum())} strong cells")

# %% [markdown]
# ### Create Stokes object

# %%
stokes = uw.systems.Stokes(
    mesh1,
    velocityField=v_soln,
    pressureField=p_soln,
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

# Velocity-block preconditioner. "auto"/"fmg" lights up geometric Full Multigrid
# when the mesh carries a refinement hierarchy (refinement>0) — anisotropy- and
# contrast-robust where the default algebraic multigrid (GAMG) cliffs as the
# plastic viscosity gradient sharpens. Falls back to GAMG with no hierarchy.
stokes.preconditioner = preconditioner


# %% [markdown]
# ##### Setup projections of model parameters to save on the mesh

# %%
def update_projections():
    strain_rate_calc = uw.systems.Projection(mesh1, edot)
    strain_rate_calc.uw_function = stokes.Unknowns.Einv2
    strain_rate_calc.smoothing = smoothing
    strain_rate_calc.solve()
    
    
    viscosity_calc = uw.systems.Projection(mesh1, visc)
    viscosity_calc.uw_function = stokes.constitutive_model.Parameters.shear_viscosity_0
    viscosity_calc.smoothing = smoothing
    viscosity_calc.solve()
    
    stress_calc = uw.systems.Projection(mesh1, stress)
    # S = stokes.stress_deviator
    stress_calc.uw_function = stokes.stress_1d[-1]
    stress_calc.smoothing = smoothing
    stress_calc.solve()


# %%
# Velocity boundary conditions
# V_conv is a uw.expression — the solver non-dimensionalises automatically
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes.add_dirichlet_bc((V_conv, 0.0), "Left")
stokes.add_dirichlet_bc((-V_conv, 0.0), "Right")

# %%
# Body force: ρ₀ * g acting downward
# Using expressions — the solver handles non-dimensionalisation
stokes.bodyforce = sympy.Matrix([0, -rho_0 * g])


# %% [markdown]
# #### Set solve options here 
# or remove default values

# %%
stokes.tolerance = 1e-8  # Balanced for speed and convergence

### see the SNES output
stokes.petsc_options["snes_converged_reason"] = None
stokes.petsc_options["snes_monitor_short"] = None
stokes.petsc_options["ksp_monitor"] = None


# %% [markdown]
# #### Alternative PETSc solver options (reference)
#
# A Schur-complement fieldsplit preconditioner with direct sub-solvers
# gives faster convergence at higher resolution (~20k elements on 4 CPUs).
# See [PETSc ex69](https://gitlab.spack.io/petsc/petsc/-/blob/xsdk-0.2.0/src/snes/examples/tutorials/output/ex69_p2p1.out).
# Uncomment the block below to use it instead of the defaults.

# %%
# Viscosity combination functions

def eta_harmonic_mean(*etas):
    """
    Calculate the harmonic mean of viscosities.
    Commonly used for combining diffusion and dislocation creep.
    """
    n = len(etas)
    return n / sympy.Add(*[1/eta for eta in etas])

def eta_minimum(*etas):
    """
    Return the minimum viscosity.
    Simple min() operation for viscosities.
    """
    return sympy.Min(*etas)

def eta_geometric_mean(*etas):
    """
    Calculate the geometric mean of viscosities.
    """
    return sympy.Mul(*etas) ** (1/len(etas))


# %% [markdown]
# #### Yield homotopy (problem-space regularisation)
#
# The plastic viscosity cap $\min(\eta_p, \eta_{bg})$ has a non-differentiable
# kink that stalls the SNES line search from a cold start — which is why the
# original benchmark leans on a Picard pre-step and the DD→DP $\alpha$
# continuation.  Instead we replace the sharp $\min$ with the unified
# $\delta$-parameterised soft-min (identical to UW3's
# ``ViscousFlowModel._combine_yield``) and ramp $\delta$ from ``DELTA_START``
# down to 0 *within each solve* via a per-iteration ``SNESSetUpdate`` callback.
# $\delta$ rides in ``constants[]`` (no JIT recompile), and because it ends at
# 0 the converged stress sits on the **exact** yield surface.
#
# This mirrors ``constitutive_model.enable_yield_homotopy()``; it is written
# inline here because the benchmark builds its piecewise viscosity by hand on a
# plain ``ViscousFlowModel`` rather than through a yield-aware constitutive model.

# %%
# δ and its onset offset as runtime-rampable constant expressions.
delta_y = uw.expression(r"\delta_y", sympy.Float(0.0), "yield soft-min regularisation")
offset_y = uw.expression(
    r"\delta_{y,0}", (-1 + sympy.sqrt(1 + delta_y**2)) / 2, "soft-min onset offset"
)


def eta_softmin(eta_pl, eta_bg):
    """δ-parameterised soft-min of (plastic, background) viscosity.

    g(f,δ)=1+(f-1+√((f-1)²+δ²))/2-offset, f=η_pl/η_bg; δ=0 ⇒ exact min."""
    f = eta_pl / eta_bg
    g = 1 + (f - 1 + sympy.sqrt((f - 1) ** 2 + delta_y**2)) / 2 - offset_y
    return eta_pl / g


class YieldHomotopy:
    """Residual-paced δ ramp, mirroring enable_yield_homotopy (absolute schedule)."""

    def __init__(self, solver, delta_start):
        self.delta_start = float(delta_start)
        self.r_ref = 0.0
        self.r_min = 0.0
        solver.petsc_options["snes_linesearch_type"] = "basic"
        if solver.petsc_options.getInt("snes_max_it", 0) < 100:
            solver.petsc_options["snes_max_it"] = 100

    def __call__(self, solver, iteration):
        try:
            rnorm = float(solver.snes.getFunctionNorm())
        except Exception:
            rnorm = None
        if iteration == 0:
            self.r_min = rnorm if rnorm else 0.0
            if rnorm and rnorm > self.r_ref:
                self.r_ref = rnorm
        elif rnorm:
            self.r_min = min(self.r_min, rnorm)
        scale = min(1.0, max(0.0, self.r_min / self.r_ref)) if self.r_ref > 0 else (
            1.0 if iteration == 0 else 0.0
        )
        delta_y.sym = sympy.Float(self.delta_start * scale)
        solver._update_constants()


# %% [markdown]
# #### Drucker-Prager plasticity
#
# The yield stress is $\sigma_y = C \cos\varphi + \sin\varphi\, P$ where $C$ is
# cohesion, $\varphi$ is the friction angle, and $P$ is the pressure used in
# the yield criterion.
#
# **Pressure decomposition** — The Stokes solver produces **total pressure**
# $p = p_{\text{litho}} + p_{\text{dyn}}$ because the body force $\rho g$ is
# included in the momentum equation.  There is no separate "reference pressure".
#
# The continuation parameter $\alpha$ (Spiegelman et al. 2016) controls
# how much of the dynamic pressure enters the yield criterion:
#
# $$P_{\text{eff}} = p_{\text{litho}} + \alpha\, p_{\text{dyn}}
#                  = (1-\alpha)\, p_{\text{litho}} + \alpha\, p_{\text{total}}$$
#
# - $\alpha = 0$: depth-dependent only (DD) — uses analytical $\rho g d$
# - $\alpha = 1$: full Drucker-Prager (DP) — uses solver pressure directly
#
# The plastic viscosity is then $\eta_p = \sigma_y\, /\, 2\dot\varepsilon_{II}$.

# %%
# Friction angle and coefficient
phi = 30                              # degrees
fc  = np.sin(np.deg2rad(phi))         # sin(φ) — Drucker-Prager coefficient

# Lithostatic pressure: ρ₀ g depth  (depth = -y, since y=0 at top)
lithoP = rho_0 * g * (-mesh1.X[1])

# Strain-rate invariant (with small regularisation to avoid 0/0)
edot_II = stokes.Unknowns.Einv2 + uw.maths.functions.vanishing

# %%
# Yield stress as a function of the continuation parameter α
#
# α = 0: DD (depth-dependent only, P_eff = P_litho)
# α = 1: full DP (P_eff = p_soln, the solver's total pressure)
#
# NOTE: p_soln IS total pressure (litho + dynamic) because the body force
# ρg is included in the Stokes momentum equation.  Do NOT add lithoP to
# p_soln — that would double-count the lithostatic component.

def yield_stress(alpha):
    """Yield stress for continuation parameter α ∈ [0, 1]."""
    P_eff = (1 - alpha) * lithoP + alpha * p_soln.sym[0]
    sigma_y = C + fc * P_eff
    # Floor at zero — tensile yield is not physical for Drucker-Prager
    return sympy.Max(sigma_y, sympy.sympify(0))

def plastic_viscosity(alpha):
    """Plastic viscosity η_p = σ_y / (2 ε̇_II) for given α."""
    return yield_stress(alpha) / (2 * edot_II)

# %%
# STAGE 1: Linear solve (constant viscosity, no plasticity)
# Stabilises the solution from scratch

uw.pprint(0, "\n" + "=" * 60)
uw.pprint(0, "STAGE 1: Solving with background viscosity only (no plasticity)")
uw.pprint(0, "=" * 60 + "\n")

stokes.petsc_options["snes_monitor"] = ":" + os.path.join(outputPath, "1_NL_dp_linear_stage_picard.txt")

# Piecewise viscosity: η_bg in "Strong" region, η_b in "Weak" region
# Using expressions — solver non-dimensionalises automatically
visc_fn_linear = sympy.Piecewise((eta_bg, mat.sym[0] > 0.5),
                                  (eta_b, True))

stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn_linear
stokes.saddle_preconditioner = 1 / visc_fn_linear

stokes.solve(zero_init_guess=False, picard=0)

uw.pprint(0, "Linear stage complete\n")

update_projections()

# Save linear strain rate for comparison with plastic solution
edot_0.data[...] = edot.data

mesh1.petsc_save_checkpoint(index=0, meshVars=[edot, visc, stress, p_soln, v_soln], outputPath=outputPath)

# Save linear solution for restart
v_linear = v_soln.data.copy()
p_linear = p_soln.data.copy()

# %%
uw.pause("Linear solve — waiting before starting non-linear solve")

# %% [markdown]
# ### Stage 2: DD → DP continuation (Spiegelman et al. 2016)
#
# Gradually introduce dynamic-pressure dependence via the continuation
# parameter $\alpha$.  At each step the effective viscosity in the strong
# layer is:
#
# $$\eta_{\text{eff}} = \eta_b + H\!\left(\eta_p(\alpha),\; \eta_{\text{bg}}\right)$$
#
# where $H$ is the harmonic mean and $\eta_p(\alpha) = \sigma_y(\alpha) / 2\dot\varepsilon_{II}$.
# The floor $\eta_b$ prevents zero viscosity; the harmonic mean with
# $\eta_{\text{bg}}$ caps the plastic viscosity at the background value.

# %%
# Continuation schedule.
#   - Yield homotopy ON: jump straight to full Drucker-Prager (α = 1) and let
#     the δ-ramp carry the hard solve from the linear guess — no DD→DP
#     continuation, no Picard pre-step.
#   - Yield homotopy OFF: the original DD→DP α-continuation + Picard pre-step.
alpha_steps = [1.0] if USE_HOMOTOPY else [0.0, 0.25, 0.5, 0.75, 1.0]

# Register the residual-paced δ ramp once (fires every SNES iteration).
if USE_HOMOTOPY:
    stokes.add_update_callback(YieldHomotopy(stokes, DELTA_START))

for step_i, alpha in enumerate(alpha_steps):

    uw.pprint(0, "\n" + "=" * 60)
    uw.pprint(0, f"STAGE 2.{step_i}: Plasticity with α = {alpha}"
                 f"{'  [yield homotopy]' if USE_HOMOTOPY else ''}")
    uw.pprint(0, "=" * 60 + "\n")

    # Restore linear solution as initial guess for each step
    with uw.synchronised_array_update():
        v_soln.data[...] = v_linear
        p_soln.data[...] = p_linear

    stokes.petsc_options["snes_monitor"] = ":" + os.path.join(
        outputPath, f"2_{step_i}_NL_alpha_{alpha:.2f}.txt"
    )

    # Effective viscosity: (soft-)min of plastic and background, plus floor.
    eta_p = plastic_viscosity(alpha)
    cap = eta_softmin(eta_p, eta_bg) if USE_HOMOTOPY else eta_minimum(eta_p, eta_bg)
    visc_top = eta_b + cap

    visc_fn_plastic = sympy.Piecewise((visc_top, mat.sym[0] > 0.5),
                                       (eta_b, True))

    stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn_plastic
    stokes.saddle_preconditioner = 1 / visc_fn_plastic

    # Homotopy carries the hard solve, so no Picard pre-step. Without it, the
    # first α-step needs Picard to establish the plastic regime.
    n_picard = 0 if USE_HOMOTOPY else (5 if step_i == 0 else 0)
    stokes.solve(zero_init_guess=False, picard=n_picard)
    uw.pprint(0, f"  SNES reason={int(stokes.snes.getConvergedReason())} "
                 f"its={stokes.snes.getIterationNumber()}")

    update_projections()
    mesh1.petsc_save_checkpoint(
        index=step_i + 1,
        meshVars=[edot, visc, stress, p_soln, v_soln],
        outputPath=outputPath,
    )

    # Use converged solution as initial guess for next α step
    v_linear = v_soln.data.copy()
    p_linear = p_soln.data.copy()

    uw.pprint(0, f"α = {alpha} complete\n")

uw.pprint(0, "Done.")

# %% [markdown]
# ### In-process strain-rate render (cell-centred)
#
# Renders the converged strain-rate localisation directly from the live ``edot``
# variable (cell-centred DG0 — no checkpoint round-trip, no point smearing), with
# the gmsh corner arcs overlaid. Serial only (a parallel render would show one
# rank's partition). One-sided ``Oranges`` colormap, log scale.

# %%
if uw.mpi.size == 1:
    import underworld3.visualisation as vis
    import pyvista as pv
    pv.OFF_SCREEN = True

    rc_ = 0.02
    centres_H = [(-(1/12+rc_), -(3/4-rc_)), (-(1/12-rc_), -(2/3+rc_)),
                 ((1/12-rc_), -(2/3+rc_)), ((1/12+rc_), -(3/4-rc_))]

    ev = np.asarray(edot.data[:, 0])
    pos = ev[ev > 0]
    clo, chi = np.percentile(pos, 5), np.percentile(pos, 99.5)
    pvm = vis.mesh_to_pv_mesh(mesh1)
    pvm.cell_data["edot"] = ev   # cell-centred

    # Coordinate frame from the pv mesh points (plain numpy, matches what is
    # drawn) so the overlaid arcs line up regardless of the coord units.
    pts = np.asarray(pvm.points)
    Hc = float(pts[:, 1].max() - pts[:, 1].min())
    Rc_ = rc_ * Hc
    centres = [(cx*Hc, cy*Hc) for cx, cy in centres_H]

    def _arcs():
        out = []
        for cx, cy in centres:
            t = np.linspace(0, 2*np.pi, 240)
            out.append(pv.lines_from_points(np.c_[cx+Rc_*np.cos(t), cy+Rc_*np.sin(t), 0*t]))
        return out

    def _render(out, title, zoom=None):
        pl = pv.Plotter(off_screen=True, window_size=(1300, 480))
        pl.set_background("white")
        pl.add_mesh(pvm, scalars="edot", cmap="Oranges", clim=(clo, chi), log_scale=True,
                    show_edges=False, lighting=False,
                    scalar_bar_args={"title": "strain rate II", "color": "black"})
        for c in _arcs():
            pl.add_mesh(c, color="red", line_width=2.0, lighting=False)
        pl.view_xy()
        if zoom is not None:
            zx, zy, zh = zoom
            pl.camera.parallel_projection = True
            pl.camera.focal_point = (zx, zy, 0); pl.camera.parallel_scale = zh
        pl.add_text(title, font_size=11, color="black")
        pl.screenshot(out); pl.close()
        uw.pprint(0, f"wrote {out}")

    _render(outputPath + "edot_localization.png", "strain-rate localization (full DP)")
    _render(outputPath + "edot_localization_notch.png", "strain-rate localization (notch)",
            zoom=(0.0, -0.72*Hc, 0.30*Hc))

# %% [markdown]
# ### Visualisation
#
# Plot the final strain-rate invariant on the mesh with velocity arrows
# overlaid, then the change in strain rate due to plasticity ($\Delta\dot\varepsilon$).
#
# The strain-rate field (`edot`) is DG-0 so we evaluate it at mesh vertices;
# velocity (`v_soln`) is P2 so we build a separate triangulated mesh through
# all its DOF points for the arrows.
#
# Arrow scaling: normalise velocity to unit magnitude, then set the arrow
# length to a fraction of the domain height.

# %%
if uw.is_notebook():
    import pyvista as pv
    import underworld3.visualisation as vis

    # --- mesh + strain rate ---
    pvmesh = vis.mesh_to_pv_mesh(mesh1)
    pvmesh.point_data["edot"] = vis.scalar_fn_to_pv_points(pvmesh, edot.sym)
    pvmesh.point_data["edot_0"] = vis.scalar_fn_to_pv_points(pvmesh, edot_0.sym)
    pvmesh.point_data["delta_edot"] = pvmesh.point_data["edot"] - pvmesh.point_data["edot_0"]
    pvmesh.point_data["visc"] = vis.scalar_fn_to_pv_points(pvmesh, visc.sym)

    # --- velocity on its own (P2) triangulation ---
    pvmesh_v = vis.meshVariable_to_pv_mesh_object(v_soln)
    pvmesh_v.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh_v, v_soln.sym)

    # Arrow sub-sampling and scaling
    vfreq = max(1, pvmesh_v.n_points // 50)
    V_max = np.sqrt((pvmesh_v.point_data["V"] ** 2).sum(axis=1)).max()
    domain_height = pvmesh.bounds[3] - pvmesh.bounds[2]
    arrow_mag = 0.03 * domain_height / max(V_max, 1e-30)

    # --- Plot 1: Strain rate + velocity arrows ---
    pl = pv.Plotter(window_size=(1000, 400))

    pl.add_mesh(
        pvmesh,
        scalars="edot_0",
        cmap="magma",
        edge_color="Grey",
        show_edges=True,
        show_scalar_bar=True,
        log_scale=True,
        scalar_bar_args={"title": r"Strain rate invariant"},
        opacity=1.0,
    )

    pl.add_arrows(
        pvmesh_v.points[::vfreq],
        pvmesh_v.point_data["V"][::vfreq],
        mag=arrow_mag,
        color="green",
        opacity=0.8,
    )

    pl.show(cpos="xy")

    # --- Plot 2: Delta strain rate (plastic - linear) ---
    pl2 = pv.Plotter(window_size=(1000, 400))

    pl2.add_mesh(
        pvmesh,
        scalars="delta_edot",
        cmap="RdBu_r",
        edge_color="Grey",
        show_edges=True,
        clim=[0,0.0005],
        show_scalar_bar=True,
        scalar_bar_args={"title": r"$\Delta\dot{\varepsilon}$ (plastic − linear)"},
        opacity=1.0,
    )

    pl2.show(cpos="xy")

# %% language="sh"
# python --version

# %%
