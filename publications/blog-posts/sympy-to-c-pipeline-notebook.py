# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How Underworld3 Turns SymPy into C
#
# This notebook accompanies the blog post of the same name.
# It walks through the pipeline from user-facing SymPy expressions
# to compiled C callbacks registered with PETSc.

# %% [markdown]
# ## At the User Level
#
# We set up a Stokes flow problem with a temperature-dependent
# (Frank-Kamenetskii) viscosity. This is a standard thermal convection
# setup — simple enough to follow, complex enough to show the pipeline.

# %%
import underworld3 as uw
import sympy

# %% [markdown]
# ### The mesh

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=0.1,
    qdegree=3,
)

# %% [markdown]
# ### Parameters as UWexpressions
#
# Each parameter is a symbolic name with a concrete value.
# These behave like SymPy symbols in expressions but carry their values
# for the compiler to extract later. (Units can be added via `uw.quantity()`
# but we keep things dimensionless here to focus on the pipeline.)

# %%
# Parameters — symbolic names with concrete values (dimensionless here)
eta_0   = uw.expression(r"\eta_0", 1.0)    # reference viscosity
gamma   = uw.expression(r"\gamma", 13.8)   # FK sensitivity
Ra      = uw.expression("Ra", 1e6)         # Rayleigh number

# %% [markdown]
# ### Mesh variables
#
# Velocity, pressure, and temperature are mesh variables — fields
# defined at every point in the mesh. We create them explicitly
# so that they have readable names in the symbolic output.

# %%
v = uw.discretisation.MeshVariable(r"u", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable(r"p", mesh, 1, degree=1, continuous=True)
T = uw.discretisation.MeshVariable(r"T", mesh, 1, degree=2)

# %% [markdown]
# ### The Stokes solver
#
# We pass our velocity and pressure variables to the solver.

# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

# %% [markdown]
# ### Building the constitutive law
#
# Frank-Kamenetskii viscosity: $\eta = \eta_0 \exp(-\gamma T)$
#
# This is ordinary SymPy arithmetic combining UWexpressions (parameters)
# and a MeshVariable (spatial field). Nothing is evaluated yet.

# %%
viscosity_fn = eta_0 * sympy.exp(-gamma * T)
viscosity_fn

# %% [markdown]
# ### Assigning the constitutive model and body force

# %%
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_fn
stokes.bodyforce = sympy.Matrix([0, -Ra * T[0, 0]])

# %% [markdown]
# ### Boundary conditions — free-slip on all walls

# %%
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

# %% [markdown]
# ## Stage 1: The Strong Form Template
#
# The Stokes equation in strong form is:
#
# $$-\nabla \cdot \underbrace{\boldsymbol{\sigma}}_{\mathbf{F_1}}
#   - \underbrace{\mathbf{f}}_{F_0} = 0$$
#
# The solver decomposes this into $\mathbf{F}_1$ (the stress flux — everything
# under the divergence) and $F_0$ (the body force — everything else).
# We can inspect each piece separately.

# %% [markdown]
# ### The body force ($F_0$)

# %%
stokes.bodyforce

# %% [markdown]
# ### The constitutive stress ($\mathbf{F}_1$)
#
# The constitutive model defines the stress as a function of the strain rate.
# Our FK viscosity appears inside it.

# %%
stokes.constitutive_model.flux

# %% [markdown]
# ### The viscosity parameter inside the model

# %%
stokes.constitutive_model.Parameters.shear_viscosity_0

# %% [markdown]
# These are all live SymPy expressions. The solver has not evaluated anything —
# it stores them symbolically and defers compilation until `solve()` is called.
# Note that the constitutive flux contains the strain rate of the unknown
# velocity field, and the FK viscosity with its dependence on temperature.

# %% [markdown]
# ## Stage 2: Automatic Jacobians
#
# The solver differentiates $F_0$ and $\mathbf{F}_1$ with respect to the unknowns
# to produce four Jacobian blocks $G_0$–$G_3$ for PETSc's Newton solver.
#
# We can see what this means by differentiating the viscosity ourselves.
# The viscosity depends on $T$, so any Jacobian term involving
# $\partial\mathbf{F}_1/\partial T$ will contain this derivative:

# %%
sympy.diff(viscosity_fn, T)

# %% [markdown]
# SymPy computes exact symbolic derivatives. No finite-difference
# approximations, no hand-coding.

# %% [markdown]
# ## The Symbolic Wrappers
#
# Let's look at what the expressions are actually made of.

# %% [markdown]
# ### UWexpression: symbol with a value

# %%
# eta_0 is a SymPy symbol...
print(f"Type: {type(eta_0)}")
print(f"In an expression: {2 * eta_0 * gamma}")

# ...but it carries a value
print(f"Stored value: {eta_0.value}")

# %%
# Display it — renders as LaTeX in the notebook
eta_0

# %% [markdown]
# ### MeshVariable symbols: spatial field data

# %%
# The temperature variable's symbolic face
T.sym

# %%
# The velocity variable — a vector of symbols
v.sym

# %% [markdown]
# ### Coordinates: also symbolic

# %%
# Mesh coordinates as SymPy symbols
mesh.X

# %%
# Use them in expressions — depth-dependent body force
depth_force = Ra * mesh.X[1]
depth_force

# %% [markdown]
# ### UWexpression values

# %%
# Each expression carries its current value
print(f"eta_0 = {eta_0.value}")
print(f"gamma = {gamma.value}")
print(f"Ra    = {Ra.value}")

# %% [markdown]
# ## Summary
#
# Everything you see in this notebook — the parameters, the viscosity law,
# the $F_0$/$\mathbf{F}_1$ terms, the constitutive flux — is simultaneously:
#
# - **Human-readable mathematics** (rendered in the notebook)
# - **A complete specification for C code generation** (the JIT compiler reads it)
# - **Symbolically differentiable** (for automatic Jacobians)
#
# The compiler just reads what was there all along.
