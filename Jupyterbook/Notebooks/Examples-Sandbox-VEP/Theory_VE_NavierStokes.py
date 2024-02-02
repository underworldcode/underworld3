# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + \frac [markdown] {"incorrectly_encoded_metadata": "{1}{12} \\mathbf{\\tau^{**}} \\right]"}
# # Viscoelastic Navier-Stokes equation
#
# Here we outline how we combine the numerical NS scheme and the numerical Visco-elastic scheme
#
# ## Navier-Stokes equation
#
# $$
#     \rho\frac{D{\mathbf{u}}}{D t} + \nabla \cdot \boldsymbol{\tau} - \nabla \mathbf{p} = \mathbf{\rho g \hat{\mathbf{z}}}
# $$
#
# The viscous constitutive law connects the stress to gradients of $\mathbf{v}$ as follows:
#
# $$
# \boldsymbol{\tau} = \eta ( \nabla \mathbf{u} + (\nabla \mathbf{u})^T )
# $$
#
# We next write a discrete problem in terms of corresponding variables defined on a mesh.
#
# $$
#     \rho\frac{\mathbf{u}_{[1]} - \mathbf{u}^*}{\Delta t}  = \rho g - \nabla \cdot \mathbf{\tau} + \nabla p
# $$
#
# where $\mathbf{u}^*$ is the value of $\mathbf{u}$ evaluated at upstream point at a time $t - \delta t$.
# Numerically, this is the value on a particle at the previous timestep. This approximation is the forward Euler integration in time for velocity because $\tau$ is defined in terms of the unknowns. $\mathbf{u}_{[1]}$ denotes that solution uses the 1st order Adams-Moulton scheme and higher order updates are well known:
#
# $$
#     \rho\frac{\mathbf{u}_{[2]} - \mathbf{u}^*}{\Delta t}  =\rho g - \nabla \cdot \left[
#                                                                 \frac{1}{2} \boldsymbol{\tau} +
#                                                                 \frac{1}{2} \boldsymbol{\tau^*}
#                                                                    \right]
#                                                                 - \nabla p
# $$
#
# and
#
# $$
#      \rho\frac{\mathbf{u}_{[3]} - \mathbf{u}^*}{\Delta t}
#              = \rho g - \nabla \cdot \left[ \frac{5}{12} \boldsymbol{\tau}
#                                                          - \frac{1}{12} \boldsymbol{\tau^{**}}
#                                                           \right] - \nabla p
# $$
#
# Where $\boldsymbol\tau^*$ and $\boldsymbol\tau^{**}$ are the upstream history values at $t - \Delta t$ and $t - 2\Delta t$ respectively.
# + \frac [markdown] {"incorrectly_encoded_metadata": "{1}{12} \\boldsymbol{\\tau^{* *}} \\right]"}
#
# In the Navier-stokes problem, it is common to write $\boldsymbol\tau=\eta \left(\nabla \mathbf u + (\nabla \mathbf u)^T \right)$ and $\boldsymbol\tau^*=\eta \left(\nabla \mathbf u^* + (\nabla \mathbf u^*)^T \right)$ which ignores  rotation and shearing of the stress during the interval $\Delta T$. This simplifies the implementation because only the velocity history is required, not the history of the stress tensor.
#
#
# ## Viscoelasticity
#
# In viscoelasticity, the elastic part of the deformation is related to the stress rate. If we approach this problem as a perturbation to the viscous Navier-Stokes equation, we first consider the constitutive behaviour
#
# $$
#  \frac{1}{2\mu}\frac{D{\boldsymbol\tau}}{Dt} + \frac{\boldsymbol\tau}{2\eta} = \dot{\boldsymbol\varepsilon}
# $$
#
# A first order difference form for ${D \tau}/{D t}$ then gives
#
# $$
#     \frac{\boldsymbol\tau - \boldsymbol\tau^{*}}{2 \Delta t \mu} + \frac{\boldsymbol\tau}{2 \eta} = \dot{\boldsymbol\varepsilon}
# $$
#
# where $\tau^*$ is the stress history along the characteristics associated with the current computational points. Rearranging to find an expression for the current stress in terms of the strain rate:
#
# $$
#     \boldsymbol\tau = 2 \dot\varepsilon \eta_{\textrm{eff}_{(1)}} + \frac{\eta \boldsymbol\tau^{*}}{\Delta t \mu + \eta}
# $$
#
# where an 'effective viscosity' is introduced, defined as follows:
#
# $$
#     \eta_{\textrm{eff}_{(1)}} = \frac{\Delta t \eta \mu}{\Delta t \mu + \eta}
# $$
#
# Substituting this definition of the stress into the forward-Euler form of the Navier-Stokes discretisation then gives
#
# $$
#     \rho\frac{\mathbf{u}_{[1]} - \mathbf{u}^*}{\Delta t}  = \rho g - \nabla \cdot \left[ 2 \dot{\boldsymbol\varepsilon}\eta_{\textrm{eff}_{(1)}} +  \frac{\eta \boldsymbol\tau^{*}}{\Delta t \mu + \eta}  \right] + \nabla p
# $$
#
# and the 2nd order (Crank-Nicholson) form becomes
#
# $$
#     \rho\frac{\mathbf{u}_{[2]} - \mathbf{u}^*}{\Delta t}  = \rho g - \frac{1}{2} \nabla \cdot \left[ 2 \dot\varepsilon \eta_{\textrm{eff}_{(1)}} + \left[\frac{\eta}{\Delta t \mu + \eta} + 1\right]\tau^*  \right] + \nabla p
# $$
#
# If we use $\tau^{**}$ in the estimate for the stress rate, we have
#
# $$
#     \frac{3 \tau - 4 \tau^{*} + \tau^{**}}{4 \Delta t \mu} + \frac{\tau}{2 \eta}  = \dot\varepsilon
# $$
#
# Giving
#
# $$
#     \boldsymbol\tau = 2 \dot{\boldsymbol\varepsilon} \eta_{\textrm{eff}_{(2)}} + \frac{4 \eta \boldsymbol\tau^{*}}{2 \Delta t \mu + 3 \eta} - \frac{\eta \boldsymbol\tau^{**}}{2 \Delta t \mu + 3 \eta}
# $$
#
# $$
#     \eta_{\textrm{eff}_{(2)}} = \frac{2 \Delta t \eta \mu}{2 \Delta t \mu + 3 \eta}
# $$
#
#
# $$
#     \frac{\mathbf{u}_{[3]} - \mathbf{u}^*}{\Delta t} =   \rho g
#                 - \nabla \cdot \left[ \frac{5 \dot{\boldsymbol\varepsilon} \eta_{\textrm{eff_(2)}}}{6} +
#                  \frac{5 \eta \boldsymbol\tau^{*}}{3 \cdot \left(2 \Delta t \mu + 3 \eta\right)} + \frac{2\boldsymbol\tau^{*}}{3}
#                - \frac{5 \eta \boldsymbol\tau^{**}}{12 \cdot \left(2 \Delta t \mu + 3 \eta\right)} - \frac{\boldsymbol\tau^{**}}{12}
#                                                                     \right] + \nabla p
# $$
#
#
#
# ---
#
# $$
# \nabla\cdot\left[ \color{blue}{ \boldsymbol{\tau} - p \boldsymbol{I}   }   \right] = \color{green}{\frac{D \boldsymbol{u}}{Dt}} - \rho g
# $$
#
# -
# $$
# \frac{5 \dot\varepsilon \eta_\textrm{eff}}{6} + \frac{5 \eta \tau^{*}}{3 \cdot \left(2 \Delta\,\!t \mu + 3 \eta\right)} - \frac{5 \eta \tau^{**}}{12 \cdot \left(2 \Delta\,\!t \mu + 3 \eta\right)} + \frac{2 \tau^{*}}{3} - \frac{\tau^{**}}{12}
# $$

# ## Mock up ... BDF / Adams-Moulton coefficients
#
# dot_f term will need BDf coefficients (https://en.wikipedia.org/wiki/Backward_differentiation_formula)
#
# Flux history for Adams Moulton
# Substitute for present stress in ADM
#
#
#
#
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# + \frac {"incorrectly_encoded_metadata": "{8}{12} \\mathbf{\\tau^*}"}
import os


# + \frac {"incorrectly_encoded_metadata": "{2 \\tau^{*}}{3}"}
os.path.join("", "test")


# +

# Symbolic: sympy + uw3

import os

os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
import underworld3 as uw
import numpy as np
import sympy
import pyvista as pv
import vtk

from underworld3 import timing

resolution = uw.options.getReal("model_resolution", default=0.033)
mu = uw.options.getInt("mu", default=0.5)
maxsteps = uw.options.getInt("max_steps", default=500)


# +
mesh1 = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.5, -0.5),
    maxCoords=(+1.5, +0.5),
    cellSize=resolution,
)

x, y = mesh1.X

# +
U = uw.discretisation.MeshVariable(
    "U", mesh1, mesh1.dim, vtype=uw.VarType.VECTOR, degree=2
)
P = uw.discretisation.MeshVariable(
    "P", mesh1, 1, vtype=uw.VarType.SCALAR, degree=1, continuous=True
)
T = uw.discretisation.MeshVariable("T", mesh1, 1, vtype=uw.VarType.SCALAR, degree=3)

# Nodal values of deviatoric stress (symmetric tensor)

work = uw.discretisation.MeshVariable(
    "W", mesh1, 1, vtype=uw.VarType.SCALAR, degree=2, continuous=False
)
St = uw.discretisation.MeshVariable(
    r"Stress",
    mesh1,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    degree=2,
    continuous=False,
    varsymbol=r"{\tau}",
)

# May need these
Edot_inv_II = uw.discretisation.MeshVariable(
    "eps_II",
    mesh1,
    1,
    vtype=uw.VarType.SCALAR,
    degree=2,
    varsymbol=r"{|\dot\varepsilon|}",
)
St_inv_II = uw.discretisation.MeshVariable(
    "tau_II", mesh1, 1, vtype=uw.VarType.SCALAR, degree=2, varsymbol=r"{|\tau|}"
)

# +
swarm = uw.swarm.Swarm(mesh=mesh1, recycle_rate=5)

material = uw.swarm.SwarmVariable(
    "M",
    swarm,
    size=1,
    vtype=uw.VarType.SCALAR,
    proxy_continuous=True,
    proxy_degree=1,
    dtype=int,
)

strain_inv_II = uw.swarm.SwarmVariable(
    "Strain",
    swarm,
    size=1,
    vtype=uw.VarType.SCALAR,
    proxy_continuous=True,
    proxy_degree=2,
    varsymbol=r"{|\varepsilon|}",
    dtype=float,
)

stress_star = uw.swarm.SwarmVariable(
    r"stress_dt",
    swarm,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    proxy_continuous=True,
    proxy_degree=2,
    varsymbol=r"{\tau^{*}_{p}}",
)

stress_star_star = uw.swarm.SwarmVariable(
    r"stress_2dt",
    swarm,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    proxy_continuous=True,
    proxy_degree=2,
    varsymbol=r"{\tau^{**}_{p}}",
)

swarm.populate(fill_param=2)

# +
stokes = uw.systems.Stokes(
    mesh1,
    velocityField=U,
    pressureField=P,
    verbose=False,
    solver_name="stokes",
)

stokes
# -
uw.systems.Stokes


# +
from sympy import UnevaluatedExpr

st = sympy.UnevaluatedExpr(stokes.constitutive_model.viscosity) * sympy.UnevaluatedExpr(
    stokes.strainrate
)
st

# +
eta_0 = sympy.sympify(10) ** -6
C_0 = sympy.log(10**6)


stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = sympy.symbols(r"\eta")
stokes.constitutive_model.Parameters.shear_modulus = sympy.symbols(r"\mu")
stokes.constitutive_model.Parameters.stress_star = stress_star.sym
stokes.constitutive_model.Parameters.dt_elastic = sympy.symbols(
    r"\Delta\ t"
)  # sympy.sympify(1) / 10
stokes.constitutive_model.Parameters.strainrate_inv_II = stokes.Unknowns.Einv2
stokes.constitutive_model.Parameters.strainrate_inv_II_min = 0
stokes.constitutive_model
# -

stokes.constitutive_model.Parameters.viscosity

# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

# Constant visc

stokes.penalty = 0

stokes.tolerance = 1.0e-4

# Velocity boundary conditions

stokes.add_dirichlet_bc((0.0, 0.0), "Inclusion", (0, 1))
stokes.add_dirichlet_bc((1.0, 0.0), "Top", (0, 1))
stokes.add_dirichlet_bc((-1.0, 0.0), "Bottom", (0, 1))
stokes.add_dirichlet_bc((0.0), "Left", (1))
stokes.add_dirichlet_bc((0.0), "Right", (1))

# -

stokes._setup_problem_description()

t0 = St.sym
t1 = stress_star.sym
t0 + t1

# +
t2 = t0 + t1
t2

t2_II_sq = (t2**2).trace() / 2
t2_II = sympy.sqrt(t2_II_sq)

overshoot = sympy.simplify(sympy.sqrt(t2_II_sq) / tauY)
overshoot
# -

t2_II_sq

stokes.constitutive_model.flux(stokes.strainrate)

# Crank-Nicholson timestep - Jacobians
(stokes.constitutive_model.flux(stokes.strainrate) / 2 + stress_star.sym / 2).diff(
    stokes._u.sym[0]
)

(stokes.stress_deviator_1d / 2 + stress_star.sym_1d / 2)[0]

# RHS
stokes._u_f0

# Jacobian terms
stokes._u_f1[1, 1].diff(stokes._L[1, 1])

# +
## Jacobians (e.g. stress rate derivatives with respect to strain rate tensor)

stokes._uu_G3

# +
## And now, second order terms

stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
    mesh1.dim
)
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0 * sympy.exp(
    -C_0 * T.sym[0]
)
stokes.constitutive_model.Parameters.shear_modulus = 100
stokes.constitutive_model.Parameters.stress_star = stress_star.sym
stokes.constitutive_model.Parameters.stress_star_star = stress_star_star.sym
stokes.constitutive_model.Parameters.dt_elastic = sympy.sympify(1) / 10
stokes.constitutive_model.Parameters.strainrate_inv_II = stokes._Einv2
stokes.constitutive_model.Parameters.strainrate_inv_II_min = 0

stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity

stokes.constitutive_model


# -

stokes.stress_deviator_1d[2]

0 / 0

stokes.solve()


# +
# Adams / Bashforth & Adams Moulton ...

s = sympy.Symbol(r"\tau")
s1 = sympy.Symbol(r"\tau^*")
s2 = sympy.Symbol(r"\tau^**")
dt = sympy.Symbol(r"\Delta\,\!t")
mu = sympy.Symbol(r"\mu")
eta = sympy.Symbol(r"\eta")
eta_p = sympy.Symbol(r"\eta_p")
eta_eff = sympy.Symbol(r"\eta_\textrm{eff}")
edot = sympy.Symbol(r"\dot\varepsilon")
tr = sympy.Symbol(r"t_r")

# Stress history

# 1st order difference expression for stress rate
sdot1 = (s - s1) / dt

# 2nd order difference for stress rate
sdot2 = (3 * s - 4 * s1 + s2) / (2 * dt)


# +
display(sdot1)
display(sdot1 / (2 * mu) + s / (2 * eta))  # + s / (2 * eta_p))
print(sympy.latex(sdot1 / (2 * mu) + s / (2 * eta) + s / (2 * eta_p)))
Seq1 = sympy.Equality(
    sympy.simplify(sdot1 / (2 * mu) + s / (2 * eta) + 0 * s / (2 * eta_p)), edot
)
display(Seq1)

# solve this for an expression in terms of the present stress, $\tau$
a = sympy.simplify(sympy.solve(Seq1, s)[0]).expand()
display(a)

# b = sympy.simplify(sympy.solve(Seq1, eta_p)[0]).expand()
# display(b)
# -

S = a.subs(edot, stokes.strainrate).subs(s1, stress_star.sym)
SII = sympy.simplify((S**2).trace())

# +
## yielding case

tauY = sympy.symbols(r"\tau_y")

# -


etaY2 = tauY / (2 * edot + (s1 - tauY) / (mu * dt))
etaY2

sympy.simplify(etaY - etaY2)

stokes.constitutive_model.Parameters.ve_effective_viscosity

sympy.simplify(a / 2 + s1 / 2).expand().collect(s1)

# +
# Identify effective viscosity

eta_eff_1 = sympy.simplify(eta * mu * dt / (mu * dt + eta))
display(eta_eff_1)
print(sympy.latex(eta_eff_1))

# rename this in the equations
b = a.subs(eta_eff_1, eta_eff)
display(b)
print(sympy.latex(b))

## An equivalent form for this is

c = 2 * eta_eff * edot + (s1 * tr / (dt + tr))
display(c)

## Validate that
sympy.simplify(b - c.subs(tr, eta / mu))

(b / 2 + s1 / 2)


# +
# Now we can try a 2nd order

display(sdot2)
print(sympy.latex(sdot2))
display(sdot2 / (2 * mu) + s / (2 * eta))
print(sympy.latex(sdot2 / (2 * mu) + s / (2 * eta)))
Seq2 = sympy.Equality(sympy.simplify(sdot2 / (2 * mu) + s / (2 * eta)), edot)
display(Seq2)
sympy.simplify(sympy.solve(Seq2, s)[0])

# +
eta_eff_2 = sympy.simplify(2 * eta * mu * dt / (2 * mu * dt + 3 * eta))
display(eta_eff_2)
print(sympy.latex(eta_eff_2))


sympy.simplify(2 * eta * sympy.solve(Seq2, s)[0] / (2 * eta_eff_2))

# solve this for an expression in terms of the present stress, $\tau$
a2 = sympy.simplify(sympy.solve(Seq2, s)[0]).expand()
display(a2)

# Identify effective viscosity


# rename this in the equations
b2 = a2.subs(eta_eff_2, eta_eff)
display(b2)
print(sympy.latex(b2))

## And this is what happens in Adams-Moulton 3rd order

display(5 * b2 / 12 + 2 * s1 / 3 - s2 / 12)
print(sympy.latex(5 * b2 / 12 + 2 * s1 / 3 - s2 / 12))


# -

a = sympy.simplify(2 * eta * sympy.solve(Seq2, s)[0] / (2 * eta_eff_2))
tau_2 = a.expand().subs(eta / mu, tr)

a


0 / 0


stokes.constitutive_model.Parameters.yield_stress.subs(
    ((strain.sym[0], 0.25), (y, 0.0))
)

stokes.constitutive_model.Parameters.viscosity


def return_points_to_domain(coords):
    new_coords = coords.copy()
    new_coords[:, 0] = (coords[:, 0] + 1.5) % 3 - 1.5
    return new_coords


ts = 0

# +
delta_t = stokes.estimate_dt()

expt_name = f"output/shear_band_sw_nonp_{mu}"

for step in range(0, 10):
    stokes.solve(zero_init_guess=False)

    nodal_strain_rate_inv2.uw_function = sympy.Max(
        0.0,
        stokes._Einv2
        - 0.5
        * stokes.constitutive_model.Parameters.yield_stress
        / stokes.constitutive_model.Parameters.bg_viscosity,
    )
    nodal_strain_rate_inv2.solve()

    with mesh1.access(strain_rate_inv2_p):
        strain_rate_inv2_p.data[...] = strain_rate_inv2.data.copy()

    nodal_strain_rate_inv2.uw_function = stokes._Einv2
    nodal_strain_rate_inv2.solve()

    with swarm.access(strain), mesh1.access():
        XX = swarm.particle_coordinates.data[:, 0]
        YY = swarm.particle_coordinates.data[:, 1]
        mask = (2 * XX / 3) ** 4  # * 1.0 - (YY * 2)**8
        strain.data[:, 0] += (
            delta_t * mask * strain_rate_inv2_p.rbf_interpolate(swarm.data)[:, 0]
            - 0.1 * delta_t
        )
        strain_dat = (
            delta_t * mask * strain_rate_inv2_p.rbf_interpolate(swarm.data)[:, 0]
        )
        print(
            f"dStrain / dt = {delta_t * (mask * strain_rate_inv2_p.rbf_interpolate(swarm.data)[:,0]).mean()}, {delta_t}"
        )

    mesh1.write_timestep_xdmf(
        f"{expt_name}",
        meshUpdates=False,
        meshVars=[p_soln, v_soln, strain_rate_inv2_p],
        swarmVars=[strain],
        index=ts,
    )

    swarm.save(f"{expt_name}.swarm.{ts}.h5")
    strain.save(f"{expt_name}.strain.{ts}.h5")

    # Update the swarm locations
    swarm.advection(v_soln.sym, delta_t=delta_t, restore_points_to_domain_func=None)

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))

    ts += 1



# +
nodal_visc_calc.uw_function = sympy.log(stokes.constitutive_model.Parameters.viscosity)
nodal_visc_calc.solve()

yield_stress_calc.uw_function = stokes.constitutive_model.Parameters.yield_stress
yield_stress_calc.solve()

nodal_tau_inv2.uw_function = (
    2 * stokes.constitutive_model.Parameters.viscosity * stokes._Einv2
)
nodal_tau_inv2.solve()

# +
# check it - NOTE - for the periodic mesh, points which have crossed the coordinate sheet are plotted somewhere
# unexpected. This is a limitation we are stuck with for the moment.

if uw.mpi.size == 1:
    
    import underworld3.visualisation as vis # use tools from here
    
    mesh1.vtk("tmp_shear_inclusion.vtk")
    pvmesh = pv.read("tmp_shear_inclusion.vtk")

    pvpoints = pvmesh.points[:, 0:2]
    usol = v_soln.rbf_interpolate(pvpoints)

    pvmesh.point_data["P"] = p_soln.rbf_interpolate(pvpoints)
    pvmesh.point_data["Edot"] = strain_rate_inv2.rbf_interpolate(pvpoints)
    pvmesh.point_data["Visc"] = np.exp(node_viscosity.rbf_interpolate(pvpoints))
    pvmesh.point_data["Edotp"] = strain_rate_inv2_p.rbf_interpolate(pvpoints)
    pvmesh.point_data["Strs"] = dev_stress_inv2.rbf_interpolate(pvpoints)
    pvmesh.point_data["StrY"] = yield_stress.rbf_interpolate(pvpoints)
    pvmesh.point_data["dStrY"] = (
        pvmesh.point_data["StrY"]
        - 2 * pvmesh.point_data["Visc"] * pvmesh.point_data["Edot"]
    )
    pvmesh.point_data["Mat"] = material.rbf_interpolate(pvpoints)
    pvmesh.point_data["Strn"] = strain._meshVar.rbf_interpolate(pvpoints)

    # Velocity arrows

    v_vectors = np.zeros_like(pvmesh.points)
    v_vectors[:, 0:2] = v_soln.rbf_interpolate(pvpoints)

    # Points (swarm)

    with swarm.access():
        plot_points = np.where(strain.data > 0.0001)
        strain_data = strain.data.copy()

        points = np.zeros((swarm.data[plot_points].shape[0], 3))
        points[:, 0] = swarm.data[plot_points[0], 0]
        points[:, 1] = swarm.data[plot_points[0], 1]
        point_cloud = pv.PolyData(points)
        point_cloud.point_data["strain"] = strain.data[plot_points]

    pl = pv.Plotter(window_size=(500, 500))

    # pl.add_arrows(pvmesh.points, v_vectors, mag=0.1, opacity=0.75)
    # pl.camera_position = "xy"

    pl.add_mesh(
        pvmesh,
        cmap="Blues",
        edge_color="Grey",
        show_edges=True,
        # clim=[-1.0,1.0],
        scalars="Edotp",
        use_transparency=False,
        opacity=0.5,
    )

    pl.add_points(
        point_cloud,
        colormap="Oranges",
        scalars="strain",
        point_size=10.0,
        opacity=0.0,
        # clim=[0.0,0.2],
    )

    pl.camera.SetPosition(0.0, 0.0, 3.0)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)

    pl.show()
# -

strain_dat.max()

# +
alph = sympy.symbols(r"\alpha_:10")
alph[9]


fn = alph[0] * U[0].sym + alph[1] * U[1].sym
# -

fn.diff(alph[0])

with swarm.access():
    print(strain.data.max())

strain_rate_inv2_p.rbf_interpolate(mesh1.data).max()


# ##

mesh1._search_lengths
