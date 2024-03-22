# ---
# jupyter:
#   jupytext:
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

# # Advection-diffusion of a hot pipe
#
# - Using the adv_diff solver.
# - Advection of the rectangular pulse vertically as it also diffuses. The velocity is 0.05 and has a diffusivity value of 1, 0.1 or 0.01
# - Benchmark comparison between 1D analytical solution and 2D UW numerical model.
#
# ![](Figures/AdvectionTestFigure.png)
#
# *Figure: typical results from this test. Quad mesh v. unstructured triangles with equivalent
# resolution. $\kappa=1$, $\mathbf{v}=(1000,0)$, $t_0 = 0.0001$, $\delta t = 0.0003$. The error looks significantly larger with triangles but you can see that it is dominated by a relatively small* phase error *where the speed of propagation is slightly different from the analytic case.*
#
#
# ## How to test advection or diffusion only
# - Set velocity to 0 to test diffusion only.
# - Set diffusivity (k) to 0 to test advection only.
#
#
# ## Analytic solution
#
# $$
# T(x,t) =
# \frac{\operatorname{erf}{\left(\frac{- \mathrm{x} + v \left(t + {t_0}\right) + \frac{{\delta}}{2} + {x_0}}{2 \sqrt{\kappa \left(t + {t_0}\right)}} \right)}}{2} + \frac{\operatorname{erf}{\left(\frac{\mathrm{x} - v \left(t + {t_0}\right) + \frac{{\delta}}{2} - {x_0}}{2 \sqrt{\kappa \left(t + {t_0}\right)}} \right)}}{2}
# $$
#
# Where $x,y$ describe the coordinate frame, $v$ is the horizontal velocity that advects the temperature, $\delta$ is the width of the temperature anomaly, $x_0$ is the initial midpoint of the temperature anomaly. $\kappa$ is the thermal diffusivity, $t_0$ is the time at which we turn on the horizontal velocity. 
#
# Note: this solution is derived from the diffusion of a step which is applied to the leading and trailing edges of the block. The solution is valid while the diffusion fronts from each interface remain independent of each other. (This is ill-defined from the problem, but the most obvious test is to look a the time that the block temperature drops below 1 to the tolerance of the solver).
#

import nest_asyncio
nest_asyncio.apply()

# +
import underworld3 as uw
import numpy as np
import sympy
import math
import os

from scipy import special
# -

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt


# ### Set up variables of the model

# +
import sys

init_t = 0.0001
dt   = 0.0003
velocity = 1000.
centre = 0.2
width = 0.2


### min and max temps
tmin = 0.  # temp min
tmax = 1.0 # temp max

# I think we should get into the habit of doing this consistently with the PETSc interface

res = uw.options.getReal("model_resolution", default=16)
kappa = uw.options.getInt("kappa", default=1.0)
Tdegree = uw.options.getInt("Tdeg", default=3)
Vdegree = uw.options.getInt("Vdeg", default=2)
simplex = uw.options.getBool("simplex", default=True)



# Tdegree = int(sys.argv[1])
# Vdegree = int(sys.argv[2])
# kappa = float(sys.argv[3]) # 1, 0.1, 0.01 # diffusive constant
# res = int(sys.argv[4])
# simplex = sys.argv[5].lower()

# +
outputPath = f'./output/adv_diff-hot_pipe/'

if uw.mpi.rank == 0:
    # checking if the directory
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
# -

# ### Set up the mesh

xmin, xmax = 0, 1
ymin, ymax = 0, 0.2

## Quads
if simplex == True:
    mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=(ymax-ymin)/res, regular=False, qdegree=max(Tdegree, Vdegree) )
else:
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(int(res)*5, int(res)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax), qdegree=max(Tdegree, Vdegree),
    )


# ### Create mesh variables
# To be used in the solver

# +
x,y = mesh.X

x0 = sympy.symbols(r"{x_0}")
t0 = sympy.symbols(r"{t_0}")
delta = sympy.symbols(r"{\delta}")
ks = sympy.symbols(r"\kappa")
ts = sympy.symbols("t")
vs = sympy.symbols("v")

Ts =  ( sympy.erf( (x0 + delta/2  - x+(vs*(ts+t0)))  / (2*sympy.sqrt(ks*(ts+t0)))) + sympy.erf( (-x0 + delta/2 + x-((ts+t0)*vs))  / (2*sympy.sqrt(ks*(ts+t0)))) ) / 2
Ts


# +
def build_analytic_fn_at_t(time): 
    fn = Ts.subs({vs:velocity, ts:time, ks:kappa, delta:width, x0:centre, t0:init_t})
    return fn


Ts0 = build_analytic_fn_at_t(time=0.0)
TsVKT = build_analytic_fn_at_t(time=dt)

# +
# Create the mesh var
T       = uw.discretisation.MeshVariable("T", mesh, 1, degree=Tdegree)

# This is the velocity field

v = sympy.Matrix([velocity, 0])
# -

# #### Create the advDiff solver

adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh,
    u_Field=T,
    V_fn=v,
    solver_name="adv_diff",
)

# ### Set up properties of the adv_diff solver
# - Constitutive model (Diffusivity)
# - Boundary conditions
# - Internal velocity
# - Initial temperature distribution 

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = kappa

adv_diff.add_dirichlet_bc(tmin, "Left")
adv_diff.add_dirichlet_bc(tmin, "Right")


adv_diff.estimate_dt(v_factor=10)
steps = int(dt // (10*adv_diff.estimate_dt()))

# ### Create points to sample the UW results

# +
### get the initial temp profile

with mesh.access(T):
    T.data[:,0] = uw.function.evalf(Ts0, T.coords)

# -

step = 0
model_time = 0.0

# +
adv_diff.petsc_options["snes_monitor_short"] = None

# if uw.mpi.size == 1:
#     adv_diff.petsc_options['pc_type'] = 'lu'
    
# -

while model_time < dt:    
    adv_diff.solve(timestep=dt/steps, zero_init_guess=False)
    model_time += dt/steps
    step += 1
    print(f"Timestep: {step}, model time {model_time}")


# %%
if uw.mpi.size == 1:

        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(mesh)
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, sympy.Matrix([velocity, 0]).T)
        pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, T.sym)
        pvmesh.point_data["Ta"] = vis.scalar_fn_to_pv_points(pvmesh, Ts0)
        pvmesh.point_data["dT"] = pvmesh.point_data["T"] - pvmesh.point_data["Ta"]

        T_points = vis.meshVariable_to_pv_cloud(T)
        T_points.point_data["T"] = vis.scalar_fn_to_pv_points(T_points, T.sym)
        T_points.point_data["Ta"] = vis.scalar_fn_to_pv_points(T_points, TsVKT)
        T_points.point_data["T0"] = vis.scalar_fn_to_pv_points(T_points, Ts0)
        T_points.point_data["Tp"] = (T_points.point_data["T0"] + T_points.point_data["Ta"])/2
        T_points.point_data["dT"] = T_points.point_data["T"] - T_points.point_data["Ta"]

        pvmesh2 = vis.mesh_to_pv_mesh(mesh)
        pvmesh2.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh2, T.sym)
        pvmesh2.point_data["T0"] = vis.scalar_fn_to_pv_points(pvmesh2, Ts0)
        pvmesh2.points[:,1] += 0.3        
    
        pvmesh3 = vis.mesh_to_pv_mesh(mesh)
        pvmesh3.point_data["Ta"] = vis.scalar_fn_to_pv_points(pvmesh2, TsVKT)
        pvmesh3.points[:,1] -= 0.3


        pl = pv.Plotter()

        pl.add_mesh(
            pvmesh2,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=True,
            scalars="T",
            use_transparency=False,
            show_scalar_bar=False,
            opacity=1,
        )


        pl.add_mesh(
                    
            pvmesh3,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=True,
            scalars="Ta",
            use_transparency=False,
            show_scalar_bar=False,
            opacity=1,
        )

        pl.add_points(T_points, color="White",
                      scalars="dT", cmap="coolwarm",
                      point_size=5.0, opacity=0.5)


        pl.add_arrows(pvmesh.points, pvmesh.point_data["V"], mag=0.00003, opacity=0.5, show_scalar_bar=False)

        # pl.add_points(pdata)

        pl.show(cpos="xy")

        # return vsol

T_points.point_data["dT"].max()

T_points.point_data["Ta"].max()

#
#
