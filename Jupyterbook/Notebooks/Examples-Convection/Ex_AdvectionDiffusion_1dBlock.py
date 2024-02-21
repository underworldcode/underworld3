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
# ## How to test advection or diffusion only
# - Set velocity to 0 to test diffusion only.
# - Set diffusivity (k) to 0 to test advection only.
#
#
# ## Analytic solution
#



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
if simplex == 'true':
    mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=1/res, regular=False, qdegree=max(Tdegree, Vdegree) )
else:
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(int(res)*5, int(res)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax), qdegree=max(Tdegree, Vdegree),
    )


# ### triangles


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


# +
adv_diff.estimate_dt(v_factor=10)
steps = int(dt // (10*adv_diff.estimate_dt()))


# -

# ### Create points to sample the UW results

# +
### y coords to sample
sample_x = np.linspace(xmin, xmax, 201) ### get the x coords from the mesh

### x coords to sample
sample_y = np.zeros_like(sample_x) 

sample_points = np.empty((sample_x.shape[0], 2))
sample_points[:, 0] = sample_x
sample_points[:, 1] = sample_y + 0.5 ###  across centre of box

# +
### get the initial temp profile
T_orig = uw.function.evalf(T.sym, sample_points)

with mesh.access(T):
    T.data[:,0] = uw.function.evalf(Ts0, T.coords)

# -

step = 0
model_time = 0.0

# +
# adv_diff.petsc_options["snes_rtol"] = 1.0e-9
# adv_diff.petsc_options["ksp_rtol"] = 1.0e-9
# adv_diff.petsc_options["snes_max_it"] = 100

adv_diff.petsc_options["snes_monitor_short"] = None

if uw.mpi.size == 1:
    adv_diff.petsc_options['pc_type'] = 'lu'
    

# +
while model_time < dt:    
    adv_diff.solve(timestep=dt/steps, zero_init_guess=False)
    model_time += dt/steps
    step += 1
    print(f"Timestep: {step}")
    
    


# +
#     start = track_time()
#     ### print some stuff
#     if uw.mpi.rank == 0:
#         print(f"Step: {str(step).rjust(3)}, time: {model_time:6.5f}\n")

    
    
#     dt = adv_diff.estimate_dt()

#     ### finish at the set final time
#     if model_time + dt > final_time:
#         dt = final_time - model_time

#     # print(f'dt: {dt}\n')

    
#     ### diffuse through underworld
#     adv_diff.solve(timestep=dt)

#     step += 1
#     model_time += dt

#     end = track_time()

#     solve_time = end - start
#     if uw.mpi.rank == 0:
#         print(f'solve time: {solve_time}\n')
# -

model_time

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

        pl = pv.Plotter()

        pl.add_mesh(pvmesh, "Black", "wireframe")

        # pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)

        # pl.add_mesh(
        #     pvmesh,
        #     cmap="coolwarm",
        #     edge_color="Black",
        #     show_edges=True,
        #     scalars="Ta",
        #     use_transparency=False,
        #     opacity=0.01,
        # )


        pl.add_points(T_points, color="White",
                      scalars="dT", cmap="coolwarm",
                      point_size=5.0, opacity=0.5)



        pl.add_arrows(pvmesh.points, pvmesh.point_data["V"], mag=0.00003, opacity=0.5)

        # pl.add_points(pdata)

        pl.show(cpos="xy")

        # return vsol

0/0

T_points.point_data["dT"].max()

T_points.point_data["Ta"].max()

# ### Check the results
#
# Compare numerical and analytic results.

#
#

with mesh.access(T_a):
    x = T_a.coords[:,0]
    T_a.data[:,0] = 0.5 * ( special.erf( (final_x  - x+(u*t))  / (2*np.sqrt(kappa*t))) + special.erf( (-start_x + x-(u*t))  / (2*np.sqrt(kappa*t))) )


def L1_norm_integration(solver, analytical_sol):
    numeric_solution   = solver.u.sym[0]
    analytic_solution  = analytical_sol.sym[0]

    I = uw.maths.Integral(solver.mesh, sympy.Abs(numeric_solution-analytic_solution))

    return I


# +
### Create columns of file if it doesn't exist
try:
    with open(f'{outputPath}AdvDiff_kappa={kappa}_Tdeg={Tdegree}_Vdeg={Vdegree}_simplex={mesh.isSimplex}.txt', 'x') as f:
        f.write(f'Tdegree,Vdegree,res,cell size,L1_norm')
except:
    pass



### Append the data
with open(f'{outputPath}AdvDiff_kappa={kappa}_Tdeg={Tdegree}_Vdeg={Vdegree}_simplex={mesh.isSimplex}.txt', 'a') as f:
    f.write(f'\n{Tdegree},{Vdegree},{res},{mesh.get_min_radius()},{L1_norm_integration(adv_diff, T_a).evaluate()}')
# -



# +
# analytical_solution = 0.5 * ( special.erf( (final_x  - x+(u*t))  / (2*np.sqrt(kappa*t))) + special.erf( (-start_x + x-(u*t))  / (2*np.sqrt(kappa*t))) )

# +
# with mesh.access(T, l1_norm):
    
#     x_coords = T.coords[:,0]
#     y_coords = T.coords[:,1]
    
#     analytical_solution = 0.5 * ( special.erf( (final_x  - x_coords+(u*t))  / (2*np.sqrt(kappa*t))) + special.erf( (-start_x + x_coords-(u*t))  / (2*np.sqrt(kappa*t))) )

#     numerical_solution  = np.copy(T.data[:,0])

#     l1_calc = np.abs(numerical_solution - analytical_solution)

#     l1_norm.data[:,0] = l1_calc

# +
# if uw.mpi.rank == 0:
#     ### save coords and l1 norm
#     np.savez(f'{outputPath}l1_norm-D={kappa}-Tdeg={Tdegree}', x_coords, y_coords, l1_calc)

# +
# if uw.mpi.size == 1:
#     plt.scatter(x_coords, y_coords, c=l1_calc)

# +
# x = sample_points[:, 0]
# u = velocity
# t = final_time
# start_x = 0.5 - (pipe_thickness/2)
# final_x = 0.5 + (pipe_thickness/2)


# T_UW_profile = uw.function.evalf(T.sym[0], sample_points)

# T_a_profile = 0.5 * ( special.erf( (final_x  - x+(u*t))  / (2*np.sqrt(kappa*t)))
# + special.erf( (-start_x + x-(u*t))  / (2*np.sqrt(kappa*t))) )
 


# +
# if uw.mpi.size == 1:
#     plt.plot(x, T_UW_profile, ls='-')
#     plt.plot(x, T_a_profile, ls=':', c='r')

# +
# if uw.mpi.rank == 0:
#     ### save the analytical and numerical profile
#     np.savez(f'{outputPath}1D_profile-D={kappa}-Tdeg={Tdegree}', x, T_a_profile, T_UW_profile)




# +
# """compare analytical and UW solutions"""
# if uw.mpi.size == 1:
#     from matplotlib import rc
#     # Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
#     rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':10})
    
#     # Set the font used for MathJax
#     rc('mathtext',**{'default':'regular'})
#     rc('figure',**{'figsize':(8,6)})
    
#     ### profile from UW
#     plt.plot(sample_points[:, 0], T_UW_profile, ls="-", c="red", label="UW numerical solution")
#     ### analytical solution
#     plt.plot(x, T_anal_profile, ls=":", c="k", label="1D analytical solution")
    
    
#     plt.title(f'time: {round(model_time, 5)}', fontsize=8)
#     plt.legend()

#     # plt.savefig(f'benchmark_figs/AdvDiff_HP_benchmark-kappa={kappa}-degree={Tdegree}.pdf')
# -


# + active=""
#
