# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Navier-Stokes DFG-2 Benchmark (SLCN)

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** advanced

## Description

Flow around a circular cylinder in a 2D channel - the classic DFG-2 benchmark
for validating incompressible Navier-Stokes solvers. Uses Semi-Lagrangian
Crank-Nicolson (SLCN) method for time integration.

## Key Concepts

- **DFG-2 benchmark**: Standard CFD validation case for Re=20-100
- **SLCN formulation**: Semi-Lagrangian Crank-Nicolson time integration
- **Mesh-based DuDt**: No explicit particle tracking required
- **Vortex shedding**: Periodic wake at higher Reynolds numbers
- **Multigrid preconditioning**: Efficient solver for large systems

## Mathematical Formulation

Navier-Stokes equations:
$$\\rho \\left( \\frac{\\partial \\mathbf{u}}{\\partial t} + \\mathbf{u} \\cdot \\nabla \\mathbf{u} \\right) = -\\nabla p + \\mu \\nabla^2 \\mathbf{u}$$

SLCN discretization:
$$\\frac{\\mathbf{u}^{n+1} - \\mathbf{u}^n_{*}}{\\Delta t} = \\frac{1}{2}\\left(\\mathcal{L}(\\mathbf{u}^{n+1}) + \\mathcal{L}(\\mathbf{u}^n_{*})\\right)$$

where $\\mathbf{u}^n_{*}$ is the velocity at the departure point.

## Model Cases

| Model | U0  | Re  | Description |
|-------|-----|-----|-------------|
| 1     | 0.3 | 20  | Steady laminar |
| 2     | 0.3 | 20  | Steady state solve |
| 3     | 1.5 | 100 | Vortex shedding |
| 4     | 3.75| 250 | High Re test |
| 5     | 15  | 1000| Very high Re test |

## Parameters

- `uw_resolution`: Mesh resolution (elements per unit length)
- `uw_refinement`: Mesh refinement levels
- `uw_model`: Model number (1-5)
- `uw_max_steps`: Maximum time steps
- `uw_restart_step`: Restart from checkpoint (-1 = fresh start)

## References

- [DFG Benchmark](http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html)
"""

# %% [markdown]
"""
## Setup and Parameters
"""

# %%
import nest_asyncio
nest_asyncio.apply()

import os

os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
import underworld3 as uw
from underworld3 import timing

import numpy as np
import sympy

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Navier_Stokes_Benchmarks_NS_DFG_2d_SLCN.py -uw_resolution 32
python Ex_Navier_Stokes_Benchmarks_NS_DFG_2d_SLCN.py -uw_model 3
python Ex_Navier_Stokes_Benchmarks_NS_DFG_2d_SLCN.py -uw_max_steps 500
```
"""

# %%
params = uw.Params(
    uw_resolution = 20,           # Mesh resolution
    uw_refinement = 0,            # Mesh refinement levels
    uw_model = 1,                 # Model number (1-5)
    uw_max_steps = 201,           # Maximum time steps
    uw_restart_step = -1,         # Restart from step (-1 = fresh start)
    uw_dt = 0.005,                # Base timestep
    uw_rho = 1000,                # Fluid density (scaled for stability)
)

resolution = int(params.uw_resolution)
refinement = int(params.uw_refinement)
model = int(params.uw_model)
maxsteps = int(params.uw_max_steps)
restart_step = int(params.uw_restart_step)
dt_ns = params.uw_dt
rho = params.uw_rho

# %% [markdown]
"""
## Model Selection
"""

# %%
if model == 1:
    U0 = 0.3
    expt_name = f"NS_benchmark_DFG2d_SLCN_1_{resolution}"
elif model == 2:
    U0 = 0.3
    expt_name = f"NS_benchmark_DFG2d_SLCN_1_ss_{resolution}"
elif model == 3:
    U0 = 1.5
    expt_name = f"NS_benchmark_DFG2d_SLCN_2_{resolution}"
elif model == 4:
    U0 = 3.75
    expt_name = f"NS_test_Re_250_SLCN_{resolution}"
elif model == 5:
    U0 = 15
    expt_name = f"NS_test_Re_1000i_SLCN_{resolution}"

# %% [markdown]
"""
## Output Directory
"""

# %%
outdir = f"output/output_res_{resolution}"
os.makedirs(".meshes", exist_ok=True)
os.makedirs(f"{outdir}", exist_ok=True)

# %% [markdown]
"""
## Mesh Generation with PyGMSH
"""

# %%
import pygmsh
from enum import Enum


class boundaries(Enum):
    bottom = 1
    right = 2
    top = 3
    left = 4
    inclusion = 5
    All_Boundaries = 1001


# Domain parameters
csize = 1.0 / resolution
csize_circle = 0.25 * csize
res = csize_circle

width = 2.2
height = 0.41
radius = 0.05
centre = (0.2, 0.2)


def pipemesh_mesh_refinement_callback(dm):
    """Project points onto the circular inclusion surface."""
    r_p = radius

    c2 = dm.getCoordinatesLocal()
    coords = c2.array.reshape(-1, 2) - centre

    R = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2).reshape(-1, 1)

    pipeIndices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
        dm, "inclusion"
    )

    coords[pipeIndices] *= r_p / R[pipeIndices]
    coords = coords + centre

    c2.array[...] = coords.reshape(-1)
    dm.setCoordinatesLocal(c2)

    return


def pipemesh_return_coords_to_bounds(coords):
    """Restore particles that leave the domain to valid positions."""
    lefty_troublemakers = coords[:, 0] < 0.0
    coords[lefty_troublemakers, 0] = 0.0001

    return coords


if uw.mpi.rank == 0:
    # Generate mesh on rank 0
    with pygmsh.geo.Geometry() as geom:
        geom.characteristic_length_max = csize

        inclusion = geom.add_circle(
            (centre[0], centre[1], 0.0),
            radius,
            make_surface=False,
            mesh_size=csize_circle
        )
        domain = geom.add_rectangle(
            xmin=0.0,
            ymin=0.0,
            xmax=width,
            ymax=height,
            z=0,
            holes=[inclusion],
            mesh_size=csize
        )

        geom.add_physical(domain.surface.curve_loop.curves[0], label=boundaries.bottom.name)
        geom.add_physical(domain.surface.curve_loop.curves[1], label=boundaries.right.name)
        geom.add_physical(domain.surface.curve_loop.curves[2], label=boundaries.top.name)
        geom.add_physical(domain.surface.curve_loop.curves[3], label=boundaries.left.name)
        geom.add_physical(inclusion.curve_loop.curves, label=boundaries.inclusion.name)
        geom.add_physical(domain.surface, label="Elements")

        geom.generate_mesh(dim=2, verbose=False)
        geom.save_geometry(f".meshes/ns_pipe_flow_{resolution}.msh")

pipemesh = uw.discretisation.Mesh(
    f".meshes/ns_pipe_flow_{resolution}.msh",
    markVertices=True,
    useMultipleTags=True,
    useRegions=True,
    refinement=refinement,
    refinement_callback=pipemesh_mesh_refinement_callback,
    return_coords_to_bounds=pipemesh_return_coords_to_bounds,
    boundaries=boundaries,
    qdegree=3
)

pipemesh.dm.view()

# %% [markdown]
"""
## Coordinate System
"""

# %%
x = pipemesh.N.x
y = pipemesh.N.y

# Relative to inclusion centre
r = sympy.sqrt((x - 0.2) ** 2 + (y - 0.2) ** 2)
th = sympy.atan2(y - 0.2, x - 0.2)

inclusion_rvec = pipemesh.rvec - 1.0 * pipemesh.N.i - 0.5 * pipemesh.N.j
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)

# Parabolic inlet velocity
Vb = (4.0 * U0 * y * (0.41 - y)) / 0.41 ** 2

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", pipemesh, pipemesh.dim, degree=2)
vs_soln = uw.discretisation.MeshVariable("Us", pipemesh, pipemesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", pipemesh, 1, degree=1, continuous=True)
p_cont = uw.discretisation.MeshVariable("Pc", pipemesh, 1, degree=2, continuous=True)
vorticity = uw.discretisation.MeshVariable("omega", pipemesh, 1, degree=1)
r_inc = uw.discretisation.MeshVariable("R", pipemesh, 1, degree=1)
rho_var = uw.discretisation.MeshVariable("rho", pipemesh, 1, degree=1, varsymbol=r"{\rho}")

# Deviatoric stress tensor
work = uw.discretisation.MeshVariable(
    "W", pipemesh, 1, vtype=uw.VarType.SCALAR, degree=1, continuous=False
)

St = uw.discretisation.MeshVariable(
    r"Stress",
    pipemesh,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    degree=1,
    continuous=False,
    varsymbol=r"{\tau}"
)

# %% [markdown]
"""
## Passive Swarm for Visualization
"""

# %%
passive_swarm = uw.swarm.Swarm(mesh=pipemesh)
passive_swarm.populate(fill_param=1)

# Add seed points at inflow
npoints = 100
passive_swarm.dm.addNPoints(npoints)
with passive_swarm.access(passive_swarm._particle_coordinates):
    for i in range(npoints):
        passive_swarm._particle_coordinates.data[-1:-(npoints + 1):-1, :] = np.array(
            [0.01, 0.195] + 0.01 * np.random.random((npoints, 2))
        )

# %% [markdown]
"""
## Vorticity Projection
"""

# %%
nodal_vorticity_from_v = uw.systems.Projection(pipemesh, vorticity)
nodal_vorticity_from_v.uw_function = sympy.vector.curl(v_soln.fn).dot(pipemesh.N.k)
nodal_vorticity_from_v.smoothing = 1.0e-3
nodal_vorticity_from_v.petsc_options.delValue("ksp_monitor")

# %% [markdown]
"""
## Navier-Stokes Solver (SLCN)
"""

# %%
navier_stokes = uw.systems.NavierStokes(
    pipemesh,
    velocityField=v_soln,
    pressureField=p_soln,
    rho=rho,
    verbose=False,
    order=2
)

navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

navier_stokes.penalty = 100
navier_stokes.bodyforce = sympy.Matrix([0, 0])

hw = 1000.0 / res
with pipemesh.access(r_inc):
    r_inc.data[:, 0] = uw.function.evalf(r, pipemesh.data, pipemesh.N)

surface_defn = sympy.exp(-(((r_inc.fn - radius) / radius) ** 2) * hw)

# %% [markdown]
"""
## Boundary Conditions
"""

# %%
navier_stokes.add_dirichlet_bc((0.0, 0.0), "inclusion")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "top")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "bottom")
navier_stokes.add_dirichlet_bc((Vb, 0.0), "left")

# %% [markdown]
"""
## Pressure Projection
"""

# %%
continuous_pressure_projection = uw.systems.Projection(pipemesh, p_cont)
continuous_pressure_projection.uw_function = p_soln.sym[0]
continuous_pressure_projection.solve()

# %% [markdown]
"""
## Solver Configuration
"""

# %%
navier_stokes.tolerance = 1.0e-4
navier_stokes.delta_t = 10  # Stokes-like initial solve

if model == 2:  # Steady state
    navier_stokes.UF0 = -(
        navier_stokes.rho * (v_soln.sym - vs_soln.sym) / navier_stokes.delta_t
    )

navier_stokes.view()

# %%
navier_stokes.petsc_options["snes_monitor"] = None
navier_stokes.petsc_options["ksp_monitor"] = None

navier_stokes.petsc_options["snes_type"] = "newtonls"
navier_stokes.petsc_options["ksp_type"] = "fgmres"

navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

navier_stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
navier_stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 2
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# %% [markdown]
"""
## Initial Solve
"""

# %%
timing.reset()
timing.start()

navier_stokes.solve(timestep=10, verbose=False)  # Stokes-like initial flow
nodal_vorticity_from_v.solve()

timing.print_table(display_fraction=0.999)

# %%
continuous_pressure_projection.solve()

# %% [markdown]
"""
## Visualization Function
"""

# %%
def plot_V_mesh(filename):
    if uw.mpi.size == 1:
        import pyvista as pv
        import underworld3.visualisation as vis

        pvmesh = vis.mesh_to_pv_mesh(pipemesh)
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
        pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)
        pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))

        velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

        pl = pv.Plotter(window_size=(1000, 750))

        # Streamline sources
        points = np.zeros((pipemesh._centroids.shape[0], 3))
        points[:, 0] = pipemesh._centroids[:, 0]
        points[:, 1] = pipemesh._centroids[:, 1]
        point_cloud = pv.PolyData(points)

        passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)

        pvstream = pvmesh.streamlines_from_source(
            point_cloud, vectors="V", integration_direction="forward", max_steps=10
        )

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=True,
            scalars="Omega",
            use_transparency=False,
            opacity=1.0,
            show_scalar_bar=False
        )

        pl.add_arrows(
            velocity_points.points, velocity_points.point_data["V"],
            mag=0.025 / U0, opacity=0.75,
            show_scalar_bar=False
        )

        pl.add_mesh(pvstream, show_scalar_bar=False)

        pl.add_points(
            passive_swarm_points,
            color="Black",
            render_points_as_spheres=True,
            point_size=5,
            opacity=0.5
        )

        pl.camera.SetPosition(0.75, 0.2, 1.5)
        pl.camera.SetFocalPoint(0.75, 0.2, 0.0)
        pl.camera.SetClippingRange(1.0, 8.0)

        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(2560, 1280),
            return_img=False
        )

        pl.clear()


# %% [markdown]
"""
## Time Evolution
"""

# %%
dt1, _ = navier_stokes.estimate_dt()

ts = 0
elapsed_time = 0.0
delta_t_diff, delta_t_adv = navier_stokes.estimate_dt()
delta_t = dt_ns

print(f"Dt_adv -> {delta_t_adv}; Dt_diff -> {delta_t_diff}")
print(f"delta_t / delta_t_adv = {delta_t / delta_t_adv}")

for step in range(0, maxsteps):

    navier_stokes.solve(timestep=delta_t, zero_init_guess=False, verbose=False)

    # Update passive swarm
    passive_swarm.advection(v_soln.sym, delta_t, order=2, corrector=False, evalf=False)

    # Add new points at inflow
    npoints = 200
    passive_swarm.dm.addNPoints(npoints)
    with passive_swarm.access(passive_swarm._particle_coordinates):
        for i in range(npoints):
            passive_swarm._particle_coordinates.data[
                -1:-(npoints + 1):-1, :
            ] = np.array([0.0, 0.195] + 0.01 * np.random.random((npoints, 2)))

    uw.pprint(f"Timestep {ts}, t {elapsed_time:.4f}, dt {delta_t:.4e}")

    if ts % 10 == 0:
        nodal_vorticity_from_v.solve()
        plot_V_mesh(filename=f"{outdir}/{expt_name}.{ts:05d}")

        pipemesh.write_timestep(
            expt_name,
            meshUpdates=True,
            meshVars=[p_soln, v_soln, vorticity, St],
            outputPath=outdir,
            index=ts
        )

        passive_swarm.write_timestep(
            expt_name,
            "passive_swarm",
            swarmVars=None,
            outputPath=outdir,
            index=ts,
            force_sequential=True
        )

    elapsed_time += delta_t
    ts += 1

# %%
print(f"DFG-2 benchmark (SLCN) complete: model={model}, resolution={resolution}, steps={ts}")
