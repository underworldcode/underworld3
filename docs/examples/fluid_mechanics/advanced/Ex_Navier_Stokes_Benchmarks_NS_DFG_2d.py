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
# Navier-Stokes DFG-2 Benchmark (Swarm-based)

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** advanced

## Description

Flow around a circular cylinder in a 2D channel - the classic DFG-2 benchmark
for validating incompressible Navier-Stokes solvers. Uses Lagrangian swarm-based
advection for the material derivative formulation.

## Key Concepts

- **DFG-2 benchmark**: Standard CFD validation case for Re=20-100
- **Lagrangian swarm DuDt**: Material derivative tracked on particles
- **Vortex shedding**: Periodic wake at higher Reynolds numbers
- **Parabolic inlet**: Fully-developed channel flow profile
- **PyGMSH mesh**: Local refinement around cylinder

## Mathematical Formulation

Navier-Stokes equations:
$$\\rho \\left( \\frac{\\partial \\mathbf{u}}{\\partial t} + \\mathbf{u} \\cdot \\nabla \\mathbf{u} \\right) = -\\nabla p + \\mu \\nabla^2 \\mathbf{u}$$

Parabolic inlet velocity:
$$V_b = \\frac{4 U_0 y (H - y)}{H^2}$$

where H = 0.41 is the channel height.

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
import petsc4py
import underworld3 as uw
import numpy as np
import sympy

import psutil
pid = os.getpid()
python_process = psutil.Process(pid)
print(f"Memory usage = {python_process.memory_info().rss // 1000000} Mb", flush=True)

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Ex_Navier_Stokes_Benchmarks_NS_DFG_2d.py -uw_resolution 32
python Ex_Navier_Stokes_Benchmarks_NS_DFG_2d.py -uw_model 3
python Ex_Navier_Stokes_Benchmarks_NS_DFG_2d.py -uw_max_steps 500
```
"""

# %%
params = uw.Params(
    uw_resolution = 16,           # Mesh resolution
    uw_refinement = 0,            # Mesh refinement levels
    uw_model = 3,                 # Model number (1-5)
    uw_max_steps = 251,           # Maximum time steps
    uw_restart_step = -1,         # Restart from step (-1 = fresh start)
    uw_dt = 0.01,                 # Base timestep
    uw_rho = 1000.0,              # Fluid density (scaled for stability)
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
    expt_name = f"NS_benchmark_DFG2d_1_{resolution}"
elif model == 2:
    U0 = 0.3
    expt_name = f"NS_benchmark_DFG2d_1_ss_{resolution}"
elif model == 3:
    U0 = 1.5
    expt_name = f"NS_benchmark_DFG2d_2iii_{resolution}"
elif model == 4:
    U0 = 3.75
    expt_name = f"NS_test_Re_250_{resolution}"
elif model == 5:
    U0 = 15
    expt_name = f"NS_test_Re_1000_{resolution}"

# %% [markdown]
"""
## Output Directory
"""

# %%
outdir = f"output_swarm_{resolution}"
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
csize_circle = 0.5 * csize
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
    far_right = coords[:, 0] > 2.2
    too_low = coords[:, 1] < 0.0
    too_high = coords[:, 1] > 0.41

    coords[lefty_troublemakers, 0] = 0.0001
    coords[far_right, 0] = 2.2 - 0.0001
    coords[too_low, 1] = 0.0001
    coords[too_high, 1] = 0.41 - 0.0001

    return coords


if uw.mpi.rank == 0:
    # Generate mesh on rank 0
    with pygmsh.geo.Geometry() as geom:
        geom.characteristic_length_max = csize

        inclusion = geom.add_circle(
            (0.2, 0.2, 0.0), radius, make_surface=False, mesh_size=csize_circle
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

        geom.add_physical(domain.surface.curve_loop.curves[0], label="bottom")
        geom.add_physical(domain.surface.curve_loop.curves[1], label="right")
        geom.add_physical(domain.surface.curve_loop.curves[2], label="top")
        geom.add_physical(domain.surface.curve_loop.curves[3], label="left")
        geom.add_physical(inclusion.curve_loop.curves, label="inclusion")
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

print(f"Memory usage = {python_process.memory_info().rss // 1000000} Mb", flush=True)

# %% [markdown]
"""
## Variables
"""

# %%
v_soln = uw.discretisation.MeshVariable("U", pipemesh, pipemesh.dim, degree=2)
vs_soln = uw.discretisation.MeshVariable("Us", pipemesh, pipemesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", pipemesh, 1, degree=1)
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
## Swarm-based Material Derivative

The Lagrangian swarm tracks velocity history for the DuDt term.
"""

# %%
swarm = uw.swarm.Swarm(mesh=pipemesh, recycle_rate=6)

DvDt = uw.systems.ddt.Lagrangian_Swarm(
    swarm,
    v_soln.sym,
    uw.VarType.VECTOR,
    degree=2,
    order=2,
    verbose=False,
    continuous=True
)

swarm.populate(fill_param=4)

# %% [markdown]
"""
## Passive Swarm for Visualization
"""

# %%
passive_swarm = uw.swarm.Swarm(mesh=pipemesh)
passive_swarm.populate(fill_param=1)

# Add seed points at inflow
npoints = 50
passive_swarm.dm.addNPoints(npoints)
with passive_swarm.access(passive_swarm._particle_coordinates):
    for i in range(npoints):
        passive_swarm._particle_coordinates.data[-1:-(npoints + 1):-1, :] = np.array(
            [0.0, 0.195] + 0.01 * np.random.random((npoints, 2))
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
## Navier-Stokes Solver
"""

# %%
navier_stokes = uw.systems.NavierStokes(
    pipemesh,
    velocityField=v_soln,
    pressureField=p_soln,
    DuDt=DvDt,
    rho=rho,
    verbose=False,
    order=2
)

navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

navier_stokes.penalty = 0.1
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

navier_stokes.tolerance = 1.0e-3
navier_stokes.delta_t = 10.0  # Stokes-like initial solve

if model == 2:  # Steady state
    # Remove d/dt term for steady state solution
    navier_stokes.UF0 = -(
        navier_stokes.rho * (v_soln.sym - vs_soln.sym) / navier_stokes.delta_t
    )

# %% [markdown]
"""
## Initial Solve
"""

# %%
navier_stokes.Unknowns.DuDt.update_pre_solve(0.0)
navier_stokes.solve(timestep=10.0, verbose=False)  # Stokes-like initial flow
nodal_vorticity_from_v.solve()

# %% [markdown]
"""
## Visualization Function
"""

# %%
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pl = pv.Plotter(window_size=(1000, 750))


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

        # Streamline sources
        points = np.zeros((pipemesh._centroids.shape[0], 3))
        points[:, 0] = pipemesh._centroids[:, 0]
        points[:, 1] = pipemesh._centroids[:, 1]
        point_cloud = pv.PolyData(points)

        passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)
        active_swarm_points = uw.visualisation.swarm_to_pv_cloud(swarm)

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

        pl.add_points(
            active_swarm_points,
            color="DarkGreen",
            render_points_as_spheres=True,
            point_size=2,
            opacity=0.25
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
ts = 0
elapsed_time = 0.0
delta_t_cfl = 5 * navier_stokes.estimate_dt()
delta_t = min(delta_t_cfl, dt_ns)

navier_stokes.DuDt.update(delta_t)

for step in range(0, maxsteps):
    delta_t_cfl = 5 * navier_stokes.estimate_dt()

    if step % 10 == 0:
        delta_t = min(delta_t_cfl, dt_ns)

    navier_stokes.solve(timestep=dt_ns, zero_init_guess=False)

    # Update passive swarm
    passive_swarm.advection(v_soln.sym, delta_t, order=2, corrector=False, evalf=False)

    # Update material point swarm
    swarm.advection(v_soln.sym, delta_t, order=2, corrector=False, evalf=False)

    # Add new points at inflow
    npoints = 200
    passive_swarm.dm.addNPoints(npoints)
    with passive_swarm.access(passive_swarm._particle_coordinates):
        for i in range(npoints):
            passive_swarm._particle_coordinates.data[
                -1:-(npoints + 1):-1, :
            ] = np.array([0.0, 0.195] + 0.01 * np.random.random((npoints, 2)))

    uw.pprint(f"Timestep {ts}, t {elapsed_time:.4f}, dt {delta_t:.4e}, dt_cfl {delta_t_cfl:.4e}")

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
print(f"DFG-2 benchmark (swarm) complete: model={model}, resolution={resolution}, steps={ts}")
