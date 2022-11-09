# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# coding: utf-8
# %% [markdown]
# # Spiegelman et al, notch-deformation benchmark
#
# This example is for the notch-localization test of Spiegelman et al. For which they supply a geometry file which gmsh can use to construct meshes at various resolutions. NOTE: we are just demonstrating the mesh here, not the solver configuration / benchmarking.
#
# The `.geo` file is provided and we show how to make this into a `.msh` file and
# how to read that into a `uw.discretisation.Mesh` object. The `.geo` file has header parameters to control the mesh refinement, and we provide a coarse version and the original version.
#
# After that, there is some cell data which we can assign to a data structure on the elements (such as a swarm).

# %%
import gmsh
import meshio

import petsc4py
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

from underworld3.cython import petsc_discretisation


# %%
mesh_res = "coarse"  # For tests
build_mesh = False

# %%
if build_mesh:
    if mesh_res == "coarse":
        gmsh.initialize()
        gmsh.model.add("Notch")
        gmsh.open("meshes/compression_mesh_rounded_coarse.geo")
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write("meshes/notch_coarse.msh")
        gmsh.finalize()

    elif mesh_res == "medium":
        gmsh.initialize()
        gmsh.model.add("Notch")
        gmsh.open("meshes/compression_mesh_rounded_medium.geo")
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write("meshes/notch_medium.msh")
        gmsh.finalize()

    else:
        gmsh.initialize()
        gmsh.model.add("Notch")
        gmsh.open("meshes/compression_mesh_rounded_refine.geo")
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write("meshes/notch_refine.msh")
        gmsh.finalize()


# Create dictionary to tell the mesh constructor about the mesh labels
# which are not embedded in the .geo file in the way the physical groups
# are embedded in the .msh file

cell_sets = []
cell_sets += [{"name": "weak", "id": 0}]
cell_sets += [{"name": "strong", "id": 1}]

face_sets = []
face_sets += [{"name": "Left", "id": 1}]
face_sets += [{"name": "Right", "id": 2}]
face_sets += [{"name": "Bottom", "id": 3}]
face_sets += [{"name": "Top", "id": 4}]

# %%
mesh1 = uw.discretisation.Mesh(
    f"./meshes/notch_{mesh_res}.msh",
    simplex=True,
    qdegree=3,
    cellSets=cell_sets,
    faceSets=face_sets,
)
mesh1.dm.view()


# %%
swarm = uw.swarm.Swarm(mesh=mesh1)
material = uw.swarm.SwarmVariable(
    "M", swarm, num_components=1, proxy_continuous=False, proxy_degree=1
)
swarm.populate(fill_param=1)

# %%
v_soln = uw.discretisation.MeshVariable(r"U", mesh1, mesh1.dim, degree=2)
p_soln = uw.discretisation.MeshVariable(r"P", mesh1, 1, degree=1, continuous=True)
p_null = uw.discretisation.MeshVariable(r"P2", mesh1, 1, degree=1, continuous=True)

edot = uw.discretisation.MeshVariable(
    r"\dot\varepsilon", mesh1, 1, degree=1, continuous=True
)
visc = uw.discretisation.MeshVariable(r"\eta", mesh1, 1, degree=1, continuous=True)
stress = uw.discretisation.MeshVariable(r"\sigma", mesh1, 1, degree=1, continuous=True)

# + [markdown] magic_args="[markdown]"
# This is how we extract cell data from the mesh. We can map it to the swarm data structure and use this to
# build material properties that depend on cell type.
# -

# Parallel ? Local or global ?

indexSetW = mesh1.dm.getStratumIS("Cell Sets", 0)
indexSetS = mesh1.dm.getStratumIS("Cell Sets", 1)


l = swarm.dm.createLocalVectorFromField("M")
lvec = l.copy()
swarm.dm.restoreField("M")

lvec.isset(indexSetW, 0.0)
lvec.isset(indexSetS, 1.0)

with swarm.access(material):
    material.data[:, 0] = lvec.array[:]

# check the mesh if in a notebook / serial

if False and uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1050, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

    mesh1.vtk("tmp_notch.vtk")
    pvmesh = pv.read("tmp_notch.vtk")

    pl = pv.Plotter()

    # points = np.zeros((mesh1._centroids.shape[0], 3))
    # points[:, 0] = mesh1._centroids[:, 0]
    # points[:, 1] = mesh1._centroids[:, 1]

    with swarm.access():
        points = np.zeros((swarm.particle_coordinates.data.shape[0], 3))
        points[:, 0] = swarm.particle_coordinates.data[:, 0]
        points[:, 1] = swarm.particle_coordinates.data[:, 1]

    point_cloud = pv.PolyData(points)

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    pvmesh.point_data["eta"] = uw.function.evaluate(
        material.sym[0], mesh1.data, mesh1.N
    )

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
        opacity=0.5,
    )
    # pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=False, point_size=10, opacity=0.66)

    pl.show(cpos="xy")


# Check that this mesh can be solved for a simple, linear problem

# Create Stokes object

stokes = uw.systems.Stokes(
    mesh1,
    velocityField=v_soln,
    pressureField=p_soln,
    solver_name="stokes",
    verbose=True,
)

# Set solve options here (or remove default values
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_atol"] = 0.001

# Level set approach to rheology:
viscosity_L = sympy.Piecewise(
    (1.0, material.sym[0] < 0.5),
    (1000.0, True),
)

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh1.dim)
stokes.constitutive_model.Parameters.viscosity = viscosity_L
stokes.saddle_preconditioner = 1 / viscosity_L
stokes.penalty = 0.0

# Velocity boundary conditions
stokes.add_dirichlet_bc((+1.0, 0), "Left", (0, 1))
stokes.add_dirichlet_bc((-1.0, 0), "Right", (0, 1))
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))
stokes.add_dirichlet_bc((0.0,), "Upper", (1,))


stokes.bodyforce = sympy.Matrix([0, -1])

# %%
x, y = mesh1.X

res = 0.1
hw = 1000.0 / res
surface_defn_fn = sympy.exp(-((y - 0) ** 2) * hw)
base_defn_fn = sympy.exp(-((y + 1) ** 2) * hw)
edges_fn = sympy.exp(-((x - 2) ** 2) / 0.025) + sympy.exp(-((x + 2) ** 2) / 0.025)
# stokes.bodyforce -= 10000.0 * surface_defn_fn * v_soln.sym[1] * mesh1.CoordinateSystem.unit_j

# This is a strategy to obtain integrals over the surface (etc)


def surface_integral(mesh, uw_function, mask_fn):

    calculator = uw.maths.Integral(mesh, uw_function * mask_fn)
    value = calculator.evaluate()

    calculator.fn = mask_fn
    norm = calculator.evaluate()

    integral = value / norm

    return integral


# %%
strain_rate_calc = uw.systems.Projection(mesh1, edot)
strain_rate_calc.uw_function = stokes._Einv2
strain_rate_calc.smoothing = 1.0e-3

viscosity_calc = uw.systems.Projection(mesh1, visc)
viscosity_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
viscosity_calc.smoothing = 1.0e-3

stress_calc = uw.systems.Projection(mesh1, stress)
S = stokes.stress_deviator
stress_calc.uw_function = (
    sympy.simplify(sympy.sqrt(((S**2).trace()) / 2)) - p_soln.sym[0]
)
stress_calc.smoothing = 1.0e-3

# %%
print(f"Stokes setup", flush=True)


# %%
stokes._setup_terms()

# %%
stokes.solve(zero_init_guess=True)

# %%
p_surface_ave = surface_integral(mesh1, p_soln.sym[0], surface_defn_fn)

print(f"Surface Average pressure: {p_surface_ave}")

with mesh1.access(p_null):
    p_null.data[:] = p_surface_ave

# %%
tau_y = 100 + 0.5 * (p_soln.sym[0] - p_null.sym[0])

viscosity_L = 999.0 * material.sym[0] + 1.0
viscosity_Y = tau_y / (2 * stokes._Einv2 + 0.001)

viscosity_H = 1 / (1 / viscosity_Y + 1 / 1)

viscosity = sympy.Piecewise(
    (1.0, material.sym[0] < 0.5),
    (viscosity_Y, True),
)

stokes.constitutive_model.Parameters.viscosity = viscosity
stokes.saddle_preconditioner = 1 / viscosity

# %%
stokes._setup_terms()

# Fake call-back version

for i in range(5):
    p_surface_ave = surface_integral(mesh1, p_soln.sym[0], surface_defn_fn)
    with mesh1.access(p_null):
        p_null.data[:] = p_surface_ave

    stokes.solve(zero_init_guess=False)

# %%
viscosity_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
stress_calc.uw_function = (
    2 * stokes.constitutive_model.Parameters.viscosity * stokes._Einv2
)

# %%
strain_rate_calc.solve()
viscosity_calc.solve()
stress_calc.solve()

# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1050, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

    mesh1.vtk("tmp_notch.vtk")
    pvmesh = pv.read("tmp_notch.vtk")

    points = np.zeros((mesh1._centroids.shape[0], 3))
    points[:, 0] = mesh1._centroids[:, 0]
    points[:, 1] = mesh1._centroids[:, 1]

    pvmesh.point_data["sfn"] = uw.function.evaluate(
        surface_defn_fn, mesh1.data, mesh1.N
    )
    pvmesh.point_data["pres"] = uw.function.evaluate(
        p_soln.sym[0] - p_null.sym[0], mesh1.data
    )
    pvmesh.point_data["edot"] = uw.function.evaluate(edot.sym[0], mesh1.data)
    pvmesh.point_data["eta"] = uw.function.evaluate(visc.sym[0], mesh1.data)
    pvmesh.point_data["str"] = uw.function.evaluate(stress.sym[0], mesh1.data)

    with mesh1.access():
        usol = v_soln.data.copy()

    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    point_cloud = pv.PolyData(points)

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    pl = pv.Plotter()

    # pl.add_arrows(arrow_loc, arrow_length, mag=0.05, opacity=0.75)

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        scalars="str",
        edge_color="Grey",
        show_edges=True,
        use_transparency=False,
        # clim=[0.5,0.8],
        opacity=0.75,
    )

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=False,
        point_size=10,
        opacity=0.3,
    )

    pl.show(cpos="xy")

# %%
surface_defn_fn = sympy.exp(-((y - 0) ** 2) * hw)
p_surface_ave = surface_integral(mesh1, p_soln.sym[0], surface_defn_fn)
print(f"Upper surface average P = {p_surface_ave}")

surface_defn_fn = sympy.exp(-((y + 1) ** 2) * hw)
p_surface_ave = surface_integral(mesh1, p_soln.sym[0], surface_defn_fn)
print(f"Lower surface average P = {p_surface_ave}")


# %%
surface_defn_fn = sympy.exp(-((y + 0.666) ** 2) * hw)
p_surface_ave = surface_integral(mesh1, edot.sym[0], surface_defn_fn)
print(f"Edot at 0.666 = {p_surface_ave}")

