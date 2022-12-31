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

# # Spiegelman et al, notch-deformation benchmark
#
# This example is for the notch-localization test of Spiegelman et al. For which they supply a geometry file which gmsh can use to construct meshes at various resolutions. NOTE: we are just demonstrating the mesh here, not the solver configuration / benchmarking.
#
# The `.geo` file is provided and we show how to make this into a `.msh` file and
# how to read that into a `uw.discretisation.Mesh` object. The `.geo` file has header parameters to control the mesh refinement, and we provide a coarse version and the original version.
#
# After that, there is some cell data which we can assign to a data structure on the elements (such as a swarm).

# +
import petsc4py
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy
import gmsh
import os

os.makedirs("meshes", exist_ok=True)

if uw.mpi.size == 1:
    os.makedirs("output", exist_ok=True)
else:
    os.makedirs(f"output_np{uw.mpi.size}", exist_ok=True)


os.environ["UW_TIMING_ENABLE"] = "1"

# Define the problem size
#      1 - ultra low res for automatic checking
#      2 - low res problem to play with this notebook
#      3 - medium resolution (be prepared to wait)
#      4 - highest resolution (benchmark case from Spiegelman et al)

problem_size = 2

# For testing and automatic generation of notebook output,
# over-ride the problem size if the UW_TESTING_LEVEL is set

uw_testing_level = os.environ.get("UW_TESTING_LEVEL")
if uw_testing_level:
    try:
        problem_size = int(uw_testing_level)
    except ValueError:
        # Accept the default value
        pass

# -
options = PETSc.Options()
options["dm_adaptor"] = "pragmatic"


from underworld3.cython import petsc_discretisation


# +
if problem_size <= 1:
    cl_1 = 0.25
    cl_2 = 0.15
    cl_2a = 0.1
    cl_3 = 0.25
    cl_4 = 0.15
elif problem_size == 2:
    cl_1 = 0.1
    cl_2 = 0.05
    cl_2a = 0.03
    cl_3 = 0.1
    cl_4 = 0.05
elif problem_size == 3:
    cl_1 = 0.06
    cl_2 = 0.03
    cl_2a = 0.015
    cl_3 = 0.04
    cl_4 = 0.02
else:
    cl_1 = 0.04
    cl_2 = 0.005
    cl_2a = 0.003
    cl_3 = 0.02
    cl_4 = 0.01

# The benchmark provides a .geo file. This is the gmsh python
# equivalent (mostly transcribed from the .geo format). The duplicated
# Point2 caused a few problems with the mesh reader at one point.

if uw.mpi.rank == 0:

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("Notch")

    Point1 = gmsh.model.geo.addPoint(-2, -1, 0, cl_1)
    # Point2 = gmsh.model.geo.addPoint(-2, -1, 0, cl_1)
    Point3 = gmsh.model.geo.addPoint(+2, -1, 0, cl_1)
    Point4 = gmsh.model.geo.addPoint(2, -0.75, 0, cl_1)
    Point5 = gmsh.model.geo.addPoint(2, 0, 0, cl_1)
    Point6 = gmsh.model.geo.addPoint(-2, 0, 0, cl_1)
    Point7 = gmsh.model.geo.addPoint(-2, -0.75, 0, cl_1)
    Point8 = gmsh.model.geo.addPoint(-0.08333333333329999, -0.75, 0, cl_2)
    Point9 = gmsh.model.geo.addPoint(0.08333333333329999, -0.75, 0, cl_2)
    Point10 = gmsh.model.geo.addPoint(0.08333333333329999, -0.6666666666667, 0, cl_2)
    Point11 = gmsh.model.geo.addPoint(-0.08333333333329999, -0.6666666666667, 0, cl_2)
    Point25 = gmsh.model.geo.addPoint(-0.75, 0, 0, cl_4)
    Point26 = gmsh.model.geo.addPoint(0.75, 0, 0, cl_4)
    Point27 = gmsh.model.geo.addPoint(0, 0, 0, cl_3)

    Line1 = gmsh.model.geo.addLine(Point1, Point3)
    Line2 = gmsh.model.geo.addLine(Point3, Point4)
    Line3 = gmsh.model.geo.addLine(Point4, Point5)
    Line4 = gmsh.model.geo.addLine(Point5, Point26)
    Line8 = gmsh.model.geo.addLine(Point26, Point27)
    Line9 = gmsh.model.geo.addLine(Point27, Point25)
    Line10 = gmsh.model.geo.addLine(Point25, Point6)
    Line6 = gmsh.model.geo.addLine(Point6, Point7)
    Line7 = gmsh.model.geo.addLine(Point7, Point1)

    Point12 = gmsh.model.geo.addPoint(-0.1033333333333, -0.75, 0, cl_2a)
    Point13 = gmsh.model.geo.addPoint(-0.0833333333333, -0.73, 0, cl_2a)
    Point14 = gmsh.model.geo.addPoint(-0.0833333333333, -0.686666666666666, 0, cl_2a)
    Point15 = gmsh.model.geo.addPoint(-0.0633333333333, -0.666666666666666, 0, cl_2a)
    Point16 = gmsh.model.geo.addPoint(0.0633333333333, -0.666666666666666, 0, cl_2a)
    Point17 = gmsh.model.geo.addPoint(0.0833333333333, -0.686666666666666, 0, cl_2a)
    Point18 = gmsh.model.geo.addPoint(0.0833333333333, -0.73, 0, cl_2a)
    Point19 = gmsh.model.geo.addPoint(0.1033333333333, -0.75, 0, cl_2a)
    Point20 = gmsh.model.geo.addPoint(-0.103333333333333, -0.73, 0, cl_2a)
    Point21 = gmsh.model.geo.addPoint(-0.063333333333333, -0.686666666666666, 0, cl_2a)
    Point22 = gmsh.model.geo.addPoint(0.063333333333333, -0.686666666666666, 0, cl_2a)
    Point24 = gmsh.model.geo.addPoint(0.103333333333333, -0.73, 0, cl_2a)

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
        name="InnerBoundary",
    )

    gmsh.model.addPhysicalGroup(2, [Surface32], tag=100, name="Weak")
    gmsh.model.addPhysicalGroup(2, [Surface34], tag=101, name="Strong")

    gmsh.model.mesh.generate(2)

    gmsh.write(f"./meshes/notch_mesh{problem_size}.msh")
    gmsh.finalize()
# -


mesh1 = uw.discretisation.Mesh(
    f"./meshes/notch_mesh{problem_size}.msh",
    simplex=True,
    qdegree=3,
    markVertices=False,
    useRegions=True,
    useMultipleTags=True,
)

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

    mesh1.vtk("tmp_notch_msh.vtk")
    pvmesh = pv.read("tmp_notch_msh.vtk")

    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh,
        "Blue",
        "wireframe",
        opacity=0.5,
    )
    # pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=False, point_size=10, opacity=0.66)

    pl.show(cpos="xy")

swarm = uw.swarm.Swarm(mesh=mesh1)
material = uw.swarm.SwarmVariable(
    "M", swarm, num_components=1, proxy_continuous=False, proxy_degree=0
)
swarm.populate(fill_param=1)

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

indexSetW = mesh1.dm.getStratumIS("Weak", 100)
indexSetS = mesh1.dm.getStratumIS("Strong", 101)


l = swarm.dm.createLocalVectorFromField("M")
lvec = l.copy()
swarm.dm.restoreField("M")

lvec.isset(indexSetW, 0.0)
lvec.isset(indexSetS, 1.0)

with swarm.access(material):
    material.data[:, 0] = lvec.array[:]

# check the mesh if in a notebook / serial

if True and uw.mpi.size == 1:
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

    pvmesh = pv.read(f"./meshes/notch_mesh{problem_size}.msh")

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

    # pl.add_mesh(
    #     pvmesh,
    #     cmap="coolwarm",
    #     edge_color="Black",
    #     show_edges=True,
    #     use_transparency=False,
    #     opacity=0.5,
    # )
    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=False,
        point_size=10,
        opacity=0.66,
    )
    pl.add_mesh(pvmesh, "Black", "wireframe")

    pl.show(cpos="xy")


# ### Check that this mesh can be solved for a simple, linear problem

# Create Stokes object

stokes = uw.systems.Stokes(
    mesh1,
    velocityField=v_soln,
    pressureField=p_soln,
    solver_name="stokes",
    verbose=False,
)


# +
# Set solve options here (or remove default values
stokes.petsc_options["ksp_monitor"] = None

stokes.tolerance = 1.0e-6
stokes.petsc_options["snes_atol"] = 1e-2

# stokes.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1e-4
# stokes.petsc_options["fieldsplit_pressure_ksp_type"] = "gmres" # gmres here for bulletproof
stokes.petsc_options[
    "fieldsplit_pressure_pc_type"
] = "gamg"  # can use gasm / gamg / lu here
stokes.petsc_options[
    "fieldsplit_pressure_pc_gasm_type"
] = "basic"  # can use gasm / gamg / lu here
stokes.petsc_options[
    "fieldsplit_pressure_pc_gamg_type"
] = "classical"  # can use gasm / gamg / lu here
stokes.petsc_options["fieldsplit_pressure_pc_gamg_classical_type"] = "direct"
# # stokes.petsc_options["fieldsplit_velocity_pc_gamg_agg_nsmooths"] = 5
# # stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
# # stokes.petsc_options["fieldsplit_pressure_mg_levels_ksp_converged_maxits"] = None


# # Fast: preonly plus gasm / gamg / mumps
# # Robust: gmres plus gasm / gamg / mumps

# stokes.petsc_options["fieldsplit_velocity_pc_type"] = "gamg"
# # stokes.petsc_options["fieldsplit_velocity_pc_gasm_type"] = "basic" # can use gasm / gamg / lu here

# stokes.petsc_options["fieldsplit_velocity_pc_gamg_agg_nsmooths"] = 2
# stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 3

# stokes.petsc_options["fieldsplit_velocity_pc_gamg_esteig_ksp_type"] = "cg"
stokes.petsc_options["fieldsplit_pressure_pc_gamg_esteig_ksp_type"] = "cg"

# -


viscosity_L = 999.0 * material.sym[0] + 1.0

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh1.dim)
stokes.constitutive_model.Parameters.viscosity = viscosity_L
stokes.saddle_preconditioner = 1 / viscosity_L
stokes.penalty = 0.1

# Velocity boundary conditions
stokes.add_dirichlet_bc(1.0, "Left", 0)
stokes.add_dirichlet_bc(0, "Left", 1)
stokes.add_dirichlet_bc(-1.0, "Right", 0)
stokes.add_dirichlet_bc(0, "Right", 1)
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))
# stokes.add_dirichlet_bc((0.0,), "Top", (1,))


stokes.bodyforce = sympy.Matrix([0, -1])


# +
x, y = mesh1.X

res = 0.1
hw = 1000.0 / res
surface_defn_fn = sympy.exp(-((y - 0) ** 2) * hw)
base_defn_fn = sympy.exp(-((y + 1) ** 2) * hw)
edges_fn = sympy.exp(-((x - 2) ** 2) / 0.025) + sympy.exp(-((x + 2) ** 2) / 0.025)
# stokes.bodyforce -= 10000.0 * surface_defn_fn * v_soln.sym[1] * mesh1.CoordinateSystem.unit_j
# -

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

# +
# First, we solve the linear problem

stokes.tolerance = 1e-4
stokes.petsc_options["snes_atol"] = 1.0e-2

# stokes.petsc_options["ksp_rtol"]  = 1.0e-4
# stokes.petsc_options["ksp_atol"]  = 1.0e-8

# stokes.petsc_options["fieldsplit_pressure_ksp_rtol"]  = 1.0e-5
# stokes.petsc_options["fieldsplit_velocity_ksp_rtol"]  = 1.0e-5

from underworld3 import timing

timing.reset()
timing.start()
stokes.solve(zero_init_guess=True)
timing.print_table()

if uw.mpi.rank == 0:
    print("Linear solve complete", flush=True)


# +

C0 = 150
for i in range(10):

    mu = 0.75
    C = C0 + (1.0 - i / 9) * 15.0
    if uw.mpi.rank == 0:
        print(f"Mu - {mu}, C = {C}", flush=True)

    tau_y = C + mu * p_soln.sym[0]
    viscosity_L = 999.0 * material.sym[0] + 1.0
    viscosity_Y = tau_y / (2 * stokes._Einv2 + 1.0 / 1000)
    viscosity = 1 / (1 / viscosity_Y + 1 / viscosity_L)

    stokes.constitutive_model.Parameters.viscosity = viscosity
    stokes.saddle_preconditioner = 1 / viscosity

    # +
    # Now use that as the guess for a better job

    # stokes.tolerance = 1e-4
    # stokes.petsc_options["ksp_rtol"]  = 1.0e-4
    # stokes.petsc_options["ksp_atol"]  = 1.0e-8

    # stokes.petsc_options["fieldsplit_pressure_ksp_rtol"]  = 1.0e-5
    # stokes.petsc_options["fieldsplit_velocity_ksp_rtol"]  = 1.0e-5
    # stokes.snes.atol = 1e-3

    timing.reset()
    timing.start()
    stokes.solve(zero_init_guess=False)
    timing.print_table()
    if uw.mpi.rank == 0:
        print(f"Completed: Mu - {mu}, C = {C}", flush=True)
# -

# %%
viscosity_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
stress_calc.uw_function = (
    2 * stokes.constitutive_model.Parameters.viscosity * stokes._Einv2
)

# %%
strain_rate_calc.solve()
viscosity_calc.solve()
stress_calc.solve()

# +
## Save data ...

savefile = f"output/notched_beam_mesh{problem_size}.h5"
mesh1.save(savefile)
v_soln.save(savefile)
p_soln.save(savefile)
# mesh1.generate_xdmf(savefile)
# -

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
    pvmesh.point_data["pres"] = uw.function.evaluate(p_soln.sym[0], mesh1.data)
    pvmesh.point_data["edot"] = uw.function.evaluate(edot.sym[0], mesh1.data)
    # pvmesh.point_data["tauy"] = uw.function.evaluate(tau_y, mesh1.data, mesh1.N)
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

    # pl.add_arrows(arrow_loc, arrow_length, mag=0.03, opacity=0.75)

    pl.add_mesh(
        pvmesh,
        cmap="RdYlGn",
        scalars="edot",
        edge_color="Grey",
        # show_edges=True,
        use_transparency=False,
        clim=[0.1, 1.25],
        opacity=1.0,
    )

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=False,
        point_size=5,
        opacity=0.1,
    )

    pl.show(cpos="xy")

# # What works:
#
# Coarse mesh, small penalty (0.1) - C >= 100, mu 0.0 (step from C=1500)
#
# Coarse mesh, small penalty (0.1) - C >= 100, mu 0.3
#
# Coarse mesh, small penalty (0.1) - C >= 100, mu 0.6
#
# Coarse mesh, small penalty (0.1) - C >= 100, mu 0.9
#
# Coarse mesh, small penalty (1.0) - C >= 75, mu 0.9
#

# +
# %%
# surface_defn_fn = sympy.exp(-((y - 0) ** 2) * hw)
# p_surface_ave = surface_integral(mesh1, p_soln.sym[0], surface_defn_fn)
# print(f"Upper surface average P = {p_surface_ave}")

# +
# surface_defn_fn = sympy.exp(-((y + 1) ** 2) * hw)
# p_surface_ave = surface_integral(mesh1, p_soln.sym[0], surface_defn_fn)
# print(f"Lower surface average P = {p_surface_ave}")


# +
# %%
# surface_defn_fn = sympy.exp(-((y + 0.666) ** 2) * hw)
# p_surface_ave = surface_integral(mesh1, edot.sym[0], surface_defn_fn)
# print(f"Edot at 0.666 = {p_surface_ave}")
# -


if uw.mpi.size == 1:
    pvmesh.point_data["eta"].min(), pvmesh.point_data["eta"].max()
