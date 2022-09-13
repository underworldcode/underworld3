# # Shear bands around a circular inclusion in a simple shear flow
#
# No slip conditions
#
#

expt_name = "ShearBand"

import petsc4py
import underworld3 as uw
import numpy as np


# +
import meshio, pygmsh

# Mesh a 2D pipe with a circular hole

csize = 0.075
csize_circle = 0.025
res = csize_circle

width = 3.0
height = 1.0
radius = 0.1

if uw.mpi.rank == 0:

    # Generate local mesh on boss process

    with pygmsh.geo.Geometry() as geom:

        geom.characteristic_length_max = csize

        inclusion = geom.add_circle(
            (width / 2.0, height / 2.0, 0.0), radius, make_surface=False, mesh_size=csize_circle
        )
        domain = geom.add_rectangle(
            xmin=0.0, ymin=0.0, xmax=width, ymax=height, z=0, holes=[inclusion], mesh_size=csize
        )

        geom.add_physical(domain.surface.curve_loop.curves[0], label="bottom")
        geom.add_physical(domain.surface.curve_loop.curves[1], label="right")
        geom.add_physical(domain.surface.curve_loop.curves[2], label="top")
        geom.add_physical(domain.surface.curve_loop.curves[3], label="left")

        geom.add_physical(inclusion.curve_loop.curves, label="inclusion")

        geom.add_physical(domain.surface, label="Elements")

        geom.generate_mesh(dim=2, verbose=False)
        geom.save_geometry("tmp_shear_inclusion.msh")

# -


mesh1 = uw.discretisation.Mesh("tmp_shear_inclusion.msh", simplex=True)
mesh1.dm.view()

# +
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

    mesh1.vtk("tmp_shear_inclusion.vtk")
    pvmesh = pv.read("tmp_shear_inclusion.vtk")

    pl = pv.Plotter()

    points = np.zeros((mesh1._centroids.shape[0], 3))
    points[:, 0] = mesh1._centroids[:, 0]
    points[:, 1] = mesh1._centroids[:, 1]

    point_cloud = pv.PolyData(points)

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, opacity=0.5)

    #

    pl.show(cpos="xy")

# +
# Define some functions on the mesh

import sympy

# radius_fn = sympy.sqrt(mesh1.rvec.dot(mesh1.rvec)) # normalise by outer radius if not 1.0
# unit_rvec = mesh1.rvec / (1.0e-10+radius_fn)

# Some useful coordinate stuff

x = mesh1.N.x
y = mesh1.N.y

# relative to the centre of the inclusion
r = sympy.sqrt((x - 1.0) ** 2 + (y - 0.5) ** 2)
th = sympy.atan2(y - 0.5, x - 1.0)

# need a unit_r_vec equivalent

inclusion_rvec = mesh1.rvec - 1.0 * mesh1.N.i - 0.5 * mesh1.N.j
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)

# -

v_soln = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh1, 1, degree=1)
vorticity = uw.discretisation.MeshVariable("omega", mesh1, 1, degree=1)
strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh1, 1, degree=1)
dev_stress_inv2 = uw.discretisation.MeshVariable("tau", mesh1, 1, degree=1)
node_viscosity = uw.discretisation.MeshVariable("eta", mesh1, 1, degree=1)
r_inc = uw.discretisation.MeshVariable("R", mesh1, 1, degree=1)


# +
# Create NS object

stokes = uw.systems.Stokes(
    mesh1,
    velocityField=v_soln,
    pressureField=p_soln,
    u_degree=v_soln.degree,
    p_degree=p_soln.degree,
    verbose=False,
    solver_name="stokes",
)


# +

nodal_strain_rate_inv2 = uw.systems.Projection(mesh1, strain_rate_inv2)
nodal_strain_rate_inv2.add_dirichlet_bc(1.0, "top", 0)
nodal_strain_rate_inv2.add_dirichlet_bc(1.0, "bottom", 0)
nodal_strain_rate_inv2.uw_function = stokes._Einv2
nodal_strain_rate_inv2.smoothing = 1.0e-3
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_tau_inv2 = uw.systems.Projection(mesh1, dev_stress_inv2)
nodal_tau_inv2.uw_function = stokes.viscosity * stokes._Einv2
nodal_tau_inv2.smoothing = 1.0e-3
nodal_tau_inv2.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh1, node_viscosity)
nodal_visc_calc.uw_function = stokes.viscosity
nodal_visc_calc.smoothing = 1.0e-3
nodal_visc_calc.petsc_options.delValue("ksp_monitor")


# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

# Constant visc

stokes.penalty = 0.0
stokes.bodyforce = 1.0e-32 * mesh1.N.i

hw = 1000.0 / res
with mesh1.access(r_inc):
    r_inc.data[:, 0] = uw.function.evaluate(r, mesh1.data)

surface_defn = sympy.exp(-(((r_inc.fn - radius) / radius) ** 2) * hw)

# Velocity boundary conditions

stokes.add_dirichlet_bc((0.0, 0.0), "inclusion", (0, 1))
stokes.add_dirichlet_bc((1.0, 0.0), "top", (0, 1))
stokes.add_dirichlet_bc((-1.0, 0.0), "bottom", (0, 1))
stokes.add_dirichlet_bc(0.0, "left", 1)
stokes.add_dirichlet_bc(0.0, "right", 1)
# -


# linear solve first
stokes.viscosity = 1.0
stokes.solve()

# +
# Now introduce the non-linearity once we have an initial strain rate

tau_y = 3.0 + 0.5 * stokes.p.fn

stokes.viscosity = sympy.Max(sympy.Min(tau_y / stokes._Einv2, 1.0), 0.1)
stokes._Ppre_fn = 1.0 / stokes.viscosity

stokes.solve(zero_init_guess=False)

# -


nodal_strain_rate_inv2.solve()

nodal_visc_calc.uw_function = stokes.viscosity
nodal_visc_calc.solve(_force_setup=True)

nodal_tau_inv2.uw_function = stokes.viscosity * stokes._Einv2
nodal_tau_inv2.solve(_force_setup=True)

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1250, 1250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh1.vtk("tmp_shear_inclusion.vtk")
    pvmesh = pv.read("tmp_shear_inclusion.vtk")

    with mesh1.access():
        usol = v_soln.data.copy()

    with mesh1.access():
        pvmesh.point_data["Vmag"] = uw.function.evaluate(sympy.sqrt(v_soln.fn.dot(v_soln.fn)), mesh1.data)
        pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn, mesh1.data)
        pvmesh.point_data["Edot"] = uw.function.evaluate(strain_rate_inv2.fn, mesh1.data)
        pvmesh.point_data["Str"] = uw.function.evaluate(dev_stress_inv2.fn, mesh1.data)
        pvmesh.point_data["Visc"] = uw.function.evaluate(node_viscosity.fn, mesh1.data)

    v_vectors = np.zeros((mesh1.data.shape[0], 3))
    v_vectors[:, 0:2] = uw.function.evaluate(v_soln.fn, mesh1.data)
    pvmesh.point_data["V"] = v_vectors

    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    # point sources at cell centres

    points = np.zeros((mesh1._centroids.shape[0], 3))
    points[:, 0] = mesh1._centroids[:, 0]
    points[:, 1] = mesh1._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(point_cloud, vectors="V", integration_direction="both", max_steps=100)

    pl = pv.Plotter(window_size=(1000, 500))

    pl.add_arrows(arrow_loc, arrow_length, mag=0.1, opacity=0.75)

    # pl.add_points(point_cloud, cmap="coolwarm",
    #               render_points_as_spheres=False,
    #               point_size=10, opacity=0.66
    #             )

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="P", use_transparency=False, opacity=1.0
    )

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)
    # pl.add_mesh(pvstream)

    # pl.remove_scalar_bar("mag")

    pl.show()
# -
pvmesh.point_data["Visc"].min(), pvmesh.point_data["Visc"].max()

pvmesh.point_data["P"].min(), pvmesh.point_data["P"].max()  # cf 4.26

pvmesh.point_data["Str"].min(), pvmesh.point_data["Str"].max()

pvmesh.point_data["Edot"].min(), pvmesh.point_data["Edot"].max()
