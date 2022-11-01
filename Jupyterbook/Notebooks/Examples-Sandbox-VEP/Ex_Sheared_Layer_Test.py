# # Validate constitutive models
#
# Simple shear with material defined by particle swarm (based on inclusion model), position, pressure, strain rate etc.
# Check Jacobians ... 
#

expt_name = "ShearBand"

# +
import petsc4py
import underworld3 as uw
import numpy as np

import pyvista as pv
import vtk

pv.global_theme.background = "white"
pv.global_theme.window_size = [1250, 1250]
pv.global_theme.antialiasing = True
pv.global_theme.jupyter_backend = "panel"
pv.global_theme.smooth_shading = True
pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
pv.global_theme.camera["position"] = [0.0, 0.0, 20.0]


# +
import meshio, pygmsh

# Mesh a 2D pipe with a circular hole

csize = 0.075
csize_circle = 0.025
res = csize_circle
cellSize = csize

width = 3.0
height = 1.0
radius = 0.05

eta1 = 100
eta2 = 10

if uw.mpi.rank == 0:
    
    import gmsh
    gmsh.initialize()
    gmsh.model.add("Periodic x")


    # %%
    boundaries = {
        "Bottom": 1,
        "Top": 2,
        "Right": 3,
        "Left": 4,
    }

    xmin, ymin = -width/2,-height/2
    xmax, ymax = +width/2,+height/2

    p1 = gmsh.model.geo.add_point(xmin, ymin, 0.0, meshSize=cellSize)
    p2 = gmsh.model.geo.add_point(xmax, ymin, 0.0, meshSize=cellSize)
    p3 = gmsh.model.geo.add_point(xmin, ymax, 0.0, meshSize=cellSize)
    p4 = gmsh.model.geo.add_point(xmax, ymax, 0.0, meshSize=cellSize)

    l1 = gmsh.model.geo.add_line(p1, p2, tag=boundaries["Bottom"])
    l2 = gmsh.model.geo.add_line(p2, p4, tag=boundaries["Right"])
    l3 = gmsh.model.geo.add_line(p4, p3, tag=boundaries["Top"])
    l4 = gmsh.model.geo.add_line(p3, p1, tag=boundaries["Left"])
    
    loops = []
    if radius > 0.0:
        p5 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=csize_circle)
        p6 = gmsh.model.geo.add_point(+radius,  0.0, 0.0, meshSize=csize_circle)
        p7 = gmsh.model.geo.add_point(-radius, 0.0, 0.0, meshSize=csize_circle)
        
        c1 = gmsh.model.geo.add_circle_arc(p6, p5, p7)
        c2 = gmsh.model.geo.add_circle_arc(p7, p5, p6)
        
        cl1 = gmsh.model.geo.add_curve_loop([c1, c2], tag=55)
        loops = [cl1] + loops

    cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
    loops = [cl] + loops

    surface = gmsh.model.geo.add_plane_surface(loops)
        
    gmsh.model.geo.synchronize()
    
    translation = [1, 0, 0, width, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(1, [boundaries["Right"]], [boundaries["Left"]], translation)

    # Add Physical groups
    
    for name, tag in boundaries.items():
        gmsh.model.add_physical_group(1, [tag], tag)
        gmsh.model.set_physical_name(1, tag, name)
        
    if radius > 0.0:
        gmsh.model.addPhysicalGroup(1, [c1, c2], 55)
        gmsh.model.setPhysicalName(1, 55, "Inclusion")

    gmsh.model.addPhysicalGroup(2, [surface], surface)
    gmsh.model.setPhysicalName(2, surface, "Elements")

    # %%
    gmsh.model.mesh.generate(2)
    gmsh.write("tmp_shear_inclusion.msh")
    gmsh.finalize()
    
# -


mesh1 = uw.discretisation.Mesh("tmp_shear_inclusion.msh", simplex=True)
mesh1.dm.view()

# +
# if uw.mpi.size == 1:

#     pvmesh = pv.read("tmp_shear_inclusion.msh")
#     pl = pv.Plotter(window_size=(500, 500))
#     pl.camera_position = "xy"

#     pl.add_mesh(
#         pvmesh,
#         "Black",
#         "wireframe",
#     )

#     pl.show()
# -

swarm = uw.swarm.Swarm(mesh=mesh1)
material = uw.swarm.SwarmVariable("M", swarm, num_components=1, proxy_continuous=False, proxy_degree=1)
swarm.populate(fill_param=1)

# +
# Define some functions on the mesh

import sympy

# Some useful coordinate stuff

x, y = mesh1.X

# relative to the centre of the inclusion
r = sympy.sqrt(x ** 2 + y ** 2)
th = sympy.atan2(y,x)

# need a unit_r_vec equivalent

inclusion_rvec = mesh1.X
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)
inclusion_unit_rvec = mesh1.vector.to_matrix(inclusion_unit_rvec)


# +
v_soln = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh1, 1, degree=1, continuous=True)
p_null = uw.discretisation.MeshVariable(r"P2",  mesh1, 1, degree=1, continuous=True)

vorticity = uw.discretisation.MeshVariable("omega", mesh1, 1, degree=1)
strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh1, 1, degree=1)
dev_stress_inv2 = uw.discretisation.MeshVariable("tau", mesh1, 1, degree=1)
yield_stress = uw.discretisation.MeshVariable("tau_y", mesh1, 1, degree=1)

node_viscosity = uw.discretisation.MeshVariable("eta", mesh1, 1, degree=1)
r_inc = uw.discretisation.MeshVariable("R", mesh1, 1, degree=1)
# -


with swarm.access(material):
    material.data[:,0] = 0.5 + 0.5 * np.sign(swarm.particle_coordinates.data[:,1])


# +
# Create NS object

stokes = uw.systems.Stokes(
    mesh1,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
    solver_name="stokes",
)

viscosity_L = sympy.Piecewise(
    (eta2, material.sym[0] < 0.0),
    (eta1, True),
)


stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh1.dim)
stokes.constitutive_model.Parameters.viscosity = viscosity_L
stokes.saddle_preconditioner = 1 / viscosity_L
stokes.penalty = 0.0

stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_atol"] = 0.01
stokes.petsc_options["snes_rtol"] = 1.0e-6


# +
nodal_strain_rate_inv2 = uw.systems.Projection(mesh1, strain_rate_inv2, solver_name="edot_II")
nodal_strain_rate_inv2.uw_function = stokes._Einv2
nodal_strain_rate_inv2.smoothing = 1.0e-3
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_tau_inv2 = uw.systems.Projection(mesh1, dev_stress_inv2, solver_name="stress_II")
nodal_tau_inv2.uw_function = 2 * stokes.constitutive_model.Parameters.viscosity * stokes._Einv2
nodal_tau_inv2.smoothing = 1.0e-3
nodal_tau_inv2.petsc_options.delValue("ksp_monitor")

yield_stress_calc = uw.systems.Projection(mesh1, yield_stress, solver_name="stress_y")
yield_stress_calc.uw_function = 0.0
yield_stress_calc.smoothing = 1.0e-3
yield_stress_calc.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh1, node_viscosity, solver_name="visc")
nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
nodal_visc_calc.smoothing = 1.0e-3
nodal_visc_calc.petsc_options.delValue("ksp_monitor")


# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

# Constant visc

stokes.penalty = 0.0
stokes.bodyforce = -10 * mesh1.CoordinateSystem.unit_e_1  # vertical force term (non-zero pressure)

hw = 1000.0 / res
surface_defn_fn = sympy.exp(-(((r - radius) / radius) ** 2) * hw)
# stokes.bodyforce -= 1.0e6 * surface_defn_fn * v_soln.sym.dot(inclusion_unit_rvec) * inclusion_unit_rvec

# Velocity boundary conditions

stokes.add_dirichlet_bc((0.0, 0.0), "Inclusion", (0, 1))
stokes.add_dirichlet_bc((1.0, 0.0), "Top", (0, 1))
stokes.add_dirichlet_bc((-1.0, 0.0), "Bottom", (0, 1))
# stokes.add_dirichlet_bc(0.0, "Left", 1)
# stokes.add_dirichlet_bc(0.0, "Right", 1)


# +
# linear solve first

stokes.solve()

# +
# This should be the same as a linear function of viscosity with depth,
# but the jacobian is different and will light up the SNES terms

viscosity_L = 1.0 + 0.1 * p_soln.sym[0]

stokes.constitutive_model.Parameters.viscosity = viscosity_L
stokes.saddle_preconditioner = 1 / viscosity_L


# +
stokes._setup_terms()
stokes.solve()

S = stokes.stress_deviator
nodal_tau_inv2.uw_function = sympy.simplify(sympy.sqrt(((S**2).trace())/2))
nodal_tau_inv2.solve()

nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
nodal_visc_calc.solve()

nodal_strain_rate_inv2.solve()

# +
# check it - NOTE - for the periodic mesh, points which have crossed the coordinate sheet are plotted somewhere 
# unexpected. This is a limitation we are stuck with for the moment.

if uw.mpi.size == 1:

    pvmesh = pv.read("tmp_shear_inclusion.msh")

    with mesh1.access():
        usol = v_soln.data.copy()
        
    pvpoints = pvmesh.points[:,0:2] 

    with mesh1.access():
        pvmesh.point_data["P"] = uw.function.evaluate(p_soln.sym[0], pvpoints)
        pvmesh.point_data["Edot"] = uw.function.evaluate(sympy.exp(-0.1 * strain_rate_inv2.fn**2), pvpoints)
        pvmesh.point_data["Str"] = uw.function.evaluate(dev_stress_inv2.fn, pvpoints)
        pvmesh.point_data["Visc"] = uw.function.evaluate(node_viscosity.fn, pvpoints)

    
    # Velocity arrows
    v_vectors = np.zeros_like(pvmesh.points)
    v_vectors[:, 0:2] = uw.function.evaluate(v_soln.fn, pvpoints)
    pvmesh.point_data["V"] = v_vectors / v_vectors.max()
    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln.coords[...]
    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    
    pl = pv.Plotter(window_size=(500, 500))

    pl.add_arrows(arrow_loc, arrow_length, mag=0.1, opacity=0.75)
    pl.camera_position = "xy"

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        # clim=[1.0,2.0],
        scalars="P",
        use_transparency=False,
        opacity=1.0,
    )

    pl.show()

# +
## Now a velocity dependence - not really physical, but tests the jacobian etc

viscosity_L = (1.0 + 0.1 * p_soln.sym[0]) * ( 1.0 - 0.1 * v_soln.sym[0]**2 )

stokes.constitutive_model.Parameters.viscosity = viscosity_L
stokes.saddle_preconditioner = 1 / viscosity_L


# +
stokes.solve()

S = stokes.stress_deviator
nodal_tau_inv2.uw_function = sympy.simplify(sympy.sqrt(((S**2).trace())/2))
nodal_tau_inv2.solve()

nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
nodal_visc_calc.solve()

nodal_strain_rate_inv2.solve()


# +
viscosity_L =  ( 1.0 - 0.0001 * (v_soln.sym[0].diff(y)+v_soln.sym[1].diff(x))**2 ) * (1.0 + 0.1 * p_soln.sym[0]) 

stokes.constitutive_model.Parameters.viscosity = viscosity_L
stokes.saddle_preconditioner = 1 / viscosity_L

# +
stokes.solve(zero_init_guess=False)

S = stokes.stress_deviator
nodal_tau_inv2.uw_function = sympy.simplify(sympy.sqrt(((S**2).trace())/2))
nodal_tau_inv2.solve()

nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
nodal_visc_calc.solve()

nodal_strain_rate_inv2.solve()

# +


steps = 5
for i in range(steps,0,-1):

    mu = 0.5
    C = 300 + (i-1) * 100
    tau_y = sympy.Max(C + mu * (stokes.p.sym[0]), 0.1)

    v_y = tau_y / (2 * stokes._Einv2 + 0.00001)
    v_l = eta2 + (eta1-eta2) * material.sym[0]
    viscosity = sympy.Min(v_y, v_l)

    stokes.constitutive_model.Parameters.viscosity = viscosity
    stokes.saddle_preconditioner = 1 / viscosity
    stokes.solve(zero_init_guess=False)



# +
S = stokes.stress_deviator
nodal_tau_inv2.uw_function = sympy.simplify(sympy.sqrt(((S**2).trace())/2))
nodal_tau_inv2.solve(zero_init_guess=False)

yield_stress_calc.uw_function = tau_y
yield_stress_calc.solve()

nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
nodal_visc_calc.solve()

nodal_strain_rate_inv2.solve()


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
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 20.0]

    pvmesh = pv.read("tmp_shear_inclusion.msh")

    with mesh1.access():
        usol = v_soln.data.copy()
        
    points = pvmesh.points[:,0:2] 

    with mesh1.access():
        pvmesh.point_data["Vmag"] = uw.function.evaluate(sympy.sqrt(v_soln.fn.dot(v_soln.fn)), points)
        pvmesh.point_data["P"] = uw.function.evaluate(p_soln.sym[0], points)
        pvmesh.point_data["Edot"] = uw.function.evaluate(strain_rate_inv2.fn, points)
        pvmesh.point_data["Y"] = uw.function.evaluate(yield_stress.fn, points)
        pvmesh.point_data["Str"] = uw.function.evaluate(dev_stress_inv2.fn, points)
        pvmesh.point_data["DStr"] = uw.function.evaluate(dev_stress_inv2.fn-yield_stress.fn, points)
        pvmesh.point_data["Visc"] = (uw.function.evaluate(node_viscosity.fn, points))
        pvmesh.point_data["D"] = pvmesh.point_data["Str"] - 2 * pvmesh.point_data["Edot"] * pvmesh.point_data["Visc"]

    v_vectors = np.zeros_like(pvmesh.points)
    v_vectors[:, 0:2] = uw.function.evaluate(v_soln.fn, points)
    pvmesh.point_data["V"] = v_vectors / v_vectors.max()

    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    # point sources at cell centres

    subsample = 10
    points = np.zeros((mesh1._centroids[::subsample].shape[0], 3))
    points[:, 0] = mesh1._centroids[::subsample, 0]
    points[:, 1] = mesh1._centroids[::subsample, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(point_cloud, vectors="V", integration_direction="both", max_steps=100)

    pl = pv.Plotter(window_size=(500, 500))

    pl.add_arrows(arrow_loc, arrow_length, mag=0.1, opacity=0.75)
    pl.camera_position = "xy"

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        # clim=[1.0,2.0],
        scalars="Edot",
        use_transparency=False,
        opacity=1.0,
    )

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)
    # pl.add_mesh(pvstream)

    # pl.remove_scalar_bar("mag")

    pl.show()