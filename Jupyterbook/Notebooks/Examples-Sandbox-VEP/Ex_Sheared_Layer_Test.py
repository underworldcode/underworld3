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

# # Validate constitutive models
#
# Simple shear with material defined by particle swarm (based on inclusion model), position, pressure, strain rate etc. 
# Check the implementation of the Jacobians using various non-linear terms.
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
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
# Mesh a 2D pipe with a circular hole

csize = resolution
csize_circle = resolution * 0.5
res = csize
cellSize = csize

width = 3.0
height = 1.0
radius = 0.0

eta1 = 1000
eta2 = 1

from enum import Enum

## NOTE: stop using pygmsh, then we can just define boundary labels ourselves and not second guess pygmsh

class boundaries(Enum):
    Bottom = 1
    Right = 3
    Top = 2
    Left  = 4
    Inclusion = 5
    All_Boundaries = 1001 



if uw.mpi.rank == 0:

    import gmsh

    gmsh.initialize()
    gmsh.model.add("Periodic x")


    xmin, ymin = -width / 2, -height / 2
    xmax, ymax = +width / 2, +height / 2

    p1 = gmsh.model.geo.add_point(xmin, ymin, 0.0, meshSize=cellSize)
    p2 = gmsh.model.geo.add_point(xmax, ymin, 0.0, meshSize=cellSize)
    p3 = gmsh.model.geo.add_point(xmin, ymax, 0.0, meshSize=cellSize)
    p4 = gmsh.model.geo.add_point(xmax, ymax, 0.0, meshSize=cellSize)

    l1 = gmsh.model.geo.add_line(p1, p2, tag=boundaries["Bottom"].value)
    l2 = gmsh.model.geo.add_line(p2, p4, tag=boundaries["Right"].value)
    l3 = gmsh.model.geo.add_line(p4, p3, tag=boundaries["Top"].value)
    l4 = gmsh.model.geo.add_line(p3, p1, tag=boundaries["Left"].value)

    loops = []
    if radius > 0.0:
        p5 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=csize_circle)
        p6 = gmsh.model.geo.add_point(+radius, 0.0, 0.0, meshSize=csize_circle)
        p7 = gmsh.model.geo.add_point(-radius, 0.0, 0.0, meshSize=csize_circle)

        c1 = gmsh.model.geo.add_circle_arc(p6, p5, p7)
        c2 = gmsh.model.geo.add_circle_arc(p7, p5, p6)

        cl1 = gmsh.model.geo.add_curve_loop([c1, c2], tag=55)
        loops = [cl1] + loops

    cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
    loops = [cl] + loops

    surface = gmsh.model.geo.add_plane_surface(loops, tag=99999)

    gmsh.model.geo.synchronize()

    # translation = [1, 0, 0, width, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    # gmsh.model.mesh.setPeriodic(
    #     1, [boundaries["Right"]], [boundaries["Left"]], translation
    # )

    # Add Physical groups

    for bd in boundaries:
        print(bd.value, flush=True)
        print(bd.name, flush=True)
        gmsh.model.add_physical_group(1, [bd.value], bd.value)
        gmsh.model.set_physical_name(1, bd.value, bd.name)

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




# +
mesh1 = uw.discretisation.Mesh("tmp_shear_inclusion.msh", 
                               simplex=True, markVertices=True, 
                               useRegions=True, boundaries=boundaries
                              )
mesh1.dm.view()

## build periodic mesh (mesh1)
# uw.cython.petsc_discretisation.petsc_dm_set_periodicity(
#     mesh1.dm, [0.1, 0.0], [-1.5, 0.0], [1.5, 0.0])

# mesh1.dm.view()

# +

swarm = uw.swarm.Swarm(mesh=mesh1, recycle_rate=5)

material = uw.swarm.SwarmVariable(
    "M", swarm, size=1, 
    proxy_continuous=True, proxy_degree=2, dtype=int,
)

strain_p = uw.swarm.SwarmVariable(
    "Strain_p", swarm, size=1, 
    proxy_continuous=True, 
    proxy_degree=2, varsymbol=r"{\varepsilon_{p}}", dtype=float,
)

stress_dt = uw.swarm.SwarmVariable(r"Stress_p", swarm, (2,2), vtype=uw.VarType.SYM_TENSOR, varsymbol=r"{\sigma^{*}_{p}}")

swarm.populate(fill_param=2)

# +
# Define some functions on the mesh

import sympy

# Some useful coordinate stuff

x, y = mesh1.X

# relative to the centre of the inclusion
r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y, x)

# need a unit_r_vec equivalent

inclusion_rvec = mesh1.X
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)
inclusion_unit_rvec = mesh1.vector.to_matrix(inclusion_unit_rvec)


# +
v_soln = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh1, 1, degree=1, continuous=True)
work   = uw.discretisation.MeshVariable(r"W", mesh1, 1, degree=1, continuous=False)
Stress = uw.discretisation.MeshVariable(r"{\sigma}", mesh1, (2,2), vtype=uw.VarType.SYM_TENSOR, degree=1, 
                                         continuous=False, varsymbol=r"{\sigma}")

vorticity = uw.discretisation.MeshVariable("omega", mesh1, 1, degree=1)
strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh1, 1, degree=2)
strain_rate_inv2_p = uw.discretisation.MeshVariable("eps_p", mesh1, 1, degree=2, varsymbol=r"\dot\varepsilon_p")
dev_stress_inv2 = uw.discretisation.MeshVariable("tau", mesh1, 1, degree=2)
yield_stress = uw.discretisation.MeshVariable("tau_y", mesh1, 1, degree=1)

node_viscosity = uw.discretisation.MeshVariable("eta", mesh1, 1, degree=1)
r_inc = uw.discretisation.MeshVariable("R", mesh1, 1, degree=1)
# -




# +
# Set the initial strain from the mesh 

with mesh1.access(): #strain_rate_inv2_p):
    XX = v_soln.coords[:,0]
    YY = v_soln.coords[:,1]
    mask = (1.0 - (YY * 2)**8) * (1 -  (2*XX/3)**6)
    # strain_rate_inv2_p.data[:,0] = 2.0 * np.floor(0.033+np.random.random(strain_rate_inv2_p.coords.shape[0])) * mask

# +
   
with swarm.access(material, strain_p), mesh1.access():
    strain_p.data[:] = strain_rate_inv2_p.rbf_interpolate(swarm.particle_coordinates.data)
    strain_array = strain_rate_inv2_p.rbf_interpolate(swarm.particle_coordinates.data)
# +
   
with swarm.access(strain_p, material), mesh1.access():
    XX = swarm.particle_coordinates.data[:,0]
    YY = swarm.particle_coordinates.data[:,1]
    mask = (1.0 - (YY * 2)**8) * (1 -  (2*XX/3)**6)
    material.data[(XX**2 + YY**2 < 0.01), 0] = 1
    strain_p.data[:,0] = 0.0 * np.random.random(swarm.particle_coordinates.data.shape[0]) * mask
# -



# +
# Create Solver object

stokes = uw.systems.Stokes(
    mesh1,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=True,
    solver_name="stokes",
)

eta1 = 1000
eta2 = 1

viscosity_L = sympy.Piecewise(
    (eta2, material.sym[0] > 0.5),
    (eta1, True),
)

# -
stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
stokes.constitutive_model.Parameters.bg_viscosity = viscosity_L
# stokes.constitutive_model.Parameters.sigma_star_fn 


stokes.constitutive_model.viscosity

stokes.constitutive_model

stokes.constitutive_model.Parameters.shear_modulus = 1.0
stokes.constitutive_model.Parameters.dt_elastic = 0.1

stokes.constitutive_model

stokes.stress_deviator_1d

sigma_projector = uw.systems.Tensor_Projection(mesh1, tensor_Field=Stress, scalar_Field=work  )
sigma_projector.uw_function = stokes.stress_1d

# +
nodal_strain_rate_inv2 = uw.systems.Projection(
    mesh1, strain_rate_inv2, solver_name="edot_II"
)
nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
nodal_strain_rate_inv2.smoothing = 1.0e-3
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_tau_inv2 = uw.systems.Projection(mesh1, dev_stress_inv2, solver_name="stress_II")
nodal_tau_inv2.uw_function = (
    2 * stokes.constitutive_model.viscosity * stokes.Unknowns.Einv2
)
nodal_tau_inv2.smoothing = 1.0e-3
nodal_tau_inv2.petsc_options.delValue("ksp_monitor")

yield_stress_calc = uw.systems.Projection(mesh1, yield_stress, solver_name="stress_y")
yield_stress_calc.uw_function = 0.0
yield_stress_calc.smoothing = 1.0e-3
yield_stress_calc.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh1, node_viscosity, solver_name="visc")
nodal_visc_calc.uw_function = stokes.constitutive_model.viscosity
nodal_visc_calc.smoothing = 1.0e-3
nodal_visc_calc.petsc_options.delValue("ksp_monitor")



# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

# Constant visc

stokes.penalty = 1.0
stokes.bodyforce = (
    -0.00000001 * mesh1.CoordinateSystem.unit_e_1.T 
)  # vertical force term (non-zero pressure)

stokes.tolerance = 1.0e-4

# stokes.bodyforce -= 1.0e6 * surface_defn_fn * v_soln.sym.dot(inclusion_unit_rvec) * inclusion_unit_rvec

# Velocity boundary conditions

if radius > 0.0:
    stokes.add_dirichlet_bc((0.0, 0.0), "Inclusion")
    
stokes.add_dirichlet_bc((1.0, 0.0), "Top")
stokes.add_dirichlet_bc((-1.0, 0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")

# -


mesh1.dm.view()

stokes._setup_pointwise_functions()
stokes._setup_discretisation(verbose=True)
stokes._setup_solver()

stokes.solve()

sigma_projector.solve()

with mesh1.access():
    print(Stress[0,0].data.max())
    print(Stress[1,1].data.max())
    print(Stress[0,1].data.max())


# +
# Now add yield without pressure dependence

eps_ref = sympy.sympify(1)
scale = sympy.sympify(25)
C0 = 2500
Cinf = 500

# C = 2 * (y * 2)**16 + 5.0 * sympy.exp(-(strain.sym[0]/0.1)**2) + 0.1
C = 2 * (y * 2)**2 + (C0-Cinf) * (1 - sympy.tanh((strain_p.sym[0]/eps_ref - 1)*scale) ) / 2 + Cinf

stokes.constitutive_model.Parameters.yield_stress = C + mu * p_soln.sym[0]
stokes.constitutive_model.Parameters.edot_II_fn = stokes.Unknowns.Einv2
stokes.constitutive_model.Parameters.min_viscosity = 0.1
stokes.saddle_preconditioner = 1 / stokes.constitutive_model.viscosity


# stokes.solve(zero_init_guess=False, picard=2)

stokes.constitutive_model
# -

stokes.constitutive_model.stress_projection()

stokes.constitutive_model.flux

0/0

with mesh1.access():
    print(p_soln.data.min(), p_soln.data.max())

# +
timing.reset()
timing.start()

stokes.snes.setType("newtontr")
stokes.solve(zero_init_guess=False, picard = -1)

timing.print_table(display_fraction=1)
# -


stokes._u_f1

# +

S = stokes.stress_deviator
nodal_tau_inv2.uw_function = sympy.simplify(sympy.sqrt(((S**2).trace()) / 2))
nodal_tau_inv2.solve()

yield_stress_calc.uw_function = stokes.constitutive_model.Parameters.yield_stress
yield_stress_calc.solve()

nodal_visc_calc.uw_function = sympy.log(stokes.constitutive_model.Parameters.viscosity)
nodal_visc_calc.solve()

nodal_strain_rate_inv2.uw_function = (sympy.Max(0.0, stokes._Einv2 - 
                        0.5 * stokes.constitutive_model.Parameters.yield_stress / stokes.constitutive_model.Parameters.bg_viscosity))
nodal_strain_rate_inv2.solve()

with mesh1.access(strain_rate_inv2_p):
    strain_rate_inv2_p.data[...] = strain_rate_inv2.data.copy()

nodal_strain_rate_inv2.uw_function = stokes._Einv2
nodal_strain_rate_inv2.solve()

# -
mesh0 = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.5,-0.5),
    maxCoords=(+1.5,+0.5),
    cellSize=0.033,
)


# +
# check it - NOTE - for the periodic mesh, points which have crossed the coordinate sheet are plotted somewhere
# unexpected. This is a limitation we are stuck with for the moment.

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)

    pvpoints = pvmesh.points[:, 0:2]
    usol = v_soln.rbf_interpolate(pvpoints)

    pvmesh.point_data["P"] = p_soln.rbf_interpolate(pvpoints)
    pvmesh.point_data["Edot"] = strain_rate_inv2.rbf_interpolate(pvpoints)**2
    pvmesh.point_data["Edotp"] = strain_rate_inv2_p.rbf_interpolate(pvpoints)
    pvmesh.point_data["Strs"] = dev_stress_inv2.rbf_interpolate(pvpoints)
    pvmesh.point_data["StrY"] =  yield_stress.rbf_interpolate(pvpoints)
    pvmesh.point_data["dStrY"] = pvmesh.point_data["Strs"] - pvmesh.point_data["StrY"]
    pvmesh.point_data["Visc"] = np.exp(node_viscosity.rbf_interpolate(pvpoints))
    pvmesh.point_data["Mat"] = material.rbf_interpolate(pvpoints)
    pvmesh.point_data["Strn"] = strain._meshVar.rbf_interpolate(pvpoints)

    # Velocity arrows
    
    v_vectors = np.zeros_like(pvmesh.points)
    v_vectors[:, 0:2] = v_soln.rbf_interpolate(pvpoints)
        
    # Points (swarm)
    
    with swarm.access():
        points = np.zeros((swarm.data.shape[0], 3))
        points[:, 0] = swarm.data[:,0]
        points[:, 1] = swarm.data[:,1]
        point_cloud = pv.PolyData(points)
        point_cloud.point_data["strain"] = strain.data[:,0]

    pl = pv.Plotter(window_size=(500, 500))

    # pl.add_arrows(pvmesh.points, v_vectors, mag=0.1, opacity=0.75)
    # pl.camera_position = "xy"

    pl.add_mesh(
        pvmesh,
        cmap="Blues",
        edge_color="Grey",
        show_edges=True,
        # clim=[0.0,1.0],
        scalars="P",
        use_transparency=False,
        opacity=0.5,
    )
    
    # pl.add_points(point_cloud, colormap="coolwarm", scalars="strain", point_size=10.0, opacity=0.5)
    
    pl.camera.SetPosition(0.0, 0.0, 3.0)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)
        
    pl.show()
# +
# Now add elasticity
# -
stokes.constitutive_model.Parameters.yield_stress.subs(((strain.sym[0],0.25), (y,0.0)))

stokes.constitutive_model.Parameters.viscosity


def return_points_to_domain(coords):
    new_coords = coords.copy()
    new_coords[:,0] = (coords[:,0] + 1.5)%3 - 1.5
    return new_coords


ts = 0

# +
delta_t = stokes.estimate_dt()

expt_name = f"output/shear_band_sw_nonp_{mu}"

for step in range(0, 10):
    
    stokes.solve(zero_init_guess=False)
    
    nodal_strain_rate_inv2.uw_function = (sympy.Max(0.0, stokes._Einv2 - 
                       0.5 * stokes.constitutive_model.Parameters.yield_stress / stokes.constitutive_model.Parameters.bg_viscosity))
    nodal_strain_rate_inv2.solve()

    with mesh1.access(strain_rate_inv2_p):
        strain_rate_inv2_p.data[...] = strain_rate_inv2.data.copy()
        
    nodal_strain_rate_inv2.uw_function = stokes._Einv2
    nodal_strain_rate_inv2.solve()
    
    with swarm.access(strain), mesh1.access():
        XX = swarm.particle_coordinates.data[:,0]
        YY = swarm.particle_coordinates.data[:,1]
        mask =  (2*XX/3)**4 # * 1.0 - (YY * 2)**8 
        strain.data[:,0] +=  delta_t * mask * strain_rate_inv2_p.rbf_interpolate(swarm.data)[:,0] - 0.1 * delta_t
        strain_dat = delta_t * mask *  strain_rate_inv2_p.rbf_interpolate(swarm.data)[:,0]
        print(f"dStrain / dt = {delta_t * (mask * strain_rate_inv2_p.rbf_interpolate(swarm.data)[:,0]).mean()}, {delta_t}")
        
    mesh1.write_timestep_xdmf(f"{expt_name}", 
                         meshUpdates=False,
                         meshVars=[p_soln,v_soln,strain_rate_inv2_p], 
                         swarmVars=[strain],
                         index=ts)
    
    swarm.save(f"{expt_name}.swarm.{ts}.h5")
    strain.save(f"{expt_name}.strain.{ts}.h5")

    # Update the swarm locations
    swarm.advection(v_soln.sym, delta_t=delta_t, 
                 restore_points_to_domain_func=None) 
    
    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))
        
    ts += 1

# -




# +
nodal_visc_calc.uw_function = sympy.log(stokes.constitutive_model.Parameters.viscosity)
nodal_visc_calc.solve()

yield_stress_calc.uw_function = stokes.constitutive_model.Parameters.yield_stress
yield_stress_calc.solve()

nodal_tau_inv2.uw_function = 2 * stokes.constitutive_model.Parameters.viscosity * stokes._Einv2
nodal_tau_inv2.solve()

# +
# check it - NOTE - for the periodic mesh, points which have crossed the coordinate sheet are plotted somewhere
# unexpected. This is a limitation we are stuck with for the moment.

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh1)

    pvpoints = pvmesh.points[:, 0:2]
    usol = v_soln.rbf_interpolate(pvpoints)

    pvmesh.point_data["P"] = p_soln.rbf_interpolate(pvpoints)
    pvmesh.point_data["Edot"] = strain_rate_inv2.rbf_interpolate(pvpoints)
    pvmesh.point_data["Visc"] = np.exp(node_viscosity.rbf_interpolate(pvpoints))
    pvmesh.point_data["Edotp"] = strain_rate_inv2_p.rbf_interpolate(pvpoints)
    pvmesh.point_data["Strs"] = dev_stress_inv2.rbf_interpolate(pvpoints)
    pvmesh.point_data["StrY"] =  yield_stress.rbf_interpolate(pvpoints)
    pvmesh.point_data["dStrY"] = pvmesh.point_data["StrY"] - 2 *  pvmesh.point_data["Visc"] * pvmesh.point_data["Edot"] 
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
        points[:, 0] = swarm.data[plot_points[0],0]
        points[:, 1] = swarm.data[plot_points[0],1]
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
    
 
    pl.add_points(point_cloud, 
                  colormap="Oranges", scalars="strain",
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



with swarm.access():
    print(strain.data.max())

strain_rate_inv2_p.rbf_interpolate(mesh1.data).max()



# ## 

mesh1._search_lengths


