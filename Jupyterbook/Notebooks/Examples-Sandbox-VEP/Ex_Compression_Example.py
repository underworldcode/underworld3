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

# # Compression / Extension with no mesh deformation
#
# This is a rigid inclusion model so it looks a lot like Ex_Shear_Band_Plasticity_PS.py but the geometry is closer to 
# what we have seen before in various papers.
#
# The yield stress is Drucker-Prager / Von Mises ($\mu$ = 0).
#
# ## Examples:
#
# Try $C = 0.1$ and $\mu = 0$ to see highly developed shear bands
#
# Try $C = 0.05$ and $\mu = 0.5$ which does not localise as strongly but is highly non-linear nonetheless.
#

# +

C0 = 0.0001
mu0 = 0.3

expt_name = "Compression_C{C0}_mu{mu0}"
# -

import petsc4py
import underworld3 as uw
import numpy as np




options = petsc4py.PETSc.Options()
options["dm_adaptor"]= "pragmatic"



# +
import gmsh

# Mesh a 2D pipe with a circular hole

csize = 0.025
csize_inclusion = 0.01
res = csize_inclusion

width = 2.0
height = 1.0
radius = 0.25

if uw.mpi.rank == 0:

    # Generate local mesh on boss process

    gmsh.initialize()
    gmsh.model.add("Notch")
    gmsh.model.geo.characteristic_length_max = csize
             
    c0  = gmsh.model.geo.add_point(0.0, 0.0, 0.0, csize_inclusion) 
    cr1 = gmsh.model.geo.add_point(-radius, 0.0, 0.0, csize_inclusion) 
    cr2 = gmsh.model.geo.add_point(0.0, radius, 0.0, csize_inclusion) 
    cr3 = gmsh.model.geo.add_point(+radius, 0.0, 0.0, csize_inclusion) 
    cr4 = gmsh.model.geo.add_point(-radius, radius, 0.0, csize_inclusion) 
    cr5 = gmsh.model.geo.add_point(+radius, radius, 0.0, csize_inclusion) 

    cp1 = gmsh.model.geo.add_point(-width, 0.0, 0.0, csize) 
    cp2 = gmsh.model.geo.add_point(+width, 0.0, 0.0, csize) 
    cp3 = gmsh.model.geo.add_point(+width, height, 0.0, csize) 
    cp4 = gmsh.model.geo.add_point(-width, height, 0.0, csize) 


    l1 = gmsh.model.geo.add_line(cr3, cp2)
    l2 = gmsh.model.geo.add_line(cp2, cp3)
    l3 = gmsh.model.geo.add_line(cp3, cp4)
    l4 = gmsh.model.geo.add_line(cp4, cp1)
    l5 = gmsh.model.geo.add_line(cp1, cr1)

    l6 = gmsh.model.geo.add_circle_arc(cr1, c0, cr2)
    l7 = gmsh.model.geo.add_circle_arc(cr2, c0, cr3)
    # l6 = gmsh.model.geo.add_line(cr1, cr4)
    # l7 = gmsh.model.geo.add_line(cr4, cr5)
    # l8 = gmsh.model.geo.add_line(cr5, cr3)
    

    cl1 = gmsh.model.geo.add_curve_loop([l1,l2,l3,l4,l5,l6,l7])
    surf1 = gmsh.model.geo.add_plane_surface([cl1], )
    
    gmsh.model.geo.synchronize()

    gmsh.model.add_physical_group(1, [l4], -1, name="Left")
    gmsh.model.add_physical_group(1, [l2], -1, name="Right")
    gmsh.model.add_physical_group(1, [l3], -1, name="Top")
    gmsh.model.add_physical_group(1, [l1,l5], -1, name="FlatBottom")
    gmsh.model.add_physical_group(1, [l6,l7], -1, name="Hump")
    gmsh.model.add_physical_group(2, [surf1], -1, name="Elements")
    
    gmsh.model.mesh.generate(2)

    gmsh.write(f"tmp_hump.msh")
    gmsh.finalize()
# -


mesh1 = uw.discretisation.Mesh("tmp_hump.msh", 
                               useRegions=True,
                               simplex=True)
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

    pvmesh = pv.read("tmp_hump.msh")

    pl = pv.Plotter()

    points = np.zeros((mesh1._centroids.shape[0], 3))
    points[:, 0] = mesh1._centroids[:, 0]
    points[:, 1] = mesh1._centroids[:, 1]

    point_cloud = pv.PolyData(points)

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
        opacity=0.5,
    )

    #

    pl.show(cpos="xy")

# +
# Define some functions on the mesh

import sympy

# radius_fn = sympy.sqrt(mesh1.rvec.dot(mesh1.rvec)) # normalise by outer radius if not 1.0
# unit_rvec = mesh1.rvec / (1.0e-10+radius_fn)

# Some useful coordinate stuff

x, y = mesh1.X

# relative to the centre of the inclusion
r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y, x)

# need a unit_r_vec equivalent

inclusion_rvec = mesh1.X
inclusion_unit_rvec = inclusion_rvec / inclusion_rvec.dot(inclusion_rvec)
inclusion_unit_rvec = mesh1.vector.to_matrix(inclusion_unit_rvec)

# Pure shear flow

vx_ps = mesh1.N.x
vy_ps = -mesh1.N.y
# +
v_soln = uw.discretisation.MeshVariable("U", mesh1, mesh1.dim, degree=2)
t_soln = uw.discretisation.MeshVariable("T", mesh1, 1, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh1, 1, degree=1)

vorticity = uw.discretisation.MeshVariable("omega", mesh1, 1, degree=1)
strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh1, 1, degree=1)
dev_stress_inv2 = uw.discretisation.MeshVariable("tau", mesh1, 1, degree=1)
node_viscosity = uw.discretisation.MeshVariable("eta", mesh1, 1, degree=1)
r_inc = uw.discretisation.MeshVariable("R", mesh1, 1, degree=1)


# +
# Create Stokes solver object

stokes = uw.systems.Stokes(
    mesh1,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
    solver_name="stokes",
)

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh1.dim)
stokes.constitutive_model.Parameters.viscosity = 1
stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity
stokes.penalty=0.1

stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_atol"] = 1.0e-4


stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "cg"
stokes.petsc_options["fieldsplit_velocity_pc_type"]  = "mg"

stokes.petsc_options["fieldsplit_pressure_ksp_type"] = "gmres"
stokes.petsc_options["fieldsplit_pressure_pc_type"] = "mg" 


# +
nodal_strain_rate_inv2 = uw.systems.Projection(
    mesh1, strain_rate_inv2, solver_name="edot_II"
)
nodal_strain_rate_inv2.add_dirichlet_bc(1.0, "Left", 0)
nodal_strain_rate_inv2.add_dirichlet_bc(1.0, "Right", 0)
nodal_strain_rate_inv2.uw_function = stokes._Einv2
nodal_strain_rate_inv2.smoothing = 0.0e-3
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_tau_inv2 = uw.systems.Projection(mesh1, dev_stress_inv2, solver_name="stress_II")

S = stokes.stress_deviator
nodal_tau_inv2.uw_function = (
    sympy.simplify(sympy.sqrt(((S**2).trace()) / 2)) - p_soln.sym[0]
)
nodal_tau_inv2.smoothing = 0.0e-3
nodal_tau_inv2.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh1, node_viscosity, solver_name="visc")
nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
nodal_visc_calc.smoothing = 1.0e-3
nodal_visc_calc.petsc_options.delValue("ksp_monitor")


# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

# Constant visc

stokes.bodyforce = -1 * mesh1.CoordinateSystem.unit_j

hw = 1000.0 / res
hump_surface_fn = sympy.exp(-(((r - radius) / radius) ** 2) * hw)
upper_surface_fn = sympy.exp(-(((y - height)) ** 2) * hw)

stokes.bodyforce -= (
    1.0e6 * hump_surface_fn * v_soln.sym.dot(inclusion_unit_rvec) * inclusion_unit_rvec
)

# stokes.bodyforce 
p_penalty = 0.0
stokes.PF0 = p_penalty * upper_surface_fn * p_soln.sym 
stokes.saddle_preconditioner = (1 / stokes.constitutive_model.Parameters.viscosity
                                + p_penalty * upper_surface_fn)
    
# Velocity boundary conditions

# stokes.add_dirichlet_bc((0.0, 0.0), "Hump", (0, 1))
# stokes.add_dirichlet_bc((vx_ps, vy_ps), ["top", "bottom", "left", "right"], (0, 1))
stokes.add_dirichlet_bc((1.0,0.0), "Left", (0,1))
stokes.add_dirichlet_bc((-1.0,0.0), "Right", (0,1))
stokes.add_dirichlet_bc((0.0,), "FlatBottom", (1,))



# +
# linear solve first

stokes.solve(zero_init_guess=False)
# +
# Calculate surface pressure

_,_,_,_,ps_sum,_,_ = mesh1.stats(p_soln.sym[0] * upper_surface_fn)
_,_,_,_,p_sum,_,_ = mesh1.stats(p_soln.sym[0])
_,_,_,_,ps_norm,_,_ = mesh1.stats(upper_surface_fn)
_,_,_,_,p_norm,_,_ = mesh1.stats(1+0.00001 * p_soln.sym[0])

print(f"Mean Surface P - {ps_sum/p_sum}")
print(f"Mean P - {p_sum/p_norm}")


# p_calculator = uw.maths.Integral(mesh1, p_soln.sym[0] * upper_surface_fn)
# value = p_calculator.evaluate()

# # calculator.fn = upper_surface_fn
# # norm = calculator.evaluate()

# integral = value # / norm

# print(f"Average surface pressure: {integral}")

# + p_penalty * upper_surface_fn)

# stokes.solve(zero_init_guess=False)


# + tags=[]
# Approach the required value by shifting the parameters

for i in range(1):
    mu = mu0
    C = C0 # + (1 - i / 4) * 0.1
    print(f"Mu - {mu}, C = {C}")
    tau_y = sympy.Max(C + mu * stokes.p.sym[0] + 1 * sympy.sin(x * sympy.pi / (2*width))**2, 0.0001)
    viscosity = 1.0 / (2 * stokes._Einv2 / tau_y + 1.0)
    
    stokes.constitutive_model.Parameters.viscosity = viscosity
    stokes.saddle_preconditioner = (1 / stokes.constitutive_model.Parameters.viscosity
                                + p_penalty * upper_surface_fn)
    stokes.solve(zero_init_guess=False)
# -

nodal_tau_inv2.uw_function = (
    stokes.constitutive_model.Parameters.viscosity * stokes._Einv2
)
nodal_tau_inv2.solve()
nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
nodal_visc_calc.solve()
nodal_strain_rate_inv2.solve()


# +

savefile = f"output/{expt_name}.h5"
mesh1.save(savefile)
v_soln.save(savefile)
p_soln.save(savefile)
dev_stress_inv2.save(savefile)
node_viscosity.save(savefile)
mesh1.generate_xdmf(savefile)


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
        pvmesh.point_data["Vmag"] = uw.function.evaluate(
            sympy.sqrt(v_soln.sym.dot(v_soln.sym)), mesh1.data
        )
        pvmesh.point_data["P"] = uw.function.evaluate(p_soln.sym[0]-0*(1-y), mesh1.data)
        pvmesh.point_data["Edot"] = uw.function.evaluate(
            strain_rate_inv2.sym[0], mesh1.data
        )
        pvmesh.point_data["Visc"] = uw.function.evaluate(
            node_viscosity.sym[0], mesh1.data
        )
        pvmesh.point_data["Str"] = uw.function.evaluate(
            dev_stress_inv2.sym[0], mesh1.data
        )

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

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="both", max_steps=100
    )


# -

if uw.mpi.size == 1:

    pl = pv.Plotter(window_size=(1000, 500))

    pl.add_arrows(arrow_loc, arrow_length, mag=0.05, opacity=0.75)

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="Edot",
        use_transparency=False,
        opacity=1.0,
        clim=[0.0,4.0],
    )



    # pl.remove_scalar_bar("mag")

    pl.show()

if uw.mpi.size == 1:

    print(pvmesh.point_data["Visc"].min(), pvmesh.point_data["Visc"].max())




