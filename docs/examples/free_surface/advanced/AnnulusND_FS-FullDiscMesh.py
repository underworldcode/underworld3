# %% [markdown]
"""
# 🎓 AnnulusND FS-FullDiscMesh

**PHYSICS:** free_surface  
**DIFFICULTY:** advanced  
**MIGRATED:** From underworld3-documentation/Notebooks

## Description
This example has been migrated from the original UW3 documentation.
Additional documentation and parameter annotations will be added.

## Migration Notes
- Original complexity preserved
- Parameters to be extracted and annotated
- Claude hints to be added in future update
"""

# %% [markdown]
"""
## Original Code
The following is the migrated code with minimal modifications.
"""

# %%
# %%
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
import underworld3 as uw
from underworld3 import timing

import nest_asyncio
nest_asyncio.apply()

import numpy as np
import sympy


# %%

r_inf = 1.5
r_o = 1
r_i = 0.5
res = 18

cellsize = 1/res

mesh = uw.meshing.DiscInternalBoundaries(radiusUpper=r_inf, 
                                         radiusInternal=r_o, 
                                         radiusLower=r_i, 
                                         cellSize_Upper=3.0*cellsize,
                                         cellSize=cellsize,
                                         cellSize_Centre=3.0 * cellsize,
                                         qdegree=3,
                                         gmsh_verbosity=5)

r, th = mesh.CoordinateSystem.R

# %%
Vr = uw.discretisation.MeshVariable("v_r", mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True)
v = uw.discretisation.MeshVariable("v", mesh, vtype=uw.VarType.VECTOR, degree=2, continuous=True, varsymbol=r"\mathbf{v}")
p = uw.discretisation.MeshVariable("p", mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True)
T = uw.discretisation.MeshVariable(r"T", mesh, vtype=uw.VarType.SCALAR, degree=3, continuous=True)

R0 = uw.discretisation.MeshVariable("r_0", mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True)
R0c = uw.discretisation.MeshVariable("r_0c", mesh, vtype=uw.VarType.SCALAR, degree=0, continuous=False, varsymbol="{r_{0,c}}")
D = uw.discretisation.MeshVariable(r"D", mesh, vtype=uw.VarType.SCALAR, degree=0, continuous=True, varsymbol="{\cal{D}}")
Mc = uw.discretisation.MeshVariable("\cal{M}", mesh, vtype=uw.VarType.SCALAR, degree=0, continuous=False, varsymbol=r"{\cal{M}}")
t = uw.discretisation.MeshVariable(r"\tau", mesh, vtype=uw.VarType.SCALAR, degree=0, continuous=False)

Phi = uw.discretisation.MeshVariable(r"\varphi", mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True)
Gvec = uw.discretisation.MeshVariable("g", mesh, vtype=uw.VarType.VECTOR, degree=1, continuous=True)


# %%
rho_c = 2
rho_m = 1
rho_0 = 0

rho_fn = sympy.Piecewise(
    (rho_c, R0.sym[0] <= r_i),
    (rho_m, R0.sym[0] <= r_o),
    (rho_0, True))

# Smooth equivalent

# rho_fn = rho_0 + (1-sympy.tanh(20*(R0.sym[0] - r_o ))) * (rho_m - rho_0)/2 + (1-sympy.tanh(20*( R0.sym[0] - r_i ))) * (rho_c - rho_m)/2

# Sticky fluids

sticky_visc = 0.001

viscosity_fn = sympy.Piecewise(
    (sticky_visc, R0.sym[0] < r_i),
    (1.00, R0.sym[0] < r_o),
    (sticky_visc, True))


# %%
with mesh.access(R0, R0c,D, Mc):
    R0.data[:,0] = uw.function.evalf(r, R0.coords)
    R0c.data[:,0] = uw.function.evalf(r, R0c.coords)
    D.data[:,0] = uw.function.evalf((r-r_i)/(r_o-r_i), D.coords)
    Mc.data[:,0] = (D.data[:,0] <= 1) * (D.data[:,0] >= 0)


with mesh.access(T):
    gauss = sympy.exp(-100*(D.sym[0]-0.5)**2)
    T.data[:,0] = uw.function.evalf( gauss * (sympy.sin(5 * th)**3 + 0.1 * sympy.sin(3*th)**3) * Mc.sym[0], T.coords  )



# %%
# Initial deformation of upper surface 


# %%
deform_fn_u = (r / r_o) * sympy.sin(5 * th) / 300
deform_fn_l = (r / r_o) * sympy.sin(7 * th) / 300

Vr_solver = uw.systems.Poisson(mesh, Vr)
Vr_solver.constitutive_model = uw.constitutive_models.DiffusionModel
Vr_solver.constitutive_model.Parameters.diffusivity = 1

Vr_solver.add_essential_bc((0), mesh.boundaries.Upper.name)
Vr_solver.add_essential_bc((deform_fn_u), mesh.boundaries.Internal.name)
Vr_solver.add_essential_bc((deform_fn_l), mesh.boundaries.Lower.name)
#Vr_solver.add_essential_bc((0), mesh.boundaries.Lower.name)

Vr_solver.tolerance = 1.0e-4

Vr_solver.solve()

# %%
G_phi_solver = uw.systems.Poisson(mesh, Phi)
G_phi_solver.constitutive_model = uw.constitutive_models.DiffusionModel
G_phi_solver.constitutive_model.Parameters.diffusivity = 1
G_phi_solver.add_essential_bc((sympy.cos(2*th)/100), mesh.boundaries.Upper.name)
G_phi_solver.f = rho_fn + T.sym[0]
G_phi_solver.tolerance = 1.0e-5
G_phi_solver.solve()

g_solver = uw.systems.Vector_Projection(mesh, Gvec)
g_solver.uw_function = mesh.vector.gradient(Phi.sym[0])
g_solver.solve()


# %%
## Now deform the mesh using this smooth field

displacement = uw.function.evalf(Vr.sym * mesh.CoordinateSystem.unit_e_0 , mesh.X.coords)
mesh._deform_mesh(mesh.X.coords + displacement)


# %%
G_phi_solver.view()

# %%
Vr_solver.view()

# %%
if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, T.sym)
    pvmesh.point_data["Vrmag"] = vis.scalar_fn_to_pv_points(pvmesh, Vr.sym)
    pvmesh.point_data["Phi"]  = vis.scalar_fn_to_pv_points(pvmesh, Phi.sym)
    pvmesh.point_data["R0"]  = vis.scalar_fn_to_pv_points(pvmesh, R0.sym)
    pvmesh.point_data["D"]  = vis.scalar_fn_to_pv_points(pvmesh, D.sym)
    pvmesh.point_data["rho"]  = vis.scalar_fn_to_pv_points(pvmesh, rho_fn + T.sym[0])
    pvmesh.point_data["g"]  = vis.vector_fn_to_pv_points(pvmesh, Gvec.sym)
    pvmesh.point_data["Vr"]  = vis.vector_fn_to_pv_points(pvmesh, Vr.sym * mesh.CoordinateSystem.unit_e_0)
    pvmesh.point_data["V"]  = vis.vector_fn_to_pv_points(pvmesh, v.sym)


    pvmesh_t = vis.meshVariable_to_pv_mesh_object(T, alpha=0.05)
    pvmesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh_t, T.sym[0])
    pvmesh_t.point_data["R0"] = vis.scalar_fn_to_pv_points(pvmesh_t, R0c.sym[0])

    with mesh.access():
        pvmesh.cell_data["R0c"] = R0c.data[...]

    with mesh.access():
        pvmesh_t.cell_data["R0c"] = uw.function.evaluate(R0.sym, np.array(pvmesh_t.cell_centers().points[:,0:2]))

    pvmesh1 = pvmesh.threshold(value=r_o, scalars="R0c", invert=True).threshold(value=r_i, scalars="R0c", invert=False)

    pvmesh_t_inner = pvmesh_t.threshold(value=r_o, scalars="R0c", invert=True).threshold(value=r_i, scalars="R0c", invert=False)


    pl = pv.Plotter(window_size=(1000, 1000))

    # pl.add_mesh(
    #     pvmesh,
    #     edge_color="Black",
    #     show_edges=True,
    #     scalars="rho",
    #     use_transparency=False,
    #     opacity=0.5,
    # )
    
    pl.add_mesh(
        pvmesh1,
        cmap="seismic",
        edge_color="Black",
        show_edges=True,
        scalars="rho",
        use_transparency=False,
        opacity=0.1)

    pl.add_mesh(
        pvmesh_t_inner,
        cmap="seismic",
        edge_color="Black",
        show_edges=True,
        scalars="T",
        use_transparency=False,
        opacity=0.9)

    

    pl.add_mesh(pvmesh1,'Black', 'wireframe', opacity=0.25)

    pl.add_arrows(pvmesh.points, 
                   pvmesh.point_data["g"], 
                   mag=0.25, color="Green", show_scalar_bar=False)
    
    pl.camera.SetPosition(0.0, 0.0, 8)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)


    pl.show()

# %%

# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_fn
stokes.penalty = 0.0

stokes.bodyforce = Gvec.sym * (rho_fn - 1.0 * T.sym[0]) - 100 * v.sym * sympy.exp(-50*r**2) # stagnant core
stokes.add_essential_bc((0.0,0.0), mesh.boundaries.Upper.name)
stokes.add_essential_bc((0.0,0.0), mesh.boundaries.Centre.name)

stokes.petsc_options["snes_monitor"]= None
stokes.petsc_options["ksp_monitor"] = None

stokes.tolerance = 1.0e-3

# stokes.solve()

# FSSA
delta_t = uw.function.expression(R"\delta t",
                                 0.1, # stokes.estimate_dt(), 
                                 "Timestep")

Gamma = mesh.Gamma / sympy.sqrt(mesh.Gamma.dot(mesh.Gamma))

# Upper Boundary - $1/2 * delta t \Gamma \cdot v_r * \delta\rho$ 
FSSA_traction_upper = delta_t * (rho_m-rho_0) * Gamma.dot(v.sym) * Gamma / 2
stokes.add_natural_bc(FSSA_traction_upper, mesh.boundaries.Internal.name)

# Lower Boundary - $1/2 * delta t \Gamma \cdot v_r * \delta\rho$ 
FSSA_traction_upper = delta_t * (rho_m-rho_c) * Gamma.dot(v.sym) * Gamma / 2
stokes.add_natural_bc(FSSA_traction_upper, mesh.boundaries.Lower.name)

stokes.solve()


# %%

# %%
## From now on, use V_r to drive deformation. Again, create a smooth, continuous field
Vr_solver._reset()
Vr_solver.add_essential_bc((0), mesh.boundaries.Upper.name)
Vr_solver.add_essential_bc((v.sym.dot(mesh.CoordinateSystem.unit_e_0)), mesh.boundaries.Internal.name)
Vr_solver.add_essential_bc((v.sym.dot(mesh.CoordinateSystem.unit_e_0)), mesh.boundaries.Lower.name)
Vr_solver.solve()

# %%
pl = pv.Plotter(window_size=(1000, 1000))
pl.clear()


def plot_V_mesh(filename):

    if uw.mpi.size == 1:
        
        import pyvista as pv
        import underworld3.visualisation as vis
    
        pvmesh = vis.mesh_to_pv_mesh(mesh)
        with mesh.access():
            pvmesh.cell_data["R0c"] = R0c.data[...]

        pvmesh_inner = pvmesh.threshold(value=r_o, scalars="R0c", invert=True).threshold(value=r_i, scalars="R0c", invert=False)
       
        pvmesh_inner.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh_inner, v.sym)
        pvmesh_inner.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh_inner, v.sym.dot(v.sym))
        pvmesh_inner.point_data["Vr"] = vis.scalar_fn_to_pv_points(pvmesh_inner, v.sym.dot(mesh.CoordinateSystem.unit_e_0 ))
        pvmesh_inner.point_data["M"] = vis.scalar_fn_to_pv_points(pvmesh_inner, Mc.sym)
        pvmesh_inner.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh_inner, T.sym)
    

        velocity_points = vis.meshVariable_to_pv_cloud(v)
        velocity_points.point_data["R0"] = vis.scalar_fn_to_pv_points(velocity_points, R0.sym[0])
    
        velocity_points_inner = velocity_points.threshold(value=r_o, scalars="R0", invert=True).threshold(value=r_i, scalars="R0", invert=False)

        velocity_points_inner.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points_inner, v.sym * Mc.sym[0])
        velocity_points_inner.point_data["Vr"] = vis.vector_fn_to_pv_points(velocity_points_inner, Vr.sym[0] * Mc.sym[0]  * mesh.CoordinateSystem.unit_e_0 )

        # function.evaluate blows up at some point (uh oh)
        # 
        
        pvmesh_t = vis.meshVariable_to_pv_mesh_object(T, alpha=0.05)
        pvmesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh_t, T.sym[0])

        with mesh.access():
            pvmesh_t.cell_data["R0c"] = uw.function.evaluate(R0c.sym, np.array(pvmesh_t.cell_centers().points[:,0:2]))

        pvmesh_t_inner = pvmesh_t.threshold(value=r_o, scalars="R0c", invert=True).threshold(value=r_i, scalars="R0c", invert=False)
        
    
        pvstream = pvmesh_inner.streamlines_from_source(
            pvmesh_inner.cell_centers(),
            vectors="V", 
            integration_direction="forward", 
            integrator_type=2,
            surface_streamlines=True,
            initial_step_length=0.1,
            max_time=0.1,
            max_steps=100)
        
    
        pl.add_mesh(
            pvmesh_t_inner,
            cmap="coolwarm",
            edge_color="Black",
            edge_opacity=0.25,
            scalars="T",
            show_edges=True,
            use_transparency=False,
            opacity=1.0,
            clim=[-1.1,1.1],
            show_scalar_bar=False)
    
        pl.add_mesh(pvmesh, style="wireframe", color="Grey", opacity=0.05)
            
        
        # pl.add_arrows(velocity_points_inner.points, 
        #               velocity_points_inner.point_data["V"], 
        #               mag=10, color="Green", show_scalar_bar=False)
        
        # pl.add_mesh(pvstream, 
        #             show_scalar_bar=False)
    
        pl.camera.SetPosition(0.0, 0.0, 5)
        pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
        pl.camera.SetClippingRange(1.0, 8.0)
    
    
        # pl.camera_position = "xz"
        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(1000, 1000),
            return_img=False)
    
        pl.clear()


# %%
ts = 0
time=0.0
outdir = "output"
expt_name = "AnnulusFreeSurface_Relax_4"

delta_t.sym = stokes.estimate_dt() / 3


# %%
## Relaxation loop

with mesh.access(T):
    gauss = sympy.exp(-50*(D.sym[0]-0.5)**2)
    T.data[:,0] = uw.function.evalf( gauss * (sympy.sin(5 * th)**3 + 0.1 * sympy.sin(3*th))**3 * Mc.sym[0], T.coords  )


for step in range(0,10):

    # Smooth deformation function
    Vr_solver.solve(zero_init_guess=False)

    # Gravitational potential
    G_phi_solver.solve()
    g_solver.solve()

    delta_t.sym = stokes.estimate_dt() / 3
    displacement = delta_t.sym * uw.function.evalf(Vr.sym * mesh.CoordinateSystem.unit_e_0, mesh.X.coords)

    print(f"ts: {ts} / Displacement - Amplitude: {np.abs(displacement).max()}")

    mesh._deform_mesh(mesh.X.coords + displacement)
    
    stokes.solve(zero_init_guess=False, _force_setup=True)

    if ts%1 == 0:
        plot_V_mesh(filename=f"{outdir}/{expt_name}.{ts:05d}")

    ts += 1
    time += delta_t

# with mesh.access(T):
#     T.data[:,0] = 0.0

# for step in range(0,25):

#     # Smooth deformation function
#     Vr_solver.solve(zero_init_guess=False)

#     # Gravitational potential
#     G_phi_solver.solve()
#     g_solver.solve()

#     delta_t.sym = stokes.estimate_dt() / 2
#     displacement = delta_t.sym * uw.function.evalf(Vr.sym * mesh.CoordinateSystem.unit_e_0, mesh.X.coords)

#     print(f"ts: {ts} / Displacement - Amplitude: {np.abs(displacement).max()}")

#     mesh._deform_mesh(mesh.X.coords + displacement)
    
#     stokes.solve(zero_init_guess=False, _force_setup=True)

#     if ts%1 == 0:
#         plot_V_mesh(filename=f"{outdir}/{expt_name}.{ts:05d}")

#     ts += 1
#     time += delta_t


# with mesh.access(T):
#     gauss = sympy.exp(-50*(D.sym[0]-0.5)**2)
#     T.data[:,0] = -uw.function.evalf( gauss * (sympy.sin(5 * th)**3 + 0.1 * sympy.sin(3*th))**3 * Mc.sym[0], T.coords  )


# for step in range(0,25):

#     # Smooth deformation function
#     Vr_solver.solve(zero_init_guess=False)

#     # Gravitational potential
#     G_phi_solver.solve()
#     g_solver.solve()

#     delta_t.sym = stokes.estimate_dt() / 2
#     displacement = delta_t.sym * uw.function.evalf(Vr.sym * mesh.CoordinateSystem.unit_e_0, mesh.X.coords)

#     print(f"ts: {ts} / Displacement - Amplitude: {np.abs(displacement).max()}")

#     mesh._deform_mesh(mesh.X.coords + displacement)
    
#     stokes.solve(zero_init_guess=False, _force_setup=True)

#     if ts%1 == 0:
#         plot_V_mesh(filename=f"{outdir}/{expt_name}.{ts:05d}")

#     ts += 1
#     time += delta_t


# with mesh.access(T):
#     T.data[:,0] = 0.0

# for step in range(0,25):

#     # Smooth deformation function
#     Vr_solver.solve(zero_init_guess=False)

#     # Gravitational potential
#     G_phi_solver.solve()
#     g_solver.solve()

#     delta_t.sym = stokes.estimate_dt() / 2
#     displacement = delta_t.sym * uw.function.evalf(Vr.sym * mesh.CoordinateSystem.unit_e_0, mesh.X.coords)

#     print(f"ts: {ts} / Displacement - Amplitude: {np.abs(displacement).max()}")

#     mesh._deform_mesh(mesh.X.coords + displacement)
    
#     stokes.solve(zero_init_guess=False, _force_setup=True)

#     if ts%1 == 0:
#         plot_V_mesh(filename=f"{outdir}/{expt_name}.{ts:05d}")

#     ts += 1
#     time += delta_t



# %%
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    with mesh.access():
            pvmesh.cell_data["R0c"] = R0c.data[...]

    pvmesh_inner = pvmesh.threshold(value=r_o, scalars="R0c", invert=True).threshold(value=r_i, scalars="R0c", invert=False)

    pvmesh_inner.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh_inner, v.sym)
    pvmesh_inner.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh_inner, v.sym.dot(v.sym))
    pvmesh_inner.point_data["Vr"] = vis.scalar_fn_to_pv_points(pvmesh_inner, v.sym.dot(mesh.CoordinateSystem.unit_e_0 ))
    pvmesh_inner.point_data["M"] = vis.scalar_fn_to_pv_points(pvmesh_inner, Mc.sym)
    pvmesh_inner.point_data["rho"] = vis.scalar_fn_to_pv_points(pvmesh_inner, rho_fn + T.sym[0])
    pvmesh_inner.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh_inner, T.sym[0])

    velocity_points = vis.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["R0"] = vis.scalar_fn_to_pv_points(velocity_points, R0.sym[0])
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym * Mc.sym[0])

    velocity_points_inner = velocity_points.threshold(value=r_o, scalars="R0", invert=True).threshold(value=r_i, scalars="R0", invert=False)
    velocity_points_inner.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points_inner, v.sym * Mc.sym[0])
    velocity_points_inner.point_data["Vr"] = vis.vector_fn_to_pv_points(velocity_points_inner, Vr.sym[0] * Mc.sym[0]  * mesh.CoordinateSystem.unit_e_0 )

    pvmesh_t = vis.meshVariable_to_pv_mesh_object(T, alpha=0.0)
    pvmesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh_t, T.sym[0])

    tri_centres = np.ascontiguousarray(pvmesh_t.cell_centers().points[:,0:2])
    
    cells = mesh.get_closest_cells(tri_centres)

    with mesh.access():
        pvmesh_t.cell_data["R0c"] = R0c.data[cells,0]

    pvmesh_t_inner = pvmesh_t.threshold(value=r_o, scalars="R0c", invert=True).threshold(value=r_i, scalars="R0c", invert=False)

    
    point_cloud = pvmesh_inner.cell_centers()

    pvstream = pvmesh_inner.streamlines_from_source(
        point_cloud, vectors="V", 
        integration_direction="forward", 
        integrator_type=2,
        surface_streamlines=True,
        initial_step_length=0.1,
        max_time=0.2,
        max_steps=1000)
    
    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
                pvmesh_t_inner,
                cmap="coolwarm",
                edge_color="Black",
                scalars="T",
                show_edges=False,
                use_transparency=False,
                opacity=1,
                interpolate_before_map=True,
                show_scalar_bar=False)

    pl.add_mesh(pvmesh_t_inner, style="wireframe", color="Grey", opacity=0.2)
    pl.add_mesh(pvmesh, style="wireframe", color="Grey", opacity=0.2)
        
    
    # pl.add_arrows(velocity_points_inner.points, 
    #               velocity_points_inner.point_data["V"], 
    #               mag=100, color="Green")
    
    pl.add_mesh(pvstream, opacity=0.5, show_scalar_bar=False)

    pl.show(cpos="xy")

# %%
pl.screenshot(filename="RelaxingMesh.png", return_img=False,  scale=2 )

# %%
cells = mesh.get_closest_cells(np.ascontiguousarray(pvmesh_t.cell_centers().points[:,0:2]))


# %%
np.ascontiguousarray()
