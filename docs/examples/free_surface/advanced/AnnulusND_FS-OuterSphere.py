# %% [markdown]
"""
# 🎓 AnnulusND FS-OuterSphere

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
res = 20

cellsize = 1/res

mesh = uw.meshing.AnnulusInternalBoundary(radiusOuter=r_inf, radiusInternal=r_o, radiusInner=r_i, 
                                          cellSize_Outer=3.0*cellsize,
                                          cellSize=cellsize, qdegree=3)

r, th = mesh.CoordinateSystem.R

# %%
Vr = uw.discretisation.MeshVariable("Vr", mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True, varsymbol=r"{v_r}")
v = uw.discretisation.MeshVariable("V", mesh, vtype=uw.VarType.VECTOR, degree=2, continuous=True, varsymbol=r"\mathbf{v}")
p = uw.discretisation.MeshVariable("P", mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True, varsymbol=r"p")
M = uw.discretisation.MeshVariable("M", mesh, vtype=uw.VarType.SCALAR, degree=0, continuous=False, varsymbol=r"{\cal{M}}")
R0 = uw.discretisation.MeshVariable("r0", mesh, vtype=uw.VarType.SCALAR, degree=1, continuous=True, varsymbol=r"{r_0}")
R0c = uw.discretisation.MeshVariable("r0c", mesh, vtype=uw.VarType.SCALAR, degree=0, continuous=False, varsymbol=r"{r_{0,c}}")


# %%
layer_fn = sympy.Piecewise(
    (1, r <= r_o),
    (0, True)
)

with mesh.access(M):
    M.data[:,0] = uw.function.evalf(layer_fn, M.coords)

with mesh.access(R0):
    R0.data[:,0] = uw.function.evalf(r, R0.coords)

with mesh.access(R0c):
    R0c.data[:,0] = uw.function.evalf(r, R0c.coords)



# %%
# deformation of upper surface


# %%
deform_fn = (r / r_o) * sympy.sin(10 * th) / 20

diffuser = uw.systems.Poisson(mesh, Vr)
diffuser.constitutive_model = uw.constitutive_models.DiffusionModel
diffuser.constitutive_model.Parameters.diffusivity = 1

diffuser.add_essential_bc((0), mesh.boundaries.Upper.name)
diffuser.add_essential_bc((deform_fn), mesh.boundaries.Internal.name)
diffuser.add_essential_bc((0), mesh.boundaries.Lower.name)

diffuser.tolerance = 1.0e-3

diffuser.solve()

# %%
## Now deform the mesh using this smooth field

displacement = uw.function.evalf(Vr.sym * mesh.CoordinateSystem.unit_e_0 , mesh.data)
mesh._deform_mesh(mesh.data + displacement)


# %%
if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["Vr"] = vis.scalar_fn_to_pv_points(pvmesh, Vr.sym)
    pvmesh.point_data["R0"]  = vis.scalar_fn_to_pv_points(pvmesh, R0.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(
        pvmesh,
        cmap="seismic",
        edge_color="Black",
        show_edges=False,
        scalars="R0",
        use_transparency=False,
        opacity=0.9)

    

    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)

    pl.show()

# %%

# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 0.01 + M.sym[0]
stokes.penalty = 0.0

stokes.bodyforce = -mesh.CoordinateSystem.unit_e_0 * M.sym[0]
stokes.add_essential_bc((0.0,0.0), mesh.boundaries.Lower.name)
stokes.add_essential_bc((0.0,0.0), mesh.boundaries.Upper.name)

stokes.petsc_options["snes_monitor"]= None
stokes.petsc_options["ksp_monitor"] = None

stokes.solve()

# FSSA
delta_t = uw.function.expression(R"\delta t", 0.5 * stokes.estimate_dt(), "Timestep")

Gamma = mesh.Gamma / sympy.sqrt(mesh.Gamma.dot(mesh.Gamma))
FSSA_traction = delta_t*Gamma.dot(v.sym) * Gamma / 2
stokes.add_natural_bc(FSSA_traction, mesh.boundaries.Internal.name)

stokes.solve()


# %%
## From now on, use V_r to drive deformation. Again, create a smooth, continuous field
diffuser._reset()
diffuser.add_essential_bc((0), mesh.boundaries.Upper.name)
diffuser.add_essential_bc(v.sym.dot(mesh.CoordinateSystem.unit_e_0), mesh.boundaries.Internal.name)
diffuser.add_essential_bc((0), mesh.boundaries.Lower.name)
diffuser.solve()

# %%
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v.sym.dot(v.sym))
    pvmesh.point_data["Vr"] = vis.scalar_fn_to_pv_points(pvmesh, Vr.sym[0])
    pvmesh.point_data["R0"] = vis.scalar_fn_to_pv_points(pvmesh,R0.sym[0])

    velocity_points = vis.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym)
    velocity_points.point_data["Vr"] = vis.vector_fn_to_pv_points(velocity_points, Vr.sym[0] * mesh.CoordinateSystem.unit_e_0 )

    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", 
        integration_direction="forward", 
        integrator_type=2,
        surface_streamlines=True,
        initial_step_length=0.1,
        max_time=0.1,
        max_steps=100
    )
    
    pl = pv.Plotter(window_size=(750, 750))

    
    pl.add_mesh(
                pvmesh,
                cmap="coolwarm",
                edge_color="Black",
                scalars="R0",
                show_edges=True,
                use_transparency=False,
                opacity=0.75)
    
    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], 
                  mag=20, color="Green")

    
    pl.add_mesh(pvstream)
    pl.show(cpos="xy")



# %%
## Relaxation loop

for step in range(0,100):
        
    diffuser._reset()
    diffuser.add_essential_bc((0), mesh.boundaries.Upper.name)
    diffuser.add_essential_bc(v.sym.dot(mesh.CoordinateSystem.unit_e_0), mesh.boundaries.Internal.name)
    diffuser.add_essential_bc((0), mesh.boundaries.Lower.name)
    diffuser.solve(zero_init_guess=False)
    
    # delta_t.value = stokes.estimate_dt()
    displacement = delta_t.value * uw.function.evalf(Vr.sym * mesh.CoordinateSystem.unit_e_0 , mesh.data)

    print(f"Displacement - Amplitude: {np.abs(displacement).max()}")

    mesh._deform_mesh(mesh.data + displacement)
    
    stokes.solve(zero_init_guess=False)


# %%
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    with mesh.access():
        pvmesh.cell_data["R0"] = M.data[...]

    pvmesh_inner = pvmesh.threshold(value=1.0, scalars="R0", invert=False)

    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v.sym.dot(v.sym))
    pvmesh.point_data["Vr"] = vis.scalar_fn_to_pv_points(pvmesh, v.sym.dot(mesh.CoordinateSystem.unit_e_0 ))
    pvmesh.point_data["M"] = vis.scalar_fn_to_pv_points(pvmesh, M.sym)

    pvmesh_inner.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh_inner, v.sym)
    pvmesh_inner.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh_inner, v.sym.dot(v.sym))
    pvmesh_inner.point_data["Vr"] = vis.scalar_fn_to_pv_points(pvmesh_inner, v.sym.dot(mesh.CoordinateSystem.unit_e_0 ))
    pvmesh_inner.point_data["M"] = vis.scalar_fn_to_pv_points(pvmesh_inner, M.sym)

    velocity_points = vis.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym * M.sym[0])
    velocity_points.point_data["Vr"] = vis.vector_fn_to_pv_points(velocity_points, Vr.sym[0] * M.sym[0]  * mesh.CoordinateSystem.unit_e_0 )

    point_cloud = pvmesh_inner.cell_centers()

    pvstream = pvmesh_inner.streamlines_from_source(
        point_cloud, vectors="V", 
        integration_direction="forward", 
        integrator_type=2,
        surface_streamlines=True,
        initial_step_length=0.1,
        max_time=0.1,
        max_steps=100
    )
    
    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
                pvmesh_inner,
                cmap="seismic",
                edge_color="Black",
                scalars="Vmag",
                show_edges=True,
                use_transparency=False,
                opacity=0.75)

    # pl.add_mesh(pvmesh, style="wireframe", color="Grey", opacity=0.1)
        
    
    pl.add_arrows(velocity_points.points, 
                  velocity_points.point_data["V"], 
                  mag=30, color="Green")
    
    pl.add_mesh(pvstream)

    pl.show(cpos="xy")

# %%

# %%
