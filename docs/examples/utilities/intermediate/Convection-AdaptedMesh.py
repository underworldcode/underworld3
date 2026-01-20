# %% [markdown]
"""
# 🔬 Convection-AdaptedMesh

**PHYSICS:** utilities  
**DIFFICULTY:** intermediate  
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
# +
## Mesh refinement ...

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import os

os.environ["UW_TIMING_ENABLE"] = "1"
os.environ["SYMPY_USE_CACHE"] = "no"

import petsc4py
from petsc4py import PETSc

from underworld3 import timing
from underworld3 import adaptivity

import underworld3 as uw
from underworld3 import function
from enum import Enum

import numpy as np
import sympy


# %%
class bd(Enum):
    Upper=2
    
batmesh = uw.discretisation.Mesh(
    "Batmesh_number_one.h5",
    coordinate_system_type=uw.coordinates.CoordinateSystemType.CYLINDRICAL2D,
    boundaries=bd)


# %%
batmesh.dm.view()

# %%
Rayleigh_number = uw.function.expression(r"\textrm{Ra}", sympy.sympify(10000000), "Rayleigh number")

# %%
v_soln  = uw.discretisation.MeshVariable("U", batmesh, batmesh.dim, degree=2, continuous=True)
v_soln1  = uw.discretisation.MeshVariable("U1", batmesh, batmesh.dim, degree=2, continuous=True)
p_soln  = uw.discretisation.MeshVariable("P", batmesh, 1, degree=1, continuous=True)
t_soln  = uw.discretisation.MeshVariable("T", batmesh, 1, degree=2, varsymbol=r"T_0")
t_soln1 = uw.discretisation.MeshVariable("T1", batmesh, 1, degree=2, varsymbol=r"T_1")
cell_properties = uw.discretisation.MeshVariable("Bat", batmesh, 1, degree=0)


# %%
cell_properties.read_timestep(
        "Batmesh_cell_properties",
         data_name="L",
         index=0)

# %%
# Create Stokes object

stokes = uw.systems.Stokes(
    batmesh,
    velocityField=v_soln,
    pressureField=p_soln)

# Constant viscosity

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
stokes.tolerance = 1.0e-5

unit_r_vec = batmesh.CoordinateSystem.unit_e_0
unit_y_vec = batmesh.CoordinateSystem.unit_j 

# free slip.
# note with petsc we always need to provide a vector of correct cardinality.

stokes.add_essential_bc([0,0], "Upper")
# stokes.add_natural_bc(1e6  * Rayleigh_number * unit_r_vec.dot(v_soln.sym) * unit_r_vec, "Upper")

stokes.bodyforce = Rayleigh_number * unit_y_vec * t_soln.sym[0] + \
                   Rayleigh_number * unit_y_vec * t_soln1.sym[0]


stokes.petsc_options.setValue("ksp_monitor", None)
stokes.petsc_options.setValue("snes_monitor", None)
stokes.petsc_options.setValue("snes_min_it", 1)

stokes.view()

# %%
adv_diff = uw.systems.AdvDiffusionSLCN(
    batmesh,
    u_Field=t_soln,
    V_fn=v_soln)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = 1
adv_diff.add_dirichlet_bc(0.0, "Upper")


adv_diff1 = uw.systems.AdvDiffusionSLCN(
    batmesh,
    u_Field=t_soln1,
    V_fn=v_soln1)

adv_diff1.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff1.constitutive_model.Parameters.diffusivity = 1
adv_diff1.add_dirichlet_bc(0.0, "Upper")

t_init = 1-cell_properties.sym[0]

with batmesh.access(t_soln, t_soln1):
    t_soln.data[:,0] = uw.function.evalf(t_init, t_soln.coords)
    t_soln1.data[:,0] = uw.function.evalf(t_init, t_soln1.coords)



# %%
stokes.solve(zero_init_guess=True, picard=0, verbose=False)
with batmesh.access(v_soln1):
    v_soln1.data[...] = -v_soln.data[...]

# %%

t_step = 0

# %%

for step in range(0, 250):
    stokes.solve(zero_init_guess=False)
    with batmesh.access(v_soln1):
        v_soln1.data[...] = -v_soln.data[...]

    
    delta_t = 5.0 * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=True)
    adv_diff1.solve(timestep=delta_t, zero_init_guess=True)

    # stats then loop
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))

    if t_step%1==0:
        batmesh.write_timestep(
            "Batmesh",
            meshUpdates=True,
            meshVars=[p_soln, v_soln, t_soln, t_soln1],
            outputPath="output",
            index=t_step)

    # with batmesh.access(t_soln):
    #     t_soln.data[:,0] = np.maximum(t_soln.data[:,0], uw.function.evalf(t_init, t_soln.coords))

    t_step += 1


# %%
if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(batmesh)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym[0])
    pvmesh.point_data["R"] = vis.scalar_fn_to_pv_points(pvmesh, batmesh.CoordinateSystem.R[0])
    pvmesh.point_data["T1"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln1.sym[0])
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["L"] = vis.scalar_fn_to_pv_points(pvmesh, cell_properties.sym[0])

    pvmesh_t = vis.meshVariable_to_pv_mesh_object(t_soln)
    pvmesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh_t, t_soln.sym[0])
    pvmesh_t.point_data["T1"] = vis.scalar_fn_to_pv_points(pvmesh_t, t_soln1.sym[0])

    pvmesh_t.point_data["To"] = (pvmesh_t.point_data["T"] > 0.25).astype(float)
    pvmesh_t.point_data["T1o"] = (pvmesh_t.point_data["T1"] > 0.25).astype(float)
    pvmesh_t.point_data["TD"] = pvmesh_t.point_data["T"] - pvmesh_t.point_data["T1"]


    pl = pv.Plotter(window_size=(750, 750))
    pl.enable_depth_peeling()

    doughnut = pvmesh.clip_scalar(scalars="R", 
                                  value=0.3, 
                                  invert=False)
    
    pvstream = pvmesh.streamlines_from_source(
        pv.PointSet(doughnut.cell_centers().points[::10]), 
        vectors="V", 
        integrator_type=45,
        surface_streamlines=True, 
        max_steps=1000,
        max_time=1.0)


    pl.add_mesh(pvmesh, 
                style="wireframe",
                color="#FEFEF0",
                opacity=0.2)


    pl.add_mesh(
                pvmesh_t,
                cmap="Blues",
                scalars="T",
                opacity="sigmoid",
                edge_color="Grey",
                show_edges=False,
                use_transparency=False,
                show_scalar_bar=False)
    
    pl.add_mesh(
                pvmesh_t,
                copy_mesh=True,
                cmap="Grays",
                scalars="T1",
                opacity="sigmoid",
                edge_color="Grey",
                show_edges=False,
                use_transparency=False,
                show_scalar_bar=False,
                # clim=[0,2]
               )
  

    pl.add_mesh(pvstream, 
            cmap=["#ED6020", "#994305", "#665500"], 
            show_scalar_bar=False,
            opacity=0.5
           )
    
    # pl.add_mesh(
    #             pvmesh,
    #             scalars="L",
    #             cmap="Greys_r",
    #             edge_color="Black",
    #             opacity="L",
    #             show_edges=False,
    #             use_transparency=True,
    #             show_scalar_bar=False,
    #            )


    # pl.add_arrows(pvmesh.points, pvmesh.point_data["V"],
    #               mag=float(0.2/Rayleigh_number.sym),
    #              show_scalar_bar=False)

    pl.camera.roll = 180


    pl.show(jupyter_backend='html')

# %%
0/0


# %%
## Animation 

# %%
## Check the results saved to file

def read_bat_data(step):

    expt_name = "Batmesh"
    output_path="Hyp"
    

    v_soln.read_timestep(expt_name, "U", step, outputPath=output_path)
    p_soln.read_timestep(expt_name, "P", step, outputPath=output_path)
    t_soln.read_timestep(expt_name, "T", step, outputPath=output_path)
    t_soln1.read_timestep(expt_name, "T1", step, outputPath=output_path)

    return





# %%
frame = 0

# %%
import pyvista as pv
import underworld3.visualisation as vis

pl = pv.Plotter(window_size=(750, 750))

for i in range(0,250, 1):
    
    read_bat_data(250-i-1)
    
    pvmesh = vis.mesh_to_pv_mesh(batmesh)
    pvmesh.point_data["R"] = vis.scalar_fn_to_pv_points(pvmesh, batmesh.CoordinateSystem.R[0])
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["L"] = vis.scalar_fn_to_pv_points(pvmesh, cell_properties.sym[0])

    pvmesh_t = vis.meshVariable_to_pv_mesh_object(t_soln)
    pvmesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh_t, t_soln.sym[0])
    pvmesh_t.point_data["T1"] = vis.scalar_fn_to_pv_points(pvmesh_t, t_soln1.sym[0])


    doughnut = pvmesh.clip_scalar(scalars="R", 
                                  value=0.3, 
                                  invert=False)
    
    pvstream = pvmesh.streamlines_from_source(
        pv.PointSet(doughnut.cell_centers().points[::10]), 
        vectors="V", 
        integrator_type=45,
        surface_streamlines=True, 
        max_steps=1000,
        max_time=1.0)


    # pl.add_mesh(pvmesh, 
    #             style="wireframe",
    #             color="Black",
    #             opacity=0.2)


    pl.add_mesh(
                pvmesh_t,
                cmap="Grays",
                scalars="T1",
                opacity="sigmoid",
                edge_color="Grey",
                show_edges=False,
                use_transparency=False,
                show_scalar_bar=False)
    
  
    pl.add_mesh(pvstream, 
            cmap=["#ED6020", "#994305", "#665500"], 
            show_scalar_bar=False,
            opacity=0.5
           )
    

    # pl.camera.roll = 180

    pl.screenshot(filename=f"BatPumpkin_{frame}.png", window_size=(750,750), scale=4 )

    frame += 1

    pl.clear()
    


# %%
for i in range(0,250, 1):
    
    read_bat_data(i)
    
    pvmesh = vis.mesh_to_pv_mesh(batmesh)
    pvmesh.point_data["R"] = vis.scalar_fn_to_pv_points(pvmesh, batmesh.CoordinateSystem.R[0])
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["L"] = vis.scalar_fn_to_pv_points(pvmesh, cell_properties.sym[0])

    pvmesh_t = vis.meshVariable_to_pv_mesh_object(t_soln)
    pvmesh_t.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh_t, t_soln.sym[0])
    pvmesh_t.point_data["T1"] = vis.scalar_fn_to_pv_points(pvmesh_t, t_soln1.sym[0])


    doughnut = pvmesh.clip_scalar(scalars="R", 
                                  value=0.3, 
                                  invert=False)
    
    pvstream = pvmesh.streamlines_from_source(
        pv.PointSet(doughnut.cell_centers().points[::10]), 
        vectors="V", 
        integrator_type=45,
        surface_streamlines=True, 
        max_steps=1000,
        max_time=1.0)


    # pl.add_mesh(pvmesh, 
    #             style="wireframe",
    #             color="Black",
    #             opacity=0.2)


    pl.add_mesh(
                pvmesh_t,
                cmap="Grays",
                scalars="T",
                opacity="sigmoid",
                edge_color="Grey",
                show_edges=False,
                use_transparency=False,
                show_scalar_bar=False)
    
  
    pl.add_mesh(pvstream, 
            cmap=["#ED6020", "#994305", "#665500"], 
            show_scalar_bar=False,
            opacity=0.5
           )
    

    # pl.camera.roll = 180

    pl.screenshot(filename=f"BatPumpkin_{frame}.png", window_size=(750,750), scale=4 )

    frame += 1

    pl.clear()
    

# %%
t_soln.stats()

# %%
t_soln1.stats()

# %%
adv_diff

# %%
adv_diff1

# %%
adv_diff1.DuDt.bdf(1)

# %%
adv_diff.DuDt.bdf(1)

# %%
batmesh.dm.view()

# %% [markdown]
# # 
