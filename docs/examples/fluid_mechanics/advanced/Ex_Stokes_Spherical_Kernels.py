# %% [markdown]
"""
# 🎓 Stokes Spherical Kernels

**PHYSICS:** fluid_mechanics  
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
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # SphericalStokes 
#
# Spherical mesh with embedded internal surface. This allows us to introduce an internal force integral
#

from IPython.display import IFrame
IFrame(src="media/stokes_sphere_plot.html", width=750, height=750)

# to fix visualisation issue
import nest_asyncio
nest_asyncio.apply()

# +
import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

res = 0.08
r_o = 1.0
r_int = 0.825
r_i = 0.55

free_slip_upper = True
free_slip_lower = True
true_delta_fn = True

expt_name = f"Spherical_Kernel_np_{uw.mpi.size}"
output_path = "output"

uw.require_dirs(["output"])
# -
meshball = uw.meshing.SphericalShellInternalBoundary(radiusOuter=r_o,                       
                                              radiusInternal=r_int, 
                                              radiusInner=r_i, 
                                              cellSize=res)


meshball.view()

# The following is a projection that allows us to access the values
# of the normal vector that PETSc applies to each face in the surface
# integral.
#
# Internal surfaces may not be oriented consistently across the sphere, especially
# if the mesh is decomposed, so this is a way to fix that and to check that it has
# been fixed because you can visualise the normals.

# +
norm_v = uw.discretisation.MeshVariable("N", meshball, 3, degree=1, varsymbol=r"{\hat{n}}")

projection = uw.systems.Vector_Projection(meshball, norm_v)
projection.uw_function = sympy.Matrix([[0,0,0]])
projection.smoothing = 1.0e-3

# Point in a consistent direction wrt vertical 
GammaNorm = meshball.Gamma.dot(meshball.CoordinateSystem.unit_e_0) / sympy.sqrt(meshball.Gamma.dot(meshball.Gamma))

# projection.add_natural_bc((0,0), "All_Edges")
projection.add_natural_bc(meshball.Gamma * GammaNorm, "Internal")
projection.add_natural_bc(meshball.Gamma * GammaNorm, "Upper")
projection.add_natural_bc(meshball.Gamma * GammaNorm, "Lower")

projection.solve(verbose=False, debug=False)

with meshball.access(norm_v):
    norm_v.data[:,:] /= np.sqrt(norm_v.data[:,0]**2 + norm_v.data[:,1]**2 + norm_v.data[:,2]**2).reshape(-1,1)

# -




# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

radius_fn = meshball.CoordinateSystem.xR[0]
unit_rvec = meshball.CoordinateSystem.unit_e_0
gravity_fn = 1  # radius_fn / r_o

# Some useful coordinate stuff

x, y, z = meshball.CoordinateSystem.X
r, th, ph = meshball.CoordinateSystem.xR

# Null space in velocity (constant v_theta) expressed in x,y coordinates
v_theta_fn_xy = r * meshball.CoordinateSystem.rRotN.T * sympy.Matrix((0,1,0))

# -
v_soln = uw.discretisation.MeshVariable("V0", meshball, 3, degree=2, varsymbol=r"{v_0}")
p_soln = uw.discretisation.MeshVariable("p", meshball, 1, degree=0, continuous=False)
p_cont = uw.discretisation.MeshVariable("pc", meshball, 1, degree=1, continuous=True)
t_init = uw.discretisation.MeshVariable("T", meshball, 1, degree=2, continuous=True)


# +
# Create Stokes object

stokes = Stokes(
    meshball, velocityField=v_soln, 
    pressureField=p_soln
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.penalty = 1.0

stokes.tolerance = 1.0e-3

stokes.petsc_options.setValue("ksp_monitor", None)
stokes.petsc_options.setValue("snes_monitor", None)
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

# Similar (equivalent horizontal pattern) but in a body force
with meshball.access(t_init):
    t_init.data[:,0] = uw.function.evaluate(
            sympy.cos(3*ph) * sympy.sin(5*th) * sympy.exp(-50.0 * ((r - r_int) ** 2)), t_init.coords) 



# +
## Three choices: 
##    Normals that have been re-oriented correctly and stored in a meshVariable
##    Normals that are calculated analytically
##    PETSc Normals that are re-oriented (as above) using a consistent direction estimate (here, radial)

# This first choice is the PETSc normals, re-oriented on the fly
GammaNorm = meshball.Gamma.dot(meshball.CoordinateSystem.unit_e_0) / sympy.sqrt(meshball.Gamma.dot(meshball.Gamma))
Gamma     = GammaNorm * meshball.Gamma #  / sympy.sqrt(meshball.Gamma.dot(meshball.Gamma))
# Gamma = norm_v.sym
# Gamma = unit_rvec


if true_delta_fn:
    stokes.add_natural_bc(sympy.cos(3*ph) * sympy.sin(5*th) * Gamma, "Internal")
else:
    # Note, these are not the same force term because the 
    # approximate delta function should be scaled according to
    # the mesh dimension. But it is simply for debugging !
    stokes.bodyforce = t_init.sym[0] * unit_rvec

if free_slip_upper:
    stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Upper")
else:
    stokes.add_essential_bc((0.0,0.0,0.0), "Upper")

if free_slip_lower:
    stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Lower")
else:
    stokes.add_essential_bc((0.0,0.0,0.0), "Lower")

stokes.solve(verbose=False)



# +
# Null space evaluation 

I0 = uw.maths.Integral(meshball, 1)

for orientation in ((1,0,0), (0,1,0), (0,0,1)):
    v_rotation = r * meshball.CoordinateSystem.rRotN.T * sympy.Matrix(orientation)

    I0.fn = v_rotation.dot(v_soln.sym)
    norm = I0.evaluate()
    I0.fn = v_rotation.dot(v_rotation)
    vnorm = I0.evaluate()

    # print(norm/vnorm, vnorm)

    with meshball.access(v_soln):
        dv = uw.function.evaluate(norm * v_theta_fn_xy, v_soln.coords) / vnorm
        v_soln.data[...] -= dv 



# +
# If we choose a discontinuous pressure, we need to create a continuous 
# equivalent for analysis / plotting

pressure_solver = uw.systems.Projection(meshball, p_cont)
pressure_solver.uw_function = p_soln.sym[0]
pressure_solver.smoothing = 1.0e-6
pressure_solver.solve()

pstats1 = p_cont.stats()
pstats0 = p_soln.stats()

uw.pprint(0, f"Pressure (C1): {pstats1}")
    print(f"Pressure (C0): {pstats0}")
    print(f"Velocity: {vnorm}")
# -

meshball.write_timestep(
    expt_name,
    meshUpdates=True,
    meshVars=[p_soln, v_soln, p_cont],
    outputPath=output_path,
    index=0)



if not uw.is_notebook:
    exit()

# +
# Optional - read in a different solution if you want to just use this for
# visualisation - set the `expt_name` here

expt_name="Spherical_Kernel_np_96"

import os

mesh_chkpt = uw.discretisation.Mesh(os.path.join(output_path, expt_name) + ".mesh.00000.h5")

v_soln_ckpt = uw.discretisation.MeshVariable("V0_c", mesh_chkpt, 3, degree=2, varsymbol=r"{v_0}")
p_soln_ckpt = uw.discretisation.MeshVariable("p_c", mesh_chkpt, 1, degree=1, continuous=False)
p_cont_ckpt = uw.discretisation.MeshVariable("pc_c", mesh_chkpt, 1, degree=1, continuous=True)

v_soln_ckpt.read_timestep(expt_name, "V0", 0, outputPath=output_path)
p_soln_ckpt.read_timestep(expt_name, "p", 0, outputPath=output_path)
p_cont_ckpt.read_timestep(expt_name, "pc", 0, outputPath=output_path)


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh_chkpt)
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln_ckpt.sym)

    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln_ckpt.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln_ckpt.sym)
    pvmesh.point_data["R"] = vis.scalar_fn_to_pv_points(pvmesh, radius_fn)

    # pvmesh_c = pvmesh.clip_scalar(scalars="R", value=0.95).clip()
    pvmesh_c = pvmesh.clip(normal="z")
        
    skip = 10
    points = np.zeros((meshball._centroids[::skip].shape[0], 3))
    points[:, 0] = meshball._centroids[::skip, 0]
    points[:, 1] = meshball._centroids[::skip, 1]
    points[:, 2] = meshball._centroids[::skip, 2]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", 
        integration_direction="both", 
        integrator_type=45,
        max_time=1,
        max_steps=500)
   

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh_c,
        cmap="coolwarm",
        edge_color="Black",
        # edge_opacity=0.33,
        scalars="P",
        show_edges=True,
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=False
    )

    pl.add_mesh(pvstream, opacity=0.23, show_scalar_bar=False, color="Black", render_lines_as_tubes=True)


    vsol_rms = np.sqrt(velocity_points.point_data["V"][:, 0] ** 2 + velocity_points.point_data["V"][:, 1] ** 2).mean()
    # print(vsol_rms)

    pl.export_html("stokes_sphere_plot_96.html")
    # pl.show(cpos="xy", jupyter_backend="trame")
# -


from IPython.display import IFrame
IFrame(src="./stokes_sphere_plot_96.html", width=750, height=750)


p_soln.stats()
