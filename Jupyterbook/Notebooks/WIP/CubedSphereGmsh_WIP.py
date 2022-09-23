from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
options["ksp_rtol"] =  1.0e-3
options["ksp_monitor_short"] = None
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
# options["snes_view"]=None
# options["snes_test_jacobian"] = None
options["snes_max_it"] = 1
options["pc_type"] = "fieldsplit"
options["pc_fieldsplit_type"] = "schur"
options["pc_fieldsplit_schur_factorization_type"] ="full"
options["pc_fieldsplit_schur_precondition"] = "a11"
options["fieldsplit_velocity_pc_type"] = "lu"
options["fieldsplit_pressure_ksp_rtol"] = 1.e-3
options["fieldsplit_pressure_pc_type"] = "lu"


# +
# This (guy) sets up the visualisation defaults

import numpy as np
import pyvista as pv
import vtk

pv.global_theme.background = 'white'
pv.global_theme.window_size = [1000, 500]
pv.global_theme.antialiasing = True
pv.global_theme.jupyter_backend = 'panel'
pv.global_theme.smooth_shading = True
# -

import pygmsh
import meshio




cubed_sphere_mesh_shell = uw.mesh.StructuredCubeSphereShellMesh(elementRes=(11,5), radius_inner=0.5,
                                        radius_outer=1.0, simplex=False)


cubed_sphere_mesh_ball = uw.mesh.StructuredCubeSphereBallMesh(elementRes=9,
                                        radius_outer=1.0, simplex=True)


cubed_sphere_mesh = cubed_sphere_mesh_ball

# +


pvmesh = cubed_sphere_mesh.mesh2pyvista()

# pvmesh.cell_data['my cell values'] = np.arange(pvmesh.n_cells)
# pvmesh.plot(scalars='my cell values', show_edges=True)


clipped_stack = pvmesh.clip(origin=(0.00001,0.0,0.0), normal=(1, 0, 0), invert=False)

pl = pv.Plotter()

# pl.add_mesh(pvstack,'Blue', 'wireframe' )
pl.add_mesh(clipped_stack, cmap="coolwarm", edge_color="Black", show_edges=True, 
              use_transparency=False)
pl.show()

# +
# Create Stokes object
stokes = Stokes(cubed_sphere_mesh,u_degree=2,p_degree=1)
# Constant visc
stokes.viscosity = 1.

# Velocity boundary conditions
stokes.add_dirichlet_bc( (0.,0.,0.), cubed_sphere_mesh.boundary.ALL_BOUNDARIES, (0,1,2) )

# +
# Create a density structure

import sympy

dens_ball = 10.
dens_other = 1.
position_ball = 0.75*cubed_sphere_mesh.N.k
radius_ball = 0.5

off_rvec = cubed_sphere_mesh.rvec - position_ball
abs_r = off_rvec.dot(off_rvec)
density = sympy.Piecewise( ( dens_ball,    abs_r < radius_ball**2 ),
                           ( dens_other,                   True ) )
density
# -

# Write density into a variable for saving
densvar = uw.mesh.MeshVariable("density",cubed_sphere_mesh,1)

with cubed_sphere_mesh.access(densvar):
    densvar.data[:,0] = uw.function.evaluate(density,densvar.coords)
    print(densvar.data.max())

unit_rvec = cubed_sphere_mesh.rvec / (1.0e-10+sympy.sqrt(cubed_sphere_mesh.rvec.dot(cubed_sphere_mesh.rvec)))
stokes.bodyforce = -unit_rvec*density
stokes.bodyforce

stokes.solve()

# +
pv_vtkmesh = cubed_sphere_mesh.mesh2pyvista()

umag = stokes.u.fn.dot(stokes.u.fn)

pv_vtkmesh.point_data['density'] = uw.function.evaluate(density,cubed_sphere_mesh.data)
# pv_vtkmesh.point_data['umag'] = uw.function.evaluate(umag,cubed_sphere_mesh.data)
# -

clipped = pv_vtkmesh.clip(normal=(1, 0, 0), invert=False)
contours = pv_vtkmesh.contour([1.0,5.0, 10.0], scalars="density")

# +
pl = pv.Plotter()
pl.add_mesh(clipped, cmap="coolwarm", edge_color="Black", show_edges=True, 
            scalars="density",  use_transparency=False)

with cubed_sphere_mesh.access():
    usol = stokes.u.data
    
pl.add_arrows(stokes.u.coords, usol, mag=10.0)
# pl.add_mesh(contours, opacity=0.5)

pl.show()
