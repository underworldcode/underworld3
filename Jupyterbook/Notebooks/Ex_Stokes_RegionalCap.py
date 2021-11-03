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
options["fieldsplit_pressure_pc_type"] = "ilu"


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


regional_cap = uw.mesh.StructuredCubeSphericalCap(elementRes=(8,8,7), 
                                                  radius_inner=0.5, radius_outer=1.0, 
                                                  simplex=True, angles=(np.pi/4,np.pi/4)
                                                     )

regional_cap.pygmesh.points.shape

regional_cap.mesh2pyvista().plot(show_edges=True)

# Create Stokes object
stokes = Stokes(regional_cap,u_degree=2,p_degree=1)
# Constant visc
stokes.viscosity = 1.
# No slip boundary conditions
stokes.add_dirichlet_bc( (0.,0.,0.), regional_cap.boundary.ALL_BOUNDARIES, (0,1,2) )

# +
# Create a density structure

import sympy

dens_ball = 10.
dens_other = 1.
position_ball = 0.75*regional_cap.N.k
radius_ball = 0.1


off_rvec = regional_cap.rvec - position_ball
abs_r = off_rvec.dot(off_rvec)
density = sympy.Piecewise( ( dens_ball,    abs_r < radius_ball**2 ),
                           ( dens_other,                   True ) )
density
# -

# Write density into a variable for saving
densvar = uw.mesh.MeshVariable("density",regional_cap,1)
with regional_cap.access(densvar):
    densvar.data[:,0] = uw.function.evaluate(density,densvar.coords)

unit_rvec = regional_cap.rvec / sympy.sqrt(regional_cap.rvec.dot(regional_cap.rvec))
stokes.bodyforce = -unit_rvec*density
stokes.bodyforce

regional_cap.rvec

stokes.solve()

# +
pv_vtkmesh = regional_cap.mesh2pyvista()

umag = stokes.u.fn.dot(stokes.u.fn)

pv_vtkmesh.point_data['density'] = uw.function.evaluate(density,regional_cap.data)




# pv_vtkmesh.point_data['umag'] = uw.function.evaluate(umag,0.005+regional_cap.data*0.99)
# pv_vtkmesh.point_data['urange'] = 0.5 + 0.5 * np.minimum(1.0,pv_vtkmesh.point_data['umag'] / pv_vtkmesh.point_data['umag'].mean())
# -

clipped = pv_vtkmesh.clip(normal=(1, 0, 0), invert=False)
contours = pv_vtkmesh.contour([1.0,5.0, 10.0], scalars="density")

# +
pl = pv.Plotter()
pl.add_mesh(clipped, cmap="coolwarm", edge_color="Black", show_edges=True, 
            scalars="density",  use_transparency=False)

with regional_cap.access():
    usol = stokes.u.data

pl.add_arrows(stokes.u.coords, usol, )
pl.add_mesh(contours, opacity=0.5)

pl.show()
# -

# I cannot manage to evaluate the stokes.u variable anywhere and the value of the u.data array appears to be zero everywhere if 


usol.max()

uw.function.evaluate(unit_rvec, pv_vtkmesh.points)

with regional_cap.access():
    usol = stokes.u.data
    psol = stokes.p.data
    

psol.min(), psol.max(), usol.min(), usol.max()


