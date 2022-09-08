from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
options["ksp_rtol"] =  1.0e-3  # For demonstration purposes we can use a loose tolerance
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




regional_cap_meshio = uw.discretisation.StructuredCubeSphericalCap.build_pygmsh(
                                elementRes=(64,64,64), 
                                angles=(np.pi/2,np.pi/2),
                                radius_inner=0.5, 
                                radius_outer=1.0, 
                                simplex=False, 
                            )

regional_cap_meshio.write("rcm.vtk")
rcm_pyvista = pv.read("rcm.vtk")
rcm_pyvista.plot(show_edges=True)

# +
regional_cap_mesh = uw.discretisation.StructuredCubeSphericalCap(
                                elementRes=(6,6,6), 
                                angles=(np.pi/2,np.pi/2),
                                radius_inner=0.5, 
                                radius_outer=1.0, 
                                simplex=True, 
                            )

regional_cap_mesh.mesh2pyvista().plot(show_edges=True)
# -

# Create Stokes object
stokes = Stokes(regional_cap_mesh,u_degree=2,p_degree=1)
# Constant visc

stokes.viscosity = 1.
# No slip boundary conditions
stokes.add_dirichlet_bc( (0.,0.,0.), regional_cap_mesh.boundary.ALL_BOUNDARIES, (0,1,2) )

# +
# Create a density structure

import sympy

dens_ball = 10.
dens_other = 1.
position_ball = 0.75*regional_cap_mesh.N.k
radius_ball = 0.25


off_rvec = regional_cap_mesh.rvec - position_ball
abs_r = off_rvec.dot(off_rvec)
density = sympy.Piecewise( ( dens_ball,    abs_r < radius_ball**2 ),
                           ( dens_other,                   True ) )
density
# -

# Write density into a variable for saving
densvar = uw.discretisation.MeshVariable("density",regional_cap_mesh,1)
with regional_cap_mesh.access(densvar):
    densvar.data[:,0] = uw.function.evaluate(density,densvar.coords)

unit_rvec = regional_cap_mesh.rvec / sympy.sqrt(regional_cap_mesh.rvec.dot(regional_cap_mesh.rvec))
stokes.bodyforce = -unit_rvec*density
stokes.bodyforce

regional_cap_mesh.rvec

stokes.solve()

# +
pv_vtkmesh = regional_cap_mesh.mesh2pyvista()

umag = stokes.u.fn.dot(stokes.u.fn)

pv_vtkmesh.point_data['density'] = uw.function.evaluate(density,regional_cap_mesh.data)

# pv_vtkmesh.point_data['umag'] = uw.function.evaluate(umag,0.005+regional_cap.data*0.99)
# pv_vtkmesh.point_data['urange'] = 0.5 + 0.5 * np.minimum(1.0,pv_vtkmesh.point_data['umag'] / pv_vtkmesh.point_data['umag'].mean())
# -

clipped = pv_vtkmesh.clip(normal=(1, 0, 0), invert=False)
contours = pv_vtkmesh.contour([1.0,5.0, 10.0], scalars="density")

# +
pl = pv.Plotter()
pl.add_mesh(clipped, cmap="coolwarm", edge_color="Black", show_edges=True, 
            scalars="density",  use_transparency=False)

with regional_cap_mesh.access():
    usol = stokes.u.data

pl.add_arrows(stokes.u.coords, usol, mag=10.0)
pl.add_mesh(contours, opacity=0.5)

pl.show()
# -

# The hex mesh always seem problematic here ... 


densvar.coords

uw.function.evaluate(unit_rvec, pv_vtkmesh.points)

with regional_cap_mesh.access():
    usol = stokes.u.data
    psol = stokes.p.data


psol.min(), psol.max(), usol.min(), usol.max()


