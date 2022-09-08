# # Cylindrical Stokes

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
import numpy as np
import sympy
# -


meshball_xyz = uw.meshing.Annulus(radiusOuter=1.0, 
                              radiusInner=0.5,
                              cellSize=0.2,
                              centre=True)

xy_vec = meshball_xyz.dm.getCoordinates()
xy = xy_vec.array.reshape(-1,2)
dmplex = meshball_xyz.dm.clone()
rtheta = np.empty_like(xy)
rtheta[:,0] = np.sqrt(xy[:,0]**2 + xy[:,1]**2)
rtheta[:,1] = np.arctan2(xy[:,1]+1.0e-16, xy[:,0]+1.0e-16)
rtheta_vec = xy_vec.copy()
rtheta_vec.array[...] = rtheta.reshape(-1)[...]
dmplex.setCoordinates(rtheta_vec)

meshball = uw.meshing.Mesh(dmplex)

# meshball._N = sympy.vector.CoordSys3D("N", vector_names=('e1','e2','e3'), variable_names=('r','t','z'))
meshball._N.x._latex_form=r"\mathrm{r}"
meshball._N.y._latex_form=r"\mathrm{\theta}"
meshball._N.z._latex_form=r"\mathrm{z}"
meshball._N.i._latex_form=r"\mathbf{\hat{e}_r}"
meshball._N.j._latex_form=r"\mathbf{\hat{e}_\theta}"
meshball._N.k._latex_form=r"\mathbf{\hat{e}_z}"

v_soln = uw.discretisation.MeshVariable('U',meshball, 2, degree=2 )
p_soln = uw.discretisation.MeshVariable('P',meshball, 1, degree=1 )
t_soln = uw.discretisation.MeshVariable("T",meshball, 1, degree=3 )


v_soln.sym[0]

# +
# check the mesh if in a notebook / serial

if uw.mpi.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    
    meshball.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")
    pvmesh.points[:,0:2] = xy[:,0:2]

    pvmesh.plot(show_edges=True, cpos="xy")


# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre 
# of the sphere to (say) 1 at the surface

radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec)) # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10+radius_fn)
gravity_fn = radius_fn

e = 0 # sympy.sympify(10)**sympy.sympify(-10)

# Some useful coordinate stuff 

r, th = meshball.X

x = r * sympy.cos(th) - sympy.sin(th) * th
y = r * sympy.sin(th) + sympy.cos(th) * th

# 
Rayleigh = 1.0e2

# +
# Create Stokes object

stokes = uw.systems.Stokes(meshball, velocityField=v_soln, 
                pressureField=p_soln, 
                solver_name="stokes")

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModelCylinder(meshball.X)
stokes.constitutive_model.material_properties = stokes.constitutive_model.Parameters(viscosity = 1)             

stokes.saddle_preconditioner = 1 / r
   
# Velocity boundary conditions

stokes.add_dirichlet_bc( (0.0,0.0), "Upper", (0,1))
stokes.add_dirichlet_bc( (0.0, 0.0), "Lower", (0,1))


# +
# Write density into a variable for saving
t_init = sympy.cos(4*th)

with meshball.access(t_soln):
    t_soln.data[:,0] = uw.function.evaluate(t_init, t_soln.coords)
    print(t_soln.data.min(), t_soln.data.max())
# -
stokes.bodyforce = sympy.Matrix([Rayleigh * t_init, 0])

stokes._setup_terms()

gradU = stokes._L.copy()
gradU[0,1] = stokes._L[0,1] / r
gradU[1,1] = stokes._L[1,1] / r

stokes._E.diff(stokes._L)



stokes._u_f1



# +
stokes.petsc_options["snes_test_jacobian"] = None
# stokes.petsc_options["snes_type"] = "newtontr"
stokes.petsc_options["snes_rtol"] = 1.0e-3
stokes.petsc_options["pc_type"] = "fieldsplit"
stokes.petsc_options["pc_fieldsplit_type"] = "schur"
stokes.petsc_options["pc_fieldsplit_schur_fact_type"] = "diag"
stokes.petsc_options["pc_fieldsplit_schur_precondition"] = "a11"
stokes.petsc_options["pc_fieldsplit_detect_saddle_point"] = None
stokes.petsc_options["pc_fieldsplit_off_diag_use_amat"] = None    # These two seem to be needed in petsc 3.17
stokes.petsc_options["pc_use_amat"] = None                        # These two seem to be needed in petsc 3.17
stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fgmres"
stokes.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1.0e-4
stokes.petsc_options["fieldsplit_velocity_pc_type"]  = "gamg"
stokes.petsc_options["fieldsplit_pressure_ksp_rtol"] = 3.e-4
stokes.petsc_options["fieldsplit_pressure_pc_type"] = "gamg" 

# stokes.petsc_options.delValue("pc_fieldsplit_off_diag_use_amat")
# stokes.petsc_options.delValue("pc_use_amat")

stokes.petsc_options["ksp_monitor"] = None
#stokes.petsc_options["fieldsplit_velocity_ksp_monitor"] = None
#stokes.petsc_options["fieldsplit_pressure_ksp_monitor"] = None


# stokes.snes.view()
# -



stokes.solve()

# +
# U_xy = meshball.Rot * v_soln.sym.T
# U_xy = meshball.Rot * stokes.bodyforce.T
# -

U_xy

# +
# An alternative is to use the swarm project_from method using these points to make a swarm

# +
# check the mesh if in a notebook / serial


if uw.mpi.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [1000, 1000]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    
    meshball.vtk("tmp.vtk")
    pvmesh = pv.read("tmp.vtk")
    
    with meshball.access():
        pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, meshball.data)
        
    usol = np.empty_like(v_soln.coords)
    usol[:,0] = uw.function.evaluate(U_xy[0], v_soln.coords)
    usol[:,1] = uw.function.evaluate(U_xy[1], v_soln.coords)

    arrow_loc = np.zeros((stokes.u.coords.shape[0],3))
    arrow_loc[:,0:2] = stokes.u.coords[...]
    
    arrow_length = np.zeros((stokes.u.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...]
# -


    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, 
                  use_transparency=False, opacity=0.5)
    pl.add_arrows(arrow_loc, arrow_length, mag=0.1)
    pl.show(cpos="xy")


