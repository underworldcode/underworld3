# # Cylindrical Stokes

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np

# options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None


# +
import meshio

meshball = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)
# -


v_soln = uw.discretisation.MeshVariable("U", meshball, 2, degree=2)
p_soln = uw.discretisation.MeshVariable(
    ["P", "h"],
    meshball,
    num_components=2,
    continuous=True,
    degree=1,
    vtype=uw.VarType.COMPOSITE,
)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
maskr = uw.discretisation.MeshVariable("M", meshball, 1, degree=1)

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

import sympy

radius_fn = sympy.sqrt(
    meshball.rvec.dot(meshball.rvec)
)  # normalise by outer radius if not 1.0
unit_rvec = meshball.vector.to_matrix(meshball.rvec / (1.0e-10 + radius_fn))
gravity_fn = radius_fn

# Some useful coordinate stuff

x = meshball.N.x
y = meshball.N.y

r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y + 1.0e-5, x + 1.0e-5)

#

Rayleigh = 1.0e2

hw = 500.0
surface_fn_r = sympy.exp(-((r - 1.0) ** 2) * hw)
surface_fn = maskr.sym[0]


# +
# Surface-driven flow, use this bc

stokes_f = Stokes(
    meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes_fixed"
)

stokes_f.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes_f.constitutive_model.material_properties = (
    stokes_f.constitutive_model.Parameters(viscosity=1)
)

# Velocity boundary conditions

stokes_f.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))
stokes_f.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))

# velocity constraints

stokes_f.constraints = sympy.sympify(sympy.Matrix([stokes_f.div_u, 0]).T)


# +
# Create Stokes object

stokes = Stokes(
    meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes"
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel(meshball.dim)
stokes.constitutive_model.material_properties = stokes.constitutive_model.Parameters(
    viscosity=1
)

# Velocity boundary conditions

stokes.add_neumann_bc((0.0, 0.0), "Upper", (0, 1))
stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))

# velocity constraints

# stokes.constraints = sympy.sympify(sympy.Matrix([ v_soln.sym.dot(unit_rvec) * mask.sym[0]]).T)
stokes.constraints = sympy.sympify(
    sympy.Matrix([stokes.div_u, v_soln.sym.dot(unit_rvec) * surface_fn_r]).T
)
# stokes.constraints = sympy.sympify(sympy.Matrix([stokes.div_u]).T)

stokes.UF0 = -p_soln.sym[1] * unit_rvec * surface_fn_r

stokes.saddle_preconditioner = 1 / (1 + Rayleigh * surface_fn)

# +
# stokes.petsc_options["snes_test_jacobian"] = None

# stokes.petsc_options["snes_type"] = "qn"

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["snes_rtol"] = 1.0e-4
stokes.petsc_options["snes_max_it"] = 10
stokes.petsc_options["ksp_rtol"] = 1.0e-4
stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fgmres"
stokes.petsc_options["fieldsplit_pressure_ksp_type"] = "fgmres"
stokes.petsc_options["pc_fieldsplit_schur_fact_type"] = "full"
stokes.petsc_options["pc_fieldsplit_schur_precondition"] = "a11"
stokes.petsc_options["pc_fieldsplit_diag_use_amat"] = None
stokes.petsc_options["pc_fieldsplit_off_diag_use_amat"] = None
stokes.petsc_options["pc_use_amat"] = None


# stokes.petsc_options["pc_fieldsplit_off_diag_use_amat"] = None    # These two seem to be needed in petsc 3.17
# stokes.petsc_options["pc_use_amat"] = None                        # These two seem to be needed in petsc 3.17

stokes.petsc_options["ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_velocity_ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_pressure_ksp_monitor"] = None


# -

t_init = sympy.cos(3 * th)

# +
# Write density into a variable for saving

with meshball.access(t_soln):
    t_soln.data[:, 0] = uw.function.evaluate(t_init, t_soln.coords)

with meshball.access(maskr):
    maskr.data[:, 0] = uw.function.evaluate(surface_fn_r, maskr.coords)
# -
unit_rvec * t_init

stokes_f.bodyforce = Rayleigh * unit_rvec * t_init
stokes.bodyforce = Rayleigh * unit_rvec * t_init

stokes_f.solve()

stokes.solve(zero_init_guess=False)

stokes._up_G0.reshape(1, 4)

stokes.snes.view()

# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 600]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    meshball.vtk("tmp_ball.vtk")
    pvmesh = pv.read("tmp_ball.vtk")

    with meshball.access():
        pvmesh.point_data["T"] = uw.function.evaluate(
            p_soln.sym[0] * surface_fn, meshball.data
        )

    with meshball.access():
        usol = stokes.u.data

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]
    # -

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
        opacity=0.5,
    )
    pl.add_arrows(arrow_loc, arrow_length, mag=0.3)
    pl.show(cpos="xy")

stokes._uu_G3

stokes._pu_G1.reshape(2, 4)

stokes._up_G2

stokes._pu_G0

stokes._p_f0

stokes._up_G0.reshape(1, 4)  # reorganise ??

stokes._u_f1
