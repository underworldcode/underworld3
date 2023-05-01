# +
## Mesh refinement ... 

import os
os.environ["UW_TIMING_ENABLE"] = "1"
os.environ["SYMPY_USE_CACHE"] = "no"

import petsc4py
from petsc4py import PETSc

from underworld3 import timing
from underworld3 import adaptivity


import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

free_slip_upper = True


# +
# Earth-like ratio of inner to outer
r_o = 1.0
r_i = 0.547
res = 1000 / 6730 

mesh0 = uw.meshing.SphericalShell(radiusOuter=r_o, 
                           radiusInner=r_i, 
                           cellSize=res,
                           filename="tmp_low_r.msh")

grad = uw.discretisation.MeshVariable(r"\nabla~T", mesh0, 1)


# +
x, y, z = mesh0.CoordinateSystem.N

t_forcing_fn = 1.0 * (
    + sympy.exp(-10.0 * (x**2 + (y - 0.8) ** 2 + z**2))
    + sympy.exp(-10.0 * ((x - 0.8) ** 2 + y**2 + z**2))
    + sympy.exp(-10.0 * (x**2 + y**2 + (z - 0.8) ** 2))
)

# -

gradient = uw.systems.Projection(mesh0, grad, solver_name="gradient")
gradient.uw_function = 1.0 + mesh0.vector.gradient(t_forcing_fn**2).dot(mesh0.vector.gradient(t_forcing_fn**2))
gradient.petsc_options["snes_rtol"] = 1.0e-2
gradient.smoothing = 1.0e-3
gradient.solve()


mesh0.dm.view()
dm1 = mesh0.dm

# +
mesh1 = uw.meshing.SphericalShell(radiusOuter=r_o, 
                           radiusInner=r_i, 
                           cellSize=res,
                           filename="tmp_refined_r.msh",
                           refinement=2)

grad1 = uw.discretisation.MeshVariable(r"\nabla~T_1", mesh1, 1)
grad2 = uw.discretisation.MeshVariable(r"\nabla~T_2", mesh1, 1)
v_soln = uw.discretisation.MeshVariable(r"u", mesh1, mesh1.dim, degree=2, vtype=uw.VarType.VECTOR)
p_soln = uw.discretisation.MeshVariable(r"p", mesh1, 1, degree=1, continuous=True)

with mesh1.access(grad1):
    grad1.data[...] = grad.rbf_interpolate(grad1.coords)

adaptivity.mesh2mesh_mapVar(grad, grad2)    
# -



# +
uw.function.interpolate_vars_on_mesh(mesh0, mesh0.data)




# -

0/0

mesh1.dm.view()

mesh1.write_timestep_xdmf("refined_write_xdmf", meshUpdates=True, meshVars=[grad1, grad2, v_soln, p_soln])

# +
import mpi4py

if mpi4py.MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.anti_aliasing = "ssaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh0.vtk("tmp_meshball0.vtk")
    pvmesh0 = pv.read("tmp_meshball0.vtk")

    mesh1.vtk("tmp_meshball.vtk")
    pvmesh = pv.read("tmp_meshball.vtk")
    
    
    with mesh1.access():
        pvmesh.point_data["grad1"] = grad1.data.copy()
        pvmesh.point_data["grad2"] = grad2.data.copy()

    pvmesh["delta"] = pvmesh.point_data["grad1"] - pvmesh.point_data["grad2"]
# -


pvmesh.points *= 0.99

# +
pl = pv.Plotter(window_size=[1000, 1000])
pl.add_axes()

pl.add_mesh(
    pvmesh, 
    cmap="coolwarm",
    # clim=[0.997, 1.0],
    edge_color="Black",
    style="surface",
    scalars="delta",
    show_edges=True,
)

pl.add_mesh(
    pvmesh0, 
    edge_color="White",
    style="wireframe",
    color="White", 
    render_lines_as_tubes=True,
)


# pl.screenshot(filename="sphere.png", window_size=(1000, 1000), return_img=False)
# OR
pl.show(cpos="xy")
# -




