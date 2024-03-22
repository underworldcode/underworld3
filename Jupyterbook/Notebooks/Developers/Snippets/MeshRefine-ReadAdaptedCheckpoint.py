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
from underworld3 import function

import numpy as np
import sympy

free_slip_upper = True


# +
meshA = uw.discretisation.Mesh("adaptor_write_xdmf.mesh.0.h5")

gradA = uw.discretisation.MeshVariable(r"\nabla~T", meshA, 1)
v_soln = uw.discretisation.MeshVariable(r"u", meshA, meshA.dim, degree=2, vtype=uw.VarType.VECTOR)
p_soln = uw.discretisation.MeshVariable(r"p", meshA, 1, degree=1, continuous=True)

# -

gradA.read_from_vertex_checkpoint("adaptor_write_xdmf.proxy.nablaT_s.0.h5", data_name="proxy_nablaT_s")
v_soln.read_from_vertex_checkpoint("adaptor_write_xdmf.u.0.h5", data_name="u")
p_soln.read_from_vertex_checkpoint("adaptor_write_xdmf.p.0.h5", data_name="p")

with meshA.access():
    print(gradA.data.max())

uw.utilities.h5_scan("adaptor_write_xdmf.proxy.nablaT_s.0.h5")

# +
# Read the raw points and look at those

# +
# gs = h5py.File("adaptor_write.gradS0.h5")
# gsdat = gs["data"][()]
# gs.close()

# gx = h5py.File("adaptor_write.X.h5")
# gxdat = gx["data"][()]
# gx.close()

# gsdat = g["data"]["nablaTSA"][()].copy()
# gsX = g["fields"]["X"][()].copy()


# +
import mpi4py

if mpi4py.MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    meshA.vtk("tmp_meshball.vtk")
    pvmeshA = pv.read("tmp_meshball.vtk")

# -


if mpi4py.MPI.COMM_WORLD.size == 1:
    
    with meshA.access():
        pvmeshA.point_data["gradS"] = gradA.data.copy()

    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[...] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[...] = v_soln.rbf_interpolate(v_soln.coords)

    clipped = pvmeshA.clip(origin=(0.0, 0.0, 0.0), normal=(0.0, 1, 0), invert=False)

# +

if mpi4py.MPI.COMM_WORLD.size == 1:

    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(
        clipped, 
        cmap="coolwarm",
        edge_color="black",
        style="surface",
        scalars = "gradS",
        show_edges=True,
    )

    pl.add_arrows(arrow_loc, arrow_length, mag=33, opacity=0.5)

    # pl.screenshot(filename="sphere.png", window_size=(1000, 1000), return_img=False)
    # OR
    pl.show(cpos="xy")
# -
# ls -rtl


