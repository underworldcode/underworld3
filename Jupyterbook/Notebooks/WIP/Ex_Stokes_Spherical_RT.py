# +
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np

options = PETSc.Options()
PETSc.Sys().pushErrorHandler("debugger")
# options["help"] = None
# options["pc_type"]  = "svd"
options["ksp_rtol"] = 1.0e-5
options["ksp_monitor_short"] = None
# options["snes_type"]  = "fas"
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
# options["snes_view"]=None
# options["snes_test_jacobian"] = None
options["snes_max_it"] = 1

options["pc_type"] = "fieldsplit"
options["pc_fieldsplit_type"] = "schur"
options["pc_fieldsplit_schur_factorization_type"] = "full"
options["pc_fieldsplit_schur_precondition"] = "a11"
options["fieldsplit_velocity_pc_type"] = "lu"
options["fieldsplit_pressure_ksp_rtol"] = 1.0e-5
options["fieldsplit_pressure_pc_type"] = "lu"
# -

cell_size = 0.01
mesh = uw.discretisation.SphericalShell(2, cell_size=cell_size)


def plot_mesh(mesh, colour_data=None):
    import k3d
    import plot

    vertices_2d = plot.mesh_coords(mesh)
    vertices = np.zeros((vertices_2d.shape[0], 3), dtype=np.float32)
    vertices[:, 0:2] = vertices_2d[:]
    indices = plot.mesh_faces(mesh)
    kplot = k3d.plot()
    options = {}
    if colour_data is not None:
        options["attribute"] = colour_data
    kplot += k3d.mesh(
        vertices, indices, color_map=k3d.colormaps.basic_color_maps.RainbowDesaturated, wireframe=True, **options
    )
    kplot.grid_visible = False
    kplot.display()
    kplot.camera = [-0.2, 0.2, 2.0, 0.0, 0.0, 0.0, -0.5, 1.0, -0.1]  # these are some adhoc settings


def plot_swarm(swarm, colour_data=None, point_size=0.02, point_shader="dot"):
    import k3d
    import plot

    kplot = k3d.plot()
    options = {}
    if colour_data is not None:
        options["attribute"] = colour_data
    with swarm.access():
        coords = np.zeros((swarm.data.shape[0], 3), dtype=np.float32)
        coords[:, 0:2] = swarm.data[:]
        pts = k3d.points(
            positions=coords,
            point_size=point_size,
            color_map=k3d.colormaps.basic_color_maps.RainbowDesaturated,
            **options,
        )
        pts.shader = point_shader
        kplot += pts
    kplot.grid_visible = False
    kplot.display()
    kplot.camera = [-0.2, 0.2, 2.0, 0.0, 0.0, 0.0, -0.5, 1.0, -0.1]  # these are some adhoc settings
    ## attempt at file write out.
    # if filename:
    #     with open(filename,"w") as f:
    #         kplot.fetch_screenshot()
    #         import time
    #         time.sleep(15) # perhaps needed cos async...
    #         f.write(kplot.screenshot)


stokes = uw.systems.Stokes(mesh)

# set force. first create swarm.
swarm = uw.swarm.Swarm(mesh)
density = swarm.add_variable("rho", 1, int)
swarm.populate(fill_param=3)
import numpy as np

with swarm.access(swarm.particle_coordinates):
    factor = 0.1 * cell_size
    swarm.particle_coordinates.data[:] += factor * np.random.rand(*swarm.particle_coordinates.data.shape)


# add no slip boundaries
stokes.add_dirichlet_bc((0.0, 0.0), mesh.boundary.ALL_BOUNDARIES, (0, 1))
# set const visc
stokes.viscosity = 1

import numpy as np
import math

with swarm.access(density):
    density.data[:] = 1.0
    for index, coord in enumerate(swarm.data):
        r = math.sqrt(coord[0] ** 2 + coord[1] ** 2)
        theta = np.arctan2(coord[1], coord[0])
        if r > (0.75 + 0.02 * np.sin(8.0 * theta)):
            density.data[index] = 10.0

with swarm.access():
    plot_swarm(swarm, density.data)

# Plot the proxy for comparison
plot_mesh(mesh, uw.function.evaluate(density.fn, mesh.data[:]))

# +
# file="annulus_rt.h5"
# mesh.save(file)
# density.save(file)
# mesh.generate_xdmf(file)
# -

# construct unit rvec
import sympy

rvec = mesh.rvec
rmag = sympy.sqrt(sympy.vector.dot(rvec, rvec))
rhat = rvec / rmag
rhat


stokes.bodyforce = -rhat * density.fn

stokes.solve()

velmag2 = uw.function.evaluate(stokes.u.fn.dot(stokes.u.fn), mesh.data[:])

plot_mesh(mesh, velmag2)

time = 0.0
time_max = 10.0

while time < time_max:
    dt = stokes.dt()
    with swarm.access(swarm.particle_coordinates):
        swarm.particle_coordinates.data[:] += dt * uw.function.evaluate(stokes.u.fn, swarm.particle_coordinates.data[:])
    time += dt
    stokes.solve()
    print(f"time={time:4.1f}, dt={dt:4.2}")

with swarm.access():
    plot_swarm(swarm, density.data)
