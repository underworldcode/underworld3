# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# # Stripped down Rayleigh Taylor problem
#
# There is a memory leak in the function.evaluate routine when mapping from mesh to swarm.
# There seems to be no memory leak with function.evaluate() on the mesh points.
#
# The visualisation tends to be leaky as well, so I have removed all of that. 
#

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
import numpy as np
import sympy

print(f"Memory usage: {uw.utilities.mem_footprint()} Mb");

cell_size = uw.options.getReal("mesh_cell_size", default=1.0/32)
particle_fill = uw.options.getInt("particle_fill", default=12)
viscosity_ratio = uw.options.getReal("rt_viscosity_ratio", default=1.0)



# +
lightIndex = 0
denseIndex = 1

boxLength = 0.9142
boxHeight = 1.0
viscosityRatio = viscosity_ratio
amplitude = 0.02
offset = 0.2
model_end_time = 300.0

# material perturbation from van Keken et al. 1997
wavelength = 2.0 * boxLength
k = 2.0 * np.pi / wavelength
# -

meshbox = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(boxLength, boxHeight),
    cellSize=cell_size,
    regular=False,
    qdegree=2,
)


# +
import sympy

# Some useful coordinate stuff

x, y = meshbox.CoordinateSystem.X
# -

v_soln = uw.discretisation.MeshVariable(r"U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable(r"P", meshbox, 1, degree=1)
m_cont = uw.discretisation.MeshVariable(r"M_c", meshbox, 1, degree=1, continuous=True)


swarm = uw.swarm.Swarm(mesh=meshbox)
material = uw.swarm.IndexSwarmVariable(
    r"M", swarm, indices=2, proxy_degree=1, proxy_continuous=False
)
swarm.populate(fill_param=particle_fill)


# +
with swarm.access(material):
    material.data[...] = 0

with swarm.access(material):
    perturbation = offset + amplitude * np.cos(
        k * swarm.particle_coordinates.data[:, 0]
    )
    material.data[:, 0] = np.where(
        perturbation > swarm.particle_coordinates.data[:, 1], lightIndex, denseIndex
    )

material.sym
# -


X = meshbox.CoordinateSystem.X

mat_density = np.array([0, 1])  # lightIndex, denseIndex
density = mat_density[0] * material.sym[0] + mat_density[1] * material.sym[1]

mat_viscosity = np.array([viscosityRatio, 1])
viscosity = mat_viscosity[0] * material.sym[0] + mat_viscosity[1] * material.sym[1]

print(f"Memory usage: {uw.utilities.mem_footprint()} Mb");

# +
# Create Stokes object

stokes = uw.systems.Stokes(
    meshbox, velocityField=v_soln, pressureField=p_soln, solver_name="stokes"
)

# Set some things
import sympy
from sympy import Piecewise

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshbox.dim)
stokes.constitutive_model.Parameters.viscosity = viscosity
stokes.bodyforce = sympy.Matrix([0, -density])
stokes.saddle_preconditioner = 1.0 / viscosity

# free slip.
# note with petsc we always need to provide a vector of correct cardinality.
stokes.add_dirichlet_bc(
    (0.0, 0.0), ["Bottom", "Top"], 1
)  # top/bottom: components, function, markers
stokes.add_dirichlet_bc(
    (0.0, 0.0), ["Left", "Right"], 0
)  # left/right: components, function, markers
# -


stokes.rtol = 1.0e-3

stokes.solve(zero_init_guess=True)
delta_t = min(10.0, stokes.estimate_dt())

# +
# Update 

for step in range(0, 10):

    print(f"Memory usage: {uw.utilities.mem_footprint()} Mb");
    
    with meshbox.access():
        print(f"Memory usage 0.1: {uw.utilities.mem_footprint()} Mb");
        uw.function.evaluate(y, meshbox.data)
        print(f"Memory usage 0.2: {uw.utilities.mem_footprint()} Mb");

    
    with swarm.access():
        print(f"Memory usage 1.1: {uw.utilities.mem_footprint()} Mb");
        uw.function.evaluate(x, swarm.particle_coordinates.data)
        print(f"Memory usage 1.2: {uw.utilities.mem_footprint()} Mb");
    
    with swarm.access():
        print(f"Memory usage 2.1: {uw.utilities.mem_footprint()} Mb");
        uw.function.evaluate(v_soln.sym[0], swarm.particle_coordinates.data)
        print(f"Memory usage 2.2: {uw.utilities.mem_footprint()} Mb");

    with meshbox.access():
        print(f"Memory usage 3.1: {uw.utilities.mem_footprint()} Mb");
        uw.function.evaluate(v_soln.sym[0], meshbox.data)
        print(f"Memory usage 3.2: {uw.utilities.mem_footprint()} Mb");

    
    # update swarm / swarm variables

    print(f"Memory usage 4.1: {uw.utilities.mem_footprint()} Mb");
    swarm.advection(v_soln.sym, delta_t)
    print(f"Memory usage 4.2: {uw.utilities.mem_footprint()} Mb");


# -


