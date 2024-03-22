# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Using the PETSc mesh integration routines
#
# This is probably better moved to become a test !

import underworld3 as uw
import numpy as np
import sympy

# +
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0, regular=True)
s_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2) # add this line to avoid petsc error for time being

x = mesh.N.x
y = mesh.N.y
z = mesh.N.z

I = uw.maths.Integral(mesh, x * y)
print(I.evaluate())  # 0.25


# +
mesh = uw.meshing.StructuredQuadBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), elementRes=(32, 32))
s_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2) # add this line to avoid petsc error for time being

x = mesh.N.x
y = mesh.N.y
z = mesh.N.z

I2 = uw.maths.Integral(mesh, x * y)
print(I2.evaluate())  # 0.25

# +
mesh = uw.meshing.StructuredQuadBox(minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), elementRes=(8, 8, 8))
s_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2) # add this line to avoid petsc error for time being

x = mesh.N.x
y = mesh.N.y
z = mesh.N.z

I3 = uw.maths.Integral(mesh, x * y * z)
print(I3.evaluate())  # 0.125

# +
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=1.0 / 8.0, regular=True
)
s_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2) # add this line to avoid petsc error for time being

x = mesh.N.x
y = mesh.N.y
z = mesh.N.z

I4 = uw.maths.Integral(mesh, x * y * z)
print(I4.evaluate())  # 0.125

# +
mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0, cellSize=0.05)
s_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2) # add this line to avoid petsc error for time being

x = mesh.N.x
y = mesh.N.y
z = mesh.N.z

I5 = uw.maths.Integral(mesh, 1.0)
print(I5.evaluate())  # 3 * pi / 4 = 2.35

# +
mesh = uw.meshing.Annulus(radiusInner=0.0, radiusOuter=1.0, cellSize=0.05)
s_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2) # add this line to avoid petsc error for time being

x = mesh.N.x
y = mesh.N.y
z = mesh.N.z

I6 = uw.maths.Integral(mesh, 1.0)
print(I6.evaluate())  # pi

# +
mesh = uw.meshing.SphericalShell(radiusInner=0.5, radiusOuter=1.0, cellSize=0.2)
s_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2) # add this line to avoid petsc error for time being

x = mesh.N.x
y = mesh.N.y
z = mesh.N.z

I7 = uw.maths.Integral(mesh, 1)
print(I7.evaluate())  # 4/3 * 7/8 * pi

# +
mesh = uw.meshing.SphericalShell(radiusInner=0.0, radiusOuter=1.0, cellSize=0.2)
s_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2) # add this line to avoid petsc error for time being

x = mesh.N.x
y = mesh.N.y
z = mesh.N.z

I8 = uw.maths.Integral(mesh, 1)
print(I8.evaluate())  # 4/3 * pi

# +
# mesh = uw.meshing.CubicSphere(radiusInner=0.5, radiusOuter=1.0, numElements=30)
# s_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2) # add this line to avoid petsc error for time being

# x = mesh.N.x
# y = mesh.N.y
# z = mesh.N.z

# I9 = uw.maths.Integral(mesh, 1)
# print(I9.evaluate())  # 4/3 * 7/8 * pi (3.634)

# +
mesh = uw.meshing.SphericalShell(radiusInner=0.0, radiusOuter=1.0, cellSize=0.5)
# mesh = uw.meshing.Annulus(radiusInner = 0.0, radiusOuter=1.0, cellSize=0.05)
s_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2) # add this line to avoid petsc error for time being

x = mesh.N.x
y = mesh.N.y
z = mesh.N.z

radius_fn = sympy.sqrt(mesh.rvec.dot(mesh.rvec))  # normalise by outer radius if not 1.0
unit_rvec = mesh.rvec / (1.0e-10 + radius_fn)

r = sympy.sqrt(x**2 + y**2)
th = sympy.atan2(y + 1.0e-10, x + 1.0e-10)

v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)

swarm = uw.swarm.Swarm(mesh=mesh)
v_star = uw.swarm.SwarmVariable("Vs", swarm, mesh.dim, proxy_degree=2)
remeshed = uw.swarm.SwarmVariable("Vw", swarm, 1, dtype="int", _proxy=False)
X_0 = uw.swarm.SwarmVariable("X0", swarm, mesh.dim, _proxy=False)

swarm.populate(fill_param=4)

I10 = uw.maths.Integral(mesh, 1)
print(I10.evaluate())

stokes = uw.systems.Stokes(
    mesh,
    velocityField=v_soln,
    pressureField=p_soln,
    # u_degree=v_soln.degree,
    # p_degree=p_soln.degree,
    verbose=False,
    solver_name="stokes",
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Upper", (0, 1, 2))

stokes.viscosity = 1.0
stokes.bodyforce = mesh.rvec * sympy.sin(th)


stokes.solve()

I10 = uw.maths.Integral(mesh, v_soln.fn.dot(v_soln.fn))
print(I10.evaluate())


# +
## Check this this sort of thing works OK

mesh = uw.meshing.SphericalShell(radiusInner=0.0, radiusOuter=1.0, cellSize=0.1)

x = mesh.N.x
y = mesh.N.y
z = mesh.N.z

meshvar = uw.discretisation.MeshVariable("phi", mesh, 1, degree=3)

I = uw.maths.Integral(mesh, 1)
print(I.evaluate())  # 4/3 * pi

I.fn = 0.75
print(I.evaluate())  # pi

I.fn = x
print(I.evaluate())  # should be zero

I.fn = x + y + z
print(I.evaluate())  # should be zero

with mesh.access(meshvar):
    meshvar.data[:, 0] = uw.function.evaluate(x + y + z, meshvar.coords)

I.fn = meshvar.fn
print(I.evaluate())  # should be zero
# -


