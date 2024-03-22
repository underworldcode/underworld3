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

#
# # Constitutive relationships in Underworld (pt 1)
#
# Introduction to how the constitutive relationships in Underworld are formulated.

import petsc4py
from petsc4py import PETSc

# %%
import sys


# %%
import underworld3 as uw
from underworld3.systems import Poisson
import numpy as np
import sympy

import pytest
from IPython.display import display  # since pytest runs pure python


# %%
mesh2 = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,0.0), 
                                              maxCoords=(1.0,1.0), 
                                              cellSize=1.0/48.0, 
                                              regular=True)

u2 = uw.discretisation.MeshVariable(r"\mathbf{u2}", mesh2, mesh2.dim, vtype=uw.VarType.VECTOR, degree=2)
p2 = uw.discretisation.MeshVariable(r"p^{(2)}", mesh2, mesh2.dim, vtype=uw.VarType.SCALAR, degree=1)
phi2 = uw.discretisation.MeshVariable(r"\phi^{(2)}", mesh2, 1, vtype=uw.VarType.SCALAR, degree=2)

mesh3 = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,0.0,0.0), 
                                              maxCoords=(1.0,1.0,1.0), 
                                              cellSize=1.0/8.0, 
                                              regular=True)

u3 = uw.discretisation.MeshVariable(r"\mathbf{u3}", mesh3, mesh3.dim, vtype=uw.VarType.VECTOR, degree=2)
p3 = uw.discretisation.MeshVariable(r"p^{(3)}", mesh3, 1, vtype=uw.VarType.SCALAR, degree=1)
phi3 = uw.discretisation.MeshVariable(r"\phi^{(3)}", mesh3, 1, vtype=uw.VarType.SCALAR, degree=2)

R  = sympy.Array(sympy.matrices.rot_axis3(sympy.pi)[0:2,0:2])
R4  = sympy.Array(sympy.matrices.rot_axis3(sympy.pi/4)[0:2,0:2])
R2 = sympy.Array(sympy.matrices.rot_axis3(sympy.pi/2)[0:2,0:2])
R34 = sympy.Array(sympy.matrices.rot_axis3(3*sympy.pi/4)[0:2,0:2])

# %%
poisson2 = uw.systems.Poisson(mesh2, u_Field=phi2)
poisson3 = uw.systems.Poisson(mesh3, u_Field=phi3)

stokes2 = uw.systems.Stokes(mesh2, velocityField=u2, pressureField=p2)
stokes3 = uw.systems.Stokes(mesh3, velocityField=u3, pressureField=p3)

# %%
stokes2

# # Tests 

# The following tests are implemented with pytest. 



# + [markdown] magic_args="[markdown]"
# ## Introduction to constitutive models
#
# Constitutive models relate fluxes of a quantity to the gradients of the unknowns. For example, in thermal diffusion, there is a relationship between heat flux and temperature gradients which is known as *Fourier's law* and which involves an empirically determined thermal conductivity.
#
# $$ q_i = k \frac{ \partial T}{\partial x_i} $$
#
# and in elasticity or fluid flow, we have a relationship between stresses and strain (gradients of the displacements) or strain-rate (gradients of velocity) respectively
#
# $$ \tau_{ij} = \mu \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) $$
#
# where $\eta$, here, is a material property (elastic shear modulus, viscosity) that we need to determine experimentally.
#
# These material properties can be non-linear (they depend upon the quantities that they relate), and they can also be anisotropic (they vary with the orientation of the material). In the latter case, the material property is described through a *consitutive tensor*. For example:
#
# $$ q_i = k_{ij} \frac{ \partial T}{\partial x_j} $$
#
# or
#
# $$ \tau_{ij} = \mu_{ijkl} \left( \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right) $$

# + [markdown] magic_args="[markdown]"
# ## Constitutive tensors
#
# In Underworld, the underlying implementation of any consitutive relationship is through the appropriate tensor representation, allowing for a very general formulation for most problems. The constitutive tensors are described symbolically using `sympy` expressions. For scalar equations, the rank 2 constitutive tensor ($k_{ij}$) can be expressed as a matrix without any loss of generality. 
#
# For vector equations, the constitutive tensor is of rank 4, but it is common in the engineering / finite element literature to transform the rank 4 tensor to a matrix representation. This has some benefits in terms of legibility and, perhaps, results in common code patterns across the formulation for scalar equations. 
#
# We can write the stress and strain rate tensors ($\tau_{ij}$ and $\dot\epsilon_{ij}$ respectively) as
#
# $$ \tau_I = \left( \tau_{00}, \tau_{11}, \tau_{22}, \tau_{12}, \tau_{02}, \tau_{01} \right) $$
#
# $$ \dot\varepsilon_I = \left( \dot\varepsilon_{00}, \dot\varepsilon_{11}, \dot\varepsilon_{22}, \dot\varepsilon_{12}, \dot\varepsilon_{02}, \dot\varepsilon_{01} \right) $$
#
# where $I$ indexes $\{i,j\}$ in a specified order and symmetry of the tensor implies only six (in 3D) unique entries in the nine entries of a general tensor. With these definitions, the constitutive tensor can be written as $C_{IJ}$ where $I$, and $J$ range over the independent degrees of freedom in stress and strain (rate). This is a re-writing of the full tensor, with the necessary symmetry required to respect the symmetry of $\tau$ and $\dot\varepsilon$ 
#
# The simplified notation comes at a cost. For example, $\tau_I \dot\varepsilon_I$ is not equivalent to the tensor inner produce $\tau_{ij} \dot\varepsilon_{ij}$, and $C_{IJ}$ does not transform correctly for a rotation of coordinates. 
#
# However, a modest transformation of the matrix equations is helpful:
#
# $$ \tau^*_{I} = C^*_{IJ} \varepsilon^*_{J} $$ 
#
# Where 
#
# $$\tau^*_{I} = P_{IJ} \tau^*_J$$
#
# $$\varepsilon^*_I = P_{IJ} \varepsilon^*_J$$
#
# $$C^*_{IJ} = P_{IK} C_{KL} P_{LJ}$$
#
# $\mathbf{P}$ is the scaling matrix 
#
# $$P = \left[\begin{matrix}1 & 0 & 0 & 0 & 0 & 0\\0 & 1 & 0 & 0 & 0 & 0\\0 & 0 & 1 & 0 & 0 & 0\\0 & 0 & 0 & \sqrt{2} & 0 & 0\\0 & 0 & 0 & 0 & \sqrt{2} & 0\\0 & 0 & 0 & 0 & 0 & \sqrt{2}\end{matrix}\right]$$
#
# In this form, known as the Mandel notation, $\tau^*_I\varepsilon^*_I \equiv \tau_{ij} \varepsilon_{ij}$, and the fourth order identity matrix:
#
# $$I_{ijkl} = \frac{1}{2} \left( \delta_{ij}\delta_{kj} + \delta_{kl}\delta_{il} \right)$$ 
#
# transforms to 
#
# $$I_{IJ} = \delta_{IJ}$$
#
#
#
#

# + [markdown] magic_args="[markdown]"
# ## Mandel form, sympy tensorial form, Voigt form
#
# The rank 4 tensor form of the constitutive equations, $c_{ijkl}$, has the following representation in `sympy`:
#
# $$\left[\begin{matrix}\left[\begin{matrix}c_{0 0 0 0} & c_{0 0 0 1}\\c_{0 0 1 0} & c_{0 0 1 1}\end{matrix}\right] & \left[\begin{matrix}c_{0 1 0 0} & c_{0 1 0 1}\\c_{0 1 1 0} & c_{0 1 1 1}\end{matrix}\right]\\\left[\begin{matrix}c_{1 0 0 0} & c_{1 0 0 1}\\c_{1 0 1 0} & c_{1 0 1 1}\end{matrix}\right] & \left[\begin{matrix}c_{1 1 0 0} & c_{1 1 0 1}\\c_{1 1 1 0} & c_{1 1 1 1}\end{matrix}\right]\end{matrix}\right]$$
#
# and the inner product $\tau_{ij} = c_{ijkl} \varepsilon_{kl} $ is written
#
# ```python
# tau = sympy.tensorcontraction(
#             sympy.tensorcontraction(
#                    sympy.tensorproduct(c, epsilon),(3,5)),(2,3))
# ```
#
# However, the `sympy.Matrix` module allows a much simpler expression
#
# ```python
# tau_star = C_star * epsilon_star
# tau = P.inv() * tau_star
# ```
#
# which we adopt in `underworld` for the display and manipulation of constitutive tensors.
#
# ### Voigt form
#
# Computation of the stress tensor using $\tau^*_I = C^*_{IJ}\varepsilon^*_J$ is equivalent to the following
#
# $$ \mathbf{P} \mathbf{\tau} =  \mathbf{P} \mathbf{C} \mathbf{P} \cdot \mathbf{P} \mathbf{\varepsilon} $$
#
# multiply by $\mathbf{P}^{-1} $ and collecting terms:
#
# $$ \mathbf{\tau} = \mathbf{C} \mathbf{P}^2 \mathbf{\varepsilon} $$
#
# This is the Voigt form of the constitutive equations and is generally what you will find in a finite element textbook. The Voigt form of the constitutive matrix is the rearrangement of the rank 4 constitutive tensor (with no scaling), the strain rate vector is usually combined with $\mathbf{P}^2$, and the stress vector is raw. For example:
#
#
# $$ \left[\begin{matrix}\tau_{0 0}\\\tau_{1 1}\\\tau_{0 1}\end{matrix}\right] =
#    \left[\begin{matrix}\eta & 0 & 0\\0 & \eta & 0\\0 & 0 & \frac{\eta}{2}\end{matrix}\right]   \left[\begin{matrix}\dot\varepsilon_{0 0}\\\dot\varepsilon_{1 1}\\2 \dot\varepsilon_{0 1}\end{matrix}\right]
# $$
#
# A full discussion of this can be found in Brannon (2018) and Helnwein (2001).
#
#
# ### References. 
#
# Brannon, R. (2018). Rotation Reflection and Frame Changes Orthogonal tensors in computational engineering mechanics. IOP Publishing. https://doi.org/10.1088/978-0-7503-1454-1
#
# Helnwein, P. (2001). Some remarks on the compressed matrix representation of symmetric second-order and fourth-order tensors. Computer Methods in Applied Mechanics and Engineering, 190(22–23), 2753–2770. https://doi.org/10.1016/S0045-7825(00)00263-2
#
#
#
# -

# %%
epsdot = uw.maths.tensor.rank2_symmetric_sym("\\dot\\varepsilon", 2)
display(epsdot)
epsdot_vec = uw.maths.tensor.rank2_to_voigt(epsdot, 2)
display(epsdot_vec)
Pm = uw.maths.tensor.P_mandel[2]

# %%
I4 = uw.maths.tensor.rank4_identity(2)
I4v = uw.maths.tensor.rank4_to_voigt(I4,2)
I4m = uw.maths.tensor.rank4_to_mandel(I4,2)

display(I4v)
display(I4m)

# # What does this show then ?

display(Pm * Pm * epsdot_vec.T)
display(Pm * I4v * Pm * epsdot_vec.T)

# This is the generic constitutive tensor with relevant symmetries for fluid mechanics problems

c4sym = uw.maths.tensor.rank4_symmetric_sym('c', 2)
display(c4sym)

# # This is the mandel form of the constitutive matrix for constant viscosity

eta = sympy.symbols("\eta")
Ceta = sympy.Matrix.diag([eta]*3)
display(Ceta)

# And this is the equivalent tensor
display(uw.maths.tensor.mandel_to_rank4(Ceta, 2))


# %%
Cv = uw.maths.tensor.rank4_to_voigt(c4sym,2)
Cm = uw.maths.tensor.rank4_to_mandel(c4sym,2)

display(Cv)
display(Cm)

# %%
d=3
display(uw.maths.tensor.rank4_to_voigt(uw.maths.tensor.rank4_identity(d), d))
sympy.Array(sympy.symarray('C',(d,d,d,d)))

# # This is how we use those things

ViscousFlow = uw.constitutive_models.TransverseIsotropicFlowModel
ViscousFlow.Parameters.eta_0=sympy.symbols(r"\eta_0")
ViscousFlow.Parameters.eta_1=sympy.symbols(r"\eta_1")
ViscousFlow.Parameters.director=sympy.Matrix([1,0,0]).T




ViscousFlow.flux

stokes3.constitutive_model = ViscousFlow
display(stokes3.strainrate)
display(stokes3.constitutive_model.flux)

# %%
ViscousFlow.Parameters.eta_0=sympy.symbols(r"\eta_0")
ViscousFlow.Parameters.eta_1=sympy.symbols(r"\eta_0")
ViscousFlow.Parameters.director=sympy.Matrix([1,0,0]) # Doesn't matter if the viscosity are the same

stokes3.constitutive_model.flux

# %%
Cmods = uw.constitutive_models.TransverseIsotropicFlowModel(u3)
Cmods.Parameters.eta_0=sympy.symbols(r"\eta_0")
Cmods.Parameters.eta_1=sympy.symbols(r"\eta_1")

Cmods.Parameters.director=sympy.Matrix([1,0,0]).T
display(Cmods.C)

Cmods.Parameters.director=sympy.Matrix([0,1,0]).T
display(Cmods.C)

# %%
# Description / help is especially useful in notebook form
Cmods.view()

# %%
gradT = mesh2.vector.gradient(phi2.sym)
Cmodp = uw.constitutive_models.Constitutive_Model(mesh2.dim, 1)
Cmodp


# %%
Cmodp.Parameters.k = sympy.symbols("\\kappa")
display(Cmodp.C)
display(Cmodp.c)

# %%
Cmodp.flux(gradT)

# %%
Cmods.k = sympy.symbols("\\eta")
Cmods.c

# %%
Cmods

# %%
Cmodv = uw.constitutive_models.ViscousFlowModel(2)
Cmodv.Parameters.viscosity = sympy.symbols(r"\eta")
Cmodv

# %%
Cmodv.flux(epsdot)

# Cvisc = sympy.symbols(r'\eta') * uw.maths.tensor.rank4_identity(2)
# Celas = sympy.symbols(r'\mu') * uw.maths.tensor.rank4_identity(2)

# Cvisc + Celas

# # Equivalence test: define tensor explicitly or in canonical form with rotation

# +
# Rotation (as defined for Muhlhaus / Moresi transverse isotropy)

n = sympy.Matrix(sympy.symarray("n",(3,)))
s = sympy.Matrix((-n[1], n[0], 0))
t = -mesh3.vector.cross(n,s).T # complete the coordinate triad

# -

R = sympy.BlockMatrix((s,n,t)).as_explicit()
R

# # Equivalence test - 2D tensor rotation 

Delta = sympy.symbols(r"\Delta")
lambda_matrix = sympy.Matrix.diag([0,0, Delta])
lambda_ani = uw.maths.tensor.mandel_to_rank4(lambda_matrix,2)
lambda_xyz = sympy.simplify(uw.maths.tensor.tensor_rotation(R2, lambda_ani))
lambda_xyz_m = sympy.simplify(uw.maths.tensor.rank4_to_mandel(lambda_xyz, 2))


lambda_matrix

# # Equivalence test - 2D tensor definition

d=2
lambda_mat2 = uw.maths.tensor.rank4_identity(2) * 0
lambda_mat2

for i in range(d):
    for j in range(d):
        for k in range(d):
            for l in range(d):
                lambda_mat2[i,j,k,l] = Delta * ((n[i]*n[k]*int(j==l) + n[j]*n[k] * int(l==i) + 
                                                 n[i]*n[l]*int(j==k) + n[j]*n[l] * int(k==i))/2 
                                                 - 2 * n[i]*n[j]*n[k]*n[l] )

lambda_mat_m = sympy.simplify(uw.maths.tensor.rank4_to_mandel(lambda_mat2,2))
difference = sympy.simplify(lambda_mat_m - lambda_xyz_m)

difference


# Are they term-by-term equivalent 
sympy.simplify(difference.subs(n[1],sympy.sqrt(1-n[0]**2)))

n = sympy.Matrix(sympy.symarray("n",(3,)))
s = (sympy.Matrix((-n[1], n[0], 0)) + sympy.Matrix((0, -n[2], n[1])))/2 
s /= sympy.sqrt((s.dot(s)))

n_ijk = mesh3.vector.to_vector(n.T)
s_ijk = mesh3.vector.to_vector(s.T)
t = mesh3.vector.to_matrix(n_ijk.cross(s_ijk)).T

R = sympy.BlockMatrix((n,s,t)).as_explicit()
R

# # Equivalence test - 3D tensor rotation 

Delta = sympy.symbols(r"\Delta")
lambda_matrix = sympy.Matrix.diag([0, 0, 0, 0, Delta , Delta])
lambda_ani = uw.maths.tensor.mandel_to_rank4(lambda_matrix,3)
lambda_xyz = sympy.simplify(uw.maths.tensor.tensor_rotation(R, lambda_ani))
lambda_xyz_m = sympy.simplify(uw.maths.tensor.rank4_to_mandel(lambda_xyz, 3))

lambda_matrix


# # Equivalence test - 3D tensor definition

d=3
lambda_mat2 = uw.maths.tensor.rank4_identity(d) * 0



for i in range(d):
    for j in range(d):
        for k in range(d):
            for l in range(d):
                lambda_mat2[i,j,k,l] = Delta * ((n[i]*n[k]*int(j==l) + n[j]*n[k] * int(l==i) + 
                                                 n[i]*n[l]*int(j==k) + n[j]*n[l] * int(k==i))/2 
                                                 - 2 * n[i]*n[j]*n[k]*n[l] )

lambda_mat_m = sympy.simplify(uw.maths.tensor.rank4_to_mandel(lambda_mat2,d))
difference = sympy.simplify(lambda_mat_m - lambda_xyz_m)

# Are they term-by-term equivalent (should be if n is a unit vector)
sympy.simplify(difference.subs(n[2],sympy.sqrt(1-n[0]**2-n[1]**2)))


# +
# Muhlhaus et al 

eta0 = sympy.symbols("\eta_0")
eta1 = sympy.symbols("\eta_1")

lambda_MM = uw.maths.tensor.rank4_identity(3) * 0
lambda_MM_mandel = uw.maths.tensor.rank4_to_mandel(lambda_MM, 3)
lambda_MM_mandel[0,0] = eta0
lambda_MM_mandel[1,1] = eta0
lambda_MM_mandel[2,2] = eta0
lambda_MM_mandel[3,3] = eta0
lambda_MM_mandel[4,4] = eta1
lambda_MM_mandel[5,5] = eta1
lambda_MM = uw.maths.tensor.mandel_to_rank4(lambda_MM_mandel, 3)
lambda_xyz = sympy.simplify(uw.maths.tensor.tensor_rotation(R, lambda_MM))
lambda_xyz_m = sympy.simplify(uw.maths.tensor.rank4_to_mandel(lambda_xyz, 3))

# -

lambda_xyz_m









# Note: the two forms are equivalent, the second is simpler to implement and sympy probably likes it better.

# %%
0/0

# # What does the identity tensor look like in C_ijkl, converted ?

d = 2

I = sympy.MutableDenseNDimArray.zeros(d,d,d,d)

for i in range(d):
    for j in range(d):
        for k in range(d):
            for l in range(d):
                I[i,j,k,l] = eta * sympy.sympify((i==k)*(j==l) + (i==l)*(j==k)) / 2

I

uw.maths.tensor.rank4_to_mandel(I, 2)

uw.maths.tensor.rank4_to_voigt(I, 2)

# %%
uw.maths.tensor.voigt_to_rank4(P.inv() * sympy.Matrix.eye(6) * P.inv(),2)

# %%
E3 = sympy.Matrix(sympy.symarray('\dot\epsilon',(3,3)))
E3[1,0] = E3[0,1] # enforce symmetry
E3[2,0] = E3[0,2] #    --- " ---
E3[2,1] = E3[1,2] #    --- " ---

mesh3.tensor.rank2_to_mandel(E3)

# %%
tau_r = 2 * sympy.tensorcontraction(sympy.tensorcontraction(sympy.tensorproduct(I, E3),(3,5)),(2,3))

tau_r

# %%
0/0

# %%
P*mesh3.tensor.rank4_to_voigt(I)*P

# %%
P * P * Cmd * Em

# %%
P = sympy.Matrix.diag([1, 1, 1, sympy.sqrt(2), sympy.sqrt(2), sympy.sqrt(2)])
eta = sympy.symbols("\eta")
C = sympy.Matrix.diag([eta]*6)

# %%
Pm = 

Cv

# %%
P * Cmods.template_L # this is the strain rate in Mandel form

# %%
0/0

# %%
0/0

# %%
sympy.Matrix(sympy.symarray('\partial\phi,',(d,)))


# %%
M = sympy.Matrix(sympy.symarray("\\left(\\nabla\\mathbf{u}\\right)",(3,)))
M

# %%
Mt = sympy.Matrix(sympy.symarray("\\left(\\nabla\\mathbf{u^*} + \\nabla\\mathbf{u^*}^T\\right)",(3,)))
Mt

# %%
C.is_symmetric()

# %%
isinstance(stokes, uw.systems.SNES_Vector)

# %%
# Set some things
poisson.k = 1. 
poisson.f = 0.
poisson.add_dirichlet_bc( 1., "Bottom" )  
poisson.add_dirichlet_bc( 0., "Top" )  

# %%
poisson._setup_terms()
poisson.snes.setTolerances(rtol=1.0e-6)
poisson.snes.ksp.monitorCancel()

# %%
# Solve time
poisson.solve()

# %%
display(poisson._f1)
display(poisson._L)


# %%
Cmod = uw.constitutive_models.Constitutive_Model(poisson)

# %%
Cmod.template_c

# %%
Cmod3 = uw.constitutive_models.Constitutive_Model(poisson3)
Cmod3.template_c

# %%
Cmods = uw.constitutive_models.Constitutive_Model(stokes)

# %%
Cmods.template_c

# %%
mesh.tensor.rank4_to_voigt(Cmods.template_c)

# %%
import sympy
K = sympy.Matrix.eye(2,2)
K[0,0] = 100

# %%
R = sympy.matrices.rot_axis3(sympy.pi)[0:2,0:2]

# %%
K

# %%
K1 = R.T * K * R

# %%
(K1 * poisson._L.T).T

# + [markdown] magic_args="[markdown]"
# https://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node250.html
#
#
# Would be useful to demonstrate how to do tensorproduct / tensor contraction rotation of a tensor 
# to make sure it works for rank 4 case. 
#
# Define ${\cal R}_{ij}$ as the rotation matrix that maps $x$ onto the $x' $ coordiate system, i.e.
#
#
# $\displaystyle a_i' = {\cal R}_{ij}\,a_j,$, for vectors,
#
# which also has this property:
#
# $\displaystyle {\cal R}_{ki}\,{\cal R}_{kj } = \delta_{ij}.$
#
# Second order tensors transform as follows:
#
# $\displaystyle a_{ij}' = {\cal R}_{ik}\,{\cal R}_{jl}\,a_{kl},$
#
# and for higher rank tensors, we just continue ... 
#
# $\displaystyle a_{ijk}' = {\cal R}_{il}\,{\cal R}_{jm}\,{\cal R}_{kn}\,a_{lmn}.$
#
#
# -

# 2D example - this should be one of the tests.

R  = sympy.Array(sympy.matrices.rot_axis3(sympy.pi)[0:2,0:2])
R4  = sympy.Array(sympy.matrices.rot_axis3(sympy.pi/4)[0:2,0:2])
R2 = sympy.Array(sympy.matrices.rot_axis3(sympy.pi/2)[0:2,0:2])
R34 = sympy.Array(sympy.matrices.rot_axis3(3*sympy.pi/4)[0:2,0:2])


display(R)
display(R2)
display(R4)
display(R34)


# %%
def tensor_rotation(R, T):
    rank = T.rank()
    print("Rank = ",rank)
    
    # Tc = T.copy()
    
    for i in range(rank):
        T = sympy.tensorcontraction(sympy.tensorproduct(R,T),(1,rank+1))
    
    return T


A = sympy.Array(sympy.symarray('a',(2,2)))
C = sympy.Array(sympy.symarray('c',(2,2,2,2)))
V = sympy.Array(sympy.symarray('v',(2,)))

A3 = sympy.Array(sympy.symarray('a',(3,3)))
C3 = sympy.Array(sympy.symarray('c',(3,3,3,3)))
V3 = sympy.Array(sympy.symarray('v',(3,)))


# %%
CC = sympy.MutableDenseNDimArray(0 * C)
eta0 = sympy.symbols("\eta_0")
eta1 = sympy.symbols("\eta_1")

CC[0,0,0,0] = CC[1,1,1,1] = 2 * eta0
CC[0,1,0,1] = CC[1,0,1,0] = eta1
CC[0,1,1,0] = CC[1,0,0,1] = eta1


E = sympy.MutableDenseNDimArray(sympy.symarray('\dot\epsilon',(2,2)))
E[0,1] = E[1,0] # enforce symmetry



CC3 = sympy.MutableDenseNDimArray(0 * C3)
eta30 = sympy.symbols("\eta_0")
eta31 = sympy.symbols("\eta_1")

CC3[0,0,0,0] = CC3[1,1,1,1] = CC3[2,2,2,2] = 2 * eta0

CC3[0,1,0,1] = CC3[1,0,1,0] = eta1
CC3[0,1,1,0] = CC3[1,0,0,1] = eta1

CC3[1,2,1,2] = CC3[2,1,2,1] = eta1
CC3[2,1,1,2] = CC3[1,2,2,1] = eta1

CC3[0,2,0,2] = CC3[2,0,2,0] = eta0
CC3[2,0,0,2] = CC3[0,2,2,0] = eta0

E3 = sympy.MutableDenseNDimArray(sympy.symarray('\dot\epsilon',(3,3)))
E3[1,0] = E3[0,1] # enforce symmetry
E3[2,0] = E3[0,2] #    --- " ---
E3[2,1] = E3[1,2] #    --- " ---

# %%
P = sympy.Matrix.zeros(6,6)
P[0,0] = P[1,1] = P[2,2] = 1
P[3,3] = P[4,4] = P[5,5] = 2

# %%
tau_v = Cv * P * Ev.T
tau_v

# %%
display(voigt_to_rank2(tau_v))
mesh3.tensor.voigt_to_rank2(tau_v)

# %%
tau = sympy.tensorcontraction(sympy.tensorcontraction(sympy.tensorproduct(CC3, E3),(3,5)),(2,3))
display(tau)

# %%
CCr = tensor_rotation(R34, CC)
tau_r = sympy.tensorcontraction(sympy.tensorcontraction(sympy.tensorproduct(CCr, E),(3,5)),(2,3))
tau_r

display(sympy.simplify(CCr))
display(sympy.simplify(tau_r))

# %%
display(tensor_rotation(R2, A))
display(tensor_rotation(R2, V))

A2 = tensor_rotation(R2, A)
A3 = tensor_rotation(R2, A2)




# %%
Crot = tensor_rotation(R2, tensor_rotation(R2, C))

# %%
Crot - C

# %%
C = sympy.tensorcontraction(sympy.tensorproduct(R,C),(1,3))
C.shape
C

# %%
sympy.tensorcontraction(sympy.tensorproduct(R,R),(1,2))

# %%
sympy.tensorproduct(R,R).shape

# %%
R.transpose()

# %%
0/0

# Check. Construct simple linear which is solution for 
# above config.  Exclude boundaries from mesh data. 

import numpy as np
with mesh.access():
    mesh_numerical_soln = uw.function.evaluate(poisson.u.fn, mesh.data)
    mesh_analytic_soln = uw.function.evaluate(1.0-mesh.N.y, mesh.data)
    if not np.allclose(mesh_analytic_soln, mesh_numerical_soln, rtol=0.01):
        raise RuntimeError("Unexpected values encountered.")

# %%
(mesh_analytic_soln - mesh_numerical_soln).max()


# Validate

from mpi4py import MPI

if MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    
    mesh.vtk("ignore_box.vtk")
    pvmesh = pv.read("ignore_box.vtk")
        


    with mesh.access():
        pvmesh.point_data["T"]  = mesh_analytic_soln
        pvmesh.point_data["T2"] = mesh_numerical_soln
        pvmesh.point_data["DT"] = pvmesh.point_data["T"] - pvmesh.point_data["T2"] 
    
    pl = pv.Plotter()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="DT",
                  use_transparency=False, opacity=0.5)
         
    pl.show(cpos="xy")

