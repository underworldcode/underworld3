# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#
# # Anisotropic Constitutive relationships in Underworld (pt 1)
#
# Introduction to how the anisotropic constitutive relationships in Underworld are formulated.
#
# This may be helpful: 
#
#
# ![](https://d32ogoqmya1dw8.cloudfront.net/images/NAGTWorkshops/mineralogy/mineral_physics/table_9.v13.png)
#
#
# (Image from: https://serc.carleton.edu/NAGTWorkshops/mineralogy/mineral_physics/tensors.html)
# This will also be helpful: https://en.wikipedia.org/wiki/Elasticity_tensor
#
# ## Incompressibility
#
# If we apply constraints to the deformation, we expect to reduce the number of independent material constants. Incompressibility should reduce the number of *independent* material constants. In an isotropic medium (or a medium with cubic symmetry), incompressibility eliminates the one volumetric material modulus (e.g. the bulk modulus). In general anisotropic media, it is not the case that changes in pressure result in uniform expansion or contraction, and an incompressibility constraint reduces the number of *independent* material constants. In the transversely isotropic case, there are five independent materia constants in general, reducing to four when the material is incompressible. 
#
# It is not a given that the stiffness matrix is trivial to construct / meaningful for incompressible anisotropy and there is some discussion here: https://rastgaragah.wordpress.com/2013/03/12/incompressibility-of-linearly-elastic-material/ (identifies the issue but no resolution) and this explains in more detail: https://doi.org/10.1016/S0022-5096(01)00121-1. 
#
#
#
#
#

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

# ## Rotations
#
# The expressions for our anisotropic tensors are provided in a Canonical reference frame which exploits the symmetry of the material to present a compact form of the constitutive tensors. In a general orientation, a rotation tensor $\cal R$ is required to transform to / from this canonical frame. For isotropic materials, $\cal R$ is the identity matrix. For cubic materials, a single angle describes the offset of the two frames, and for transversely isotropic materials, we need to define a rotation matrix that aligns the normal to the symmetry plane with the vertical axis in the canonical frame. These examples have fewer degrees of freedom than a full rotation matrix and this fact can be used to simplify the general (rotated) form of the constitutive tensors. 
#
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
# In `underworld` tensor rotation is provided for rank 2 and rank 4 tensors by `uw.maths.tensor.tensor_rotation(R, tensor_expression)`
#
# ### Example
#
#
# First we define the rotation for the transversely isotropic case which depends on the normal vector to the symmetry plane.
#

# A rotation matrix for a transversely isotropic medium is defined by specifying the normal of the symmetry plane ($\hat{\mathbf{n}} = \{ n_0, n_1, n_2\} $). 
# The other orientations are arbitrary, so we simply derive them from $\hat{\mathbf{n}}$ - one vector specified to be perpendicular in the horizontal plane and the third vector is then found from their cross product ($\hat{\mathbf{s}}$ and $\hat{\mathbf{t}}$ respectively)
#
# $$\cal{R} = \left[\begin{matrix}\frac{n_{1}}{\sqrt{n_{0}^{2} + n_{1}^{2}}} & - \frac{n_{0} n_{2}}{\sqrt{n_{0}^{2} + n_{1}^{2}}} & n_{0}\\- \frac{n_{0}}{\sqrt{n_{0}^{2} + n_{1}^{2}}} & - \frac{n_{1} n_{2}}{\sqrt{n_{0}^{2} + n_{1}^{2}}} & n_{1}\\0 & \frac{n_{0}^{2}}{\sqrt{n_{0}^{2} + n_{1}^{2}}} + \frac{n_{1}^{2}}{\sqrt{n_{0}^{2} + n_{1}^{2}}} & n_{2}\end{matrix}\right]$$

# +
## uw / sympy implementation

import underworld3 as uw
import sympy

n = sympy.Matrix(sympy.symarray("n",(3,)))

# Or give a specific value (just specify a vector, then normalise)
# n = sympy.Matrix([1,1,1]) 
# n /= sympy.sqrt(n.dot(n))

if n[0] == 0:
    s = sympy.Matrix([1,0,0])
    t = sympy.Matrix([0,1,0])
    R = sympy.eye(3)

else:
    s = sympy.Matrix((n[1] , -n[0], 0 ))
    s /= sympy.sqrt(s.dot(s))
    t = -n.cross(s) # complete the coordinate triad
    R = sympy.BlockMatrix((s,t,n)).as_explicit()

display(R)

# print(sympy.latex(R, mode='plain'))
# -

# ## Validation
#
# A simple check on this is to rotate the isotropic constitutive tensor and validate that
# it is invariant under rotation. 
#
# $$C_{IJ} = \left[\begin{matrix}2 \eta_{0} & 0 & 0 & 0 & 0 & 0\\0 & 2 \eta_{0} & 0 & 0 & 0 & 0\\0 & 0 & 2 \eta_{0} & 0 & 0 & 0\\0 & 0 & 0 & 2 \eta_{0} & 0 & 0\\0 & 0 & 0 & 0 & 2 \eta_{0} & 0\\0 & 0 & 0 & 0 & 0 & 2 \eta_{0}\end{matrix}\right]$$
#
# Noting that the rotation of the Mandel or Voigt constitutive matrices is complicated by the $\mathbf{P}$ scaling matrices, we compute rotations on the rank 4 tensor, $c_{ijkl}$ and transform to the matrix forms as required. We denote the rotation from $\{IJ\}$ coordinates to $\{I'J'\}$ as
#
# $$C_{I'J'} = {\cal R}[C_{IJ}]$$
#
# The rotated constitutive model has the following form:
#
# $$C_{I'J'} = \left[\begin{matrix}\frac{2 \eta_{0} \left(n_{0}^{2} n_{2}^{2} + n_{0}^{2} \left(n_{0}^{2} + n_{1}^{2}\right) + n_{1}^{2}\right)^{2}}{\left(n_{0}^{2} + n_{1}^{2}\right)^{2}} & \frac{2 \eta_{0} n_{0}^{2} n_{1}^{2} \left(n_{0}^{2} + n_{1}^{2} + n_{2}^{2} - 1\right)^{2}}{\left(n_{0}^{2} + n_{1}^{2}\right)^{2}} & 0 & 0 & 0 & \frac{2 \sqrt{2} \eta_{0} n_{0} n_{1} \left(n_{0}^{6} + 2 n_{0}^{4} n_{1}^{2} + 2 n_{0}^{4} n_{2}^{2} - n_{0}^{4} + n_{0}^{2} n_{1}^{4} + 2 n_{0}^{2} n_{1}^{2} n_{2}^{2} + n_{0}^{2} n_{2}^{4} - n_{0}^{2} n_{2}^{2} + n_{1}^{4} + n_{1}^{2} n_{2}^{2} - n_{1}^{2}\right)}{n_{0}^{4} + 2 n_{0}^{2} n_{1}^{2} + n_{1}^{4}}\\\frac{2 \eta_{0} n_{0}^{2} n_{1}^{2} \left(n_{0}^{2} + n_{1}^{2} + n_{2}^{2} - 1\right)^{2}}{\left(n_{0}^{2} + n_{1}^{2}\right)^{2}} & \frac{2 \eta_{0} \left(n_{0}^{2} + n_{1}^{2} n_{2}^{2} + n_{1}^{2} \left(n_{0}^{2} + n_{1}^{2}\right)\right)^{2}}{\left(n_{0}^{2} + n_{1}^{2}\right)^{2}} & 0 & 0 & 0 & \frac{2 \sqrt{2} \eta_{0} n_{0} n_{1} \left(n_{0}^{4} n_{1}^{2} + n_{0}^{4} + 2 n_{0}^{2} n_{1}^{4} + 2 n_{0}^{2} n_{1}^{2} n_{2}^{2} + n_{0}^{2} n_{2}^{2} - n_{0}^{2} + n_{1}^{6} + 2 n_{1}^{4} n_{2}^{2} - n_{1}^{4} + n_{1}^{2} n_{2}^{4} - n_{1}^{2} n_{2}^{2}\right)}{n_{0}^{4} + 2 n_{0}^{2} n_{1}^{2} + n_{1}^{4}}\\0 & 0 & 2 \eta_{0} \left(n_{0}^{2} + n_{1}^{2} + n_{2}^{2}\right)^{2} & 0 & 0 & 0\\0 & 0 & 0 & \frac{2 \eta_{0} \left(n_{0}^{2} + n_{1}^{2} + n_{2}^{2}\right) \left(n_{0}^{2} + n_{1}^{2} n_{2}^{2} + n_{1}^{2} \left(n_{0}^{2} + n_{1}^{2}\right)\right)}{n_{0}^{2} + n_{1}^{2}} & \frac{2 \eta_{0} n_{0} n_{1} \left(n_{0}^{2} + n_{1}^{2} + n_{2}^{2}\right) \left(n_{0}^{2} + n_{1}^{2} + n_{2}^{2} - 1\right)}{n_{0}^{2} + n_{1}^{2}} & 0\\0 & 0 & 0 & \frac{2 \eta_{0} n_{0} n_{1} \left(n_{0}^{2} + n_{1}^{2} + n_{2}^{2}\right) \left(n_{0}^{2} + n_{1}^{2} + n_{2}^{2} - 1\right)}{n_{0}^{2} + n_{1}^{2}} & \frac{2 \eta_{0} \left(n_{0}^{2} + n_{1}^{2} + n_{2}^{2}\right) \left(n_{0}^{2} n_{2}^{2} + n_{0}^{2} \left(n_{0}^{2} + n_{1}^{2}\right) + n_{1}^{2}\right)}{n_{0}^{2} + n_{1}^{2}} & 0\\\frac{2 \sqrt{2} \eta_{0} n_{0} n_{1} \left(n_{0}^{2} n_{2}^{2} + n_{0}^{2} \left(n_{0}^{2} + n_{1}^{2}\right) + n_{1}^{2}\right) \left(n_{0}^{2} + n_{1}^{2} + n_{2}^{2} - 1\right)}{\left(n_{0}^{2} + n_{1}^{2}\right)^{2}} & \frac{2 \sqrt{2} \eta_{0} n_{0} n_{1} \left(n_{0}^{2} + n_{1}^{2} n_{2}^{2} + n_{1}^{2} \left(n_{0}^{2} + n_{1}^{2}\right)\right) \left(n_{0}^{2} + n_{1}^{2} + n_{2}^{2} - 1\right)}{\left(n_{0}^{2} + n_{1}^{2}\right)^{2}} & 0 & 0 & 0 & \frac{2 \eta_{0} \left(n_{0}^{2} n_{2}^{2} \left(n_{0}^{2} + 2 n_{1}^{2} n_{2}^{2} + 2 n_{1}^{2} \left(n_{0}^{2} + n_{1}^{2}\right) - n_{1}^{2}\right) + n_{0}^{2} \left(n_{0}^{2} + n_{1}^{2}\right) \left(n_{0}^{2} + 2 n_{1}^{2} n_{2}^{2} + 2 n_{1}^{2} \left(n_{0}^{2} + n_{1}^{2}\right) - n_{1}^{2}\right) + n_{1}^{2} \cdot \left(2 n_{0}^{2} + n_{2}^{2} \left(- n_{0}^{2} + n_{1}^{2}\right) - \left(n_{0}^{2} - n_{1}^{2}\right) \left(n_{0}^{2} + n_{1}^{2}\right)\right)\right)}{\left(n_{0}^{2} + n_{1}^{2}\right)^{2}}\end{matrix}\right]$$
#
# However, $\{n_0, n_1, n_2\}$ are not independent because $\hat{\mathbf{n}}$ is a unit vector. If we add this information and simplify, we recover the isotropic form of $C$

# +
# construct a symbolic, isotropic matrix (Mandel form)

eta0 = sympy.symbols("\eta_0")
C_IJm = 2 * sympy.Matrix.diag([eta0]*6)
display(C_IJm)

## Rotate the matrix 

c_ijkl = uw.maths.tensor.mandel_to_rank4(C_IJm, 3)
C_IJv = uw.maths.tensor.rank4_to_voigt(c_ijkl, 3)
c_ijkl_R = sympy.simplify(uw.maths.tensor.tensor_rotation(R, c_ijkl))
C_IJm_R = sympy.simplify(uw.maths.tensor.rank4_to_mandel(c_ijkl_R, 3))
display(C_IJm_R)

## Is this really invariant under rotation ?? 
## Have to do some manipulation to identify the unit-vector component relationships

C_IJm_R_s1 = C_IJm_R.subs(n[0]**2+n[1]**2+n[2]**2,1)
C_IJm_R_s2 = C_IJm_R_s1.subs(n[0]**2+n[1]**2, 1-n[2]**2).applyfunc(sympy.factor)
C_IJm_R_s2.subs(n[0]**2+n[1]**2, 1-n[2]**2).simplify()

# -

# ## Muhlhaus / Moresi transversely isotropic tensor
#
# The Muhlhaus / Moresi transversely isotropic consitituve model is designed to model 
# materials that have a single (usually) weak plane (e.g. an embedded fault). 
#
# The constitutive model is 
#
# $$\left[\begin{matrix}2 \eta_{0} & 0 & 0 & 0 & 0 & 0\\0 & 2 \eta_{0} & 0 & 0 & 0 & 0\\0 & 0 & 2 \eta_{0} & 0 & 0 & 0\\0 & 0 & 0 & - 2 \Delta\eta + 2 \eta_{0} & 0 & 0\\0 & 0 & 0 & 0 & - 2 \Delta\eta + 2 \eta_{0} & 0\\0 & 0 & 0 & 0 & 0 & 2 \eta_{0}\end{matrix}\right]$$
#
# where $\Delta\eta$ represents the change (usually reduction) in viscosity.
#
# Rotation using $\cal R$ as defined from the normal to the weak plane (above) gives (Mandel form):
#
# $$\left[\begin{matrix}- \frac{4 \Delta\eta n_{0}^{2} \left(n_{0}^{2} n_{2}^{2} + n_{1}^{2}\right)}{n_{0}^{2} + n_{1}^{2}} + 2 \eta_{0} & \frac{4 \Delta\eta n_{0}^{2} n_{1}^{2} \cdot \left(1 - n_{2}^{2}\right)}{n_{0}^{2} + n_{1}^{2}} & 4 \Delta\eta n_{0}^{2} n_{2}^{2} & \frac{2 \sqrt{2} \Delta\eta n_{0}^{2} n_{1} n_{2} \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2} + 1\right)}{n_{0}^{2} + n_{1}^{2}} & \frac{2 \sqrt{2} \Delta\eta n_{0} n_{2} \left(n_{0}^{2} \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2}\right) - n_{1}^{2}\right)}{n_{0}^{2} + n_{1}^{2}} & \frac{2 \sqrt{2} \Delta\eta n_{0} n_{1} \left(- 2 n_{0}^{2} n_{2}^{2} + n_{0}^{2} - n_{1}^{2}\right)}{n_{0}^{2} + n_{1}^{2}}\\\frac{4 \Delta\eta n_{0}^{2} n_{1}^{2} \cdot \left(1 - n_{2}^{2}\right)}{n_{0}^{2} + n_{1}^{2}} & - \frac{4 \Delta\eta n_{1}^{2} \left(n_{0}^{2} + n_{1}^{2} n_{2}^{2}\right)}{n_{0}^{2} + n_{1}^{2}} + 2 \eta_{0} & 4 \Delta\eta n_{1}^{2} n_{2}^{2} & - \frac{2 \sqrt{2} \Delta\eta n_{1} n_{2} \left(n_{0}^{2} - n_{1}^{2} \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2}\right)\right)}{n_{0}^{2} + n_{1}^{2}} & \frac{2 \sqrt{2} \Delta\eta n_{0} n_{1}^{2} n_{2} \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2} + 1\right)}{n_{0}^{2} + n_{1}^{2}} & \frac{2 \sqrt{2} \Delta\eta n_{0} n_{1} \left(- n_{0}^{2} - 2 n_{1}^{2} n_{2}^{2} + n_{1}^{2}\right)}{n_{0}^{2} + n_{1}^{2}}\\4 \Delta\eta n_{0}^{2} n_{2}^{2} & 4 \Delta\eta n_{1}^{2} n_{2}^{2} & - 4 \Delta\eta n_{2}^{2} \left(n_{0}^{2} + n_{1}^{2}\right) + 2 \eta_{0} & 2 \sqrt{2} \Delta\eta n_{1} n_{2} \left(- n_{0}^{2} - n_{1}^{2} + n_{2}^{2}\right) & 2 \sqrt{2} \Delta\eta n_{0} n_{2} \left(- n_{0}^{2} - n_{1}^{2} + n_{2}^{2}\right) & 4 \sqrt{2} \Delta\eta n_{0} n_{1} n_{2}^{2}\\\frac{2 \sqrt{2} \Delta\eta n_{0}^{2} n_{1} n_{2} \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2} + 1\right)}{n_{0}^{2} + n_{1}^{2}} & \frac{2 \sqrt{2} \Delta\eta n_{1} n_{2} \left(n_{0}^{2} n_{1}^{2} - n_{0}^{2} + n_{1}^{4} - n_{1}^{2} n_{2}^{2}\right)}{n_{0}^{2} + n_{1}^{2}} & 2 \sqrt{2} \Delta\eta n_{1} n_{2} \left(- n_{0}^{2} - n_{1}^{2} + n_{2}^{2}\right) & - \frac{2 \Delta\eta \left(n_{0}^{2} n_{2}^{2} - n_{1}^{2} n_{2}^{2} \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2}\right) + n_{1}^{2} \left(n_{0}^{2} + n_{1}^{2}\right) \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2}\right)\right)}{n_{0}^{2} + n_{1}^{2}} + 2 \eta_{0} & \frac{2 \Delta\eta n_{0} n_{1} \left(n_{2}^{2} \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2}\right) + n_{2}^{2} - \left(n_{0}^{2} + n_{1}^{2}\right) \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2}\right)\right)}{n_{0}^{2} + n_{1}^{2}} & \frac{2 \Delta\eta n_{0} n_{2} \cdot \left(2 n_{0}^{2} n_{1}^{2} - n_{0}^{2} + 2 n_{1}^{4} - 2 n_{1}^{2} n_{2}^{2} + n_{1}^{2}\right)}{n_{0}^{2} + n_{1}^{2}}\\\frac{2 \sqrt{2} \Delta\eta n_{0} n_{2} \left(n_{0}^{4} + n_{0}^{2} n_{1}^{2} - n_{0}^{2} n_{2}^{2} - n_{1}^{2}\right)}{n_{0}^{2} + n_{1}^{2}} & \frac{2 \sqrt{2} \Delta\eta n_{0} n_{1}^{2} n_{2} \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2} + 1\right)}{n_{0}^{2} + n_{1}^{2}} & 2 \sqrt{2} \Delta\eta n_{0} n_{2} \left(- n_{0}^{2} - n_{1}^{2} + n_{2}^{2}\right) & \frac{2 \Delta\eta n_{0} n_{1} \left(n_{2}^{2} \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2}\right) + n_{2}^{2} - \left(n_{0}^{2} + n_{1}^{2}\right) \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2}\right)\right)}{n_{0}^{2} + n_{1}^{2}} & - \frac{2 \Delta\eta \left(- n_{0}^{2} n_{2}^{2} \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2}\right) + n_{0}^{2} \left(n_{0}^{2} + n_{1}^{2}\right) \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2}\right) + n_{1}^{2} n_{2}^{2}\right)}{n_{0}^{2} + n_{1}^{2}} + 2 \eta_{0} & \frac{2 \Delta\eta n_{1} n_{2} \cdot \left(2 n_{0}^{4} + 2 n_{0}^{2} n_{1}^{2} - 2 n_{0}^{2} n_{2}^{2} + n_{0}^{2} - n_{1}^{2}\right)}{n_{0}^{2} + n_{1}^{2}}\\\frac{2 \sqrt{2} \Delta\eta n_{0} n_{1} \left(- 2 n_{0}^{2} n_{2}^{2} + n_{0}^{2} - n_{1}^{2}\right)}{n_{0}^{2} + n_{1}^{2}} & \frac{2 \sqrt{2} \Delta\eta n_{0} n_{1} \left(- n_{0}^{2} - 2 n_{1}^{2} n_{2}^{2} + n_{1}^{2}\right)}{n_{0}^{2} + n_{1}^{2}} & 4 \sqrt{2} \Delta\eta n_{0} n_{1} n_{2}^{2} & \frac{2 \Delta\eta n_{0} n_{2} \left(- n_{0}^{2} + 2 n_{1}^{2} \left(n_{0}^{2} + n_{1}^{2} - n_{2}^{2}\right) + n_{1}^{2}\right)}{n_{0}^{2} + n_{1}^{2}} & \frac{2 \Delta\eta n_{1} n_{2} \cdot \left(2 n_{0}^{4} + 2 n_{0}^{2} n_{1}^{2} - 2 n_{0}^{2} n_{2}^{2} + n_{0}^{2} - n_{1}^{2}\right)}{n_{0}^{2} + n_{1}^{2}} & \frac{2 \Delta\eta \left(- n_{0}^{4} - 4 n_{0}^{2} n_{1}^{2} n_{2}^{2} + 2 n_{0}^{2} n_{1}^{2} - n_{1}^{4}\right)}{n_{0}^{2} + n_{1}^{2}} + 2 \eta_{0}\end{matrix}\right]$$
#
# and we can easily demonstrate that this collapses back to the isotropic form if $\Delta\eta \leftarrow 0$. We can also show this is equivalent to the alternate expression for the tensor provided in the original papers (an exercise for the reader !!).
#
# The `underworld` implementation is:

# +
# Muhlhaus definition of C_IJ (mandel form)

eta1 = sympy.symbols("\eta_1")
delta_eta = sympy.symbols("\Delta\eta")

C_ijkl_MM = uw.maths.tensor.rank4_identity(3) * 0
C_IJm_MM = uw.maths.tensor.rank4_to_mandel(C_ijkl_MM, 3)
C_IJm_MM[0,0] = 2*eta0
C_IJm_MM[1,1] = 2*eta0
C_IJm_MM[2,2] = 2*eta0
C_IJm_MM[3,3] = 2*(eta0-delta_eta)  # yz
C_IJm_MM[4,4] = 2*(eta0-delta_eta)  # xz
C_IJm_MM[5,5] = 2*eta0  # xy

display(C_IJm_MM)

## We know that the isotropic part is invariant under rotation, so we only need to 
## examine the non-isotropic part.

C_ijkl_MM = uw.maths.tensor.mandel_to_rank4(C_IJm_MM - C_IJm , 3)
C_ijkl_MM_R = sympy.simplify(uw.maths.tensor.tensor_rotation(R, C_ijkl_MM))

C_IJm_MM_R = sympy.simplify(uw.maths.tensor.rank4_to_mandel(C_ijkl_MM_R, 3)) + C_IJm
C_IJv_MM_R = sympy.simplify(uw.maths.tensor.rank4_to_voigt(C_ijkl_MM_R, 3)) + C_IJv

display(C_IJm_MM_R)

# Check what happens if we set delta eta to zero
C_IJm_MM_iso = C_IJm_MM_R.subs(delta_eta,0).applyfunc(sympy.factor).subs(n[0]**2+n[1]**2, 1-n[2]**2).simplify()
display(C_IJm_MM_iso)

# -


# ## Han & Wahr (full transverse isotropic tensor)
#
# In the Han & Wahr paper, the expression for incompressible transverse-isotropy is as follows
#
# $$\left[\begin{matrix}2 \eta_{0} + \mu_{0} & \mu_{0} & 0 & 0 & 0 & 0\\\mu_{0} & 2 \eta_{0} + \mu_{0} & 0 & 0 & 0 & 0\\0 & 0 & - 2 \Delta\eta + 2 \eta_{0} + \mu_{1} & 0 & 0 & 0\\0 & 0 & 0 & - 2 \Delta\eta + 2 \eta_{0} & 0 & 0\\0 & 0 & 0 & 0 & - 2 \Delta\eta + 2 \eta_{0} & 0\\0 & 0 & 0 & 0 & 0 & 2 \eta_{0}\end{matrix}\right]$$
#
# Note that the notation differs from their paper. I have replaced their $\nu1, \nu2$ with $\eta0, \eta1$ to be consistent with the forms defined above. I have replaced their $\eta$ with $\mu$ to avoid the confusion that results from the first change.
#
# Applying the rotation, $\cal R$ and attempting to coerce `sympy` to simplify the constitutive matrix:
#
# $$\left[\begin{matrix}\frac{2 \Delta\eta n_{0}^{4} n_{2}^{4} - 2 \Delta\eta n_{0}^{4} + 4 \Delta\eta n_{0}^{2} n_{1}^{2} n_{2}^{2} - 4 \Delta\eta n_{0}^{2} n_{1}^{2} + 2 \eta_{0} n_{2}^{4} - 4 \eta_{0} n_{2}^{2} + 2 \eta_{0} + \mu_{0} n_{0}^{4} n_{2}^{4} + 2 \mu_{0} n_{0}^{2} n_{1}^{2} n_{2}^{2} + \mu_{0} n_{1}^{4} + \mu_{1} n_{0}^{4} n_{2}^{4} - 2 \mu_{1} n_{0}^{4} n_{2}^{2} + \mu_{1} n_{0}^{4}}{\left(n_{2} - 1\right)^{2} \left(n_{2} + 1\right)^{2}} & \frac{2 \Delta\eta n_{0}^{2} n_{1}^{2} n_{2}^{4} - 4 \Delta\eta n_{0}^{2} n_{1}^{2} n_{2}^{2} + 2 \Delta\eta n_{0}^{2} n_{1}^{2} + \mu_{0} n_{0}^{4} n_{2}^{2} + \mu_{0} n_{0}^{2} n_{1}^{2} n_{2}^{4} + \mu_{0} n_{0}^{2} n_{1}^{2} + \mu_{0} n_{1}^{4} n_{2}^{2} + \mu_{1} n_{0}^{2} n_{1}^{2} n_{2}^{4} - 2 \mu_{1} n_{0}^{2} n_{1}^{2} n_{2}^{2} + \mu_{1} n_{0}^{2} n_{1}^{2}}{\left(n_{2} - 1\right)^{2} \left(n_{2} + 1\right)^{2}} & 2 \Delta\eta n_{0}^{2} n_{2}^{2} + \mu_{0} n_{0}^{2} n_{2}^{2} + \mu_{0} n_{1}^{2} + \mu_{1} n_{0}^{2} n_{2}^{2} & \frac{\sqrt{2} n_{1} n_{2} \cdot \left(2 \Delta\eta n_{0}^{2} n_{2}^{2} - 2 \Delta\eta n_{0}^{2} + \mu_{0} n_{0}^{2} n_{2}^{2} + \mu_{0} n_{1}^{2} - \mu_{1} n_{0}^{4} - \mu_{1} n_{0}^{2} n_{1}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & \frac{\sqrt{2} n_{0} n_{2} \cdot \left(2 \Delta\eta n_{0}^{2} n_{2}^{2} + 2 \Delta\eta n_{1}^{2} + \mu_{0} n_{0}^{2} n_{2}^{2} + \mu_{0} n_{1}^{2} - \mu_{1} n_{0}^{4} - \mu_{1} n_{0}^{2} n_{1}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & \frac{\sqrt{2} n_{0} n_{1} \cdot \left(2 \Delta\eta n_{0}^{2} n_{2}^{2} + 2 \Delta\eta n_{1}^{2} + \mu_{0} n_{0}^{2} n_{2}^{2} + \mu_{0} n_{1}^{2} + \mu_{1} n_{0}^{2} n_{2}^{2} - \mu_{1} n_{0}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)}\\\frac{2 \Delta\eta n_{0}^{2} n_{1}^{2} n_{2}^{4} - 4 \Delta\eta n_{0}^{2} n_{1}^{2} n_{2}^{2} + 2 \Delta\eta n_{0}^{2} n_{1}^{2} + \mu_{0} n_{0}^{4} n_{2}^{2} + \mu_{0} n_{0}^{2} n_{1}^{2} n_{2}^{4} + \mu_{0} n_{0}^{2} n_{1}^{2} + \mu_{0} n_{1}^{4} n_{2}^{2} + \mu_{1} n_{0}^{2} n_{1}^{2} n_{2}^{4} - 2 \mu_{1} n_{0}^{2} n_{1}^{2} n_{2}^{2} + \mu_{1} n_{0}^{2} n_{1}^{2}}{\left(n_{2} - 1\right)^{2} \left(n_{2} + 1\right)^{2}} & \frac{4 \Delta\eta n_{0}^{2} n_{1}^{2} n_{2}^{2} - 4 \Delta\eta n_{0}^{2} n_{1}^{2} + 2 \Delta\eta n_{1}^{4} n_{2}^{4} - 2 \Delta\eta n_{1}^{4} + 2 \eta_{0} n_{2}^{4} - 4 \eta_{0} n_{2}^{2} + 2 \eta_{0} + \mu_{0} n_{0}^{4} + 2 \mu_{0} n_{0}^{2} n_{1}^{2} n_{2}^{2} + \mu_{0} n_{1}^{4} n_{2}^{4} + \mu_{1} n_{1}^{4} n_{2}^{4} - 2 \mu_{1} n_{1}^{4} n_{2}^{2} + \mu_{1} n_{1}^{4}}{\left(n_{2} - 1\right)^{2} \left(n_{2} + 1\right)^{2}} & 2 \Delta\eta n_{1}^{2} n_{2}^{2} + \mu_{0} n_{0}^{2} + \mu_{0} n_{1}^{2} n_{2}^{2} + \mu_{1} n_{1}^{2} n_{2}^{2} & \frac{\sqrt{2} n_{1} n_{2} \cdot \left(2 \Delta\eta n_{0}^{2} + 2 \Delta\eta n_{1}^{2} n_{2}^{2} + \mu_{0} n_{0}^{2} + \mu_{0} n_{1}^{2} n_{2}^{2} - \mu_{1} n_{0}^{2} n_{1}^{2} - \mu_{1} n_{1}^{4}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & \frac{\sqrt{2} n_{0} n_{2} \cdot \left(2 \Delta\eta n_{1}^{2} n_{2}^{2} - 2 \Delta\eta n_{1}^{2} + \mu_{0} n_{0}^{2} + \mu_{0} n_{1}^{2} n_{2}^{2} - \mu_{1} n_{0}^{2} n_{1}^{2} - \mu_{1} n_{1}^{4}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & \frac{\sqrt{2} n_{0} n_{1} \cdot \left(2 \Delta\eta n_{0}^{2} + 2 \Delta\eta n_{1}^{2} n_{2}^{2} + \mu_{0} n_{0}^{2} + \mu_{0} n_{1}^{2} n_{2}^{2} + \mu_{1} n_{1}^{2} n_{2}^{2} - \mu_{1} n_{1}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)}\\2 \Delta\eta n_{0}^{2} n_{2}^{2} + \mu_{0} n_{0}^{2} n_{2}^{2} + \mu_{0} n_{1}^{2} + \mu_{1} n_{0}^{2} n_{2}^{2} & 2 \Delta\eta n_{1}^{2} n_{2}^{2} + \mu_{0} n_{0}^{2} + \mu_{0} n_{1}^{2} n_{2}^{2} + \mu_{1} n_{1}^{2} n_{2}^{2} & 2 \Delta\eta n_{2}^{4} - 4 \Delta\eta n_{2}^{2} + 2 \eta_{0} + \mu_{0} n_{2}^{4} - 2 \mu_{0} n_{2}^{2} + \mu_{0} + \mu_{1} n_{2}^{4} & - \sqrt{2} n_{1} n_{2} \cdot \left(2 \Delta\eta n_{0}^{2} + 2 \Delta\eta n_{1}^{2} + \mu_{0} n_{0}^{2} + \mu_{0} n_{1}^{2} - \mu_{1} n_{2}^{2}\right) & - \sqrt{2} n_{0} n_{2} \cdot \left(2 \Delta\eta n_{0}^{2} + 2 \Delta\eta n_{1}^{2} + \mu_{0} n_{0}^{2} + \mu_{0} n_{1}^{2} - \mu_{1} n_{2}^{2}\right) & \sqrt{2} n_{0} n_{1} \cdot \left(2 \Delta\eta n_{2}^{2} + \mu_{0} n_{2}^{2} - \mu_{0} + \mu_{1} n_{2}^{2}\right)\\\frac{\sqrt{2} n_{1} n_{2} \cdot \left(2 \Delta\eta n_{0}^{2} n_{2}^{2} - 2 \Delta\eta n_{0}^{2} + \mu_{0} n_{0}^{2} n_{2}^{2} + \mu_{0} n_{1}^{2} + \mu_{1} n_{0}^{2} n_{2}^{2} - \mu_{1} n_{0}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & \frac{\sqrt{2} n_{1} n_{2} \cdot \left(2 \Delta\eta n_{0}^{2} + 2 \Delta\eta n_{1}^{2} n_{2}^{2} + \mu_{0} n_{0}^{2} + \mu_{0} n_{1}^{2} n_{2}^{2} + \mu_{1} n_{1}^{2} n_{2}^{2} - \mu_{1} n_{1}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & - \sqrt{2} n_{1} n_{2} \cdot \left(2 \Delta\eta n_{0}^{2} + 2 \Delta\eta n_{1}^{2} + \mu_{0} n_{0}^{2} + \mu_{0} n_{1}^{2} - \mu_{1} n_{2}^{2}\right) & \frac{2 \left(\Delta\eta n_{0}^{2} n_{2}^{2} + 2 \Delta\eta n_{1}^{2} n_{2}^{4} - 2 \Delta\eta n_{1}^{2} n_{2}^{2} + \Delta\eta n_{1}^{2} + \eta_{0} n_{2}^{2} - \eta_{0} + \mu_{0} n_{1}^{2} n_{2}^{4} - \mu_{0} n_{1}^{2} n_{2}^{2} + \mu_{1} n_{1}^{2} n_{2}^{4} - \mu_{1} n_{1}^{2} n_{2}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & 2 n_{0} n_{1} \cdot \left(2 \Delta\eta n_{2}^{2} - \Delta\eta + \mu_{0} n_{2}^{2} + \mu_{1} n_{2}^{2}\right) & \frac{2 n_{0} n_{2} \left(\Delta\eta n_{0}^{2} + 2 \Delta\eta n_{1}^{2} n_{2}^{2} - \Delta\eta n_{1}^{2} + \mu_{0} n_{1}^{2} n_{2}^{2} - \mu_{0} n_{1}^{2} + \mu_{1} n_{1}^{2} n_{2}^{2} - \mu_{1} n_{1}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)}\\\frac{\sqrt{2} n_{0} n_{2} \cdot \left(2 \Delta\eta n_{0}^{2} n_{2}^{2} + 2 \Delta\eta n_{1}^{2} + \mu_{0} n_{0}^{2} n_{2}^{2} + \mu_{0} n_{1}^{2} + \mu_{1} n_{0}^{2} n_{2}^{2} - \mu_{1} n_{0}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & \frac{\sqrt{2} n_{0} n_{2} \cdot \left(2 \Delta\eta n_{1}^{2} n_{2}^{2} - 2 \Delta\eta n_{1}^{2} + \mu_{0} n_{0}^{2} + \mu_{0} n_{1}^{2} n_{2}^{2} + \mu_{1} n_{1}^{2} n_{2}^{2} - \mu_{1} n_{1}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & - \sqrt{2} n_{0} n_{2} \cdot \left(2 \Delta\eta n_{0}^{2} + 2 \Delta\eta n_{1}^{2} + \mu_{0} n_{0}^{2} + \mu_{0} n_{1}^{2} - \mu_{1} n_{2}^{2}\right) & 2 n_{0} n_{1} \cdot \left(2 \Delta\eta n_{2}^{2} - \Delta\eta + \mu_{0} n_{2}^{2} + \mu_{1} n_{2}^{2}\right) & \frac{2 \cdot \left(2 \Delta\eta n_{0}^{2} n_{2}^{4} - 2 \Delta\eta n_{0}^{2} n_{2}^{2} + \Delta\eta n_{0}^{2} + \Delta\eta n_{1}^{2} n_{2}^{2} + \eta_{0} n_{2}^{2} - \eta_{0} + \mu_{0} n_{0}^{2} n_{2}^{4} - \mu_{0} n_{0}^{2} n_{2}^{2} + \mu_{1} n_{0}^{2} n_{2}^{4} - \mu_{1} n_{0}^{2} n_{2}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & \frac{2 n_{1} n_{2} \cdot \left(2 \Delta\eta n_{0}^{2} n_{2}^{2} - \Delta\eta n_{0}^{2} + \Delta\eta n_{1}^{2} + \mu_{0} n_{0}^{2} n_{2}^{2} - \mu_{0} n_{0}^{2} + \mu_{1} n_{0}^{2} n_{2}^{2} - \mu_{1} n_{0}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)}\\\frac{\sqrt{2} n_{0} n_{1} \cdot \left(2 \Delta\eta n_{0}^{2} n_{2}^{2} + 2 \Delta\eta n_{1}^{2} + \mu_{0} n_{0}^{2} n_{2}^{2} + \mu_{0} n_{1}^{2} + \mu_{1} n_{0}^{2} n_{2}^{2} - \mu_{1} n_{0}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & \frac{\sqrt{2} n_{0} n_{1} \cdot \left(2 \Delta\eta n_{0}^{2} + 2 \Delta\eta n_{1}^{2} n_{2}^{2} + \mu_{0} n_{0}^{2} + \mu_{0} n_{1}^{2} n_{2}^{2} + \mu_{1} n_{1}^{2} n_{2}^{2} - \mu_{1} n_{1}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & \sqrt{2} n_{0} n_{1} \cdot \left(2 \Delta\eta n_{2}^{2} + \mu_{0} n_{2}^{2} - \mu_{0} + \mu_{1} n_{2}^{2}\right) & \frac{2 n_{0} n_{2} \left(\Delta\eta n_{0}^{2} + 2 \Delta\eta n_{1}^{2} n_{2}^{2} - \Delta\eta n_{1}^{2} + \mu_{0} n_{1}^{2} n_{2}^{2} - \mu_{0} n_{1}^{2} - \mu_{1} n_{0}^{2} n_{1}^{2} - \mu_{1} n_{1}^{4}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & \frac{2 n_{1} n_{2} \cdot \left(2 \Delta\eta n_{0}^{2} n_{2}^{2} - \Delta\eta n_{0}^{2} + \Delta\eta n_{1}^{2} + \mu_{0} n_{0}^{2} n_{2}^{2} - \mu_{0} n_{0}^{2} - \mu_{1} n_{0}^{4} - \mu_{1} n_{0}^{2} n_{1}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)} & \frac{2 \left(\Delta\eta n_{0}^{4} + 2 \Delta\eta n_{0}^{2} n_{1}^{2} n_{2}^{2} + \Delta\eta n_{1}^{4} + \eta_{0} n_{2}^{2} - \eta_{0} + \mu_{0} n_{0}^{2} n_{1}^{2} n_{2}^{2} - \mu_{0} n_{0}^{2} n_{1}^{2} + \mu_{1} n_{0}^{2} n_{1}^{2} n_{2}^{2} - \mu_{1} n_{0}^{2} n_{1}^{2}\right)}{\left(n_{2} - 1\right) \left(n_{2} + 1\right)}\end{matrix}\right]$$
#
# The `underworld` / `sympy` implementation follows:
#

# +
## The full incompressible, trans-iso model (4 independent unknowns) from Han and Wahr paper

# Extra viscosity terms
mu0 = sympy.symbols(r"\mu_0")
mu1 = sympy.symbols(r"\mu_1")

I = uw.maths.tensor.rank4_identity(3) * 0
C_IJm_HW = uw.maths.tensor.rank4_to_mandel(I, 3)
C_IJm_HW[0,0] = mu0 + 2 * eta0 
C_IJm_HW[0,1] = mu0  
C_IJm_HW[1,0] = mu0  
C_IJm_HW[1,1] = mu0 + 2 * eta0
C_IJm_HW[2,2] = mu1 + 2 * (eta0 - delta_eta)
C_IJm_HW[3,3] = 2 * (eta0 - delta_eta) # yz
C_IJm_HW[4,4] = 2 * (eta0 - delta_eta)  # xz
C_IJm_HW[5,5] = 2 * eta0  # xy

display(C_IJm_HW)
# -

# ## Rotation (symbolic)
#
# Compute the rotation (of the non-isotropic part) 

# +
C_ijkl_HW = uw.maths.tensor.mandel_to_rank4(C_IJm_HW - C_IJm, 3)
display(uw.maths.tensor.rank4_to_mandel(C_ijkl_HW, 3)) 

C_ijkl_HW_R = sympy.simplify(uw.maths.tensor.tensor_rotation(R, C_ijkl_HW))
C_IJm_HW_R = sympy.simplify(uw.maths.tensor.rank4_to_mandel(C_ijkl_HW_R, 3)) + C_IJm

display(C_IJm_HW_R)

# +
## Maybe this can be simplified if we use the unit vector relationships among n0,n1,n2

C_IJm_HW_R_s1 = C_IJm_HW_R.subs(n[0]**2+n[1]**2+n[2]**2,1)
C_IJm_HW_R_s2 = C_IJm_HW_R_s1.subs(n[0]**2+n[1]**2, 1-n[2]**2).applyfunc(sympy.factor)
display(C_IJm_HW_R_s2)

## Yeah, maybe not ... 
# -

# ## Orthotropic medium
#
# **Note** all the caveats above regarding incompressibility. The Browaeys & Chevrot elastic tensors have a bulk modulus term, so it is not completely obvious how to square the assumptions in the first two implementations with this set. 
#
# The full formulation should look like this:
#
# $$\left[\begin{matrix}2 \eta_{00} & 2 \eta_{01} & 2 \eta_{02} & 0 & 0 & 0\\2 \eta_{01} & 2 \eta_{11} & 2 \eta_{12} & 0 & 0 & 0\\2 \eta_{02} & 2 \eta_{12} & 2 \eta_{22} & 0 & 0 & 0\\0 & 0 & 0 & 2 \eta_{33} & 0 & 0\\0 & 0 & 0 & 0 & 2 \eta_{44} & 0\\0 & 0 & 0 & 0 & 0 & 2 \eta_{55}\end{matrix}\right]$$
#
# Rotation in this case should be general as it is no longer enough to specify the symmetry plane. 
#
# $$\left[\begin{matrix}s_{0} & t_{0} & n_{0}\\s_{1} & t_{1} & n_{1}\\s_{2} & t_{2} & n_{2}\end{matrix}\right]$$
#
# $\hat{\mathbf{n}}$,  $\hat{\mathbf{s}}$ and $\hat{\mathbf{t}}$ are an arbitrary orthogonal triad of unit vectors (we keep the notation from the Mühlhaus formulation). It is probably not useful to code up this form.
#
# $$\left[\begin{matrix}2 \eta_{00} s_{0}^{4} + 4 \eta_{01} s_{0}^{2} t_{0}^{2} + 4 \eta_{02} n_{0}^{2} s_{0}^{2} + 2 \eta_{11} t_{0}^{4} + 4 \eta_{12} n_{0}^{2} t_{0}^{2} + 2 \eta_{22} n_{0}^{4} + 4 \eta_{33} n_{0}^{2} t_{0}^{2} + 4 \eta_{44} n_{0}^{2} s_{0}^{2} + 4 \eta_{55} s_{0}^{2} t_{0}^{2} & 2 n_{0} \left(\eta_{33} n_{1} t_{0} t_{1} + \eta_{44} n_{1} s_{0} s_{1} + n_{0} \left(\eta_{02} s_{1}^{2} + \eta_{12} t_{1}^{2} + \eta_{22} n_{1}^{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{0} n_{1} s_{1} + \eta_{55} s_{1} t_{0} t_{1} + s_{0} \left(\eta_{00} s_{1}^{2} + \eta_{01} t_{1}^{2} + \eta_{02} n_{1}^{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{0} n_{1} t_{1} + \eta_{55} s_{0} s_{1} t_{1} + t_{0} \left(\eta_{01} s_{1}^{2} + \eta_{11} t_{1}^{2} + \eta_{12} n_{1}^{2}\right)\right) & 2 n_{0} \left(\eta_{33} n_{2} t_{0} t_{2} + \eta_{44} n_{2} s_{0} s_{2} + n_{0} \left(\eta_{02} s_{2}^{2} + \eta_{12} t_{2}^{2} + \eta_{22} n_{2}^{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{0} n_{2} s_{2} + \eta_{55} s_{2} t_{0} t_{2} + s_{0} \left(\eta_{00} s_{2}^{2} + \eta_{01} t_{2}^{2} + \eta_{02} n_{2}^{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{0} n_{2} t_{2} + \eta_{55} s_{0} s_{2} t_{2} + t_{0} \left(\eta_{01} s_{2}^{2} + \eta_{11} t_{2}^{2} + \eta_{12} n_{2}^{2}\right)\right) & \sqrt{2} \left(n_{0} \left(\eta_{33} t_{0} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{44} s_{0} \left(n_{1} s_{2} + n_{2} s_{1}\right) + 2 n_{0} \left(\eta_{02} s_{1} s_{2} + \eta_{12} t_{1} t_{2} + \eta_{22} n_{1} n_{2}\right)\right) + s_{0} \left(\eta_{44} n_{0} \left(n_{1} s_{2} + n_{2} s_{1}\right) + \eta_{55} t_{0} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 s_{0} \left(\eta_{00} s_{1} s_{2} + \eta_{01} t_{1} t_{2} + \eta_{02} n_{1} n_{2}\right)\right) + t_{0} \left(\eta_{33} n_{0} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{55} s_{0} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 t_{0} \left(\eta_{01} s_{1} s_{2} + \eta_{11} t_{1} t_{2} + \eta_{12} n_{1} n_{2}\right)\right)\right) & \sqrt{2} \left(n_{0} \left(\eta_{33} t_{0} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{44} s_{0} \left(n_{0} s_{2} + n_{2} s_{0}\right) + 2 n_{0} \left(\eta_{02} s_{0} s_{2} + \eta_{12} t_{0} t_{2} + \eta_{22} n_{0} n_{2}\right)\right) + s_{0} \left(\eta_{44} n_{0} \left(n_{0} s_{2} + n_{2} s_{0}\right) + \eta_{55} t_{0} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 s_{0} \left(\eta_{00} s_{0} s_{2} + \eta_{01} t_{0} t_{2} + \eta_{02} n_{0} n_{2}\right)\right) + t_{0} \left(\eta_{33} n_{0} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{55} s_{0} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 t_{0} \left(\eta_{01} s_{0} s_{2} + \eta_{11} t_{0} t_{2} + \eta_{12} n_{0} n_{2}\right)\right)\right) & \sqrt{2} \left(n_{0} \left(\eta_{33} t_{0} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{44} s_{0} \left(n_{0} s_{1} + n_{1} s_{0}\right) + 2 n_{0} \left(\eta_{02} s_{0} s_{1} + \eta_{12} t_{0} t_{1} + \eta_{22} n_{0} n_{1}\right)\right) + s_{0} \left(\eta_{44} n_{0} \left(n_{0} s_{1} + n_{1} s_{0}\right) + \eta_{55} t_{0} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 s_{0} \left(\eta_{00} s_{0} s_{1} + \eta_{01} t_{0} t_{1} + \eta_{02} n_{0} n_{1}\right)\right) + t_{0} \left(\eta_{33} n_{0} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{55} s_{0} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 t_{0} \left(\eta_{01} s_{0} s_{1} + \eta_{11} t_{0} t_{1} + \eta_{12} n_{0} n_{1}\right)\right)\right)\\2 n_{1} \left(\eta_{33} n_{0} t_{0} t_{1} + \eta_{44} n_{0} s_{0} s_{1} + n_{1} \left(\eta_{02} s_{0}^{2} + \eta_{12} t_{0}^{2} + \eta_{22} n_{0}^{2}\right)\right) + 2 s_{1} \left(\eta_{44} n_{0} n_{1} s_{0} + \eta_{55} s_{0} t_{0} t_{1} + s_{1} \left(\eta_{00} s_{0}^{2} + \eta_{01} t_{0}^{2} + \eta_{02} n_{0}^{2}\right)\right) + 2 t_{1} \left(\eta_{33} n_{0} n_{1} t_{0} + \eta_{55} s_{0} s_{1} t_{0} + t_{1} \left(\eta_{01} s_{0}^{2} + \eta_{11} t_{0}^{2} + \eta_{12} n_{0}^{2}\right)\right) & 2 \eta_{00} s_{1}^{4} + 4 \eta_{01} s_{1}^{2} t_{1}^{2} + 4 \eta_{02} n_{1}^{2} s_{1}^{2} + 2 \eta_{11} t_{1}^{4} + 4 \eta_{12} n_{1}^{2} t_{1}^{2} + 2 \eta_{22} n_{1}^{4} + 4 \eta_{33} n_{1}^{2} t_{1}^{2} + 4 \eta_{44} n_{1}^{2} s_{1}^{2} + 4 \eta_{55} s_{1}^{2} t_{1}^{2} & 2 n_{1} \left(\eta_{33} n_{2} t_{1} t_{2} + \eta_{44} n_{2} s_{1} s_{2} + n_{1} \left(\eta_{02} s_{2}^{2} + \eta_{12} t_{2}^{2} + \eta_{22} n_{2}^{2}\right)\right) + 2 s_{1} \left(\eta_{44} n_{1} n_{2} s_{2} + \eta_{55} s_{2} t_{1} t_{2} + s_{1} \left(\eta_{00} s_{2}^{2} + \eta_{01} t_{2}^{2} + \eta_{02} n_{2}^{2}\right)\right) + 2 t_{1} \left(\eta_{33} n_{1} n_{2} t_{2} + \eta_{55} s_{1} s_{2} t_{2} + t_{1} \left(\eta_{01} s_{2}^{2} + \eta_{11} t_{2}^{2} + \eta_{12} n_{2}^{2}\right)\right) & \sqrt{2} \left(n_{1} \left(\eta_{33} t_{1} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{44} s_{1} \left(n_{1} s_{2} + n_{2} s_{1}\right) + 2 n_{1} \left(\eta_{02} s_{1} s_{2} + \eta_{12} t_{1} t_{2} + \eta_{22} n_{1} n_{2}\right)\right) + s_{1} \left(\eta_{44} n_{1} \left(n_{1} s_{2} + n_{2} s_{1}\right) + \eta_{55} t_{1} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 s_{1} \left(\eta_{00} s_{1} s_{2} + \eta_{01} t_{1} t_{2} + \eta_{02} n_{1} n_{2}\right)\right) + t_{1} \left(\eta_{33} n_{1} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{55} s_{1} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 t_{1} \left(\eta_{01} s_{1} s_{2} + \eta_{11} t_{1} t_{2} + \eta_{12} n_{1} n_{2}\right)\right)\right) & \sqrt{2} \left(n_{1} \left(\eta_{33} t_{1} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{44} s_{1} \left(n_{0} s_{2} + n_{2} s_{0}\right) + 2 n_{1} \left(\eta_{02} s_{0} s_{2} + \eta_{12} t_{0} t_{2} + \eta_{22} n_{0} n_{2}\right)\right) + s_{1} \left(\eta_{44} n_{1} \left(n_{0} s_{2} + n_{2} s_{0}\right) + \eta_{55} t_{1} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 s_{1} \left(\eta_{00} s_{0} s_{2} + \eta_{01} t_{0} t_{2} + \eta_{02} n_{0} n_{2}\right)\right) + t_{1} \left(\eta_{33} n_{1} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{55} s_{1} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 t_{1} \left(\eta_{01} s_{0} s_{2} + \eta_{11} t_{0} t_{2} + \eta_{12} n_{0} n_{2}\right)\right)\right) & \sqrt{2} \left(n_{1} \left(\eta_{33} t_{1} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{44} s_{1} \left(n_{0} s_{1} + n_{1} s_{0}\right) + 2 n_{1} \left(\eta_{02} s_{0} s_{1} + \eta_{12} t_{0} t_{1} + \eta_{22} n_{0} n_{1}\right)\right) + s_{1} \left(\eta_{44} n_{1} \left(n_{0} s_{1} + n_{1} s_{0}\right) + \eta_{55} t_{1} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 s_{1} \left(\eta_{00} s_{0} s_{1} + \eta_{01} t_{0} t_{1} + \eta_{02} n_{0} n_{1}\right)\right) + t_{1} \left(\eta_{33} n_{1} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{55} s_{1} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 t_{1} \left(\eta_{01} s_{0} s_{1} + \eta_{11} t_{0} t_{1} + \eta_{12} n_{0} n_{1}\right)\right)\right)\\2 n_{2} \left(\eta_{33} n_{0} t_{0} t_{2} + \eta_{44} n_{0} s_{0} s_{2} + n_{2} \left(\eta_{02} s_{0}^{2} + \eta_{12} t_{0}^{2} + \eta_{22} n_{0}^{2}\right)\right) + 2 s_{2} \left(\eta_{44} n_{0} n_{2} s_{0} + \eta_{55} s_{0} t_{0} t_{2} + s_{2} \left(\eta_{00} s_{0}^{2} + \eta_{01} t_{0}^{2} + \eta_{02} n_{0}^{2}\right)\right) + 2 t_{2} \left(\eta_{33} n_{0} n_{2} t_{0} + \eta_{55} s_{0} s_{2} t_{0} + t_{2} \left(\eta_{01} s_{0}^{2} + \eta_{11} t_{0}^{2} + \eta_{12} n_{0}^{2}\right)\right) & 2 n_{2} \left(\eta_{33} n_{1} t_{1} t_{2} + \eta_{44} n_{1} s_{1} s_{2} + n_{2} \left(\eta_{02} s_{1}^{2} + \eta_{12} t_{1}^{2} + \eta_{22} n_{1}^{2}\right)\right) + 2 s_{2} \left(\eta_{44} n_{1} n_{2} s_{1} + \eta_{55} s_{1} t_{1} t_{2} + s_{2} \left(\eta_{00} s_{1}^{2} + \eta_{01} t_{1}^{2} + \eta_{02} n_{1}^{2}\right)\right) + 2 t_{2} \left(\eta_{33} n_{1} n_{2} t_{1} + \eta_{55} s_{1} s_{2} t_{1} + t_{2} \left(\eta_{01} s_{1}^{2} + \eta_{11} t_{1}^{2} + \eta_{12} n_{1}^{2}\right)\right) & 2 \eta_{00} s_{2}^{4} + 4 \eta_{01} s_{2}^{2} t_{2}^{2} + 4 \eta_{02} n_{2}^{2} s_{2}^{2} + 2 \eta_{11} t_{2}^{4} + 4 \eta_{12} n_{2}^{2} t_{2}^{2} + 2 \eta_{22} n_{2}^{4} + 4 \eta_{33} n_{2}^{2} t_{2}^{2} + 4 \eta_{44} n_{2}^{2} s_{2}^{2} + 4 \eta_{55} s_{2}^{2} t_{2}^{2} & \sqrt{2} \left(n_{2} \left(\eta_{33} t_{2} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{44} s_{2} \left(n_{1} s_{2} + n_{2} s_{1}\right) + 2 n_{2} \left(\eta_{02} s_{1} s_{2} + \eta_{12} t_{1} t_{2} + \eta_{22} n_{1} n_{2}\right)\right) + s_{2} \left(\eta_{44} n_{2} \left(n_{1} s_{2} + n_{2} s_{1}\right) + \eta_{55} t_{2} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 s_{2} \left(\eta_{00} s_{1} s_{2} + \eta_{01} t_{1} t_{2} + \eta_{02} n_{1} n_{2}\right)\right) + t_{2} \left(\eta_{33} n_{2} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{55} s_{2} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 t_{2} \left(\eta_{01} s_{1} s_{2} + \eta_{11} t_{1} t_{2} + \eta_{12} n_{1} n_{2}\right)\right)\right) & \sqrt{2} \left(n_{2} \left(\eta_{33} t_{2} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{44} s_{2} \left(n_{0} s_{2} + n_{2} s_{0}\right) + 2 n_{2} \left(\eta_{02} s_{0} s_{2} + \eta_{12} t_{0} t_{2} + \eta_{22} n_{0} n_{2}\right)\right) + s_{2} \left(\eta_{44} n_{2} \left(n_{0} s_{2} + n_{2} s_{0}\right) + \eta_{55} t_{2} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 s_{2} \left(\eta_{00} s_{0} s_{2} + \eta_{01} t_{0} t_{2} + \eta_{02} n_{0} n_{2}\right)\right) + t_{2} \left(\eta_{33} n_{2} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{55} s_{2} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 t_{2} \left(\eta_{01} s_{0} s_{2} + \eta_{11} t_{0} t_{2} + \eta_{12} n_{0} n_{2}\right)\right)\right) & \sqrt{2} \left(n_{2} \left(\eta_{33} t_{2} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{44} s_{2} \left(n_{0} s_{1} + n_{1} s_{0}\right) + 2 n_{2} \left(\eta_{02} s_{0} s_{1} + \eta_{12} t_{0} t_{1} + \eta_{22} n_{0} n_{1}\right)\right) + s_{2} \left(\eta_{44} n_{2} \left(n_{0} s_{1} + n_{1} s_{0}\right) + \eta_{55} t_{2} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 s_{2} \left(\eta_{00} s_{0} s_{1} + \eta_{01} t_{0} t_{1} + \eta_{02} n_{0} n_{1}\right)\right) + t_{2} \left(\eta_{33} n_{2} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{55} s_{2} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 t_{2} \left(\eta_{01} s_{0} s_{1} + \eta_{11} t_{0} t_{1} + \eta_{12} n_{0} n_{1}\right)\right)\right)\\\sqrt{2} \cdot \left(2 n_{1} \left(\eta_{33} n_{0} t_{0} t_{2} + \eta_{44} n_{0} s_{0} s_{2} + n_{2} \left(\eta_{02} s_{0}^{2} + \eta_{12} t_{0}^{2} + \eta_{22} n_{0}^{2}\right)\right) + 2 s_{1} \left(\eta_{44} n_{0} n_{2} s_{0} + \eta_{55} s_{0} t_{0} t_{2} + s_{2} \left(\eta_{00} s_{0}^{2} + \eta_{01} t_{0}^{2} + \eta_{02} n_{0}^{2}\right)\right) + 2 t_{1} \left(\eta_{33} n_{0} n_{2} t_{0} + \eta_{55} s_{0} s_{2} t_{0} + t_{2} \left(\eta_{01} s_{0}^{2} + \eta_{11} t_{0}^{2} + \eta_{12} n_{0}^{2}\right)\right)\right) & \sqrt{2} \cdot \left(2 n_{1} \left(\eta_{33} n_{1} t_{1} t_{2} + \eta_{44} n_{1} s_{1} s_{2} + n_{2} \left(\eta_{02} s_{1}^{2} + \eta_{12} t_{1}^{2} + \eta_{22} n_{1}^{2}\right)\right) + 2 s_{1} \left(\eta_{44} n_{1} n_{2} s_{1} + \eta_{55} s_{1} t_{1} t_{2} + s_{2} \left(\eta_{00} s_{1}^{2} + \eta_{01} t_{1}^{2} + \eta_{02} n_{1}^{2}\right)\right) + 2 t_{1} \left(\eta_{33} n_{1} n_{2} t_{1} + \eta_{55} s_{1} s_{2} t_{1} + t_{2} \left(\eta_{01} s_{1}^{2} + \eta_{11} t_{1}^{2} + \eta_{12} n_{1}^{2}\right)\right)\right) & \sqrt{2} \cdot \left(2 n_{1} n_{2} \left(\eta_{02} s_{2}^{2} + \eta_{12} t_{2}^{2} + \eta_{22} n_{2}^{2} + \eta_{33} t_{2}^{2} + \eta_{44} s_{2}^{2}\right) + 2 s_{1} s_{2} \left(\eta_{00} s_{2}^{2} + \eta_{01} t_{2}^{2} + \eta_{02} n_{2}^{2} + \eta_{44} n_{2}^{2} + \eta_{55} t_{2}^{2}\right) + 2 t_{1} t_{2} \left(\eta_{01} s_{2}^{2} + \eta_{11} t_{2}^{2} + \eta_{12} n_{2}^{2} + \eta_{33} n_{2}^{2} + \eta_{55} s_{2}^{2}\right)\right) & 2 n_{1} \left(\eta_{33} t_{2} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{44} s_{2} \left(n_{1} s_{2} + n_{2} s_{1}\right) + 2 n_{2} \left(\eta_{02} s_{1} s_{2} + \eta_{12} t_{1} t_{2} + \eta_{22} n_{1} n_{2}\right)\right) + 2 s_{1} \left(\eta_{44} n_{2} \left(n_{1} s_{2} + n_{2} s_{1}\right) + \eta_{55} t_{2} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 s_{2} \left(\eta_{00} s_{1} s_{2} + \eta_{01} t_{1} t_{2} + \eta_{02} n_{1} n_{2}\right)\right) + 2 t_{1} \left(\eta_{33} n_{2} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{55} s_{2} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 t_{2} \left(\eta_{01} s_{1} s_{2} + \eta_{11} t_{1} t_{2} + \eta_{12} n_{1} n_{2}\right)\right) & 2 n_{1} \left(\eta_{33} t_{2} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{44} s_{2} \left(n_{0} s_{2} + n_{2} s_{0}\right) + 2 n_{2} \left(\eta_{02} s_{0} s_{2} + \eta_{12} t_{0} t_{2} + \eta_{22} n_{0} n_{2}\right)\right) + 2 s_{1} \left(\eta_{44} n_{2} \left(n_{0} s_{2} + n_{2} s_{0}\right) + \eta_{55} t_{2} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 s_{2} \left(\eta_{00} s_{0} s_{2} + \eta_{01} t_{0} t_{2} + \eta_{02} n_{0} n_{2}\right)\right) + 2 t_{1} \left(\eta_{33} n_{2} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{55} s_{2} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 t_{2} \left(\eta_{01} s_{0} s_{2} + \eta_{11} t_{0} t_{2} + \eta_{12} n_{0} n_{2}\right)\right) & 2 n_{1} \left(\eta_{33} t_{2} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{44} s_{2} \left(n_{0} s_{1} + n_{1} s_{0}\right) + 2 n_{2} \left(\eta_{02} s_{0} s_{1} + \eta_{12} t_{0} t_{1} + \eta_{22} n_{0} n_{1}\right)\right) + 2 s_{1} \left(\eta_{44} n_{2} \left(n_{0} s_{1} + n_{1} s_{0}\right) + \eta_{55} t_{2} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 s_{2} \left(\eta_{00} s_{0} s_{1} + \eta_{01} t_{0} t_{1} + \eta_{02} n_{0} n_{1}\right)\right) + 2 t_{1} \left(\eta_{33} n_{2} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{55} s_{2} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 t_{2} \left(\eta_{01} s_{0} s_{1} + \eta_{11} t_{0} t_{1} + \eta_{12} n_{0} n_{1}\right)\right)\\\sqrt{2} \cdot \left(2 n_{0} \left(\eta_{33} n_{0} t_{0} t_{2} + \eta_{44} n_{0} s_{0} s_{2} + n_{2} \left(\eta_{02} s_{0}^{2} + \eta_{12} t_{0}^{2} + \eta_{22} n_{0}^{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{0} n_{2} s_{0} + \eta_{55} s_{0} t_{0} t_{2} + s_{2} \left(\eta_{00} s_{0}^{2} + \eta_{01} t_{0}^{2} + \eta_{02} n_{0}^{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{0} n_{2} t_{0} + \eta_{55} s_{0} s_{2} t_{0} + t_{2} \left(\eta_{01} s_{0}^{2} + \eta_{11} t_{0}^{2} + \eta_{12} n_{0}^{2}\right)\right)\right) & \sqrt{2} \cdot \left(2 n_{0} \left(\eta_{33} n_{1} t_{1} t_{2} + \eta_{44} n_{1} s_{1} s_{2} + n_{2} \left(\eta_{02} s_{1}^{2} + \eta_{12} t_{1}^{2} + \eta_{22} n_{1}^{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{1} n_{2} s_{1} + \eta_{55} s_{1} t_{1} t_{2} + s_{2} \left(\eta_{00} s_{1}^{2} + \eta_{01} t_{1}^{2} + \eta_{02} n_{1}^{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{1} n_{2} t_{1} + \eta_{55} s_{1} s_{2} t_{1} + t_{2} \left(\eta_{01} s_{1}^{2} + \eta_{11} t_{1}^{2} + \eta_{12} n_{1}^{2}\right)\right)\right) & \sqrt{2} \cdot \left(2 n_{0} n_{2} \left(\eta_{02} s_{2}^{2} + \eta_{12} t_{2}^{2} + \eta_{22} n_{2}^{2} + \eta_{33} t_{2}^{2} + \eta_{44} s_{2}^{2}\right) + 2 s_{0} s_{2} \left(\eta_{00} s_{2}^{2} + \eta_{01} t_{2}^{2} + \eta_{02} n_{2}^{2} + \eta_{44} n_{2}^{2} + \eta_{55} t_{2}^{2}\right) + 2 t_{0} t_{2} \left(\eta_{01} s_{2}^{2} + \eta_{11} t_{2}^{2} + \eta_{12} n_{2}^{2} + \eta_{33} n_{2}^{2} + \eta_{55} s_{2}^{2}\right)\right) & 2 n_{0} \left(\eta_{33} t_{2} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{44} s_{2} \left(n_{1} s_{2} + n_{2} s_{1}\right) + 2 n_{2} \left(\eta_{02} s_{1} s_{2} + \eta_{12} t_{1} t_{2} + \eta_{22} n_{1} n_{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{2} \left(n_{1} s_{2} + n_{2} s_{1}\right) + \eta_{55} t_{2} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 s_{2} \left(\eta_{00} s_{1} s_{2} + \eta_{01} t_{1} t_{2} + \eta_{02} n_{1} n_{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{2} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{55} s_{2} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 t_{2} \left(\eta_{01} s_{1} s_{2} + \eta_{11} t_{1} t_{2} + \eta_{12} n_{1} n_{2}\right)\right) & 2 n_{0} \left(\eta_{33} t_{2} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{44} s_{2} \left(n_{0} s_{2} + n_{2} s_{0}\right) + 2 n_{2} \left(\eta_{02} s_{0} s_{2} + \eta_{12} t_{0} t_{2} + \eta_{22} n_{0} n_{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{2} \left(n_{0} s_{2} + n_{2} s_{0}\right) + \eta_{55} t_{2} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 s_{2} \left(\eta_{00} s_{0} s_{2} + \eta_{01} t_{0} t_{2} + \eta_{02} n_{0} n_{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{2} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{55} s_{2} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 t_{2} \left(\eta_{01} s_{0} s_{2} + \eta_{11} t_{0} t_{2} + \eta_{12} n_{0} n_{2}\right)\right) & 2 n_{0} \left(\eta_{33} t_{2} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{44} s_{2} \left(n_{0} s_{1} + n_{1} s_{0}\right) + 2 n_{2} \left(\eta_{02} s_{0} s_{1} + \eta_{12} t_{0} t_{1} + \eta_{22} n_{0} n_{1}\right)\right) + 2 s_{0} \left(\eta_{44} n_{2} \left(n_{0} s_{1} + n_{1} s_{0}\right) + \eta_{55} t_{2} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 s_{2} \left(\eta_{00} s_{0} s_{1} + \eta_{01} t_{0} t_{1} + \eta_{02} n_{0} n_{1}\right)\right) + 2 t_{0} \left(\eta_{33} n_{2} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{55} s_{2} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 t_{2} \left(\eta_{01} s_{0} s_{1} + \eta_{11} t_{0} t_{1} + \eta_{12} n_{0} n_{1}\right)\right)\\\sqrt{2} \cdot \left(2 n_{0} \left(\eta_{33} n_{0} t_{0} t_{1} + \eta_{44} n_{0} s_{0} s_{1} + n_{1} \left(\eta_{02} s_{0}^{2} + \eta_{12} t_{0}^{2} + \eta_{22} n_{0}^{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{0} n_{1} s_{0} + \eta_{55} s_{0} t_{0} t_{1} + s_{1} \left(\eta_{00} s_{0}^{2} + \eta_{01} t_{0}^{2} + \eta_{02} n_{0}^{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{0} n_{1} t_{0} + \eta_{55} s_{0} s_{1} t_{0} + t_{1} \left(\eta_{01} s_{0}^{2} + \eta_{11} t_{0}^{2} + \eta_{12} n_{0}^{2}\right)\right)\right) & \sqrt{2} \cdot \left(2 n_{0} n_{1} \left(\eta_{02} s_{1}^{2} + \eta_{12} t_{1}^{2} + \eta_{22} n_{1}^{2} + \eta_{33} t_{1}^{2} + \eta_{44} s_{1}^{2}\right) + 2 s_{0} s_{1} \left(\eta_{00} s_{1}^{2} + \eta_{01} t_{1}^{2} + \eta_{02} n_{1}^{2} + \eta_{44} n_{1}^{2} + \eta_{55} t_{1}^{2}\right) + 2 t_{0} t_{1} \left(\eta_{01} s_{1}^{2} + \eta_{11} t_{1}^{2} + \eta_{12} n_{1}^{2} + \eta_{33} n_{1}^{2} + \eta_{55} s_{1}^{2}\right)\right) & \sqrt{2} \cdot \left(2 n_{0} \left(\eta_{33} n_{2} t_{1} t_{2} + \eta_{44} n_{2} s_{1} s_{2} + n_{1} \left(\eta_{02} s_{2}^{2} + \eta_{12} t_{2}^{2} + \eta_{22} n_{2}^{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{1} n_{2} s_{2} + \eta_{55} s_{2} t_{1} t_{2} + s_{1} \left(\eta_{00} s_{2}^{2} + \eta_{01} t_{2}^{2} + \eta_{02} n_{2}^{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{1} n_{2} t_{2} + \eta_{55} s_{1} s_{2} t_{2} + t_{1} \left(\eta_{01} s_{2}^{2} + \eta_{11} t_{2}^{2} + \eta_{12} n_{2}^{2}\right)\right)\right) & 2 n_{0} \left(\eta_{33} t_{1} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{44} s_{1} \left(n_{1} s_{2} + n_{2} s_{1}\right) + 2 n_{1} \left(\eta_{02} s_{1} s_{2} + \eta_{12} t_{1} t_{2} + \eta_{22} n_{1} n_{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{1} \left(n_{1} s_{2} + n_{2} s_{1}\right) + \eta_{55} t_{1} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 s_{1} \left(\eta_{00} s_{1} s_{2} + \eta_{01} t_{1} t_{2} + \eta_{02} n_{1} n_{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{1} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{55} s_{1} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 t_{1} \left(\eta_{01} s_{1} s_{2} + \eta_{11} t_{1} t_{2} + \eta_{12} n_{1} n_{2}\right)\right) & 2 n_{0} \left(\eta_{33} t_{1} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{44} s_{1} \left(n_{0} s_{2} + n_{2} s_{0}\right) + 2 n_{1} \left(\eta_{02} s_{0} s_{2} + \eta_{12} t_{0} t_{2} + \eta_{22} n_{0} n_{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{1} \left(n_{0} s_{2} + n_{2} s_{0}\right) + \eta_{55} t_{1} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 s_{1} \left(\eta_{00} s_{0} s_{2} + \eta_{01} t_{0} t_{2} + \eta_{02} n_{0} n_{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{1} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{55} s_{1} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 t_{1} \left(\eta_{01} s_{0} s_{2} + \eta_{11} t_{0} t_{2} + \eta_{12} n_{0} n_{2}\right)\right) & 2 n_{0} \left(\eta_{33} t_{1} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{44} s_{1} \left(n_{0} s_{1} + n_{1} s_{0}\right) + 2 n_{1} \left(\eta_{02} s_{0} s_{1} + \eta_{12} t_{0} t_{1} + \eta_{22} n_{0} n_{1}\right)\right) + 2 s_{0} \left(\eta_{44} n_{1} \left(n_{0} s_{1} + n_{1} s_{0}\right) + \eta_{55} t_{1} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 s_{1} \left(\eta_{00} s_{0} s_{1} + \eta_{01} t_{0} t_{1} + \eta_{02} n_{0} n_{1}\right)\right) + 2 t_{0} \left(\eta_{33} n_{1} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{55} s_{1} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 t_{1} \left(\eta_{01} s_{0} s_{1} + \eta_{11} t_{0} t_{1} + \eta_{12} n_{0} n_{1}\right)\right)\end{matrix}\right]$$
#
# In the Browaeys formulation, the orthorhombic part has two unique values (in the Voigt notation this gives three unique entries in the $C_{IJ}$ matrix). The canonical (Mandel) form is
#
# $$\left[\begin{matrix}2 \xi_{0} & 0 & 2 \xi_{1} & 0 & 0 & 0\\0 & - 2 \xi_{0} & - 2 \xi_{1} & 0 & 0 & 0\\2 \xi_{1} & - 2 \xi_{1} & 0 & 0 & 0 & 0\\0 & 0 & 0 & - 2 \xi_{1} & 0 & 0\\0 & 0 & 0 & 0 & 2 \xi_{1} & 0\\0 & 0 & 0 & 0 & 0 & 0\end{matrix}\right]$$
#
# The rotated form:
#
# $$\left[\begin{matrix}2 \eta_{00} s_{0}^{4} + 4 \eta_{01} s_{0}^{2} t_{0}^{2} + 4 \eta_{02} n_{0}^{2} s_{0}^{2} + 2 \eta_{11} t_{0}^{4} + 4 \eta_{12} n_{0}^{2} t_{0}^{2} + 2 \eta_{22} n_{0}^{4} + 4 \eta_{33} n_{0}^{2} t_{0}^{2} + 4 \eta_{44} n_{0}^{2} s_{0}^{2} + 4 \eta_{55} s_{0}^{2} t_{0}^{2} & 2 n_{0} \left(\eta_{33} n_{1} t_{0} t_{1} + \eta_{44} n_{1} s_{0} s_{1} + n_{0} \left(\eta_{02} s_{1}^{2} + \eta_{12} t_{1}^{2} + \eta_{22} n_{1}^{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{0} n_{1} s_{1} + \eta_{55} s_{1} t_{0} t_{1} + s_{0} \left(\eta_{00} s_{1}^{2} + \eta_{01} t_{1}^{2} + \eta_{02} n_{1}^{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{0} n_{1} t_{1} + \eta_{55} s_{0} s_{1} t_{1} + t_{0} \left(\eta_{01} s_{1}^{2} + \eta_{11} t_{1}^{2} + \eta_{12} n_{1}^{2}\right)\right) & 2 n_{0} \left(\eta_{33} n_{2} t_{0} t_{2} + \eta_{44} n_{2} s_{0} s_{2} + n_{0} \left(\eta_{02} s_{2}^{2} + \eta_{12} t_{2}^{2} + \eta_{22} n_{2}^{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{0} n_{2} s_{2} + \eta_{55} s_{2} t_{0} t_{2} + s_{0} \left(\eta_{00} s_{2}^{2} + \eta_{01} t_{2}^{2} + \eta_{02} n_{2}^{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{0} n_{2} t_{2} + \eta_{55} s_{0} s_{2} t_{2} + t_{0} \left(\eta_{01} s_{2}^{2} + \eta_{11} t_{2}^{2} + \eta_{12} n_{2}^{2}\right)\right) & \sqrt{2} \left(n_{0} \left(\eta_{33} t_{0} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{44} s_{0} \left(n_{1} s_{2} + n_{2} s_{1}\right) + 2 n_{0} \left(\eta_{02} s_{1} s_{2} + \eta_{12} t_{1} t_{2} + \eta_{22} n_{1} n_{2}\right)\right) + s_{0} \left(\eta_{44} n_{0} \left(n_{1} s_{2} + n_{2} s_{1}\right) + \eta_{55} t_{0} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 s_{0} \left(\eta_{00} s_{1} s_{2} + \eta_{01} t_{1} t_{2} + \eta_{02} n_{1} n_{2}\right)\right) + t_{0} \left(\eta_{33} n_{0} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{55} s_{0} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 t_{0} \left(\eta_{01} s_{1} s_{2} + \eta_{11} t_{1} t_{2} + \eta_{12} n_{1} n_{2}\right)\right)\right) & \sqrt{2} \left(n_{0} \left(\eta_{33} t_{0} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{44} s_{0} \left(n_{0} s_{2} + n_{2} s_{0}\right) + 2 n_{0} \left(\eta_{02} s_{0} s_{2} + \eta_{12} t_{0} t_{2} + \eta_{22} n_{0} n_{2}\right)\right) + s_{0} \left(\eta_{44} n_{0} \left(n_{0} s_{2} + n_{2} s_{0}\right) + \eta_{55} t_{0} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 s_{0} \left(\eta_{00} s_{0} s_{2} + \eta_{01} t_{0} t_{2} + \eta_{02} n_{0} n_{2}\right)\right) + t_{0} \left(\eta_{33} n_{0} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{55} s_{0} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 t_{0} \left(\eta_{01} s_{0} s_{2} + \eta_{11} t_{0} t_{2} + \eta_{12} n_{0} n_{2}\right)\right)\right) & \sqrt{2} \left(n_{0} \left(\eta_{33} t_{0} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{44} s_{0} \left(n_{0} s_{1} + n_{1} s_{0}\right) + 2 n_{0} \left(\eta_{02} s_{0} s_{1} + \eta_{12} t_{0} t_{1} + \eta_{22} n_{0} n_{1}\right)\right) + s_{0} \left(\eta_{44} n_{0} \left(n_{0} s_{1} + n_{1} s_{0}\right) + \eta_{55} t_{0} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 s_{0} \left(\eta_{00} s_{0} s_{1} + \eta_{01} t_{0} t_{1} + \eta_{02} n_{0} n_{1}\right)\right) + t_{0} \left(\eta_{33} n_{0} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{55} s_{0} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 t_{0} \left(\eta_{01} s_{0} s_{1} + \eta_{11} t_{0} t_{1} + \eta_{12} n_{0} n_{1}\right)\right)\right)\\2 n_{1} \left(\eta_{33} n_{0} t_{0} t_{1} + \eta_{44} n_{0} s_{0} s_{1} + n_{1} \left(\eta_{02} s_{0}^{2} + \eta_{12} t_{0}^{2} + \eta_{22} n_{0}^{2}\right)\right) + 2 s_{1} \left(\eta_{44} n_{0} n_{1} s_{0} + \eta_{55} s_{0} t_{0} t_{1} + s_{1} \left(\eta_{00} s_{0}^{2} + \eta_{01} t_{0}^{2} + \eta_{02} n_{0}^{2}\right)\right) + 2 t_{1} \left(\eta_{33} n_{0} n_{1} t_{0} + \eta_{55} s_{0} s_{1} t_{0} + t_{1} \left(\eta_{01} s_{0}^{2} + \eta_{11} t_{0}^{2} + \eta_{12} n_{0}^{2}\right)\right) & 2 \eta_{00} s_{1}^{4} + 4 \eta_{01} s_{1}^{2} t_{1}^{2} + 4 \eta_{02} n_{1}^{2} s_{1}^{2} + 2 \eta_{11} t_{1}^{4} + 4 \eta_{12} n_{1}^{2} t_{1}^{2} + 2 \eta_{22} n_{1}^{4} + 4 \eta_{33} n_{1}^{2} t_{1}^{2} + 4 \eta_{44} n_{1}^{2} s_{1}^{2} + 4 \eta_{55} s_{1}^{2} t_{1}^{2} & 2 n_{1} \left(\eta_{33} n_{2} t_{1} t_{2} + \eta_{44} n_{2} s_{1} s_{2} + n_{1} \left(\eta_{02} s_{2}^{2} + \eta_{12} t_{2}^{2} + \eta_{22} n_{2}^{2}\right)\right) + 2 s_{1} \left(\eta_{44} n_{1} n_{2} s_{2} + \eta_{55} s_{2} t_{1} t_{2} + s_{1} \left(\eta_{00} s_{2}^{2} + \eta_{01} t_{2}^{2} + \eta_{02} n_{2}^{2}\right)\right) + 2 t_{1} \left(\eta_{33} n_{1} n_{2} t_{2} + \eta_{55} s_{1} s_{2} t_{2} + t_{1} \left(\eta_{01} s_{2}^{2} + \eta_{11} t_{2}^{2} + \eta_{12} n_{2}^{2}\right)\right) & \sqrt{2} \left(n_{1} \left(\eta_{33} t_{1} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{44} s_{1} \left(n_{1} s_{2} + n_{2} s_{1}\right) + 2 n_{1} \left(\eta_{02} s_{1} s_{2} + \eta_{12} t_{1} t_{2} + \eta_{22} n_{1} n_{2}\right)\right) + s_{1} \left(\eta_{44} n_{1} \left(n_{1} s_{2} + n_{2} s_{1}\right) + \eta_{55} t_{1} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 s_{1} \left(\eta_{00} s_{1} s_{2} + \eta_{01} t_{1} t_{2} + \eta_{02} n_{1} n_{2}\right)\right) + t_{1} \left(\eta_{33} n_{1} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{55} s_{1} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 t_{1} \left(\eta_{01} s_{1} s_{2} + \eta_{11} t_{1} t_{2} + \eta_{12} n_{1} n_{2}\right)\right)\right) & \sqrt{2} \left(n_{1} \left(\eta_{33} t_{1} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{44} s_{1} \left(n_{0} s_{2} + n_{2} s_{0}\right) + 2 n_{1} \left(\eta_{02} s_{0} s_{2} + \eta_{12} t_{0} t_{2} + \eta_{22} n_{0} n_{2}\right)\right) + s_{1} \left(\eta_{44} n_{1} \left(n_{0} s_{2} + n_{2} s_{0}\right) + \eta_{55} t_{1} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 s_{1} \left(\eta_{00} s_{0} s_{2} + \eta_{01} t_{0} t_{2} + \eta_{02} n_{0} n_{2}\right)\right) + t_{1} \left(\eta_{33} n_{1} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{55} s_{1} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 t_{1} \left(\eta_{01} s_{0} s_{2} + \eta_{11} t_{0} t_{2} + \eta_{12} n_{0} n_{2}\right)\right)\right) & \sqrt{2} \left(n_{1} \left(\eta_{33} t_{1} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{44} s_{1} \left(n_{0} s_{1} + n_{1} s_{0}\right) + 2 n_{1} \left(\eta_{02} s_{0} s_{1} + \eta_{12} t_{0} t_{1} + \eta_{22} n_{0} n_{1}\right)\right) + s_{1} \left(\eta_{44} n_{1} \left(n_{0} s_{1} + n_{1} s_{0}\right) + \eta_{55} t_{1} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 s_{1} \left(\eta_{00} s_{0} s_{1} + \eta_{01} t_{0} t_{1} + \eta_{02} n_{0} n_{1}\right)\right) + t_{1} \left(\eta_{33} n_{1} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{55} s_{1} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 t_{1} \left(\eta_{01} s_{0} s_{1} + \eta_{11} t_{0} t_{1} + \eta_{12} n_{0} n_{1}\right)\right)\right)\\2 n_{2} \left(\eta_{33} n_{0} t_{0} t_{2} + \eta_{44} n_{0} s_{0} s_{2} + n_{2} \left(\eta_{02} s_{0}^{2} + \eta_{12} t_{0}^{2} + \eta_{22} n_{0}^{2}\right)\right) + 2 s_{2} \left(\eta_{44} n_{0} n_{2} s_{0} + \eta_{55} s_{0} t_{0} t_{2} + s_{2} \left(\eta_{00} s_{0}^{2} + \eta_{01} t_{0}^{2} + \eta_{02} n_{0}^{2}\right)\right) + 2 t_{2} \left(\eta_{33} n_{0} n_{2} t_{0} + \eta_{55} s_{0} s_{2} t_{0} + t_{2} \left(\eta_{01} s_{0}^{2} + \eta_{11} t_{0}^{2} + \eta_{12} n_{0}^{2}\right)\right) & 2 n_{2} \left(\eta_{33} n_{1} t_{1} t_{2} + \eta_{44} n_{1} s_{1} s_{2} + n_{2} \left(\eta_{02} s_{1}^{2} + \eta_{12} t_{1}^{2} + \eta_{22} n_{1}^{2}\right)\right) + 2 s_{2} \left(\eta_{44} n_{1} n_{2} s_{1} + \eta_{55} s_{1} t_{1} t_{2} + s_{2} \left(\eta_{00} s_{1}^{2} + \eta_{01} t_{1}^{2} + \eta_{02} n_{1}^{2}\right)\right) + 2 t_{2} \left(\eta_{33} n_{1} n_{2} t_{1} + \eta_{55} s_{1} s_{2} t_{1} + t_{2} \left(\eta_{01} s_{1}^{2} + \eta_{11} t_{1}^{2} + \eta_{12} n_{1}^{2}\right)\right) & 2 \eta_{00} s_{2}^{4} + 4 \eta_{01} s_{2}^{2} t_{2}^{2} + 4 \eta_{02} n_{2}^{2} s_{2}^{2} + 2 \eta_{11} t_{2}^{4} + 4 \eta_{12} n_{2}^{2} t_{2}^{2} + 2 \eta_{22} n_{2}^{4} + 4 \eta_{33} n_{2}^{2} t_{2}^{2} + 4 \eta_{44} n_{2}^{2} s_{2}^{2} + 4 \eta_{55} s_{2}^{2} t_{2}^{2} & \sqrt{2} \left(n_{2} \left(\eta_{33} t_{2} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{44} s_{2} \left(n_{1} s_{2} + n_{2} s_{1}\right) + 2 n_{2} \left(\eta_{02} s_{1} s_{2} + \eta_{12} t_{1} t_{2} + \eta_{22} n_{1} n_{2}\right)\right) + s_{2} \left(\eta_{44} n_{2} \left(n_{1} s_{2} + n_{2} s_{1}\right) + \eta_{55} t_{2} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 s_{2} \left(\eta_{00} s_{1} s_{2} + \eta_{01} t_{1} t_{2} + \eta_{02} n_{1} n_{2}\right)\right) + t_{2} \left(\eta_{33} n_{2} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{55} s_{2} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 t_{2} \left(\eta_{01} s_{1} s_{2} + \eta_{11} t_{1} t_{2} + \eta_{12} n_{1} n_{2}\right)\right)\right) & \sqrt{2} \left(n_{2} \left(\eta_{33} t_{2} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{44} s_{2} \left(n_{0} s_{2} + n_{2} s_{0}\right) + 2 n_{2} \left(\eta_{02} s_{0} s_{2} + \eta_{12} t_{0} t_{2} + \eta_{22} n_{0} n_{2}\right)\right) + s_{2} \left(\eta_{44} n_{2} \left(n_{0} s_{2} + n_{2} s_{0}\right) + \eta_{55} t_{2} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 s_{2} \left(\eta_{00} s_{0} s_{2} + \eta_{01} t_{0} t_{2} + \eta_{02} n_{0} n_{2}\right)\right) + t_{2} \left(\eta_{33} n_{2} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{55} s_{2} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 t_{2} \left(\eta_{01} s_{0} s_{2} + \eta_{11} t_{0} t_{2} + \eta_{12} n_{0} n_{2}\right)\right)\right) & \sqrt{2} \left(n_{2} \left(\eta_{33} t_{2} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{44} s_{2} \left(n_{0} s_{1} + n_{1} s_{0}\right) + 2 n_{2} \left(\eta_{02} s_{0} s_{1} + \eta_{12} t_{0} t_{1} + \eta_{22} n_{0} n_{1}\right)\right) + s_{2} \left(\eta_{44} n_{2} \left(n_{0} s_{1} + n_{1} s_{0}\right) + \eta_{55} t_{2} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 s_{2} \left(\eta_{00} s_{0} s_{1} + \eta_{01} t_{0} t_{1} + \eta_{02} n_{0} n_{1}\right)\right) + t_{2} \left(\eta_{33} n_{2} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{55} s_{2} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 t_{2} \left(\eta_{01} s_{0} s_{1} + \eta_{11} t_{0} t_{1} + \eta_{12} n_{0} n_{1}\right)\right)\right)\\\sqrt{2} \cdot \left(2 n_{1} \left(\eta_{33} n_{0} t_{0} t_{2} + \eta_{44} n_{0} s_{0} s_{2} + n_{2} \left(\eta_{02} s_{0}^{2} + \eta_{12} t_{0}^{2} + \eta_{22} n_{0}^{2}\right)\right) + 2 s_{1} \left(\eta_{44} n_{0} n_{2} s_{0} + \eta_{55} s_{0} t_{0} t_{2} + s_{2} \left(\eta_{00} s_{0}^{2} + \eta_{01} t_{0}^{2} + \eta_{02} n_{0}^{2}\right)\right) + 2 t_{1} \left(\eta_{33} n_{0} n_{2} t_{0} + \eta_{55} s_{0} s_{2} t_{0} + t_{2} \left(\eta_{01} s_{0}^{2} + \eta_{11} t_{0}^{2} + \eta_{12} n_{0}^{2}\right)\right)\right) & \sqrt{2} \cdot \left(2 n_{1} \left(\eta_{33} n_{1} t_{1} t_{2} + \eta_{44} n_{1} s_{1} s_{2} + n_{2} \left(\eta_{02} s_{1}^{2} + \eta_{12} t_{1}^{2} + \eta_{22} n_{1}^{2}\right)\right) + 2 s_{1} \left(\eta_{44} n_{1} n_{2} s_{1} + \eta_{55} s_{1} t_{1} t_{2} + s_{2} \left(\eta_{00} s_{1}^{2} + \eta_{01} t_{1}^{2} + \eta_{02} n_{1}^{2}\right)\right) + 2 t_{1} \left(\eta_{33} n_{1} n_{2} t_{1} + \eta_{55} s_{1} s_{2} t_{1} + t_{2} \left(\eta_{01} s_{1}^{2} + \eta_{11} t_{1}^{2} + \eta_{12} n_{1}^{2}\right)\right)\right) & \sqrt{2} \cdot \left(2 n_{1} n_{2} \left(\eta_{02} s_{2}^{2} + \eta_{12} t_{2}^{2} + \eta_{22} n_{2}^{2} + \eta_{33} t_{2}^{2} + \eta_{44} s_{2}^{2}\right) + 2 s_{1} s_{2} \left(\eta_{00} s_{2}^{2} + \eta_{01} t_{2}^{2} + \eta_{02} n_{2}^{2} + \eta_{44} n_{2}^{2} + \eta_{55} t_{2}^{2}\right) + 2 t_{1} t_{2} \left(\eta_{01} s_{2}^{2} + \eta_{11} t_{2}^{2} + \eta_{12} n_{2}^{2} + \eta_{33} n_{2}^{2} + \eta_{55} s_{2}^{2}\right)\right) & 2 n_{1} \left(\eta_{33} t_{2} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{44} s_{2} \left(n_{1} s_{2} + n_{2} s_{1}\right) + 2 n_{2} \left(\eta_{02} s_{1} s_{2} + \eta_{12} t_{1} t_{2} + \eta_{22} n_{1} n_{2}\right)\right) + 2 s_{1} \left(\eta_{44} n_{2} \left(n_{1} s_{2} + n_{2} s_{1}\right) + \eta_{55} t_{2} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 s_{2} \left(\eta_{00} s_{1} s_{2} + \eta_{01} t_{1} t_{2} + \eta_{02} n_{1} n_{2}\right)\right) + 2 t_{1} \left(\eta_{33} n_{2} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{55} s_{2} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 t_{2} \left(\eta_{01} s_{1} s_{2} + \eta_{11} t_{1} t_{2} + \eta_{12} n_{1} n_{2}\right)\right) & 2 n_{1} \left(\eta_{33} t_{2} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{44} s_{2} \left(n_{0} s_{2} + n_{2} s_{0}\right) + 2 n_{2} \left(\eta_{02} s_{0} s_{2} + \eta_{12} t_{0} t_{2} + \eta_{22} n_{0} n_{2}\right)\right) + 2 s_{1} \left(\eta_{44} n_{2} \left(n_{0} s_{2} + n_{2} s_{0}\right) + \eta_{55} t_{2} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 s_{2} \left(\eta_{00} s_{0} s_{2} + \eta_{01} t_{0} t_{2} + \eta_{02} n_{0} n_{2}\right)\right) + 2 t_{1} \left(\eta_{33} n_{2} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{55} s_{2} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 t_{2} \left(\eta_{01} s_{0} s_{2} + \eta_{11} t_{0} t_{2} + \eta_{12} n_{0} n_{2}\right)\right) & 2 n_{1} \left(\eta_{33} t_{2} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{44} s_{2} \left(n_{0} s_{1} + n_{1} s_{0}\right) + 2 n_{2} \left(\eta_{02} s_{0} s_{1} + \eta_{12} t_{0} t_{1} + \eta_{22} n_{0} n_{1}\right)\right) + 2 s_{1} \left(\eta_{44} n_{2} \left(n_{0} s_{1} + n_{1} s_{0}\right) + \eta_{55} t_{2} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 s_{2} \left(\eta_{00} s_{0} s_{1} + \eta_{01} t_{0} t_{1} + \eta_{02} n_{0} n_{1}\right)\right) + 2 t_{1} \left(\eta_{33} n_{2} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{55} s_{2} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 t_{2} \left(\eta_{01} s_{0} s_{1} + \eta_{11} t_{0} t_{1} + \eta_{12} n_{0} n_{1}\right)\right)\\\sqrt{2} \cdot \left(2 n_{0} \left(\eta_{33} n_{0} t_{0} t_{2} + \eta_{44} n_{0} s_{0} s_{2} + n_{2} \left(\eta_{02} s_{0}^{2} + \eta_{12} t_{0}^{2} + \eta_{22} n_{0}^{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{0} n_{2} s_{0} + \eta_{55} s_{0} t_{0} t_{2} + s_{2} \left(\eta_{00} s_{0}^{2} + \eta_{01} t_{0}^{2} + \eta_{02} n_{0}^{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{0} n_{2} t_{0} + \eta_{55} s_{0} s_{2} t_{0} + t_{2} \left(\eta_{01} s_{0}^{2} + \eta_{11} t_{0}^{2} + \eta_{12} n_{0}^{2}\right)\right)\right) & \sqrt{2} \cdot \left(2 n_{0} \left(\eta_{33} n_{1} t_{1} t_{2} + \eta_{44} n_{1} s_{1} s_{2} + n_{2} \left(\eta_{02} s_{1}^{2} + \eta_{12} t_{1}^{2} + \eta_{22} n_{1}^{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{1} n_{2} s_{1} + \eta_{55} s_{1} t_{1} t_{2} + s_{2} \left(\eta_{00} s_{1}^{2} + \eta_{01} t_{1}^{2} + \eta_{02} n_{1}^{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{1} n_{2} t_{1} + \eta_{55} s_{1} s_{2} t_{1} + t_{2} \left(\eta_{01} s_{1}^{2} + \eta_{11} t_{1}^{2} + \eta_{12} n_{1}^{2}\right)\right)\right) & \sqrt{2} \cdot \left(2 n_{0} n_{2} \left(\eta_{02} s_{2}^{2} + \eta_{12} t_{2}^{2} + \eta_{22} n_{2}^{2} + \eta_{33} t_{2}^{2} + \eta_{44} s_{2}^{2}\right) + 2 s_{0} s_{2} \left(\eta_{00} s_{2}^{2} + \eta_{01} t_{2}^{2} + \eta_{02} n_{2}^{2} + \eta_{44} n_{2}^{2} + \eta_{55} t_{2}^{2}\right) + 2 t_{0} t_{2} \left(\eta_{01} s_{2}^{2} + \eta_{11} t_{2}^{2} + \eta_{12} n_{2}^{2} + \eta_{33} n_{2}^{2} + \eta_{55} s_{2}^{2}\right)\right) & 2 n_{0} \left(\eta_{33} t_{2} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{44} s_{2} \left(n_{1} s_{2} + n_{2} s_{1}\right) + 2 n_{2} \left(\eta_{02} s_{1} s_{2} + \eta_{12} t_{1} t_{2} + \eta_{22} n_{1} n_{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{2} \left(n_{1} s_{2} + n_{2} s_{1}\right) + \eta_{55} t_{2} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 s_{2} \left(\eta_{00} s_{1} s_{2} + \eta_{01} t_{1} t_{2} + \eta_{02} n_{1} n_{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{2} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{55} s_{2} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 t_{2} \left(\eta_{01} s_{1} s_{2} + \eta_{11} t_{1} t_{2} + \eta_{12} n_{1} n_{2}\right)\right) & 2 n_{0} \left(\eta_{33} t_{2} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{44} s_{2} \left(n_{0} s_{2} + n_{2} s_{0}\right) + 2 n_{2} \left(\eta_{02} s_{0} s_{2} + \eta_{12} t_{0} t_{2} + \eta_{22} n_{0} n_{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{2} \left(n_{0} s_{2} + n_{2} s_{0}\right) + \eta_{55} t_{2} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 s_{2} \left(\eta_{00} s_{0} s_{2} + \eta_{01} t_{0} t_{2} + \eta_{02} n_{0} n_{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{2} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{55} s_{2} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 t_{2} \left(\eta_{01} s_{0} s_{2} + \eta_{11} t_{0} t_{2} + \eta_{12} n_{0} n_{2}\right)\right) & 2 n_{0} \left(\eta_{33} t_{2} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{44} s_{2} \left(n_{0} s_{1} + n_{1} s_{0}\right) + 2 n_{2} \left(\eta_{02} s_{0} s_{1} + \eta_{12} t_{0} t_{1} + \eta_{22} n_{0} n_{1}\right)\right) + 2 s_{0} \left(\eta_{44} n_{2} \left(n_{0} s_{1} + n_{1} s_{0}\right) + \eta_{55} t_{2} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 s_{2} \left(\eta_{00} s_{0} s_{1} + \eta_{01} t_{0} t_{1} + \eta_{02} n_{0} n_{1}\right)\right) + 2 t_{0} \left(\eta_{33} n_{2} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{55} s_{2} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 t_{2} \left(\eta_{01} s_{0} s_{1} + \eta_{11} t_{0} t_{1} + \eta_{12} n_{0} n_{1}\right)\right)\\\sqrt{2} \cdot \left(2 n_{0} \left(\eta_{33} n_{0} t_{0} t_{1} + \eta_{44} n_{0} s_{0} s_{1} + n_{1} \left(\eta_{02} s_{0}^{2} + \eta_{12} t_{0}^{2} + \eta_{22} n_{0}^{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{0} n_{1} s_{0} + \eta_{55} s_{0} t_{0} t_{1} + s_{1} \left(\eta_{00} s_{0}^{2} + \eta_{01} t_{0}^{2} + \eta_{02} n_{0}^{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{0} n_{1} t_{0} + \eta_{55} s_{0} s_{1} t_{0} + t_{1} \left(\eta_{01} s_{0}^{2} + \eta_{11} t_{0}^{2} + \eta_{12} n_{0}^{2}\right)\right)\right) & \sqrt{2} \cdot \left(2 n_{0} n_{1} \left(\eta_{02} s_{1}^{2} + \eta_{12} t_{1}^{2} + \eta_{22} n_{1}^{2} + \eta_{33} t_{1}^{2} + \eta_{44} s_{1}^{2}\right) + 2 s_{0} s_{1} \left(\eta_{00} s_{1}^{2} + \eta_{01} t_{1}^{2} + \eta_{02} n_{1}^{2} + \eta_{44} n_{1}^{2} + \eta_{55} t_{1}^{2}\right) + 2 t_{0} t_{1} \left(\eta_{01} s_{1}^{2} + \eta_{11} t_{1}^{2} + \eta_{12} n_{1}^{2} + \eta_{33} n_{1}^{2} + \eta_{55} s_{1}^{2}\right)\right) & \sqrt{2} \cdot \left(2 n_{0} \left(\eta_{33} n_{2} t_{1} t_{2} + \eta_{44} n_{2} s_{1} s_{2} + n_{1} \left(\eta_{02} s_{2}^{2} + \eta_{12} t_{2}^{2} + \eta_{22} n_{2}^{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{1} n_{2} s_{2} + \eta_{55} s_{2} t_{1} t_{2} + s_{1} \left(\eta_{00} s_{2}^{2} + \eta_{01} t_{2}^{2} + \eta_{02} n_{2}^{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{1} n_{2} t_{2} + \eta_{55} s_{1} s_{2} t_{2} + t_{1} \left(\eta_{01} s_{2}^{2} + \eta_{11} t_{2}^{2} + \eta_{12} n_{2}^{2}\right)\right)\right) & 2 n_{0} \left(\eta_{33} t_{1} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{44} s_{1} \left(n_{1} s_{2} + n_{2} s_{1}\right) + 2 n_{1} \left(\eta_{02} s_{1} s_{2} + \eta_{12} t_{1} t_{2} + \eta_{22} n_{1} n_{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{1} \left(n_{1} s_{2} + n_{2} s_{1}\right) + \eta_{55} t_{1} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 s_{1} \left(\eta_{00} s_{1} s_{2} + \eta_{01} t_{1} t_{2} + \eta_{02} n_{1} n_{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{1} \left(n_{1} t_{2} + n_{2} t_{1}\right) + \eta_{55} s_{1} \left(s_{1} t_{2} + s_{2} t_{1}\right) + 2 t_{1} \left(\eta_{01} s_{1} s_{2} + \eta_{11} t_{1} t_{2} + \eta_{12} n_{1} n_{2}\right)\right) & 2 n_{0} \left(\eta_{33} t_{1} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{44} s_{1} \left(n_{0} s_{2} + n_{2} s_{0}\right) + 2 n_{1} \left(\eta_{02} s_{0} s_{2} + \eta_{12} t_{0} t_{2} + \eta_{22} n_{0} n_{2}\right)\right) + 2 s_{0} \left(\eta_{44} n_{1} \left(n_{0} s_{2} + n_{2} s_{0}\right) + \eta_{55} t_{1} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 s_{1} \left(\eta_{00} s_{0} s_{2} + \eta_{01} t_{0} t_{2} + \eta_{02} n_{0} n_{2}\right)\right) + 2 t_{0} \left(\eta_{33} n_{1} \left(n_{0} t_{2} + n_{2} t_{0}\right) + \eta_{55} s_{1} \left(s_{0} t_{2} + s_{2} t_{0}\right) + 2 t_{1} \left(\eta_{01} s_{0} s_{2} + \eta_{11} t_{0} t_{2} + \eta_{12} n_{0} n_{2}\right)\right) & 2 n_{0} \left(\eta_{33} t_{1} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{44} s_{1} \left(n_{0} s_{1} + n_{1} s_{0}\right) + 2 n_{1} \left(\eta_{02} s_{0} s_{1} + \eta_{12} t_{0} t_{1} + \eta_{22} n_{0} n_{1}\right)\right) + 2 s_{0} \left(\eta_{44} n_{1} \left(n_{0} s_{1} + n_{1} s_{0}\right) + \eta_{55} t_{1} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 s_{1} \left(\eta_{00} s_{0} s_{1} + \eta_{01} t_{0} t_{1} + \eta_{02} n_{0} n_{1}\right)\right) + 2 t_{0} \left(\eta_{33} n_{1} \left(n_{0} t_{1} + n_{1} t_{0}\right) + \eta_{55} s_{1} \left(s_{0} t_{1} + s_{1} t_{0}\right) + 2 t_{1} \left(\eta_{01} s_{0} s_{1} + \eta_{11} t_{0} t_{1} + \eta_{12} n_{0} n_{1}\right)\right)\end{matrix}\right]$$
#
#

# +
## I can't see much benefit in expanding the full orthorhombic model
## in this way - keeping the rotations separated makes more sense.

eta_00 = sympy.symbols(r"\eta_00")
eta_11 = sympy.symbols(r"\eta_11")
eta_22 = sympy.symbols(r"\eta_22")
eta_33 = sympy.symbols(r"\eta_33")
eta_44 = sympy.symbols(r"\eta_44")
eta_55 = sympy.symbols(r"\eta_55")
eta_01 = sympy.symbols(r"\eta_01")
eta_02 = sympy.symbols(r"\eta_02")
eta_12 = sympy.symbols(r"\eta_12")

I_ijkl = uw.maths.tensor.rank4_identity(3) * 0
C_IJm_ORTHO = uw.maths.tensor.rank4_to_mandel(I_ijkl, 3)

C_IJm_ORTHO[0,0] = 2*eta_00
C_IJm_ORTHO[1,1] = 2*eta_11
C_IJm_ORTHO[2,2] = 2*eta_22
C_IJm_ORTHO[3,3] = 2*eta_33  # yz
C_IJm_ORTHO[4,4] = 2*eta_44  # xz
C_IJm_ORTHO[5,5] = 2*eta_55  # xy
C_IJm_ORTHO[0,1] = C_IJm_ORTHO[1,0] = 2*eta_01
C_IJm_ORTHO[0,2] = C_IJm_ORTHO[2,0] = 2*eta_02
C_IJm_ORTHO[1,2] = C_IJm_ORTHO[2,1] = 2*eta_12

C_ijkl_ORTHO = uw.maths.tensor.mandel_to_rank4(C_IJm_ORTHO, 3)
C_IJv_ORTHO = sympy.simplify(uw.maths.tensor.rank4_to_voigt(C_ijkl_ORTHO, 3))

C_IJv_ORTHO


# +
# Rotation: Use 3 orthogonal unit vectors to define cannonical orientation

n = sympy.Matrix(sympy.symarray("n",(3,)))
s = sympy.Matrix(sympy.symarray("s",(3,)))
t = sympy.Matrix(sympy.symarray("t",(3,)))

# This would work but is less clear in terms of notation
# t = -mesh3.vector.cross(n,s).T # complete the coordinate triad

Rx = sympy.BlockMatrix((s,t,n)).as_explicit()

Rx

# +
C_ijkl_ORTHO = uw.maths.tensor.mandel_to_rank4(C_IJm_ORTHO, 3)
C_ijkl_ORTHO_R = sympy.simplify(uw.maths.tensor.tensor_rotation(Rx, C_ijkl_ORTHO))
uw.maths.tensor.rank4_to_mandel(C_ijkl_ORTHO_R, 3)

# print(sympy.latex(uw.maths.tensor.rank4_to_mandel(C_ijkl_ORTHO_R, 3)))


# + jupyter={"outputs_hidden": true}

xi_0 = sympy.symbols(r"\xi_0")
xi_1 = sympy.symbols(r"\xi_1")

I_ijkl = uw.maths.tensor.rank4_identity(3) * 0
C_IJm_BC = uw.maths.tensor.rank4_to_mandel(I_ijkl, 3)

C_IJm_BC[0,0] =  2*xi_0
C_IJm_BC[1,1] = -2*xi_0
C_IJm_BC[3,3] = -2*xi_1  # yz
C_IJm_BC[4,4] = 2*xi_1  # xz
C_IJm_BC[5,5] = 0
C_IJm_BC[0,2] = C_IJm_BC[2,0] = 2*xi_1
C_IJm_BC[1,2] = C_IJm_BC[2,1] = -2*xi_1

C_ijkl_BC = uw.maths.tensor.mandel_to_rank4(C_IJm_BC, 3)
C_IJv_BC = sympy.simplify(uw.maths.tensor.rank4_to_voigt(C_ijkl_BC, 3))

print(sympy.latex(C_IJm_BC))

C_IJv_BC

# Rotation by Rx

C_ijkl_BC = uw.maths.tensor.mandel_to_rank4(C_IJm_BC, 3)
C_ijkl_BC_R = sympy.simplify(uw.maths.tensor.tensor_rotation(Rx, C_ijkl_ORTHO))
uw.maths.tensor.rank4_to_mandel(C_ijkl_BC_R, 3)

print(sympy.latex(uw.maths.tensor.rank4_to_mandel(C_ijkl_BC_R, 3)))

# -
display(C_ijkl_BC_R)




