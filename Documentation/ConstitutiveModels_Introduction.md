---
title: "Underworld 3: Constitutive Models"
keywords: [underworld3, sympy, PETSc]
authors: 
  - name: Underworld Team
exports:
  - format: pdf
    template: lapreprint
    theme_color: blue
---

# Introduction to constitutive relationships in Underworld3 

Constitutive models relate fluxes of a quantity to the gradients of the unknowns. For example, in thermal diffusion, there is a relationship between heat flux and temperature gradients which is known as *Fourier's law* and which involves an empirically determined thermal conductivity.

$$ q_i = k \frac{ \partial T}{\partial x_i} $$

and in elasticity or fluid flow, we have a relationship between stresses and strain (gradients of the displacements) or strain-rate (gradients of velocity) respectively

$$ \tau_{ij} = \mu \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) $$

where $\eta$, here, is a material property (elastic shear modulus, viscosity) that we need to determine experimentally.

These material properties can be non-linear (they depend upon the quantities that they relate), and they can also be anisotropic (they vary with the orientation of the material). In the latter case, the material property is described through a *consitutive tensor*. For example:

$$ q_i = k_{ij} \frac{ \partial T}{\partial x_j} $$

or

$$ \tau_{ij} = \mu_{ijkl} \left( \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right) $$


+++ {"magic_args": "[markdown]"}

## Constitutive tensors

In Underworld, the underlying implementation of any consitutive relationship is through the appropriate tensor representation, allowing for a very general formulation for most problems. The constitutive tensors are described symbolically using `sympy` expressions. For scalar equations, the rank 2 constitutive tensor ($k_{ij}$) can be expressed as a matrix without any loss of generality. 

For vector equations, the constitutive tensor is of rank 4, but it is common in the engineering / finite element literature to transform the rank 4 tensor to a matrix representation. This has some benefits in terms of legibility and, perhaps, results in common code patterns across the formulation for scalar equations. 

We can write the stress and strain rate tensors ($\tau_{ij}$ and $\dot\epsilon_{ij}$ respectively) as

$$ \tau_I = \left( \tau_{00}, \tau_{11}, \tau_{22}, \tau_{12}, \tau_{02}, \tau_{01} \right) $$

$$ \dot\varepsilon_I = \left( \dot\varepsilon_{00}, \dot\varepsilon_{11}, \dot\varepsilon_{22}, \dot\varepsilon_{12}, \dot\varepsilon_{02}, \dot\varepsilon_{01} \right) $$

where $I$ indexes $\{i,j\}$ in a specified order and symmetry of the tensor implies only six (in 3D) unique entries in the nine entries of a general tensor. With these definitions, the constitutive tensor can be written as $C_{IJ}$ where $I$, and $J$ range over the independent degrees of freedom in stress and strain (rate). This is a re-writing of the full tensor, with the necessary symmetry required to respect the symmetry of $\tau$ and $\dot\varepsilon$ 

The simplified notation comes at a cost. For example, $\tau_I \dot\varepsilon_I$ is not equivalent to the tensor inner produce $\tau_{ij} \dot\varepsilon_{ij}$, and $C_{IJ}$ does not transform correctly for a rotation of coordinates. 

However, a modest transformation of the matrix equations is helpful:

$$ \tau^*_{I} = C^*_{IJ} \varepsilon^*_{J} $$ 

Where 

$$\tau^*_{I} = P_{IJ} \tau^*_J$$

$$\varepsilon^*_I = P_{IJ} \varepsilon^*_J$$

$$C^*_{IJ} = P_{IK} C_{KL} P_{LJ}$$

$\mathbf{P}$ is the scaling matrix 

$$P = \left[\begin{matrix}1 & 0 & 0 & 0 & 0 & 0\\0 & 1 & 0 & 0 & 0 & 0\\0 & 0 & 1 & 0 & 0 & 0\\0 & 0 & 0 & \sqrt{2} & 0 & 0\\0 & 0 & 0 & 0 & \sqrt{2} & 0\\0 & 0 & 0 & 0 & 0 & \sqrt{2}\end{matrix}\right]$$

In this form, known as the Mandel notation, $\tau^*_I\varepsilon^*_I \equiv \tau_{ij} \varepsilon_{ij}$, and the fourth order identity matrix:

$$I_{ijkl} = \frac{1}{2} \left( \delta_{ij}\delta_{kj} + \delta_{kl}\delta_{il} \right)$$ 

transforms to 

$$I_{IJ} = \delta_{IJ}$$

### Voigt form

Computation of the stress tensor using $\tau^*_I = C^*_{IJ}\varepsilon^*_J$ is equivalent to the following

$$ \mathbf{P} \mathbf{\tau} =  \mathbf{P} \mathbf{C} \mathbf{P} \cdot \mathbf{P} \mathbf{\varepsilon} $$

multiply by $\mathbf{P}^{-1} $ and collecting terms:

$$ \mathbf{\tau} = \mathbf{C} \mathbf{P}^2 \mathbf{\varepsilon} $$

This is the Voigt form of the constitutive equations and is generally what you will find in a finite element textbook. The Voigt form of the constitutive matrix is the rearrangement of the rank 4 constitutive tensor (with no scaling), the strain rate vector is usually combined with $\mathbf{P}^2$, and the stress vector is raw. For example:


$$ \left[\begin{matrix}\tau_{0 0}\\\tau_{1 1}\\\tau_{0 1}\end{matrix}\right] =
   \left[\begin{matrix}\eta & 0 & 0\\0 & \eta & 0\\0 & 0 & \frac{\eta}{2}\end{matrix}\right]   \left[\begin{matrix}\dot\varepsilon_{0 0}\\\dot\varepsilon_{1 1}\\2 \dot\varepsilon_{0 1}\end{matrix}\right]
$$

A full discussion of this can be found in {cite}`brannonRotationReflectionFrame2018` and in {cite}`helnweinRemarksCompressedMatrix2001`

+++ {"magic_args": "[markdown]"}

## Sympy tensorial form, Voigt form

The rank 4 tensor form of the constitutive equations, $c_{ijkl}$, has the following (2D) representation in `sympy`:

$$\left[\begin{matrix}\left[\begin{matrix}c_{0 0 0 0} & c_{0 0 0 1}\\c_{0 0 1 0} & c_{0 0 1 1}\end{matrix}\right] & \left[\begin{matrix}c_{0 1 0 0} & c_{0 1 0 1}\\c_{0 1 1 0} & c_{0 1 1 1}\end{matrix}\right]\\\left[\begin{matrix}c_{1 0 0 0} & c_{1 0 0 1}\\c_{1 0 1 0} & c_{1 0 1 1}\end{matrix}\right] & \left[\begin{matrix}c_{1 1 0 0} & c_{1 1 0 1}\\c_{1 1 1 0} & c_{1 1 1 1}\end{matrix}\right]\end{matrix}\right]$$

and the inner product $\tau_{ij} = c_{ijkl} \varepsilon_{kl} $ is written

```python
tau = sympy.tensorcontraction(
            sympy.tensorcontraction(
                   sympy.tensorproduct(c, epsilon),(3,5)),(2,3))
```

However, the `sympy.Matrix` module allows a much simpler expression

```python
tau_star = C_star * epsilon_star
tau = P.inv() * tau_star
```

which we adopt in `underworld` for the display and manipulation of constitutive tensors in either the Mandel form or the (more common in Engineering texts) Voigt form.

