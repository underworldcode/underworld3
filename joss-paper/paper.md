---
title: 'Underworld3: Mathematically Self-Describing Modelling in Python for Desktop, HPC and Cloud'
tags:
  - Python
  - Geodynamics
  - PETSc
  - sympy
  - symbolic algebra
  - finite element
authors:
  - name: Louis Moresi
    orcid: 0000-0003-3685-174X
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: John Mansour
    equal-contrib: true
    orcid: 0000-0001-5865-1664
    affiliation: "2"
  - name: Julian Giordani
    equal-contrib: true
    orcid: 0000-0003-4515-9296
    affiliation: "3"
  - name: Matt Knepley
    orcid: 0000-0002-2292-0735
    affiliation: "4"
  - name: Ben Knight
    orcid: 0000-0001-7919-2575
    affiliation: "5"
  - name: Juan Carlos Graciosa
    orcid: 0000-0003-0817-354X
    affiliation: "1"
  - name: Thyagarajulu Gollapalli
    orcid: 0000-0001-9394-4104
    affiliation: "2"
  - name: Neng Lu
    orcid: 0000-0001-9424-2315
    affiliation: "1"
  - name: Romain Beucher
    orcid: 0000-0003-3891-5444
    affiliation: "1"

affiliations:
 - name: Research School of Earth Sciences, Australian National University, Canberra, Australia
   index: 1
 - name: School of Earth, Atmospheric & Environmental Science, Monash University
   index: 2
 - name: University of Sydney, Sydney, Australia
   index: 3
 - name: Computer Science and Engineering, University at Buffalo
   index: 4
 - name: Curtin University, Perth, Australia
   index: 5


date: 30 June 2024
bibliography: paper.bib
---

# Summary


`Underworld3` is a finite element, geophysical-fluid-dynamics modelling framework
 designed to be both straightforward to use and highly scalable to peak high-performance computing environments.
 `Underworld3` inherits the design patterns of earlier versions of `underworld` including: (1) A python user interface that is inherently safe for parallel computation. (2) A symbolic interface based on `sympy` that allows users to construct and simplify combinations of mathematical functions, unknowns and the spatial gradients of unknowns on the fly. (3) Interchangeable Lagrangian, Semi-Lagrangian and Eulerian time derivatives with symbolic representations wrapping the underlying implementation. (4) Fast, robust, parallel numerical solvers based on `PETSc` and `petsc4py`, (5) Flexible, Lagrangian "particle"  swarms for handling transport-dominated unknowns that are fully interchangeable with other data-types and can also be treated as symbolic quantities. (6) Unstructured and adaptive meshing that is fully compatible with the symbolic framework.

The symbolic forms in (2,3) are used to construct a finite element representation using `sympy` and `cython`. These forms are just-in-time (JIT) compiled as `C` functions libraries and pointers to these libraries are given to PETSc to describe the finite element weak forms (surface and volume integrals), Jacobian derivatives and boundary conditions.

Users of `underworld3` typically develop python scripts within `jupyter` notebooks and, in this environment, `underworld3` provides introspection of its native classes both as python objects as well as mathematical ones. This allows symbolic prototyping and validation of PDE solvers in scripts that can immediately be deployed in a parallel HPC environment.

# Statement of need

The problems in global planetary dynamics and tectonics that `underworld3` is designed to address have a number of defining characteristics:  geomaterials are non-linear, viscoelastic/plastic and have a propensity for strain-dependent softening during deformation; strain localisation is very common as a consequence. Geological structures that we seek to understand are often emergent over the course of loading and are observed in the very-large deformation limit. Material properties have strong spatial gradients arising from pressure and temperature dependence and jumps of several orders of magnitude resulting from material interfaces.

Requirement for HPC ...





Geophysical fluid dynamics - complex and non-linear problems that also require high performance, parallelism, good quality derivatives etc.

Introspection at the mathematical level is important because problems typically cascade in complexity quickly.

Time discretisation is complicated because it may introduce history terms in both fluxes (derivatives of unknowns including material history) and the unknowns themselves.


PETSc is just impossible, isn't it ? Need I say more ?


# Mathematical Framework

PETSc provides a template form for the automatic generation of weak forms. The strong-form of the problem is defined through the functional $\cal{F}$ that expresses the balance between fluxes and unknowns:

$$
\cal{F}(\mathbf{u}) \sim \color{Black}{\nabla \cdot \mathbf{f}_1 (u, \nabla u)} - \color{Black}{{f_0} (u, \nabla u)} = 0
$$

The discrete weak form and its Jacobian derivative can be expressed as follows

$$ \cal{F}(\cal{u}) \sim \sum_e \epsilon_e^T \left[ B^T W f_0(u^q, \nabla u^q) + \sum_k D_k^T W \mathbf{f}_1^k (u^q, \nabla u^q) \right] = 0
$$

$$ \cal{F'}(\cal{u}) \sim \sum_e \epsilon_e^T
                \left[ \begin{array}{cc}
                    B^T  & \mathbf{D}^T \\
                \end{array} \right]
                \mathbf{W}
                \left[ \begin{array}{cc}
                    f_{0,0} & f_{0,1} \\
                    \mathbf{f}_{1,0} & \mathbf{f}_{1,1} \\
                \end{array}\right]
                 \left[ \begin{array}{c}
                    B^T  \\
                    \mathbf{D}^T
                \end{array} \right] \epsilon_e
\quad \mathrm{and} \quad
                f_{[i,j]} =
                \left[ \begin{array}{cc}
                    \partial f_0 / \partial u & \partial f_0 / \partial \nabla u \\
                    \partial \mathbf{f}_1 / \partial u & \partial \mathbf{f}_1 / \partial \nabla u
                \end{array}\right]
$$

The symbolic form corresponding to [ ref above ] is

$$
\color{DarkGreen}{\underbrace{ \Bigl[ {D u}/{D t} \Bigr]}_{\dot{\mathbf{f} } } }
 \color{Blue}{
- \nabla \cdot \underbrace{\Bigl[ \mathrm{F}(u, \nabla u) \Bigr]}_{\mathbf{F}}}
- \color {Maroon}{\underbrace{\Bigl[
   \mathrm{H}(\mathbf{x},t) \Bigr]}_{\mathbf{f}}}
= 0
$$

This symbolic form contains material / time derivatives of the unknowns which are not present in [ ref above ] because, after discretisation, they simplify to produce terms that are combinations of fluxes and flux history terms (which modify $\mathrm{F}$) and forces (which modify $\mathbf{f}$. In `Underworld3` this simplification is handled by sympy which allows arbitrary combinations of numerical time-discretisation with constitutive models that have history terms or the force-like terms which often occur in non-Cartesian coordinate systems.

# Design Patterns


# Documentation


# Discussion


Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

<!-- You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text. -->

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

# Figures

<!-- Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

We acknowledge contributions from AuScope, mainly. Anyone else ?

# References
