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
    orcid: 0000-0001-5865-1664
    affiliation: "2"
  - name: Julian Giordani
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

`Underworld3` is a finite element, geophysical-fluid-dynamics modelling framework designed to be both straightforward to use and highly scalable to peak high-performance computing environments. It implements the Lagrangian-particle finite element methodology outlined in @moresiLagrangianIntegrationPoint2003.

 `Underworld3` inherits the design patterns of earlier versions of `underworld` including: (1) A python user interface that is inherently safe for parallel computation. (2) A symbolic interface based on `sympy` that allows users to construct and simplify combinations of mathematical functions, unknowns and the spatial gradients of unknowns on the fly. (3) Interchangeable Lagrangian, Semi-Lagrangian and Eulerian time derivatives with symbolic representations wrapping the underlying implementation. (4) Fast, robust, parallel numerical solvers based on `PETSc` [@balayPETScTAOUsers2024] and `petsc4py` [@dalcinpazklercosimo2011], (5) Flexible, Lagrangian "particle"  swarms for handling transport-dominated unknowns that are fully interchangeable with other data-types and can also be treated as symbolic quantities. (6) Unstructured and adaptive meshing that is fully compatible with the symbolic framework.

The symbolic forms in (2,3) are used to construct a finite element representation using `sympy` [@meurerSymPySymbolicComputing2017] and `cython` [@behnel2011cython]. These forms are just-in-time (JIT) compiled as `C` functions libraries and pointers to these libraries are given to PETSc to describe the finite element weak forms (surface and volume integrals), Jacobian derivatives and boundary conditions.

Users of `underworld3` typically develop python scripts within `jupyter` notebooks and, in this environment, `underworld3` provides introspection of its native classes both as python objects as well as mathematical ones. This allows symbolic prototyping and validation of PDE solvers in scripts that can immediately be deployed in a parallel HPC environment.

# Statement of need

Typical problems in geodynamics usually require computing material deformation, damage evolution, and interface tracking in the large-deformation limit. These are typically not well supported by standard engineering finite element simulation codes. Underworld is a python software framework that is intended to solve geodynamics problems that sit at the interface between computational fluid mechanics and solid mechanics (often known as *complex fluids*). It does so by putting Lagrangian and Eulerian variables on an equal footing at both the user and computational levels.

Underworld is built around a general, symbolic partial differential equation solver but provides template forms to solve common geophysical fluid dynamics problems such as the Stokes equation for mantle convection, subduction-zone evolution, lithospheric deformation, glacial isostatic adjustment, ice flow; Navier-Stokes equations for finite Prandtl number fluid flow and short-timescale, viscoelastic deformation; and Darcy Flow for porous media problems including groundwater flow and contaminant transport.

These problems have a number of defining characteristics:  geomaterials are non-linear, viscoelastic/plastic and have a propensity for strain-dependent softening during deformation; strain localisation is very common as a consequence. Geological structures that we seek to understand are often emergent over the course of loading and are observed in the very-large deformation limit. Material properties have strong spatial gradients arising from pressure and temperature dependence and jumps of several orders of magnitude resulting from material interfaces.

`underworld3` automatically handles much of the complexity of combining the non-linearities in rheology, boundary conditions and time-discretisation, forming their derivatives, and simplifying expressions to generate an efficient, parallel `PETSc` script. `underworld3` provides a textbook-like mathematical experience for users who are confident in understanding physical modelling. A number of equation-system templates are provided for typical geophysical fluid dynamics problems such as Stokes-flow, Navier-Stokes-flow, and Darcy flow which provide both usage and mathematical documentation at run-time.

## Mathematical Framework

The symbolic layer of `underworld3` works with the "strong form" of a problem which is typically how the governing equations are derived and disseminated in publications and textbooks. The finite element method is based on a corresponding weak or variational form [e.g. 'standard' finite element textbooks such as @hughesFiniteElementMethod1987, @batheFiniteElementMethod2008, @zienkiewiczFiniteElementMethod2013].

`PETSc` provides a template form for the automatic generation of weak forms [see @knepleyAchievingHighPerformance2013]. We start from the strong-form of the problem which is defined through the functional $\mathcal{F}_s$ that expresses the balance between fluxes ($F(u, \nabla u)$), forces, $f(u, \nabla u)$, and unknowns $u$:

\begin{equation}\label{eq:petsc-strong-form}
\mathcal{F}_s(u) \sim \nabla \cdot F(u, \nabla u) - f(u, \nabla u) = 0
\end{equation}

The discrete weak form and its Jacobian derivative would then be expressed through the related functional $\mathcal{F}_w$  as follows:

\begin{equation}\label{eq:petsc-weak-form}
 \mathcal{F}_w(u) \sim \sum_e \epsilon_e^T \left[ B^T W f(u^q, \nabla u^q) + \sum_k D_k^T W F^k (u^q, \nabla u^q) \right] = 0
\end{equation}

\begin{equation}\label{eq:petsc-jacobian}
 \mathcal{F}_w'(u) \sim \sum _e \epsilon _{e^T}
                \left[ \begin{array}{cc}
                    B^T  & D^T \\
                \end{array} \right]
                W
                \left[ \begin{array}{cc}
                \partial {f}/{\partial {u}} &
                \partial {f}/{\partial \nabla {u}} \\
                \partial {F}/{\partial {u}} &
                \partial {F}/{\partial \nabla {u}} \\
                \end{array}\right]
                 \left[
                    \begin{array}{c}
                      B^T  \\
                      D^T
                    \end{array} \right]
                \epsilon _{e}
\end{equation}


Here $\epsilon$ is the element restriction operator; $B$ is the matrix of basis function derivatives and $D$ is the constitutive matrix that, together, describe the relation between the unknowns and the flux. $q$ indicates that the values are determined at a set of quadrature points, and $W$ is a diagonal matrix of weights for these points.

The symbolic representation of the strong-form that is encoded in `underworld3` is:

\begin{equation}\label{eq:sympy-strong-form}
 \Bigl[ {D u}/{D t} \Bigr]
-\nabla \cdot \Bigl[ \sigma(u, \nabla u, \mathbf{x}, t) \Bigr]
-\Bigl[ \mathrm{H}(u, \nabla u, \mathbf{x},t) \Bigr]
= 0
\end{equation}

Here $\mathrm{H}$ represents sources and sinks of $u$, and ${D u}/{D t}$ is the material time derivative of $u$. The time derivatives of the unknowns are not present in the `PETSc` template but, after time-discretisation, they produce terms that are combinations of fluxes and flux history terms (which combine with $\boldsymbol{\sigma}$ to contribute to $F$) and forces (which combine with $\mathbf{h}$ to contribute to $f$). The explicit time / position dependence in $\sigma$ is to highlight potential changes to boundary conditions or constitutive properties.

In `underworld3`, the user interacts with the time derivatives explicitly, and provides strong-form expressions for the template \ref{eq:sympy-strong-form}. `Sympy` automatically gathers all the flux-like terms and all the force-like terms into the form required by the `PETSc` template. All evaluations, derivatives and simplifications of functions in the `underworld3` symbolic layer are deferred until final assembly of the `PETSc` template and the compilation of the `C` functions.

The main benefits of combining `sympy` with the `PETSc` weak form template is a user environment that 1) provides symbolic, mathematical introspection, particularly in the context of Jupyter notebooks; 2) eliminates much of the `python` or `C` coding required for complex constitutive models; 3) eliminates any need for users to compute derivatives for the Newton solvers in `PETSc`.

# State of the Field

`Underworld3` is one among a small number of codes for studying Earth deformation on medium to long geological time-scales. Early geodynamics codes, of which there were too many to recite individually, were highly specialised for specific tasks with little flexibility for user-defined problems. A subsequent generation of codes, currently in use, was built around generic partial differential equation solvers with scriptable interfaces.

These include: Aspect [$\textrm{C\nolinebreak\hspace{-.05em}\raisebox{.4ex}{\tiny\bf +}\nolinebreak\hspace{-.10em}\raisebox{.4ex}{\tiny\bf +}}$ plugin architecture: @heisterHighAccuracyMantle2017], Underworld 1 and 2 [xml object composition / python scripting respectively, @mansourUnderworld2PythonGeodynamics2020; @moresiComputationalApproachesStudying2007], Fluidity [xml combined with python scripting, @daviesFluidityFullyUnstructured2011a], Milamin [Matlab front end, @dabrowskiMILAMINMATLABbasedFinite2008], LaMEM [julia scripting @boriskausLaMEM2024], TerraFERMA [Unified Form Language, @wilsonTerraFERMARansparent2017], GAdopt [Unified Form Language / python @daviesAutomaticFiniteelementMethods2022].

`Underworld3` uses python and the python package `sympy` as the scripting interface that overlies the generic partial differential equation layer. The advantage of `sympy` is that it is a fully featured symbolic algebra package which allows much of the logic of the mathematical problem description to be defined symbolically and dynamically rather than as static relationships between objects. It also provides deep, mathematical introspection when developing and debugging models.

# Discussion

The aim of `underworld3` is to provide strong support to users in developing sophisticated mathematical models, and provide the ability to interrogate those models during development and at run-time. `Underworld3` encodes the mathematical structure of the equations it solves and will display, in a publishable mathematical form, the derivations and simplifications that it makes as it sets up the numerical solution.

Despite this symbolic, interactive layer, `underworld3` python scripts are inherently-parallel codes that seamlessly deploy as scripts in a high-performance computing parallel environment with very little performance overhead.

`Underworld3` documentation is accessible in a rich, mathematical format within jupyter notebooks for model development and analysis but is also incorporated into the API documentation in the same format.

# Acknowledgements

AuScope provides direct support for the core development team behind the underworld codes and the underworld cloud suite of tools. AuScope is funded by the Australian Government through the National Collaborative Research Infrastructure Strategy, NCRIS.

The development and testing of our codes is also supported by computational resources provided by the Australian Government through the National Computing Infrastructure (NCI) under the National Computational Merit Allocation Scheme (project m18). This work was also supported by resources provided by the Pawsey Supercomputing Research Centre’s
[Setonix Supercomputer](https://doi.org/10.48569/18sb-8s43), with funding from the Australian Government and the Government of Western Australia.

The Australian Research Council (ARC) supported the development of novel algorithms, computational methods and applications under the Discovery Project and Linkage Project programs. AuScope funding was used to make these methods widely and freely available in the underworld codes. Direct support for Underworld was provided by ARC Industrial Transformation Research Hub Program (The Basin Genesis Hub).

# References
